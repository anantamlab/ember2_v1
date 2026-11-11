"""Run 19 execution interface: the laptop develops, the RTX 5090 workstation only executes pinned runs.

  prepare-run      exact git commit + exact HF revisions + a committed run config (data/runs/*.json)
                   -> download, verify every hash, write a frozen execution manifest
  execute-run      one AUTHORIZED run from its manifest; refuses code / data / model / environment drift;
                   writes durable logs, keeps partial output apart from final output; resumable only with an
                   explicit --resume; a COMPLETE run is never executed again
  status           state of one run
  publish-results  upload a run's outputs to the private experiments repo (immutable path), print the revision
  fetch-results    (laptop) download a published result at an exact revision, verify it, write a review

Run kinds: `env-check` (GPU/driver/CUDA/PyTorch Blackwell checks + pinned install audit), `trace-replay` (CPU:
recorded V1 traces through the simulator), `v1-reproduce` (GPU: load V1 and re-predict recorded steps),
`v1-policy-ablation` (GPU: arms A/B/C, see policy_arms; its config is NOT authorized).

Layout under the workstation root (`--root`, env EMBER_WS_ROOT, default ~/ember-ws):
  artifacts/<repo>/<revision>/<path>/   verified downloads (immutable per revision)
  runs/<run_id>/attempt-<n>/            execution_manifest.json, state.json, logs/, partial/, final/
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import hfstore as hs

VERSION = "ember-workstation@1.0.0"
REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK = "infra/p0b-requirements.lock.txt"
REV_ARGS = {"models": "model_revision", "datasets": "dataset_revision", "experiments": "experiments_revision"}
KINDS = {"env-check": {"gpu": True, "model": False}, "trace-replay": {"gpu": False, "model": False},
         "v1-reproduce": {"gpu": True, "model": True}, "v1-policy-ablation": {"gpu": True, "model": True}}
STATES = ("PREPARED", "RUNNING", "INTERRUPTED", "FAILED", "COMPLETE")


class WsError(RuntimeError):
    """Refused: identity, authorization, state or environment check failed."""


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read(p: Path) -> Any:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _write(p: Path, doc: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_bytes(hs.dumps(doc))
    os.replace(tmp, p)


def lf_sha256(p: Path) -> str:
    return "sha256:" + hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def default_root() -> Path:
    return Path(os.environ.get("EMBER_WS_ROOT") or Path.home() / "ember-ws")


# ----- code identity ------------------------------------------------------------------------------------------
def git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise WsError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout.strip()


def code_identity(repo: Path, commit: str) -> dict[str, Any]:
    """The running code must be a clean checkout of exactly `commit` (no branch heads, no local edits)."""
    if not re.fullmatch(r"[0-9a-f]{40}", commit or ""):
        raise WsError(f"--commit must be a full 40-hex git commit, got {commit!r}")
    head = git(repo, "rev-parse", "HEAD")
    if head != commit:
        raise WsError(f"code checkout is at {head}, not the requested {commit}")
    dirty = git(repo, "status", "--porcelain")
    if dirty:
        raise WsError(f"code checkout has local changes (execution machines never edit code):\n{dirty[:500]}")
    return {"commit": head, "clean": True, "checkout": str(repo)}


# ----- environment --------------------------------------------------------------------------------------------
def lock_pins(repo: Path) -> dict[str, str]:
    pins = {}
    for line in (repo / LOCK).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s\\;]+)", line)
        if m:
            pins[m.group(1).lower().replace("_", "-")] = m.group(2)
    return pins


def lock_drift(repo: Path) -> list[str]:
    from importlib import metadata
    bad = []
    for name, want in sorted(lock_pins(repo).items()):
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            bad.append(f"{name}: missing (lock {want})")
            continue
        if have != want:
            bad.append(f"{name}: {have} (lock {want})")
    return bad


def gpu_facts() -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        out["nvidia_smi"] = q.stdout.strip() if q.returncode == 0 else f"error: {q.stderr.strip()[:200]}"
    except (OSError, subprocess.TimeoutExpired) as exc:
        out["nvidia_smi"] = f"unavailable: {exc}"
    try:
        import torch
        out["torch"] = torch.__version__
        out["torch_cuda"] = torch.version.cuda
        out["cuda_available"] = bool(torch.cuda.is_available())
        if out["cuda_available"]:
            out["device"] = torch.cuda.get_device_name(0)
            out["capability"] = list(torch.cuda.get_device_capability(0))
            out["arch_list"] = torch.cuda.get_arch_list()
            out["memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 2)
    except ImportError:
        out["torch"] = None
        out["cuda_available"] = False
    return out


def environment(repo: Path, *, gpu: bool) -> dict[str, Any]:
    env = {"python": platform.python_version(), "implementation": platform.python_implementation(),
           "platform": platform.platform(), "machine": platform.machine(), "host": socket.gethostname(),
           "wsl": "microsoft" in platform.release().lower(), "lock_sha256_lf": lf_sha256(repo / LOCK)}
    from importlib import metadata
    try:
        env["huggingface_hub"] = metadata.version("huggingface_hub")
    except metadata.PackageNotFoundError:
        env["huggingface_hub"] = None
    if gpu:
        env["lock_drift"] = lock_drift(repo)
        env["gpu"] = gpu_facts()
    return env


def check_environment(env: Mapping[str, Any], *, gpu: bool) -> None:
    if not env["python"].startswith("3.11."):
        raise WsError(f"Python {env['python']}: the pinned stack is Python 3.11")
    if gpu:
        if env["lock_drift"]:
            raise WsError(f"installed packages differ from {LOCK}: {env['lock_drift'][:8]}")
        g = env["gpu"]
        if not g.get("cuda_available"):
            raise WsError(f"CUDA is not available to PyTorch: {g}")
        cap = g.get("capability") or [0, 0]
        if cap[0] >= 12 and f"sm_{cap[0]}{cap[1]}" not in (g.get("arch_list") or []):
            raise WsError(f"this PyTorch build has no kernels for sm_{cap[0]}{cap[1]} ({g.get('arch_list')})")


# ----- artifacts ----------------------------------------------------------------------------------------------
def registry(repo: Path) -> dict[str, Any]:
    reg = hs.load_registry(repo / hs.REGISTRY)
    if not reg.get("namespace"):
        raise WsError("data/hf/registry.json has no namespace (laptop: run hfstore init-repos)")
    return reg


def fetch_inputs(cfg: Mapping[str, Any], revs: Mapping[str, str], root: Path, reg: Mapping[str, Any],
                 hf: Any, log: Callable[[str], None]) -> dict[str, Any]:
    ids = hs.repo_ids(reg["namespace"])
    out = {}
    for name, spec in cfg["inputs"].items():
        repo = spec["repo"]
        rev = revs.get(repo)
        if not rev:
            raise WsError(f"input {name!r} lives in the {repo} repo: pass --{REV_ARGS[repo].replace('_', '-')}")
        known = reg["artifacts"].get(spec["artifact_id"])
        if known and known["manifest_sha256"] != spec["manifest_sha256"]:
            raise WsError(f"{spec['artifact_id']}: config manifest differs from the registry")
        rid, rtype = ids[repo]
        dest = root / "artifacts" / repo / rev
        r = hs.fetch(hf, repo_id=rid, repo_type=rtype, path=spec["path"], revision=rev, dest=dest,
                     manifest_sha256=spec["manifest_sha256"], log=log)
        out[name] = {**spec, "revision": rev, "root": r["root"], "status": r["status"]}
    return out


def fetch_base_model(model_root: Path, log: Callable[[str], None]) -> dict[str, Any]:
    """The pinned upstream base (public), into HF_HOME; every file checked against identity.json."""
    ident = _read(model_root / "identity.json")
    b = ident["base_model"]
    from huggingface_hub import snapshot_download
    snap = Path(snapshot_download(repo_id=b["repo"], revision=b["revision"]))
    bad = []
    for rel, f in b["files"].items():
        p = snap / rel
        if not p.exists():
            bad.append(f"{rel}: missing")
            continue
        h = hs.hash_file(p, scan=False)
        ok = h["size"] == f["size"] and (h["sha256"] == f["sha256"] if f.get("sha256") else
                                         h["git_sha1"] == f["git_sha1"])
        if not ok:
            bad.append(f"{rel}: hash/size differs")
    if bad:
        raise WsError(f"base model {b['repo']}@{b['revision']} fails its pinned hashes: {bad[:5]}")
    log(f"base model {b['repo']}@{b['revision'][:12]}: {len(b['files'])} files verified")
    return {"repo": b["repo"], "revision": b["revision"], "snapshot": str(snap), "files_verified": len(b["files"])}


def quick_verify(inputs: Mapping[str, Any]) -> None:
    """Before execution: manifest bytes unchanged and every file present with its size (no rehash)."""
    for name, s in inputs.items():
        root = Path(s["root"])
        mb = (root / hs.MANIFEST).read_bytes()
        if hs.sha256_bytes(mb) != s["manifest_sha256"]:
            raise WsError(f"input {name}: ARTIFACT.json changed since prepare-run")
        for f in json.loads(mb)["files"]:
            p = root / f["path"]
            if not p.is_file() or p.stat().st_size != f["size"]:
                raise WsError(f"input {name}: {f['path']} missing or resized since prepare-run")


# ----- prepare ------------------------------------------------------------------------------------------------
def run_dir(root: Path, run_id: str, attempt: int) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9._@+-]+", run_id):
        raise WsError(f"bad run id {run_id!r}")
    return root / "runs" / run_id / f"attempt-{attempt}"


def prepare_run(*, config: str, commit: str, revisions: Mapping[str, str], root: Path, attempt: int = 1,
                repo: Path = REPO_ROOT, hf: Any = None, skip_code_check: bool = False,
                log: Callable[[str], None] = print) -> Path:
    code = {"commit": commit, "clean": None, "checkout": str(repo), "note": "NOT CHECKED (test)"} \
        if skip_code_check else code_identity(repo, commit)
    for k, v in revisions.items():
        if v and not re.fullmatch(r"[0-9a-f]{40}", v):
            raise WsError(f"{k} must be a full 40-hex HF commit, got {v!r}")
    cpath = (repo / config).resolve()
    if Path(config).is_absolute() or repo.resolve() not in cpath.parents:
        raise WsError(f"--config must be a path inside the checkout, got {config!r}")
    cfg = _read(cpath)
    if cfg.get("config_version") != "ember-run-config@1":
        raise WsError(f"{config}: not an ember-run-config@1 file")
    if (cfg.get("authorization") or {}).get("status") != "AUTHORIZED":
        raise WsError(f"{cfg['run_id']}: NOT AUTHORIZED ({cfg.get('authorization')}); the laptop must commit an "
                      f"authorization before any execution")
    kind = KINDS.get(cfg["kind"])
    if kind is None:
        raise WsError(f"unknown run kind {cfg['kind']!r}")
    rd = run_dir(root, cfg["run_id"], attempt)
    revs = {k: v for k, v in revisions.items() if v}
    conf = {"path": config, "sha256_lf": lf_sha256(cpath), "doc": cfg}
    ident = {"run_id": cfg["run_id"], "attempt": attempt, "config": conf, "code": code, "revisions": revs,
             "inputs": {n: {"manifest_sha256": s["manifest_sha256"], "revision": revs.get(s["repo"]),
                            "path": s["path"]} for n, s in cfg["inputs"].items()}}
    identity = hs.sha256_bytes(hs.dumps(ident))
    mp = rd / "execution_manifest.json"
    if mp.exists():                                       # checked before any download
        old = _read(mp)
        if old["identity_sha256"] != identity:
            raise WsError(f"{rd} was prepared with other inputs ({old['identity_sha256']}); use a new --attempt")
        log(f"already prepared: {mp} (identity {identity[:19]})")
        return rd
    env = environment(repo, gpu=kind["gpu"])
    check_environment(env, gpu=kind["gpu"])
    hf = hf or hs.api()
    reg = registry(repo)
    inputs = fetch_inputs(cfg, revisions, root, reg, hf, log)
    base = fetch_base_model(Path(inputs["model"]["root"]), log) if kind["model"] else None
    man = {"manifest_version": VERSION, "run_id": cfg["run_id"], "attempt": attempt, "kind": cfg["kind"],
           "config": conf, "code": code, "revisions": revs, "inputs": inputs, "base_model": base,
           "environment": env, "namespace": reg["namespace"], "result_path": f"{cfg['result_path']}/attempt-{attempt}",
           "identity_sha256": identity, "prepared_utc": utc()}
    _write(mp, man)
    _write(rd / "state.json", {"status": "PREPARED", "updated_utc": utc(), "history": [["PREPARED", utc()]]})
    log(f"prepared {rd} identity {man['identity_sha256']}")
    return rd


# ----- execute ------------------------------------------------------------------------------------------------
def _state(rd: Path) -> dict[str, Any]:
    return _read(rd / "state.json")


def _set_state(rd: Path, status: str, **extra: Any) -> None:
    st = _state(rd)
    st.update(extra)
    st["status"], st["updated_utc"] = status, utc()
    st.setdefault("history", []).append([status, utc()])
    _write(rd / "state.json", st)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, SystemExit):
        return False
    return True


class Guards:
    def __init__(self, max_wall_s: float, t0: float, prior_s: float) -> None:
        self.max_wall_s, self.t0, self.prior_s = max_wall_s, t0, prior_s

    def __call__(self) -> Any:
        from types import SimpleNamespace
        left = self.max_wall_s - self.prior_s - (time.monotonic() - self.t0)
        return SimpleNamespace(wall_cap_s=left, usd_cap=float("inf"))


def execute_run(rd: Path, *, resume: bool = False, repo: Path = REPO_ROOT, skip_code_check: bool = False,
                adapter_factory: Callable[..., Any] | None = None) -> dict[str, Any]:
    rd = Path(rd)
    man = _read(rd / "execution_manifest.json")
    st = _state(rd)
    if st["status"] == "COMPLETE":
        raise WsError(f"{rd} is COMPLETE ({st['updated_utc']}); a completed run is never executed again")
    lock = rd / "RUNNING.lock"
    if lock.exists():
        info = _read(lock)
        if info["host"] == socket.gethostname() and _pid_alive(int(info["pid"])):
            raise WsError(f"{rd} is already running (pid {info['pid']})")
        if st["status"] == "RUNNING":
            _set_state(rd, "INTERRUPTED", interrupted_note=f"stale lock of pid {info['pid']} (process gone)")
            st = _state(rd)
        lock.unlink()
    if st["status"] in ("INTERRUPTED", "FAILED", "RUNNING") and not resume:
        raise WsError(f"{rd} is {st['status']}: pass --resume to continue from its partial output "
                      f"(nothing is rerun silently)")
    kind = KINDS[man["kind"]]
    if not skip_code_check:
        code_identity(repo, man["code"]["commit"])
    if lf_sha256(repo / man["config"]["path"]) != man["config"]["sha256_lf"]:
        raise WsError("run config changed since prepare-run")
    env = environment(repo, gpu=kind["gpu"])
    check_environment(env, gpu=kind["gpu"])
    for k in ("python", "lock_sha256_lf", "huggingface_hub"):
        if env[k] != man["environment"][k]:
            raise WsError(f"environment drift since prepare-run: {k} {man['environment'][k]} -> {env[k]}")
    if kind["gpu"] and env["gpu"].get("device") != man["environment"]["gpu"].get("device"):
        raise WsError(f"GPU changed since prepare-run: {man['environment']['gpu'].get('device')} -> "
                      f"{env['gpu'].get('device')}")
    quick_verify(man["inputs"])
    _write(lock, {"pid": os.getpid(), "host": socket.gethostname(), "started_utc": utc()})
    logs = rd / "logs"
    logs.mkdir(exist_ok=True)
    log_path = logs / f"execute-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.log"
    logf = open(log_path, "a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"{utc()} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()
    prior_s = float(st.get("elapsed_total_s", 0.0))
    t0 = time.monotonic()
    _set_state(rd, "RUNNING", pid=os.getpid(), resumed=bool(resume), environment_at_start=env)

    def on_signal(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"signal {signum}")
    old = {s: signal.signal(s, on_signal) for s in (signal.SIGTERM, getattr(signal, "SIGHUP", signal.SIGTERM))}
    partial, final = rd / "partial", rd / "final"
    partial.mkdir(exist_ok=True)
    try:
        log(f"execute {man['run_id']} attempt {man['attempt']} kind {man['kind']} commit {man['code']['commit'][:12]}"
            f"{' (resume)' if resume else ''}")
        cfg = man["config"]["doc"]
        guards = Guards(float(cfg.get("max_wall_s", 86400)), t0, prior_s)
        res = RUNNERS[man["kind"]](man, cfg, partial, guards=guards, log=log, repo=repo,
                                   adapter_factory=adapter_factory)
        elapsed = prior_s + time.monotonic() - t0
        if res.get("stopped"):
            _write(partial / "metrics.json", res)
            _set_state(rd, "INTERRUPTED", elapsed_total_s=elapsed, stop=res["stopped"])
            log(f"STOPPED (partial kept): {res['stopped']}")
            return res
        res.update({"elapsed_total_s": round(elapsed, 1), "run_id": man["run_id"], "attempt": man["attempt"],
                    "identity_sha256": man["identity_sha256"], "workstation_version": VERSION})
        _write(partial / "metrics.json", res)
        os.replace(partial, final)
        _set_state(rd, "COMPLETE", elapsed_total_s=elapsed, verdict=res.get("verdict"),
                   metrics_sha256=hs.sha256_bytes((final / "metrics.json").read_bytes()))
        log(f"COMPLETE verdict={res.get('verdict')} in {elapsed:.0f} s")
        return res
    except KeyboardInterrupt as exc:
        _set_state(rd, "INTERRUPTED", elapsed_total_s=prior_s + time.monotonic() - t0, stop=str(exc) or "SIGINT")
        log(f"INTERRUPTED ({exc}); partial output kept in {partial}; continue with --resume")
        raise
    except Exception as exc:
        _set_state(rd, "FAILED", elapsed_total_s=prior_s + time.monotonic() - t0,
                   error=f"{type(exc).__name__}: {str(exc)[:2000]}")
        log(f"FAILED {type(exc).__name__}: {exc}")
        raise
    finally:
        for s, h in old.items():
            signal.signal(s, h)
        with contextlib.suppress(FileNotFoundError):
            lock.unlink()
        logf.close()


# ----- run kinds ----------------------------------------------------------------------------------------------
def _jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def _append(p: Path, doc: Any) -> None:
    with open(p, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(doc, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _selected(man: Mapping[str, Any], cfg: Mapping[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """(rollout record, scenario list item) pairs named by the config, in config order."""
    rec_root = Path(man["inputs"]["records"]["root"])
    recs = {r["scenario_id"]: r for r in _jsonl(rec_root / cfg["params"]["records"])}
    items = {i["scenario_id"]: i for i in _jsonl(rec_root / cfg["params"]["scenarios"])}
    out = []
    ids = cfg["params"]["scenario_ids"]
    for sid in (list(recs) if ids == "ALL" else ids):
        if sid not in recs or sid not in items:
            raise WsError(f"{sid}: not in the recorded rollouts / scenario list")
        if recs[sid]["scenario_hash"] != items[sid]["scenario_hash"]:
            raise WsError(f"{sid}: record and scenario list disagree on the scenario hash")
        out.append((recs[sid], items[sid]))
    return out


def run_env_check(man: Mapping[str, Any], cfg: Mapping[str, Any], out: Path, *, log: Callable[[str], None],
                  repo: Path, **_: Any) -> dict[str, Any]:
    import torch
    g = gpu_facts()
    log(f"gpu {g}")
    checks: dict[str, Any] = {}
    dev = torch.device("cuda")
    a = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(20):
        c = a @ a
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / 20
    checks["bf16_matmul_tflops"] = round(2 * 4096 ** 3 / dt / 1e12, 1)
    checks["bf16_matmul_finite"] = bool(torch.isfinite(c.float()).all())
    q = torch.randn(1, 8, 2048, 128, device=dev, dtype=torch.bfloat16)
    o = torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
    checks["sdpa_finite"] = bool(torch.isfinite(o.float()).all())
    x = torch.randn(2, 64, 1024, device=dev)
    conv = torch.nn.Conv1d(64, 64, 4, groups=64, padding=3).to(dev)
    checks["depthwise_conv_finite"] = bool(torch.isfinite(conv(x)).all())
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def _add(xp, yp, n, BLOCK: tl.constexpr):
            i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            m = i < n
            tl.store(yp + i, tl.load(xp + i, mask=m) + 1.0, mask=m)
        xs = torch.arange(1000, device=dev, dtype=torch.float32)
        ys = torch.empty_like(xs)
        _add[(4,)](xs, ys, 1000, BLOCK=256)
        checks["triton_kernel"] = bool(torch.equal(ys, xs + 1))
        checks["triton"] = triton.__version__
    except Exception as exc:                       # recorded, and fails the verdict
        checks["triton_kernel"] = False
        checks["triton_error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
    vi = subprocess.run([sys.executable, "-m", "ember_v1.p0b", "verify-install"], capture_output=True, text=True,
                        cwd=str(repo), env={**os.environ, "PYTHONPATH": str(repo / "src")})
    checks["verify_install_exit"] = vi.returncode
    (out / "verify_install.txt").write_text(vi.stdout[-20000:] + "\n--- stderr ---\n" + vi.stderr[-20000:],
                                            encoding="utf-8")
    ok = (checks["bf16_matmul_finite"] and checks["sdpa_finite"] and checks["depthwise_conv_finite"]
          and checks["triton_kernel"] and vi.returncode == 0)
    log(f"checks {checks}")
    return {"kind": "env-check", "gpu": g, "checks": checks, "verdict": "PASS" if ok else "FAIL"}


def run_trace_replay(man: Mapping[str, Any], cfg: Mapping[str, Any], out: Path, *, log: Callable[[str], None],
                     guards: Any, **_: Any) -> dict[str, Any]:
    """CPU: every selected recorded V1 trace replayed through the simulator must end in the recorded outcome."""
    from .sim import Scenario
    from .student_run import replay_trace
    path = out / "replay.jsonl"
    done = {r["scenario_id"] for r in _jsonl(path)} if path.exists() else set()
    qs = {"question_style": cfg["params"]["question_style"]}
    for rec, item in _selected(man, cfg):
        if rec["scenario_id"] in done:
            continue
        if guards().wall_cap_s <= 0:
            return {"stopped": {"reason": "max_wall_s"}}
        scn = Scenario.from_dict(item["scenario"])
        if scn.content_hash() != item["scenario_hash"]:
            raise WsError(f"{rec['scenario_id']}: scenario content hash differs")
        rp = replay_trace(scn, rec["trace"], qs)
        want = {"steps": rec["decisions"], **{k: rec[k] for k in ("evacuated", "casualties", "unresolved")}}
        _append(path, {"scenario_id": rec["scenario_id"], "replayed": rp, "recorded": want, "match": rp == want})
        log(f"replay {rec['scenario_id']}: match={rp == want}")
    rows = _jsonl(path)
    n_ok = sum(r["match"] for r in rows)
    return {"kind": "trace-replay", "scenarios": len(rows), "matched": n_ok,
            "verdict": "PASS" if rows and n_ok == len(rows) else "FAIL"}


def _adapter(man: Mapping[str, Any], repo: Path, factory: Callable[..., Any] | None) -> tuple[Any, dict[str, Any]]:
    from . import student_train as st
    ident = _read(Path(man["inputs"]["model"]["root"]) / "identity.json")
    proto = st.load_protocol(repo / ident["protocol"]["path"])
    if proto["protocol_hash"] != ident["protocol"]["protocol_hash"]:
        raise WsError("protocol hash differs from the model identity")
    cfg = st.config_from_protocol(proto)
    ck = Path(man["inputs"]["model"]["root"]) / ident["checkpoint_dir"]
    make = factory or (lambda c, checkpoint=None: st.ClefLoraAdapter(c, checkpoint=checkpoint))
    return make(cfg, checkpoint=str(ck)), cfg


def run_v1_reproduce(man: Mapping[str, Any], cfg: Mapping[str, Any], out: Path, *, log: Callable[[str], None],
                     guards: Any, repo: Path, adapter_factory: Callable[..., Any] | None = None,
                     **_: Any) -> dict[str, Any]:
    """Re-predict recorded V1 steps on identical inputs: the recorded trace rebuilds each visited state; the
    loaded checkpoint predicts it again; the step's argmax actions must equal the recorded ones."""
    from .policy_arms import ScoreTap
    from .sim import Scenario
    from .student_run import StudentPolicy, replay_trace
    t = time.monotonic()
    ad, scfg = _adapter(man, repo, adapter_factory)
    log(f"model loaded in {time.monotonic() - t:.0f} s")
    tap = ScoreTap(ad)
    pol = StudentPolicy(tap, scfg, "V1-reproduce")
    path = out / "steps.jsonl"
    done = {(r["scenario_id"], r["step"]) for r in _jsonl(path)} if path.exists() else set()
    max_steps = int(cfg["params"]["max_steps"])
    stopped = None

    class Stop(Exception):
        pass

    for rec, item in _selected(man, cfg):
        sid = rec["scenario_id"]

        def visit(step: int, obs: Mapping[str, Any], acts: Mapping[str, str], sid: str = sid,
                  rec: Mapping[str, Any] = rec) -> None:
            if step >= max_steps or (sid, step) in done:
                return
            if guards().wall_cap_s <= 0:
                raise Stop
            tap.last, pol.trace = {}, []
            w = time.perf_counter()
            pol.decide(obs)
            got, want = pol.trace[-1], rec["trace"][step]
            margins = {}
            for pid, s in tap.last.items():
                v = sorted(s.values(), reverse=True)
                margins[pid] = round(v[0] - v[1], 6) if len(v) > 1 else None
            _append(path, {"scenario_id": sid, "step": step, "recorded": want, "reproduced": got,
                           "match": got == want, "scores": {p: {a: round(x, 6) for a, x in s.items()}
                                                            for p, s in sorted(tap.last.items())},
                           "margins": margins, "wall_ms": round((time.perf_counter() - w) * 1000, 1)})
            log(f"{sid} step {step}: match={got == want}")
        try:
            replay_trace(Scenario.from_dict(item["scenario"]), rec["trace"], {"question_style": scfg["question_style"]},
                         visit=visit)
        except Stop:
            stopped = {"reason": "max_wall_s"}
            break
    rows = _jsonl(path) if path.exists() else []
    if stopped:
        return {"stopped": stopped, "steps_done": len(rows)}
    return {"kind": "v1-reproduce", **reproduce_summary(rows, near_tie=float(cfg["params"].get("near_tie", 0.05))),
            "weights_digest": _digest(ad), "recorded_weights_digest":
            _read(Path(man["inputs"]["model"]["root"]) / "identity.json")["training"]["final_weights_digest"],
            "peak_memory_gb": round(ad.memory_bytes() / 2 ** 30, 3) if hasattr(ad, "memory_bytes") else None}


def _digest(ad: Any) -> str | None:
    try:
        return ad.weights_digest()
    except Exception as exc:  # informational only (parameter names may differ after reload)
        return f"unavailable: {type(exc).__name__}"


def reproduce_summary(rows: Sequence[Mapping[str, Any]], near_tie: float = 0.05) -> dict[str, Any]:
    """Per step and per person agreement. A mismatch is a near tie when the reproduced top-2 margin of every
    disagreeing person is below `near_tie` (numerics, not a different model)."""
    from . import student_train as st
    people = agree = 0
    mism = []
    for r in rows:
        a, b = r["recorded"], r["reproduced"]
        if isinstance(a, str) and isinstance(b, str) and len(a) == len(b):
            ids = sorted(r["scores"])
            for i, (x, y) in enumerate(zip(a, b)):
                people += 1
                agree += x == y
                if x != y:
                    pid = ids[i]
                    mism.append({"scenario_id": r["scenario_id"], "step": r["step"], "person": pid,
                                 "recorded": st.ACTION_ORDER[int(x)], "reproduced": st.ACTION_ORDER[int(y)],
                                 "margin": r["margins"].get(pid)})
        elif a != b:
            mism.append({"scenario_id": r["scenario_id"], "step": r["step"], "person": None,
                         "recorded": a, "reproduced": b, "margin": None})
    steps_ok = sum(r["match"] for r in rows)
    ties = bool(mism) and all(m["margin"] is not None and m["margin"] < near_tie for m in mism)
    verdict = ("REPRODUCED" if rows and steps_ok == len(rows) else
               "REPRODUCED_EXCEPT_NEAR_TIES" if rows and ties else "NOT_REPRODUCED")
    return {"steps": len(rows), "steps_exact": steps_ok, "people": people, "people_agree": agree,
            "mismatches": mism[:200], "n_mismatches": len(mism), "near_tie_threshold": near_tie, "verdict": verdict}


def run_policy_ablation(man: Mapping[str, Any], cfg: Mapping[str, Any], out: Path, *, log: Callable[[str], None],
                        guards: Any, repo: Path, adapter_factory: Callable[..., Any] | None = None,
                        **_: Any) -> dict[str, Any]:
    from .policy_arms import ARMS, run_arms
    p = cfg["params"]
    arms = p["arms"]
    if [a for a in arms if a not in ARMS]:
        raise WsError(f"unknown arm in {arms}")
    lst = out / "scenarios.jsonl"
    if not lst.exists():                            # the matched scenario list, frozen in the run output
        src = Path(man["inputs"]["records"]["root"]) / p["scenarios"]
        keep = [x for x in _jsonl(src) if x["set"] in p["sets"]]
        if p.get("limit"):
            keep = keep[:int(p["limit"])]
        tmp = lst.with_name("scenarios.jsonl.tmp")
        tmp.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in keep), encoding="utf-8", newline="\n")
        os.replace(tmp, lst)
    ad, scfg = _adapter(man, repo, adapter_factory)
    res = run_arms(ad, scfg, lst, out, arms=arms, label="V1", guards=guards, block=int(p.get("block", 4)), log=log)
    if res["stopped"]:
        return {"stopped": res["stopped"], "arms": res["arms"]}
    return {"kind": "v1-policy-ablation", "arms": res["arms"], "scenarios": res["scenarios"],
            "verdict": "COMPLETE" if all(r["finished"] == res["scenarios"] for r in res["arms"].values())
            else "INCOMPLETE"}


RUNNERS: dict[str, Callable[..., dict[str, Any]]] = {
    "env-check": run_env_check, "trace-replay": run_trace_replay, "v1-reproduce": run_v1_reproduce,
    "v1-policy-ablation": run_policy_ablation}


# ----- publish / fetch ----------------------------------------------------------------------------------------
def publish_results(rd: Path, *, allow_incomplete: bool = False, hf: Any = None,
                    log: Callable[[str], None] = print) -> dict[str, Any]:
    rd = Path(rd)
    man, st = _read(rd / "execution_manifest.json"), _state(rd)
    if st["status"] == "RUNNING":
        raise WsError("the run is RUNNING; publish after it ends")
    if st["status"] != "COMPLETE" and not allow_incomplete:
        raise WsError(f"the run is {st['status']}; pass --allow-incomplete to publish it marked INCOMPLETE")
    status = "RESULT" if st["status"] == "COMPLETE" else ("FAILED" if st["status"] == "FAILED" else "INCOMPLETE")
    path = man["result_path"] if status == "RESULT" else \
        f"{man['result_path']}-{status.lower()}-{st['updated_utc'].replace(':', '').replace('-', '')}"
    stage = rd / "publish" / path.rsplit("/", 1)[-1]
    if not stage.exists():
        tmp = stage.with_name(stage.name + ".staging")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        for name in ("execution_manifest.json", "state.json"):
            shutil.copy2(rd / name, tmp / name)
        shutil.copytree(rd / "logs", tmp / "logs")
        shutil.copytree(rd / ("final" if status == "RESULT" else "partial"), tmp / "outputs")
        os.replace(tmp, stage)
    hf = hf or hs.api()
    rid, rtype = hs.repo_ids(man["namespace"])["experiments"]
    files = hs.file_table(stage, log=log)
    metrics = _read(stage / "outputs" / "metrics.json") if (stage / "outputs" / "metrics.json").exists() else {}
    mf = hs.build_manifest(artifact_id=f"{man['run_id']}.a{man['attempt']}", kind=f"run-result/{man['kind']}",
                           status=status, repo_id=rid, repo_type=rtype, path=path, files=files,
                           source={"git_commit": man["code"]["commit"], "uploader": VERSION,
                                   "host": man["environment"]["host"]},
                           provenance={"identity_sha256": man["identity_sha256"], "run_state": st["status"],
                                       "verdict": metrics.get("verdict"), "revisions": man["revisions"],
                                       "inputs": {n: {k: s[k] for k in ("artifact_id", "manifest_sha256", "revision")}
                                                  for n, s in man["inputs"].items()},
                                       "gpu": (man["environment"].get("gpu") or {}).get("device"),
                                       "warnings": [] if status == "RESULT" else [f"run state {st['status']}"]})
    e = hs.publish(hf, mf, stage, log=log)
    _write(rd / "published.json", {**e, "published_utc": utc()})
    log(f"PUBLISHED {rid}:{path} manifest commit {e['manifest_commit']} (verified {e['verified_commit']})")
    return e


def fetch_results(*, repo_id: str, path: str, revision: str, dest: Path, records: Path | None = None,
                  hf: Any = None, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Laptop: download + verify a published run result; independent re-check of the verdict; review.json."""
    hf = hf or hs.api()
    r = hs.fetch(hf, repo_id=repo_id, repo_type="dataset", path=path, revision=revision, dest=dest, log=log)
    root = Path(r["root"])
    man = _read(root / "execution_manifest.json")
    metrics = _read(root / "outputs" / "metrics.json") if (root / "outputs" / "metrics.json").exists() else {}
    review: dict[str, Any] = {"result": f"{repo_id}:{path}@{revision}", "manifest_sha256": r["manifest_sha256"],
                              "status": r["status"], "run_id": man["run_id"], "attempt": man["attempt"],
                              "kind": man["kind"], "code_commit": man["code"]["commit"],
                              "environment": {k: man["environment"].get(k) for k in ("python", "platform", "host")},
                              "gpu": (man["environment"].get("gpu") or {}), "verdict_reported": metrics.get("verdict")}
    if man["kind"] == "v1-reproduce" and (root / "outputs" / "steps.jsonl").exists():
        rows = _jsonl(root / "outputs" / "steps.jsonl")
        again = reproduce_summary(rows, near_tie=float(man["config"]["doc"]["params"].get("near_tie", 0.05)))
        review["verdict_rechecked"] = again["verdict"]
        review["agreement"] = {k: again[k] for k in ("steps", "steps_exact", "people", "people_agree", "n_mismatches")}
        review["mismatches"] = again["mismatches"][:20]
        if records is not None:                     # the laptop's own original V1 records, not the downloaded copy
            orig = {x["scenario_id"]: x["trace"] for x in _jsonl(records)}
            review["baseline_trace_mismatch"] = [
                [x["scenario_id"], x["step"]] for x in rows
                if orig.get(x["scenario_id"], [None] * (x["step"] + 1))[x["step"]] != x["recorded"]]
    elif man["kind"] == "trace-replay" and (root / "outputs" / "replay.jsonl").exists():
        rows = _jsonl(root / "outputs" / "replay.jsonl")
        review["verdict_rechecked"] = "PASS" if rows and all(x["match"] for x in rows) else "FAIL"
        review["agreement"] = {"scenarios": len(rows), "matched": sum(x["match"] for x in rows)}
    else:
        review["verdict_rechecked"] = metrics.get("verdict")
    _write(Path(dest) / "reviews" / f"{man['run_id']}.a{man['attempt']}.json", review)
    return review


# ----- CLI ----------------------------------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ember_v1.workstation", description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=None, help="workstation root (env EMBER_WS_ROOT, ~/ember-ws)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare-run")
    p.add_argument("--config", required=True, help="run config path inside the checkout, e.g. data/runs/x.json")
    p.add_argument("--commit", required=True)
    p.add_argument("--model-revision", default="")
    p.add_argument("--dataset-revision", default="")
    p.add_argument("--experiments-revision", default="")
    p.add_argument("--attempt", type=int, default=1)
    for name in ("execute-run", "status", "publish-results"):
        q = sub.add_parser(name)
        q.add_argument("--run", required=True, help="run id")
        q.add_argument("--attempt", type=int, default=1)
        if name == "execute-run":
            q.add_argument("--resume", action="store_true", help="continue an INTERRUPTED / FAILED run")
        if name == "publish-results":
            q.add_argument("--allow-incomplete", action="store_true")
    f = sub.add_parser("fetch-results")
    f.add_argument("--path", required=True, help="result path in the experiments repo")
    f.add_argument("--revision", required=True, help="the manifest commit printed by publish-results")
    f.add_argument("--dest", required=True, type=Path)
    f.add_argument("--records", type=Path, help="laptop copy of the original V1 rollouts.jsonl")
    f.add_argument("--repo-id", default="", help="default: <namespace>/ember-v1-experiments from the registry")
    a = ap.parse_args(argv)
    root = a.root or default_root()
    try:
        if a.cmd == "prepare-run":
            rd = prepare_run(config=a.config, commit=a.commit, root=root, attempt=a.attempt,
                             revisions={"models": a.model_revision, "datasets": a.dataset_revision,
                                        "experiments": a.experiments_revision})
            print(json.dumps({"run_dir": str(rd), "next": f"execute-run --run {_read(rd / 'execution_manifest.json')['run_id']} --attempt {a.attempt}"}))
        elif a.cmd == "fetch-results":
            rid = a.repo_id or hs.repo_ids(registry(REPO_ROOT)["namespace"])["experiments"][0]
            print(json.dumps(fetch_results(repo_id=rid, path=a.path, revision=a.revision, dest=a.dest,
                                           records=a.records), indent=1))
        else:
            # the run id names a directory; read the manifest to find it
            rd = run_dir(root, a.run, a.attempt)
            if not (rd / "execution_manifest.json").exists():
                raise WsError(f"{rd}: not prepared")
            if a.cmd == "status":
                st = _state(rd)
                print(json.dumps({"run_dir": str(rd), **{k: st.get(k) for k in
                                                         ("status", "updated_utc", "verdict", "stop", "error",
                                                          "elapsed_total_s")}}, indent=1))
            elif a.cmd == "execute-run":
                res = execute_run(rd, resume=a.resume)
                print(json.dumps({k: res.get(k) for k in ("verdict", "stopped")}))
                return 0 if not res.get("stopped") else 3
            else:
                print(json.dumps(publish_results(rd, allow_incomplete=a.allow_incomplete), indent=1))
    except (WsError, hs.StoreError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
