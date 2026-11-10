"""Read-only preflight for attempt 1 of p1-12-train-pilot@1 (eight TRAIN scenarios, five teachers, CPU).

Runs NO teacher and writes nothing inside the execution checkout. Run it from ANOTHER checkout (or a copy of
this file) against the execution checkout, so the execution tree stays byte-clean:

    python scripts/pilot_preflight.py --repo <execution checkout> --out <path outside it>/preflight.json

It imports `ember_v1` from `<repo>/src` (refused if it resolves anywhere else), with bytecode writing off,
and checks, item by item: git identity and ancestry, the frozen plan and its hash, a live plan rebuild
(scenario, teacher, simulator, generator and collector identities), regeneration of the eight scenarios,
teacher pins, attempt markers and partial outputs (this checkout, every worktree, origin/main), a clean tree,
machine capacity, and finally the runner's own gate `train_pilot.run_preconditions` (the source of truth).

Status: READY_AUTHORIZED (every gate passes, including a verified GitHub approval), PENDING_AUTHORIZATION
(every non-authorization check passes; only the authorization is missing or not yet approved) or BLOCKED.
Exit code 0 / 2 / 1. Unknown runner reasons count as BLOCKED (fail closed).
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PREFLIGHT_VERSION = "p1-12-pilot-preflight@1.0.0"
PLAN_HASH = "sha256:eb8dbe293db7f1c1e1b4f342c07f21853031e9e6ab5419eda2833c51f761b664"
FREEZE_COMMIT = "827e88fa79d9fae9667f2fda590b4b23852988fc"
EXECUTION_COMMIT = "1a0128ca5711557fba9db837f0ed251233bf6359"
OUT_DIR = "data/collection/p1-12-train-pilot@1"
PLAN_PATH = f"{OUT_DIR}/plan.json"
AUTH_PATH = f"{OUT_DIR}/authorization.json"
# Machine requirements of this preflight (NOT part of the frozen plan): one supervisor plus five concurrent
# teacher processes per group; the CPU caps are CPU seconds, so fewer cores only lengthen wall time.
MIN_LOGICAL_CPUS = 6
MIN_AVAILABLE_RAM_GIB = 4.0
MIN_FREE_DISK_GIB = 1.0
SUPPORTED_PYTHON = ((3, 11), (3, 12))
AUTH_PREFIXES = ("authorization:", "approval:", "verified approvers:")
READY, PENDING, BLOCKED = "READY_AUTHORIZED", "PENDING_AUTHORIZATION", "BLOCKED"


def git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=120)


def git_out(repo: Path, *args: str) -> str:
    r = git(repo, *args)
    return r.stdout.strip() if r.returncode == 0 else ""


def is_ancestor(repo: Path, older: str, newer: str) -> bool:
    return bool(older and newer) and git(repo, "merge-base", "--is-ancestor", older, newer).returncode == 0


def memory_gib() -> dict[str, float | None]:
    """Total and available physical memory in GiB (None if the platform does not tell)."""
    try:
        if sys.platform == "win32":
            class MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MemStatus()
            m.dwLength = ctypes.sizeof(MemStatus)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return {"total": None, "available": None}
            return {"total": round(m.ullTotalPhys / 2 ** 30, 2), "available": round(m.ullAvailPhys / 2 ** 30, 2)}
        if Path("/proc/meminfo").exists():
            kv = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                k, _, v = line.partition(":")
                kv[k] = int(v.split()[0]) * 1024
            return {"total": round(kv["MemTotal"] / 2 ** 30, 2),
                    "available": round(kv.get("MemAvailable", kv.get("MemFree", 0)) / 2 ** 30, 2)}
        if sys.platform == "darwin":
            total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout)
            return {"total": round(total / 2 ** 30, 2), "available": None}
    except (OSError, ValueError, KeyError, AttributeError):
        pass
    return {"total": None, "available": None}


def machine_check(repo: Path, mem: dict[str, float | None] | None = None, cpus: int | None = None,
                  python: tuple[int, int] | None = None) -> dict[str, Any]:
    mem = memory_gib() if mem is None else mem
    cpus = os.cpu_count() if cpus is None else cpus
    python = tuple(sys.version_info[:2]) if python is None else python
    free = round(shutil.disk_usage(repo).free / 2 ** 30, 2)
    problems = []
    if not cpus or cpus < MIN_LOGICAL_CPUS:
        problems.append(f"logical CPUs {cpus} < {MIN_LOGICAL_CPUS}")
    if mem.get("available") is None or mem["available"] < MIN_AVAILABLE_RAM_GIB:
        problems.append(f"available RAM {mem.get('available')} GiB < {MIN_AVAILABLE_RAM_GIB} (or unknown)")
    if free < MIN_FREE_DISK_GIB:
        problems.append(f"free disk {free} GiB < {MIN_FREE_DISK_GIB}")
    if python not in SUPPORTED_PYTHON:
        problems.append(f"Python {python[0]}.{python[1]} is not 3.11 or 3.12")
    return {"pass": not problems, "problems": problems, "logical_cpus": cpus, "ram_gib": mem,
            "free_disk_gib": free, "python": platform.python_version(), "platform": platform.platform(),
            "requirements": {"min_logical_cpus": MIN_LOGICAL_CPUS, "min_available_ram_gib": MIN_AVAILABLE_RAM_GIB,
                             "min_free_disk_gib": MIN_FREE_DISK_GIB, "python": ["3.11", "3.12"],
                             "source": "this preflight, not the frozen plan"},
            "note": "run on an otherwise idle machine; CPU load is not measured here"}


def attempt_artifacts(root: Path) -> list[str]:
    """Every attempt marker or attempt directory under <root>/OUT_DIR."""
    d = root / OUT_DIR
    if not d.is_dir():
        return []
    return sorted(str(p) for p in d.glob("attempt-*"))


def worktrees(repo: Path) -> list[Path]:
    out = git_out(repo, "worktree", "list", "--porcelain")
    return [Path(line[len("worktree "):]) for line in out.splitlines() if line.startswith("worktree ")]


def classify(reasons: list[str], own_failures: list[str]) -> str:
    """Overall status. PENDING only when the runner's every reason is an authorization reason (fail closed)."""
    if own_failures:
        return BLOCKED
    if not reasons:
        return READY
    if all(r.strip().startswith(AUTH_PREFIXES) for r in reasons):
        return PENDING
    return BLOCKED


def import_target(repo: Path) -> Any:
    sys.dont_write_bytecode = True
    src = str(repo / "src")
    sys.path.insert(0, src)
    for name in [m for m in sys.modules if m == "ember_v1" or m.startswith("ember_v1.")]:
        del sys.modules[name]
    import ember_v1
    if Path(ember_v1.__file__).resolve().parent != (repo / "src" / "ember_v1").resolve():
        raise SystemExit(f"ember_v1 imported from {ember_v1.__file__}, not from {src}")
    from ember_v1 import train_pilot
    return train_pilot


def preflight(repo: Path, *, auth_commit: str | None = None, trusted_ref: str = "origin/main",
              live: Any = None, machine: dict[str, Any] | None = None) -> dict[str, Any]:
    repo = repo.resolve()
    tp = import_target(repo)
    from ember_v1 import trust
    from ember_v1.teacher import confirmation as C
    from ember_v1.teacher import registry as reg
    checks: dict[str, dict[str, Any]] = {}

    def record(name: str, ok: bool, **detail: Any) -> None:
        checks[name] = {"pass": bool(ok), **detail}

    head = git_out(repo, "rev-parse", "HEAD")
    main = git_out(repo, "rev-parse", trusted_ref)
    record("git_identity", head == EXECUTION_COMMIT and bool(main), head=head, expected=EXECUTION_COMMIT,
           trusted_ref=trusted_ref, trusted_ref_sha=main)
    record("git_ancestry", is_ancestor(repo, FREEZE_COMMIT, head) and is_ancestor(repo, head, trusted_ref),
           freeze_is_ancestor_of_head=is_ancestor(repo, FREEZE_COMMIT, head),
           head_in_trusted_ref=is_ancestor(repo, head, trusted_ref))
    changed = [p for p in git_out(repo, "diff", "--name-only", FREEZE_COMMIT, head).splitlines() if p]
    record("no_unauthorized_modifications", not [p for p in changed if p.startswith(("src/", "data/"))],
           paths_changed_since_freeze=changed,
           tree_equal_to_freeze=git_out(repo, "rev-parse", f"{head}^{{tree}}")
           == git_out(repo, "rev-parse", f"{FREEZE_COMMIT}^{{tree}}"))
    dirty = C.tree_problems(repo)
    record("clean_tree", not dirty, problems=dirty[:50])

    plan_ref = {"kind": "git", "commit": FREEZE_COMMIT, "path": PLAN_PATH}
    plan, meta = trust.load_trusted_json(repo, plan_ref, trusted_ref=trusted_ref)
    head_blob = git_out(repo, "rev-parse", f"{head}:{PLAN_PATH}")
    record("plan_hash", plan.get("plan_hash") == PLAN_HASH and not tp.check_plan(plan)
           and head_blob == git_out(repo, "rev-parse", f"{FREEZE_COMMIT}:{PLAN_PATH}"),
           plan_hash=plan.get("plan_hash"), status=plan.get("status"), plan_problems=tp.check_plan(plan),
           plan_sha256_lf=meta["sha256"], plan_blob_unchanged_at_head=bool(head_blob))

    t0 = time.monotonic()
    fresh = (live or (lambda: tp.live_plan(repo)))()
    rebuild_s = round(time.monotonic() - t0, 1)
    body_old, body_new = tp.plan_body(plan), tp.plan_body(fresh)
    record("live_rebuild", fresh.get("plan_hash") == plan["plan_hash"], rebuilt=fresh.get("plan_hash"),
           differs=sorted(k for k in body_old if body_old[k] != body_new.get(k)), seconds=rebuild_s)
    record("scenario_fingerprints", fresh.get("scenarios") == plan["scenarios"],
           scenarios=[{"scenario_id": s["scenario_id"], "scenario_hash": s["scenario_hash"]}
                      for s in plan["scenarios"]])
    record("teacher_fingerprints", tp.fingerprints_of(fresh["teacher_pins"]) == tp.fingerprints_of(plan["teacher_pins"])
           and not reg.check_pins() and not reg.check_registry(),
           teachers=tp.fingerprints_of(plan["teacher_pins"]), pin_problems=reg.check_pins(),
           registry_problems=reg.check_registry())
    record("simulator_collector_identities", fresh.get("identities") == plan["identities"],
           identities=plan["identities"])
    try:
        train = json.loads((repo / "data/splits/p1-02-v1/train.json").read_text(encoding="utf-8"))
        regen = tp.pilot_scenarios(train, plan)
        record("scenarios_regenerate", len(regen) == tp.SCENARIOS, count=len(regen))
    except (ValueError, OSError, KeyError) as e:
        record("scenarios_regenerate", False, error=str(e))

    local = attempt_artifacts(repo)
    others = {str(w): attempt_artifacts(w) for w in worktrees(repo)}
    committed = [p for p in git_out(repo, "ls-tree", "-r", "--name-only", trusted_ref, "--", OUT_DIR).splitlines()
                 if Path(p).name.startswith("attempt-")]
    record("attempt_1_unused", not local and not any(others.values()) and not committed,
           local=local, other_worktrees={k: v for k, v in others.items() if v}, committed_on_trusted_ref=committed)
    m = machine or machine_check(repo)
    record("machine", m["pass"], **{k: v for k, v in m.items() if k != "pass"})

    auth_commit = auth_commit or main
    auth_present = git(repo, "cat-file", "-e", f"{auth_commit}:{AUTH_PATH}").returncode == 0
    auth_ref = {"kind": "git", "commit": auth_commit, "path": AUTH_PATH}
    try:
        tp.run_preconditions(repo, plan_ref, auth_ref, trusted_ref=trusted_ref, live=lambda: fresh)
        reasons: list[str] = []
    except tp.PilotBlocked as e:
        reasons = [r for r in str(e).split("; ") if r]
    own = sorted(k for k, v in checks.items() if not v["pass"])
    status = classify(reasons, own)
    return {"preflight_version": PREFLIGHT_VERSION, "status": status, "repo": str(repo),
            "plan_ref": plan_ref, "auth_ref": auth_ref, "authorization_present": auth_present,
            "failed_checks": own, "runner_gate_reasons": reasons, "checks": checks,
            "teacher_calls": 0, "writes_in_checkout": 0}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pilot_preflight.py", description=__doc__.splitlines()[0])
    ap.add_argument("--repo", required=True, type=Path, help="the execution checkout")
    ap.add_argument("--auth-commit", default=None, help="40-hex commit holding authorization.json "
                                                        "(default: the trusted ref)")
    ap.add_argument("--trusted-ref", default="origin/main")
    ap.add_argument("--out", type=Path, default=None, help="report path OUTSIDE the execution checkout")
    a = ap.parse_args(argv)
    repo = a.repo.resolve()
    if a.out is not None and repo in [a.out.resolve(), *a.out.resolve().parents]:
        raise SystemExit("--out must be outside the execution checkout (it would dirty the tree)")
    auth_commit = a.auth_commit
    if auth_commit is None:
        auth_commit = git_out(repo, "rev-parse", a.trusted_ref)
    rep = preflight(repo, auth_commit=auth_commit, trusted_ref=a.trusted_ref)
    text = json.dumps(rep, indent=1, sort_keys=True) + "\n"
    if a.out is not None:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text, encoding="utf-8", newline="\n")
    sys.stdout.write(json.dumps({"status": rep["status"], "failed_checks": rep["failed_checks"],
                                 "runner_gate_reasons": rep["runner_gate_reasons"]}, indent=1) + "\n")
    return {READY: 0, PENDING: 2}.get(rep["status"], 1)


if __name__ == "__main__":
    raise SystemExit(main())
