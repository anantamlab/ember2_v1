"""P1-18 Stage 3: matched five-teacher collection (T0-T4) on the p1-18-expand@1 scenarios, local CPU.

Same teachers, versions, seeds and per-decision CPU limits as the P1-17 full-TRAIN collection (its frozen plan);
the launch refuses if a teacher code fingerprint differs. One group-runner process per scenario spawns one
killable process per teacher (`pilot_supervisor.run_group`); the supervisor replays every stream in its own
simulator (`build_group`) and the row records `multiteacher.verify_fixture`. All five teachers start from the
same scenario seed (matched seeds).

Resumable: a scenario with a row is done (its group file sha256 is checked); a scenario that started without a
row (crash or stop) is rerun from scratch after its partial files move to `stale/` (a local CPU rerun costs no
money; the partial run is never used). Order (documented adaptive design, T2 is ~99% of CPU in P1-17): TRAIN
partition first, then val, then diagnostic; inside a partition longest predicted T2 cost first (people x
max_steps), which shortens the tail at a fixed concurrency. Teacher behavior is not changed.

This module does not edit `train_collect` / `multiteacher` (their fingerprints are pinned in P1-17's frozen plan);
it registers its own scope `p1-18-expand@1` in `multiteacher.SCOPES` at runtime (`register_scope`).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import shutil
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import expand as E
from . import train_collect as tc
from . import train_pilot as tp
from .sim.canonical import digest

VERSION = "p1-18-expand-collect@1.0.0"
NAMESPACE = E.NAMESPACE
SCOPE = E.NAMESPACE
DEV_SCOPE = "p1-18-dev-teachers@1"         # T0-T4 on the 48 DEV scenarios (evaluation only, never training)
SCOPES = (SCOPE, DEV_SCOPE)
P17_PLAN = "data/collection/p1-17-train-full@1/plan.json"
STORAGE_ROOT = "D:/Coding/ember-evidence/p1-18-expand@1"
ORDER = ("train", "val", "diagnostic")
GROUP_WALL_S = 7200
POLL_S = 1.0


class ExpandCollectError(RuntimeError):
    pass


def register_scope() -> None:
    from . import multiteacher as mt
    mt.SCOPES = (*mt.SCOPES, *(s for s in SCOPES if s not in mt.SCOPES))


def paths(root: Path) -> dict[str, Path]:
    return {"root": root, "log": root / "log.jsonl", "started": root / "started", "groups": root / "groups",
            "rows": root / "rows", "stale": root / "stale", "logs": root / "logs",
            "progress": root / "progress.json", "stop": root / "STOP"}


def build_plan(repo: Path, manifest: Mapping[str, Any], *, concurrency: int, min_free_gib: float,
               scope: str = SCOPE) -> dict[str, Any]:
    p17 = json.loads((repo / P17_PLAN).read_text(encoding="utf-8"))
    pins = p17["teacher_pins"]
    rank = {p: i for i, p in enumerate(ORDER)}
    sc = sorted(manifest["scenarios"], key=lambda e: (rank.get(e["partition"], len(ORDER)),
                                                      -e["n_people"] * e["diagnostics"]["max_steps"], e["scenario_id"]))
    body = {"version": VERSION, "namespace": scope, "scope": scope, "manifest_hash": manifest["manifest_hash"],
            "expand_plan_hash": manifest["plan_hash"], "teacher_pins": pins,
            "teacher_pins_source": {"plan": P17_PLAN, "plan_hash": p17["plan_hash"]},
            "identities": tp.code_identities(),
            "limits": {"decision_cpu_s": p17["limits"]["decision_cpu_s"], "group_wall_s": GROUP_WALL_S,
                       "max_concurrent_groups": concurrency, "min_free_ram_gib_to_start": min_free_gib, "retries": 0},
            "order_rule": "partition train, val, diagnostic; then people x max_steps descending; then scenario_id",
            "scenarios": [{k: e[k] for k in ("scenario_id", "scenario_hash", "lineage_id", "family_id", "partition",
                                             "stratum", "n_people")} for e in sc]}
    return {**body, "plan_hash": digest(body)}


def check_teachers(plan: Mapping[str, Any]) -> list[str]:
    now, want = tp.fingerprints_of(tp.teacher_pins()), tp.fingerprints_of(plan["teacher_pins"])
    return [f"{t}: teacher pin differs from P1-17" for t in want if now.get(t) != want[t]]


# ----- group runner (child process) ----------------------------------------------------------------------------
def _group_main(job: Mapping[str, Any], stop_event: Any, target: Callable[..., None] | None) -> None:
    """One matched group: five teacher workers, replay, verify, write group + row (P1-17 runner, own scope)."""
    from . import multiteacher as mt
    from . import pilot_supervisor as ps
    from .sim import Scenario
    from .teacher import registry as reg
    from .teacher.confirmation import AttemptLog
    register_scope()
    t_start = tc._utc()
    sid = job["scenario_id"]
    scn = Scenario.from_dict(job["scenario"])
    base = AttemptLog(Path(job["log"]))
    mem = tc._MemSampler()

    def log(event: str, **f: Any) -> None:
        if event == "worker_start":
            mem.pids[f["teacher"]] = f["pid"]
        base(event, scenario_id=sid, **f)

    parent = mp.parent_process()

    def stop() -> bool:
        return stop_event.is_set() or (parent is not None and not parent.is_alive())

    mem.pids["runner"] = os.getpid()
    mem.start()
    insts = {}
    for j in job["jobs"]:
        cls = reg.teacher_class(j["name"], j["identity"]["version"])
        insts[j["name"]] = cls() if j["seed"] is None else cls(seed=j["seed"])
    identities = {n: p.identity.to_dict() for n, p in insts.items()}
    seeds = {j["name"]: j["seed"] for j in job["jobs"]}
    prov = mt.provenance(scn, insts, {"kind": job["scope"], "manifest_hash": job["manifest_hash"], "scenario_id": sid,
                                      "lineage_id": job["lineage_id"], "partition": job["partition"]})
    run = ps.run_group(scn, job["jobs"], decision_cpu_s=job["limits"]["decision_cpu_s"],
                       group_wall_s=job["limits"]["group_wall_s"], log=log, target=target, stop_requested=stop)
    extra = {"collection": {"plan_hash": job["plan_hash"], "namespace": job["scope"], "scenario_id": sid,
                            "partition": job["partition"]}}
    rep, err = None, None
    try:
        rep = ps.build_group(scn, run, identities, seeds, prov, extra=extra, run_kind="primary", scope=job["scope"])
    except ps.GroupIntegrityError as e:
        err = f"integrity_error: {e}"
    resources = mem.stop()
    out = Path(job["group_file"])
    doc = rep if rep is not None else {"scenario_id": sid, "integrity_error": err, "group_complete": False,
                                       "training_eligible": False, "runtime": {"supervision": run["supervision"]},
                                       "failures": run["failures"], "extra": extra}
    tc.write_once(out, doc)
    row = tp._group_summary(sid, rep, scn, run, "ran", err)
    row.update(session=job["session"], partition=job["partition"], stratum=job["stratum"],
               n_people=len(scn.people), started_utc=t_start, ended_utc=tc._utc(), interrupted=run["interrupted"],
               process_cpu_s=run["supervision"]["process_cpu_s"],
               peak_live_decision_cpu_ms=run["supervision"]["peak_live_decision_cpu_ms"], **resources,
               file_sha256=tc._sha256_file(out),
               final_transition_hashes={n: r["final_transition_hash"] for n, r in rep["policies"].items()}
               if rep else None,
               outcomes={n: r["outcome"] for n, r in rep["policies"].items()} if rep else None,
               failures=run["failures"])
    tc.write_once(Path(job["row_file"]), row)


# ----- driver --------------------------------------------------------------------------------------------------
def _state(plan: Mapping[str, Any], P: Mapping[str, Path]) -> dict[str, str]:
    out = {}
    for s in plan["scenarios"]:
        sid = s["scenario_id"]
        out[sid] = ("row" if (P["rows"] / f"{sid}.json").exists() else
                    "started" if (P["started"] / f"{sid}.json").exists() else "pending")
    return out


def run(repo: Path, root: Path, *, concurrency: int, min_free_gib: float = 2.0, limit: int | None = None,
        target: Callable[..., None] | None = None, manifest: Mapping[str, Any] | None = None,
        scenarios: Sequence[tuple[Mapping[str, Any], Any]] | None = None, poll_s: float = POLL_S,
        scope: str = SCOPE) -> dict[str, Any]:
    """Run (or resume) the collection. `limit` caps how many new groups start in this session."""
    register_scope()
    manifest = manifest or json.loads((repo / E.OUT_DIR / "manifest.json").read_text(encoding="utf-8"))
    P = paths(root)
    root.mkdir(parents=True, exist_ok=True)
    for k in ("started", "groups", "rows", "stale", "logs"):
        P[k].mkdir(exist_ok=True)
    plan_file = root / "plan.json"
    if plan_file.exists():
        plan = json.loads(plan_file.read_text(encoding="utf-8"))
        if plan["manifest_hash"] != manifest["manifest_hash"]:
            raise ExpandCollectError("storage root holds another manifest's collection")
    else:
        plan = build_plan(repo, manifest, concurrency=concurrency, min_free_gib=min_free_gib, scope=scope)
        tc.write_once(plan_file, plan)
    bad = check_teachers(plan)
    if bad:
        raise ExpandCollectError("; ".join(bad))
    from .teacher.confirmation import AttemptLog
    sessions = sorted(root.glob("session-*.json"))
    session = 1 + sum(1 for p in sessions if not p.name.endswith("-end.json"))
    tc.write_once(root / f"session-{session}.json", {"session": session, "plan_hash": plan["plan_hash"],
                                                     "started_utc": tc._utc(), "pid": os.getpid(),
                                                     "concurrency": concurrency, "limit": limit,
                                                     "platform": {"python": platform.python_version(),
                                                                  "cpu_count": os.cpu_count()}})
    alog = AttemptLog(P["log"])

    def log(event: str, **f: Any) -> None:
        alog(event, session=session, **f)

    st = _state(plan, P)
    for sid, v in st.items():
        if v == "row":
            row = tc._read(P["rows"] / f"{sid}.json")
            g = P["groups"] / f"{sid}.json"
            if not g.exists() or row.get("file_sha256") != tc._sha256_file(g):
                raise ExpandCollectError(f"{sid}: finished group fails its row sha256")
        elif v == "started":           # crashed or stopped: rerun from scratch, keep the partial files aside
            for k in ("started", "groups", "logs"):
                src = P[k] / (f"{sid}.jsonl" if k == "logs" else f"{sid}.json")
                if src.exists():
                    shutil.move(str(src), str(P["stale"] / f"s{session}-{k}-{src.name}"))
            log("rerun_after_interruption", scenario_id=sid)
            st[sid] = "pending"
    by_id = {e["scenario_id"]: (e, s) for e, s in (scenarios if scenarios is not None
                                                     else E.materialize(manifest))}
    jobs = tp._jobs(plan)
    pending = [s for s in plan["scenarios"] if st[s["scenario_id"]] == "pending"]
    if limit is not None:
        pending = pending[:limit]

    def job(s: Mapping[str, Any]) -> dict[str, Any]:
        e, scn = by_id[s["scenario_id"]]
        sid = s["scenario_id"]
        return {"scenario_id": sid, "session": session, "scenario": scn.to_dict(), "scenario_hash": scn.content_hash(),
                "lineage_id": e["lineage_id"], "partition": e["partition"], "stratum": e["stratum"],
                "manifest_hash": plan["manifest_hash"], "plan_hash": plan["plan_hash"], "limits": plan["limits"],
                "scope": plan.get("scope", SCOPE),
                "jobs": jobs, "log": str(P["logs"] / f"{sid}.jsonl"), "group_file": str(P["groups"] / f"{sid}.json"),
                "row_file": str(P["rows"] / f"{sid}.json")}

    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()
    running: dict[str, Any] = {}
    queue = [job(s) for s in pending]
    log("session_start", pending=len(queue), concurrency=concurrency)
    t0, last, ram_waits, host_min, halt = time.monotonic(), 0.0, 0, None, None
    counts: Counter = Counter()
    try:
        while queue or running:
            if P["stop"].exists() and not halt:
                halt = "operator_stop"
                log("admission_closed", reason=halt)
            if queue and not halt and len(running) < concurrency:
                free = tc.host_free_ram_gib()
                host_min = free if host_min is None or (free is not None and free < host_min) else host_min
                if free is None or free >= min_free_gib:
                    j = queue.pop(0)
                    tc.write_once(P["started"] / f"{j['scenario_id']}.json",
                                  {"scenario_id": j["scenario_id"], "session": session, "started_utc": tc._utc(),
                                   "scenario_hash": j["scenario_hash"], "host_free_gib": free})
                    p = ctx.Process(target=_group_main, args=(j, stop_event, target),
                                    name=f"p1-18-group-{j['scenario_id']}", daemon=False)
                    p.start()
                    running[j["scenario_id"]] = (p, j)
                    continue
                ram_waits += 1
            if halt and queue:
                queue = []
            for sid, (p, j) in list(running.items()):
                if p.is_alive():
                    continue
                p.join()
                del running[sid]
                ok = Path(j["row_file"]).exists()
                counts["row" if ok else "runner_crashed"] += 1
                log("group_end" if ok else "group_runner_crashed", scenario_id=sid, exit_code=p.exitcode)
            now = time.monotonic()
            if now - last > 30:
                last = now
                s2 = _state(plan, P)
                tc._replace_json(P["progress"], {"session": session, "updated_utc": tc._utc(),
                                                 "planned": len(s2), "finished": sum(v == "row" for v in s2.values()),
                                                 "running": sorted(running), "queued": len(queue),
                                                 "host_free_gib": tc.host_free_ram_gib(), "halt": halt})
            time.sleep(poll_s)
    except KeyboardInterrupt:
        stop_event.set()
        for p, _ in running.values():
            p.join(60)
        halt = "interrupted"
    end = {"session": session, "ended_utc": tc._utc(), "wall_s": round(time.monotonic() - t0, 3), "halt": halt,
           "groups": dict(counts), "ram_waits": ram_waits, "host_min_free_gib": host_min}
    tc.write_once(root / f"session-{session}-end.json", end)
    log("session_end", **{k: v for k, v in end.items() if k != "session"})
    return end


def dev_manifest(repo: Path) -> tuple[dict[str, Any], list[tuple[dict[str, Any], Any]]]:
    """The 48 DEV scenarios (hash-checked, regenerated) in this collector's manifest shape (partition `dev`)."""
    from .teacher.evaluate import load_dev
    raw = json.loads((repo / "data/splits/p1-02-v1/dev.json").read_text(encoding="utf-8"))
    by_lin = {e["lineage_id"]: e for e in raw["scenarios"]}
    meta, dev = load_dev(repo / "data/splits/p1-02-v1/index.json", repo / "data/splits/p1-02-v1/dev.json")
    sc = []
    for d in dev:
        e = by_lin[d["lineage_id"]]
        sc.append(({**e, "partition": "dev", "stratum": "dev", "n_people": len(d["scenario"].people)}, d["scenario"]))
    return {"manifest_hash": meta["manifest_hash"], "plan_hash": raw["plan_hash"],
            "scenarios": [e for e, _ in sc]}, sc


def summarize(root: Path) -> dict[str, Any]:
    """Coverage and per-teacher outcome totals over the finished rows (no teacher is ranked here)."""
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    rows = [tc._read(p) for p in sorted((root / "rows").glob("*.json"))]
    ok = [r for r in rows if r.get("group_complete") and r.get("verified")]
    by = Counter((r["partition"], r["stratum"]) for r in ok)
    t2 = [r.get("t2_cpu_s") or 0.0 for r in rows]
    return {"planned": len(plan["scenarios"]), "rows": len(rows), "complete_verified": len(ok),
            "by_partition_stratum": {f"{a}/{b}": c for (a, b), c in sorted(by.items())},
            "t2_cpu_s_total": round(sum(t2), 1), "budget_limited_decisions":
                sum(sum((r.get("budget_limited_decisions") or {}).values()) for r in rows),
            "not_complete": sorted(r["scenario_id"] for r in rows if r not in ok)}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ember_v1.expand_collect")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--repo", type=Path, default=Path("."))
    r.add_argument("--root", type=Path, default=Path(STORAGE_ROOT))
    r.add_argument("--concurrency", type=int, default=12)
    r.add_argument("--min-free-gib", type=float, default=2.0)
    r.add_argument("--limit", type=int, default=None)
    d = sub.add_parser("dev")
    d.add_argument("--repo", type=Path, default=Path("."))
    d.add_argument("--root", type=Path, default=Path("D:/Coding/ember-evidence/p1-18-dev-teachers@1"))
    d.add_argument("--concurrency", type=int, default=8)
    s = sub.add_parser("summary")
    s.add_argument("--root", type=Path, default=Path(STORAGE_ROOT))
    a = ap.parse_args(argv)
    if a.cmd == "run":
        print(json.dumps(run(a.repo, a.root, concurrency=a.concurrency, min_free_gib=a.min_free_gib, limit=a.limit)))
    elif a.cmd == "dev":
        m, scs = dev_manifest(a.repo)
        print(json.dumps(run(a.repo, a.root, concurrency=a.concurrency, manifest=m, scenarios=scs, scope=DEV_SCOPE)))
    else:
        print(json.dumps(summarize(a.root), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
