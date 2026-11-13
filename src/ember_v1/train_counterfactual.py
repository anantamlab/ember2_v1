"""P1-17 Phase 7: one bounded, preregistered batch of real-scope step-0 counterfactual studies.

Preregistration: `data/protocols/p1-17-counterfactual-batch.json` (FROZEN before the batch runs).
Source: verified groups of `p1-17-train-full@1` (collection gate of the P1-14 dataset protocol).
State: step 0, the matched initial state. Interventions: the distinct STORED step-0 joint maps of the five
teachers (`counterfactual.step0_study`; no live teacher call, so T2's budget-dependent decisions are not re-run).
Continuations: T1 and T4 (registry-pinned, retained). Replicates: 3 under stochastic fire (matched forks), 1 under
deterministic fire. Horizon: the scenario's max_steps (to termination; a branch still active is censored).

Every study is written once, then verified by `verify_shared_state_study` with the continuation RE-RUN (every
post-intervention action must be chosen again), then turned into class C/D records by `preferences`. Preference
approval is disabled in the dataset protocol: every class C record stays `candidate_not_approved`, weight 0.
No retry: a failed study is recorded and counted. A larger batch needs a new preregistration.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import counterfactual as cf
from . import multiteacher_dataset as ds
from .sim.canonical import digest
from .teacher import registry as reg

VERSION = "p1-17-cf-batch@1.0.0"
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "data" / "protocols" / "p1-17-counterfactual-batch.json"
PLAN_FILE = "data/collection/p1-17-train-full@1/plan.json"
TRAIN_FILE = "data/splits/p1-02-v1/train.json"


class BatchBlocked(RuntimeError):
    """A gate failed; nothing was written."""


def _read(p: Path) -> Any:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _write_once(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "x", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


def select_scenarios(plan: Mapping[str, Any], per_bin: int) -> list[str]:
    """The `per_bin` lowest-rank scenarios of each people bin (rank = the plan's preregistered salted hash)."""
    from .train_collect import N_BINS, _bin
    out = []
    for b in N_BINS:
        inb = sorted((s for s in plan["scenarios"] if _bin(s["people"]) == b), key=lambda s: s["rank"])
        out += [s["scenario_id"] for s in inb[:per_bin]]
    return out


def jobs_of(spec: Mapping[str, Any], plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    by = {s["scenario_id"]: s for s in plan["scenarios"]}
    out = []
    for sid in spec["selection"]["scenario_ids"]:
        s = by[sid]
        reps = spec["replicates"][s["fire_model"]]
        for c in spec["continuations"]:
            out.append({"scenario_id": sid, "continuation": c["name"], "continuation_version": c["version"],
                        "replicates": reps, "horizon": s["max_steps"], "family_id": s["family_id"],
                        "lineage_id": s["lineage_id"], "people": s["people"], "fire_model": s["fire_model"]})
    out.sort(key=lambda j: (-j["people"] * j["replicates"], j["scenario_id"], j["continuation"]))
    return out


def check_spec(spec: Mapping[str, Any], plan: Mapping[str, Any], verification: Mapping[str, Any]) -> list[str]:
    p = []
    if spec.get("status") != "FROZEN":
        p.append("batch preregistration is not FROZEN")
    if spec["source"]["plan_hash"] != plan["plan_hash"] or spec["source"]["plan_hash"] != verification["plan_hash"]:
        p.append("batch source plan is not the verified collection's plan")
    if spec["source"]["verification_hash"] != verification.get("verification_hash"):
        p.append("batch source verification hash differs")
    if verification.get("full_train_verified") is not True:
        p.append("collection is not full_train_verified")
    if select_scenarios(plan, spec["selection"]["per_bin"]) != spec["selection"]["scenario_ids"]:
        p.append("scenario selection does not recompute from the plan")
    for c in spec["continuations"]:
        try:
            i = reg.identity(c["name"], c["version"])
            if i.status != reg.RETAINED:
                p.append(f"continuation {c['name']} {c['version']} is not retained")
        except reg.RegistryError as e:
            p.append(f"continuation {c['name']}: {e}")
    if digest({k: v for k, v in spec.items() if k != "spec_hash"}) != spec.get("spec_hash"):
        p.append("batch preregistration hash does not recompute")
    return p


# ----- worker -------------------------------------------------------------------------------------------------
def _study(job: Mapping[str, Any]) -> dict[str, Any]:
    sid, cname = job["scenario_id"], job["continuation"]
    t0 = time.time()
    c0 = time.process_time()
    out = {"scenario_id": sid, "continuation": cname, "status": "failed", "verified": False, "records": 0}
    try:
        cls = reg.teacher_class(cname, job["continuation_version"])
        group = _read(Path(job["group_path"]))
        scenario = job["scenario"]
        study = cf.step0_study(scenario, group, family_id=job["family_id"], lineage_id=job["lineage_id"],
                               horizon=job["horizon"], max_horizon=job["horizon"], replicates=job["replicates"],
                               continuation_factory=cls, cpu_ms_per_branch_cap=job["cpu_ms_per_branch_cap"])
        _write_once(Path(job["study_path"]), json.dumps(study, sort_keys=True, separators=(",", ":")))
        out["status"] = study["status"]
        v = cf.verify_shared_state_study(study, scenario, group, continuation_factory=cls)
        out["verified"] = bool(v.get("verified"))
        recs = ds.preferences(study, scenario, group)
        _write_once(Path(job["records_path"]),
                    "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in recs))
        out["records"] = len(recs)
        if study["status"] == "contrasted":
            rep = study["report"]
            out.update({"alternatives": len(rep["branches"]),
                        "proposers": sorted("+".join(b["sources"]["proposed_by"]) for b in rep["branches"].values()),
                        "verdicts": dict(Counter(c["verdict"] for c in rep["contrasts"])),
                        "censored_contrasts": sum(1 for c in rep["contrasts"] if c["censored"]),
                        "classes": dict(Counter(r["class"] for r in recs)),
                        "continuation_cpu_ms": round(sum(sum(b["runtime"]["continuation_cpu_ms"])
                                                         for b in rep["branches"].values()), 1),
                        "report_hash": rep["report_hash"]})
    except Exception as e:  # noqa: BLE001 - every failure is recorded, never retried
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()[-2000:]
    out["wall_s"] = round(time.time() - t0, 2)
    out["process_cpu_s"] = round(time.process_time() - c0, 2)
    return out


def run(*, root: Path, verification_path: Path, workers: int | None = None,
        spec_path: Path = SPEC_PATH, plan_path: Path | None = None,
        scenarios: Sequence[tuple[Mapping[str, Any], Any]] | None = None, group_root: Path | None = None
        ) -> dict[str, Any]:
    spec = _read(spec_path)
    plan = _read(plan_path or REPO / PLAN_FILE)
    verification = _read(verification_path)
    problems = check_spec(spec, plan, verification)
    out_root = Path(spec["storage"]["root"]) if root is None else root
    if out_root.exists():
        problems.append(f"{out_root} exists: the batch is write-once (no rerun)")
    if problems:
        raise BatchBlocked("; ".join(problems))
    group_root = group_root or Path(plan["storage"]["root"])
    if scenarios is None:
        from .train_collect import collection_scenarios
        sub = {**plan, "scenarios": [s for s in plan["scenarios"] if s["scenario_id"] in
                                     set(spec["selection"]["scenario_ids"])]}
        scenarios = collection_scenarios(_read(REPO / TRAIN_FILE), sub)
    by_id = {e["scenario_id"]: s for e, s in scenarios}
    jobs = []
    for j in jobs_of(spec, plan):
        name = f"{j['scenario_id']}-{j['continuation']}"
        jobs.append({**j, "scenario": by_id[j["scenario_id"]],
                     "group_path": str(group_root / "groups" / f"{j['scenario_id']}.json"),
                     "cpu_ms_per_branch_cap": spec["cpu_ms_per_branch_cap"],
                     "study_path": str(out_root / "studies" / f"{name}.json"),
                     "records_path": str(out_root / "records" / f"{name}.jsonl")})
    out_root.mkdir(parents=True)
    _write_once(out_root / "batch_start.json", json.dumps(
        {"version": VERSION, "spec_hash": spec["spec_hash"], "jobs": len(jobs),
         "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1, sort_keys=True))
    t0 = time.time()
    n = workers or spec["concurrency"]
    if n <= 1:
        results = [_study(j) for j in jobs]
    else:
        with mp.get_context("spawn").Pool(n) as pool:
            results = list(pool.imap_unordered(_study, jobs, chunksize=1))
    results.sort(key=lambda r: (r["scenario_id"], r["continuation"]))
    summary = summarize(spec, results, out_root)
    body = {"version": VERSION, "spec_hash": spec["spec_hash"], "wall_s": round(time.time() - t0, 1),
            "results": [{k: v for k, v in r.items() if k != "traceback"} for r in results], **summary}
    for r in results:
        if "traceback" in r:
            _write_once(out_root / "failures" / f"{r['scenario_id']}-{r['continuation']}.txt", r["traceback"])
    _write_once(out_root / "batch.json", json.dumps({**body, "batch_hash": digest(body)}, indent=1, sort_keys=True))
    return {**body, "batch_hash": digest(body)}


def summarize(spec: Mapping[str, Any], results: Sequence[Mapping[str, Any]], root: Path) -> dict[str, Any]:
    recs = [json.loads(x) for p in sorted((root / "records").glob("*.jsonl"))
            for x in p.read_text(encoding="utf-8").splitlines() if x]
    weighted = ds.assign_weights(recs) if recs else []
    v = ds.validate_dataset(weighted) if weighted else {"ok": True, "problems": [], "audit": {}}
    by_cont: dict[str, Counter] = {}
    for r in results:
        c = by_cont.setdefault(r["continuation"], Counter())
        c["studies"] += 1
        c[r["status"]] += 1
        c["verified"] += bool(r["verified"])
        for k, n in (r.get("verdicts") or {}).items():
            c[f"verdict_{k}"] += n
        c["censored_contrasts"] += r.get("censored_contrasts", 0)
    return {"studies": len(results), "failed": sum(r["status"] == "failed" for r in results),
            "verified": sum(bool(r["verified"]) for r in results),
            "all_verified": all(r["verified"] for r in results),
            "by_continuation": {k: dict(sorted(c.items())) for k, c in sorted(by_cont.items())},
            "records": len(recs), "record_classes": dict(Counter(r["class"] for r in recs)),
            "records_valid": v["ok"], "record_problems": v["problems"][:20],
            "approved_preferences": sum(1 for r in weighted if r["class"] == "C"
                                        and r["approval"]["status"] == "approved"),
            "cpu_s_total": round(sum(r.get("process_cpu_s", 0) for r in results), 1),
            "training_eligible": False}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ember_v1.train_counterfactual", description=__doc__.splitlines()[0])
    ap.add_argument("--verification", required=True)
    ap.add_argument("--root", default=None, help="default: the preregistered storage root")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args(argv)
    try:
        res = run(root=Path(a.root) if a.root else None, verification_path=Path(a.verification), workers=a.workers)
    except BatchBlocked as e:
        print(json.dumps({"status": "BLOCKED", "reason": str(e)}))
        return 2
    print(json.dumps({k: res[k] for k in ("studies", "failed", "verified", "all_verified", "records",
                                          "record_classes", "records_valid", "batch_hash")}))
    return 0 if res["all_verified"] and res["records_valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
