"""Single frozen DEV comparison of T4 v1.1.0 (A2) and its A0/A1 ablations with frozen T1 (overnight P5).

    python -m ember_v1.teacher.t4_v2_eval [--pilot-report PATH] [--out PATH]

Preregistered in docs/P1_T4_RISK_PRIORITY_V2.md section 7. RUN ONCE. It refuses
to run unless the v2 pilot report (`t4_v2_pilot`) has gate decision
T4_V2_FULL_DEV_FEASIBLE for the same v1.1.0 identities, and it refuses to
overwrite an existing report. Same DEV gate as P1-03..P1-08
(`t1_eval.load_dev`): only the `dev` split of plan `p1-02-v1`. Train,
calibration, the locked test, blind sets and EMBERBench are never read.

The per-episode harness, evaluator-side metrics (frozen v1.0.0 risk module on
public observations), decision rule and worse-case listing are reused from
`t4_eval` unchanged. Added: v1.1 identities, own bootstrap seeds, and a check
that T1 reproduces every final transition hash of the committed T1 DEV
baseline (`data/baselines/p1-04-t1-dev-v1.json`). DEV was already used by
P1-03..P1-08; it is a development comparison, not a fresh confirmation set.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..sim.canonical import digest
from ..sim.scenario import Scenario
from . import evaluate as ev
from . import t1_eval
from .interface import ReferencePolicy
from .t1 import T1CongestionAware
from .t4_eval import (PAIRED_METRICS, GateError, risk_policy, run_episode, strata_totals, verdict, wall_limit,
                      worse)
from .t4_v11 import ABLATIONS_V11, T4DistancePriorityV11, T4NoPriorityV11, T4RiskPriorityV11

EVAL_VERSION = "t4-v11-vs-t1-dev@2.0.0"
DEFAULT_OUT = ev.REPO / "data/baselines/p1-08-t4-v2-dev.json"
DEFAULT_PILOT = ev.REPO / "data/baselines/p1-08-t4-v2-pilot.json"
T1_BASELINE = ev.REPO / "data/baselines/p1-04-t1-dev-v1.json"
POLICY_ORDER = ("T1", "A0", "A1", "A2")
COMPARISONS = (("A2", "T1"), ("A2", "A0"), ("A2", "A1"), ("A1", "A0"))
PRIMARY = ("A2", "T1", "casualties")
BOOTSTRAP_SEEDS = {f"{a}-{b}": f"t4v2-{a.lower()}-vs-{b.lower()}-family-cluster-bootstrap@2" for a, b in COMPARISONS}
PASS_DECISION = "T4_V2_FULL_DEV_FEASIBLE"


def policies() -> dict[str, ReferencePolicy]:
    return {"T1": T1CongestionAware(), "A0": T4NoPriorityV11(), "A1": T4DistancePriorityV11(),
            "A2": T4RiskPriorityV11()}


def check_gate_v2(pilot: dict[str, Any]) -> float:
    """Return the wall limit (s) if the v2 pilot permits the DEV run for these exact identities, else raise."""
    dec = pilot.get("gate", {}).get("decision")
    if dec != PASS_DECISION:
        raise GateError(f"v2 pilot gate decision is {dec!r}, not {PASS_DECISION}")
    want = {k: cls().identity.config_hash for k, cls in ABLATIONS_V11.items()}
    got = {k: pilot.get("teachers", {}).get(k, {}).get("config_hash") for k in want}
    if got != want:
        raise GateError("v2 pilot was run for different T4 v1.1 / ablation configurations")
    return wall_limit(pilot["estimate"]["upper"]["comparison_cpu_s_total"])


def t1_reference() -> dict[int, str]:
    base = json.loads(T1_BASELINE.read_text(encoding="utf-8"))
    return {r["position"]: r["T1"]["final_transition_hash"] for r in base["rows"]}


def evaluate(pilot_report: str | Path = DEFAULT_PILOT) -> dict[str, Any]:
    pilot = json.loads(Path(pilot_report).read_text(encoding="utf-8"))
    limit_s = check_gate_v2(pilot)
    cfg = dict(risk_policy().identity.config)
    ref = t1_reference()
    meta, items = t1_eval.load_dev()
    rows: list[dict[str, Any]] = []
    latencies: dict[str, list[float]] = {n: [] for n in POLICY_ORDER}
    t1_mismatch: list[int] = []
    start = time.perf_counter()
    complete = True
    for it in items:
        if time.perf_counter() - start > limit_s:
            complete = False
            break
        s: Scenario = it["scenario"]
        row: dict[str, Any] = {"position": it["position"], "lineage_id": it["lineage_id"],
                               "family_id": it["family_id"], "scenario_hash": it["scenario_hash"],
                               "simulator_seed": s.seed, "label": it["label"], "grid": [s.height, s.width],
                               "fire_model": s.fire_model, "enter_fire": s.enter_fire, "max_steps": s.max_steps}
        for name, pol in policies().items():
            r, lat = run_episode(s, pol, cfg)
            row[name] = r
            latencies[name].extend(lat)
        if ref.get(it["position"]) != row["T1"]["final_transition_hash"]:
            t1_mismatch.append(it["position"])
        rows.append(row)
        print(json.dumps({"position": it["position"], **{f"{n}_cas": row[n]["casualties"] for n in POLICY_ORDER}}),
              flush=True)
    paired = {f"{a}-{b}": t1_eval.paired(rows, a, b, BOOTSTRAP_SEEDS[f"{a}-{b}"], PAIRED_METRICS) if rows else None
              for a, b in COMPARISONS}
    rules = {k: {m: verdict(e, k.split("-")[0]) for m, e in v.items()} if v else None for k, v in paired.items()}
    det_rows = [{k: (ev.deterministic_view(v) if k in POLICY_ORDER else v) for k, v in r.items()} for r in rows]
    full = complete and len(rows) == len(items)
    return {
        "eval_version": EVAL_VERSION, "dev": meta, "complete": full,
        "scenarios_evaluated": len(rows), "families_evaluated": len({r["family_id"] for r in rows}),
        "policies": {n: p.identity.to_dict() for n, p in policies().items()},
        "primary": {"comparison": f"{PRIMARY[0]}-{PRIMARY[1]}", "metric": PRIMARY[2],
                    "verdict": rules["A2-T1"]["casualties"] if rules.get("A2-T1") and full else "not_evaluable"},
        "t1_reproduction": {"baseline": str(T1_BASELINE.relative_to(ev.REPO)).replace("\\", "/"),
                            "mismatched_positions": t1_mismatch, "pass": full and not t1_mismatch},
        "wall_limit_s": limit_s, "elapsed_wall_s": time.perf_counter() - start,
        "pilot_gate": pilot["gate"],
        "matched_seeds": "all policies run each scenario with its own scenario seed",
        "bootstrap": {**t1_eval.BOOTSTRAP, "seeds": BOOTSTRAP_SEEDS},
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "processor": platform.processor(), **ev._revision()},
        "aggregates": t1_eval.aggregate(rows, latencies, POLICY_ORDER) if rows else None,
        "strata_totals": strata_totals(rows) if rows else None,
        "paired": paired, "decision_rule": rules,
        "a2_worse_than_t1": worse(rows, "A2", "T1"), "a2_worse_than_a0": worse(rows, "A2", "A0"),
        "deterministic_digest": digest(det_rows),
        "dev_status": "development comparison on a previously used DEV split; not a confirmation of generalization",
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pilot-report", type=Path, default=DEFAULT_PILOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    if args.out.exists():
        sys.exit(f"{args.out} exists: the v2 DEV comparison runs once (protocol section 7)")
    try:
        report = evaluate(args.pilot_report)
    except GateError as exc:
        sys.exit(f"T4 v2 DEV evaluation refused: {exc}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "complete": report["complete"], "primary": report["primary"],
                      "t1_reproduction": report["t1_reproduction"]["pass"],
                      "deterministic_digest": report["deterministic_digest"]}, indent=1))


if __name__ == "__main__":
    main()
