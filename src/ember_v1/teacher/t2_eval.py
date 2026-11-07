"""Closed-loop paired comparison of T2 with frozen T1 on independent DEV (issue #9).

    python -m ember_v1.teacher.t2_eval --pilot-report PATH [--limit N] [--out PATH]

Preregistered in docs/P1_T2_ROLLOUT.md section 7. It refuses to run unless the
CPU feasibility pilot report (`t2_pilot`) has gate decision FULL_DEV_FEASIBLE
for the same T2 identity, teacher seed and continuation. Same DEV gate as
P1-03/P1-04 (`t1_eval.load_dev`): only the `dev` split of plan `p1-02-v1`;
train, calibration, the locked test, blind sets and EMBERBench are never read.
T1 and T2 run every scenario with the scenario's own simulator seed (the
fire trajectory is identical for both). A policy receives only
`Simulator.observe()`. Family-cluster bootstrap and family-level sign tests
as in P1-04, with the seed `t2-vs-t1-family-cluster-bootstrap@1`. One worker.
A run that passes the preregistered wall limit stops and is marked incomplete;
an incomplete run supports no comparison claim.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..sim.canonical import digest
from ..sim.scenario import Scenario
from . import evaluate as ev
from . import t1_eval
from .t1 import T1CongestionAware
from .t2 import T2StochasticRollout

EVAL_VERSION = "t2-vs-t1-dev@1.0.0"
DEFAULT_OUT = ev.REPO / "data/baselines/p1-05-t2-dev-v1.json"
DEFAULT_PILOT = ev.REPO / "data/baselines/p1-05-t2-pilot.json"
POLICY_ORDER = ("T1", "T2")
BOOTSTRAP_SEED = "t2-vs-t1-family-cluster-bootstrap@1"
WALL_LIMIT_FACTOR = 1.5            # x the pilot's upper full-DEV estimate
WALL_LIMIT_MAX_S = 6 * 3600.0
PAIRED_METRICS = (("casualties", -1), ("evacuated_fraction", +1), ("unresolved", -1), ("trapped_at_end", -1),
                  ("steps", -1), ("completion_step", -1), ("blocked_per_100_attempts", -1), ("collisions", -1),
                  ("non_progress_rate", -1), ("reversals_per_100_moves", -1), ("exit_utilization", +1),
                  ("policy_cpu_ms", -1))


class GateError(RuntimeError):
    """The CPU feasibility gate does not permit the full DEV run."""


def check_gate(pilot: dict[str, Any], t2: T2StochasticRollout) -> float:
    """Return the wall limit (s) if the pilot permits the DEV run for this exact T2, else raise GateError."""
    if pilot.get("gate", {}).get("decision") != "FULL_DEV_FEASIBLE":
        raise GateError(f"pilot gate decision is {pilot.get('gate', {}).get('decision')!r}, not FULL_DEV_FEASIBLE")
    if pilot.get("teacher", {}).get("config_hash") != t2.identity.config_hash:
        raise GateError("pilot was run for a different T2 configuration")
    if pilot.get("teacher_seed") != t2.seed or pilot.get("continuation") != t2.continuation:
        raise GateError("pilot was run for a different teacher seed or continuation")
    upper = pilot["estimate"]["upper"]["t2_cpu_s_total"]
    return min(WALL_LIMIT_MAX_S, WALL_LIMIT_FACTOR * upper)


class _Recorder:
    """Wraps a policy to collect T2 diagnostics; the policy still sees only the observation."""

    def __init__(self, pol: T2StochasticRollout) -> None:
        self.pol = pol
        self.identity = pol.identity
        self.reset()

    def reset(self) -> None:
        self.status: Counter = Counter()
        self.deviations = self.scored = self.rollouts = self.decisions = 0
        self.limited: Counter = Counter()
        self.joint: Counter = Counter()

    def decide(self, obs):
        d = self.pol.decide(obs)
        dg = d.diagnostics
        self.decisions += 1
        self.status[dg["status"]] += 1
        self.deviations += len(dg["deviations"])
        self.scored += len(dg["scored"])
        self.rollouts += dg["budget"]["rollouts"]
        if dg["budget"]["limit_hit"]:
            self.limited[dg["budget"]["limit_hit"]] += 1
        v = dg["joint"]["validation"]
        self.joint[v["result"] if v else ("single_or_none")] += 1
        return d

    def summary(self) -> dict[str, Any]:
        return {"decisions": self.decisions, "status": dict(sorted(self.status.items())),
                "deviating_person_decisions": self.deviations, "scored_person_decisions": self.scored,
                "rollouts": self.rollouts, "limit_hit": dict(sorted(self.limited.items())),
                "joint_validation": dict(sorted(self.joint.items()))}


def evaluate(pilot_report: str | Path = DEFAULT_PILOT, limit: int | None = None) -> dict[str, Any]:
    t2 = T2StochasticRollout()
    pilot = json.loads(Path(pilot_report).read_text(encoding="utf-8"))
    wall_limit = check_gate(pilot, t2)
    meta, items = t1_eval.load_dev()
    if limit is not None:
        items = items[:limit]
    rec = _Recorder(t2)
    rows: list[dict[str, Any]] = []
    latencies: dict[str, list[float]] = {n: [] for n in POLICY_ORDER}
    start = time.perf_counter()
    complete = True
    for it in items:
        if time.perf_counter() - start > wall_limit:
            complete = False
            break
        s: Scenario = it["scenario"]
        row: dict[str, Any] = {"position": it["position"], "lineage_id": it["lineage_id"],
                               "family_id": it["family_id"], "scenario_hash": it["scenario_hash"],
                               "simulator_seed": s.seed, "label": it["label"], "grid": [s.height, s.width],
                               "fire_model": s.fire_model, "enter_fire": s.enter_fire, "max_steps": s.max_steps}
        r1, lat1 = t1_eval.run_episode(s, T1CongestionAware())
        rec.reset()
        r2, lat2 = t1_eval.run_episode(s, rec)
        r2["t2"] = rec.summary()
        row["T1"], row["T2"] = r1, r2
        latencies["T1"].extend(lat1)
        latencies["T2"].extend(lat2)
        rows.append(row)
        print(json.dumps({"position": it["position"], "T1_cas": r1["casualties"], "T2_cas": r2["casualties"],
                          "T2_cpu_s": round(r2["timing"]["policy_cpu_ms"] / 1000, 1)}), flush=True)
    det_rows = [{k: (ev.deterministic_view(v) if k in POLICY_ORDER else v) for k, v in r.items()} for r in rows]
    return {
        "eval_version": EVAL_VERSION, "dev": meta, "complete": complete and len(rows) == len(items),
        "scenarios_evaluated": len(rows), "families_evaluated": len({r["family_id"] for r in rows}),
        "policies": {"T1": T1CongestionAware().identity.to_dict(), "T2": t2.identity.to_dict()},
        "teacher_seed": t2.seed, "continuation": t2.continuation, "wall_limit_s": wall_limit,
        "pilot_gate": pilot["gate"],
        "bootstrap": {**t1_eval.BOOTSTRAP, "seed": BOOTSTRAP_SEED},
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "processor": platform.processor(), **ev._revision()},
        "aggregates": t1_eval.aggregate(rows, latencies, POLICY_ORDER) if rows else None,
        "paired_t2_minus_t1": t1_eval.paired(rows, "T2", "T1", BOOTSTRAP_SEED, PAIRED_METRICS) if rows else None,
        "deterministic_digest": digest(det_rows),
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pilot-report", type=Path, default=DEFAULT_PILOT)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    try:
        report = evaluate(args.pilot_report, args.limit)
    except GateError as exc:
        sys.exit(f"T2 DEV evaluation refused: {exc}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "complete": report["complete"],
                      "deterministic_digest": report["deterministic_digest"]}, indent=1))


if __name__ == "__main__":
    main()
