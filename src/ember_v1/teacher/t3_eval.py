"""Closed-loop paired comparison of T3 with frozen T0 and T1 on independent DEV (issue #23).

    python -m ember_v1.teacher.t3_eval --pilot-report PATH [--limit N] [--out PATH]

Preregistered in docs/P1_T3_ACO.md section 8. It refuses to run unless the CPU
feasibility pilot report (`t3_pilot`) has gate decision FULL_DEV_FEASIBLE for
the same T3 identity and teacher seed. Same DEV gate as P1-03/P1-04/P1-05
(`t1_eval.load_dev`): only the `dev` split of plan `p1-02-v1`; train,
calibration, the locked test, blind sets and EMBERBench are never read. T0, T1
and T3 run every scenario with the scenario's own simulator seed (the fire
trajectory is identical for all three). A policy receives only
`Simulator.observe()`. T2 is not run and its P1-05 results are not read.
Family-cluster bootstrap and family-level sign tests as in P1-04, with seeds
`t3-vs-t1-family-cluster-bootstrap@1` and `t3-vs-t0-family-cluster-bootstrap@1`.
One worker. The wall limit is min(6 h, max(900 s, 1.5 x the pilot's upper
estimate)); it is checked between episodes (amendment A1, docs/P1_T3_ACO.md
section 8.1). A run that passes it stops and is marked incomplete; an
incomplete run supports no comparison claim.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
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
from .t0 import T0ShortestPath
from .t1 import T1CongestionAware
from .t3 import T3DynamicACO

EVAL_VERSION = "t3-vs-t0-t1-dev@1.0.0"
DEFAULT_OUT = ev.REPO / "data/baselines/p1-06-t3-dev-v1.json"
DEFAULT_PILOT = ev.REPO / "data/baselines/p1-06-t3-pilot.json"
POLICY_ORDER = ("T0", "T1", "T3")
BOOTSTRAP_SEEDS = {"T1": "t3-vs-t1-family-cluster-bootstrap@1", "T0": "t3-vs-t0-family-cluster-bootstrap@1"}
WALL_LIMIT_FACTOR = 1.5            # x the pilot's upper full-DEV comparison estimate
WALL_LIMIT_MIN_S = 900.0           # amendment A1: the estimate is policy CPU only, the guard is elapsed wall
WALL_LIMIT_MAX_S = 6 * 3600.0
PAIRED_METRICS = (("casualties", -1), ("evacuated_fraction", +1), ("unresolved", -1), ("trapped_at_end", -1),
                  ("steps", -1), ("completion_step", -1), ("blocked_per_100_attempts", -1), ("collisions", -1),
                  ("non_progress_rate", -1), ("reversals_per_100_moves", -1), ("exit_utilization", +1),
                  ("policy_cpu_ms", -1))


class GateError(RuntimeError):
    """The CPU feasibility gate does not permit the full DEV run."""


def check_gate(pilot: dict[str, Any], t3: T3DynamicACO) -> float:
    """Return the wall limit (s) if the pilot permits the DEV run for this exact T3, else raise GateError."""
    if pilot.get("gate", {}).get("decision") != "FULL_DEV_FEASIBLE":
        raise GateError(f"pilot gate decision is {pilot.get('gate', {}).get('decision')!r}, not FULL_DEV_FEASIBLE")
    if pilot.get("teacher", {}).get("config_hash") != t3.identity.config_hash:
        raise GateError("pilot was run for a different T3 configuration")
    if pilot.get("teacher_seed") != t3.seed:
        raise GateError("pilot was run for a different teacher seed")
    return wall_limit(pilot["estimate"]["upper"]["comparison_cpu_s_total"])


def wall_limit(upper_estimate_s: float) -> float:
    """Elapsed-wall guard: min(6 h, max(900 s, 1.5 x upper estimate)) (amendment A1)."""
    return min(WALL_LIMIT_MAX_S, max(WALL_LIMIT_MIN_S, WALL_LIMIT_FACTOR * upper_estimate_s))


class _Recorder:
    """Wraps T3 to collect ACO diagnostics; the policy still sees only the observation."""

    def __init__(self, pol: T3DynamicACO) -> None:
        self.pol = pol
        self.identity = pol.identity
        self.reset()

    def reset(self) -> None:
        self.status: Counter = Counter()
        self.limited: Counter = Counter()
        self.reasons: Counter = Counter()
        self.decisions = self.ants = self.ants_ok = self.ant_steps = self.iterations = 0
        self.converged = self.routable = self.unsafe = self.repair_rounds = self.converted = 0
        self.alternatives = 0
        self.last_exit: dict[str, Any] = {}
        self.exit_switches = self.route_lost = 0
        self.aco_len = self.aco_d0 = self.detours = 0

    def decide(self, obs):
        d = self.pol.decide(obs)
        dg = d.diagnostics
        col = dg["colony"]
        self.decisions += 1
        self.status[dg["status"]] += 1
        if dg["budget"]["limit_hit"]:
            self.limited[dg["budget"]["limit_hit"]] += 1
        self.reasons.update(p["reason"] for p in dg["people"].values())
        self.ants += col["ants"]
        self.ants_ok += col["ants_reached_exit"]
        self.ant_steps += col["ant_steps"]
        self.iterations += col["iterations_run"]
        self.converged += col["converged_people"]
        self.routable += sum(p["static_distance"] is not None and p["static_distance"] > 0
                             for p in dg["people"].values())
        self.unsafe += len(dg["unsafe_route_people"])
        self.repair_rounds += dg["coordination"]["repair_rounds"]
        self.converted += len(dg["coordination"]["converted_to_stay"])
        self.alternatives += dg["coordination"]["alternative_moves"]
        for pid, p in dg["people"].items():                # cross-step route changes (trace only)
            prev, cur = self.last_exit.get(pid), p["target_exit"] if p["route_found"] else None
            self.exit_switches += prev is not None and cur is not None and cur != prev
            self.route_lost += prev is not None and cur is None
            self.last_exit[pid] = cur
            if p["reason"] == "aco_route" and p["static_distance"]:
                self.aco_len += p["route_length"]
                self.aco_d0 += p["static_distance"]
                self.detours += p["route_length"] > p["static_distance"]
        return d

    def summary(self) -> dict[str, Any]:
        person_decisions = sum(self.reasons.values())
        return {"decisions": self.decisions, "status": dict(sorted(self.status.items())),
                "limit_hit": dict(sorted(self.limited.items())), "person_reasons": dict(sorted(self.reasons.items())),
                "fallback_person_decisions": self.reasons["fallback_bfs_descent"],
                "fallback_rate": self.reasons["fallback_bfs_descent"] / person_decisions if person_decisions else None,
                "ants": self.ants, "ants_reached_exit": self.ants_ok,
                "ant_success_rate": self.ants_ok / self.ants if self.ants else None,
                "ant_steps": self.ant_steps, "iterations_run": self.iterations,
                "converged_person_decisions": self.converged, "routable_person_decisions": self.routable,
                "converged_fraction": self.converged / self.routable if self.routable else None,
                "unsafe_route_person_decisions": self.unsafe, "repair_rounds": self.repair_rounds,
                "converted_to_stay": self.converted, "alternative_moves": self.alternatives,
                "target_exit_switches": self.exit_switches, "route_lost_person_decisions": self.route_lost,
                "aco_route_length_sum": self.aco_len, "aco_static_distance_sum": self.aco_d0,
                "aco_detour_person_decisions": self.detours}


def verdict(entry: dict[str, Any], new: str = "T3") -> str:
    """Preregistered rule (section 8): `better`/`worse` only if the family-cluster CI excludes 0 and the
    family sign test has p < 0.05 in the same direction; else `no_evidence_of_difference`."""
    a = new.lower()
    ci, p = entry["family_cluster_bootstrap_ci95"], entry["family_sign_test_p_two_sided"]
    if ci is None or p >= 0.05 or ci[0] <= 0 <= ci[1]:
        return "no_evidence_of_difference"
    sign = 1 if entry["direction"] == "higher_is_better" else -1
    ci_better = sign * ci[0] > 0
    fam_better = entry[f"family_{a}_better"] > entry[f"family_{a}_worse"]
    if ci_better != fam_better:
        return "no_evidence_of_difference"
    return "better" if ci_better else "worse"


def verdicts(paired_stats: dict[str, Any] | None) -> dict[str, str] | None:
    return {m: verdict(e) for m, e in paired_stats.items()} if paired_stats else None


def worse(rows: list[dict[str, Any]], new: str, ref: str) -> list[dict[str, Any]]:
    """Every scenario where `new` is worse than `ref` on casualties, evacuated fraction or unresolved."""
    out = []
    keys = ("evacuated", "casualties", "unresolved", "steps")
    for r in rows:
        a, b = r[new], r[ref]
        if a["casualties"] > b["casualties"] or a["evacuated_fraction"] < b["evacuated_fraction"] \
                or a["unresolved"] > b["unresolved"]:
            out.append({"position": r["position"], "family_id": r["family_id"], "label": r["label"],
                        "fire_model": r["fire_model"], "enter_fire": r["enter_fire"], "people": a["people"],
                        ref: {k: b[k] for k in keys}, new: {k: a[k] for k in keys},
                        f"{ref}_casualty_context": b["casualty_context"],
                        f"{new}_casualty_context": a["casualty_context"]})
    return out


def strata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("label", "fire_model", "enter_fire"):
        out[key] = {}
        for val in sorted({r[key] for r in rows}):
            sub = [r for r in rows if r[key] == val]
            out[key][val] = {"scenarios": len(sub), **{
                name: {"casualties": sum(r[name]["casualties"] for r in sub),
                       "evacuated_fraction_mean": statistics.fmean(r[name]["evacuated_fraction"] for r in sub),
                       "steps_mean": statistics.fmean(r[name]["steps"] for r in sub)} for name in POLICY_ORDER}}
    return out


def evaluate(pilot_report: str | Path = DEFAULT_PILOT, limit: int | None = None) -> dict[str, Any]:
    t3 = T3DynamicACO()
    pilot = json.loads(Path(pilot_report).read_text(encoding="utf-8"))
    limit_s = check_gate(pilot, t3)
    meta, items = t1_eval.load_dev()
    if limit is not None:
        items = items[:limit]
    rec = _Recorder(t3)
    rows: list[dict[str, Any]] = []
    latencies: dict[str, list[float]] = {n: [] for n in POLICY_ORDER}
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
        r0, lat0 = t1_eval.run_episode(s, T0ShortestPath())
        r1, lat1 = t1_eval.run_episode(s, T1CongestionAware())
        rec.reset()
        r3, lat3 = t1_eval.run_episode(s, rec)
        r3["t3"] = rec.summary()
        row["T0"], row["T1"], row["T3"] = r0, r1, r3
        for name, lat in (("T0", lat0), ("T1", lat1), ("T3", lat3)):
            latencies[name].extend(lat)
        rows.append(row)
        print(json.dumps({"position": it["position"], "T0_cas": r0["casualties"], "T1_cas": r1["casualties"],
                          "T3_cas": r3["casualties"], "T3_cpu_s": round(r3["timing"]["policy_cpu_ms"] / 1000, 2)}),
              flush=True)
    p31 = t1_eval.paired(rows, "T3", "T1", BOOTSTRAP_SEEDS["T1"], PAIRED_METRICS) if rows else None
    p30 = t1_eval.paired(rows, "T3", "T0", BOOTSTRAP_SEEDS["T0"], PAIRED_METRICS) if rows else None
    det_rows = [{k: (ev.deterministic_view(v) if k in POLICY_ORDER else v) for k, v in r.items()} for r in rows]
    return {
        "eval_version": EVAL_VERSION, "dev": meta, "complete": complete and len(rows) == len(items),
        "scenarios_evaluated": len(rows), "families_evaluated": len({r["family_id"] for r in rows}),
        "policies": {"T0": T0ShortestPath().identity.to_dict(), "T1": T1CongestionAware().identity.to_dict(),
                     "T3": t3.identity.to_dict()},
        "teacher_seed": t3.seed, "wall_limit_s": limit_s, "elapsed_wall_s": time.perf_counter() - start,
        "wall_limit_rule": "min(6 h, max(900 s, 1.5 x pilot upper comparison estimate)), checked between episodes",
        "pilot_gate": pilot["gate"],
        "matched_seeds": "all policies run each scenario with its own scenario seed",
        "bootstrap": {**t1_eval.BOOTSTRAP, "seeds": BOOTSTRAP_SEEDS},
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "processor": platform.processor(), **ev._revision()},
        "aggregates": t1_eval.aggregate(rows, latencies, POLICY_ORDER) if rows else None,
        "paired_t3_minus_t1": p31,
        "paired_t3_minus_t0": p30,
        "decision_rule": {"t3_minus_t1": verdicts(p31), "t3_minus_t0": verdicts(p30)},
        "strata": strata(rows) if rows else None,
        "t3_worse_than_t1": worse(rows, "T3", "T1"),
        "t3_worse_than_t0": worse(rows, "T3", "T0"),
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
        sys.exit(f"T3 DEV evaluation refused: {exc}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "complete": report["complete"],
                      "deterministic_digest": report["deterministic_digest"]}, indent=1))


if __name__ == "__main__":
    main()
