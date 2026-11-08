"""Closed-loop diagnostic comparison of T0 with simple controls on independent DEV.

    python -m ember_v1.teacher.evaluate [--limit N] [--out PATH]

Only the generated `dev` split of plan `p1-02-v1` is accepted. Train,
calibration, the locked test, blind sets and EMBERBench are refused or never
read. Every policy runs on the same scenarios with the same simulator seed
(the scenario seed), so the fire trajectory is identical for all policies
(fire never depends on actions). The policy receives only `Simulator.observe()`.
Evaluator-private values (trapped set, final statuses) are read after each step
for metrics only and never reach a policy.

This is a diagnostic baseline, not a teacher-hyperparameter search: T0 has no
tunable parameters, and nothing here selects a configuration.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..gen.splits import load_index, load_manifest, materialize, plan_from_index
from ..sim.canonical import digest
from ..sim.movement import OUTCOMES
from ..sim.scenario import Scenario
from ..sim.simulator import InvalidActionError, Simulator
from .controls import GreedyNearestExitControl, RandomControl, StayControl
from .interface import ReferencePolicy
from .t0 import T0ShortestPath

EVAL_VERSION = "t0-dev-baseline@1.0.0"
REPO = Path(__file__).resolve().parents[3]
DEFAULT_INDEX = REPO / "data/splits/p1-02-v1/index.json"
DEFAULT_DEV = REPO / "data/splits/p1-02-v1/dev.json"
DEFAULT_OUT = REPO / "data/baselines/p1-03-t0-dev-v1.json"
ALLOWED_SPLIT = "dev"
POLICY_ORDER = ("T0", "C-stay", "C-random", "C-greedy")
COLLISION_OUTCOMES = ("blocked_occupied", "blocked_conflict", "blocked_cycle")


def load_dev(index_path: str | Path = DEFAULT_INDEX, dev_path: str | Path = DEFAULT_DEV) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Verified DEV scenarios (hash-checked and regenerated). Any other split is refused."""
    idx = load_index(index_path)
    raw = json.loads(Path(dev_path).read_text(encoding="utf-8"))
    if raw.get("split") != ALLOWED_SPLIT:
        raise PermissionError(f"T0 baseline evaluation accepts only the {ALLOWED_SPLIT!r} split, not {raw.get('split')!r}")
    m = load_manifest(dev_path, index_data=idx)
    plan = plan_from_index(idx)
    items = [{"position": i, "lineage_id": e["lineage_id"], "scenario_hash": e["scenario_hash"], "label": g.label,
              "scenario": g.scenario} for i, (e, g) in enumerate(materialize(m, plan.spec))]
    meta = {"plan_id": m["plan_id"], "split": m["split"], "manifest_hash": m["manifest_hash"],
            "index_hash": idx["index_hash"], "scenarios": len(items)}
    return meta, items


def random_control_seed(plan_id: str, position: int) -> int:
    """Seed of C-random for one scenario: from the plan ID and manifest position only."""
    return int(digest({"control": "C-random", "plan_id": plan_id, "position": position})[7:23], 16)


def policies_for(plan_id: str, position: int) -> list[ReferencePolicy]:
    return [T0ShortestPath(), StayControl(), RandomControl(random_control_seed(plan_id, position)),
            GreedyNearestExitControl()]


def run_episode(scenario: Scenario, policy: ReferencePolicy) -> tuple[dict[str, Any], list[float]]:
    """One closed-loop episode. Returns (record, per-decision wall ms)."""
    sim = Simulator(scenario)                     # simulator seed = scenario seed (matched across policies)
    trapped_start = len(sim.private_state()["trapped"])
    outcomes = Counter({o: 0 for o in OUTCOMES})
    casualty_reasons: Counter = Counter()
    wall_ms: list[float] = []
    policy_cpu_ms = 0.0
    failure = None
    no_route_person_steps = 0
    mismatches = 0
    casualty_context: Counter = Counter()
    cpu0, wall0 = time.process_time(), time.perf_counter()
    while not sim.terminated:
        obs = sim.observe()
        try:
            decision = policy.decide(obs)
        except Exception as exc:  # recorded as a policy failure, never repaired
            failure = f"policy_error:{type(exc).__name__}"
            break
        wall_ms.append(decision.runtime["wall_ms"])
        policy_cpu_ms += decision.runtime["cpu_ms"]
        try:
            rec = sim.step(decision.actions)
        except InvalidActionError:
            failure = "invalid_actions"
            break
        for pid, mv in rec["movements"].items():
            outcomes[mv["outcome"]] += 1
        casualty_reasons.update(rec["casualties"].values())
        people = decision.diagnostics.get("people")
        if people is not None:                    # T0: predicted joint outcome vs simulator outcome
            no_route_person_steps += len(decision.diagnostics["no_route"])
            mismatches += sum(people[pid]["predicted_outcome"] != mv["outcome"] for pid, mv in rec["movements"].items())
            for pid in rec["casualties"]:         # T0's last decision for each casualty
                route = "route" if people[pid]["route_found"] else "no_route"
                casualty_context[f"{route}/{rec['movements'][pid]['outcome']}"] += 1
    episode_cpu_s, episode_wall_s = time.process_time() - cpu0, time.perf_counter() - wall0
    final = sim.private_state()
    status = Counter(p["status"] for p in final["people"])
    n = len(final["people"])
    moved = outcomes["moved"]
    blocked = sum(v for k, v in outcomes.items() if k.startswith("blocked_"))
    record = {
        "policy": policy.identity.name,
        "policy_seed": policy.seed,
        "steps": final["step"],
        "termination_reason": sim.termination_reason,
        "completion_step": final["step"] if sim.termination_reason == "all_resolved" else None,
        "people": n,
        "evacuated": status["evacuated"],
        "casualties": status["casualty"],
        "casualty_reasons": dict(sorted(casualty_reasons.items())),
        "unresolved": status["active"],
        "trapped_at_start": trapped_start,
        "trapped_at_end": len(final["trapped"]),
        "evacuated_fraction": status["evacuated"] / n,
        "casualty_fraction": status["casualty"] / n,
        "movement_outcomes": dict(sorted(outcomes.items())),
        "moves_attempted": moved + blocked,
        "blocked": blocked,
        "collisions": sum(outcomes[o] for o in COLLISION_OUTCOMES),
        "decisions": len(wall_ms),
        "failure": failure,
        "final_transition_hash": sim.chain_hash,
    }
    if policy.identity.name == "T0":
        record["t0"] = {"no_route_person_steps": no_route_person_steps, "prediction_mismatches": mismatches,
                        "casualty_context": dict(sorted(casualty_context.items()))}
    record["timing"] = {
        "decision_wall_ms_mean": statistics.fmean(wall_ms) if wall_ms else None,
        "decision_wall_ms_max": max(wall_ms) if wall_ms else None,
        "policy_cpu_ms_total": policy_cpu_ms,
        "episode_cpu_s": episode_cpu_s,
        "episode_wall_s": episode_wall_s,
    }
    return record, wall_ms


def _pct(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))]   # nearest-rank


def _sign_test(better: int, worse: int) -> float:
    """Exact two-sided sign test on non-tied pairs."""
    n = better + worse
    if n == 0:
        return 1.0
    k = min(better, worse)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def deterministic_view(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k != "timing"}


def aggregate(rows: list[dict[str, Any]], latencies: dict[str, list[float]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in POLICY_ORDER:
        rs = [r[name] for r in rows]
        completed = [x["completion_step"] for x in rs if x["completion_step"] is not None]
        moves = sum(x["moves_attempted"] for x in rs)
        mo = Counter()
        for x in rs:
            mo.update(x["movement_outcomes"])
        lat = latencies[name]
        agg = {
            "scenarios": len(rs), "failures": sum(x["failure"] is not None for x in rs),
            "invalid_action_episodes": sum(x["failure"] == "invalid_actions" for x in rs),
            "people": sum(x["people"] for x in rs), "evacuated": sum(x["evacuated"] for x in rs),
            "casualties": sum(x["casualties"] for x in rs), "unresolved": sum(x["unresolved"] for x in rs),
            "trapped_at_end": sum(x["trapped_at_end"] for x in rs),
            "evacuated_fraction_mean": statistics.fmean(x["evacuated_fraction"] for x in rs),
            "casualty_fraction_mean": statistics.fmean(x["casualty_fraction"] for x in rs),
            "all_resolved_episodes": len(completed),
            "completion_steps_mean": statistics.fmean(completed) if completed else None,
            "steps_mean": statistics.fmean(x["steps"] for x in rs),
            "movement_outcomes": dict(sorted(mo.items())),
            "blocked": sum(x["blocked"] for x in rs), "collisions": sum(x["collisions"] for x in rs),
            "blocked_per_100_attempted_moves": 100 * sum(x["blocked"] for x in rs) / moves if moves else None,
            "decisions": len(lat),
            "decision_wall_ms": {"mean": statistics.fmean(lat) if lat else None, "p50": _pct(lat, 0.5),
                                 "p95": _pct(lat, 0.95), "max": max(lat) if lat else None},
            "episode_cpu_s": {"mean": statistics.fmean(x["timing"]["episode_cpu_s"] for x in rs),
                              "max": max(x["timing"]["episode_cpu_s"] for x in rs)},
            "policy_cpu_s_total": sum(x["timing"]["policy_cpu_ms_total"] for x in rs) / 1000,
        }
        if name == "T0":
            agg["no_route_person_steps"] = sum(x["t0"]["no_route_person_steps"] for x in rs)
            agg["prediction_mismatches"] = sum(x["t0"]["prediction_mismatches"] for x in rs)
            ctx: Counter = Counter()
            for x in rs:
                ctx.update(x["t0"]["casualty_context"])
            agg["casualty_context"] = dict(sorted(ctx.items()))
            agg["trapped_at_start"] = sum(x["trapped_at_start"] for x in rs)
        out[name] = agg
    return out


def paired(rows: list[dict[str, Any]], a: str = "T0") -> dict[str, Any]:
    """Per-scenario paired comparison of `a` with each control (same scenario, same seed)."""
    metrics = (("evacuated_fraction", +1), ("casualties", -1), ("unresolved", -1))
    out: dict[str, Any] = {}
    for b in POLICY_ORDER:
        if b == a:
            continue
        res = {}
        for metric, sign in metrics:
            diffs = [r[a][metric] - r[b][metric] for r in rows]
            better = sum(sign * d > 0 for d in diffs)
            worse = sum(sign * d < 0 for d in diffs)
            res[metric] = {"mean_diff": statistics.fmean(diffs), "t0_better": better,
                           "equal": len(diffs) - better - worse, "t0_worse": worse,
                           "sign_test_p_two_sided": _sign_test(better, worse)}
        out[f"{a}_vs_{b}"] = res
    return out


def by_label(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = sorted({r["label"] for r in rows})
    return {lab: {name: {"scenarios": sum(r["label"] == lab for r in rows),
                         "evacuated_fraction_mean": statistics.fmean(r[name]["evacuated_fraction"] for r in rows if r["label"] == lab)}
                  for name in POLICY_ORDER} for lab in labels}


def _revision() -> dict[str, Any]:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src"], cwd=REPO, capture_output=True,
                                    text=True, check=True).stdout.strip())
        return {"git_sha": sha, "src_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_sha": None, "src_dirty": None}


def evaluate(limit: int | None = None, index_path: str | Path = DEFAULT_INDEX,
             dev_path: str | Path = DEFAULT_DEV) -> dict[str, Any]:
    meta, items = load_dev(index_path, dev_path)
    if limit is not None:
        items = items[:limit]
    rows: list[dict[str, Any]] = []
    latencies: dict[str, list[float]] = {n: [] for n in POLICY_ORDER}
    identities: dict[str, Any] = {}
    for it in items:
        s: Scenario = it["scenario"]
        row: dict[str, Any] = {"position": it["position"], "lineage_id": it["lineage_id"],
                               "scenario_hash": it["scenario_hash"], "simulator_seed": s.seed, "label": it["label"],
                               "grid": [s.height, s.width], "fire_model": s.fire_model, "enter_fire": s.enter_fire,
                               "max_steps": s.max_steps}
        for pol in policies_for(meta["plan_id"], it["position"]):
            rec, lat = run_episode(s, pol)
            row[pol.identity.name] = rec
            latencies[pol.identity.name].extend(lat)
            ident = pol.identity.to_dict()
            identities.setdefault(pol.identity.name, ident)
        rows.append(row)
    det_rows = [{k: (deterministic_view(v) if k in POLICY_ORDER else v) for k, v in r.items()} for r in rows]
    return {
        "eval_version": EVAL_VERSION, "dev": meta, "scenarios_evaluated": len(rows),
        "policies": identities,
        "random_control_seed_rule": "int(sha256 canonical {control, plan_id, position}[0:16 hex])",
        "matched_seeds": "every policy runs each scenario with its own scenario seed",
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "processor": platform.processor(), **_revision()},
        "aggregates": aggregate(rows, latencies),
        "paired": paired(rows),
        "by_label": by_label(rows),
        "deterministic_digest": digest(det_rows),
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    report = evaluate(limit=args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "deterministic_digest": report["deterministic_digest"],
                      "aggregates": {k: {m: v[m] for m in ("evacuated", "casualties", "unresolved", "collisions")}
                                     for k, v in report["aggregates"].items()}}, indent=1))


if __name__ == "__main__":
    main()
