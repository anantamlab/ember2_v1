"""Closed-loop paired comparison of T1 with frozen T0 on independent DEV (issue #8).

    python -m ember_v1.teacher.t1_eval [--limit N] [--out PATH]

Same gate as the T0 harness (`evaluate.load_dev`): only the generated `dev`
split of plan `p1-02-v1`; train, calibration, the locked test, blind sets and
EMBERBench are refused or never read. T0 and T1 run every scenario with the
scenario's own simulator seed, so the fire trajectory is identical for both
(fire never depends on actions). A policy receives only `Simulator.observe()`.
Evaluator-private values (trapped set, final statuses, family IDs, labels) are
read for metrics only and never reach a policy.

Uncertainty: scenarios share topology families (3 lineages per family), so
confidence intervals come from a family-cluster bootstrap (families resampled
with replacement, fixed seed) and a family-level sign test is reported next to
the scenario-level one. Nothing here selects or tunes a configuration: the T1
configuration is frozen in `t1.py` and docs/P1_T1_PLANNER.md before this runs.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..sim.canonical import digest
from ..sim.movement import OUTCOMES
from ..sim.scenario import Scenario
from ..sim.simulator import InvalidActionError, Simulator
from . import evaluate as ev
from .interface import ReferencePolicy
from .t0 import T0ShortestPath
from .t1 import T1CongestionAware

EVAL_VERSION = "t1-vs-t0-dev@1.0.0"
DEFAULT_OUT = ev.REPO / "data/baselines/p1-04-t1-dev-v1.json"
POLICY_ORDER = ("T0", "T1")
COLLISIONS = ("blocked_occupied", "blocked_conflict", "blocked_cycle")
BOOTSTRAP = {"resamples": 10000, "seed": "t1-vs-t0-family-cluster-bootstrap@1", "level": 0.95}
BUDGET_STATUSES = ("budget_partial", "budget_fallback")
NO_ADMISSIBLE_STATUSES = ("no_successor_greedy", "no_successor_least_hazard")
# metric -> +1 if larger is better, -1 if smaller is better
PAIRED_METRICS = (("casualties", -1), ("evacuated_fraction", +1), ("unresolved", -1), ("steps", -1),
                  ("completion_step", -1), ("blocked_per_100_attempts", -1), ("collisions", -1),
                  ("non_progress_rate", -1), ("reversals_per_100_moves", -1), ("exit_utilization", +1),
                  ("policy_cpu_ms", -1))


def load_dev(index_path: str | Path = ev.DEFAULT_INDEX, dev_path: str | Path = ev.DEFAULT_DEV
             ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """`evaluate.load_dev` (split gate, hash checks, regeneration) plus evaluator-side family IDs."""
    meta, items = ev.load_dev(index_path, dev_path)
    raw = json.loads(Path(dev_path).read_text(encoding="utf-8"))
    family = {e["lineage_id"]: e["family_id"] for e in raw["scenarios"]}
    for it in items:
        it["family_id"] = family[it["lineage_id"]]
    return meta, items


def policies() -> list[ReferencePolicy]:
    return [T0ShortestPath(), T1CongestionAware()]


def _corridor_cells(s: Scenario) -> frozenset[tuple[int, int]]:
    """Evaluator-side static bottleneck cells: exits and open cells with <= 2 open 4-neighbours."""
    walls = frozenset(s.obstacles)
    exits = {p for _, p in s.exits}
    out = set(exits)
    for r in range(s.height):
        for c in range(s.width):
            if (r, c) in walls:
                continue
            n = sum(1 for q in ((r - 1, c), (r + 1, c), (r, c + 1), (r, c - 1))
                    if 0 <= q[0] < s.height and 0 <= q[1] < s.width and q not in walls)
            if n <= 2:
                out.add((r, c))
    return frozenset(out)


def run_episode(scenario: Scenario, policy: ReferencePolicy) -> tuple[dict[str, Any], list[float]]:
    """One closed-loop episode with congestion, exit and planning metrics. Returns (record, wall ms)."""
    sim = Simulator(scenario)
    trapped_start = len(sim.private_state()["trapped"])
    bottleneck = _corridor_cells(scenario)
    exit_pos = {i: p for i, p in scenario.exits}
    outcomes = Counter({o: 0 for o in OUTCOMES})
    casualty_reasons: Counter = Counter()
    casualty_context: Counter = Counter()
    status_counts: Counter = Counter()
    exit_evac: Counter = Counter({i: 0 for i in exit_pos})
    wall_ms: list[float] = []
    policy_cpu_ms = 0.0
    failure = None
    person_steps = open_exit_steps = reversals = bottleneck_blocked = 0
    mismatches = no_route_steps = budget_decisions = repair_rounds = converted = 0
    budget_person_decisions = no_admissible_person_decisions = 0
    last_from: dict[str, list[int]] = {}
    cpu0, wall0 = time.process_time(), time.perf_counter()
    while not sim.terminated:
        obs = sim.observe()
        burning = {tuple(b) for b in obs["burning"]}
        open_exit_steps += sum(p not in burning for p in exit_pos.values())
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
        person_steps += len(rec["movements"])
        for pid, mv in rec["movements"].items():
            outcomes[mv["outcome"]] += 1
            if mv["outcome"] == "moved":
                if last_from.get(pid) == mv["to"]:
                    reversals += 1
                last_from[pid] = mv["from"]
            elif mv["outcome"].startswith("blocked_"):
                dr, dc = {"north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1)}[rec["actions"][pid]]
                if (mv["from"][0] + dr, mv["from"][1] + dc) in bottleneck:
                    bottleneck_blocked += 1
        exit_evac.update(rec["evacuations"].values())
        casualty_reasons.update(rec["casualties"].values())
        dg = decision.diagnostics
        people = dg["people"]
        no_route_steps += len(dg["no_route"])
        mismatches += sum(people[pid]["predicted_outcome"] != mv["outcome"] for pid, mv in rec["movements"].items())
        for pid in rec["casualties"]:
            route = "route" if people[pid]["route_found"] else "no_route"
            casualty_context[f"{route}/{rec['movements'][pid]['outcome']}"] += 1
        if policy.identity.name == "T1":
            status_counts.update(dg["plan_status"])
            budget_decisions += bool(dg["budget_exhausted"] or any(dg["plan_status"].get(s) for s in BUDGET_STATUSES))
            budget_person_decisions += sum(dg["plan_status"].get(s, 0) for s in BUDGET_STATUSES)
            no_admissible_person_decisions += sum(dg["plan_status"].get(s, 0) for s in NO_ADMISSIBLE_STATUSES)
            repair_rounds += dg["repair_rounds"]
            converted += len(dg["converted_to_stay"])
    episode_cpu_s, episode_wall_s = time.process_time() - cpu0, time.perf_counter() - wall0
    final = sim.private_state()
    status = Counter(p["status"] for p in final["people"])
    n = len(final["people"])
    moved = outcomes["moved"]
    blocked = sum(v for k, v in outcomes.items() if k.startswith("blocked_"))
    attempts = moved + blocked
    evacuated = status["evacuated"]
    record: dict[str, Any] = {
        "policy": policy.identity.name,
        "steps": final["step"],
        "termination_reason": sim.termination_reason,
        "completion_step": final["step"] if sim.termination_reason == "all_resolved" else None,
        "people": n, "evacuated": evacuated, "casualties": status["casualty"], "unresolved": status["active"],
        "trapped_at_start": trapped_start, "trapped_at_end": len(final["trapped"]),
        "evacuated_fraction": evacuated / n, "casualty_fraction": status["casualty"] / n,
        "casualty_reasons": dict(sorted(casualty_reasons.items())),
        "casualty_context": dict(sorted(casualty_context.items())),
        "movement_outcomes": dict(sorted(outcomes.items())),
        "moves_attempted": attempts, "blocked": blocked,
        "collisions": sum(outcomes[o] for o in COLLISIONS),
        "blocked_per_100_attempts": 100 * blocked / attempts if attempts else 0.0,
        "person_steps": person_steps,
        "waiting_rate": outcomes["stayed"] / person_steps if person_steps else 0.0,
        "non_progress_rate": (outcomes["stayed"] + blocked) / person_steps if person_steps else 0.0,
        "reversals": reversals,
        "reversals_per_100_moves": 100 * reversals / moved if moved else 0.0,
        "bottleneck_blocked": bottleneck_blocked,
        "exit_evacuations": dict(sorted(exit_evac.items())),
        "open_exit_steps": open_exit_steps,
        "exit_utilization": evacuated / open_exit_steps if open_exit_steps else 0.0,
        "busiest_exit_share": max(exit_evac.values()) / evacuated if evacuated else None,
        "no_route_person_steps": no_route_steps,
        "prediction_mismatches": mismatches,
        "decisions": len(wall_ms),
        "failure": failure,
        "final_transition_hash": sim.chain_hash,
    }
    if policy.identity.name == "T1":
        record["t1"] = {"plan_status": dict(sorted(status_counts.items())),
                        "budget_failure_decisions": budget_decisions,
                        "budget_failure_person_decisions": budget_person_decisions,
                        "no_admissible_action_person_decisions": no_admissible_person_decisions,
                        "repair_rounds": repair_rounds, "converted_to_stay": converted}
    record["timing"] = {"policy_cpu_ms": policy_cpu_ms, "episode_cpu_s": episode_cpu_s,
                        "episode_wall_s": episode_wall_s,
                        "decision_wall_ms_mean": statistics.fmean(wall_ms) if wall_ms else None,
                        "decision_wall_ms_max": max(wall_ms) if wall_ms else None}
    return record, wall_ms


def _metric(rec: dict[str, Any], name: str) -> float | None:
    return rec["timing"]["policy_cpu_ms"] if name == "policy_cpu_ms" else rec[name]


def cluster_bootstrap(diffs_by_family: dict[str, list[float]], resamples: int, seed: str,
                      level: float) -> tuple[float, float] | None:
    """Percentile CI of the mean scenario difference; families are resampled with replacement."""
    fams = sorted(f for f, d in diffs_by_family.items() if d)
    if not fams:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        tot, cnt = 0.0, 0
        for _ in fams:
            d = diffs_by_family[fams[rng.randrange(len(fams))]]
            tot += sum(d)
            cnt += len(d)
        means.append(tot / cnt)
    means.sort()
    lo = means[int((1 - level) / 2 * resamples)]
    hi = means[min(resamples - 1, int((1 + level) / 2 * resamples))]
    return lo, hi


def paired(rows: list[dict[str, Any]], new: str = "T1", ref: str = "T0", seed: str = BOOTSTRAP["seed"],
           metrics: Sequence[tuple[str, int]] = PAIRED_METRICS) -> dict[str, Any]:
    """Paired `new - ref` per scenario. Defaults give the P1-04 T1-vs-T0 report unchanged."""
    a_, b_ = new.lower(), ref.lower()
    out: dict[str, Any] = {}
    for metric, sign in metrics:
        diffs, by_fam = [], {}
        for r in rows:
            a, b = _metric(r[new], metric), _metric(r[ref], metric)
            if a is None or b is None:
                continue
            diffs.append(a - b)
            by_fam.setdefault(r["family_id"], []).append(a - b)
        better = sum(sign * d > 0 for d in diffs)
        worse = sum(sign * d < 0 for d in diffs)
        fam_means = {f: statistics.fmean(d) for f, d in by_fam.items()}
        fb = sum(sign * d > 0 for d in fam_means.values())
        fw = sum(sign * d < 0 for d in fam_means.values())
        ci = cluster_bootstrap(by_fam, BOOTSTRAP["resamples"], f"{seed}/{metric}", BOOTSTRAP["level"])
        out[metric] = {"direction": "higher_is_better" if sign > 0 else "lower_is_better", "pairs": len(diffs),
                       f"mean_diff_{a_}_minus_{b_}": statistics.fmean(diffs) if diffs else None,
                       f"{a_}_better": better, "equal": len(diffs) - better - worse, f"{a_}_worse": worse,
                       "sign_test_p_two_sided": ev._sign_test(better, worse),
                       "families": len(fam_means), f"family_{a_}_better": fb, f"family_{a_}_worse": fw,
                       "family_sign_test_p_two_sided": ev._sign_test(fb, fw),
                       "family_cluster_bootstrap_ci95": list(ci) if ci else None}
    return out


def aggregate(rows: list[dict[str, Any]], latencies: dict[str, list[float]],
              names: Sequence[str] = POLICY_ORDER) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in names:
        rs = [r[name] for r in rows]
        completed = [x["completion_step"] for x in rs if x["completion_step"] is not None]
        mo: Counter = Counter()
        for x in rs:
            mo.update(x["movement_outcomes"])
        attempts = sum(x["moves_attempted"] for x in rs)
        person_steps = sum(x["person_steps"] for x in rs)
        lat = latencies[name]
        cpu = [x["timing"]["policy_cpu_ms"] for x in rs]
        ctx: Counter = Counter()
        for x in rs:
            ctx.update(x["casualty_context"])
        agg = {
            "scenarios": len(rs), "failures": sum(x["failure"] is not None for x in rs),
            "people": sum(x["people"] for x in rs), "evacuated": sum(x["evacuated"] for x in rs),
            "casualties": sum(x["casualties"] for x in rs), "unresolved": sum(x["unresolved"] for x in rs),
            "trapped_at_start": sum(x["trapped_at_start"] for x in rs),
            "trapped_at_end": sum(x["trapped_at_end"] for x in rs),
            "evacuated_fraction_mean": statistics.fmean(x["evacuated_fraction"] for x in rs),
            "casualty_fraction_mean": statistics.fmean(x["casualty_fraction"] for x in rs),
            "all_resolved_episodes": len(completed),
            "completion_steps_mean": statistics.fmean(completed) if completed else None,
            "steps_mean": statistics.fmean(x["steps"] for x in rs),
            "movement_outcomes": dict(sorted(mo.items())),
            "moves_attempted": attempts, "blocked": sum(x["blocked"] for x in rs),
            "collisions": sum(x["collisions"] for x in rs),
            "blocked_per_100_attempts": 100 * sum(x["blocked"] for x in rs) / attempts if attempts else None,
            "person_steps": person_steps,
            "waiting_rate": mo["stayed"] / person_steps if person_steps else None,
            "non_progress_rate": (mo["stayed"] + sum(x["blocked"] for x in rs)) / person_steps if person_steps else None,
            "reversals": sum(x["reversals"] for x in rs),
            "reversals_per_100_moves": 100 * sum(x["reversals"] for x in rs) / mo["moved"] if mo["moved"] else None,
            "bottleneck_blocked": sum(x["bottleneck_blocked"] for x in rs),
            "exit_utilization": (sum(x["evacuated"] for x in rs) / sum(x["open_exit_steps"] for x in rs)),
            "busiest_exit_share_mean": statistics.fmean(x["busiest_exit_share"] for x in rs
                                                        if x["busiest_exit_share"] is not None),
            "casualty_context": dict(sorted(ctx.items())),
            "prediction_mismatches": sum(x["prediction_mismatches"] for x in rs),
            "decisions": len(lat),
            "decision_wall_ms": {"mean": statistics.fmean(lat) if lat else None, "p50": ev._pct(lat, 0.5),
                                 "p95": ev._pct(lat, 0.95), "p99": ev._pct(lat, 0.99), "max": max(lat) if lat else None},
            "policy_cpu_ms_per_episode": {"mean": statistics.fmean(cpu), "max": max(cpu)},
            "policy_cpu_s_total": sum(cpu) / 1000,
            "episode_cpu_s": {"mean": statistics.fmean(x["timing"]["episode_cpu_s"] for x in rs),
                              "max": max(x["timing"]["episode_cpu_s"] for x in rs)},
        }
        if name == "T1":
            st: Counter = Counter()
            for x in rs:
                st.update(x["t1"]["plan_status"])
            dec = agg["decisions"]
            bf = sum(x["t1"]["budget_failure_decisions"] for x in rs)
            agg["t1"] = {"plan_status": dict(sorted(st.items())),
                         "budget_failure_decisions": bf,
                         "planning_failure_rate": (bf + agg["failures"]) / dec if dec else None,
                         "budget_failure_person_decisions": sum(x["t1"]["budget_failure_person_decisions"] for x in rs),
                         "no_admissible_action_person_decisions":
                             sum(x["t1"]["no_admissible_action_person_decisions"] for x in rs),
                         "repair_rounds": sum(x["t1"]["repair_rounds"] for x in rs),
                         "converted_to_stay": sum(x["t1"]["converted_to_stay"] for x in rs)}
        out[name] = agg
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


def t1_worse(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every scenario where T1 is worse than T0 on casualties, evacuated fraction or unresolved."""
    out = []
    for r in rows:
        a, b = r["T1"], r["T0"]
        if a["casualties"] > b["casualties"] or a["evacuated_fraction"] < b["evacuated_fraction"] \
                or a["unresolved"] > b["unresolved"]:
            out.append({"position": r["position"], "family_id": r["family_id"], "label": r["label"],
                        "fire_model": r["fire_model"], "enter_fire": r["enter_fire"], "people": a["people"],
                        "T0": {k: b[k] for k in ("evacuated", "casualties", "unresolved", "steps")},
                        "T1": {k: a[k] for k in ("evacuated", "casualties", "unresolved", "steps")},
                        "T0_casualty_context": b["casualty_context"], "T1_casualty_context": a["casualty_context"]})
    return out


def evaluate(limit: int | None = None, index_path: str | Path = ev.DEFAULT_INDEX,
             dev_path: str | Path = ev.DEFAULT_DEV) -> dict[str, Any]:
    meta, items = load_dev(index_path, dev_path)
    if limit is not None:
        items = items[:limit]
    rows: list[dict[str, Any]] = []
    latencies: dict[str, list[float]] = {n: [] for n in POLICY_ORDER}
    identities: dict[str, Any] = {}
    for it in items:
        s: Scenario = it["scenario"]
        row: dict[str, Any] = {"position": it["position"], "lineage_id": it["lineage_id"],
                               "family_id": it["family_id"], "scenario_hash": it["scenario_hash"],
                               "simulator_seed": s.seed, "label": it["label"], "grid": [s.height, s.width],
                               "fire_model": s.fire_model, "enter_fire": s.enter_fire, "max_steps": s.max_steps}
        for pol in policies():
            rec, lat = run_episode(s, pol)
            row[pol.identity.name] = rec
            latencies[pol.identity.name].extend(lat)
            identities.setdefault(pol.identity.name, pol.identity.to_dict())
        rows.append(row)
    det_rows = [{k: (ev.deterministic_view(v) if k in POLICY_ORDER else v) for k, v in r.items()} for r in rows]
    return {
        "eval_version": EVAL_VERSION, "dev": meta, "scenarios_evaluated": len(rows),
        "families_evaluated": len({r["family_id"] for r in rows}),
        "policies": identities,
        "matched_seeds": "both policies run each scenario with its own scenario seed",
        "bootstrap": BOOTSTRAP,
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "processor": platform.processor(), **ev._revision()},
        "aggregates": aggregate(rows, latencies),
        "paired_t1_minus_t0": paired(rows),
        "strata": strata(rows),
        "t1_worse_than_t0": t1_worse(rows),
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
