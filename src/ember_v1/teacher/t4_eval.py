"""Gated paired comparison of T4 (A2) and its A0/A1 ablations with frozen T1 on independent DEV (issue #24).

    python -m ember_v1.teacher.t4_eval --pilot-report PATH [--out PATH]

Preregistered in docs/P1_T4_RISK_PRIORITY.md section 7. It refuses to run
unless the CPU feasibility pilot report (`t4_pilot`) has gate decision
FULL_DEV_FEASIBLE for the same T4, T4-A0 and T4-A1 identities. Same DEV gate
as P1-03..P1-06 (`t1_eval.load_dev`): only the `dev` split of plan
`p1-02-v1`; train, calibration, the locked test, blind sets and EMBERBench
are never read. T1, A0, A1 and A2 run every scenario with the scenario's own
simulator seed (identical fire trajectory). A policy receives only
`Simulator.observe()`.

Evaluator-side metrics (`step_metrics`, `person_outcomes`) use the frozen T4
risk module on PUBLIC observations only, with the same frozen configuration
for every policy. Initial risk strata come from the step-0 observation, never
from outcomes. Evaluator-private values (final statuses, trapped set, family
IDs, labels) are read for metrics only and never reach a policy.

THIS HARNESS HAS NOT BEEN RUN ON DEV. It needs the reviewed pilot report.
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
from ..sim.simulator import Simulator
from . import evaluate as ev
from . import t1_eval
from .grid import PublicGrid
from .interface import ReferencePolicy
from .t1 import T1CongestionAware
from .t4 import ABLATIONS, T4DistancePriority, T4NoPriority, T4RiskPriority
from .t4_risk import STRATA, assess

EVAL_VERSION = "t4-vs-t1-dev@1.0.0"
DEFAULT_OUT = ev.REPO / "data/baselines/p1-08-t4-dev-v1.json"
DEFAULT_PILOT = ev.REPO / "data/baselines/p1-08-t4-pilot.json"
POLICY_ORDER = ("T1", "A0", "A1", "A2")
AT_RISK = ("critical", "high")
COMPARISONS = (("A2", "T1"), ("A2", "A0"), ("A2", "A1"), ("A1", "A0"))
PRIMARY = ("A2", "T1", "casualties")
BOOTSTRAP_SEEDS = {f"{a}-{b}": f"t4-{a.lower()}-vs-{b.lower()}-family-cluster-bootstrap@1" for a, b in COMPARISONS}
WALL_LIMIT_FACTOR = 1.5
WALL_LIMIT_MIN_S = 900.0
WALL_LIMIT_MAX_S = 6 * 3600.0
PAIRED_METRICS = (("casualties", -1), ("evacuated_fraction", +1), ("unresolved", -1),
                  ("casualties_plus_unresolved", -1), ("at_risk_evacuated_fraction", +1),
                  ("at_risk_casualties", -1), ("steps", -1), ("completion_step", -1),
                  ("blocked_per_100_attempts", -1), ("collisions", -1), ("reversals_per_100_moves", -1),
                  ("at_risk_wait_rate", -1), ("low_risk_wait_rate", -1), ("unnecessary_waits", -1),
                  ("priority_enforcement_rate", +1), ("policy_cpu_ms", -1))


class GateError(RuntimeError):
    """The CPU feasibility gate does not permit the full DEV run."""


def risk_policy() -> T4RiskPriority:
    """The frozen risk module configuration used for every evaluator-side stratum."""
    return T4RiskPriority()


def policies() -> dict[str, ReferencePolicy]:
    return {"T1": T1CongestionAware(), "A0": T4NoPriority(), "A1": T4DistancePriority(), "A2": T4RiskPriority()}


# ----- evaluator-side metrics (public observation + simulator record) --------------------------------
def _intended(ra, pid: str) -> int | None:
    r = ra.people[pid]
    if r["nominal_exit"] is None:
        return None
    d = ra.fields[r["nominal_exit"]].dist
    s = ra.pos[pid]
    for _, n in ra.cells.succ[s]:
        if d[n] == d[s] - 1:
            return n
    return None


def step_metrics(obs: dict[str, Any], actions: dict[str, str], rec: dict[str, Any], cfg: dict[str, Any]
                 ) -> dict[str, Any]:
    """Priority contests, enforcement, waiting and unnecessary waits for one step (same rule for all policies).

    Contest: an at-risk person p (stratum critical/high) whose intended cell (first step of the static descent
    to its nominal viable exit) is occupied by, or also intended by, a less urgent person q. Success: p
    advances (reaches an exit or a cell nearer its nominal exit). `over_id`: some such q has a lower ID than
    p, so the resolver alone would favour q in a direct contest.
    """
    pg = PublicGrid.from_observation(obs)
    ra = assess(pg, obs, cfg)
    g = ra.cells
    people = ra.people
    rank = {pid: STRATA.index(r["stratum"]) for pid, r in people.items()}
    intended = {pid: _intended(ra, pid) for pid in people}
    occ = {i: pid for pid, i in ra.pos.items()}
    mv = rec["movements"]
    end = {pid: g.idx(tuple(m["to"])) for pid, m in mv.items()}
    entered = {end[pid] for pid, m in mv.items() if m["outcome"] == "moved"}
    burning = set(g.burning_idx)

    def advanced(pid: str) -> bool:
        r = people[pid]
        if mv[pid]["outcome"] != "moved" or r["nominal_exit"] is None:
            return False
        d = ra.fields[r["nominal_exit"]].dist
        e = end[pid]
        return g.safe_exit[e] or d[e] < d[ra.pos[pid]]

    out: Counter = Counter()
    for pid, r in people.items():
        st = r["stratum"]
        nonprog = mv[pid]["outcome"] != "moved"
        out[f"person_steps/{st}"] += 1
        out[f"waits/{st}"] += nonprog
        t = intended[pid]
        if actions[pid] == "stay" and t is not None and t not in burning and t not in entered:
            o = occ.get(t)
            if o is None or mv[o]["outcome"] == "moved":
                out["unnecessary_waits"] += 1
        if st not in AT_RISK or t is None:
            continue
        rivals = [q for q in people if q != pid and rank[q] > rank[pid] and (occ.get(t) == q or intended[q] == t)]
        if not rivals:
            continue
        ok = advanced(pid)
        out["contests"] += 1
        out["contest_success"] += ok
        if any(q < pid for q in rivals):
            out["contests_over_id"] += 1
            out["contest_success_over_id"] += ok
    return dict(out)


class _Recorder:
    """Wraps a policy: records observations and actions; the policy still sees only the observation."""

    def __init__(self, pol: ReferencePolicy) -> None:
        self.pol = pol
        self.identity = pol.identity
        self.obs: list[dict[str, Any]] = []
        self.acts: list[dict[str, str]] = []
        self.diag: Counter = Counter()

    def decide(self, obs):
        d = self.pol.decide(obs)
        self.obs.append(obs)
        self.acts.append(d.actions)
        dg = d.diagnostics
        if "priority" in dg:
            pr = dg["priority"]
            self.diag["decisions"] += 1
            self.diag["budget_limited_decisions"] += dg["budget"]["limit_hit"] is not None
            self.diag["fallback_person_decisions"] += len(dg["fallback_people"])
            self.diag["routable_person_decisions"] += sum(p["route_found"] for p in dg["people"].values())
            self.diag["person_decisions"] += dg["active"]
            self.diag["repair_rounds"] += dg["repair_rounds"]
            self.diag["converted_to_stay"] += len(dg["converted_to_stay"])
            self.diag["diverted"] += len(dg["diverted"])
            for k in ("yields", "yields_over_id_rule", "effective_yields", "effective_yields_over_id_rule",
                      "vacate_refused", "free_progress_unused"):
                self.diag[k] += pr[k]
            for k, v in pr["effects"].items():
                self.diag[f"effect/{k}"] += v
        return d


def person_outcomes(scenario: Scenario, rec: _Recorder, cfg: dict[str, Any]) -> dict[str, Any]:
    """Replay the recorded actions (no policy call) and derive per-stratum outcomes and step metrics."""
    sim = Simulator(scenario)
    init = assess(PublicGrid.from_observation(sim.observe()), sim.observe(), cfg).people
    stratum0 = {pid: r["stratum"] for pid, r in init.items()}
    totals: Counter = Counter()
    for obs, acts in zip(rec.obs, rec.acts):
        if obs != sim.observe():
            raise AssertionError("replay diverged from the recorded observation")
        r = sim.step(acts)
        totals.update(step_metrics(obs, acts, r, cfg))
    final = {p["id"]: p["status"] for p in sim.private_state()["people"]}
    by: dict[str, dict[str, int]] = {}
    for st in STRATA:
        ids = [pid for pid, s in stratum0.items() if s == st]
        c = Counter(final[pid] for pid in ids)
        by[st] = {"people": len(ids), "evacuated": c["evacuated"], "casualties": c["casualty"],
                  "unresolved": c["active"]}
    at = [pid for pid, s in stratum0.items() if s in AT_RISK]
    c = Counter(final[pid] for pid in at)
    contests = totals["contests"]
    at_steps = sum(totals[f"person_steps/{s}"] for s in AT_RISK)
    low_steps = sum(totals[f"person_steps/{s}"] for s in ("elevated", "low"))
    return {
        "initial_strata": dict(Counter(stratum0.values())), "by_initial_stratum": by,
        "at_risk_people": len(at), "at_risk_evacuated": c["evacuated"], "at_risk_casualties": c["casualty"],
        "at_risk_evacuated_fraction": c["evacuated"] / len(at) if at else None,
        "step_metrics": dict(sorted(totals.items())),
        "priority_enforcement_rate": totals["contest_success"] / contests if contests else None,
        "priority_enforcement_rate_over_id": (totals["contest_success_over_id"] / totals["contests_over_id"]
                                              if totals["contests_over_id"] else None),
        "at_risk_wait_rate": sum(totals[f"waits/{s}"] for s in AT_RISK) / at_steps if at_steps else None,
        "low_risk_wait_rate": sum(totals[f"waits/{s}"] for s in ("elevated", "low")) / low_steps if low_steps else None,
        "unnecessary_waits": totals["unnecessary_waits"],
        "final_transition_hash": sim.chain_hash,
    }


def run_episode(scenario: Scenario, pol: ReferencePolicy, cfg: dict[str, Any]) -> tuple[dict[str, Any], list[float]]:
    rec = _Recorder(pol)
    record, lat = t1_eval.run_episode(scenario, rec)
    if record["failure"] is None:
        po = person_outcomes(scenario, rec, cfg)
        if po.pop("final_transition_hash") != record["final_transition_hash"]:
            raise AssertionError("replay hash differs from the episode")
        record.update(po)
    record["casualties_plus_unresolved"] = record["casualties"] + record["unresolved"]
    if rec.diag:
        record["t4"] = dict(sorted(rec.diag.items()))
    return record, lat


# ----- gate and statistics -----------------------------------------------------------------------------
def check_gate(pilot: dict[str, Any]) -> float:
    """Return the wall limit (s) if the pilot permits the DEV run for these exact identities, else raise."""
    if pilot.get("gate", {}).get("decision") != "FULL_DEV_FEASIBLE":
        raise GateError(f"pilot gate decision is {pilot.get('gate', {}).get('decision')!r}, not FULL_DEV_FEASIBLE")
    want = {k: cls().identity.config_hash for k, cls in ABLATIONS.items()}
    got = {k: v.get("config_hash") for k, v in pilot.get("teachers", {}).items()}
    if got != want:
        raise GateError("pilot was run for different T4 / ablation configurations")
    return wall_limit(pilot["estimate"]["upper"]["comparison_cpu_s_total"])


def wall_limit(upper_estimate_s: float) -> float:
    return min(WALL_LIMIT_MAX_S, max(WALL_LIMIT_MIN_S, WALL_LIMIT_FACTOR * upper_estimate_s))


def verdict(entry: dict[str, Any], new: str) -> str:
    """Preregistered rule: better/worse only if the family-cluster CI excludes 0 and the family sign test has
    p < 0.05 in the same direction; else no_evidence_of_difference."""
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


def worse(rows: list[dict[str, Any]], new: str, ref: str) -> list[dict[str, Any]]:
    out = []
    keys = ("evacuated", "casualties", "unresolved", "steps", "at_risk_casualties")
    for r in rows:
        a, b = r[new], r[ref]
        if a["casualties"] > b["casualties"] or a["evacuated_fraction"] < b["evacuated_fraction"] \
                or a["unresolved"] > b["unresolved"]:
            out.append({"position": r["position"], "family_id": r["family_id"], "label": r["label"],
                        "fire_model": r["fire_model"], "enter_fire": r["enter_fire"], "people": a["people"],
                        ref: {k: b.get(k) for k in keys}, new: {k: a.get(k) for k in keys}})
    return out


def strata_totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in POLICY_ORDER:
        tot: dict[str, Counter] = {st: Counter() for st in STRATA}
        for r in rows:
            for st, v in r[name].get("by_initial_stratum", {}).items():
                tot[st].update(v)
        out[name] = {st: dict(c) for st, c in tot.items()}
    return out


def evaluate(pilot_report: str | Path = DEFAULT_PILOT) -> dict[str, Any]:
    pilot = json.loads(Path(pilot_report).read_text(encoding="utf-8"))
    limit_s = check_gate(pilot)
    cfg = dict(risk_policy().identity.config)
    meta, items = t1_eval.load_dev()
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
        for name, pol in policies().items():
            r, lat = run_episode(s, pol, cfg)
            row[name] = r
            latencies[name].extend(lat)
        rows.append(row)
        print(json.dumps({"position": it["position"], **{f"{n}_cas": row[n]["casualties"] for n in POLICY_ORDER}}),
              flush=True)
    paired = {f"{a}-{b}": t1_eval.paired(rows, a, b, BOOTSTRAP_SEEDS[f"{a}-{b}"], PAIRED_METRICS) if rows else None
              for a, b in COMPARISONS}
    rules = {k: {m: verdict(e, k.split("-")[0]) for m, e in v.items()} if v else None for k, v in paired.items()}
    det_rows = [{k: (ev.deterministic_view(v) if k in POLICY_ORDER else v) for k, v in r.items()} for r in rows]
    return {
        "eval_version": EVAL_VERSION, "dev": meta, "complete": complete and len(rows) == len(items),
        "scenarios_evaluated": len(rows), "families_evaluated": len({r["family_id"] for r in rows}),
        "policies": {n: p.identity.to_dict() for n, p in policies().items()},
        "primary": {"comparison": f"{PRIMARY[0]}-{PRIMARY[1]}", "metric": PRIMARY[2]},
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
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pilot-report", type=Path, default=DEFAULT_PILOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    try:
        report = evaluate(args.pilot_report)
    except GateError as exc:
        sys.exit(f"T4 DEV evaluation refused: {exc}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "complete": report["complete"],
                      "deterministic_digest": report["deterministic_digest"]}, indent=1))


if __name__ == "__main__":
    main()
