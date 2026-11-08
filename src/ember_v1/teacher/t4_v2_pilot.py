"""T4 v2 independent synthetic confirmation pilot (overnight P4), preregistered in docs/P1_T4_RISK_PRIORITY_V2.md.

    python -m ember_v1.teacher.t4_v2_pilot [--out PATH] [--wall-limit-s S]

RUN ONCE. A failed gate is a result; the protocol forbids changing v1.1.0 and
re-running this sample. One CPU process, no GPU, no network.

1 Scenarios: synthetic only, namespace `t4-pilot@2/...` (disjoint from the v1
  pilot `t4-pilot@1/...`; content hashes are checked against every v1 pilot
  scenario and against the two inspected crowded v1 layouts; a match is
  rejected, recorded and not replaced). Cells: `pilot_cells_v2`.
2 Closed loop: A0, A1 and A2 (v1.1.0) on every cell. 16x20 cells run full
  episodes, 48x64 cells the first PREFIX_DECISIONS decisions.
3 Neutral paired states: frozen T1 runs every cell closed loop. On every T1
  observation A0 and A2 (v1.1.0) decide (not executed). These states are not
  created by A0 or A2. They give G5c and the paired immediate-action metrics.
4 DEV cost: the v1 cost model (`t4_pilot.estimate`) on the frozen-T1 DEV
  workload profile (`t4_pilot.dev_profile`; public workload and T1 time only;
  no T4 decision on DEV).
5 Gate (all must hold): G1-G4 exactly as v1 (`t4_pilot` thresholds); G5a
  precedence, G5b A0 integrity, G5c demonstrable priority effect (see `gate_v2`).
  T4_V2_FULL_DEV_FEASIBLE if all pass, else T4_V2_BLOCKED (with the failed list).
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from ..contract import validate_actions, validate_observation
from ..gen.layouts import BUILDERS, components
from ..gen.rng import GenRng, derive_seed
from ..sim.canonical import digest
from ..sim.movement import resolve_moves
from ..sim.scenario import Scenario
from ..sim.simulator import Simulator
from . import evaluate as ev
from .grid import PublicGrid, exit_distance_field
from .t1 import T1CongestionAware
from .t2_pilot import LAYOUT_PARAMS, peak_rss_mb
from .t4_eval import step_metrics
from .t4_pilot import (BUDGET_HIT_LIMIT, DEV_CPU_LIMIT_S, EPISODE_FACTOR, FALLBACK_LIMIT, MEMORY_LIMIT_MB,
                       OVERRUN_MS, PREFIX_DECISIONS, _bfs, _pct, dev_profile, estimate, nominal_work, pilot_cells,
                       pilot_scenario, workload)
from .t4_risk import STRATA, assess
from .t4_v11 import ABLATIONS_V11, YIELD_EFFECTS

PILOT_VERSION = "t4-v2-confirmation-pilot@1.0.0"
NAMESPACE = "t4-pilot@2"
DEFAULT_OUT = ev.REPO / "data/baselines/p1-08-t4-v2-pilot.json"
WALL_LIMIT_S = 3600.0
SMALL, LARGE = (16, 20), (48, 64)
TOPOLOGIES = ("open_room", "corridor", "multi_room", "irregular")
SIZES = (1, 10, 50, 100, 200)
VARIANTS = ("A0", "A1", "A2")
AT_RISK = ("critical", "high")
# The two crowded v1 pilot cells inspected during the v1 diagnosis (excluded as confirmation cases).
INSPECTED_V1 = ({"kind": "base", "topology": "multi_room", "grid": SMALL, "n": 200, "model": "stochastic", "exits": 3},
                {"kind": "base", "topology": "open_room", "grid": SMALL, "n": 100, "model": "deterministic", "exits": 3})


def pilot_cells_v2() -> list[dict[str, Any]]:
    def c(kind, topo, grid, n, model="stochastic", exits=3):
        return {"kind": kind, "topology": topo, "grid": grid, "n": n, "model": model, "exits": exits,
                "mode": "full_episode" if grid == SMALL else "prefix"}
    cells = [c("base", t, g, n) for g in (SMALL, LARGE) for t in TOPOLOGIES for n in SIZES]
    cells += [c("base", t, g, n, "deterministic") for g in (SMALL, LARGE) for t in ("open_room", "corridor", "multi_room")
              for n in (10, 100)]
    cells += [c("exits", t, LARGE, 50, exits=e) for t in ("open_room", "multi_room") for e in (1, 6)]
    cells += [c("bottleneck", "corridor", SMALL, n, m, exits=1) for n in (50, 100) for m in ("stochastic", "deterministic")]
    cells += [c("high_id_risk", t, SMALL, 50, m) for t in ("open_room", "multi_room", "corridor")
              for m in ("stochastic", "deterministic")]
    cells += [c("near_entrapment", "corridor", SMALL, n, m) for n in (10, 50) for m in ("stochastic", "deterministic")]
    cells += [c("unreachable", "multi_room", g, 50, m) for g in (SMALL, LARGE) for m in ("stochastic", "deterministic")]
    return cells


def pilot_scenario_v2(cell: dict[str, Any], *, namespace: str = NAMESPACE, sid_prefix: str = "t4-pilot2") -> Scenario:
    """Synthetic v2 pilot scenario (own namespace and seeds; not a generator plan, family or split).
    The keyword defaults are the frozen v2 values; later pilots pass their own namespace and prefix."""
    (h, w), topo, n, model, n_exits, kind = cell["grid"], cell["topology"], cell["n"], cell["model"], \
        cell["exits"], cell["kind"]
    parts = (namespace, kind, topo, h, w, n, model, n_exits)
    rng = GenRng(derive_seed(*parts))
    params = dict(LAYOUT_PARAMS)
    if kind == "unreachable":
        params["sealed_room_pct"] = 100
    walls = BUILDERS[topo](h, w, rng, params)
    comps = sorted(components(h, w, walls), key=len, reverse=True)
    main = comps[0]
    border = sorted(p for p in main if p[0] in (0, h - 1) or p[1] in (0, w - 1)) or sorted(main)
    exits = rng.sample(border, min(n_exits, len(border)))
    rest = sorted(set(main) - set(exits))
    ign = rng.choice(rest)
    rest = [p for p in rest if p != ign]
    dist = _bfs(h, w, walls, ign)
    if kind == "near_entrapment":
        pool = [p for p in rest if dist.get(p, 99) <= 6]
        people = rng.sample(pool, min(n, len(pool)))
    elif kind == "unreachable":
        sealed = sorted(p for comp in comps[1:] for p in comp)
        k = min(n // 2, len(sealed))
        people = rng.sample(sealed, k) + rng.sample(rest, min(n - k, len(rest)))
    else:
        people = rng.sample(rest, min(n, len(rest) - 1))
    if kind == "high_id_risk":                       # nearest to the fire gets the highest ID
        people = sorted(people, key=lambda p: (-dist.get(p, 10 ** 6), p))
    fire = {"model": model, "spread_interval": 2, "exits_flammable": False}
    if model == "stochastic":
        fire["spread_probability"] = 0.5
    sid = f"{sid_prefix}-{kind}-{topo}-{h}x{w}-{n}-{model}-{n_exits}"
    return Scenario.from_dict({
        "schema_version": "1.0", "scenario_id": sid, "grid": {"width": w, "height": h},
        "obstacles": [list(p) for p in sorted(walls)],
        "exits": [{"id": f"E{i}", "position": list(p)} for i, p in enumerate(exits)],
        "people": [{"id": f"P{i:03d}", "position": list(p)} for i, p in enumerate(people)],
        "ignitions": [list(ign)], "fire": fire,
        "movement": {"conflict_resolution": "lowest_id_wins", "enter_fire": "blocked"},
        "termination": {"max_steps": 2 * (h + w)}, "seed": derive_seed(f"{namespace}/sim", *parts[1:])})


def content_hash(s: Scenario) -> str:
    """Scenario content without its ID and simulator seed."""
    d = s.to_dict()
    d.pop("scenario_id", None)
    d.pop("seed", None)
    return digest(d)


def layout_hash(s: Scenario) -> str:
    d = s.to_dict()
    return digest({"grid": d["grid"], "obstacles": d["obstacles"], "exits": d["exits"]})


def select_cells() -> tuple[list[tuple[dict[str, Any], Scenario]], list[dict[str, Any]], dict[str, Any]]:
    v1_content = {content_hash(pilot_scenario(c)) for c in pilot_cells()}
    v1_seeds = {pilot_scenario(c).seed for c in pilot_cells()}
    inspected = {layout_hash(pilot_scenario({**c, "mode": "full_episode"})) for c in INSPECTED_V1}
    kept, rejected, seen = [], [], set()
    for cell in pilot_cells_v2():
        s = pilot_scenario_v2(cell)
        ch, lh = content_hash(s), layout_hash(s)
        why = ("content_equals_v1_pilot" if ch in v1_content else "seed_equals_v1_pilot" if s.seed in v1_seeds
               else "layout_equals_inspected_v1_crowded_cell" if lh in inspected
               else "duplicate_content_in_v2" if ch in seen else None)
        if why:
            rejected.append({"cell": _cell_json(cell), "reason": why, "content_hash": ch})
            continue
        seen.add(ch)
        kept.append((cell, s))
    return kept, rejected, {"v1_scenarios_checked": len(v1_content), "inspected_v1_layouts": len(inspected)}


def _cell_json(cell: dict[str, Any]) -> dict[str, Any]:
    return {k: (list(v) if k == "grid" else v) for k, v in cell.items()}


# ----- closed-loop runs ---------------------------------------------------------------------------------
def _spearman(xs: list[float], ys: list[float]) -> float | None:
    def ranks(v):
        order = sorted(range(len(v)), key=v.__getitem__)
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    if len(xs) < 3:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    sx = sum((a - mx) ** 2 for a in rx) ** 0.5
    sy = sum((b - my) ** 2 for b in ry) ** 0.5
    return None if sx == 0 or sy == 0 else sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (sx * sy)


def run_cell(cell: dict[str, Any], s: Scenario, v: str, deadline: float, *, ablations: dict = ABLATIONS_V11
             ) -> tuple[list[dict[str, Any]], dict[str, Any], bool]:
    pol = ablations[v]()
    risk_cfg = dict(ablations["A2"]().identity.config)
    sim = Simulator(s)
    rows: list[dict[str, Any]] = []
    tot: Counter = Counter()
    stall: Counter = Counter()
    max_stall: Counter = Counter()
    init = None
    while not sim.terminated and (cell["mode"] == "full_episode" or len(rows) < PREFIX_DECISIONS):
        if time.perf_counter() > deadline:
            return rows, {}, False
        obs = sim.observe()
        if init is None:
            init = {p: r["stratum"] for p, r in assess(PublicGrid.from_observation(obs), obs, risk_cfg).people.items()}
        wl = workload(obs)
        d = pol.decide(obs)
        dg = d.diagnostics
        validate_actions(validate_observation(obs), d.actions)
        rec = sim.step(d.actions)
        mm = sum(dg["people"][pid]["predicted_outcome"] != m["outcome"] for pid, m in rec["movements"].items())
        blocked = sum(m["outcome"].startswith("blocked_") for m in rec["movements"].values())
        e0 = time.perf_counter_ns()
        sm = step_metrics(obs, d.actions, rec, risk_cfg)
        eval_ms = (time.perf_counter_ns() - e0) / 1e6
        tot.update(sm)
        for pid, m in rec["movements"].items():
            if dg["people"][pid]["route_found"] and m["outcome"] != "moved":
                stall[pid] += 1
                max_stall[pid] = max(max_stall[pid], stall[pid])
            else:
                stall[pid] = 0
        pr, tx = dg["priority"], dg["transaction"]
        order = pr["order"]
        rank_d = [(i, dg["people"][p]["static_distance_target"]) for i, p in enumerate(order)
                  if dg["people"][p]["route_found"]]
        rho = _spearman([a for a, _ in rank_d], [b for _, b in rank_d])
        a0_inv = sum(1 for pid, p in dg["people"].items() if p["yielded_to"] is not None
                     and p["priority_effect"] in YIELD_EFFECTS and p["yield_origin"] is not None
                     and pid < p["yield_origin"]) if v == "A0" else 0
        a0_bad_over_id = sum(1 for pid, p in dg["people"].items() if p["priority_effect"] in YIELD_EFFECTS
                             and p["yield_over_id_rule"] and not (p["yield_kind"] == "chain"
                                                                  and p["yield_origin"] is not None
                                                                  and p["yield_origin"] < pid)) if v == "A0" else 0
        w = nominal_work(wl)
        rows.append({
            "variant": v, "kind": cell["kind"], "topology": cell["topology"], "grid": list(cell["grid"]),
            "people_requested": cell["n"], "fire_model": cell["model"], "exits": cell["exits"],
            "mode": cell["mode"], "step": obs["step"], **wl, "nominal_work": w,
            "limit_hit": dg["budget"]["limit_hit"], "fallback_people": len(dg["fallback_people"]),
            "route_people": sum(p["route_found"] for p in dg["people"].values()),
            "no_route_people": len(dg["no_route"]), "diverted": len(dg["diverted"]),
            "expansions": dg["expansions"], "converted_to_stay": len(dg["converted_to_stay"]),
            "predicted_blocked": sum(n for o, n in dg["predicted_outcomes"].items() if o.startswith("blocked_")),
            "blocked_moves": blocked, "prediction_mismatches": mm,
            "verification_mismatches": tx["verification_mismatches"],
            "precedence_audit_violations": tx["precedence_audit_violations"],
            "precedence_inversions": pr["precedence_inversions"],
            "a0_id_inversions": a0_inv, "a0_unexplained_over_id_yields": a0_bad_over_id,
            "yields": pr["yields"], "yields_over_id_rule": pr["yields_over_id_rule"],
            "chain_yields": pr["chain_yields"], "chain_yields_over_id_rule": pr["chain_yields_over_id_rule"],
            "effective_yields": pr["effective_yields"],
            "effective_yields_over_id_rule": pr["effective_yields_over_id_rule"],
            "vacate_requests": tx.get("vacate_requests", 0), "vacate_success": tx.get("vacate_success", 0),
            "vacate_refusals": sum(n for k, n in tx.items() if k.startswith("vacate_failed_")),
            "push_rollbacks": tx.get("push_rollbacks", 0), "max_push_depth": tx.get("max_push_depth", 0),
            "free_progress_unused": pr["free_progress_unused"],
            "rank_vs_target_distance_spearman": rho,
            "contests": sm.get("contests", 0), "contest_success": sm.get("contest_success", 0),
            "contests_over_id": sm.get("contests_over_id", 0),
            "contest_success_over_id": sm.get("contest_success_over_id", 0),
            "unnecessary_waits": sm.get("unnecessary_waits", 0),
            "cpu_ms": d.runtime["cpu_ms"], "wall_ms": d.runtime["wall_ms"], "eval_ms": eval_ms,
            "alpha": d.runtime["wall_ms"] / w, "alpha_eval": eval_ms / w,
        })
    ep: dict[str, Any] = {}
    if cell["mode"] == "full_episode":
        fin = {p["id"]: p["status"] for p in sim.private_state()["people"]}
        by = Counter(f"{init.get(pid, 'n/a')}/{st}" for pid, st in fin.items()) if init else Counter()
        c = Counter(fin.values())
        ep = {**_cell_json(cell), "variant": v, "people": len(s.people), "steps": sim.observe()["step"],
              "termination_reason": sim.termination_reason, "evacuated": c["evacuated"], "casualties": c["casualty"],
              "unresolved": c["active"], "decisions": len(rows), "final_by_initial_stratum": dict(sorted(by.items())),
              "max_consecutive_stall": max(max_stall.values(), default=0),
              "step_metrics": dict(sorted(tot.items()))}
    return rows, ep, True


# ----- neutral paired public states (frozen T1 closed loop) -----------------------------------------------
def _ends(obs: dict[str, Any], acts: dict[str, str]) -> dict[str, tuple[int, int]]:
    pg = PublicGrid.from_observation(obs)
    res = resolve_moves(height=pg.height, width=pg.width, walls=pg.walls, burning=pg.burning,
                        enter_fire=pg.enter_fire, positions=dict(pg.active), actions=acts)
    return {pid: pos for pid, (_, pos) in res.items()}


def paired_state(obs: dict[str, Any], a0: dict[str, Any], a2: dict[str, Any], strata: dict[str, str]
                 ) -> tuple[Counter, list[dict[str, Any]]]:
    """Immediate-action comparison of A0 and A2 on ONE public observation (no execution)."""
    pg = PublicGrid.from_observation(obs)
    field = exit_distance_field(pg)
    exits = {tuple(e["position"]) for e in obs["map"]["exits"]} if "map" in obs else set()
    pos = dict(pg.active)
    e0, e2 = _ends(obs, a0.actions), _ends(obs, a2.actions)

    def prog(pid: str, end: tuple[int, int]) -> int | None:
        """Static exit-distance decrease of one person's resolved move (end cell without an exit path: 0)."""
        s = pos[pid]
        if s not in field:
            return None
        de = 0 if end in exits else field[end][0] if end in field else field[s][0]
        return field[s][0] - de
    p0 = {pid: prog(pid, e0[pid]) for pid in pos}
    p2 = {pid: prog(pid, e2[pid]) for pid in pos}
    out: Counter = Counter()
    cases: list[dict[str, Any]] = []
    sidx = {pid: STRATA.index(st) for pid, st in strata.items()}
    d0, d2 = a0.diagnostics["people"], a2.diagnostics["people"]
    for pid in pos:
        if p0[pid] is None:
            continue
        at = strata[pid] in AT_RISK
        diff = p2[pid] - p0[pid]
        key = "at_risk" if at else "lower_risk"
        out[f"{key}/persons"] += 1
        out[f"{key}/progress_a0"] += p0[pid]
        out[f"{key}/progress_a2"] += p2[pid]
        out[f"{key}/a2_more_progress"] += diff > 0
        out[f"{key}/a2_less_progress"] += diff < 0
        if diff <= 0:
            continue
        for low, r in d2.items():                     # verified links from a lower-risk person
            if sidx[low] <= sidx[pid]:
                continue
            link = None
            if r["priority_effect"] in YIELD_EFFECTS and r["yielded_to"] == pid and \
                    r["contested_cell"] == list(e2[pid]):
                link = "yield"
            elif r.get("target_exit") is not None and d0[low].get("target_exit") is not None and \
                    r["target_exit"] != d0[low]["target_exit"] and d2[pid].get("target_exit") == d0[low]["target_exit"]:
                link = "exit_reassignment"
            if link:
                out[f"g5c/{link}"] += 1
                if len(cases) < 20:
                    cases.append({"step": obs["step"], "higher_risk": pid, "stratum": strata[pid], "lower_risk": low,
                                  "lower_stratum": strata[low], "link": link, "progress_a0": p0[pid],
                                  "progress_a2": p2[pid], "lower_progress_a0": p0[low], "lower_progress_a2": p2[low]})
                break
    return out, cases


def run_neutral(cell: dict[str, Any], s: Scenario, deadline: float, *, ablations: dict = ABLATIONS_V11
                ) -> tuple[Counter, list[dict[str, Any]], int, bool]:
    t1 = T1CongestionAware()
    a0, a2 = ablations["A0"](), ablations["A2"]()
    risk_cfg = dict(a2.identity.config)
    sim = Simulator(s)
    tot: Counter = Counter()
    cases: list[dict[str, Any]] = []
    n = 0
    while not sim.terminated and (cell["mode"] == "full_episode" or n < PREFIX_DECISIONS):
        if time.perf_counter() > deadline:
            return tot, cases, n, False
        obs = sim.observe()
        strata = {p: r["stratum"] for p, r in assess(PublicGrid.from_observation(obs), obs, risk_cfg).people.items()}
        d0, d2 = a0.decide(obs), a2.decide(obs)
        for d, v in ((d0, "A0"), (d2, "A2")):
            tot[f"{v}/precedence_audit_violations"] += d.diagnostics["transaction"]["precedence_audit_violations"]
            tot[f"{v}/precedence_inversions"] += d.diagnostics["priority"]["precedence_inversions"]
            tot[f"{v}/predicted_blocked"] += sum(c for o, c in d.diagnostics["predicted_outcomes"].items()
                                                 if o.startswith("blocked_"))
        c, cs = paired_state(obs, d0, d2, strata)
        tot.update(c)
        cases.extend(cs[: max(0, 20 - len(cases))])
        n += 1
        sim.step(t1.act(obs))
    return tot, cases, n, True


# ----- gate -------------------------------------------------------------------------------------------------
def gate_v2(cfg: dict[str, Any], pilot: list[dict[str, Any]], neutral: Counter, complete: bool,
            peak_mb: float | None, est: dict[str, Any]) -> dict[str, Any]:
    budget = float(cfg["cpu_budget_ms"])
    max_cpu = max((r["cpu_ms"] for r in pilot), default=None)
    g1 = {"pilot_complete_inside_wall_limit": complete,
          "predicted_blocked": sum(r["predicted_blocked"] for r in pilot),
          "blocked_moves": sum(r["blocked_moves"] for r in pilot),
          "prediction_mismatches": sum(r["prediction_mismatches"] for r in pilot),
          "verification_mismatches": sum(r["verification_mismatches"] for r in pilot),
          "max_decision_cpu_ms": max_cpu, "max_allowed_cpu_ms": budget + OVERRUN_MS,
          "peak_rss_mb": peak_mb, "memory_limit_mb": MEMORY_LIMIT_MB}
    g1["pass"] = bool(complete and pilot and g1["predicted_blocked"] == 0 and g1["blocked_moves"] == 0
                      and g1["prediction_mismatches"] == 0 and max_cpu is not None and max_cpu <= budget + OVERRUN_MS
                      and peak_mb is not None and peak_mb <= MEMORY_LIMIT_MB)
    up = est.get("upper") or {}
    g2 = {"upper_dev_comparison_cpu_s": up.get("comparison_cpu_s_total"), "limit_s": DEV_CPU_LIMIT_S}
    g2["pass"] = g2["upper_dev_comparison_cpu_s"] is not None and g2["upper_dev_comparison_cpu_s"] <= DEV_CPU_LIMIT_S
    fr = {v: (up.get("variants") or {}).get(v, {}).get("predicted_cap_fraction") for v in VARIANTS}
    g3 = {"upper_cap_fraction": fr, "limit": BUDGET_HIT_LIMIT}
    g3["pass"] = all(x is not None and x <= BUDGET_HIT_LIMIT for x in fr.values())
    g4: dict[str, Any] = {"fallback_limit": FALLBACK_LIMIT, "budget_limit": BUDGET_HIT_LIMIT}
    ok4 = True
    for v in VARIANTS:
        rs = [r for r in pilot if r["variant"] == v]
        routable = sum(r["route_people"] for r in rs)
        fb = sum(r["fallback_people"] for r in rs)
        lim = sum(r["limit_hit"] is not None for r in rs)
        g4[v] = {"fallback_person_decisions": fb, "routable_person_decisions": routable,
                 "fallback_fraction": fb / routable if routable else None, "budget_limited_decisions": lim,
                 "decisions": len(rs), "budget_limited_fraction": lim / len(rs) if rs else None}
        ok4 = ok4 and bool(rs and routable and fb / routable <= FALLBACK_LIMIT and lim / len(rs) <= BUDGET_HIT_LIMIT)
    g4["pass"] = ok4
    audit = sum(r["precedence_audit_violations"] for r in pilot) + sum(
        neutral[f"{v}/precedence_audit_violations"] for v in ("A0", "A2"))
    inv = sum(r["precedence_inversions"] for r in pilot) + sum(neutral[f"{v}/precedence_inversions"] for v in ("A0", "A2"))
    g5a = {"precedence_audit_violations": audit, "precedence_inversions": inv,
           "neutral_state_predicted_blocked": neutral["A0/predicted_blocked"] + neutral["A2/predicted_blocked"],
           "pass": audit == 0 and inv == 0}
    a0 = [r for r in pilot if r["variant"] == "A0"]
    g5b = {"a0_id_inversions": sum(r["a0_id_inversions"] for r in a0),
           "a0_unexplained_over_id_yields": sum(r["a0_unexplained_over_id_yields"] for r in a0),
           "a0_yields_over_id_rule_all_chain": sum(r["yields_over_id_rule"] for r in a0),
           "a0_chain_yields_over_id_rule": sum(r["chain_yields_over_id_rule"] for r in a0)}
    g5b["pass"] = bool(a0) and g5b["a0_id_inversions"] == 0 and g5b["a0_unexplained_over_id_yields"] == 0
    g5c = {"verified_cases_yield": neutral["g5c/yield"], "verified_cases_exit_reassignment":
           neutral["g5c/exit_reassignment"], "neutral_decisions": neutral["neutral_decisions"],
           "at_risk_progress_a0": neutral["at_risk/progress_a0"], "at_risk_progress_a2": neutral["at_risk/progress_a2"],
           "at_risk_person_states": neutral["at_risk/persons"]}
    # (i) at least one verified direct yield to an at-risk person; (ii) at-risk immediate progress not below A0.
    g5c["pass"] = bool(g5c["verified_cases_yield"] >= 1 and g5c["at_risk_person_states"] > 0
                       and g5c["at_risk_progress_a2"] >= g5c["at_risk_progress_a0"])
    checks = (("G1_bounded", g1), ("G2_dev_cpu", g2), ("G3_cap_hits", g3), ("G4_planner_health", g4),
              ("G5a_precedence", g5a), ("G5b_a0_integrity", g5b), ("G5c_priority_effect", g5c))
    failed = [n for n, g in checks if not g["pass"]]
    return {"G1": g1, "G2": g2, "G3": g3, "G4": g4, "G5a": g5a, "G5b": g5b, "G5c": g5c, "failed": failed,
            "decision": "T4_V2_FULL_DEV_FEASIBLE" if not failed else "T4_V2_BLOCKED"}


def mechanism(pilot: list[dict[str, Any]], episodes: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("yields", "yields_over_id_rule", "chain_yields", "chain_yields_over_id_rule", "effective_yields",
            "effective_yields_over_id_rule", "vacate_requests", "vacate_success", "vacate_refusals", "push_rollbacks",
            "free_progress_unused", "contests", "contest_success", "contests_over_id", "contest_success_over_id",
            "unnecessary_waits", "diverted", "converted_to_stay", "blocked_moves", "prediction_mismatches",
            "verification_mismatches", "precedence_audit_violations", "precedence_inversions")
    out: dict[str, Any] = {}
    for v in VARIANTS:
        rs = [r for r in pilot if r["variant"] == v]
        m = {k: sum(r[k] for r in rs) for k in keys}
        m["max_push_depth"] = max((r["max_push_depth"] for r in rs), default=0)
        m["priority_enforcement_rate"] = m["contest_success"] / m["contests"] if m["contests"] else None
        rho = [r["rank_vs_target_distance_spearman"] for r in rs if r["rank_vs_target_distance_spearman"] is not None]
        m["rank_vs_target_distance_spearman_mean"] = statistics.fmean(rho) if rho else None
        m["decisions"] = len(rs)
        m["cpu_s_total"] = sum(r["cpu_ms"] for r in rs) / 1000
        m["wall_ms"] = {q: _pct([r["wall_ms"] for r in rs], p) for q, p in (("p50", .5), ("p95", .95), ("p99", .99))}
        m["wall_ms"]["max"] = max((r["wall_ms"] for r in rs), default=None)
        eps = [e for e in episodes if e["variant"] == v]
        fb: Counter = Counter()
        for e in eps:
            fb.update(e["final_by_initial_stratum"])
        m["episodes"] = {"count": len(eps), "people": sum(e["people"] for e in eps),
                         "evacuated": sum(e["evacuated"] for e in eps), "casualties": sum(e["casualties"] for e in eps),
                         "unresolved": sum(e["unresolved"] for e in eps),
                         "final_by_initial_stratum": dict(sorted(fb.items())),
                         "max_consecutive_stall": max((e["max_consecutive_stall"] for e in eps), default=0)}
        out[v] = m
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--wall-limit-s", type=float, default=WALL_LIMIT_S)
    args = ap.parse_args(argv)
    if args.out.exists():
        sys.exit(f"{args.out} exists: the v2 confirmation pilot runs once (protocol section 6)")
    t_start = time.perf_counter()
    kept, rejected, excl = select_cells()
    report = build_report(kept, rejected, excl, t_start=t_start, wall_limit_s=args.wall_limit_s)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "gate": report["gate"], "resources": report["resources"]}, indent=1))


def build_report(kept: list[tuple[dict[str, Any], Scenario]], rejected: list[dict[str, Any]], excl: dict[str, Any], *,
                 t_start: float, wall_limit_s: float, ablations: dict = ABLATIONS_V11, namespace: str = NAMESPACE,
                 pilot_version: str = PILOT_VERSION) -> dict[str, Any]:
    """Run every kept cell (A0, A1, A2 closed loop and neutral paired states), the DEV cost profile and the gate.
    The keyword defaults are the frozen v2 pilot; later pilots pass their own identities and namespace."""
    deadline = t_start + wall_limit_s
    cpu0 = time.process_time()
    teachers = {v: cls().identity.to_dict() for v, cls in ablations.items()}
    teachers["T1"] = T1CongestionAware().identity.to_dict()
    cfg = dict(ablations["A2"]().identity.config)
    pilot: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    neutral: Counter = Counter()
    g5c_cases: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    complete = True
    for cell, s in kept:
        try:
            for v in VARIANTS:
                r, ep, ok = run_cell(cell, s, v, deadline, ablations=ablations)
                pilot.extend(r)
                complete &= ok
                if ep:
                    episodes.append(ep)
            tot, cases, n, ok = run_neutral(cell, s, deadline, ablations=ablations)
            neutral.update(tot)
            neutral["neutral_decisions"] += n
            g5c_cases.extend({"cell": _cell_json(cell), **c} for c in cases[: max(0, 40 - len(g5c_cases))])
            complete &= ok
        except Exception as exc:              # recorded; the pilot is then incomplete and G1 fails
            errors.append({"cell": _cell_json(cell), "error": f"{type(exc).__name__}: {exc}"})
            complete = False
        print(json.dumps({"cell": f"{cell['kind']} {cell['topology']} {cell['grid']} N={cell['n']} {cell['model']} "
                                  f"exits={cell['exits']}", "rows": len(pilot), "errors": len(errors)}), flush=True)
        if not complete and time.perf_counter() > deadline:
            break
    pilot_s = time.perf_counter() - t_start
    profile, ref_s, profile_complete = dev_profile(deadline) if complete else ([], {}, False)
    est = estimate(cfg, pilot, profile, ref_s) if profile_complete else {"error": "dev profile incomplete"}
    peak = peak_rss_mb()
    return {
        "pilot_version": pilot_version, "namespace": namespace, "teachers": teachers,
        "limits": {"wall_limit_s": wall_limit_s, "overrun_ms": OVERRUN_MS, "memory_limit_mb": MEMORY_LIMIT_MB,
                   "dev_cpu_limit_s": DEV_CPU_LIMIT_S, "budget_hit_limit": BUDGET_HIT_LIMIT,
                   "fallback_limit": FALLBACK_LIMIT, "episode_factor": EPISODE_FACTOR, "workers": 1},
        "cells_defined": len(pilot_cells_v2()), "cells_run": len(kept), "rejected_cells": rejected,
        "exclusion_checks": excl,
        "cell_content_hashes": [{"cell": _cell_json(c), "content_hash": content_hash(s)} for c, s in kept],
        "errors": errors, "mechanism": mechanism(pilot, episodes), "neutral_paired_states": dict(sorted(neutral.items())),
        "g5c_example_cases": g5c_cases, "pilot_episodes": episodes, "pilot": pilot,
        "pilot_complete": complete, "dev_profile_complete": profile_complete,
        "dev_profile_summary": {"decisions": len(profile),
                                "active_mean": statistics.fmean(p["active"] for p in profile) if profile else None},
        "estimate": est,
        "gate": gate_v2(cfg, pilot, neutral, complete and profile_complete, peak, est),
        "resources": {"wall_s": time.perf_counter() - t_start, "pilot_wall_s": pilot_s,
                      "process_cpu_s": time.process_time() - cpu0, "peak_rss_mb": peak},
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "processor": platform.processor(), **ev._revision()},
        "dev_use": "frozen T1 replay for public workload and reference timing only; no T4 decision on DEV",
    }


if __name__ == "__main__":
    main()
