"""Bounded CPU feasibility pilot for T4 and its A0/A1 ablations (issue #24), and the full-DEV cost gate.

    python -m ember_v1.teacher.t4_pilot [--out PATH] [--wall-limit-s S]

Preregistered in docs/P1_T4_RISK_PRIORITY.md section 6 before it runs. One CPU
process, no multiprocessing, no GPU, no network.

1 Microbenchmark: one decision of each variant on sim.bench.room 32x32, N = 50.
2 Pilot on SYNTHETIC scenarios only (`gen.layouts` builders, pilot-only seeds
  `t4-pilot@1/...`; not a generator plan, family or split). Person IDs are
  zero-padded (P000...) so string order = numeric order. Cells (`pilot_cells`):
    base:          16x20 and 48x64 x {open_room, corridor, multi_room} x N = 1, 10, 50, 100, 200,
                   stochastic p = 0.5, 3 exits, random ID arrangement;
    deterministic: both grids x {open_room, corridor} x N = 10, 100;
    exits:         48x64 x {open_room, multi_room} x N = 50 with 1 and 6 exits;
    high_id_risk:  16x20 x {open_room, multi_room} x {stochastic, deterministic} x N = 50; IDs ascend
                   with decreasing fire distance (the nearest person has the highest ID);
    near_entrapment: 16x20 corridor x N = 10, 50; people only within 6 cells of the ignition;
    unreachable:   16x20 and 48x64 multi_room with a sealed room x N = 50; half the people
                   start in components with no exit (initially unreachable).
  Every cell runs A0, A1 and A2. 16x20 cells run full closed-loop episodes,
  48x64 cells the first PREFIX_DECISIONS decisions. No DEV, train, calibration,
  test, blind or EMBERBench scenario is used for any T4 decision.
3 DEV workload profile: frozen T1 replayed on the 48 `p1-02-v1` DEV scenarios
  (T1 DEV results are public since P1-04): public workload per decision and
  T1 decision time. T4 / A0 / A1 are NOT run on DEV.
4 Cost model per DEV decision d and variant v (wall ms, one thread):
      W_d = area_d * (1 + E_d * (3 + [stochastic])) + 40 * active_d
      est_{v,d} = alpha_v * W_d, capped at cpu_budget_ms + OVERRUN_MS;
  evaluator metrics: alpha_eval * W_d per decision and policy (4 policies).
  alpha from pilot decisions without a budget hit. Central: median alpha;
  upper: p90 alpha x episode factor 1.25. Full DEV comparison = T1 (measured)
  + A0 + A1 + A2 + evaluator metrics.
5 Gate (all must hold):
  G1 bounded: all cells inside the wall limit; complete valid maps; 0 blocked
     moves and 0 prediction mismatches; max decision CPU <= cpu_budget_ms +
     OVERRUN_MS; peak RSS <= MEMORY_LIMIT_MB;
  G2 upper full-DEV comparison CPU <= DEV_CPU_LIMIT_S;
  G3 predicted DEV decisions at the cap (upper alpha) <= 5 % for every variant;
  G4 planner health per variant: fallback person-decisions <= 5 % of routable
     person-decisions; budget-limited decisions <= 5 %;
  G5 mechanism: A2 has >= 1 effective yield over the ID rule; A0 has 0 yields
     over the ID rule (its planning order is the ID order).
  FULL_DEV_FEASIBLE if all pass; BLOCKED_BY_COMPUTE if G1-G4 fail;
  BLOCKED_BY_MECHANISM_CHECK if only G5 fails.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

from ..contract import validate_actions, validate_observation
from ..gen.layouts import BUILDERS, components
from ..gen.rng import GenRng, derive_seed
from ..sim.bench import room
from ..sim.scenario import Scenario
from ..sim.simulator import Simulator
from . import evaluate as ev
from .grid import PublicGrid, exit_distance_field
from .t1 import T1CongestionAware
from .t2_pilot import LAYOUT_PARAMS, peak_rss_mb
from .t4 import ABLATIONS
from .t4_eval import step_metrics

PILOT_VERSION = "t4-cpu-pilot@1.0.0"
DEFAULT_OUT = ev.REPO / "data/baselines/p1-08-t4-pilot.json"
WALL_LIMIT_S = 3600.0
OVERRUN_MS = 500.0
MEMORY_LIMIT_MB = 1024.0
DEV_CPU_LIMIT_S = 2 * 3600.0
BUDGET_HIT_LIMIT = 0.05
FALLBACK_LIMIT = 0.05
EPISODE_FACTOR = 1.25
EVAL_POLICIES = 4
SMALL, LARGE = (16, 20), (48, 64)
TOPOLOGIES = ("open_room", "corridor", "multi_room")
SIZES = (1, 10, 50, 100, 200)
PREFIX_DECISIONS = 12
VARIANTS = ("A0", "A1", "A2")


def _bfs(h: int, w: int, walls: frozenset, src: tuple[int, int]) -> dict[tuple[int, int], int]:
    d = {src: 0}
    q = deque([src])
    while q:
        r, c = q.popleft()
        for n in ((r - 1, c), (r + 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= n[0] < h and 0 <= n[1] < w and n not in walls and n not in d:
                d[n] = d[(r, c)] + 1
                q.append(n)
    return d


def pilot_scenario(cell: dict[str, Any]) -> Scenario:
    """Synthetic pilot scenario. Pilot-only seed path; not a generator plan, family or split."""
    (h, w), topo, n, model, n_exits, kind = cell["grid"], cell["topology"], cell["n"], cell["model"], \
        cell["exits"], cell["kind"]
    parts = ("t4-pilot@1", kind, topo, h, w, n, model, n_exits)
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
    sid = f"t4-pilot-{kind}-{topo}-{h}x{w}-{n}-{model}-{n_exits}"
    return Scenario.from_dict({
        "schema_version": "1.0", "scenario_id": sid, "grid": {"width": w, "height": h},
        "obstacles": [list(p) for p in sorted(walls)],
        "exits": [{"id": f"E{i}", "position": list(p)} for i, p in enumerate(exits)],
        "people": [{"id": f"P{i:03d}", "position": list(p)} for i, p in enumerate(people)],
        "ignitions": [list(ign)], "fire": fire,
        "movement": {"conflict_resolution": "lowest_id_wins", "enter_fire": "blocked"},
        "termination": {"max_steps": 2 * (h + w)}, "seed": derive_seed("t4-pilot@1/sim", *parts[1:])})


def pilot_cells() -> list[dict[str, Any]]:
    def c(kind, topo, grid, n, model="stochastic", exits=3):
        return {"kind": kind, "topology": topo, "grid": grid, "n": n, "model": model, "exits": exits,
                "mode": "full_episode" if grid == SMALL else "prefix"}
    cells = [c("base", t, g, n) for g in (SMALL, LARGE) for t in TOPOLOGIES for n in SIZES]
    cells += [c("base", t, g, n, "deterministic") for g in (SMALL, LARGE) for t in ("open_room", "corridor")
              for n in (10, 100)]
    cells += [c("base", t, LARGE, 50, exits=e) for t in ("open_room", "multi_room") for e in (1, 6)]
    cells += [c("high_id_risk", t, SMALL, 50, m) for t in ("open_room", "multi_room")
              for m in ("stochastic", "deterministic")]
    cells += [c("near_entrapment", "corridor", SMALL, n, m) for n in (10, 50) for m in ("stochastic", "deterministic")]
    cells += [c("unreachable", "multi_room", g, 50) for g in (SMALL, LARGE)]
    return cells


def workload(obs: dict[str, Any]) -> dict[str, Any]:
    """Public workload of one observation (no T4 decision)."""
    o = validate_observation(obs)
    g = PublicGrid.from_observation(o)
    field = exit_distance_field(g)
    return {"active": len(g.active), "routable": sum(p in field for _, p in g.active),
            "area": g.height * g.width, "safe_exits": len(g.safe_exits()),
            "stochastic": o["rules"]["fire_model"] == "stochastic"}


def nominal_work(wl: dict[str, Any]) -> float:
    return wl["area"] * (1 + wl["safe_exits"] * (3 + int(wl["stochastic"]))) + 40 * wl["active"]


def microbench() -> dict[str, Any]:
    out = {}
    for v, cls in ABLATIONS.items():
        pol = cls()
        sim = Simulator(room(32, 50, 1, "stochastic"))
        for _ in range(3):
            sim.step(pol.act(sim.observe()))
        obs = sim.observe()
        best = float("inf")
        for _ in range(5):
            w0 = time.perf_counter_ns()
            pol.decide(obs)
            best = min(best, (time.perf_counter_ns() - w0) / 1e6)
        out[v] = round(best, 3)
    return {"scene": "sim.bench.room 32x32 N=50 stochastic, step 3", "decision_wall_ms_best_of_5": out}


def run_cell(cell: dict[str, Any], v: str, deadline: float) -> tuple[list[dict[str, Any]], dict[str, Any], bool]:
    pol = ABLATIONS[v]()
    risk_cfg = dict(ABLATIONS["A2"]().identity.config)
    s = pilot_scenario(cell)
    sim = Simulator(s)
    rows: list[dict[str, Any]] = []
    tot: Counter = Counter()
    while not sim.terminated and (cell["mode"] == "full_episode" or len(rows) < PREFIX_DECISIONS):
        if time.perf_counter() > deadline:
            return rows, {}, False
        obs = sim.observe()
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
        pr = dg["priority"]
        w = nominal_work(wl)
        rows.append({
            "variant": v, "kind": cell["kind"], "topology": cell["topology"], "grid": list(cell["grid"]),
            "people_requested": cell["n"], "fire_model": cell["model"], "exits": cell["exits"],
            "mode": cell["mode"], "step": obs["step"], **wl, "nominal_work": w,
            "limit_hit": dg["budget"]["limit_hit"], "fallback_people": len(dg["fallback_people"]),
            "route_people": sum(p["route_found"] for p in dg["people"].values()),
            "no_route_people": len(dg["no_route"]), "diverted": len(dg["diverted"]),
            "expansions": dg["expansions"], "repair_rounds": dg["repair_rounds"],
            "converted_to_stay": len(dg["converted_to_stay"]),
            "predicted_blocked": sum(n for o, n in dg["predicted_outcomes"].items() if o.startswith("blocked_")),
            "blocked_moves": blocked, "prediction_mismatches": mm,
            "yields": pr["yields"], "yields_over_id_rule": pr["yields_over_id_rule"],
            "effective_yields": pr["effective_yields"],
            "effective_yields_over_id_rule": pr["effective_yields_over_id_rule"],
            "vacate_refused": pr["vacate_refused"], "free_progress_unused": pr["free_progress_unused"],
            "contests": sm.get("contests", 0), "contest_success": sm.get("contest_success", 0),
            "contests_over_id": sm.get("contests_over_id", 0),
            "contest_success_over_id": sm.get("contest_success_over_id", 0),
            "unnecessary_waits": sm.get("unnecessary_waits", 0),
            "cpu_ms": d.runtime["cpu_ms"], "wall_ms": d.runtime["wall_ms"], "eval_ms": eval_ms,
            "alpha": d.runtime["wall_ms"] / w, "alpha_eval": eval_ms / w,
        })
    ep: dict[str, Any] = {}
    if cell["mode"] == "full_episode":
        fin = Counter(p["status"] for p in sim.private_state()["people"])
        ep = {**{k: (list(x) if k == "grid" else x) for k, x in cell.items()}, "variant": v, "people": len(s.people),
              "steps": sim.observe()["step"], "termination_reason": sim.termination_reason,
              "evacuated": fin["evacuated"], "casualties": fin["casualty"], "unresolved": fin["active"],
              "decisions": len(rows), "step_metrics": dict(sorted(tot.items()))}
    return rows, ep, True


def run_pilot(deadline: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    for cell in pilot_cells():
        for v in VARIANTS:
            r, ep, ok = run_cell(cell, v, deadline)
            rows.extend(r)
            if not ok:
                return rows, episodes, False
            if ep:
                episodes.append(ep)
            print(json.dumps({"cell": f"{cell['kind']} {cell['topology']} {cell['grid']} N={cell['n']} "
                                      f"{cell['model']} exits={cell['exits']}", "variant": v, "decisions": len(r),
                              "max_wall_ms": round(max(x["wall_ms"] for x in r), 1) if r else None}), flush=True)
    return rows, episodes, True


def dev_profile(deadline: float) -> tuple[list[dict[str, Any]], dict[str, float], bool]:
    """Frozen-T1 DEV replay: public workload per decision and T1 policy time. T4 is not run on DEV."""
    from .t1_eval import load_dev
    _, items = load_dev()
    t1 = T1CongestionAware()
    out: list[dict[str, Any]] = []
    t1_ms = 0.0
    for it in items:
        sim = Simulator(it["scenario"])
        while not sim.terminated:
            if time.perf_counter() > deadline:
                return out, {"t1_wall_s": t1_ms / 1000}, False
            obs = sim.observe()
            w0 = time.perf_counter_ns()
            acts = t1.act(obs)
            t1_ms += (time.perf_counter_ns() - w0) / 1e6
            out.append({"position": it["position"], "step": obs["step"], **workload(obs)})
            sim.step(acts)
    return out, {"t1_wall_s": t1_ms / 1000}, True


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def estimate(cfg: dict[str, Any], pilot: list[dict[str, Any]], profile: list[dict[str, Any]],
             ref_s: dict[str, float]) -> dict[str, Any]:
    cap = float(cfg["cpu_budget_ms"]) + OVERRUN_MS
    alphas: dict[str, dict[str, Any]] = {}
    for v in VARIANTS:
        xs = [r["alpha"] for r in pilot if r["variant"] == v and r["limit_hit"] is None]
        alphas[v] = {"n": len(xs), "median": _pct(xs, 0.5), "p90": _pct(xs, 0.9), "max": max(xs) if xs else None}
    xe = [r["alpha_eval"] for r in pilot]
    alphas["evaluator"] = {"n": len(xe), "median": _pct(xe, 0.5), "p90": _pct(xe, 0.9), "max": max(xe) if xe else None}
    out: dict[str, Any] = {"alpha_wall_ms_per_work_unit": alphas, "dev_decisions": len(profile),
                           "dev_reference_policy_wall_s": ref_s,
                           "hard_bound_t4_variants_s": 3 * len(profile) * cap * EPISODE_FACTOR / 1000}
    if any(alphas[v]["median"] is None for v in (*VARIANTS, "evaluator")):
        return {**out, "error": "no unbudgeted pilot decisions to fit alpha"}
    for name, key, factor in (("central", "median", 1.0), ("upper", "p90", EPISODE_FACTOR)):
        per: dict[str, Any] = {}
        total = 0.0
        for v in VARIANTS:
            ms, hits = 0.0, 0
            per_pos: Counter = Counter()
            for p in profile:
                est = alphas[v][key] * nominal_work(p)
                if est > cap:
                    hits += 1
                    est = cap
                ms += est
                per_pos[p["position"]] += est
            s = ms * factor / 1000
            total += s
            per[v] = {"cpu_s_total": s, "predicted_cap_decisions": hits,
                      "predicted_cap_fraction": hits / len(profile) if profile else None,
                      "max_episode_cpu_s": max(per_pos.values()) * factor / 1000 if per_pos else None}
        eval_s = EVAL_POLICIES * sum(alphas["evaluator"][key] * nominal_work(p) for p in profile) * factor / 1000
        t1_s = ref_s["t1_wall_s"] * factor
        out[name] = {"variants": per, "evaluator_cpu_s": eval_s, "t1_cpu_s": t1_s, "episode_factor": factor,
                     "comparison_cpu_s_total": total + eval_s + t1_s,
                     "comparison_cpu_h_total": (total + eval_s + t1_s) / 3600}
    return out


def gate(cfg: dict[str, Any], pilot: list[dict[str, Any]], complete: bool, peak_mb: float | None,
         est: dict[str, Any]) -> dict[str, Any]:
    budget = float(cfg["cpu_budget_ms"])
    max_cpu = max((r["cpu_ms"] for r in pilot), default=None)
    g1 = {"pilot_complete_inside_wall_limit": complete,
          "predicted_blocked": sum(r["predicted_blocked"] for r in pilot),
          "blocked_moves": sum(r["blocked_moves"] for r in pilot),
          "prediction_mismatches": sum(r["prediction_mismatches"] for r in pilot),
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
    a2 = sum(r["effective_yields_over_id_rule"] for r in pilot if r["variant"] == "A2")
    a0 = sum(r["yields_over_id_rule"] for r in pilot if r["variant"] == "A0")
    g5 = {"a2_effective_yields_over_id_rule": a2, "a0_yields_over_id_rule": a0, "pass": a2 >= 1 and a0 == 0}
    compute = [n for n, g in (("G1_bounded", g1), ("G2_dev_cpu", g2), ("G3_cap_hits", g3),
                              ("G4_planner_health", g4)) if not g["pass"]]
    limiting = compute + ([] if g5["pass"] else ["G5_mechanism"])
    decision = ("FULL_DEV_FEASIBLE" if not limiting else "BLOCKED_BY_COMPUTE" if compute
                else "BLOCKED_BY_MECHANISM_CHECK")
    return {"G1": g1, "G2": g2, "G3": g3, "G4": g4, "G5": g5, "limiting": limiting, "decision": decision}


def summarize(pilot: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for v in VARIANTS:
        for grid in (SMALL, LARGE):
            for n in SIZES:
                rs = [r for r in pilot if r["variant"] == v and r["people_requested"] == n and tuple(r["grid"]) == grid]
                if not rs:
                    continue
                wall = [r["wall_ms"] for r in rs]
                cpu = [r["cpu_ms"] for r in rs]
                routable = sum(r["route_people"] for r in rs)
                out[f"{v} {grid[0]}x{grid[1]} N={n}"] = {
                    "decisions": len(rs), "active_mean": statistics.fmean(r["active"] for r in rs),
                    "wall_ms_p50": _pct(wall, 0.5), "wall_ms_p95": _pct(wall, 0.95), "wall_ms_p99": _pct(wall, 0.99),
                    "wall_ms_max": max(wall), "cpu_ms_p50": _pct(cpu, 0.5), "cpu_ms_max": max(cpu),
                    "fallback_fraction": sum(r["fallback_people"] for r in rs) / routable if routable else None,
                    "budget_limited": sum(r["limit_hit"] is not None for r in rs)}
    return out


def mechanism(pilot: list[dict[str, Any]], episodes: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("yields", "yields_over_id_rule", "effective_yields", "effective_yields_over_id_rule", "vacate_refused",
            "free_progress_unused", "contests", "contest_success", "contests_over_id", "contest_success_over_id",
            "unnecessary_waits", "diverted", "repair_rounds", "converted_to_stay", "blocked_moves",
            "prediction_mismatches")
    out: dict[str, Any] = {}
    for v in VARIANTS:
        rs = [r for r in pilot if r["variant"] == v]
        m = {k: sum(r[k] for r in rs) for k in keys}
        m["priority_enforcement_rate"] = m["contest_success"] / m["contests"] if m["contests"] else None
        m["priority_enforcement_rate_over_id"] = (m["contest_success_over_id"] / m["contests_over_id"]
                                                  if m["contests_over_id"] else None)
        m["decisions"] = len(rs)
        m["cpu_s_total"] = sum(r["cpu_ms"] for r in rs) / 1000
        m["wall_ms"] = {"p50": _pct([r["wall_ms"] for r in rs], 0.5), "p95": _pct([r["wall_ms"] for r in rs], 0.95),
                        "p99": _pct([r["wall_ms"] for r in rs], 0.99), "max": max(r["wall_ms"] for r in rs)}
        eps = [e for e in episodes if e["variant"] == v]
        sm: Counter = Counter()
        for e in eps:
            sm.update(e["step_metrics"])
        m["episodes"] = {"count": len(eps), "people": sum(e["people"] for e in eps),
                         "evacuated": sum(e["evacuated"] for e in eps), "casualties": sum(e["casualties"] for e in eps),
                         "unresolved": sum(e["unresolved"] for e in eps),
                         "waits_by_stratum": {k.split("/")[1]: sm[k] for k in sorted(sm) if k.startswith("waits/")},
                         "person_steps_by_stratum": {k.split("/")[1]: sm[k] for k in sorted(sm)
                                                     if k.startswith("person_steps/")}}
        out[v] = m
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--wall-limit-s", type=float, default=WALL_LIMIT_S)
    args = ap.parse_args(argv)
    t_start = time.perf_counter()
    deadline = t_start + args.wall_limit_s
    cpu0 = time.process_time()
    teachers = {v: cls().identity.to_dict() for v, cls in ABLATIONS.items()}
    cfg = dict(ABLATIONS["A2"]().identity.config)
    micro = microbench()
    print(json.dumps({"microbench": micro}), flush=True)
    pilot, episodes, pilot_complete = run_pilot(deadline)
    pilot_s = time.perf_counter() - t_start
    if pilot_complete:
        profile, ref_s, profile_complete = dev_profile(deadline)
    else:
        profile, ref_s, profile_complete = [], {}, False
    est = estimate(cfg, pilot, profile, ref_s) if profile_complete else {"error": "dev profile incomplete"}
    peak = peak_rss_mb()
    report = {
        "pilot_version": PILOT_VERSION, "teachers": teachers,
        "limits": {"wall_limit_s": args.wall_limit_s, "overrun_ms": OVERRUN_MS, "memory_limit_mb": MEMORY_LIMIT_MB,
                   "dev_cpu_limit_s": DEV_CPU_LIMIT_S, "budget_hit_limit": BUDGET_HIT_LIMIT,
                   "fallback_limit": FALLBACK_LIMIT, "episode_factor": EPISODE_FACTOR, "workers": 1},
        "microbench": micro,
        "cells": len(pilot_cells()),
        "pilot_summary_by_variant_grid_n": summarize(pilot), "mechanism": mechanism(pilot, episodes),
        "pilot_episodes": episodes, "pilot": pilot,
        "pilot_complete": pilot_complete, "dev_profile_complete": profile_complete,
        "dev_profile_summary": {"decisions": len(profile),
                                "active_mean": statistics.fmean(p["active"] for p in profile) if profile else None,
                                "stochastic_decisions": sum(p["stochastic"] for p in profile)},
        "estimate": est,
        "gate": gate(cfg, pilot, pilot_complete and profile_complete, peak, est),
        "resources": {"wall_s": time.perf_counter() - t_start, "pilot_wall_s": pilot_s,
                      "process_cpu_s": time.process_time() - cpu0, "peak_rss_mb": peak},
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "processor": platform.processor(), **ev._revision()},
        "dev_use": "frozen T1 replay for public workload and reference timing only; no T4/A0/A1 decision on DEV",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "gate": report["gate"], "estimate": est,
                      "resources": report["resources"]}, indent=1))


if __name__ == "__main__":
    main()
