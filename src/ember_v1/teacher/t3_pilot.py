"""Bounded CPU feasibility pilot for T3 (issue #23) and the full-DEV cost gate.

    python -m ember_v1.teacher.t3_pilot [--out PATH] [--wall-limit-s S]

Preregistered in docs/P1_T3_ACO.md section 7 before it runs. One CPU process,
no multiprocessing, no GPU, no network.

1 Microbenchmark of one T3 decision on one synthetic room (sim.bench.room).
2 Pilot on SYNTHETIC scenarios only (`gen.layouts` builders with pilot-only
  seeds `t3-pilot@1/...`; not a generator plan, family or split):
    stochastic: grids 16x20 and 48x64 x {open_room, corridor, multi_room}
                x N = 1, 10, 50, 100, 200 (3 exits);
    deterministic: both grids x {open_room, corridor} x N = 10, 100 (3 exits);
    exit configurations: 48x64 stochastic x {open_room, multi_room} x N = 50
                with 1 and 6 exits.
  Small-grid cells run as full closed-loop T3 episodes (route quality:
  evacuated, casualties, unresolved, blocked moves, prediction mismatches).
  Large-grid cells run the first PREFIX_DECISIONS closed-loop T3 decisions.
  Every T3 decision is recorded. No DEV, train, calibration, test, blind or
  EMBERBench scenario is used for a T3 decision.
3 DEV workload profile: the frozen T1 is replayed on the 48 `p1-02-v1` DEV
  scenarios (as in P1-04 and P1-05; T1 DEV results are already public) to
  record per decision the public workload of T3 (active people, routable
  people, mean static exit distance, grid area) and T1's own decision time.
  T3 is NOT run on DEV here and no T3 output on DEV exists.
4 Cost model per DEV decision d (wall ms, one thread):
      W_d = iterations * (min(ants_per_person * R_d, max_ants_per_iteration) + R_d) * (dbar_d + 1) + area_d
      est_d = alpha * W_d, capped at cpu_budget_ms + OVERRUN_MS,
  R = routable people, dbar = their mean static exit distance. alpha is
  measured on pilot decisions without a budget hit. Central: median alpha;
  upper: 90th percentile alpha x episode-length factor 1.25 (T3 episodes may
  be longer than T1's). Hard bound: every DEV decision at the cap.
  Full DEV comparison cost = T0 + T1 (measured) + T3 (estimated).
5 Gate (all must hold, else BLOCKED_BY_COMPUTE):
  G1 bounded: every planned pilot cell ran inside the wall limit; every map
     complete, no predicted blocked move, predicted = actual outcomes; max
     decision CPU <= cpu_budget_ms + OVERRUN_MS; peak RSS <= MEMORY_LIMIT_MB;
  G2 upper full-DEV comparison CPU (T0 + T1 + T3) <= DEV_CPU_LIMIT_S;
  G3 predicted DEV decisions at the cap (upper alpha) <= 5 %;
  G4 ACO health on the pilot: fallback person-decisions <= 5 % of routable
     person-decisions and budget-limited decisions <= 5 %.
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
from ..sim.bench import room
from ..sim.scenario import Scenario
from ..sim.simulator import Simulator
from . import evaluate as ev
from .grid import PublicGrid, exit_distance_field
from .t0 import T0ShortestPath
from .t1 import T1CongestionAware
from .t2_pilot import LAYOUT_PARAMS, peak_rss_mb
from .t3 import T3DynamicACO

PILOT_VERSION = "t3-cpu-pilot@1.0.0"
DEFAULT_OUT = ev.REPO / "data/baselines/p1-06-t3-pilot.json"
WALL_LIMIT_S = 2700.0
OVERRUN_MS = 500.0                     # allowed CPU past cpu_budget_ms (one ant + coordination)
MEMORY_LIMIT_MB = 1024.0
DEV_CPU_LIMIT_S = 2 * 3600.0           # upper full-DEV comparison CPU (T0 + T1 + T3), one worker
BUDGET_HIT_LIMIT = 0.05
FALLBACK_LIMIT = 0.05
EPISODE_FACTOR = 1.25
SMALL, LARGE = (16, 20), (48, 64)
TOPOLOGIES = ("open_room", "corridor", "multi_room")
SIZES = (1, 10, 50, 100, 200)
PREFIX_DECISIONS = 12


def pilot_scenario(topology: str, n: int, model: str, grid: tuple[int, int], n_exits: int = 3) -> Scenario:
    """Synthetic pilot scenario. Pilot-only seed path; not a generator plan, family or split."""
    h, w = grid
    rng = GenRng(derive_seed("t3-pilot@1", topology, h, w, n, model, n_exits))
    walls = BUILDERS[topology](h, w, rng, dict(LAYOUT_PARAMS))
    main = max(components(h, w, walls), key=len)
    border = sorted(p for p in main if p[0] in (0, h - 1) or p[1] in (0, w - 1)) or sorted(main)
    exits = rng.sample(border, min(n_exits, len(border)))
    rest = sorted(set(main) - set(exits))
    ign = rng.choice(rest)
    rest = [p for p in rest if p != ign]
    people = rng.sample(rest, min(n, len(rest) - 1))
    fire = {"model": model, "spread_interval": 2, "exits_flammable": False}
    if model == "stochastic":
        fire["spread_probability"] = 0.5
    return Scenario.from_dict({
        "schema_version": "1.0", "scenario_id": f"t3-pilot-{topology}-{h}x{w}-{n}-{model}-{n_exits}",
        "grid": {"width": w, "height": h}, "obstacles": [list(p) for p in sorted(walls)],
        "exits": [{"id": f"E{i}", "position": list(p)} for i, p in enumerate(exits)],
        "people": [{"id": f"P{i}", "position": list(p)} for i, p in enumerate(people)],
        "ignitions": [list(ign)], "fire": fire,
        "movement": {"conflict_resolution": "lowest_id_wins", "enter_fire": "blocked"},
        "termination": {"max_steps": 2 * (h + w)}, "seed": derive_seed("t3-pilot@1/sim", topology, h, w, n, model,
                                                                       n_exits)})


def pilot_cells() -> list[dict[str, Any]]:
    cells = [{"topology": t, "grid": g, "n": n, "model": "stochastic", "exits": 3}
             for g in (SMALL, LARGE) for t in TOPOLOGIES for n in SIZES]
    cells += [{"topology": t, "grid": g, "n": n, "model": "deterministic", "exits": 3}
              for g in (SMALL, LARGE) for t in ("open_room", "corridor") for n in (10, 100)]
    cells += [{"topology": t, "grid": LARGE, "n": 50, "model": "stochastic", "exits": e}
              for t in ("open_room", "multi_room") for e in (1, 6)]
    for c in cells:
        c["mode"] = "full_episode" if c["grid"] == SMALL else "prefix"
    return cells


def workload(obs: dict[str, Any]) -> dict[str, Any]:
    """Public T3 workload of one observation (no T3 decision)."""
    o = validate_observation(obs)
    g = PublicGrid.from_observation(o)
    field = exit_distance_field(g)
    ds = [field[p][0] for _, p in g.active if p in field and field[p][0] > 0]
    return {"active": len(g.active), "routable": len(ds), "dbar": statistics.fmean(ds) if ds else 0.0,
            "area": g.height * g.width}


def nominal_work(cfg: dict[str, Any], wl: dict[str, Any]) -> float:
    r = wl["routable"]
    ants = min(cfg["ants_per_person"] * r, cfg["max_ants_per_iteration"])
    return cfg["iterations"] * (ants + r) * (wl["dbar"] + 1) + wl["area"]


def _stretch(dg: dict[str, Any]) -> list[float]:
    return [p["route_length"] / p["static_distance"] for p in dg["people"].values()
            if p["reason"] == "aco_route" and p["static_distance"]]


def microbench(t3: T3DynamicACO) -> dict[str, Any]:
    sim = Simulator(room(32, 50, 1, "stochastic"))
    for _ in range(3):
        sim.step(t3.act(sim.observe()))
    obs = sim.observe()
    best = float("inf")
    for _ in range(5):
        w0 = time.perf_counter_ns()
        d = t3.decide(obs)
        best = min(best, (time.perf_counter_ns() - w0) / 1e6)
    c = d.diagnostics["colony"]
    return {"scene": "sim.bench.room 32x32 N=50 stochastic, step 3", "t3_decision_wall_ms_best_of_5": round(best, 3),
            "ants": c["ants"], "ant_steps": c["ant_steps"],
            "wall_us_per_ant_step": round(1000 * best / c["ant_steps"], 3) if c["ant_steps"] else None}


def run_pilot(t3: T3DynamicACO, deadline: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    cfg = t3.identity.config
    rows: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    for cell in pilot_cells():
        if time.perf_counter() > deadline:
            return rows, episodes, False
        s = pilot_scenario(cell["topology"], cell["n"], cell["model"], cell["grid"], cell["exits"])
        sim = Simulator(s)
        n_dec = 0
        blocked = mismatches = 0
        while not sim.terminated and (cell["mode"] == "full_episode" or n_dec < PREFIX_DECISIONS):
            if time.perf_counter() > deadline:
                return rows, episodes, False
            obs = sim.observe()
            wl = workload(obs)
            d = t3.decide(obs)
            dg = d.diagnostics
            validate_actions(validate_observation(obs), d.actions)
            rec = sim.step(d.actions)
            n_dec += 1
            col = dg["colony"]
            pred_blocked = sorted(o for o in dg["predicted_outcomes"] if o.startswith("blocked_"))
            mm = sum(dg["people"][pid]["predicted_outcome"] != mv["outcome"] for pid, mv in rec["movements"].items())
            blocked += sum(mv["outcome"].startswith("blocked_") for mv in rec["movements"].values())
            mismatches += mm
            st = _stretch(dg)
            w = nominal_work(cfg, wl)
            rows.append({
                "topology": cell["topology"], "grid": list(cell["grid"]), "people_requested": cell["n"],
                "fire_model": cell["model"], "exits": cell["exits"], "mode": cell["mode"], "step": obs["step"],
                **wl, "nominal_work": w,
                "status": dg["status"], "limit_hit": dg["budget"]["limit_hit"],
                "iterations_run": col["iterations_run"], "ants": col["ants"], "ant_steps": col["ant_steps"],
                "ant_success_rate": col["ant_success_rate"], "final_greedy_ants": col["final_greedy_ants"],
                "converged_people": col["converged_people"],
                "best_changed_by_iteration": col["best_changed_by_iteration"],
                "aco_route_people": dg["aco_route_people"], "fallback_people": len(dg["fallback_people"]),
                "no_route_people": len(dg["no_route"]), "unsafe_route_people": len(dg["unsafe_route_people"]),
                "route_stretch_mean": statistics.fmean(st) if st else None,
                "repair_rounds": dg["coordination"]["repair_rounds"],
                "converted_to_stay": len(dg["coordination"]["converted_to_stay"]),
                "predicted_blocked_outcomes": pred_blocked, "prediction_mismatches": mm,
                "t3_cpu_ms": d.runtime["cpu_ms"], "t3_wall_ms": d.runtime["wall_ms"],
                "alpha": d.runtime["wall_ms"] / w if w else None,
            })
        if cell["mode"] == "full_episode":
            fin = Counter(p["status"] for p in sim.private_state()["people"])
            episodes.append({**{k: (list(v) if k == "grid" else v) for k, v in cell.items()},
                             "people": len(s.people), "steps": sim.observe()["step"],
                             "termination_reason": sim.termination_reason, "evacuated": fin["evacuated"],
                             "casualties": fin["casualty"], "unresolved": fin["active"],
                             "blocked_moves": blocked, "prediction_mismatches": mismatches, "decisions": n_dec})
        print(json.dumps({"cell": f"{cell['topology']} {cell['grid']} N={cell['n']} {cell['model']} "
                                  f"exits={cell['exits']}", "decisions": n_dec,
                          "max_wall_ms": round(max(r["t3_wall_ms"] for r in rows[-n_dec:]), 1) if n_dec else None}),
              flush=True)
    return rows, episodes, True


def dev_profile(deadline: float) -> tuple[list[dict[str, Any]], dict[str, float], bool]:
    """Frozen-T1 DEV replay: public T3 workload per decision and T0/T1 policy time. T3 is not run."""
    from .t1_eval import load_dev                      # DEV-only gate of the P1-04 harness
    _, items = load_dev()
    t0, t1 = T0ShortestPath(), T1CongestionAware()
    out: list[dict[str, Any]] = []
    t0_ms = t1_ms = 0.0
    for it in items:
        s: Scenario = it["scenario"]
        sim = Simulator(s)
        while not sim.terminated:
            if time.perf_counter() > deadline:
                return out, {"t0_wall_s": t0_ms / 1000, "t1_wall_s": t1_ms / 1000}, False
            obs = sim.observe()
            w0 = time.perf_counter_ns()
            t0.decide(obs)
            w1 = time.perf_counter_ns()
            acts = t1.act(obs)
            w2 = time.perf_counter_ns()
            t0_ms += (w1 - w0) / 1e6
            t1_ms += (w2 - w1) / 1e6
            out.append({"position": it["position"], "fire_model": s.fire_model, "step": obs["step"], **workload(obs)})
            sim.step(acts)
    return out, {"t0_wall_s": t0_ms / 1000, "t1_wall_s": t1_ms / 1000}, True


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def estimate(cfg: dict, pilot: list[dict[str, Any]], profile: list[dict[str, Any]],
             ref_s: dict[str, float]) -> dict[str, Any]:
    cap = float(cfg["cpu_budget_ms"]) + OVERRUN_MS
    xs = [r["alpha"] for r in pilot if r["alpha"] is not None and r["limit_hit"] is None]
    alpha = {"n": len(xs), "median": _pct(xs, 0.5), "p90": _pct(xs, 0.9), "max": max(xs) if xs else None}
    out: dict[str, Any] = {"alpha_wall_ms_per_work_unit": alpha, "dev_decisions": len(profile),
                           "dev_reference_policy_wall_s": ref_s,
                           "hard_bound_t3_s": len(profile) * cap * EPISODE_FACTOR / 1000}
    if alpha["median"] is None:
        return {**out, "error": "no unbudgeted pilot decisions to fit alpha"}
    ref = ref_s["t0_wall_s"] + ref_s["t1_wall_s"]
    for name, key, factor in (("central", "median", 1.0), ("upper", "p90", EPISODE_FACTOR)):
        total_ms, hits = 0.0, 0
        per_pos: dict[int, float] = {}
        for p in profile:
            est = alpha[key] * nominal_work(cfg, p)
            if est > cap:
                hits += 1
                est = cap
            total_ms += est
            per_pos[p["position"]] = per_pos.get(p["position"], 0.0) + est
        t3_s = total_ms * factor / 1000
        out[name] = {"t3_cpu_s_total": t3_s, "episode_factor": factor,
                     "comparison_cpu_s_total": t3_s + ref * factor,
                     "comparison_cpu_h_total": (t3_s + ref * factor) / 3600,
                     "predicted_cap_decisions": hits,
                     "predicted_cap_fraction": hits / len(profile) if profile else None,
                     "max_episode_t3_cpu_s": max(per_pos.values()) * factor / 1000 if per_pos else None}
    return out


def gate(cfg: dict, pilot: list[dict[str, Any]], complete: bool, peak_mb: float | None,
         est: dict[str, Any]) -> dict[str, Any]:
    budget = float(cfg["cpu_budget_ms"])
    max_cpu = max((r["t3_cpu_ms"] for r in pilot), default=None)
    g1 = {"pilot_complete_inside_wall_limit": complete,
          "all_maps_valid_no_predicted_blocked": all(not r["predicted_blocked_outcomes"] for r in pilot),
          "prediction_mismatches": sum(r["prediction_mismatches"] for r in pilot),
          "max_decision_cpu_ms": max_cpu, "max_allowed_cpu_ms": budget + OVERRUN_MS,
          "peak_rss_mb": peak_mb, "memory_limit_mb": MEMORY_LIMIT_MB}
    g1["pass"] = bool(complete and g1["all_maps_valid_no_predicted_blocked"] and g1["prediction_mismatches"] == 0
                      and max_cpu is not None and max_cpu <= budget + OVERRUN_MS
                      and peak_mb is not None and peak_mb <= MEMORY_LIMIT_MB)
    up = est.get("upper") or {}
    g2 = {"upper_dev_comparison_cpu_s": up.get("comparison_cpu_s_total"), "limit_s": DEV_CPU_LIMIT_S}
    g2["pass"] = g2["upper_dev_comparison_cpu_s"] is not None and g2["upper_dev_comparison_cpu_s"] <= DEV_CPU_LIMIT_S
    g3 = {"upper_cap_fraction": up.get("predicted_cap_fraction"), "limit": BUDGET_HIT_LIMIT}
    g3["pass"] = g3["upper_cap_fraction"] is not None and g3["upper_cap_fraction"] <= BUDGET_HIT_LIMIT
    routable = sum(r["routable"] for r in pilot)
    fb = sum(r["fallback_people"] for r in pilot)
    lim = sum(r["limit_hit"] is not None for r in pilot)
    g4 = {"fallback_person_decisions": fb, "routable_person_decisions": routable,
          "fallback_fraction": fb / routable if routable else None, "fallback_limit": FALLBACK_LIMIT,
          "budget_limited_decisions": lim, "decisions": len(pilot),
          "budget_limited_fraction": lim / len(pilot) if pilot else None, "budget_limit": BUDGET_HIT_LIMIT}
    g4["pass"] = bool(pilot and routable and g4["fallback_fraction"] <= FALLBACK_LIMIT
                      and g4["budget_limited_fraction"] <= BUDGET_HIT_LIMIT)
    limiting = [name for name, g in (("G1_bounded", g1), ("G2_dev_cpu", g2), ("G3_cap_hits", g3),
                                     ("G4_aco_health", g4)) if not g["pass"]]
    return {"G1": g1, "G2": g2, "G3": g3, "G4": g4, "limiting": limiting,
            "decision": "FULL_DEV_FEASIBLE" if not limiting else "BLOCKED_BY_COMPUTE"}


def summarize(pilot: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for grid in (SMALL, LARGE):
        for n in SIZES:
            rs = [r for r in pilot if r["people_requested"] == n and tuple(r["grid"]) == grid]
            if not rs:
                continue
            wall = [r["t3_wall_ms"] for r in rs]
            sr = [r["ant_success_rate"] for r in rs if r["ant_success_rate"] is not None]
            stretch = [r["route_stretch_mean"] for r in rs if r["route_stretch_mean"] is not None]
            routable = sum(r["routable"] for r in rs)
            out[f"{grid[0]}x{grid[1]} N={n}"] = {
                "decisions": len(rs), "active_mean": statistics.fmean(r["active"] for r in rs),
                "wall_ms_p50": _pct(wall, 0.5), "wall_ms_p95": _pct(wall, 0.95), "wall_ms_p99": _pct(wall, 0.99),
                "wall_ms_max": max(wall), "cpu_ms_total": sum(r["t3_cpu_ms"] for r in rs),
                "ants_mean": statistics.fmean(r["ants"] for r in rs),
                "ant_success_rate_mean": statistics.fmean(sr) if sr else None,
                "route_stretch_mean": statistics.fmean(stretch) if stretch else None,
                "fallback_fraction": sum(r["fallback_people"] for r in rs) / routable if routable else None,
                "budget_limited": sum(r["limit_hit"] is not None for r in rs)}
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--wall-limit-s", type=float, default=WALL_LIMIT_S)
    args = ap.parse_args(argv)
    t_start = time.perf_counter()
    deadline = t_start + args.wall_limit_s
    cpu0 = time.process_time()
    t3 = T3DynamicACO()
    cfg = dict(t3.identity.config)
    micro = microbench(t3)
    print(json.dumps({"microbench": micro}), flush=True)
    pilot, episodes, pilot_complete = run_pilot(t3, deadline)
    pilot_s = time.perf_counter() - t_start
    if pilot_complete:
        profile, ref_s, profile_complete = dev_profile(deadline)
    else:
        profile, ref_s, profile_complete = [], {}, False
    est = estimate(cfg, pilot, profile, ref_s) if profile_complete else {"error": "dev profile incomplete"}
    peak = peak_rss_mb()
    wall = [r["t3_wall_ms"] for r in pilot]
    report = {
        "pilot_version": PILOT_VERSION, "teacher": t3.identity.to_dict(), "teacher_seed": t3.seed,
        "limits": {"wall_limit_s": args.wall_limit_s, "overrun_ms": OVERRUN_MS, "memory_limit_mb": MEMORY_LIMIT_MB,
                   "dev_cpu_limit_s": DEV_CPU_LIMIT_S, "budget_hit_limit": BUDGET_HIT_LIMIT,
                   "fallback_limit": FALLBACK_LIMIT, "episode_factor": EPISODE_FACTOR, "workers": 1},
        "microbench": micro,
        "pilot_latency_wall_ms": {"decisions": len(wall), "p50": _pct(wall, 0.5), "p95": _pct(wall, 0.95),
                                  "p99": _pct(wall, 0.99), "max": max(wall) if wall else None},
        "pilot_cpu_s_total": sum(r["t3_cpu_ms"] for r in pilot) / 1000,
        "pilot_summary_by_grid_and_n": summarize(pilot), "pilot_episodes": episodes, "pilot": pilot,
        "pilot_complete": pilot_complete, "dev_profile_complete": profile_complete,
        "dev_profile_summary": {"decisions": len(profile),
                                "active_mean": statistics.fmean(p["active"] for p in profile) if profile else None,
                                "routable_mean": statistics.fmean(p["routable"] for p in profile) if profile else None,
                                "stochastic_decisions": sum(p["fire_model"] == "stochastic" for p in profile)},
        "estimate": est,
        "gate": gate(cfg, pilot, pilot_complete and profile_complete, peak, est),
        "resources": {"wall_s": time.perf_counter() - t_start, "pilot_wall_s": pilot_s,
                      "process_cpu_s": time.process_time() - cpu0, "peak_rss_mb": peak},
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "processor": platform.processor(), **ev._revision()},
        "dev_use": "frozen T0/T1 replay for public workload and reference timing only; no T3 decision on DEV",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "gate": report["gate"], "estimate": est,
                      "resources": report["resources"]}, indent=1))


if __name__ == "__main__":
    main()
