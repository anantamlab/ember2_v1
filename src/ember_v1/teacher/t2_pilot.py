"""Bounded CPU feasibility pilot for T2 (issue #9) and the full-DEV cost gate.

    python -m ember_v1.teacher.t2_pilot [--out PATH] [--wall-limit-s S]

Preregistered in docs/P1_T2_ROLLOUT.md section 6 before it runs. One CPU
process, no multiprocessing, no GPU, no network.

1 Microbenchmark of T2's parts on one synthetic room (sampler, resolver, one
  T1 continuation call, one rollout).
2 Pilot decisions on SYNTHETIC scenarios only: four layout archetypes
  (`gen.layouts` builders with pilot-only seeds) x N = 1, 10, 50, 100, 200,
  stochastic fire, plus deterministic cells. Observations come from a short
  frozen-T1 prefix. No DEV, train, calibration, test, blind or EMBERBench
  scenario is used for a T2 decision.
3 DEV workload profile: the frozen T1 is replayed on the 48 `p1-02-v1` DEV
  scenarios (as in P1-04) to record, per decision, the active count, the
  remaining steps, the fire model and T1's own decision time. T2 is NOT run on
  DEV here and no T2 output on DEV exists.
4 Cost model per DEV decision d:
      est_d = c1_d * (1 + alpha * nominal_d),
      nominal_d = K_eff * (1 + 4 * min(N_d, max_scored_people)) * min(H, max_steps - t_d)
  alpha (cost of one nominal rollout step relative to one T1 decision) is
  measured on the pilot decisions that did not hit a budget. Central: median
  alpha; upper: 90th percentile alpha and an episode-length factor 1.25.
  est_d is capped at cpu_budget_ms + c1_d (the budget stops a decision).
5 Gate (all must hold, else BLOCKED_BY_COMPUTE):
  G1 bounded: every planned pilot decision ran inside the wall limit, every
     action map is complete and has no predicted blocked move, max decision
     CPU <= cpu_budget_ms + OVERRUN_MS, peak RSS <= MEMORY_LIMIT_MB;
  G2 upper full-DEV T2 CPU <= DEV_CPU_LIMIT_S on one worker;
  G3 predicted DEV decisions that hit cpu_budget_ms (upper alpha) <= 5 %.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from ..contract import validate_actions, validate_observation
from ..gen.layouts import BUILDERS, components
from ..gen.rng import GenRng, derive_seed
from ..sim.bench import room
from ..sim.scenario import Scenario
from ..sim.simulator import Simulator
from . import evaluate as ev
from .t1 import T1CongestionAware
from .t2 import T2StochasticRollout, World, _Decision, public_fire_digest, sample_futures

PILOT_VERSION = "t2-cpu-pilot@1.0.0"
DEFAULT_OUT = ev.REPO / "data/baselines/p1-05-t2-pilot.json"
WALL_LIMIT_S = 2700.0                  # whole pilot (microbench + pilot + DEV profile)
OVERRUN_MS = 2000.0                    # allowed CPU past cpu_budget_ms (one T1 proposal + one rollout step)
MEMORY_LIMIT_MB = 1024.0
DEV_CPU_LIMIT_S = 4 * 3600.0           # upper full-DEV T2 estimate, one worker
BUDGET_HIT_LIMIT = 0.05
EPISODE_FACTOR = 1.25
TOPOLOGIES = ("open_room", "corridor", "multi_room", "irregular")
SIZES = (1, 10, 50, 100, 200)
GRID = (32, 40)
PREFIX_STEPS = (0, 4)
DETERMINISTIC_CELLS = (("open_room", 10), ("open_room", 100), ("multi_room", 10), ("multi_room", 100))
LAYOUT_PARAMS = {"corridor_width": 2, "door_width": 2, "min_room_side": 5, "extra_door_pct": 35,
                 "sealed_room_pct": 0, "cave_fill_pct": 40, "cave_smoothing": 4}


def peak_rss_mb() -> float | None:
    """Peak resident memory of this process (Windows: PeakWorkingSetSize; POSIX: ru_maxrss)."""
    try:
        if sys.platform == "win32":
            class PMC(ctypes.Structure):
                _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            k32 = ctypes.windll.kernel32
            psapi = ctypes.windll.psapi
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            if not psapi.GetProcessMemoryInfo(ctypes.c_void_p(k32.GetCurrentProcess()), ctypes.byref(pmc), pmc.cb):
                return None
            return pmc.PeakWorkingSetSize / 2 ** 20
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 2 ** 20 if sys.platform == "darwin" else r / 1024
    except Exception:  # measurement only; never affects a decision
        return None


def pilot_scenario(topology: str, n: int, model: str, h: int = GRID[0], w: int = GRID[1]) -> Scenario:
    """Synthetic pilot scenario. Pilot-only seed path; not a generator plan, family or split."""
    rng = GenRng(derive_seed("t2-pilot@1", topology, h, w, n, model))
    walls = BUILDERS[topology](h, w, rng, dict(LAYOUT_PARAMS))
    main = max(components(h, w, walls), key=len)
    border = sorted(p for p in main if p[0] in (0, h - 1) or p[1] in (0, w - 1)) or sorted(main)
    exits = rng.sample(border, min(3, len(border)))
    rest = sorted(set(main) - set(exits))
    ign = rng.choice(rest)
    rest = [p for p in rest if p != ign]
    people = rng.sample(rest, min(n, len(rest) - 1))
    fire = {"model": model, "spread_interval": 2, "exits_flammable": False}
    if model == "stochastic":
        fire["spread_probability"] = 0.5
    return Scenario.from_dict({
        "schema_version": "1.0", "scenario_id": f"t2-pilot-{topology}-{n}-{model}",
        "grid": {"width": w, "height": h}, "obstacles": [list(p) for p in sorted(walls)],
        "exits": [{"id": f"E{i}", "position": list(p)} for i, p in enumerate(exits)],
        "people": [{"id": f"P{i}", "position": list(p)} for i, p in enumerate(people)],
        "ignitions": [list(ign)], "fire": fire,
        "movement": {"conflict_resolution": "lowest_id_wins", "enter_fire": "blocked"},
        "termination": {"max_steps": 2 * (h + w)}, "seed": derive_seed("t2-pilot@1/sim", topology, n, model)})


def _t1_wall_ms(t1: T1CongestionAware, obs: dict, reps: int = 2) -> float:
    best = float("inf")
    for _ in range(reps):
        w0 = time.perf_counter_ns()
        t1.decide(obs)
        best = min(best, (time.perf_counter_ns() - w0) / 1e6)
    return best


def nominal_steps(cfg: dict, model: str, n_active: int, step: int, max_steps: int) -> int:
    k = 1 if model == "deterministic" else cfg["samples_stochastic"]
    return k * (1 + 4 * min(n_active, cfg["max_scored_people"])) * min(cfg["horizon"], max_steps - step)


def microbench(t2: T2StochasticRollout) -> dict[str, Any]:
    sim = Simulator(room(32, 50, 1, "stochastic"))
    t1 = T1CongestionAware()
    for _ in range(3):
        sim.step(t1.act(sim.observe()))
    obs = validate_observation(sim.observe())
    cfg = t2.identity.config
    out: dict[str, Any] = {"scene": "sim.bench.room 32x32 N=50 stochastic, step 3"}

    def timed(fn, reps):
        best = float("inf")
        for _ in range(reps):
            w0 = time.perf_counter_ns()
            fn()
            best = min(best, (time.perf_counter_ns() - w0) / 1e6)
        return round(best, 3)

    world = World(obs)
    fd = public_fire_digest(obs)
    out["sample_futures_ms"] = timed(lambda: sample_futures(world, cfg["horizon"], cfg["samples_stochastic"],
                                                            t2.seed, fd), 5)
    dec = _Decision(t2, obs)
    base = t2.t1_actions(obs)
    out["resolve_ms"] = timed(lambda: dec.resolve(dec.active, base, world.burning0), 20)
    out["t1_continuation_call_ms"] = timed(lambda: t2.t1_actions(obs), 5)
    dec.futures, _ = sample_futures(world, cfg["horizon"], cfg["samples_stochastic"], t2.seed, fd)
    dec.rep, dec._exit_dist = list(range(len(dec.futures))), [{} for _ in dec.futures]
    out["one_rollout_ms"] = timed(lambda: (dec.t1_cache.clear(), dec.rollout(base, 0)), 3)
    return out


def pilot_cells() -> list[tuple[str, int, str]]:
    cells = [(t, n, "stochastic") for t in TOPOLOGIES for n in SIZES]
    return cells + [(t, n, "deterministic") for t, n in DETERMINISTIC_CELLS]


def run_pilot(t2: T2StochasticRollout, deadline: float) -> tuple[list[dict[str, Any]], bool]:
    cfg = t2.identity.config
    t1 = T1CongestionAware()
    rows: list[dict[str, Any]] = []
    for topology, n, model in pilot_cells():
        s = pilot_scenario(topology, n, model)
        sim = Simulator(s)
        for target in PREFIX_STEPS:
            while sim.observe()["step"] < target and not sim.terminated:
                sim.step(t1.act(sim.observe()))
            if sim.terminated:
                break
            if time.perf_counter() > deadline:
                return rows, False
            obs = sim.observe()
            n_active = sum(p["status"] == "active" for p in obs["people"])
            c1 = _t1_wall_ms(t1, obs)
            d = t2.decide(obs)
            dg = d.diagnostics
            validate_actions(validate_observation(obs), d.actions)
            blocked = sorted(o for o in dg["predicted_outcomes"] if o.startswith("blocked_"))
            nom = nominal_steps(cfg, model, n_active, obs["step"], obs["rules"]["max_steps"])
            scored = len(dg["scored"])
            cpu_s = d.runtime["cpu_ms"] / 1000
            rows.append({
                "topology": topology, "people_requested": n, "active": n_active, "fire_model": model,
                "step": obs["step"], "grid": [s.height, s.width], "walls": len(s.obstacles),
                "status": dg["status"], "limit_hit": dg["budget"]["limit_hit"], "scored": scored,
                "deviations": len(dg["deviations"]), "samples": dg["samples"],
                "distinct_futures": dg["distinct_futures"], "rollouts": dg["budget"]["rollouts"],
                "rollout_steps": dg["budget"]["rollout_steps"],
                "t1_continuation_calls": dg["budget"]["t1_continuation_calls"],
                "t1_cache_hits": dg["budget"]["t1_cache_hits"], "nominal_rollout_steps": nom,
                "predicted_blocked_outcomes": blocked,
                "t2_cpu_ms": d.runtime["cpu_ms"], "t2_wall_ms": d.runtime["wall_ms"], "t1_wall_ms": c1,
                "cpu_ms_per_evaluated_action": d.runtime["cpu_ms"] / (5 * scored) if scored else None,
                "rollouts_per_cpu_s": dg["budget"]["rollouts"] / cpu_s if cpu_s > 0 else None,
                "alpha": ((d.runtime["wall_ms"] / c1 - 1) / nom) if nom and c1 > 0 else None,
            })
            print(json.dumps({k: rows[-1][k] for k in ("topology", "active", "fire_model", "step", "status",
                                                        "scored", "rollouts", "t2_cpu_ms", "t1_wall_ms")}),
                  flush=True)
    return rows, True


def dev_profile(deadline: float) -> tuple[list[dict[str, Any]], bool]:
    """Frozen-T1 DEV workload (decision count, active N, remaining steps, T1 time). T2 is not run."""
    from .t1_eval import load_dev                      # DEV-only gate of the P1-04 harness
    _, items = load_dev()
    t1 = T1CongestionAware()
    out = []
    for it in items:
        s: Scenario = it["scenario"]
        sim = Simulator(s)
        while not sim.terminated:
            if time.perf_counter() > deadline:
                return out, False
            obs = sim.observe()
            c1 = _t1_wall_ms(t1, obs)
            out.append({"position": it["position"], "fire_model": s.fire_model, "step": obs["step"],
                        "max_steps": s.max_steps, "area": s.height * s.width,
                        "active": sum(p["status"] == "active" for p in obs["people"]), "t1_wall_ms": c1})
            sim.step(t1.act(obs))
    return out, True


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def estimate(cfg: dict, pilot: list[dict[str, Any]], profile: list[dict[str, Any]]) -> dict[str, Any]:
    budget = float(cfg["cpu_budget_ms"])
    alphas: dict[str, dict[str, float | None]] = {}
    pooled = [r["alpha"] for r in pilot if r["alpha"] is not None and r["limit_hit"] is None]
    for model in ("stochastic", "deterministic"):
        xs = [r["alpha"] for r in pilot if r["fire_model"] == model and r["alpha"] is not None
              and r["limit_hit"] is None]
        xs = xs if len(xs) >= 3 else pooled
        alphas[model] = {"n": len(xs), "median": _pct(xs, 0.5), "p90": _pct(xs, 0.9)}
    out: dict[str, Any] = {"alpha": alphas, "dev_decisions": len(profile),
                           "dev_t1_wall_s_total": sum(p["t1_wall_ms"] for p in profile) / 1000}
    for name, key, factor in (("central", "median", 1.0), ("upper", "p90", EPISODE_FACTOR)):
        total_ms, hits = 0.0, 0
        per_pos: dict[int, float] = {}
        for p in profile:
            a = alphas[p["fire_model"]][key]
            if a is None:
                return {**out, "error": "no unbudgeted pilot decisions to fit alpha"}
            nom = nominal_steps(cfg, p["fire_model"], p["active"], p["step"], p["max_steps"])
            est = p["t1_wall_ms"] * (1 + a * nom)
            if est > budget:
                hits += 1
                est = budget + p["t1_wall_ms"]
            total_ms += est
            per_pos[p["position"]] = per_pos.get(p["position"], 0.0) + est
        out[name] = {"t2_cpu_s_total": total_ms * factor / 1000, "t2_cpu_h_total": total_ms * factor / 3.6e6,
                     "episode_factor": factor, "predicted_budget_hit_decisions": hits,
                     "predicted_budget_hit_fraction": hits / len(profile) if profile else None,
                     "max_episode_cpu_s": max(per_pos.values()) * factor / 1000 if per_pos else None}
    return out


def gate(cfg: dict, pilot: list[dict[str, Any]], complete: bool, peak_mb: float | None,
         est: dict[str, Any]) -> dict[str, Any]:
    budget = float(cfg["cpu_budget_ms"])
    max_cpu = max((r["t2_cpu_ms"] for r in pilot), default=None)
    g1 = {"pilot_complete_inside_wall_limit": complete,
          "all_maps_valid_no_predicted_blocked": all(not r["predicted_blocked_outcomes"] for r in pilot),
          "max_decision_cpu_ms": max_cpu, "max_allowed_cpu_ms": budget + OVERRUN_MS,
          "peak_rss_mb": peak_mb, "memory_limit_mb": MEMORY_LIMIT_MB}
    g1["pass"] = bool(complete and g1["all_maps_valid_no_predicted_blocked"] and max_cpu is not None
                      and max_cpu <= budget + OVERRUN_MS and peak_mb is not None and peak_mb <= MEMORY_LIMIT_MB)
    up = est.get("upper") or {}
    g2 = {"upper_dev_cpu_s": up.get("t2_cpu_s_total"), "limit_s": DEV_CPU_LIMIT_S}
    g2["pass"] = g2["upper_dev_cpu_s"] is not None and g2["upper_dev_cpu_s"] <= DEV_CPU_LIMIT_S
    g3 = {"upper_budget_hit_fraction": up.get("predicted_budget_hit_fraction"), "limit": BUDGET_HIT_LIMIT}
    g3["pass"] = g3["upper_budget_hit_fraction"] is not None and g3["upper_budget_hit_fraction"] <= BUDGET_HIT_LIMIT
    limiting = [name for name, g in (("G1_bounded", g1), ("G2_dev_cpu", g2), ("G3_budget_hits", g3))
                if not g["pass"]]
    return {"G1": g1, "G2": g2, "G3": g3, "limiting": limiting,
            "decision": "FULL_DEV_FEASIBLE" if not limiting else "BLOCKED_BY_COMPUTE"}


def summarize(pilot: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for n in SIZES:
        rs = [r for r in pilot if r["people_requested"] == n and r["fire_model"] == "stochastic"]
        if not rs:
            continue
        cpu = [r["t2_cpu_ms"] for r in rs]
        per_act = [r["cpu_ms_per_evaluated_action"] for r in rs if r["cpu_ms_per_evaluated_action"] is not None]
        rps = [r["rollouts_per_cpu_s"] for r in rs if r["rollouts_per_cpu_s"] is not None]
        out[f"N={n}"] = {"decisions": len(rs), "active_mean": statistics.fmean(r["active"] for r in rs),
                         "t2_cpu_ms_median": statistics.median(cpu), "t2_cpu_ms_max": max(cpu),
                         "t1_wall_ms_median": statistics.median(r["t1_wall_ms"] for r in rs),
                         "cpu_ms_per_evaluated_action_median": statistics.median(per_act) if per_act else None,
                         "rollouts_per_cpu_s_median": statistics.median(rps) if rps else None,
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
    t2 = T2StochasticRollout()
    cfg = dict(t2.identity.config)
    micro = microbench(t2)
    print(json.dumps({"microbench": micro}), flush=True)
    pilot, pilot_complete = run_pilot(t2, deadline)
    profile, profile_complete = dev_profile(deadline) if pilot_complete else ([], False)
    est = estimate(cfg, pilot, profile) if profile_complete else {"error": "dev profile incomplete"}
    peak = peak_rss_mb()
    report = {
        "pilot_version": PILOT_VERSION, "teacher": t2.identity.to_dict(), "teacher_seed": t2.seed,
        "continuation": t2.continuation,
        "limits": {"wall_limit_s": args.wall_limit_s, "overrun_ms": OVERRUN_MS, "memory_limit_mb": MEMORY_LIMIT_MB,
                   "dev_cpu_limit_s": DEV_CPU_LIMIT_S, "budget_hit_limit": BUDGET_HIT_LIMIT,
                   "episode_factor": EPISODE_FACTOR, "workers": 1},
        "microbench": micro, "pilot_summary_stochastic_by_n": summarize(pilot), "pilot": pilot,
        "pilot_complete": pilot_complete, "dev_profile_complete": profile_complete,
        "dev_profile_summary": {"decisions": len(profile),
                                "active_mean": statistics.fmean(p["active"] for p in profile) if profile else None,
                                "stochastic_decisions": sum(p["fire_model"] == "stochastic" for p in profile)},
        "estimate": est,
        "gate": gate(cfg, pilot, pilot_complete and profile_complete, peak, est),
        "resources": {"wall_s": time.perf_counter() - t_start, "process_cpu_s": time.process_time() - cpu0,
                      "peak_rss_mb": peak},
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "processor": platform.processor(), **ev._revision()},
        "dev_use": "frozen T1 replay for workload and T1 timing only; no T2 decision on DEV",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "gate": report["gate"], "estimate": est,
                      "resources": report["resources"]}, indent=1))


if __name__ == "__main__":
    main()
