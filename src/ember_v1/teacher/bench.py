"""Descriptive CPU decision latency of T0: `python -m ember_v1.teacher.bench`.

For N = 1, 10, 50, 100, 200 active people on synthetic open rooms
(`sim.bench.room`, 32x32 and 64x64, two exits, one ignition), times
`T0ShortestPath.decide` (validation + BFS + resolver diagnostics) on
observations from closed-loop T0 episodes. Several layouts and repetitions per
size. Prints one JSON line per (grid, N). Measured values only.
"""
from __future__ import annotations

import json
import platform
import statistics
import sys
import time

from ..sim.bench import room
from ..sim.simulator import Simulator
from .t0 import T0ShortestPath

SIZES = (1, 10, 50, 100, 200)
SIDES = (32, 64)


def _observations(side: int, n: int, layout_seed: int, max_obs: int) -> list[dict]:
    sim = Simulator(room(side, n, layout_seed))
    t0 = T0ShortestPath()
    out = []
    while not sim.terminated and len(out) < max_obs:
        obs = sim.observe()
        out.append(obs)
        sim.step(t0.act(obs))
    return out


def measure(side: int, n: int, layouts: int = 5, max_obs: int = 10, reps: int = 5) -> dict:
    t0 = T0ShortestPath()
    samples: list[float] = []
    active: list[int] = []
    for layout_seed in range(1, layouts + 1):
        for obs in _observations(side, n, layout_seed, max_obs):
            active.append(sum(p["status"] == "active" for p in obs["people"]))
            for _ in range(reps):
                w0 = time.perf_counter_ns()
                t0.decide(obs)
                samples.append((time.perf_counter_ns() - w0) / 1e6)
    s = sorted(samples)
    return {"grid": f"{side}x{side}", "people": n, "samples": len(s), "observations": len(active),
            "mean_active": round(statistics.fmean(active), 1),
            "ms_median": round(statistics.median(s), 3), "ms_mean": round(statistics.fmean(s), 3),
            "ms_p95": round(s[min(len(s) - 1, int(0.95 * len(s)))], 3), "ms_min": round(s[0], 3),
            "ms_max": round(s[-1], 3), "python": sys.version.split()[0], "platform": platform.platform()}


def main() -> None:
    for side in SIDES:
        for n in SIZES:
            print(json.dumps(measure(side, n)), flush=True)


if __name__ == "__main__":
    main()
