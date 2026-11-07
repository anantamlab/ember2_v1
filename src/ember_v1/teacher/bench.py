"""Descriptive CPU decision latency: `python -m ember_v1.teacher.bench [--policy T0|T1]`.

For N = 1, 10, 50, 100, 200 active people on synthetic open rooms
(`sim.bench.room`, 32x32 and 64x64, two exits, one ignition), times
`policy.decide` (validation + planning + resolver diagnostics) on observations
from closed-loop episodes of the same policy. Several layouts and repetitions
per size. Prints one JSON line per (grid, N). Measured values only. Default
policy T0 (P1-03); T1 added for P1-04.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time

from ..sim.bench import room
from ..sim.simulator import Simulator
from .t0 import T0ShortestPath
from .t1 import T1CongestionAware

SIZES = (1, 10, 50, 100, 200)
SIDES = (32, 64)
POLICIES = {"T0": T0ShortestPath, "T1": T1CongestionAware}


def _observations(side: int, n: int, layout_seed: int, max_obs: int, policy: str = "T0") -> list[dict]:
    sim = Simulator(room(side, n, layout_seed))
    pol = POLICIES[policy]()
    out = []
    while not sim.terminated and len(out) < max_obs:
        obs = sim.observe()
        out.append(obs)
        sim.step(pol.act(obs))
    return out


def measure(side: int, n: int, layouts: int = 5, max_obs: int = 10, reps: int = 5, policy: str = "T0") -> dict:
    pol = POLICIES[policy]()
    samples: list[float] = []
    active: list[int] = []
    for layout_seed in range(1, layouts + 1):
        for obs in _observations(side, n, layout_seed, max_obs, policy):
            active.append(sum(p["status"] == "active" for p in obs["people"]))
            for _ in range(reps):
                w0 = time.perf_counter_ns()
                pol.decide(obs)
                samples.append((time.perf_counter_ns() - w0) / 1e6)
    s = sorted(samples)
    return {"policy": policy, "grid": f"{side}x{side}", "people": n, "samples": len(s), "observations": len(active),
            "mean_active": round(statistics.fmean(active), 1),
            "ms_median": round(statistics.median(s), 3), "ms_mean": round(statistics.fmean(s), 3),
            "ms_p95": round(s[min(len(s) - 1, int(0.95 * len(s)))], 3),
            "ms_p99": round(s[min(len(s) - 1, int(0.99 * len(s)))], 3), "ms_min": round(s[0], 3),
            "ms_max": round(s[-1], 3), "python": sys.version.split()[0], "platform": platform.platform()}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy", choices=sorted(POLICIES), default="T0")
    args = ap.parse_args(argv)
    for side in SIDES:
        for n in SIZES:
            print(json.dumps(measure(side, n, policy=args.policy)), flush=True)


if __name__ == "__main__":
    main()
