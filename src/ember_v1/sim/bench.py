"""CPU throughput of the simulator: `python -m ember_v1.sim.bench`.

Random-action policy on synthetic open rooms. Measures simulator steps only
(no policy cost beyond a seeded random choice). Prints one JSON line per size.
"""
from __future__ import annotations

import json
import random
import sys
import time

from .movement import ACTIONS
from .scenario import Scenario
from .simulator import Simulator


def room(side: int, n_people: int, layout_seed: int, fire_model: str = "stochastic") -> Scenario:
    rnd = random.Random(layout_seed)
    cells = [(r, c) for r in range(side) for c in range(side)]
    exits = [(0, side // 2), (side - 1, side // 2)]
    walls = rnd.sample([p for p in cells if p not in exits], side)
    free = [p for p in cells if p not in exits and p not in walls]
    chosen = rnd.sample(free, n_people + 1)
    fire = {"model": fire_model, "spread_interval": 2, "exits_flammable": True}
    if fire_model == "stochastic":
        fire["spread_probability"] = 0.3
    return Scenario.from_dict({
        "schema_version": "1.0", "scenario_id": f"bench-{side}-{n_people}",
        "grid": {"width": side, "height": side}, "obstacles": [list(p) for p in walls],
        "exits": [{"id": f"E{i}", "position": list(p)} for i, p in enumerate(exits)],
        "people": [{"id": f"P{i}", "position": list(p)} for i, p in enumerate(chosen[:-1])],
        "ignitions": [list(chosen[-1])], "fire": fire,
        "movement": {"conflict_resolution": "lowest_id_wins", "enter_fire": "casualty"},
        "termination": {"max_steps": 200}, "seed": layout_seed,
    })


def measure(side: int, n_people: int, steps: int = 100, layout_seed: int = 7) -> dict:
    sim = Simulator(room(side, n_people, layout_seed))
    rnd = random.Random(layout_seed)
    plans = []
    done = 0
    t0 = time.perf_counter()
    while done < steps and not sim.terminated:
        obs = sim.observe()
        actions = {p["id"]: rnd.choice(ACTIONS) for p in obs["people"] if p["status"] == "active"}
        plans.append(len(actions))
        sim.step(actions)
        done += 1
    elapsed = time.perf_counter() - t0
    return {"grid": f"{side}x{side}", "people": n_people, "steps": done,
            "mean_active": round(sum(plans) / max(done, 1), 1),
            "ms_per_step_incl_observe": round(1000 * elapsed / max(done, 1), 3),
            "python": sys.version.split()[0]}


def main() -> None:
    for side, n in ((12, 10), (24, 50), (32, 200), (64, 500)):
        print(json.dumps(measure(side, n)))


if __name__ == "__main__":
    main()
