"""Shared small plans and random scenarios for generator tests."""
from __future__ import annotations

import random

from ember_v1.gen.spec import GeneratorSpec
from ember_v1.gen.splits import DEFAULT_PLAN, SplitPlan
from ember_v1.sim import Scenario

SMALL_SPEC = GeneratorSpec.with_overrides({"grid": {"short_side": [8, 14], "aspect_pct": [100, 200]},
                                           "crowd": {"count": [1, 30]}})


def small_plan(seed: int = 7, plan_id: str = "test-small") -> SplitPlan:
    d = dict(DEFAULT_PLAN)
    d.update({"plan_id": plan_id, "master_seed": seed, "spec": SMALL_SPEC.to_dict(),
              "families": {"train": 6, "dev": 2, "calibration": 2, "locked_test": 3},
              "lineages_per_family": 2})
    return SplitPlan.from_dict(d)


def random_scenario(rnd: random.Random, *, h: int | None = None, w: int | None = None,
                    model: str | None = None, max_steps: int | None = None) -> Scenario:
    """Unstructured random scenario (independent of the generator) for analysis oracles."""
    h = h or rnd.randint(3, 9)
    w = w or rnd.randint(3, 9)
    cells = [(r, c) for r in range(h) for c in range(w)]
    rnd.shuffle(cells)
    n_walls = rnd.randint(0, len(cells) // 3)
    walls, rest = cells[:n_walls], cells[n_walls:]
    n_exits = rnd.randint(1, 3)
    exits, rest = rest[:n_exits], rest[n_exits:]
    flammable = rnd.random() < 0.5
    n_ign = rnd.randint(1, 2)
    ign_pool = rest + (exits if flammable else [])
    ign = rnd.sample(ign_pool, min(n_ign, len(ign_pool)))
    rest = [p for p in rest if p not in ign]
    if not rest:
        return random_scenario(rnd, h=h, w=w, model=model, max_steps=max_steps)
    people = rest[: rnd.randint(1, min(4, len(rest)))]
    model = model or rnd.choice(["deterministic", "stochastic"])
    fire = {"model": model, "spread_interval": rnd.randint(1, 3), "exits_flammable": flammable}
    if model == "stochastic":
        fire["spread_probability"] = rnd.choice([0.0, 0.2, 0.5, 0.8, 1.0])
    return Scenario.from_dict({
        "schema_version": "1.0", "scenario_id": "rnd", "grid": {"width": w, "height": h},
        "obstacles": [list(p) for p in walls], "exits": [{"id": f"E{i}", "position": list(p)} for i, p in enumerate(exits)],
        "people": [{"id": f"P{i}", "position": list(p)} for i, p in enumerate(people)],
        "ignitions": [list(p) for p in ign], "fire": fire,
        "movement": {"conflict_resolution": "lowest_id_wins", "enter_fire": rnd.choice(["blocked", "casualty"])},
        "termination": {"max_steps": max_steps or rnd.randint(3, 25)}, "seed": rnd.getrandbits(64),
    })
