"""Fresh geometry-only TRAIN curriculum and independent DEV for counterfactual RL."""
from __future__ import annotations

import hashlib
import random

from .rl_env import safe_distances
from .sim import Scenario

STAGES = ("navigation", "congestion", "dynamic-fire", "competing-exits", "large-unseen")


def scenarios(namespace, split, count, *, stage=None, max_steps=48):
    if split not in {"train", "select", "report"} or stage not in (None, 0, 1, 2, 3, 4):
        raise ValueError("invalid counterfactual curriculum request")
    rows = []
    for index in range(count):
        level = index % 5 if stage is None else stage
        key = hashlib.sha256(f"{namespace}/{split}/{level}/{index}".encode()).hexdigest()
        rng = random.Random(int(key, 16))
        h, w = [rng.randint(8+2*level, 10+2*level) for _ in range(2)]
        cells = [(r, c) for r in range(h) for c in range(w)]
        walls = {p for p in cells if 0 < p[0] < h-1 and 0 < p[1] < w-1 and rng.random() < (.025 if level == 0 else .07)}
        if level in {1, 3, 4}:
            columns = [rng.randrange(2, w-2)] if level < 4 else list(range(3, w-2, 4))
            for col in columns:
                door = rng.randrange(1, h-1)
                walls.update((r, col) for r in range(h) if r not in {door, door+(level == 3)})
        interior = [p for p in cells if p not in walls and 0 < p[0] < h-1 and 0 < p[1] < w-1]
        walls.update(rng.sample(interior, min(len(interior), rng.randint(6, 9))))
        fr, fc, tr = rng.randrange(2), rng.randrange(2), rng.randrange(2)
        def transform(p):
            r, c = (h-1-p[0] if fr else p[0]), (w-1-p[1] if fc else p[1])
            return (c, r) if tr else (r, c)
        walls = {transform(p) for p in walls}
        if tr:
            h, w = w, h
        free = sorted({(r, c) for r in range(h) for c in range(w)}-walls)
        edge = [p for p in free if p[0] in {0, h-1} or p[1] in {0, w-1}]
        exits = rng.sample(edge, rng.randint(2, 4) if level >= 3 else rng.randint(1, 2))
        stub = {"map": {"height": h, "width": w, "obstacles": [list(p) for p in walls],
                        "exits": [{"position": list(p)} for p in exits]}, "burning": []}
        accessible = sorted(set(safe_distances(stub))-set(exits))
        lo, hi = ((4, 6), (8, 12), (8, 16), (12, 24), (24, 32))[level]
        if len(accessible) < hi+3:
            components, unseen = [], set(free)
            while unseen:
                seed = min(unseen)
                stub["map"]["exits"] = [{"position": list(seed)}]
                component = set(safe_distances(stub))
                unseen -= component
                boundary = sorted(component & set(edge))
                if boundary:
                    components.append((len(component), boundary[0]))
            exits[0] = max(components)[1]
            exits = list(dict.fromkeys(exits))
            stub["map"]["exits"] = [{"position": list(p)} for p in exits]
            accessible = sorted(set(safe_distances(stub))-set(exits))
        rng.shuffle(accessible)
        # Prespecified density coverage: each eight-episode block includes its
        # largest population, including in the small fixed DEV panels.
        n = min(hi if index % 8 == 0 else rng.randint(lo, hi), len(accessible)-3)
        if n < lo:
            raise RuntimeError("insufficient public curriculum capacity")
        people = accessible[:n]
        ignitions = rng.sample(sorted(set(free)-set(exits)-set(people)), rng.randint(1, 3) if level >= 2 else 1)
        fire = {"model": rng.choice(["deterministic", "stochastic"]), "exits_flammable": True,
                "spread_interval": rng.randint(8, 12) if level == 0 else rng.randint(3, 8)}
        if fire["model"] == "stochastic":
            fire["spread_probability"] = rng.choice([.2, .4, .6, .7])
        s = Scenario.from_dict({"schema_version": "1.0", "scenario_id": f"cf-{split}-{key[:24]}",
            "grid": {"height": h, "width": w}, "obstacles": [list(p) for p in sorted(walls)],
            "exits": [{"id": f"E{i}", "position": list(p)} for i, p in enumerate(exits)],
            "people": [{"id": f"P{i}", "position": list(p)} for i, p in enumerate(people)],
            "ignitions": [list(p) for p in ignitions], "fire": fire,
            "movement": {"conflict_resolution": "lowest_id_wins", "enter_fire": "blocked"},
            "termination": {"max_steps": max_steps}, "seed": int(key[:16], 16)})
        rows.append((s, {"stage": level, "family": STAGES[level], "index": index, "split": split}))
    return rows
