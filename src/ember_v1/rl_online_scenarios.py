"""Fresh geometry-only online scenarios. No dataset, teacher or benchmark inputs."""
from __future__ import annotations

import hashlib
import random

from .rl_env import safe_distances
from .sim import Scenario

STRATA = ("open", "irregular", "door", "two_doors", "corridor", "pillars", "rooms", "inaccessible")


def build_scenarios(namespace, split, count, max_steps):
    if split not in {"train", "select", "report"} or count < 1:
        raise ValueError("invalid online split/count")
    rows = []
    for index in range(count):
        key = hashlib.sha256(f"{namespace}/{split}/{index}".encode()).hexdigest()
        rng = random.Random(int(key, 16))
        stratum = STRATA[index % len(STRATA)]
        h, w = rng.randint(8, 16), rng.randint(8, 16)
        cells = [(r, c) for r in range(h) for c in range(w)]
        walls = set()
        if stratum in {"irregular", "open"}:
            walls = {p for p in cells if rng.random() < (0.12 if stratum == "irregular" else 0.025)}
        if stratum in {"door", "two_doors", "rooms", "inaccessible"}:
            col = rng.randrange(2, w-2)
            doors = rng.sample(range(1, h-1), 2 if stratum == "two_doors" else 1)
            walls.update((r, col) for r in range(h) if r not in doors)
            if stratum == "rooms":
                row, door = rng.randrange(2, h-2), rng.randrange(1, w-1)
                walls.update((row, c) for c in range(w) if c != door)
            if stratum == "inaccessible":
                # A sealed side region; accessibility differs without consulting outcomes.
                walls.update((r, col) for r in range(h))
        if stratum == "corridor":
            row = rng.randrange(2, h-3)
            doors = rng.sample(range(1, w-1), 2)
            walls.update((r, c) for r in (row, row+2) for c in range(w) if c not in doors)
        if stratum == "pillars":
            walls.update((r, c) for r in range(1, h-1, 3) for c in range(1, w-1, 3))
        # Random transpose/reflections remove a fixed doorway/exit orientation.
        flip_r, flip_c, transpose = rng.randrange(2), rng.randrange(2), rng.randrange(2)
        def transform(p):
            r, c = p
            r, c = (h-1-r if flip_r else r), (w-1-c if flip_c else c)
            return (c, r) if transpose else (r, c)
        walls = {transform(p) for p in walls}
        if transpose:
            h, w = w, h
        cells = [(r, c) for r in range(h) for c in range(w)]
        free = sorted(set(cells)-walls)
        boundary = [p for p in free if p[0] in {0, h-1} or p[1] in {0, w-1}]
        exits = rng.sample(boundary, rng.randint(1, 4))
        stub = {"map": {"height": h, "width": w, "obstacles": [list(p) for p in walls],
                        "exits": [{"position": list(p)} for p in exits]}, "burning": []}
        reachable = sorted(set(safe_distances(stub))-set(exits))
        if len(reachable) < 7:
            # A random exit can land in a tiny isolated corner. Ensure capacity using
            # only connected components of public geometry, never simulator outcomes.
            unseen, components = set(free), []
            while unseen:
                seed = min(unseen)
                stub["map"]["exits"] = [{"position": list(seed)}]
                component = set(safe_distances(stub))
                unseen -= component
                edge = sorted(component & set(boundary))
                if edge:
                    components.append((len(component), edge[0]))
            exits[0] = max(components)[1]
            exits = list(dict.fromkeys(exits))
            stub["map"]["exits"] = [{"position": list(p)} for p in exits]
            reachable = sorted(set(safe_distances(stub))-set(exits))
        candidates = sorted(set(free)-set(exits)) if stratum == "inaccessible" else reachable
        rng.shuffle(candidates)
        # Alternating density strata, independent of fire/geometry outcomes.
        n = min(rng.randint(4, 8) if (index//8) % 2 == 0 else rng.randint(12, 20), len(candidates)-3)
        people = candidates[:n]
        ignitions = rng.sample(sorted(set(free)-set(exits)-set(people)), rng.randint(1, 3))
        model = rng.choice(["deterministic", "stochastic"])
        fire = {"model": model, "spread_interval": rng.randint(3, 8), "exits_flammable": True}
        if model == "stochastic":
            fire["spread_probability"] = rng.choice([0.15, 0.3, 0.5, 0.7])
        scenario = Scenario.from_dict({"schema_version": "1.0", "scenario_id": f"online-{split}-{key[:24]}",
            "grid": {"height": h, "width": w}, "obstacles": [list(p) for p in sorted(walls)],
            "exits": [{"id": f"E{i}", "position": list(p)} for i, p in enumerate(exits)],
            "people": [{"id": f"P{i}", "position": list(p)} for i, p in enumerate(people)],
            "ignitions": [list(p) for p in ignitions], "fire": fire,
            "movement": {"conflict_resolution": "lowest_id_wins", "enter_fire": "blocked"},
            "termination": {"max_steps": max_steps}, "seed": int(key[:16], 16)})
        rows.append((scenario, {"split": split, "index": index, "stratum": stratum,
                               "initial_reachable": sum(p in reachable for p in people)}))
    return rows


def select_checkpoint(rows, baseline):
    """DEV selection only; reporting DEV is deliberately absent from this interface."""
    def metrics(row):
        s = row["selection"]
        return (s["evacuated"]-2*s["casualties"]-.75*s["unresolved"])/s["people"]
    eligible = [r for r in rows if r["selection"]["casualties"]/r["selection"]["people"]
                <= baseline["casualties"]/baseline["people"] + .02]
    best = max(eligible or rows, key=lambda r: (metrics(r), -r["update"]))
    return {**best, "safety_eligible": bool(eligible), "selection_utility": metrics(best)}
