"""ASCII builders for simulator tests.

Map characters: `.` floor, `#` wall, `F` burning floor (ignition), `E` exit,
`B` exit that burns at step 0. Exit IDs are E1, E2, ... in row-major order.
People are passed separately as `{id: (row, col)}` so IDs keep their real text.
"""
from __future__ import annotations

from ember_v1.sim import Scenario, resolve_moves


def parse(rows: list[str]) -> dict:
    walls, fire, exits = [], [], []
    for r, line in enumerate(rows):
        for c, ch in enumerate(line):
            if ch == "#":
                walls.append([r, c])
            if ch in "FB":
                fire.append([r, c])
            if ch in "EB":
                exits.append([r, c])
            if ch not in ".#FEB":
                raise ValueError(f"bad map char {ch!r}")
    return {"height": len(rows), "width": len(rows[0]), "walls": walls, "fire": fire,
            "exits": [{"id": f"E{i + 1}", "position": p} for i, p in enumerate(exits)]}


def scenario(rows: list[str], people: dict[str, tuple[int, int]], *, fire: str = "deterministic",
             interval: int = 1, p: float | None = None, exits_flammable: bool = True,
             enter_fire: str = "casualty", max_steps: int = 50, seed: int = 1,
             sid: str = "test", description: str | None = None) -> Scenario:
    m = parse(rows)
    fire_cfg = {"model": fire, "spread_interval": interval, "exits_flammable": exits_flammable}
    if fire == "stochastic":
        fire_cfg["spread_probability"] = p
    data = {
        "schema_version": "1.0", "scenario_id": sid,
        "grid": {"width": m["width"], "height": m["height"]}, "obstacles": m["walls"],
        "exits": m["exits"], "people": [{"id": k, "position": list(v)} for k, v in people.items()],
        "ignitions": m["fire"], "fire": fire_cfg,
        "movement": {"conflict_resolution": "lowest_id_wins", "enter_fire": enter_fire},
        "termination": {"max_steps": max_steps}, "seed": seed,
    }
    if description is not None:
        data["description"] = description
    return Scenario.from_dict(data)


def resolve(rows: list[str], people: dict[str, tuple[int, int]], actions: dict[str, str],
            enter_fire: str = "blocked") -> tuple[dict[str, str], dict[str, tuple[int, int]]]:
    m = parse(rows)
    res = resolve_moves(height=m["height"], width=m["width"],
                        walls=frozenset(map(tuple, m["walls"])), burning=frozenset(map(tuple, m["fire"])),
                        enter_fire=enter_fire, positions=people, actions=actions)
    return {k: v[0] for k, v in res.items()}, {k: v[1] for k, v in res.items()}
