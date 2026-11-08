"""Shared public-grid utilities for reference policies (T0-T4).

Pure functions of a validated `Observation.to_dict()`. No seed, RNG state,
future fire, trapped set, scenario ID or generator metadata is read here.
Version: GRID_UTILS_VERSION. Every decision records it (P1_TEACHER_DIRECTION §3).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

GRID_UTILS_VERSION = "grid-utils@1.0.0"

Pos = tuple[int, int]

# Fixed tie-break order for equally short first moves: the public action order
# without "stay" (north, south, east, west).
MOVES: tuple[tuple[str, Pos], ...] = (("north", (-1, 0)), ("south", (1, 0)), ("east", (0, 1)), ("west", (0, -1)))


@dataclass(frozen=True)
class PublicGrid:
    """Static map plus the current burning set and active people, from the observation only."""
    height: int
    width: int
    walls: frozenset[Pos]
    burning: frozenset[Pos]
    exits: tuple[tuple[str, Pos], ...]      # sorted by exit ID
    active: tuple[tuple[str, Pos], ...]     # sorted by person ID
    enter_fire: str

    @classmethod
    def from_observation(cls, obs: Mapping[str, Any]) -> "PublicGrid":
        """`obs` must already be validated by `contract.validate_observation`."""
        m = obs["map"]
        return cls(
            height=m["height"], width=m["width"],
            walls=frozenset(tuple(p) for p in m["obstacles"]),
            burning=frozenset(tuple(p) for p in obs["burning"]),
            exits=tuple(sorted((x["id"], tuple(x["position"])) for x in m["exits"])),
            active=tuple(sorted((p["id"], tuple(p["position"])) for p in obs["people"] if p["status"] == "active")),
            enter_fire=obs["rules"]["enter_fire"],
        )

    def in_grid(self, p: Pos) -> bool:
        return 0 <= p[0] < self.height and 0 <= p[1] < self.width

    def passable(self, p: Pos) -> bool:
        """In the grid, not a wall, not burning now."""
        return self.in_grid(p) and p not in self.walls and p not in self.burning

    def safe_exits(self) -> tuple[tuple[str, Pos], ...]:
        """Exits that are not burning now (an agent evacuates only on such an exit)."""
        return tuple((i, p) for i, p in self.exits if p not in self.burning)


def exit_distance_field(grid: PublicGrid) -> dict[Pos, tuple[int, str]]:
    """Multi-source BFS from every safe exit over passable cells (4-neighbour).

    Returns `{cell: (distance, exit_id)}` for each cell with a path to a safe exit.
    `exit_id` is the lowest exit ID among the exits at that shortest distance.
    The field is built layer by layer: a cell at distance d+1 takes the lowest
    label of its neighbours at distance d, which equals the lowest ID among all
    nearest exits. A cell not in the field has no safe exit path now.
    """
    field: dict[Pos, tuple[int, str]] = {}
    layer: list[Pos] = []
    for eid, p in grid.safe_exits():
        if grid.passable(p):
            field[p] = (0, eid)
            layer.append(p)
    h, w, walls, burning = grid.height, grid.width, grid.walls, grid.burning
    d = 0
    while layer:
        nxt: dict[Pos, str] = {}
        for p in layer:
            label = field[p][1]
            r, c = p
            for n in ((r - 1, c), (r + 1, c), (r, c + 1), (r, c - 1)):   # same test as grid.passable
                if n in field or not (0 <= n[0] < h and 0 <= n[1] < w) or n in walls or n in burning:
                    continue
                if n not in nxt or label < nxt[n]:
                    nxt[n] = label
        d += 1
        for n, label in nxt.items():
            field[n] = (d, label)
        layer = list(nxt)
    return field
