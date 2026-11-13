"""Human point-of-view observations.

One view per active person. A view holds only what that person could know:
cells inside a limited sight radius with line of sight, a memory of cells seen
earlier, the person's own last action and its result, and a route estimate planned
on that memory alone. No absolute coordinates, no map size, no hidden simulator
state and no other person's identity enter a view.

The public observation stays the single source of truth; this module only removes
information from it. It imports nothing from the benchmark.
"""
from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from .contract import ACTIONS, validate_observation
from .sim.movement import DELTAS

VIEW_VERSION = "1.0"
MOVES = ("north", "south", "east", "west")
WALL, FLOOR, EXIT, FIRE, PERSON, UNSEEN, SELF = "#", ".", "E", "F", "P", "?", "@"
KINDS = {WALL: "wall", FLOOR: "floor", EXIT: "exit", FIRE: "fire", PERSON: "person"}
GOALS = ("exit", "explore", "flee", "none")
SOCIAL_BLOCKS = ("blocked_occupied", "blocked_conflict", "blocked_cycle")
_FLEE_CAP = 99
Pos = tuple[int, int]


@dataclass(frozen=True)
class ViewConfig:
    """What a person can perceive and remember.

    radius: sight distance in cells (Chebyshev). line_of_sight: walls hide cells behind them.
    memory: cells seen earlier stay known. exit_knowledge: "seen" = only exits the person saw;
    "one" = the person also knows one exit position from the start (the door they came in by);
    "all" = every exit position is known (a familiar building). hints: put the person's own
    route estimate in the view. fire_detour: extra steps the person accepts to stay away from
    cells next to fire.
    """
    radius: int = 4
    line_of_sight: bool = True
    memory: bool = True
    exit_knowledge: str = "seen"
    hints: bool = True
    fire_detour: int = 6

    def __post_init__(self) -> None:
        if type(self.radius) is not int or not 1 <= self.radius <= 15:
            raise ValueError("radius must be an integer in [1, 15]")
        if self.exit_knowledge not in ("seen", "one", "all"):
            raise ValueError("exit_knowledge must be seen, one or all")
        if type(self.fire_detour) is not int or self.fire_detour < 0:
            raise ValueError("fire_detour must be a nonnegative integer")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _between(a: Pos, b: Pos) -> list[Pos]:
    """Bresenham cells strictly between a and b (deterministic, integer only)."""
    (r0, c0), (r1, c1) = a, b
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr, sc = (1 if r1 > r0 else -1), (1 if c1 > c0 else -1)
    err, r, c, cells = dr - dc, r0, c0, []
    while (r, c) != (r1, c1):
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
        if (r, c) != (r1, c1):
            cells.append((r, c))
    return cells


def visible_cells(origin: Pos, opaque, radius: int, line_of_sight: bool = True) -> set[Pos]:
    """Cells in the square window that the person at `origin` sees. `opaque(cell)` is true for a
    wall or a cell outside the grid. An opaque cell is seen itself; it hides what is behind it.
    The eight neighbours are always seen."""
    r0, c0 = origin
    seen = set()
    for r in range(r0 - radius, r0 + radius + 1):
        for c in range(c0 - radius, c0 + radius + 1):
            if not line_of_sight or not any(opaque(x) for x in _between(origin, (r, c))):
                seen.add((r, c))
    return seen


class Memory:
    """One person's knowledge of the building, in the simulator's coordinates (never shown)."""

    def __init__(self) -> None:
        self.cells: dict[Pos, str] = {}     # WALL / FLOOR / EXIT / FIRE; outside the grid is WALL
        self.exits: set[Pos] = set()        # exit positions, also when the exit burns
        self.told: set[Pos] = set()         # exit positions known but not seen yet

    def see(self, cell: Pos, kind: str) -> None:
        self.cells[cell] = kind
        if kind == EXIT:
            self.exits.add(cell)
        self.told.discard(cell)

    def passable(self, cell: Pos) -> bool:
        return self.cells.get(cell) in (FLOOR, EXIT)

    def fire(self) -> set[Pos]:
        return {p for p, k in self.cells.items() if k == FIRE}

    def safe_exits(self) -> set[Pos]:
        return {p for p in self.exits | self.told if self.cells.get(p, EXIT) == EXIT}


def _bfs(sources, allowed) -> dict[Pos, int]:
    dist = {p: 0 for p in sources}
    todo = deque(sorted(dist))
    while todo:
        r, c = todo.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, 1), (0, -1)):
            p = (r + dr, c + dc)
            if p not in dist and allowed(p):
                dist[p] = dist[r, c] + 1
                todo.append(p)
    return dist


def plan(memory: Memory, pos: Pos, fire_detour: int = 6) -> dict[str, Any]:
    """The person's own route estimate: `goal` and, per action, the steps left after that action.

    exit:    shortest way to a known safe exit. Cells never seen count as free (a person walks
             toward a door they know and finds out on the way).
    explore: no usable exit is known; go to the nearest known cell that borders unseen space.
    flee:    nothing to reach; lower value = farther from known fire.
    none:    nothing to do.
    Lower is always better. None = the action cannot be taken (wall, fire) or leads nowhere.
    """
    fire = memory.fire()
    near_fire = {(r + dr, c + dc) for r, c in fire for dr, dc in DELTAS.values()} - fire
    known = set(memory.cells) | memory.told | {pos}
    lo_r, hi_r = min(p[0] for p in known) - 1, max(p[0] for p in known) + 1
    lo_c, hi_c = min(p[1] for p in known) - 1, max(p[1] for p in known) + 1

    def optimistic(p: Pos) -> bool:
        if p in memory.cells:
            return memory.passable(p)
        return lo_r <= p[0] <= hi_r and lo_c <= p[1] <= hi_c

    def options(dist: Mapping[Pos, int]) -> dict[str, int | None]:
        out: dict[str, int | None] = {}
        for a in ACTIONS:
            cell = (pos[0] + DELTAS[a][0], pos[1] + DELTAS[a][1])
            out[a] = dist.get(cell) if a == "stay" or memory.passable(cell) else None
        return out

    def best(steps: Mapping[str, int | None]) -> int | None:
        values = [v for v in steps.values() if v is not None]
        return min(values) if values else None

    def careful(sources: set[Pos], allowed) -> dict[str, int | None]:
        """Prefer a route that keeps off cells next to fire, unless it costs too many steps."""
        direct = options(_bfs(sources, allowed))
        if not near_fire or best(direct) is None:
            return direct
        calm = options(_bfs(sources - near_fire, lambda p: allowed(p) and p not in near_fire))
        if best(calm) is not None and best(calm) <= best(direct) + fire_detour:
            return calm
        return direct

    exits = memory.safe_exits()
    if exits:
        steps = careful(exits, optimistic)
        if best(steps) is not None:
            return {"goal": "exit", "steps": steps}
    frontier = {p for p in memory.cells if memory.passable(p)
                and any((p[0] + dr, p[1] + dc) not in memory.cells for dr, dc in DELTAS.values())}
    if frontier:
        steps = careful(frontier, memory.passable)
        if best(steps) is not None:
            return {"goal": "explore", "steps": steps}
    if fire:
        away = _bfs(fire, lambda p: memory.passable(p))
        steps = options({p: max(0, _FLEE_CAP - d) for p, d in away.items()})
        if steps["stay"] is None:       # fire cannot reach this pocket through known cells
            steps = {a: (0 if a == "stay" else None) for a in ACTIONS}
        return {"goal": "flee", "steps": steps}
    return {"goal": "none", "steps": {a: (0 if a == "stay" else None) for a in ACTIONS}}


class HumanObserver:
    """Turns successive public observations of ONE episode into per-person views.

    Call `observe(observation)` after reset and `observe(observation, record)` after each step,
    where `record` is the simulator's step record (it supplies each person's own last action and
    outcome). Memory is per person and never shared.
    """

    def __init__(self, config: ViewConfig = ViewConfig()) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._memory: dict[str, Memory] = {}
        self._plans: dict[str, dict[str, Any]] = {}
        self._step: int | None = None

    def memory(self, pid: str) -> Memory:
        return self._memory[pid]

    def plans(self) -> dict[str, dict[str, Any]]:
        """The latest route estimate per active person (also available when hints are off)."""
        return {pid: {"goal": p["goal"], "steps": dict(p["steps"])} for pid, p in self._plans.items()}

    def observe(self, observation: Mapping[str, Any], record: Mapping[str, Any] | None = None
                ) -> dict[str, dict[str, Any]]:
        obs = validate_observation(observation)
        if self._step is not None and (record is None or obs["step"] != self._step + 1
                                       or record.get("step") != obs["step"]):
            raise ValueError("observations must follow one episode step by step; call reset() first")
        if self._step is None and record is not None and record.get("step") != obs["step"]:
            raise ValueError("the record does not belong to this observation")
        self._step = obs["step"]
        cfg, m = self.config, obs["map"]
        h, w = m["height"], m["width"]
        walls = {tuple(x) for x in m["obstacles"]}
        burning = {tuple(x) for x in obs["burning"]}
        exits = sorted((e["id"], tuple(e["position"])) for e in m["exits"])
        exit_cells = {p for _, p in exits}
        active = {p["id"]: tuple(p["position"]) for p in obs["people"] if p["status"] == "active"}
        occupied = set(active.values())

        def opaque(cell: Pos) -> bool:
            return not (0 <= cell[0] < h and 0 <= cell[1] < w) or cell in walls

        def kind(cell: Pos) -> str:
            if opaque(cell):
                return WALL
            if cell in burning:
                return FIRE
            return EXIT if cell in exit_cells else FLOOR

        views, self._plans = {}, {}
        for pid in sorted(active):
            pos = active[pid]
            mem = self._memory.get(pid)
            if mem is None or not cfg.memory:   # without memory only the prior knowledge persists
                mem = self._memory[pid] = Memory()
                if cfg.exit_knowledge == "all":
                    mem.told = set(exit_cells)
                elif cfg.exit_knowledge == "one" and exits:
                    pick = int(hashlib.sha256(pid.encode()).hexdigest(), 16) % len(exits)
                    mem.told = {exits[pick][1]}
            seen = visible_cells(pos, opaque, cfg.radius, cfg.line_of_sight)
            for cell in seen:
                mem.see(cell, kind(cell))
            people = (seen & occupied) - {pos}
            window = []
            for r in range(pos[0] - cfg.radius, pos[0] + cfg.radius + 1):
                row = []
                for c in range(pos[1] - cfg.radius, pos[1] + cfg.radius + 1):
                    cell = (r, c)
                    row.append(SELF if cell == pos else PERSON if cell in people
                               else mem.cells.get(cell, UNSEEN))
                window.append("".join(row))
            moves = {}
            for a in MOVES:
                cell = (pos[0] + DELTAS[a][0], pos[1] + DELTAS[a][1])
                moves[a] = KINDS[PERSON if cell in people else mem.cells[cell]]
            route = plan(mem, pos, cfg.fire_detour)
            self._plans[pid] = route
            last = (record or {}).get("movements", {}).get(pid)
            views[pid] = {
                "view_version": VIEW_VERSION, "person": pid, "step": obs["step"], "rules": dict(obs["rules"]),
                "radius": cfg.radius, "window": window, "moves": moves,
                "last_action": record["actions"][pid] if last else None,
                "last_outcome": last["outcome"] if last else None,
                "goal": route["goal"] if cfg.hints else None,
                "steps": dict(route["steps"]) if cfg.hints else None,
                "exits": sorted([p[0] - pos[0], p[1] - pos[1]] for p in mem.safe_exits()),
            }
        for pid in set(self._memory) - set(active):
            del self._memory[pid]       # resolved people keep no state
        return views


def admissible(view: Mapping[str, Any]) -> tuple[str, ...]:
    """Actions a person can see are not wasted: no wall, no edge, no fire. "stay" always stays.
    This is the same rule as `shield.admissible_actions`, read from the person's own view."""
    return tuple(a for a in ACTIONS if a == "stay" or view["moves"][a] not in ("wall", "fire"))


def validate_view(view: Mapping[str, Any]) -> None:
    """Fail closed on a malformed view (data boundary for stored datasets)."""
    fields = {"view_version", "person", "step", "rules", "radius", "window", "moves", "last_action",
              "last_outcome", "goal", "steps", "exits"}
    if not isinstance(view, Mapping) or set(view) != fields or view["view_version"] != VIEW_VERSION:
        raise ValueError("not a human view")
    side = 2 * view["radius"] + 1
    window = view["window"]
    if len(window) != side or any(len(row) != side or set(row) - set("#.EFP?@") for row in window):
        raise ValueError("bad view window")
    if window[view["radius"]][view["radius"]] != SELF or sum(row.count(SELF) for row in window) != 1:
        raise ValueError("the person must be at the window centre")
    if set(view["moves"]) != set(MOVES) or set(view["moves"].values()) - set(KINDS.values()):
        raise ValueError("bad move kinds")
    if view["last_action"] not in (None, *ACTIONS):
        raise ValueError("bad last action")
    if (view["goal"] is None) != (view["steps"] is None):
        raise ValueError("goal and steps go together")
    if view["goal"] is not None and (view["goal"] not in GOALS or set(view["steps"]) != set(ACTIONS)):
        raise ValueError("bad route estimate")
