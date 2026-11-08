"""Generator-private scenario analysis: reachability, fire timing, bottlenecks, signatures.

These functions read the full scenario, including the seed and the true fire
trajectory. Their results are labels and diagnostics for the generator and the
split audit. They are never policy inputs.

Escape model (`latest_departure`). Let F(c) be the first step after which cell
c burns (0 for ignitions; `inf` if it does not burn within the horizon). In
step t a person moves from a to b (b = a or a 4-neighbour). The move needs
b not burning at the start of the step (t - 1 < F(b)). An exit evacuates at
once. A non-exit cell must survive the fire phase of step t (t < F(b)).
Burning only grows, so waiting is safe while the current cell does not burn,
and L(c), the latest time from which a lone person at c can still escape, is
well defined. A person at p escapes alone iff L(p) >= 0. This ignores other
people (collisions, queues), so it is an optimistic, necessary condition.
"""
from __future__ import annotations

import heapq
from collections import deque
from collections.abc import Iterable, Mapping, Sequence

from ember_v1.sim.canonical import digest
from ember_v1.sim.fire import Terrain, spread
from ember_v1.sim.fire import neighbours as fire_neighbours
from ember_v1.sim.rng import SplitMix64, stream_seed
from ember_v1.sim.scenario import Scenario

from .layouts import neighbours4

Pos = tuple[int, int]
INF = float("inf")
SIG_K = 16


def bfs(h: int, w: int, blocked: set[Pos] | frozenset[Pos], sources: Iterable[Pos],
        stop: set[Pos] | frozenset[Pos] = frozenset()) -> dict[Pos, int]:
    """Multi-source BFS distances over non-blocked cells. Cells in `stop` are reached but not expanded."""
    dist: dict[Pos, int] = {}
    queue: deque[Pos] = deque()
    for s in sources:
        if s not in blocked and s not in dist:
            dist[s] = 0
            queue.append(s)
    while queue:
        p = queue.popleft()
        if p in stop and dist[p] > 0:
            continue
        for n in neighbours4(p, h, w):
            if n not in blocked and n not in dist:
                dist[n] = dist[p] + 1
                queue.append(n)
    return dist


def core_hash(s: Scenario) -> str:
    """Content hash without scenario_id and seed: equal for two copies of one situation."""
    d = s.to_dict()
    d.pop("scenario_id")
    d.pop("seed")
    return digest(d)


def exit_region(s: Scenario, burning: frozenset[Pos] | None = None) -> set[Pos]:
    """Cells with a 4-path to a non-burning exit over non-wall, non-burning cells (the simulator's rule)."""
    burn = frozenset(s.ignitions) if burning is None else burning
    blocked = set(s.obstacles) | burn
    return set(bfs(s.height, s.width, blocked, [p for _, p in s.exits if p not in burn]))


def fire_arrival(s: Scenario) -> dict[Pos, int]:
    """First step after which each cell burns, for the scenario's own seed, up to `max_steps`.

    Equal to iterating `ember_v1.sim.fire.spread` on the fire stream, but
    O(frontier) per spread step instead of O(burning):
      deterministic  every spread step ignites the whole frontier, so
                     F(c) = spread_interval * (flammable BFS distance).
      stochastic     the same candidates (row-major), the same draw per
                     burning-neighbour edge, the same stream.
    Tests compare it with `spread` over the full horizon; the probe episode
    compares it with the simulator's ignitions at every step it runs.
    """
    terrain = Terrain(s.height, s.width, frozenset(s.obstacles), frozenset(p for _, p in s.exits),
                      s.exits_flammable)
    burning = set(s.ignitions)
    arrival = {p: 0 for p in burning}
    front = {n for b in burning for n in fire_neighbours(b) if n not in burning and terrain.flammable(n)}
    if s.fire_model == "deterministic":
        t = s.spread_interval
        while front and t <= s.max_steps:
            for c in front:
                arrival[c] = t
            burning |= front
            front = {n for b in front for n in fire_neighbours(b) if n not in burning and terrain.flammable(n)}
            t += s.spread_interval
        return arrival
    rng = SplitMix64(stream_seed(s.seed, "fire"))
    p = s.spread_probability
    for t in range(s.spread_interval, s.max_steps + 1, s.spread_interval):
        if not front:
            break
        ignited = []
        for c in sorted(front):
            hit = False
            for n in fire_neighbours(c):
                if n in burning:
                    hit = rng.bernoulli(p) or hit   # one draw per edge, no short-circuit
            if hit:
                ignited.append(c)
        for c in ignited:
            arrival[c] = t
            burning.add(c)
            front.discard(c)
        for c in ignited:
            for n in fire_neighbours(c):
                if n not in burning and terrain.flammable(n):
                    front.add(n)
    return arrival


def fire_arrival_reference(s: Scenario) -> dict[Pos, int]:
    """Slow reference: iterate the simulator's `spread` for every step (tests only)."""
    terrain = Terrain(s.height, s.width, frozenset(s.obstacles), frozenset(p for _, p in s.exits),
                      s.exits_flammable)
    rng = SplitMix64(stream_seed(s.seed, "fire"))
    burning = frozenset(s.ignitions)
    arrival = {p: 0 for p in burning}
    for t in range(1, s.max_steps + 1):
        new = spread(terrain, burning, t, s.fire_model, s.spread_interval, s.spread_probability, rng)
        for p in new:
            arrival[p] = t
        burning = burning | new
    return arrival


def latest_departure(s: Scenario, arrival: Mapping[Pos, int]) -> dict[Pos, float]:
    """L(c) for every non-exit free cell from which a lone escape exists (see module doc)."""
    walls = set(s.obstacles)
    exits = {p for _, p in s.exits}
    F = lambda p: arrival.get(p, INF)  # noqa: E731
    best: dict[Pos, float] = {}
    heap: list[tuple[float, Pos]] = []

    def offer(cell: Pos, value: float) -> None:
        value = min(value, F(cell) - 1)
        if value >= 0 and value > best.get(cell, -1):
            best[cell] = value
            heapq.heappush(heap, (-value, cell))

    for e in sorted(exits):
        g = min(F(e) - 1, s.max_steps - 1)
        for n in neighbours4(e, s.height, s.width):
            if n not in walls and n not in exits:
                offer(n, g)
    while heap:
        neg, cell = heapq.heappop(heap)
        if -neg < best.get(cell, -1):
            continue
        for n in neighbours4(cell, s.height, s.width):
            if n not in walls and n not in exits:
                offer(n, -neg - 1)
    return best


class RouteGraph:
    """Escape-route graph at one burning set, with cut vertices relative to safety.

    Nodes: free, non-burning, non-exit cells plus a virtual sink. A cell next
    to a non-burning exit has an edge to the sink. Exits are merged into the
    sink because stepping on an exit evacuates, so no route passes through one.
    One iterative DFS from the sink gives Tarjan low-links. Cell u separates the
    DFS subtree of its child c from safety iff low[c] >= disc[u] (u is not the
    sink). A person has two vertex-disjoint routes iff no cell separates it
    from the sink and it reaches at least two exits.
    """

    SINK = (-1, -1)

    def __init__(self, h: int, w: int, walls: Iterable[Pos], exit_cells: Iterable[Pos],
                 burning: Iterable[Pos] = ()) -> None:
        burning = set(burning)
        exits = set(exit_cells)
        live_exits = exits - burning
        blocked = set(walls) | burning | exits
        SINK = self.SINK
        adj: dict[Pos, list[Pos]] = {SINK: []}

        def nbrs(p: Pos) -> list[Pos]:
            out = adj.get(p)
            if out is None:
                out = [n for n in neighbours4(p, h, w) if n not in blocked]
                if any(n in live_exits for n in neighbours4(p, h, w)):
                    out.append(SINK)
                adj[p] = out
            return out

        adj[SINK] = sorted({n for e in live_exits for n in neighbours4(e, h, w) if n not in blocked})
        disc = {SINK: 0}
        low = {SINK: 0}
        parent: dict[Pos, Pos] = {}
        order = [SINK]
        stack = [(SINK, iter(adj[SINK]))]
        while stack:
            u, it = stack[-1]
            v = next(it, None)
            if v is None:
                stack.pop()
                if stack:
                    pu = stack[-1][0]
                    low[pu] = min(low[pu], low[u])
                continue
            if v not in disc:
                disc[v] = low[v] = len(order)
                order.append(v)
                parent[v] = u
                stack.append((v, iter(nbrs(v))))
            elif v != parent.get(u):
                low[u] = min(low[u], disc[v])
        self.disc, self.low, self.parent, self.order = disc, low, parent, order

    def reachable(self, p: Pos) -> bool:
        return p in self.disc and p != self.SINK

    def separating_edges(self) -> list[tuple[Pos, Pos]]:
        """(u, c): cell u cuts the subtree of its DFS child c off from safety."""
        return [(u, c) for c, u in self.parent.items() if u != self.SINK and self.low[c] >= self.disc[u]]

    def cut_cells(self) -> list[Pos]:
        return sorted({u for u, _ in self.separating_edges()})

    def bottlenecks_for(self, people: Iterable[Pos]) -> tuple[list[Pos], set[Pos]]:
        """(cells that separate at least one person from safety, people with such a cell)."""
        count = {v: 0 for v in self.order}
        for p in people:
            if p in count and p != self.SINK:
                count[p] += 1
        for v in reversed(self.order[1:]):            # children before parents
            count[self.parent[v]] += count[v]
        cuts = {u for u, c in self.separating_edges() if count[c] > 0}
        cut_below = {c for u, c in self.separating_edges()}
        dependent: set[Pos] = set()
        has: dict[Pos, bool] = {self.SINK: False}
        for v in self.order[1:]:                      # parents before children
            has[v] = has[self.parent[v]] or v in cut_below
        for p in people:
            if has.get(p):
                dependent.add(p)
        return sorted(cuts), dependent


def route_structure(s: Scenario, people: Sequence[Pos]) -> dict[str, object]:
    """Per-crowd route facts at step 0: bottleneck cells, single-exit dependence, disjoint alternatives."""
    exits = [p for _, p in s.exits if p not in set(s.ignitions)]
    g = RouteGraph(s.height, s.width, s.obstacles, [p for _, p in s.exits], s.ignitions)
    cuts, dependent = g.bottlenecks_for(people)
    blocked = set(s.obstacles) | set(s.ignitions)
    stop = {p for _, p in s.exits}
    per_exit = [_exit_reach(s.height, s.width, blocked, e, stop) for e in exits]
    alternatives = single_exit = reachable = 0
    for p in people:
        if not g.reachable(p):
            continue
        reachable += 1
        n_exits = sum(1 for d in per_exit if p in d)
        single_exit += n_exits == 1
        alternatives += n_exits >= 2 and p not in dependent
    return {"bottleneck_cells": cuts, "people_with_disjoint_alternatives": alternatives,
            "people_single_exit": single_exit, "people_reachable": reachable}


def _exit_reach(h: int, w: int, blocked: set[Pos], exit_pos: Pos, stop: set[Pos]) -> set[Pos]:
    return set(bfs(h, w, blocked, [exit_pos], stop=stop))


def cut_cells(h: int, w: int, walls: Iterable[Pos], exit_cells: Iterable[Pos]) -> list[Pos]:
    """Cells of the static map that separate some free cell from every exit (bottlenecks)."""
    return RouteGraph(h, w, walls, exit_cells).cut_cells()


def capacity_feasible(s: Scenario, arrival: Mapping[Pos, int], people: Sequence[Pos]) -> bool:
    """Necessary condition: one person per exit per step, within each exit's usable window.

    Exit e admits people in steps d_e..min(F(e), max_steps), where d_e is the
    shortest distance from any person. If the total window is below the crowd
    size, some person cannot evacuate whatever the policy does.
    """
    walls = set(s.obstacles)
    exits = [p for _, p in s.exits]
    blocked = walls | set(s.ignitions)
    dist = bfs(s.height, s.width, blocked, people, stop=set(exits))
    total = 0
    for e in exits:
        if e not in dist:
            continue
        last = min(arrival.get(e, INF), s.max_steps)
        total += max(0, int(last) - dist[e] + 1)
    return total >= len(people)


# ----- topology identity and near-duplicate signatures ----------------------

def _transforms(h: int, w: int) -> list[tuple[tuple[int, int], object]]:
    """The 8 symmetries of the rectangle: (new dims, cell map)."""
    return [
        ((h, w), lambda r, c: (r, c)), ((h, w), lambda r, c: (r, w - 1 - c)),
        ((h, w), lambda r, c: (h - 1 - r, c)), ((h, w), lambda r, c: (h - 1 - r, w - 1 - c)),
        ((w, h), lambda r, c: (c, r)), ((w, h), lambda r, c: (w - 1 - c, r)),
        ((w, h), lambda r, c: (c, h - 1 - r)), ((w, h), lambda r, c: (w - 1 - c, h - 1 - r)),
    ]


def topology_hash(h: int, w: int, walls: Iterable[Pos], exits: Iterable[Pos]) -> str:
    """Hash of the static map, invariant under the 8 grid symmetries and exit renaming."""
    walls, exits = list(walls), list(exits)
    forms = []
    for dims, f in _transforms(h, w):
        forms.append((dims, sorted(f(*p) for p in walls), sorted(f(*p) for p in exits)))  # type: ignore[operator]
    dims, ws, es = min(forms)
    return digest({"grid": list(dims), "walls": [list(p) for p in ws], "exits": [list(p) for p in es]})


def coarse(h: int, w: int, walls: Iterable[Pos]) -> list[list[float]]:
    """Area-weighted wall fraction on a SIG_K x SIG_K grid over the unit square.

    Area weighting commutes with the grid symmetries, so the 8 transforms of
    this matrix equal the matrices of the transformed maps.
    """
    K = SIG_K
    grid = [[0.0] * K for _ in range(K)]

    def spans(i: int, n: int) -> list[tuple[int, float]]:
        lo, hi = i * K, (i + 1) * K  # cell [i/n, (i+1)/n) scaled by n*K
        out = []
        for b in range(lo // n, min(K - 1, (hi - 1) // n) + 1):
            o = min(hi, (b + 1) * n) - max(lo, b * n)
            if o > 0:
                out.append((b, o / (n * K)))
        return out

    rows = [spans(r, h) for r in range(h)]
    cols = [spans(c, w) for c in range(w)]
    for r, c in walls:
        for br, fr in rows[r]:
            for bc, fc in cols[c]:
                grid[br][bc] += fr * fc
    cell = 1.0 / (K * K)
    return [[v / cell for v in row] for row in grid]


def _sym(m: list[list[float]]) -> list[list[list[float]]]:
    t = [list(x) for x in zip(*m)]
    out = []
    for base in (m, t):
        out.append(base)
        out.append([row[::-1] for row in base])
        out.append(base[::-1])
        out.append([row[::-1] for row in base[::-1]])
    return out


def _exit_points(h: int, w: int, exits: Iterable[Pos]) -> list[tuple[float, float]]:
    return [((r + 0.5) / h, (c + 0.5) / w) for r, c in exits]


def _sym_points(pts: list[tuple[float, float]]) -> list[list[tuple[float, float]]]:
    out = []
    for base in (pts, [(c, r) for r, c in pts]):
        out.append(base)
        out.append([(r, 1 - c) for r, c in base])
        out.append([(1 - r, c) for r, c in base])
        out.append([(1 - r, 1 - c) for r, c in base])
    return out


def _chamfer(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    if not a or not b:
        return INF
    d = lambda p, q: max(abs(p[0] - q[0]), abs(p[1] - q[1]))  # noqa: E731
    ab = sum(min(d(p, q) for q in b) for p in a) / len(a)
    ba = sum(min(d(p, q) for q in a) for p in b) / len(b)
    return max(ab, ba)


class MapSignature:
    """Precomputed near-duplicate signature of one static map."""

    __slots__ = ("h", "w", "walls_frac", "mats", "exits")

    def __init__(self, h: int, w: int, walls: Iterable[Pos], exits: Iterable[Pos]) -> None:
        walls = list(walls)
        self.h, self.w = h, w
        self.walls_frac = len(walls) / (h * w)
        self.mats = [[v for row in m for v in row] for m in _sym(coarse(h, w, walls))]
        self.exits = _sym_points(_exit_points(h, w, exits))


# Thresholds (documented in docs/P1_SCENARIO_GENERATOR.md section 5).
NEAR_SCALE = 1.3        # area ratio
NEAR_ASPECT = 1.15      # aspect ratio ratio (orientation free)
NEAR_WALL_MAD = 0.03    # mean absolute wall-fraction difference on the coarse grid
NEAR_EXIT = 0.1         # symmetric chamfer distance of exits (unit square, Chebyshev)
# Exit-free wall-geometry rule (`wall_near_duplicate`): applies only when both maps have at
# least this wall fraction. In sparser halls the exits carry the structure, so only the
# exit-aware `near_duplicate` rule applies there.
NEAR_WALL_MIN_FRAC = 0.05


def _comparable(a: MapSignature, b: MapSignature) -> range | None:
    """Symmetries to compare, or None if the scale or aspect gates fail."""
    area_a, area_b = a.h * a.w, b.h * b.w
    if max(area_a, area_b) > NEAR_SCALE * min(area_a, area_b):
        return None
    asp_a, asp_b = max(a.h, a.w) / min(a.h, a.w), max(b.h, b.w) / min(b.h, b.w)
    if max(asp_a, asp_b) > NEAR_ASPECT * min(asp_a, asp_b):
        return None
    if min(asp_a, asp_b) < NEAR_ASPECT:          # near square: any orientation
        return range(8)
    return range(4) if (a.h <= a.w) == (b.h <= b.w) else range(4, 8)   # keep long axes aligned


def near_duplicate(a: MapSignature, b: MapSignature) -> tuple[bool, float, float]:
    """(is near-duplicate, best wall distance, exit distance at that transform)."""
    allowed = _comparable(a, b)
    if allowed is None:
        return False, INF, INF
    if abs(a.walls_frac - b.walls_frac) > NEAR_WALL_MAD:   # MAD >= |difference of means|
        return False, INF, INF
    base, n = a.mats[0], len(a.mats[0])
    best_hit = best_any = (INF, INF)
    for k in allowed:
        mad = sum(abs(x - y) for x, y in zip(base, b.mats[k])) / n
        ex = _chamfer(a.exits[0], b.exits[k]) if mad <= NEAR_WALL_MAD else INF
        best_any = min(best_any, (mad, ex))
        if mad <= NEAR_WALL_MAD and ex <= NEAR_EXIT:
            best_hit = min(best_hit, (mad, ex))
    if best_hit[0] < INF:
        return True, best_hit[0], best_hit[1]
    return False, best_any[0], best_any[1]


def wall_near_duplicate(a: MapSignature, b: MapSignature) -> tuple[bool, float]:
    """(same wall geometry whatever the exits, best wall distance).

    Catches one building reused with other exits, which `near_duplicate`
    allows because its exit distance is large. Maps below NEAR_WALL_MIN_FRAC
    walls are out of scope: two sparse halls are not one building because
    both have few walls.
    """
    allowed = _comparable(a, b)
    if (allowed is None or min(a.walls_frac, b.walls_frac) < NEAR_WALL_MIN_FRAC
            or abs(a.walls_frac - b.walls_frac) > NEAR_WALL_MAD):
        return False, INF
    base, n = a.mats[0], len(a.mats[0])
    mad = min(sum(abs(x - y) for x, y in zip(base, b.mats[k])) / n for k in allowed)
    return mad <= NEAR_WALL_MAD, mad
