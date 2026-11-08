"""T3: dynamic Ant Colony Optimization (ACO) reference agent (issue #23).

CPU only, observation only. T3 is a standalone policy with its own name,
version, configuration, teacher seed and evaluation. It imports no T0, T1 or
T2 code. It uses only shared public utilities: the observation contract, the
canonical digest, `grid-utils` (PublicGrid, the static exit distance field, the
move order) and the public movement resolver (`sim.movement.resolve_moves`).

One decision (all stages are functions of the validated observation, the
public rules, the frozen configuration and the teacher seed):

  1 Graph. Vertices: grid cells that are not walls and do not burn now.
    Edges: 4-neighbour moves. Destinations: exits that do not burn now (an
    agent evacuates on any of them, so they are absorbing).

  2 Public fire risk (T3's own model, never a future state). Layer BFS from the
    burning set over flammable cells (not wall; exits only if exits_flammable)
    gives the earliest number of spread steps l(c) that can ignite c. Spread
    happens in absolute step t only if t % spread_interval == 0. With n(k)
    spread steps in the next k steps, hazard(c, k):
      deterministic: 1 if n(k) >= l(c), else 0 (exact for the public rule);
      stochastic, l = 1: 1 - (1 - p)^(b * n), b = burning neighbours now;
      stochastic, l >= 2: P(Binomial(n, p) >= l), single-front lower bound.
    On a safe exit the hazard is hazard(c, k - 1): evacuation comes before
    spread. Stochastic hazards are soft costs only. Deterministic hazards of 1
    are certain under the public rule, so those (cell, step) pairs are pruned.

  3 Route cost of a path c_1..c_m (c_m a safe exit), arrival step k = index:
      sum_k [1 + w_fire*hazard(c_k, k) + w_exposure*burning_neighbours_now(c_k)
             + w_congestion*load(c_k, k) + w_bottleneck*load(c_k, k)*[bottleneck]
             + w_occupied*[k == 1 and c_1 is occupied by another person now]]
      + w_queue*exit_wait(c_m, m) + w_unsafe*[max hazard >= unsafe_threshold].
    load(c, k): other people's current best routes at (c, k). Iteration 0 uses
    the static BFS descent routes as the initial flow estimate. exit_wait: a
    throughput-1 FIFO queue of the other people's arrivals at that exit.
    Bottleneck: an open cell with <= 2 open 4-neighbours.

  4 Colony. One pheromone table on directed edges, shared by all ants of the
    decision and reset at every decision. Per iteration each routable person
    launches `ants_per_person` ants (at most `max_ants_per_iteration` in
    total; a rotating window chooses the ants when the cap binds). An ant
    walks from the person's cell without revisiting cells. Transition (ACS
    pseudo-random proportional rule): with probability q0 the argmax of
    tau^alpha * eta^beta (exploitation, ties in N, S, E, W order), else a
    roulette draw proportional to it (exploration). Heuristic (reduced cost):
    eta = 1 / (step_cost(v, k) + h(v) - h(u) + 1), h = static exit distance.
    An ant dies at a dead end or after the maximum path length.
    After the ants: candidate routes are re-scored, evaporation
    tau <- max(tau_min, (1 - rho) * tau) on every edge, then deposits: the
    iteration-best feasible ant of each person deposits deposit_q * d0 / C on
    its edges and the best-so-far route of each person deposits
    elite_weight * deposit_q * d0 / C (d0 = static distance, C = route cost,
    so the quality is in (0, 1]); tau <= tau_max. Then load and exit queues are
    rebuilt from the new best routes. After the last iteration, each routable
    person whose ants found no route (e.g. no ant under the cap) gets one
    greedy ant (q0 = 1, no random draws) on the final pheromone.

  5 Candidates. Per person the best route for each first move is kept and
    re-scored with the final load. Ranking: (cost, length, move order).

  6 Joint action. People with a route, in priority (best route length, ID),
    then people without a route. One-step reservations: a first move is taken
    only if no one has claimed its target cell, the target's occupant has not
    decided to stay, and it is not a swap. Else the next candidate first move,
    else stay. The public resolver then checks the full map (lowest-ID
    precedence, follow chains, cycles); a blocked mover tries its next option
    for up to `repair_rounds` rounds; a move still predicted blocked becomes
    `stay` (end positions asserted unchanged).

  7 No route. A person with no static path to a safe exit, or whose every
    found route is certain to burn (deterministic), shelters: stay or move to
    an open, unclaimed neighbour with the least step-1 hazard, then the latest
    fire layer. Reason `no_route_shelter` / `no_feasible_route_shelter`.

  8 Budgets and fallback. `max_ant_steps` (deterministic cap) and
    `cpu_budget_ms` (thread CPU, checked after every ant) stop the colony. A
    person with a static route but no feasible ACO route gets the first step
    of the static BFS descent (shared grid-utils field), labeled
    `fallback_bfs_descent` in the diagnostics. No T0, T1 or T2 code is called.

Randomness: one teacher-owned stream per decision,
`random.Random(int(sha256(ACO_STREAM/<teacher seed>/<canonical public digest>)[:8]))`.
The canonical digest covers only the public observation, with people and exits
sorted by ID. T3 keeps no state between decisions.
"""
from __future__ import annotations

import hashlib
import math
import random
import time
from bisect import bisect_right
from collections import Counter
from typing import Any

from ..sim.canonical import digest
from ..sim.movement import resolve_moves
from .grid import MOVES, PublicGrid, exit_distance_field
from .interface import ReferencePolicy

Pos = tuple[int, int]
T3_DEFAULT_SEED = 20261123
ACO_STREAM = "ember_v1/t3/aco@1"
DELTA = dict(MOVES)
MOVE_RANK = {a: i for i, (a, _) in enumerate(MOVES)}

_DESCRIPTIVE = {
    "graph": "grid_cells; 4_neighbour; walls_and_burning_now_removed; safe_exits_absorbing_destinations",
    "fire_risk": "public_rule_layer_bfs; deterministic_exact_pruned_at_1; stochastic_soft: "
                 "layer1 1-(1-p)^(b*n), layer>=2 P(Binomial(n,p)>=l) lower bound; exit uses k-1",
    "route_cost": "sum_k[1 + w_fire*hazard + w_exposure*burning_neighbours_now + w_congestion*load "
                  "+ w_bottleneck*load*bottleneck + w_occupied*occupied_now(k=1)] + w_queue*exit_wait "
                  "+ w_unsafe*[max_hazard >= unsafe_threshold]",
    "arrival_time": "path_index_no_waiting",
    "congestion_model": "time_indexed_load_of_other_best_routes; iteration0_static_bfs_descent_routes; "
                        "exit_queue_fifo_throughput_1 (own entry approximated by -1)",
    "pheromone": "directed_edges; one_colony_per_decision; reset_every_decision; init tau0",
    "transition_rule": "acs_pseudo_random_proportional: argmax(tau^alpha*eta^beta) w.p. q0 else roulette; "
                       "ties N,S,E,W",
    "heuristic": "eta = 1/(step_cost(v,k) + h(v) - h(u) + 1); h = static exit distance",
    "tabu": "no_revisit_within_ant",
    "evaporation_rule": "after ants: tau <- max(tau_min, (1-rho)*tau) on every edge",
    "deposit_rule": "after evaporation: iteration_best feasible ant per person deposit_q*d0/C; "
                    "best_so_far per person elite_weight*deposit_q*d0/C; tau <= tau_max",
    "ant_allocation": "ants_per_person per routable person (ID order); cap max_ants_per_iteration; "
                      "rotating window when capped",
    "final_ant": "after iterations: one greedy ant (q0=1, no draws) per routable person without a route",
    "path_length_limit": "min(max_path_length, floor(max_path_factor*d0) + max_path_extra)",
    "feasibility": "reaches safe exit over cells not burning now; deterministic certain hazard pruned; "
                   "stochastic forecasts never make a route infeasible",
    "candidate_routes": "best route per (person, first move); re-scored with final load; "
                        "rank (cost, length, move order)",
    "coordination": "priority (best route length, person id), routed people first; one_step_target_reservation; "
                    "no swap; resolver repair rounds; unrepaired blocked -> stay",
    "no_route_mode": "shelter: stay or open unclaimed neighbour, min step1 hazard then max fire layer",
    "fallback": "static_bfs_descent_first_move labeled fallback_bfs_descent; never T0/T1/T2",
    "rng": "random.Random(sha256(ember_v1/t3/aco@1/<teacher_seed>/<canonical_public_digest>)[:8])",
    "memory": "stateless_between_decisions",
}
_NUMERIC = {
    "iterations": (int, 1, 1000),
    "ants_per_person": (int, 1, 1000),
    "max_ants_per_iteration": (int, 1, 10 ** 6),
    "alpha": (float, 0.0, 10.0),
    "beta": (float, 0.0, 10.0),
    "evaporation": (float, 0.0, 1.0),
    "q0": (float, 0.0, 1.0),
    "tau0": (float, 1e-6, 1e6),
    "tau_min": (float, 0.0, 1e6),
    "tau_max": (float, 1e-6, 1e6),
    "deposit_q": (float, 0.0, 1e6),
    "elite_weight": (float, 0.0, 1e6),
    "max_path_factor": (float, 1.0, 100.0),
    "max_path_extra": (int, 0, 10 ** 5),
    "max_path_length": (int, 1, 10 ** 6),
    "w_fire": (float, 0.0, 1e6),
    "w_exposure": (float, 0.0, 1e6),
    "w_congestion": (float, 0.0, 1e6),
    "w_bottleneck": (float, 0.0, 1e6),
    "w_occupied": (float, 0.0, 1e6),
    "w_queue": (float, 0.0, 1e6),
    "w_unsafe": (float, 0.0, 1e6),
    "unsafe_threshold": (float, 0.0, 1.0),
    "repair_rounds": (int, 0, 1000),
    "max_ant_steps": (int, 0, 10 ** 9),
    "cpu_budget_ms": (float, 0.0, 1e8),
}


def canonical_public_digest(obs: dict[str, Any]) -> str:
    """Digest of the validated public observation with people and exits sorted by ID (order invariant)."""
    m = obs["map"]
    return digest({"step": obs["step"], "rules": obs["rules"],
                   "map": {"height": m["height"], "width": m["width"], "obstacles": sorted(m["obstacles"]),
                           "exits": sorted(m["exits"], key=lambda x: x["id"])},
                   "people": sorted(obs["people"], key=lambda p: p["id"]), "burning": sorted(obs["burning"])})


def aco_seed(teacher_seed: int, public_digest: str) -> int:
    h = hashlib.sha256(f"{ACO_STREAM}/{teacher_seed}/{public_digest}".encode("ascii")).digest()
    return int.from_bytes(h[:8], "big")


def _nbrs(p: Pos) -> tuple[Pos, ...]:
    r, c = p
    return (r - 1, c), (r + 1, c), (r, c + 1), (r, c - 1)


class FireRisk:
    """T3's public-rule hazard model (section 2 of the module docstring)."""

    def __init__(self, obs: dict[str, Any], grid: PublicGrid) -> None:
        rules = obs["rules"]
        self.model = rules["fire_model"]
        self.p = rules["spread_probability"]
        self.interval = rules["spread_interval"]
        self.t0 = obs["step"]
        exits = {p for _, p in grid.exits}
        flammable_exits = rules["exits_flammable"]
        h, w, walls, burning = grid.height, grid.width, grid.walls, grid.burning
        layer: dict[Pos, int] = {}
        frontier = sorted(burning)
        d = 0
        seen = set(burning)
        while frontier:
            d += 1
            nxt = []
            for b in frontier:
                for n in _nbrs(b):
                    if n in seen or not (0 <= n[0] < h and 0 <= n[1] < w) or n in walls:
                        continue
                    if n in exits and not flammable_exits:
                        continue
                    seen.add(n)
                    layer[n] = d
                    nxt.append(n)
            frontier = nxt
        self.layer = layer
        self.adjacent = {c: sum(n in burning for n in _nbrs(c)) for c in layer if layer[c] == 1}
        self._cache: dict[tuple[Pos, int], float] = {}
        self._tail: dict[tuple[int, int], float] = {}

    def spreads(self, k: int) -> int:
        """Spread steps among the next k absolute steps t0+1..t0+k."""
        if k <= 0:
            return 0
        return (self.t0 + k) // self.interval - self.t0 // self.interval

    def _binom_tail(self, n: int, l: int) -> float:
        key = (n, l)
        v = self._tail.get(key)
        if v is None:
            p = self.p
            v = 0.0 if l > n else sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(l, n + 1))
            v = min(1.0, max(0.0, v))
            self._tail[key] = v
        return v

    def hazard(self, c: Pos, k: int) -> float:
        """Probability-like risk that c is burning at the end of relative step k (k <= 0: 0)."""
        if k <= 0:
            return 0.0
        l = self.layer.get(c)
        if l is None:
            return 0.0
        key = (c, k)
        v = self._cache.get(key)
        if v is not None:
            return v
        n = self.spreads(k)
        if self.model == "deterministic":
            v = 1.0 if n >= l else 0.0
        elif n == 0:
            v = 0.0
        elif l == 1:
            v = 1.0 - (1.0 - self.p) ** (self.adjacent[c] * n)
        else:
            v = self._binom_tail(n, l)
        self._cache[key] = v
        return v


class T3DynamicACO(ReferencePolicy):
    name = "T3"
    policy = "dynamic_ant_colony_optimization"
    version = "1.0.0"

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        # Frozen preregistered configuration (docs/P1_T3_ACO.md section 3).
        return {**_DESCRIPTIVE,
                "iterations": 10, "ants_per_person": 2, "max_ants_per_iteration": 64,
                "alpha": 1.0, "beta": 2.0, "evaporation": 0.2, "q0": 0.5,
                "tau0": 1.0, "tau_min": 0.1, "tau_max": 5.0, "deposit_q": 1.0, "elite_weight": 1.0,
                "max_path_factor": 2.0, "max_path_extra": 12, "max_path_length": 400,
                "w_fire": 20.0, "w_exposure": 1.0, "w_congestion": 1.0, "w_bottleneck": 1.0,
                "w_occupied": 0.5, "w_queue": 1.0, "w_unsafe": 20.0, "unsafe_threshold": 0.5,
                "repair_rounds": 6, "max_ant_steps": 100000, "cpu_budget_ms": 3000.0}

    def __init__(self, config: dict[str, Any] | None = None, *, seed: int = T3_DEFAULT_SEED) -> None:
        if seed is None:
            raise ValueError("T3: a teacher seed is required")
        super().__init__(config, seed=seed)
        cfg = self.identity.config
        for key, value in _DESCRIPTIVE.items():
            if cfg[key] != value:
                raise ValueError(f"T3: {key} is not configurable")
        for key, (typ, lo, hi) in _NUMERIC.items():
            v = cfg[key]
            ok = (type(v) is int) if typ is int else (type(v) in (int, float) and math.isfinite(v))
            if not ok or not lo <= v <= hi:
                raise ValueError(f"T3: {key} must be {typ.__name__} in [{lo}, {hi}]")
        if cfg["tau_min"] > cfg["tau0"] or cfg["tau0"] > cfg["tau_max"]:
            raise ValueError("T3: need tau_min <= tau0 <= tau_max")

    def _plan(self, obs: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
        return _Colony(self, obs).run()


class _Route:
    __slots__ = ("path", "cost", "hmax", "exit", "first")

    def __init__(self, path: tuple[Pos, ...], cost: float, hmax: float, exit_id: str, first: str) -> None:
        self.path, self.cost, self.hmax, self.exit, self.first = path, cost, hmax, exit_id, first

    def key(self) -> tuple:
        return (self.cost, len(self.path), MOVE_RANK[self.first], self.path)


class _Colony:
    def __init__(self, pol: T3DynamicACO, obs: dict[str, Any]) -> None:
        self.cfg = cfg = pol.identity.config
        self.obs = obs
        self.grid = g = PublicGrid.from_observation(obs)
        self.teacher_seed = pol.seed
        self.public_digest = canonical_public_digest(obs)
        self.seed = aco_seed(pol.seed, self.public_digest)
        self.rng = random.Random(self.seed)
        self.cpu0 = time.thread_time_ns()
        self.cpu_cap_ns = int(float(cfg["cpu_budget_ms"]) * 1e6)
        self.det = obs["rules"]["fire_model"] == "deterministic"
        self.fire = FireRisk(obs, g)
        self.active = dict(g.active)
        self.order = [pid for pid, _ in g.active]                           # ascending ID
        self.occupant = {p: pid for pid, p in g.active}
        self.exit_id = {p: i for i, p in g.safe_exits()}
        field = exit_distance_field(g)
        self.h = {c: d for c, (d, _) in field.items()}
        self.field = field
        self.nb: dict[Pos, tuple[tuple[str, Pos], ...]] = {}
        self.bottleneck: set[Pos] = set()
        for r in range(g.height):
            for c in range(g.width):
                u = (r, c)
                if not g.passable(u):
                    continue
                opts = tuple((a, (r + dr, c + dc)) for a, (dr, dc) in MOVES if g.passable((r + dr, c + dc)))
                self.nb[u] = opts
                if len(opts) <= 2 and u not in self.exit_id:
                    self.bottleneck.add(u)
        self.adjburn = {c: sum(n in g.burning for n in _nbrs(c)) for c in self.nb}
        self.adjburn = {c: v for c, v in self.adjburn.items() if v}
        self.pher: dict[tuple[Pos, Pos], float] = {}
        self.unseen = float(cfg["tau0"])
        self.ant_steps = self.ants = self.ants_ok = 0
        self.ant_fail: Counter = Counter()
        self.limit: str | None = None
        self.candidates: dict[str, dict[str, _Route]] = {pid: {} for pid in self.order}
        self.best: dict[str, _Route | None] = {pid: None for pid in self.order}
        self.ants_by_person: Counter = Counter()
        self.ants_ok_by_person: Counter = Counter()
        self.routable = [pid for pid in self.order if self.h.get(self.active[pid], 0) > 0]
        self.d0 = {pid: self.h[self.active[pid]] for pid in self.routable}
        self.lmax = {pid: min(cfg["max_path_length"], int(cfg["max_path_factor"] * self.d0[pid]) + cfg["max_path_extra"])
                     for pid in self.routable}
        self.bfs_path = {pid: self._bfs_path(self.active[pid]) for pid in self.routable}

    # ----- static helpers -----------------------------------------------------------
    def _bfs_path(self, src: Pos) -> tuple[Pos, ...]:
        """Static BFS descent to the nearest safe exit (N, S, E, W tie order): the labeled fallback."""
        path = [src]
        u = src
        while u not in self.exit_id:
            du = self.h[u]
            for a, v in self.nb[u]:
                if self.h.get(v) == du - 1:
                    u = v
                    break
            else:  # pragma: no cover - BFS field invariant
                raise AssertionError("exit distance field has no descending neighbour")
            path.append(u)
        return tuple(path)

    def _over_cpu(self) -> bool:
        if time.thread_time_ns() - self.cpu0 > self.cpu_cap_ns:
            self.limit = "cpu_budget_ms"
            return True
        return False

    def hazard_at(self, v: Pos, k: int) -> float:
        return self.fire.hazard(v, k - 1 if v in self.exit_id else k)

    # ----- congestion tables --------------------------------------------------------
    def _route_of(self, pid: str) -> tuple[Pos, ...] | None:
        b = self.best.get(pid)
        if b is not None:
            return b.path
        return self.bfs_path.get(pid)

    def build_load(self) -> None:
        load: Counter = Counter()
        arrivals: dict[Pos, list[int]] = {}
        self.own: dict[str, tuple[tuple[Pos, ...], Pos, int]] = {}
        for pid in self.routable:
            path = self._route_of(pid)
            if path is None:
                continue
            for k in range(1, len(path) - 1):
                load[(path[k], k)] += 1
            m = len(path) - 1
            arrivals.setdefault(path[-1], []).append(m)
            self.own[pid] = (path, path[-1], m)
        self.load = load
        self.queue: dict[Pos, tuple[list[int], list[int]]] = {}
        for e, arr in arrivals.items():
            arr.sort()
            deps, last = [], -10 ** 9
            for a in arr:
                last = max(a, last + 1)
                deps.append(last)
            self.queue[e] = (arr, deps)

    def exit_wait(self, pid: str, e: Pos, t: int) -> int:
        q = self.queue.get(e)
        if q is None:
            return 0
        arr, deps = q
        i = bisect_right(arr, t)
        if i == 0:
            return 0
        wait = max(0, deps[i - 1] + 1 - t)
        own = self.own.get(pid)
        if own is not None and own[1] == e and own[2] <= t:
            wait = max(0, wait - 1)
        return wait

    def step_cost(self, pid: str, v: Pos, k: int, z: float, own_path: tuple[Pos, ...] | None) -> float:
        cfg = self.cfg
        sc = 1.0 + cfg["w_fire"] * z
        b = self.adjburn.get(v)
        if b:
            sc += cfg["w_exposure"] * b
        if v not in self.exit_id:
            ld = self.load.get((v, k), 0)
            if ld and own_path is not None and k < len(own_path) and own_path[k] == v:
                ld -= 1
            if ld > 0:
                sc += cfg["w_congestion"] * ld
                if v in self.bottleneck:
                    sc += cfg["w_bottleneck"] * ld
        if k == 1:
            occ = self.occupant.get(v)
            if occ is not None and occ != pid:
                sc += cfg["w_occupied"]
        return sc

    def finish_cost(self, pid: str, cost: float, hmax: float, e: Pos, m: int) -> float:
        cost += self.cfg["w_queue"] * self.exit_wait(pid, e, m)
        if hmax >= self.cfg["unsafe_threshold"]:
            cost += self.cfg["w_unsafe"]
        return cost

    def score(self, pid: str, path: tuple[Pos, ...]) -> tuple[float, float] | None:
        """(cost, max hazard) of a stored path under the current tables; None if certainly blocked."""
        own = self.own.get(pid)
        own_path = own[0] if own else None
        cost, hmax = 0.0, 0.0
        for k in range(1, len(path)):
            v = path[k]
            z = self.hazard_at(v, k)
            if self.det and z >= 1.0:
                return None
            hmax = max(hmax, z)
            cost += self.step_cost(pid, v, k, z, own_path)
        return self.finish_cost(pid, cost, hmax, path[-1], len(path) - 1), hmax

    # ----- ants ---------------------------------------------------------------------
    def tau(self, u: Pos, v: Pos) -> float:
        return self.pher.get((u, v), self.unseen)

    def walk(self, pid: str, greedy: bool = False) -> _Route | None:
        cfg = self.cfg
        alpha, beta, q0 = cfg["alpha"], cfg["beta"], cfg["q0"]
        rng = self.rng
        h, nb, exit_id, det = self.h, self.nb, self.exit_id, self.det
        own = self.own.get(pid)
        own_path = own[0] if own else None
        src = self.active[pid]
        u = src
        path = [src]
        visited = {src}
        cost = hmax = 0.0
        steps = 0
        for k in range(1, self.lmax[pid] + 1):
            hu = h[u]
            opts = []
            tot = 0.0
            for a, v in nb[u]:
                if v in visited:
                    continue
                hv = h.get(v)
                if hv is None:
                    continue
                z = self.hazard_at(v, k)
                if det and z >= 1.0:
                    continue
                sc = self.step_cost(pid, v, k, z, own_path)
                wgt = self.tau(u, v) ** alpha * (1.0 / (sc + hv - hu + 1)) ** beta
                opts.append((wgt, v, sc, z))
                tot += wgt
            if not opts:
                self.ant_steps += steps
                self.ant_fail["dead_end"] += 1
                return None
            if greedy or rng.random() < q0:
                choice = opts[0]
                for o in opts[1:]:
                    if o[0] > choice[0]:
                        choice = o
            else:
                x = rng.random() * tot
                choice = opts[-1]
                acc = 0.0
                for o in opts:
                    acc += o[0]
                    if x < acc:
                        choice = o
                        break
            _, v, sc, z = choice
            steps += 1
            cost += sc
            if z > hmax:
                hmax = z
            path.append(v)
            visited.add(v)
            if v in exit_id:
                self.ant_steps += steps
                first = next(a for a, w in nb[src] if w == path[1])
                return _Route(tuple(path), self.finish_cost(pid, cost, hmax, v, k), hmax, exit_id[v], first)
            u = v
        self.ant_steps += steps
        self.ant_fail["too_long"] += 1
        return None

    def _ant_slots(self, it: int) -> list[str]:
        slots = [pid for pid in self.routable for _ in range(self.cfg["ants_per_person"])]
        cap = self.cfg["max_ants_per_iteration"]
        if len(slots) <= cap:
            return slots
        start = (it * cap) % len(slots)
        return [slots[(start + i) % len(slots)] for i in range(cap)]

    def rescore(self) -> None:
        for pid in self.routable:
            cands = self.candidates[pid]
            for a in list(cands):
                r = cands[a]
                s = self.score(pid, r.path)
                if s is None:
                    del cands[a]
                else:
                    r.cost, r.hmax = s
            self.best[pid] = min(cands.values(), key=_Route.key) if cands else None

    def offer(self, pid: str, r: _Route) -> None:
        cands = self.candidates[pid]
        cur = cands.get(r.first)
        if cur is None or r.key() < cur.key():
            cands[r.first] = r

    def deposit(self, path: tuple[Pos, ...], amount: float) -> None:
        tmax, pher, unseen = self.cfg["tau_max"], self.pher, self.unseen
        for i in range(len(path) - 1):
            e = (path[i], path[i + 1])
            pher[e] = min(tmax, pher.get(e, unseen) + amount)

    def colony(self) -> dict[str, Any]:
        cfg = self.cfg
        rho, tmin = cfg["evaporation"], cfg["tau_min"]
        mean_best: list[float | None] = []
        changed: list[int] = []
        iterations_run = 0
        self.build_load()
        stop = False
        for it in range(cfg["iterations"]):
            self.rescore()
            before = {pid: (b.path if b else None) for pid, b in self.best.items()}
            it_best: dict[str, _Route] = {}
            for pid in self._ant_slots(it):
                if self.ant_steps + self.lmax[pid] > cfg["max_ant_steps"]:
                    self.limit = "max_ant_steps"
                    stop = True
                    break
                self.ants += 1
                self.ants_by_person[pid] += 1
                r = self.walk(pid)
                if r is not None:
                    self.ants_ok += 1
                    self.ants_ok_by_person[pid] += 1
                    self.offer(pid, r)
                    if pid not in it_best or r.key() < it_best[pid].key():
                        it_best[pid] = r
                if self._over_cpu():
                    stop = True
                    break
            for pid in self.routable:
                cands = self.candidates[pid]
                self.best[pid] = min(cands.values(), key=_Route.key) if cands else None
            # evaporation on every edge (stored and unseen), then deposits
            for e in self.pher:
                self.pher[e] = max(tmin, (1 - rho) * self.pher[e])
            self.unseen = max(tmin, (1 - rho) * self.unseen)
            for pid in sorted(it_best):
                r = it_best[pid]
                self.deposit(r.path, cfg["deposit_q"] * self.d0[pid] / max(r.cost, 1e-9))
            for pid in self.routable:
                b = self.best[pid]
                if b is not None:
                    self.deposit(b.path, cfg["elite_weight"] * cfg["deposit_q"] * self.d0[pid] / max(b.cost, 1e-9))
            self.build_load()
            iterations_run += 1
            costs = [b.cost for b in self.best.values() if b is not None]
            mean_best.append(sum(costs) / len(costs) if costs else None)
            changed.append(sum((self.best[p].path if self.best[p] else None) != before[p] for p in self.routable))
            if stop:
                break
        # Final exploitation ant (q0 = 1, no draws) on the final pheromone for each routable person
        # whose ants found no route (e.g. no ant under the cap). Skipped once a budget is hit.
        greedy = greedy_ok = 0
        for pid in self.routable:
            if self.candidates[pid] or self.limit is not None:
                continue
            if self.ant_steps + self.lmax[pid] > cfg["max_ant_steps"]:
                self.limit = "max_ant_steps"
                break
            greedy += 1
            self.ants += 1
            self.ants_by_person[pid] += 1
            r = self.walk(pid, greedy=True)
            if r is not None:
                greedy_ok += 1
                self.ants_ok += 1
                self.ants_ok_by_person[pid] += 1
                self.offer(pid, r)
            if self._over_cpu():
                break
        if greedy_ok:
            self.build_load()
        self.rescore()
        return {"iterations_run": iterations_run, "iterations_planned": cfg["iterations"], "ants": self.ants,
                "final_greedy_ants": greedy, "final_greedy_ants_reached_exit": greedy_ok,
                "ants_reached_exit": self.ants_ok,
                "ant_success_rate": self.ants_ok / self.ants if self.ants else None,
                "ant_failures": dict(sorted(self.ant_fail.items())), "ant_steps": self.ant_steps,
                "pheromone_edges": len(self.pher),
                "mean_best_cost_by_iteration": [None if c is None else round(c, 6) for c in mean_best],
                "best_changed_by_iteration": changed,
                "converged_people": (len(self.routable) - changed[-1]) if changed else 0}

    # ----- joint action -------------------------------------------------------------
    def shelter_options(self, pid: str) -> list[str]:
        u = self.active[pid]
        f = self.fire

        def key(c: Pos) -> tuple:
            l = f.layer.get(c)
            return (self.hazard_at(c, 1), -(l if l is not None else 10 ** 9))
        opts = [("stay", u)] + list(self.nb.get(u, ()))
        best = min(range(len(opts)), key=lambda i: (key(opts[i][1]), i))
        ordered = [opts[best][0]] + [a for i, (a, _) in enumerate(opts) if i != best]
        return ordered

    def run(self) -> tuple[dict[str, str], dict[str, Any]]:
        colony = self.colony()
        people: dict[str, dict[str, Any]] = {}
        options: dict[str, list[str]] = {}
        for pid in self.order:
            cands = sorted(self.candidates.get(pid, {}).values(), key=_Route.key)
            info: dict[str, Any] = {"ants": self.ants_by_person[pid], "ants_reached_exit": self.ants_ok_by_person[pid],
                                    "candidates": len(cands), "fallback": False}
            if self.h.get(self.active[pid]) == 0:          # defensive: an active person on a safe exit
                info.update(route_found=True, reason="on_safe_exit", target_exit=self.exit_id[self.active[pid]],
                            route_length=0, route_cost=0.0, route_hazard_max=0.0, route_unsafe=False,
                            static_distance=0)
                options[pid] = ["stay"]
            elif cands:
                b = cands[0]
                info.update(route_found=True, reason="aco_route", target_exit=b.exit, route_length=len(b.path) - 1,
                            route_cost=round(b.cost, 6), route_hazard_max=round(b.hmax, 6),
                            route_unsafe=b.hmax >= self.cfg["unsafe_threshold"],
                            static_distance=self.d0[pid])
                options[pid] = [r.first for r in cands] + ["stay"]
            elif pid in self.d0 and not (self.det and self._all_blocked(pid)):
                path = self.bfs_path[pid]
                first = next(a for a, w in self.nb[path[0]] if w == path[1])
                info.update(route_found=True, reason="fallback_bfs_descent", fallback=True,
                            target_exit=self.exit_id[path[-1]], route_length=len(path) - 1, route_cost=None,
                            route_hazard_max=None, route_unsafe=None, static_distance=self.d0[pid])
                options[pid] = [first, "stay"]
            else:
                info.update(route_found=False,
                            reason="no_feasible_route_shelter" if pid in self.d0 else "no_route_shelter",
                            target_exit=None, route_length=None, route_cost=None, route_hazard_max=None,
                            route_unsafe=None, static_distance=self.d0.get(pid))
                options[pid] = self.shelter_options(pid)
            people[pid] = info
        actions, coord = self.coordinate(people, options)
        for pid, a in actions.items():
            people[pid]["action"] = a
        res = self.resolve(actions)
        for pid, (o, _) in res.items():
            people[pid]["predicted_outcome"] = o
        targets = Counter(d["target_exit"] for d in people.values() if d["route_found"])
        diagnostics = {
            "status": "complete" if self.limit is None else "budget_limited",
            "active": len(actions), "safe_exits": [i for i, _ in self.grid.safe_exits()],
            "colony": colony,
            "budget": {"limit_hit": self.limit, "ant_steps": self.ant_steps, "max_ant_steps": self.cfg["max_ant_steps"],
                       "cpu_budget_ms": self.cfg["cpu_budget_ms"]},
            "aco_route_people": sum(d["reason"] == "aco_route" for d in people.values()),
            "fallback_people": sorted(pid for pid, d in people.items() if d["fallback"]),
            "no_route": sorted(pid for pid, d in people.items() if not d["route_found"]),
            "unsafe_route_people": sorted(pid for pid, d in people.items() if d.get("route_unsafe")),
            "target_exit_load": dict(sorted(targets.items())),
            "coordination": coord,
            "predicted_outcomes": dict(sorted(Counter(o for o, _ in res.values()).items())),
            "people": dict(sorted(people.items())),
            "provenance": {"stream": ACO_STREAM, "teacher_seed": self.teacher_seed, "aco_seed": self.seed,
                           "public_observation_digest": self.public_digest},
        }
        return actions, diagnostics

    def _all_blocked(self, pid: str) -> bool:
        """Deterministic fire: the static BFS route is certain to burn before arrival (and no ant found another)."""
        path = self.bfs_path[pid]
        return any(self.hazard_at(path[k], k) >= 1.0 for k in range(1, len(path)))

    def resolve(self, actions: dict[str, str]) -> dict[str, tuple[str, Pos]]:
        g = self.grid
        return resolve_moves(height=g.height, width=g.width, walls=g.walls, burning=g.burning,
                             enter_fire=g.enter_fire, positions=self.active, actions=actions)

    def target(self, pid: str, a: str) -> Pos:
        r, c = self.active[pid]
        dr, dc = DELTA[a]
        return (r + dr, c + dc)

    def coordinate(self, people: dict[str, dict[str, Any]], options: dict[str, list[str]]
                   ) -> tuple[dict[str, str], dict[str, Any]]:
        routed = sorted((pid for pid in self.order if people[pid]["route_found"]),
                        key=lambda p: (people[p]["route_length"], p))
        rest = [pid for pid in self.order if not people[pid]["route_found"]]
        priority = routed + rest
        claimed: dict[Pos, str] = {}
        actions: dict[str, str] = {}
        tried: dict[str, set[str]] = {pid: set() for pid in priority}
        option_rank: dict[str, int] = {}

        def admissible(pid: str, a: str) -> bool:
            if a == "stay":
                return True
            t = self.target(pid, a)
            if t not in self.nb or t in claimed:
                return False
            occ = self.occupant.get(t)
            if occ is not None and occ in actions:
                oa = actions[occ]
                if oa == "stay" or self.target(occ, oa) == self.active[pid]:
                    return False                     # occupant stays, or a swap
            return True

        def choose(pid: str) -> None:
            for i, a in enumerate(options[pid]):
                if a in tried[pid] or not admissible(pid, a):
                    continue
                tried[pid].add(a)
                actions[pid] = a
                option_rank[pid] = i
                if a != "stay":
                    claimed[self.target(pid, a)] = pid
                return
            actions[pid] = "stay"
            option_rank[pid] = len(options[pid])

        for pid in priority:
            choose(pid)
        rounds = 0
        rank = {pid: i for i, pid in enumerate(priority)}
        for _ in range(self.cfg["repair_rounds"]):
            res = self.resolve(actions)
            blocked = sorted((pid for pid, (o, _) in res.items() if o.startswith("blocked_")), key=rank.get)
            if not blocked:
                break
            rounds += 1
            for pid in blocked:
                a = actions[pid]
                if a != "stay":
                    t = self.target(pid, a)
                    if claimed.get(t) == pid:
                        del claimed[t]
                del actions[pid]
            for pid in blocked:
                choose(pid)
        res = self.resolve(actions)
        converted = sorted(pid for pid, (o, _) in res.items() if o.startswith("blocked_"))
        if converted:
            ends = {pid: e for pid, (_, e) in res.items()}
            for pid in converted:
                actions[pid] = "stay"
                option_rank[pid] = len(options[pid])
            res2 = self.resolve(actions)
            if {pid: e for pid, (_, e) in res2.items()} != ends:  # pragma: no cover - resolver invariant
                raise AssertionError("stay conversion changed end positions")
        for pid in priority:
            people[pid]["option_rank"] = option_rank[pid]
            people[pid]["planned_first_move"] = options[pid][0]
        moves_kept = sum(actions[p] == options[p][0] and options[p][0] != "stay" for p in priority)
        return actions, {"priority": "best_route_length_then_id", "repair_rounds": rounds,
                         "converted_to_stay": converted,
                         "first_choice_moves": moves_kept,
                         "alternative_moves": sum(0 < option_rank[p] < len(options[p]) and actions[p] != "stay"
                                                  for p in priority),
                         "stays": sum(actions[p] == "stay" for p in priority)}
