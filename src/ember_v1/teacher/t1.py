"""T1: congestion-aware cooperative reference planner (issue #8).

Deterministic, CPU only, observation only. T1 is a standalone policy; it shares
`grid.PublicGrid` and the public resolver with T0, but no T0 code path.

One decision (all stages are pure functions of the validated observation):

  1 Public fire forecast (`_Fire`). Layer BFS from the burning set over
    flammable cells (not wall; exits only if `exits_flammable`). Layer l is the
    earliest spread step that can ignite the cell. Spread steps follow the public
    rule `t % spread_interval == 0`. Q(c, tau) = P(c burns after tau more steps):
      deterministic  1 if the l-th spread step is <= tau, else 0 (exact rule);
      stochastic     l = 1: 1-(1-p)^(k*n), k = burning neighbours now,
                     n = spread steps in tau; l >= 2: P(Binomial(n, p) >= l),
                     a single-front LOWER bound (optimistic, documented).
    Hazard of being on cell c at the end of relative step tau: Q(c, tau); for a
    safe exit Q(c, tau-1), because evacuation (phase 6) comes before spread.
    No RNG, seed or future fire is read; nothing is sampled (no rollouts: T2).

  2 Static per-exit distance fields: BFS from each safe exit over cells that
    are in the grid, not walls and not burning now. Other exits are absorbing
    (a person evacuates on any safe exit), so a field never passes through them.

  3 Congestion-aware exit assignment, ascending person ID. For each of the
    `candidate_exits` nearest safe exits, walk the static shortest path (fixed
    north/south/east/west descent). On congestion cells (safe exits and corridor
    cells: open cells with <= 2 open neighbours) a nominal space-time table
    holds one person per (cell, step); a later (higher-ID) person waits until a
    free step. This models exit throughput (one person per exit per step) and
    corridor queues in lowest-ID-wins order. Exit cost:
      arrival_with_delays + alt_exit_margin * [not the nearest exit]
      + risk_weight * sum(hazard along the delayed path).
    Exits with max hazard >= risk_hard are infeasible; the best feasible exit is
    chosen (ties: static distance, exit ID). If no exit is feasible, the least
    costly one is still the target and `fire_feasible` is false (no route is
    invented). The chosen path is committed to the nominal table.

  4 Windowed cooperative space-time A*, ascending person ID (planning order).
    State (cell, tau), tau <= reservation_horizon. Successors: north, south,
    east, west, stay. A successor is pruned if the cell is not open now, the
    (cell, tau) vertex is reserved by an earlier-planned person, the reverse
    edge is reserved (swap), or its hazard is >= risk_hard. Cost: 1 per step
    plus risk_weight * hazard plus move_extra_cost per move (< 1, so it never
    trades a step of time; it prefers waiting over equal-time shuffles).
    Heuristic: static distance to the assigned exit * (1 + move_extra_cost)
    (admissible). Goals: any safe exit (absorbing; +alt_exit_margin if it is
    not the assigned exit) or tau = horizon with that cost-to-go.
    Heap order (f, h, generation order): deterministic.
    A person's current cell is free to earlier planners at tau = 1, which
    allows follow chains in either ID order. Follow dependency: if an earlier
    planner reserved my cell at tau = 1, I first search for a plan that leaves
    it (the earlier planner has precedence); only if that search finds no full
    plan may I stay on my own cell at tau = 1. Step 5 verifies that every
    leader really leaves.
    Budgets: `node_budget_per_person`, `node_budget_total`. Budget hit: best
    partial path (min f, then deepest); the same path choice when the window
    has no conflict-free path to the horizon (`boxed_partial`). Total budget gone: reservation-
    consistent greedy descent. No admissible successor: least-hazard action.
    People with no static path to a safe exit get a shelter plan (move toward
    the reachable cells with the largest fire layer); `route_found` is false.

  5 Joint verification with the public resolver (`sim.movement.resolve_moves`,
    the simulator's own movement rule; the simulator resolves collisions, T1
    never overrides it). Blocked planned moves are repaired for at most
    `repair_rounds` rounds: a conflict loser, a person queued directly behind a
    stayer, or the highest ID of a closed cycle (rotation) is re-planned with
    that tau = 1 cell forbidden. Moves still predicted blocked after the last
    round become `stay`; end positions are checked to be unchanged.

What T1 does NOT do: stochastic rollouts or Monte Carlo (T2), ACO (T3),
risk-based evacuation priority (T4), learned components, cross-step memory.
"""
from __future__ import annotations

import heapq
import math
from collections import Counter, deque
from typing import Any

from ..sim.movement import resolve_moves
from .grid import PublicGrid
from .interface import ReferencePolicy

INF = 1 << 30
# Plan statuses that are complete search results (boxed_partial: no conflict-free path to the horizon exists
# inside the window; the best partial path is still a valid search result). Others are fallbacks.
FULL_PLAN_STATUSES = ("goal_exit", "window", "boxed_partial")
_DESCRIPTIVE = {
    "planning_order": "ascending_person_id",
    "fire_forecast": "public_rule_layer_bfs; deterministic exact; stochastic single_front_binomial_lower_bound",
    "congestion_cells": "safe_exits_and_corridor_cells_with_at_most_2_open_neighbours",
    "exit_cost": "nominal_space_time_arrival + alt_exit_margin*[not_nearest] + risk_weight*sum_hazard",
    "search": "windowed_space_time_astar_vertex_and_edge_reservations",
    "move_order": ["north", "south", "east", "west", "stay"],
    "search_tie_break": ["f", "h", "generation_order"],
    "no_route_mode": "shelter_max_fire_layer",
    "budget_fallback": "reservation_consistent_greedy_descent",
    "follow_dependency": "leader_vacates_if_a_full_plan_exists_else_may_stay",
    "no_successor_fallback": "least_hazard_action",
    "unrepaired_blocked_moves": "stay",
}
_NUMERIC = {
    "reservation_horizon": (int, 1, 64),
    "candidate_exits": (int, 1, 16),
    "alt_exit_margin": (float, 0.0, 1000.0),
    "risk_weight": (float, 0.0, 10000.0),
    "risk_hard": (float, 0.0, 1.0),
    "node_budget_per_person": (int, 1, 10 ** 6),
    "node_budget_total": (int, 0, 10 ** 8),
    "repair_rounds": (int, 0, 64),
    "move_extra_cost": (float, 0.0, 0.99),
}


class _Fire:
    """Fire hazard from the current burning set and the public fire rules only."""

    def __init__(self, g: "_Grid", obs: dict[str, Any]) -> None:
        rules = obs["rules"]
        self.p = 1.0 if rules["fire_model"] == "deterministic" else float(rules["spread_probability"])
        self.interval = rules["spread_interval"]
        self.t0 = obs["step"]
        self.base = self.t0 // self.interval
        a = g.area
        flam = [g.open_or_burning[i] and (rules["exits_flammable"] or not g.is_exit[i]) for i in range(a)]
        self.layer = layer = [INF] * a
        self.kburn = kburn = [0] * a
        q: deque[int] = deque()
        for i in g.burning_idx:
            layer[i] = 0
            q.append(i)
        while q:
            i = q.popleft()
            for j in g.nbrs4[i]:
                if flam[j] and layer[j] == INF:
                    layer[j] = layer[i] + 1
                    q.append(j)
                if layer[i] == 0 and layer[j] == 1:
                    kburn[j] += 1
        self._tail: dict[tuple[int, int], float] = {}

    def spreads(self, tau: int) -> int:
        return (self.t0 + tau) // self.interval - self.base

    def prob_burning(self, i: int, tau: int) -> float:
        """P(cell i burns after `tau` more steps) under the public rule (lower bound if stochastic)."""
        lay = self.layer[i]
        if lay == 0:
            return 1.0
        if lay >= INF or tau <= 0:
            return 0.0
        n = self.spreads(tau)
        if n < lay:
            return 0.0
        p = self.p
        if p >= 1.0:
            return 1.0
        if p <= 0.0:
            return 0.0
        if lay == 1:
            return 1.0 - (1.0 - p) ** (self.kburn[i] * n)
        key = (n, lay)
        v = self._tail.get(key)
        if v is None:
            v = 1.0 - sum(math.comb(n, j) * p ** j * (1.0 - p) ** (n - j) for j in range(lay))
            self._tail[key] = v = min(1.0, max(0.0, v))
        return v


class _Grid:
    """Integer-indexed view of `PublicGrid` (cell index = row * width + col)."""

    def __init__(self, pg: PublicGrid) -> None:
        h, w = pg.height, pg.width
        self.height, self.width, self.area = h, w, h * w
        a = self.area
        walls = {r * w + c for r, c in pg.walls}
        self.burning_idx = sorted(r * w + c for r, c in pg.burning)
        burning = set(self.burning_idx)
        self.open = [i not in walls and i not in burning for i in range(a)]
        self.open_or_burning = [i not in walls for i in range(a)]
        self.is_exit = [False] * a
        for _, (r, c) in pg.exits:
            self.is_exit[r * w + c] = True
        self.safe_exits = [(eid, r * w + c) for eid, (r, c) in pg.safe_exits()]
        self.safe_exit = [False] * a
        for _, i in self.safe_exits:
            self.safe_exit[i] = True
        self.nbrs4: list[tuple[int, ...]] = []
        self.succ: list[tuple[tuple[str, int], ...]] = []
        for i in range(a):
            r, c = divmod(i, w)
            cand = (("north", r - 1, c), ("south", r + 1, c), ("east", r, c + 1), ("west", r, c - 1))
            inside = [(act, rr * w + cc) for act, rr, cc in cand if 0 <= rr < h and 0 <= cc < w]
            self.nbrs4.append(tuple(j for _, j in inside))
            self.succ.append(tuple((act, j) for act, j in inside if self.open[j]))
        self.congest = [self.safe_exit[i] or (self.open[i] and len(self.succ[i]) <= 2) for i in range(a)]

    def rc(self, i: int) -> tuple[int, int]:
        return divmod(i, self.width)


class T1CongestionAware(ReferencePolicy):
    name = "T1"
    policy = "congestion_aware_cooperative_planner"
    version = "1.0.0"

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        # Frozen preregistered configuration (docs/P1_T1_PLANNER.md section 3).
        return {**_DESCRIPTIVE, "reservation_horizon": 8, "candidate_exits": 3, "alt_exit_margin": 2.0,
                "risk_weight": 20.0, "risk_hard": 0.5, "node_budget_per_person": 256,
                "node_budget_total": 16000, "repair_rounds": 6, "move_extra_cost": 0.25}

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config, seed=None)
        cfg = self.identity.config
        for key, value in _DESCRIPTIVE.items():
            if cfg[key] != value:
                raise ValueError(f"T1: {key} is not configurable")
        for key, (typ, lo, hi) in _NUMERIC.items():
            v = cfg[key]
            ok = (type(v) is int) if typ is int else (type(v) in (int, float) and math.isfinite(v))
            if not ok or not lo <= v <= hi:
                raise ValueError(f"T1: {key} must be {typ.__name__} in [{lo}, {hi}]")

    def _plan(self, obs: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
        return _Decision(self.identity.config, obs).run()


class _Decision:
    def __init__(self, cfg: dict[str, Any], obs: dict[str, Any]) -> None:
        self.cfg = cfg
        self.pg = PublicGrid.from_observation(obs)
        self.g = _Grid(self.pg)
        self.fire = _Fire(self.g, obs)
        self.W = cfg["reservation_horizon"]
        self.hard = float(cfg["risk_hard"])
        self.rw = float(cfg["risk_weight"])
        self.margin = float(cfg["alt_exit_margin"])
        w = self.g.width
        self.order = [pid for pid, _ in self.pg.active]                   # ascending person ID
        self.pos = {pid: r * w + c for pid, (r, c) in self.pg.active}
        self._hz: dict[tuple[int, int], float] = {}
        f = self.fire                       # first relative step with a possibly nonzero hazard
        self.hz_from = [INF if lay >= INF else 0 if lay == 0 else
                        (f.base + lay) * f.interval - f.t0 + (1 if self.g.safe_exit[i] else 0)
                        for i, lay in enumerate(f.layer)]
        self.vres: dict[int, str] = {}       # cell*(W+1)+tau -> pid
        self.eres: dict[int, str] = {}       # (from*A+to)*(W+1)+tau -> pid
        self.keys: dict[str, list[tuple[int, int]]] = {}
        self.plans: dict[str, list[int]] = {}
        self.info: dict[str, dict[str, Any]] = {}
        self.forbid: dict[str, set[int]] = {}
        self.budget_left = cfg["node_budget_total"]
        self.budget_exhausted = False
        self._own_stay = True

    # ----- hazard --------------------------------------------------------------
    def hazard(self, i: int, tau: int) -> float:
        if tau < self.hz_from[i]:
            return 0.0
        key = (i, tau)
        v = self._hz.get(key)
        if v is None:
            v = self.fire.prob_burning(i, tau - 1 if self.g.safe_exit[i] else tau)
            self._hz[key] = v
        return v

    # ----- stage 2: per-exit fields -------------------------------------------
    def exit_fields(self) -> dict[str, list[int]]:
        g = self.g
        fields: dict[str, list[int]] = {}
        for eid, e in g.safe_exits:
            d = [INF] * g.area
            d[e] = 0
            q = deque([e])
            while q:
                i = q.popleft()
                di = d[i] + 1
                for _, j in g.succ[i]:
                    if d[j] == INF and not g.safe_exit[j]:
                        d[j] = di
                        q.append(j)
            fields[eid] = d
        return fields

    # ----- stage 3: congestion-aware exit assignment ----------------------------
    def assign_exits(self, fields: dict[str, list[int]]) -> None:
        g, k = self.g, self.cfg["candidate_exits"]
        used: set[tuple[int, int]] = set()
        nxt_cache: dict[str, dict[int, int]] = {eid: {} for eid in fields}
        for pid in self.order:
            s = self.pos[pid]
            cands = sorted((d[s], eid) for eid, d in fields.items() if d[s] < INF)[:k]
            if not cands:
                self.info[pid] = {"mode": "shelter", "route_found": False, "reason": "no_safe_exit_reachable",
                                  "target_exit": None, "static_distance": None}
                continue
            nearest = cands[0][1]
            best = None
            for dist, eid in cands:
                arrival, delay, risk, maxq, marks = self._nominal(s, fields[eid], nxt_cache[eid], used)
                cost = arrival + (self.margin if eid != nearest else 0.0) + self.rw * risk
                key = (maxq >= self.hard, cost, dist, eid)
                if best is None or key < best[0]:
                    best = (key, eid, dist, arrival, delay, maxq, marks)
            (infeasible, cost, _, _), eid, dist, arrival, delay, maxq, marks = best
            used.update(marks)
            self.info[pid] = {"mode": "evacuate", "route_found": True,
                              "reason": "assigned_exit" if not infeasible else "fire_infeasible_best_effort",
                              "target_exit": eid, "nearest_exit": nearest, "static_distance": dist,
                              "nominal_arrival": arrival, "nominal_delay": delay, "exit_cost": round(cost, 6),
                              "fire_feasible": not infeasible}

    def _nominal(self, s: int, d: list[int], nxt: dict[int, int], used: set[tuple[int, int]]):
        g, cur, tau, risk, maxq, delay, marks = self.g, s, 0, 0.0, 0.0, 0, []
        hz, hz_from, congest, succ = self.hazard, self.hz_from, g.congest, g.succ
        while d[cur] > 0:
            n = nxt.get(cur)
            if n is None:
                want = d[cur] - 1
                n = next(j for _, j in succ[cur] if d[j] == want)
                nxt[cur] = n
            tau += 1
            if congest[n]:
                while (n, tau) in used:
                    if tau >= hz_from[cur]:
                        q = hz(cur, tau)
                        risk += q
                        if q > maxq:
                            maxq = q
                    tau += 1
                    delay += 1
                marks.append((n, tau))
            if tau >= hz_from[n]:
                q = hz(n, tau)
                risk += q
                if q > maxq:
                    maxq = q
            cur = n
        return tau, delay, risk, maxq, marks

    # ----- shelter fields for people with no static exit path --------------------
    def shelter_field(self, s: int, cache: dict[int, list[int]], label: list[int]) -> list[int]:
        if label[s] >= 0:
            return cache[label[s]]
        g, lay = self.g, self.fire.layer
        comp, q = [s], deque([s])
        label[s] = s
        while q:
            i = q.popleft()
            for _, j in g.succ[i]:
                if label[j] < 0:
                    label[j] = s
                    comp.append(j)
                    q.append(j)
        top = max(lay[i] for i in comp)
        d = [INF] * g.area
        q = deque(i for i in sorted(comp) if lay[i] == top)
        for i in q:
            d[i] = 0
        while q:
            i = q.popleft()
            for _, j in g.succ[i]:
                if d[j] == INF and label[j] == s:
                    d[j] = d[i] + 1
                    q.append(j)
        cache[s] = d
        return d

    # ----- stage 4: windowed cooperative A* ---------------------------------------
    def _blocked(self, pid: str, s: int, c: int, n: int, t1: int) -> bool:
        W1, a = self.W + 1, self.g.area
        if not (t1 == 1 and n == s == c and self._own_stay):  # own-cell stay at tau=1 (if not yielding)
            owner = self.vres.get(n * W1 + t1)
            if owner is not None and owner != pid:
                return True
        if n != c:
            owner = self.eres.get((n * a + c) * W1 + t1)
            if owner is not None and owner != pid:
                return True
        return t1 == 1 and n in self.forbid.get(pid, ())

    def search(self, pid: str, s: int, h: list[int], target: int | None) -> tuple[list[int] | None, str, int]:
        g, W, hard, rw, margin = self.g, self.W, self.hard, self.rw, self.margin
        budget = min(self.cfg["node_budget_per_person"], self.budget_left)
        W1 = W + 1
        mx = float(self.cfg["move_extra_cost"])
        hs = 1.0 + mx                       # each remaining cell needs one move: h is admissible
        heap = [(h[s] * hs, h[s] * hs, 0, s, 0)]
        gcost = {s * W1: 0.0}
        parent: dict[int, int] = {}
        closed: set[int] = set()
        cnt, expanded, best = 0, 0, None
        succ, safe_exit, hazard, blocked = g.succ, g.safe_exit, self.hazard, self._blocked
        goal = None
        while heap and expanded < budget:
            f, hh, _, c, tau = heapq.heappop(heap)
            key = c * W1 + tau
            if key in closed:
                continue
            closed.add(key)
            expanded += 1
            if tau == W or (tau > 0 and safe_exit[c]):
                goal = key
                break
            if tau > 0 and (best is None or (f, -tau) < best[0]):
                best = ((f, -tau), key)
            gc = gcost[key]
            t1 = tau + 1
            for _, n in succ[c] + (("stay", c),):
                if blocked(pid, s, c, n, t1):
                    continue
                q = hazard(n, t1)
                if q >= hard:
                    continue
                if safe_exit[n]:
                    hn = 0 if n == target else margin
                else:
                    hn = h[n]
                    if hn >= INF:
                        continue
                    hn *= hs
                nk = n * W1 + t1
                ng = gc + 1.0 + rw * q + (mx if n != c else 0.0)
                if nk in closed or ng >= gcost.get(nk, math.inf):
                    continue
                gcost[nk] = ng
                parent[nk] = key
                cnt += 1
                heapq.heappush(heap, (ng + hn, hn, cnt, n, t1))
        self.budget_left -= expanded
        if self.budget_left <= 0:
            self.budget_exhausted = True
        if goal is not None:
            status = "goal_exit" if safe_exit[goal // W1] else "window"
        elif best is not None:
            goal, status = best[1], "budget_partial" if expanded >= budget else "boxed_partial"
        else:
            return None, "no_successor", expanded
        path = [goal]
        while path[-1] in parent:
            path.append(parent[path[-1]])
        return [k // W1 for k in reversed(path)], status, expanded

    def greedy(self, pid: str, s: int, h: list[int], target: int | None) -> list[int] | None:
        g = self.g
        opts = []
        for rank, (_, n) in enumerate(g.succ[s] + (("stay", s),)):
            hn = -1 if g.safe_exit[n] else h[n]
            opts.append((hn, rank, n))
        for hn, _, n in sorted(opts):
            if hn < INF and not self._blocked(pid, s, s, n, 1) and self.hazard(n, 1) < self.hard:
                return [s, n]
        return None

    def least_hazard(self, s: int) -> list[int]:
        g = self.g
        opts = [(self.hazard(n, 1), rank, n) for rank, (_, n) in enumerate(g.succ[s] + (("stay", s),))]
        return [s, min(opts)[2]]

    def reserve(self, pid: str, path: list[int], hold: bool) -> None:
        W1, a, W = self.W + 1, self.g.area, self.W
        keys = self.keys.setdefault(pid, [])
        for tau in range(1, len(path)):
            c, n = path[tau - 1], path[tau]
            k = n * W1 + tau
            self.vres[k] = pid
            keys.append((0, k))
            if n != c:
                e = (c * a + n) * W1 + tau
                self.eres[e] = pid
                keys.append((1, e))
        last = path[-1]
        if hold and not self.g.safe_exit[last]:
            for tau in range(len(path), W + 1):
                k = last * W1 + tau
                if k not in self.vres:
                    self.vres[k] = pid
                    keys.append((0, k))

    def unreserve(self, pid: str) -> None:
        for kind, k in self.keys.pop(pid, []):
            table = self.vres if kind == 0 else self.eres
            if table.get(k) == pid:
                del table[k]

    def plan_one(self, pid: str) -> None:
        s, info = self.pos[pid], self.info[pid]
        h = self.hfield[pid]
        target = self.targets.get(pid)
        info["yielded"] = False
        path, status, exp, hold = None, "", 0, True
        owner = self.vres.get(s * (self.W + 1) + 1)
        if self.budget_left > 0 and owner is not None and owner != pid:
            self._own_stay = False                           # follow dependency: try to vacate first
            try:
                path, status, exp = self.search(pid, s, h, target)
            finally:
                self._own_stay = True
            if status in ("goal_exit", "window"):
                info["yielded"] = True
            else:
                info["expansions"] = info.get("expansions", 0) + exp
                path, exp = None, 0
        if path is None:
            if self.budget_left <= 0:
                self.budget_exhausted = True
                path, status, exp, hold = self.greedy(pid, s, h, target), "budget_fallback", 0, False
            else:
                path, status, exp = self.search(pid, s, h, target)
                if path is None:
                    path, hold = self.greedy(pid, s, h, target), False
                    status = "no_successor_greedy" if path is not None else status
        if path is None:
            path, status, hold = self.least_hazard(s), "no_successor_least_hazard", False
        info["plan_status"] = status
        info["expansions"] = info.get("expansions", 0) + exp
        self.plans[pid] = path
        self.reserve(pid, path, hold)

    # ----- stage 5: joint verification and repair -------------------------------
    def actions(self) -> dict[str, str]:
        out = {}
        for pid, path in self.plans.items():
            s, n = path[0], path[1] if len(path) > 1 else path[0]
            out[pid] = "stay" if n == s else next(a for a, j in self.g.succ[s] if j == n)
        return out

    def resolve(self, actions: dict[str, str]) -> dict[str, tuple[str, tuple[int, int]]]:
        pg = self.pg
        return resolve_moves(height=pg.height, width=pg.width, walls=pg.walls, burning=pg.burning,
                             enter_fire=pg.enter_fire, positions=dict(pg.active), actions=actions)

    def repair_roots(self, actions: dict[str, str], res: dict[str, tuple[str, Any]]) -> list[str]:
        occupant = {i: pid for pid, i in self.pos.items()}
        tgt = {pid: self.plans[pid][1] for pid, a in actions.items() if a != "stay"}
        roots: set[str] = set()
        for pid, t in tgt.items():
            o = res[pid][0]
            if o == "moved":
                continue
            if o == "blocked_cycle":
                cyc, cur = [pid], occupant.get(t)
                while cur is not None and cur != pid and cur in tgt and len(cyc) <= len(tgt):
                    cyc.append(cur)
                    cur = occupant.get(tgt[cur])
                roots.add(max(cyc))
            elif o == "blocked_occupied":
                occ = occupant.get(t)
                if occ is None or actions[occ] == "stay":
                    roots.add(pid)
            else:
                roots.add(pid)
        return sorted(roots)

    def run(self) -> tuple[dict[str, str], dict[str, Any]]:
        fields = self.exit_fields()
        self.assign_exits(fields)
        exit_idx = dict(self.g.safe_exits)
        self.hfield: dict[str, list[int]] = {}
        self.targets: dict[str, int | None] = {}
        cache: dict[int, list[int]] = {}
        label = [-1] * self.g.area
        for pid in self.order:
            info = self.info[pid]
            if info["route_found"]:
                self.hfield[pid] = fields[info["target_exit"]]
                self.targets[pid] = exit_idx[info["target_exit"]]
            else:
                self.hfield[pid] = self.shelter_field(self.pos[pid], cache, label)
                self.targets[pid] = None
        for pid in self.order:                                   # planning order: ascending ID
            self.plan_one(pid)
        rounds = 0
        repaired: Counter = Counter()
        for _ in range(self.cfg["repair_rounds"]):
            acts = self.actions()
            roots = self.repair_roots(acts, self.resolve(acts))
            if not roots:
                break
            rounds += 1
            for pid in roots:
                self.forbid.setdefault(pid, set()).add(self.plans[pid][1])
                self.unreserve(pid)
            for pid in roots:
                repaired[pid] += 1
                self.plan_one(pid)
        acts = self.actions()
        res = self.resolve(acts)
        converted = sorted(pid for pid, a in acts.items() if a != "stay" and res[pid][0] != "moved")
        if converted:
            final = {**acts, **{pid: "stay" for pid in converted}}
            res2 = self.resolve(final)
            if any(res2[p][1] != res[p][1] for p in res):        # construction invariant
                raise AssertionError("converting blocked moves to stay changed end positions")
            acts, res = final, res2
        people: dict[str, dict[str, Any]] = {}
        for pid in self.order:
            info = self.info[pid]
            nxt = self.plans[pid][1] if len(self.plans[pid]) > 1 else self.pos[pid]
            if pid in converted:
                nxt = self.pos[pid]
            people[pid] = {**info, "action": acts[pid], "repairs": repaired[pid],
                           "converted_to_stay": pid in converted, "predicted_outcome": res[pid][0],
                           "next_hazard": round(self.hazard(nxt, 1), 6)}
        fallbacks = Counter(p["plan_status"] for p in people.values())
        load = Counter(p["target_exit"] for p in people.values() if p["route_found"])
        diagnostics = {
            "active": len(acts),
            "safe_exits": [eid for eid, _ in self.g.safe_exits],
            "no_route": sorted(pid for pid, p in people.items() if not p["route_found"]),
            "fire_infeasible": sorted(pid for pid, p in people.items() if p["route_found"] and not p["fire_feasible"]),
            "target_exit_load": dict(sorted(load.items())),
            "plan_status": dict(sorted(fallbacks.items())),
            "expansions": self.cfg["node_budget_total"] - self.budget_left,
            "budget_exhausted": self.budget_exhausted,
            "repair_rounds": rounds,
            "converted_to_stay": converted,
            "planning_fallbacks": sum(v for k, v in fallbacks.items() if k not in FULL_PLAN_STATUSES),
            "predicted_outcomes": dict(sorted(Counter(o for o, _ in res.values()).items())),
            "people": dict(sorted(people.items())),
        }
        return acts, diagnostics
