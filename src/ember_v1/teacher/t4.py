"""T4: risk-based evacuation priority reference agent and its A0/A1 ablations (issue #24).

Deterministic, CPU only, observation only, stateless between decisions.
Design and preregistered protocol: docs/P1_T4_RISK_PRIORITY.md.

Three policies share ONE movement planner (`_PriorityPlanner`). They differ
only in the planning order and the priority class (`priority_scheme`):
  T4     (A2) "risk"           strata, then urgency from `t4_risk.score`;
  T4-A0       "none"           ascending person ID (neutral; the resolver's own order);
  T4-A1       "fire_distance"  current fire layer distance only.
Everything else (fire forecast, exit viability, exit assignment, space-time
reservations, yielding, repair, fairness controls, budgets, fallback) is the
same code with the same frozen numbers.

One decision:
  1 `t4_risk.assess`: public fire forecast, per-exit static and latest-departure
    fields, per-person signals (urgency and stratum for A2 only).
  2 Priority: planning order and class from the scheme.
  3 Priority-aware exit assignment, planning order. Candidates: the
    `candidate_exits` nearest exits that are viable under T_late, within
    `max_detour` steps of the nearest viable one (fairness control F1). A
    nominal descent on each exit field holds one person per (narrow cell, step);
    later (lower-priority) people wait in the table, so their cost grows and
    they are diverted to other exits (alternative routing for lower priority).
    Cost = arrival + alt_exit_margin*[not nearest viable] + risk_weight*sum hazard
    + deadline_weight*max(0, delay - L_early). Infeasible if delay > L_late.
    No viable exit: shelter mode (toward the reachable cells with the latest
    T_late), labeled.
  4 Windowed cooperative space-time A* in planning order with vertex and edge
    (swap) reservations. An earlier planner may reserve a later person's
    current cell at tau = 1. The later person must then VACATE (deliberate
    yield): it searches for a full plan that leaves its cell. A protected person
    (class "critical", fairness control F2) may vacate only by a progress move.
    If no such plan exists, it stays; the earlier planner is then repaired.
    Lower-priority people never take a (cell, tau) an earlier planner holds,
    so the endangered person's first move is not contested at the resolver.
  5 Verification with the public resolver (`sim.movement.resolve_moves`; T4
    never overrides lowest-ID collision resolution). Priority-aware repair for
    `repair_rounds` rounds: a cycle re-plans its lowest-priority member; a
    contest whose ID winner has lower priority than a loser re-plans the
    winner (it yields the cell); else the blocked mover. Moves still predicted
    blocked become `stay` (end positions asserted unchanged).
  6 Priority accounting per person: intended move (static descent), actual
    action, predicted outcome, and whether priority made the person move,
    wait, step aside or reroute, and for whom.
"""
from __future__ import annotations

import heapq
import math
import time
from collections import Counter, deque
from typing import Any, ClassVar

from ..sim.movement import resolve_moves
from .grid import PublicGrid
from .interface import ReferencePolicy
from .t4_risk import COMPONENTS, INF, RISK_MODEL_VERSION, STRATA, assess, public_view

PLANNER_VERSION = "t4-priority-planner@1.0.0"
SCHEMES = ("none", "fire_distance", "risk")
FULL_PLAN_STATUSES = ("goal_exit", "window", "boxed_partial")
FALLBACK_STATUSES = ("budget_partial", "cpu_budget_fallback", "node_budget_fallback", "no_successor_greedy",
                     "no_successor_least_hazard")
PRIORITY_EFFECTS = ("moved_as_intended", "moved_alternative", "waited", "yielded_wait", "yielded_step_aside",
                    "yielded_reroute", "vacate_refused", "converted_to_stay", "sheltering")
YIELD_EFFECTS = ("yielded_wait", "yielded_step_aside", "yielded_reroute")

_DESCRIPTIVE = {
    "risk_model": RISK_MODEL_VERSION,
    "planner": PLANNER_VERSION,
    "fire_forecast": "public_rule_layer_bfs; deterministic exact; stochastic l1 exact, l>=2 single_front_binomial_lower_bound",
    "arrival_estimate": "first_relative_step_with_P_burn_ge_q; CAP_after_episode_end",
    "deadline_field": "per_exit_latest_departure_max_min_dijkstra",
    "urgency": "weighted_sum_of_bounded_components(margin,exposure,proximity,options,congestion,evacuation)",
    "margin": "latest_departure_slack_early_minus_fifo_exit_congestion_delay",
    "strata": "no_viable_route if no exit viable under T_late; critical if slack_early<0 or margin<stratum_critical; "
              "high, elevated, low by margin thresholds",
    "risk_order_tie_break": ["stratum", "-urgency", "nominal_distance", "row", "col"],
    "fire_distance_order_tie_break": ["no_static_route_last", "fire_layer", "static_distance", "row", "col"],
    "none_order": "ascending_person_id",
    "exit_assignment": "priority_order_nominal_descent_one_person_per_narrow_cell_step",
    "search": "windowed_space_time_astar_vertex_and_edge_reservations_in_priority_order",
    "move_order": ["north", "south", "east", "west", "stay"],
    "search_tie_break": ["f", "h", "generation_order"],
    "yield_rule": "later_planner_vacates_cell_reserved_at_tau1_by_earlier_planner_if_full_plan_exists",
    "fairness": "F1 max_detour cap; F2 critical class vacates only by progress moves; F3 stateless (no aging)",
    "repair": "cycle_lowest_priority_member; contest_winner_if_lower_priority_than_a_loser; else_blocked_mover",
    "no_route_mode": "shelter_toward_latest_T_late_cells",
    "budget_fallback": "reservation_consistent_greedy_descent",
    "no_successor_fallback": "least_hazard_action",
    "unrepaired_blocked_moves": "stay",
    "memory": "stateless_per_decision",
}
# Frozen preregistered numbers (docs/P1_T4_RISK_PRIORITY.md section 4), shared by T4, T4-A0 and T4-A1.
_FROZEN = {
    "q_early": 0.25, "q_late": 0.75,
    "w_margin": 0.35, "w_exposure": 0.20, "w_proximity": 0.10, "w_options": 0.15, "w_congestion": 0.10,
    "w_evacuation": 0.10,
    "scale_margin": 6.0, "scale_exposure": 8.0, "scale_proximity": 4.0, "scale_congestion": 8.0,
    "scale_evacuation": 16.0, "urgency_digits": 6,
    "stratum_critical": 3.0, "stratum_high": 10.0, "stratum_elevated": 25.0,
    "band_critical": 3, "band_high": 8, "band_elevated": 16,
    "reservation_horizon": 8, "candidate_exits": 3, "alt_exit_margin": 2.0, "max_detour": 12,
    "risk_weight": 20.0, "risk_hard": 0.5, "deadline_weight": 4.0, "move_extra_cost": 0.25,
    "node_budget_per_person": 256, "node_budget_total": 20000, "repair_rounds": 6, "cpu_budget_ms": 2000.0,
}
_NUMERIC = {
    "q_early": (float, 0.0, 1.0), "q_late": (float, 0.0, 1.0),
    **{f"w_{k}": (float, 0.0, 1.0) for k in COMPONENTS},
    **{k: (float, 0.001, 1e6) for k in ("scale_margin", "scale_exposure", "scale_proximity", "scale_congestion",
                                        "scale_evacuation")},
    "urgency_digits": (int, 1, 12),
    "stratum_critical": (float, -1e6, 1e6), "stratum_high": (float, -1e6, 1e6), "stratum_elevated": (float, -1e6, 1e6),
    "band_critical": (int, 0, 10 ** 6), "band_high": (int, 0, 10 ** 6), "band_elevated": (int, 0, 10 ** 6),
    "reservation_horizon": (int, 1, 64), "candidate_exits": (int, 1, 16), "alt_exit_margin": (float, 0.0, 1000.0),
    "max_detour": (int, 0, 10 ** 6), "risk_weight": (float, 0.0, 10000.0), "risk_hard": (float, 0.0, 1.0),
    "deadline_weight": (float, 0.0, 10000.0), "move_extra_cost": (float, 0.0, 0.99),
    "node_budget_per_person": (int, 1, 10 ** 6), "node_budget_total": (int, 0, 10 ** 8),
    "repair_rounds": (int, 0, 64), "cpu_budget_ms": (float, 0.0, 10 ** 6),
}


class _T4Base(ReferencePolicy):
    scheme: ClassVar[str]
    version = "1.0.0"

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return {**_DESCRIPTIVE, "priority_scheme": cls.scheme, **_FROZEN}

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config, seed=None)
        cfg = self.identity.config
        for key, value in {**_DESCRIPTIVE, "priority_scheme": self.scheme}.items():
            if cfg[key] != value:
                raise ValueError(f"{self.name}: {key} is not configurable")
        for key, (typ, lo, hi) in _NUMERIC.items():
            v = cfg[key]
            ok = (type(v) is int) if typ is int else (type(v) in (int, float) and math.isfinite(v))
            if not ok or not lo <= v <= hi:
                raise ValueError(f"{self.name}: {key} must be {typ.__name__} in [{lo}, {hi}]")
        if abs(sum(cfg[f"w_{k}"] for k in COMPONENTS) - 1.0) > 1e-9:
            raise ValueError(f"{self.name}: urgency weights must sum to 1")
        if not cfg["q_early"] <= cfg["q_late"]:
            raise ValueError(f"{self.name}: q_early must be <= q_late")
        if not cfg["stratum_critical"] <= cfg["stratum_high"] <= cfg["stratum_elevated"]:
            raise ValueError(f"{self.name}: stratum thresholds must be ordered")
        if not cfg["band_critical"] <= cfg["band_high"] <= cfg["band_elevated"]:
            raise ValueError(f"{self.name}: distance bands must be ordered")

    def _plan(self, obs: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
        return _PriorityPlanner(self.identity.config, obs, self.scheme).run()


class T4RiskPriority(_T4Base):
    """A2: full preregistered risk-based urgency priority."""
    name = "T4"
    policy = "risk_based_priority"
    scheme = "risk"


class T4NoPriority(_T4Base):
    """A0 ablation: neutral planning order (ascending person ID), no risk-based urgency."""
    name = "T4-A0"
    policy = "risk_priority_ablation_no_priority"
    scheme = "none"


class T4DistancePriority(_T4Base):
    """A1 ablation: priority by current fire distance only."""
    name = "T4-A1"
    policy = "risk_priority_ablation_fire_distance"
    scheme = "fire_distance"


ABLATIONS = {"A0": T4NoPriority, "A1": T4DistancePriority, "A2": T4RiskPriority}


class _PriorityPlanner:
    def __init__(self, cfg: dict[str, Any], obs: dict[str, Any], scheme: str) -> None:
        self.cpu0 = time.thread_time_ns()
        self.cfg, self.scheme = cfg, scheme
        self.pg = PublicGrid.from_observation(obs)
        self.ra = assess(self.pg, obs, cfg, urgency=scheme == "risk")
        self.g, self.fc, self.pos, self.risk = self.ra.cells, self.ra.forecast, self.ra.pos, self.ra.people
        self.W = cfg["reservation_horizon"]
        self.hard, self.rw = float(cfg["risk_hard"]), float(cfg["risk_weight"])
        self.margin = float(cfg["alt_exit_margin"])
        self.mx = float(cfg["move_extra_cost"])
        self.cpu_budget_ns = float(cfg["cpu_budget_ms"]) * 1e6
        fc, g = self.fc, self.g
        self.hz_from = [INF if lay >= INF else 0 if lay == 0 else fc.tau_of(lay) + (1 if g.safe_exit[i] else 0)
                        for i, lay in enumerate(fc.layer)]
        self._hz: dict[tuple[int, int], float] = {}
        self.vres: dict[int, str] = {}
        self.eres: dict[int, str] = {}
        self.keys: dict[str, list[tuple[int, int]]] = {}
        self.plans: dict[str, list[int]] = {}
        self.info: dict[str, dict[str, Any]] = {}
        self.forbid: dict[str, set[int]] = {}
        self.budget_left = cfg["node_budget_total"]
        self.limit_hit: str | None = None
        self._own_stay = True
        self._progress_only: list[int] | None = None
        self.resolver_calls = 0
        self._prioritize()

    # ----- priority ------------------------------------------------------------------------
    def _prioritize(self) -> None:
        cfg, risk, w = self.cfg, self.risk, self.g.width
        pids = sorted(self.pos)
        if self.scheme == "none":
            order = pids
            cls = {pid: "neutral" for pid in pids}
        elif self.scheme == "fire_distance":
            def key(pid: str) -> tuple:
                r = risk[pid]
                lay, d = r["fire_layer"], r["static_distance"]
                return (d is None, INF if lay is None else lay, INF if d is None else d, *divmod(self.pos[pid], w))
            order = sorted(pids, key=key)
            cls = {}
            for pid in pids:
                lay = risk[pid]["fire_layer"]
                cls[pid] = ("no_static_route" if risk[pid]["static_distance"] is None else
                            "low" if lay is None or lay > cfg["band_elevated"] else
                            "critical" if lay <= cfg["band_critical"] else
                            "high" if lay <= cfg["band_high"] else "elevated")
        elif self.scheme == "risk":
            def key(pid: str) -> tuple:
                r = risk[pid]
                d = r["nominal_distance"] if r["nominal_distance"] is not None else r["static_distance"]
                return (STRATA.index(r["stratum"]), -r["urgency"], INF if d is None else d,
                        *divmod(self.pos[pid], w))
            order = sorted(pids, key=key)
            cls = {pid: risk[pid]["stratum"] for pid in pids}
        else:
            raise ValueError(f"unknown priority scheme {self.scheme!r}")
        self.order = order
        self.rank = {pid: i for i, pid in enumerate(order)}
        self.pclass = cls
        self.protected = {pid: cls[pid] == "critical" for pid in pids}

    # ----- hazard --------------------------------------------------------------------------
    def hazard(self, i: int, tau: int) -> float:
        if tau < self.hz_from[i]:
            return 0.0
        key = (i, tau)
        v = self._hz.get(key)
        if v is None:
            v = self.fc.prob(i, tau - 1 if self.g.safe_exit[i] else tau)
            self._hz[key] = v
        return v

    # ----- stage 3: priority-aware exit assignment -----------------------------------------------
    def _nominal(self, s: int, d: list[int], nxt: dict[int, int], used: set[tuple[int, int]]):
        g, cur, tau, risk, maxq, delay, marks = self.g, s, 0, 0.0, 0.0, 0, []
        hz, hz_from, narrow, succ = self.hazard, self.hz_from, g.narrow, g.succ
        while d[cur] > 0:
            n = nxt.get(cur)
            if n is None:
                want = d[cur] - 1
                n = next(j for _, j in succ[cur] if d[j] == want)
                nxt[cur] = n
            tau += 1
            if narrow[n]:
                while (n, tau) in used:
                    if tau >= hz_from[cur]:
                        q = hz(cur, tau)
                        risk += q
                        maxq = max(maxq, q)
                    tau += 1
                    delay += 1
                marks.append((n, tau))
            if tau >= hz_from[n]:
                q = hz(n, tau)
                risk += q
                maxq = max(maxq, q)
            cur = n
        return tau, delay, risk, maxq, marks

    def assign_exits(self) -> None:
        cfg, fields = self.cfg, self.ra.fields
        used: set[tuple[int, int]] = set()
        nxt_cache: dict[str, dict[int, int]] = {eid: {} for eid in fields}
        for pid in self.order:
            s, r = self.pos[pid], self.risk[pid]
            viable = sorted((fields[e].dist[s], e) for e in r["viable_exits"])
            if not viable:
                self.info[pid] = {"mode": "shelter", "route_found": False, "target_exit": None,
                                  "nearest_exit": None, "diverted": False, "nominal_delay": None,
                                  "reason": "no_static_route" if r["static_distance"] is None
                                  else "no_viable_exit_under_forecast"}
                continue
            d0, nearest = viable[0]
            cands = [(dist, e) for dist, e in viable if dist <= d0 + cfg["max_detour"]][:cfg["candidate_exits"]]
            best = None
            for dist, eid in cands:
                f = fields[eid]
                arrival, delay, hz_sum, maxq, marks = self._nominal(s, f.dist, nxt_cache[eid], used)
                slack_e = f.slack_early[s] if f.slack_early[s] > -INF else -1
                over = max(0, delay - slack_e)
                cost = (arrival + (self.margin if eid != nearest else 0.0) + self.rw * hz_sum
                        + cfg["deadline_weight"] * over)
                key = (delay > f.slack_late[s] or maxq >= self.hard, cost, dist, eid)
                if best is None or key < best[0]:
                    best = (key, eid, dist, arrival, delay, marks)
            (infeasible, cost, _, _), eid, dist, arrival, delay, marks = best
            used.update(marks)
            self.info[pid] = {"mode": "evacuate", "route_found": True, "target_exit": eid, "nearest_exit": nearest,
                              "diverted": eid != nearest, "static_distance_target": dist,
                              "nominal_arrival": arrival, "nominal_delay": delay, "exit_cost": round(cost, 6),
                              "forecast_feasible": not infeasible,
                              "reason": "assigned_exit" if not infeasible else "forecast_infeasible_best_effort"}

    def shelter_field(self, s: int, cache: dict[int, list[int]], label: list[int]) -> list[int]:
        if label[s] >= 0:
            return cache[label[s]]
        g, tl = self.g, self.ra.t_late
        comp, q = [s], deque([s])
        label[s] = s
        while q:
            i = q.popleft()
            for _, j in g.succ[i]:
                if label[j] < 0:
                    label[j] = s
                    comp.append(j)
                    q.append(j)
        top = max(tl[i] for i in comp)
        d = [INF] * g.area
        q = deque(i for i in sorted(comp) if tl[i] == top)
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

    # ----- stage 4: space-time A* in priority order ---------------------------------------------
    def _blocked(self, pid: str, s: int, c: int, n: int, t1: int, h: list[int]) -> bool:
        W1, a = self.W + 1, self.g.area
        if not (t1 == 1 and n == s == c and self._own_stay):
            owner = self.vres.get(n * W1 + t1)
            if owner is not None and owner != pid:
                return True
        if n != c:
            owner = self.eres.get((n * a + c) * W1 + t1)
            if owner is not None and owner != pid:
                return True
        if t1 == 1:
            if n in self.forbid.get(pid, ()):
                return True
            if self._progress_only is not None and n != c and not self.g.safe_exit[n] and h[n] >= h[c]:
                return True       # F2: a protected person vacates only by a progress move
        return False

    def search(self, pid: str, s: int, h: list[int], target: int | None) -> tuple[list[int] | None, str, int]:
        g, W, hard, rw, margin, mx = self.g, self.W, self.hard, self.rw, self.margin, self.mx
        budget = min(self.cfg["node_budget_per_person"], self.budget_left)
        W1 = W + 1
        hs = 1.0 + mx
        heap = [(h[s] * hs, h[s] * hs, 0, s, 0)]
        gcost = {s * W1: 0.0}
        parent: dict[int, int] = {}
        closed: set[int] = set()
        cnt = expanded = 0
        best = goal = None
        succ, safe_exit, hazard, blocked = g.succ, g.safe_exit, self.hazard, self._blocked
        while heap and expanded < budget:
            f, _, _, c, tau = heapq.heappop(heap)
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
                if blocked(pid, s, c, n, t1, h):
                    continue
                q = hazard(n, t1)
                if q >= hard:
                    continue
                if safe_exit[n]:
                    hn = 0.0 if target is None or n == target else margin
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

    def greedy(self, pid: str, s: int, h: list[int]) -> list[int] | None:
        g = self.g
        opts = sorted((-1 if g.safe_exit[n] else h[n], rank, n) for rank, (_, n) in enumerate(g.succ[s] + (("stay", s),)))
        for hn, _, n in opts:
            if hn < INF and not self._blocked(pid, s, s, n, 1, h) and self.hazard(n, 1) < self.hard:
                return [s, n]
        return None

    def least_hazard(self, s: int) -> list[int]:
        g = self.g
        return [s, min((self.hazard(n, 1), rank, n) for rank, (_, n) in enumerate(g.succ[s] + (("stay", s),)))[2]]

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

    def _out_of_cpu(self) -> bool:
        # >= : a coarse thread clock (about 15.6 ms on Windows) can read 0 elapsed; budget 0 must always trip.
        return time.thread_time_ns() - self.cpu0 >= self.cpu_budget_ns

    def plan_one(self, pid: str) -> None:
        s, info, h, target = self.pos[pid], self.info[pid], self.hfield[pid], self.targets[pid]
        W1 = self.W + 1
        path, status, exp, hold = None, "", 0, True
        owner = self.vres.get(s * W1 + 1)
        info["vacate_requested_by"] = owner if owner is not None and owner != pid else None
        info["vacated"] = False
        if self.limit_hit is None and self._out_of_cpu():
            self.limit_hit = "cpu_budget_ms"
        if self.limit_hit is None and self.budget_left <= 0:
            self.limit_hit = "node_budget_total"
        if self.limit_hit is None and info["vacate_requested_by"] is not None:
            self._own_stay = False
            self._progress_only = h if self.protected[pid] else None
            try:
                path, status, exp = self.search(pid, s, h, target)
            finally:
                self._own_stay, self._progress_only = True, None
            if status in ("goal_exit", "window") and path is not None and len(path) > 1 and path[1] != s:
                info["vacated"] = True
            else:
                info["expansions"] = info.get("expansions", 0) + exp
                path, exp = None, 0
        if path is None:
            if self.limit_hit is not None:
                path, hold = self.greedy(pid, s, h), False
                status = "cpu_budget_fallback" if self.limit_hit == "cpu_budget_ms" else "node_budget_fallback"
            else:
                path, status, exp = self.search(pid, s, h, target)
                if path is None:
                    path, hold = self.greedy(pid, s, h), False
                    status = "no_successor_greedy" if path is not None else status
        if path is None:
            path, status, hold = self.least_hazard(s), "no_successor_least_hazard", False
        info["plan_status"] = status
        info["expansions"] = info.get("expansions", 0) + exp
        self.plans[pid] = path
        self.reserve(pid, path, hold)

    # ----- stage 5: verification and priority-aware repair ---------------------------------------
    def actions(self) -> dict[str, str]:
        out = {}
        for pid, path in self.plans.items():
            s, n = path[0], path[1] if len(path) > 1 else path[0]
            out[pid] = "stay" if n == s else next(a for a, j in self.g.succ[s] if j == n)
        return out

    def resolve(self, actions: dict[str, str]) -> dict[str, tuple[str, tuple[int, int]]]:
        pg = self.pg
        self.resolver_calls += 1
        return resolve_moves(height=pg.height, width=pg.width, walls=pg.walls, burning=pg.burning,
                             enter_fire=pg.enter_fire, positions=dict(pg.active), actions=actions)

    def repair_roots(self, actions: dict[str, str], res: dict[str, tuple[str, Any]]) -> list[str]:
        occupant = {i: pid for pid, i in self.pos.items()}
        tgt = {pid: self.plans[pid][1] for pid, a in actions.items() if a != "stay"}
        claims: dict[int, list[str]] = {}
        for pid, t in tgt.items():
            claims.setdefault(t, []).append(pid)
        rank = self.rank
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
                roots.add(max(cyc, key=rank.__getitem__))          # lowest-priority member re-plans
            elif o == "blocked_conflict":
                winner = min(claims[t])                              # the resolver's lowest-ID winner
                roots.add(winner if rank[winner] > rank[pid] else pid)
            elif o == "blocked_occupied":
                occ = occupant.get(t)
                if occ is None or actions[occ] == "stay":
                    roots.add(pid)
            else:
                roots.add(pid)
        return sorted(roots, key=rank.__getitem__)

    # ----- stage 6: priority accounting ------------------------------------------------------------
    def _intended(self, pid: str) -> int:
        s, h = self.pos[pid], self.hfield[pid]
        if h[s] <= 0 or h[s] >= INF:
            return s
        for _, n in self.g.succ[s]:
            if h[n] == h[s] - 1:
                return n
        return s

    def run(self) -> tuple[dict[str, str], dict[str, Any]]:
        self.assign_exits()
        fields, exit_idx = self.ra.fields, dict(self.g.safe_exits)
        self.hfield: dict[str, list[int]] = {}
        self.targets: dict[str, int | None] = {}
        cache: dict[int, list[int]] = {}
        label = [-1] * self.g.area
        for pid in self.order:
            info = self.info[pid]
            if info["route_found"]:
                self.hfield[pid] = fields[info["target_exit"]].dist
                self.targets[pid] = exit_idx[info["target_exit"]]
            else:
                self.hfield[pid] = self.shelter_field(self.pos[pid], cache, label)
                self.targets[pid] = None
        for pid in self.order:
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
            if any(res2[p][1] != res[p][1] for p in res):
                raise AssertionError("converting blocked moves to stay changed end positions")
            acts, res = final, res2
        return acts, self.diagnostics(acts, res, converted, rounds, repaired)

    def diagnostics(self, acts, res, converted, rounds, repaired) -> dict[str, Any]:
        g, W1 = self.g, self.W + 1
        end = {pid: g.idx(res[pid][1]) for pid in acts}
        targets = {end[pid] for pid in acts if res[pid][0] == "moved"}
        stayers = {self.pos[pid] for pid in acts if acts[pid] == "stay"}
        people: dict[str, dict[str, Any]] = {}
        for pid in self.order:
            info, s = self.info[pid], self.pos[pid]
            intended = self._intended(pid)
            moved = res[pid][0] == "moved"
            took = self.vres.get(intended * W1 + 1) if intended != s else None
            took = took if took is not None and took != pid and self.rank[took] < self.rank[pid] else None
            swap = self.eres.get((intended * g.area + s) * W1 + 1) if intended != s else None
            if took is None and swap is not None and swap != pid and self.rank[swap] < self.rank[pid]:
                took = swap
            yielded_to, contested = None, None
            if info["mode"] == "shelter":
                effect = "sheltering"
            elif pid in converted:
                effect = "converted_to_stay"
            elif moved and end[pid] == intended:
                effect = "moved_as_intended"
            elif moved and info["vacated"]:          # left its cell for an earlier planner
                h = self.hfield[pid]
                effect = "yielded_step_aside" if h[end[pid]] >= h[s] and not g.safe_exit[end[pid]] else "yielded_reroute"
                yielded_to, contested = info["vacate_requested_by"], s
            elif moved:
                effect, yielded_to = ("yielded_reroute", took) if took is not None else ("moved_alternative", None)
                contested = intended if took is not None else None
            elif info["vacate_requested_by"] is not None:
                effect = "vacate_refused"
            elif took is not None:
                effect, yielded_to, contested = "yielded_wait", took, intended
            else:
                effect = "waited"
            free = (info["mode"] == "evacuate" and not moved and intended != s and intended not in targets
                    and intended not in stayers and self.hazard(intended, 1) < self.hard)
            row = {
                "rank": self.rank[pid], "priority_class": self.pclass[pid], "protected": self.protected[pid],
                **{k: v for k, v in info.items() if k not in ("expansions",)},
                "expansions": info.get("expansions", 0),
                "fallback": info["plan_status"] in FALLBACK_STATUSES,
                "fallback_reason": info["plan_status"] if info["plan_status"] in FALLBACK_STATUSES else None,
                "intended_move": self._action(s, intended), "action": acts[pid], "predicted_outcome": res[pid][0],
                "priority_effect": effect, "yielded_to": yielded_to,
                "contested_cell": None if contested is None else list(g.rc(contested)),
                "yield_over_id_rule": yielded_to is not None and pid < yielded_to,
                "free_progress_unused": free, "repairs": repaired[pid],
                "next_hazard": round(self.hazard(end[pid], 1), 6),
            }
            if self.scheme == "risk":
                row["risk"] = public_view(self.risk[pid])
            else:
                r = self.risk[pid]
                row["risk"] = {"fire_layer": r["fire_layer"], "static_distance": r["static_distance"],
                               "options": r["options"]}
            people[pid] = row
        for pid, p in people.items():
            p["beneficiary_of"] = sorted(q for q, r in people.items() if r["yielded_to"] == pid)
        yields = [p for p in people.values() if p["priority_effect"] in YIELD_EFFECTS]
        # Effective yield: the beneficiary is predicted to move into the contested cell.
        eff = [p for p in yields if people[p["yielded_to"]]["predicted_outcome"] == "moved"
               and list(g.rc(end[p["yielded_to"]])) == p["contested_cell"]]
        status = Counter(p["plan_status"] for p in people.values())
        load = Counter(p["target_exit"] for p in people.values() if p["route_found"])
        return {
            "active": len(acts),
            "scheme": self.scheme,
            "safe_exits": [eid for eid, _ in g.safe_exits],
            "no_route": sorted(pid for pid, p in people.items() if not p["route_found"]),
            "target_exit_load": dict(sorted(load.items())),
            "diverted": sorted(pid for pid, p in people.items() if p.get("diverted")),
            "plan_status": dict(sorted(status.items())),
            "fallback_people": sorted(pid for pid, p in people.items() if p["fallback"]),
            "expansions": self.cfg["node_budget_total"] - self.budget_left,
            "budget": {"limit_hit": self.limit_hit, "node_budget_total": self.cfg["node_budget_total"],
                       "cpu_budget_ms": self.cfg["cpu_budget_ms"]},
            "repair_rounds": rounds, "resolver_calls": self.resolver_calls,
            "converted_to_stay": converted,
            "predicted_outcomes": dict(sorted(Counter(o for o, _ in res.values()).items())),
            "priority": {
                "order": list(self.order),
                "classes": dict(sorted(Counter(self.pclass.values()).items())),
                "effects": dict(sorted(Counter(p["priority_effect"] for p in people.values()).items())),
                "yields": len(yields),
                "yields_over_id_rule": sum(p["yield_over_id_rule"] for p in yields),
                "effective_yields": len(eff),
                "effective_yields_over_id_rule": sum(p["yield_over_id_rule"] for p in eff),
                "vacate_refused": sum(p["priority_effect"] == "vacate_refused" for p in people.values()),
                "free_progress_unused": sum(p["free_progress_unused"] for p in people.values()),
            },
            "people": dict(sorted(people.items())),
        }

    def _action(self, s: int, n: int) -> str:
        return "stay" if n == s else next(a for a, j in self.g.succ[s] if j == n)
