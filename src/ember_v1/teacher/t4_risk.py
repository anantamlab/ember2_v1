"""T4 public risk and urgency estimation (issue #24). Design: docs/P1_T4_RISK_PRIORITY.md section 2.

Pure functions of a validated `Observation.to_dict()` and the frozen T4
configuration. No seed, RNG state, future burning set, trapped flag, scenario
ID, label or generator metadata is read. This module plans nothing; the T4
planner (`t4.py`) and the evaluator-side risk strata (`t4_eval.py`) both use it.

Fire forecast (public rule only). Layer l(c) = least number of spread steps
that can ignite c (BFS from the burning set over flammable cells). Spread
happens in absolute step t only if t % spread_interval == 0; n(tau) = spread
steps among the next tau steps. P(c burns by the end of relative step tau):
  deterministic   1 if n(tau) >= l(c) else 0 (exact);
  stochastic l=1  1 - (1-p)^(k*n), k = burning 4-neighbours now (exact);
  stochastic l>=2 P(Binomial(n, p) >= l), a single-front LOWER bound.
Arrival estimate T_q(c) = first relative step with P >= q (q_early and q_late).
A cell whose estimate is after the episode end (max_steps) gets CAP.
T_q is exact for deterministic fire (q has no effect). For stochastic fire it
is an estimate: the l >= 2 bound makes it later (optimistic) when several fire
fronts converge.

Per safe exit e, over cells that are open now (other safe exits absorbing):
  d_e     static BFS distance;
  L_e^q   latest departure: the latest relative step at which a person can
          still be on c and reach e with every visited cell unburnt
          (L(e) = T_q(e); L(c) = min(T_q(c) - 1, max over neighbours L(n) - 1)),
          computed by a max-min Dijkstra; m_e^q = length of that route;
  G_e     shortest-route viability under T_early (DP over the BFS DAG): a
          shortest route from c is fire-viable iff G_e(c) > d_e(c).
People, waiting and congestion are not in these fields (estimates).
"""
from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass
from typing import Any

from .grid import PublicGrid

RISK_MODEL_VERSION = "t4-risk@1.0.0"
INF = 1 << 30
CAP = 1 << 20                 # stands for "not before the episode ends"
STRATA = ("critical", "high", "elevated", "low", "no_viable_route")
COMPONENTS = ("margin", "exposure", "proximity", "options", "congestion", "evacuation")


class Cells:
    """Integer-indexed public grid (cell index = row * width + col)."""

    def __init__(self, pg: PublicGrid) -> None:
        h, w = pg.height, pg.width
        self.height, self.width, self.area = h, w, h * w
        a = self.area
        walls = {r * w + c for r, c in pg.walls}
        self.burning_idx = sorted(r * w + c for r, c in pg.burning)
        burning = set(self.burning_idx)
        self.wall = [i in walls for i in range(a)]
        self.open = [i not in walls and i not in burning for i in range(a)]
        self.is_exit = [False] * a
        for _, (r, c) in pg.exits:
            self.is_exit[r * w + c] = True
        self.safe_exits = [(eid, r * w + c) for eid, (r, c) in pg.safe_exits()]
        self.safe_exit = [False] * a
        for _, i in self.safe_exits:
            self.safe_exit[i] = True
        self.nbrs: list[tuple[int, ...]] = []
        self.succ: list[tuple[tuple[str, int], ...]] = []
        for i in range(a):
            r, c = divmod(i, w)
            cand = (("north", r - 1, c), ("south", r + 1, c), ("east", r, c + 1), ("west", r, c - 1))
            inside = [(act, rr * w + cc) for act, rr, cc in cand if 0 <= rr < h and 0 <= cc < w]
            self.nbrs.append(tuple(j for _, j in inside))
            self.succ.append(tuple((act, j) for act, j in inside if self.open[j]))
        # Congestion cells: safe exits and narrow cells (open, <= 2 open 4-neighbours).
        self.narrow = [self.safe_exit[i] or (self.open[i] and len(self.succ[i]) <= 2) for i in range(a)]

    def idx(self, p: tuple[int, int]) -> int:
        return p[0] * self.width + p[1]

    def rc(self, i: int) -> tuple[int, int]:
        return divmod(i, self.width)


class FireForecast:
    """Public-rule fire forecast from the current burning set (no RNG, no future fire)."""

    def __init__(self, cells: Cells, rules: dict[str, Any], step: int) -> None:
        self.deterministic = rules["fire_model"] == "deterministic"
        self.p = 1.0 if self.deterministic else float(rules["spread_probability"])
        self.interval = rules["spread_interval"]
        self.t0 = step
        self.base = step // self.interval
        self.remaining = max(0, rules["max_steps"] - step)
        self.n_max = self.spreads(self.remaining)
        a = cells.area
        flam = [not cells.wall[i] and (rules["exits_flammable"] or not cells.is_exit[i]) for i in range(a)]
        self.layer = layer = [INF] * a
        self.kburn = kburn = [0] * a
        q: deque[int] = deque()
        for i in cells.burning_idx:
            layer[i] = 0
            q.append(i)
        while q:
            i = q.popleft()
            for j in cells.nbrs[i]:
                if flam[j] and layer[j] == INF:
                    layer[j] = layer[i] + 1
                    q.append(j)
                if layer[i] == 0 and layer[j] == 1:
                    kburn[j] += 1
        self._need: dict[tuple[int, int, float], int | None] = {}
        self._tail: dict[tuple[int, int], float] = {}

    def spreads(self, tau: int) -> int:
        """Spread steps among the next `tau` steps."""
        return (self.t0 + tau) // self.interval - self.base

    def tau_of(self, n: int) -> int:
        """Relative step of the n-th future spread step (n >= 1)."""
        return (self.base + n) * self.interval - self.t0

    def prob(self, i: int, tau: int) -> float:
        """P(cell i burns by the end of relative step tau) (lower bound for stochastic l >= 2)."""
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

    def spreads_needed(self, i: int, q: float) -> int | None:
        """Least number of future spread steps after which P(burn) >= q; None if not in the episode."""
        lay = self.layer[i]
        if lay == 0:
            return 0
        p = self.p
        if lay >= INF or p <= 0.0 or lay > self.n_max:
            return None
        if p >= 1.0:
            return lay
        k = self.kburn[i] if lay == 1 else 1
        key = (lay, k, q)
        if key in self._need:
            return self._need[key]
        out = None
        if lay == 1:
            for n in range(1, self.n_max + 1):
                if 1.0 - (1.0 - p) ** (k * n) >= q:
                    out = n
                    break
        else:   # negative binomial CDF: P(Bin(n, p) >= lay) = P(lay-th success by trial n)
            log_pmf = lay * math.log(p)
            log_fail = math.log1p(-p)
            cdf = math.exp(log_pmf)
            n = lay
            while True:
                if cdf >= q:
                    out = n
                    break
                if n >= self.n_max:
                    break
                log_pmf += math.log(n) - math.log(n + 1 - lay) + log_fail
                n += 1
                cdf += math.exp(log_pmf)
        self._need[key] = out
        return out

    def arrival(self, i: int, q: float) -> int:
        """First relative step whose end has P(cell i burns) >= q; CAP if not before the episode ends."""
        n = self.spreads_needed(i, q)
        if n is None:
            return CAP
        return 0 if n == 0 else self.tau_of(n)


@dataclass
class ExitField:
    exit_id: str
    cell: int
    dist: list[int]                     # static BFS distance (INF = no path)
    order: list[int]                    # BFS order (cells with finite dist)
    viable_shortest: list[bool]         # a shortest route is viable under T_early
    slack_early: list[int]              # latest departure L_e under T_early (-INF = none)
    len_early: list[int]
    slack_late: list[int]               # latest departure L_e under T_late
    len_late: list[int]


def _bfs(cells: Cells, e: int) -> tuple[list[int], list[int]]:
    d = [INF] * cells.area
    d[e] = 0
    order = [e]
    q = deque([e])
    succ, safe = cells.succ, cells.safe_exit
    while q:
        i = q.popleft()
        di = d[i] + 1
        for _, j in succ[i]:
            if d[j] == INF and not safe[j]:
                d[j] = di
                order.append(j)
                q.append(j)
    return d, order


def _latest_departure(cells: Cells, e: int, T: list[int]) -> tuple[list[int], list[int]]:
    """Max-min Dijkstra: L(e) = T(e); L(c) = min(T(c) - 1, max_n L(n) - 1); ties: shorter route."""
    a = cells.area
    L = [-INF] * a
    m = [INF] * a
    L[e], m[e] = T[e], 0
    heap = [(-T[e], 0, e)]
    succ, safe = cells.succ, cells.safe_exit
    while heap:
        nl, mm, i = heapq.heappop(heap)
        li = -nl
        if li != L[i] or mm != m[i]:
            continue
        for _, j in succ[i]:
            if safe[j]:
                continue
            cand = T[j] - 1
            if li - 1 < cand:
                cand = li - 1
            if cand < 0:
                continue
            if cand > L[j] or (cand == L[j] and mm + 1 < m[j]):
                L[j], m[j] = cand, mm + 1
                heapq.heappush(heap, (-cand, mm + 1, j))
    return L, m


def exit_field(cells: Cells, eid: str, e: int, t_early: list[int], t_late: list[int], same: bool) -> ExitField:
    d, order = _bfs(cells, e)
    g = [-INF] * cells.area
    g[e] = t_early[e] + 1
    succ = cells.succ
    for c in order[1:]:
        dc = d[c]
        best = -INF
        for _, n in succ[c]:
            if d[n] == dc - 1 and g[n] > best:
                best = g[n]
        v = t_early[c] + dc
        g[c] = best if best < v else v
    viable = [d[i] < INF and g[i] > d[i] for i in range(cells.area)]
    le, me = _latest_departure(cells, e, t_early)
    ll, ml = (le, me) if same else _latest_departure(cells, e, t_late)
    return ExitField(eid, e, d, order, viable, le, me, ll, ml)


def _fin(x: float | int | None) -> float | int | None:
    """Diagnostic view: values at or beyond CAP/2 mean "not before the episode ends" -> None."""
    if x is None or x >= CAP // 2:
        return None
    return x


def _ratio(scale: float, x: float) -> float:
    """scale / (scale + max(0, x)) in (0, 1]: 1 at x <= 0, 1/2 at x = scale, -> 0 as x grows."""
    return scale / (scale + max(0.0, x))


@dataclass
class RiskAssessment:
    cells: Cells
    forecast: FireForecast
    t_early: list[int]
    t_late: list[int]
    fields: dict[str, ExitField]
    pos: dict[str, int]
    people: dict[str, dict[str, Any]]


def assess(pg: PublicGrid, obs: dict[str, Any], cfg: dict[str, Any], *, urgency: bool = True) -> RiskAssessment:
    """Fields and per-person risk signals. With `urgency=False` the urgency score and stratum are skipped."""
    cells = Cells(pg)
    fc = FireForecast(cells, obs["rules"], obs["step"])
    qe, ql = float(cfg["q_early"]), float(cfg["q_late"])
    a = cells.area
    t_early = [fc.arrival(i, qe) for i in range(a)]
    same = fc.deterministic or fc.p >= 1.0 or qe == ql
    t_late = t_early if same else [fc.arrival(i, ql) for i in range(a)]
    fields = {eid: exit_field(cells, eid, e, t_early, t_late, same) for eid, e in cells.safe_exits}
    pos = {pid: cells.idx(p) for pid, p in pg.active}
    people: dict[str, dict[str, Any]] = {}
    for pid, s in pos.items():
        static = [(f.dist[s], eid) for eid, f in fields.items() if f.dist[s] < INF]
        viable_late = sorted((f.dist[s], eid) for eid, f in fields.items() if f.slack_late[s] >= 0)
        # Latest-departure route: largest slack, then shorter route, then lowest exit ID.
        early = sorted((-f.slack_early[s], f.len_early[s], eid) for eid, f in fields.items() if f.slack_early[s] >= 0)
        best_early = early[0] if early else None
        slack_late = max((f.slack_late[s] for f in fields.values()), default=-INF)
        shortest = [f.dist[s] for f in fields.values() if f.viable_shortest[s]]
        if shortest:
            v, v_exact = min(shortest), True
        elif early:
            v, v_exact = min(fields[eid].len_early[s] for _, _, eid in early), False
        elif viable_late:
            v, v_exact = min(fields[eid].len_late[s] for _, eid in viable_late), False
        else:
            v, v_exact = None, False
        people[pid] = {
            "fire_layer": fc.layer[s] if fc.layer[s] < INF else None,
            "exposure_early": t_early[s], "exposure_late": t_late[s],
            "static_distance": min(static)[0] if static else None,
            "nearest_static_exit": min(static)[1] if static else None,
            "viable_exits": [eid for _, eid in viable_late],
            "options": len(viable_late),
            "nominal_exit": viable_late[0][1] if viable_late else None,
            "nominal_distance": viable_late[0][0] if viable_late else None,
            "slack_early": -best_early[0] if best_early else -INF,
            "route_len_early": best_early[1] if best_early else None,
            "slack_late": slack_late,
            "shortest_viable": v, "shortest_viable_exact": v_exact,
        }
    # Congestion delay: FIFO per nominal exit, 1 evacuation per step, arrival = static distance;
    # people with equal arrival share the mean delay of their group (ID-independent).
    by_exit: dict[str, list[tuple[int, str]]] = {}
    for pid, r in people.items():
        if r["nominal_exit"] is not None:
            by_exit.setdefault(r["nominal_exit"], []).append((r["nominal_distance"], pid))
    for pid in people:
        people[pid]["congestion_delay"] = 0.0
    for lst in by_exit.values():
        lst.sort()
        prev = -INF
        i = 0
        while i < len(lst):
            arr = lst[i][0]
            j = i
            while j < len(lst) and lst[j][0] == arr:
                j += 1
            g = j - i
            first = max(arr, prev + 1)
            mean = first - arr + (g - 1) / 2.0
            for _, pid in lst[i:j]:
                people[pid]["congestion_delay"] = mean
            prev = first + g - 1
            i = j
    if urgency:
        for r in people.values():
            score(r, cfg)
    return RiskAssessment(cells, fc, t_early, t_late, fields, pos, people)


def score(r: dict[str, Any], cfg: dict[str, Any]) -> None:
    """Urgency components (each in [0, 1], higher = more urgent), weighted urgency and risk stratum."""
    viable = r["options"] > 0
    q = r["congestion_delay"]
    slack = r["slack_early"]
    margin = slack - q if slack > -INF else None
    r["margin"] = margin
    r["time_to_route_loss"] = slack + r["route_len_early"] if slack > -INF else None
    r["evacuation_time"] = r["route_len_early"] + q if slack > -INF else None
    lay = r["fire_layer"]
    comp = {
        "margin": 1.0 if margin is None else _ratio(cfg["scale_margin"], margin),
        "exposure": 0.0 if r["exposure_early"] >= CAP else _ratio(cfg["scale_exposure"], r["exposure_early"]),
        "proximity": 0.0 if lay is None else _ratio(cfg["scale_proximity"], lay - 1),
        "options": 1.0 / r["options"] if viable else 1.0,
        "congestion": q / (q + cfg["scale_congestion"]),
        "evacuation": 0.0 if r["shortest_viable"] is None else
        r["shortest_viable"] / (r["shortest_viable"] + cfg["scale_evacuation"]),
    }
    digits = cfg["urgency_digits"]
    r["components"] = {k: round(v, digits) for k, v in comp.items()}
    r["urgency"] = round(sum(cfg[f"w_{k}"] * comp[k] for k in COMPONENTS), digits)
    if not viable:
        stratum = "no_viable_route"
    elif slack < 0 or margin < cfg["stratum_critical"]:
        stratum = "critical"
    elif margin < cfg["stratum_high"]:
        stratum = "high"
    elif margin < cfg["stratum_elevated"]:
        stratum = "elevated"
    else:
        stratum = "low"
    r["stratum"] = stratum
    r["viability"] = "none" if not viable else "uncertain" if slack < 0 else "viable"


def public_view(r: dict[str, Any]) -> dict[str, Any]:
    """JSON view of one person's risk record for diagnostics (CAP-sized values -> None)."""
    slack_e = r["slack_early"] if r["slack_early"] > -INF else None
    slack_l = r["slack_late"] if r["slack_late"] > -INF else None
    return {
        "urgency": r.get("urgency"), "components": r.get("components"), "stratum": r.get("stratum"),
        "signals": {
            "fire_layer": r["fire_layer"], "exposure_time": _fin(r["exposure_early"]),
            "shortest_viable_time": r["shortest_viable"], "shortest_viable_exact": r["shortest_viable_exact"],
            "congestion_delay": r["congestion_delay"], "time_to_route_loss": _fin(r.get("time_to_route_loss")),
            "evacuation_time": r.get("evacuation_time"), "margin": _fin(r.get("margin")),
            "options": r["options"], "viable_exits": r["viable_exits"],
        },
        "uncertainty": {
            "viability": r.get("viability"), "exposure_time_early": _fin(r["exposure_early"]),
            "exposure_time_late": _fin(r["exposure_late"]), "slack_early": _fin(slack_e), "slack_late": _fin(slack_l),
            "slack_band": None if _fin(slack_e) is None or _fin(slack_l) is None else slack_l - slack_e,
        },
    }
