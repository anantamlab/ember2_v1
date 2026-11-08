"""T4 v1.2.0: v1.1.0 planner plus an exact infeasibility proof before each vacate (P1, one bounded attempt).

The v1.0.0 (`t4.py`) and v1.1.0 (`t4_v11.py`) modules are frozen and unchanged. This module subclasses the
v1.1.0 transactional planner and changes ONLY one thing: a vacate request that provably fails is answered
"fail" without running its A* searches and push sub-chain. Fire forecast, urgency weights, risk thresholds,
strata, planning order, exit assignment, the A* cost, the reservation model (T1-T7 of v1.1.0) and every
frozen number are the v1.1.0 code and values. Design and evidence: docs/P1_T4_V12_NODE_BUDGET.md.

Measured v1.1.0 defect (exploratory namespace `t4-explore@v12`, never a pilot namespace): every fallback
of a limited decision is `node_budget_fallback`; one early root transaction can spend 6 000-16 000 of the
20 000 nodes on a depth-first walk of push chains through a boxed crowd (up to 792 vacate requests, all of
which fail); every later person then gets the greedy fallback.

Proof rule (`_may_vacate`, run at the entry of `plan_tx(o)` for a requester). In v1.1.0 a vacate of `o`
succeeds only through a push chain o = q0 -> q1 -> ... -> qk of distinct unplanned people, where each q(i)
plans a tau = 1 move into the cell of q(i+1), and qk moves into a cell that is empty or held by a planned
mover. Each such move must pass `_blocked` at its time. These blocks hold at frame entry and stay true for
the whole frame, because inside the frame only frame members add reservations and plans:
  B1 the tau = 1 hazard is >= hard, or the cell has no finite distance (static, as in A*);
  B2 F2: a protected person's non-progress move (static; every frame member is a vacate);
  B3 the vertex key (n, 1) or the reverse edge key is held now (held keys belong to people outside the
     frame; no frame member can release them);
  B4 the cell's planned occupant has a planned tau = 1 walk that already reaches q(i)'s cell, or reaches
     o's cell: o is planned (an ancestor of every other member) whenever a member searches, so the cycle
     check of v1.1.0 (T5) then closes through the chain;
  B5 the cell is o's cell for a member other than o (o is planned; the same cycle as B4).
The rule does a breadth-first search over unplanned people from `o` with these blocks only. If no chain can
reach an empty or follow cell, the v1.1.0 vacate must fail, and a failed v1.1.0 vacate restores every
reservation, plan, origin and forbid set (T3). v1.2.0 then returns the same failure without spending nodes.
Every other condition of v1.1.0 (forbid sets, A* full-plan requirement, later-tau reservations, depth and
node limits) can only cause more failures, so it is left out of the proof.

Consequences: when v1.1.0 hits no limit, v1.2.0 decisions are identical to v1.1.0 decisions; when v1.1.0
hits a limit, v1.2.0 spends the saved nodes on later people. Node accounting is unchanged: `expansions`
counts A* nodes actually expanded; `transaction.vacate_failed_infeasible` counts proved failures.
"""
from __future__ import annotations

import math
from typing import Any

from .interface import ReferencePolicy
from .t4 import INF
from .t4_risk import COMPONENTS
from .t4_v11 import _DESCRIPTIVE_V11, _FROZEN_V11, _NUMERIC_V11, TransactionalPlanner, _T4V11Base

PLANNER_VERSION_V12 = "t4-priority-planner@1.2.0"

_DESCRIPTIVE_V12 = {
    **_DESCRIPTIVE_V11,
    "planner": PLANNER_VERSION_V12,
    "vacate_infeasibility_proof": "entry_time_push_chain_reachability_exact_no_search_on_proved_failure",
}
_FROZEN_V12 = dict(_FROZEN_V11)          # every v1.1.0 number, unchanged
_NUMERIC_V12 = dict(_NUMERIC_V11)


class _T4V12Base(_T4V11Base):
    version = "1.2.0"

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return {**_DESCRIPTIVE_V12, "priority_scheme": cls.scheme, **_FROZEN_V12}

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        ReferencePolicy.__init__(self, config, seed=None)
        cfg = self.identity.config
        for key, value in {**_DESCRIPTIVE_V12, "priority_scheme": self.scheme}.items():
            if cfg[key] != value:
                raise ValueError(f"{self.name}: {key} is not configurable")
        for key, (typ, lo, hi) in _NUMERIC_V12.items():
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
        return ProofPlanner(self.identity.config, obs, self.scheme).run()


class T4RiskPriorityV12(_T4V12Base):
    """A2 v1.2.0: risk-based urgency priority, transactional reservations, exact vacate infeasibility proof."""
    name = "T4"
    policy = "risk_based_priority"
    scheme = "risk"


class T4NoPriorityV12(_T4V12Base):
    """A0 v1.2.0: ascending person ID order."""
    name = "T4-A0"
    policy = "risk_priority_ablation_no_priority"
    scheme = "none"


class T4DistancePriorityV12(_T4V12Base):
    """A1 v1.2.0: fire-distance order."""
    name = "T4-A1"
    policy = "risk_priority_ablation_fire_distance"
    scheme = "fire_distance"


ABLATIONS_V12 = {"A0": T4NoPriorityV12, "A1": T4DistancePriorityV12, "A2": T4RiskPriorityV12}


class ProofPlanner(TransactionalPlanner):
    def _walk_cells(self, u: str) -> list[int]:
        """Cells of the planned tau = 1 walk from planned person `u`, as the v1.1.0 cycle check follows it."""
        cells: list[int] = []
        cur = u
        while cur is not None and len(cells) <= len(self.plans):
            plan = self.plans.get(cur)
            if plan is None or plan[1] == self.pos[cur]:
                break
            cells.append(plan[1])
            cur = self.occ.get(plan[1])
        return cells

    def _may_vacate(self, o: str) -> bool:
        """False only if the v1.1.0 vacate of `o` must fail (blocks B1-B5 of the module docstring)."""
        g, W1, a, hard = self.g, self.W + 1, self.g.area, self.hard
        vres, eres, plans, occ, s_o = self.vres, self.eres, self.plans, self.occ, self.pos[o]
        seen, todo = {o}, [o]
        while todo:
            q = todo.pop()
            s, h, prot = self.pos[q], self.hfield[q], self.protected[q]
            for _, n in g.succ[s]:
                if n == s or self.hazard(n, 1) >= hard:                                    # B1
                    continue
                if not g.safe_exit[n] and (h[n] >= INF or (prot and h[n] >= h[s])):       # B1, B2
                    continue
                if n * W1 + 1 in vres or (n * a + s) * W1 + 1 in eres:                     # B3
                    continue
                u = occ.get(n)
                if u is None:
                    return True
                if u not in plans:
                    if u != o and u not in seen:                                            # B5
                        seen.add(u)
                        todo.append(u)
                    continue
                walk = self._walk_cells(u)
                if s in walk or s_o in walk:                                                # B4
                    continue
                return True
        return False

    def plan_tx(self, pid: str, requester: str | None, root: str, depth: int) -> bool:
        if requester is not None:
            self._limits()
            if self.limit_hit is None and depth <= self.cfg["push_depth_limit"] and not self._may_vacate(pid):
                info = self.info[pid]
                info["vacate_requested_by"], info["vacated"], info["origin"] = None, False, None
                info["push_refusals"] = info.get("push_refusals", 0) + 1
                self.tx["vacate_requests"] += 1
                self.tx["max_push_depth"] = max(self.tx["max_push_depth"], depth)
                self.tx["vacate_failed_infeasible"] += 1
                return False
        return super().plan_tx(pid, requester, root, depth)
