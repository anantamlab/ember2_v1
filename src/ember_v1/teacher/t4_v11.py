"""T4 v1.1.0: shared planner with a transactional reservation model (overnight P2).

The v1.0.0 policies in `t4.py` are frozen and unchanged. This module subclasses
the v1.0.0 planner and replaces ONLY the reservation, vacate and repair steps.
Fire forecast, urgency, strata, planning order, exit assignment, the A* cost
and the frozen numbers are the v1.0.0 code and values.
Design: docs/P1_T4_RISK_PRIORITY_V2.md section 2; defect analysis:
docs/overnight/T4_V1_FAILURE_ANALYSIS.md.

v1.0.0 defects (traced on the frozen pilot):
  D1 `reserve()` silently overwrote another person's tau = 1 vertex reservation
     when a person that failed to vacate stayed on its own cell.
  D2 repair re-planned a root while LATER-ranked reservations stayed in the
     table, so a later-ranked person could constrain an earlier-ranked one.
  D3 people whose move depended on a repaired person kept their plans; the
     resolver then blocked them one repair round at a time (forced stays).

v1.1.0 model (each person is planned once, in planning order, as a ROOT):
  T1 A root plans with A* against the reservations already held. Every held
     reservation belongs to an earlier root or to a person that an earlier
     root pushed (its origin precedes the root): invariant I1.
  T2 If the root's tau = 1 cell is the cell of a person not yet planned, that
     person must vacate now (push). It plans immediately, as a dependent of
     the root (same origin), with the root's reservations in place. It may
     push the next person (chain). A protected (critical) person vacates only
     by a progress move (F2, unchanged). A vacate succeeds only with a full
     plan that leaves the cell (v1.0.0 rule, unchanged).
  T3 If a vacate fails, the whole tentative sub-chain is rolled back (no
     reservation of it remains), the requester forbids that tau = 1 cell and
     plans again (I3). The chain is linear, so a rollback never strands a
     dependent.
  T4 A reservation never overwrites another owner's key; an attempt raises
     (I2). A person not yet planned always finds its own (cell, tau = 1) free,
     because any claim on it triggers T2 at once.
  T5 A tau = 1 move into the cell of a planned mover is a follow move. It is
     rejected if it closes a rotation cycle (the resolver blocks cycles).
  T6 The joint action is checked with the public resolver. By construction no
     move is blocked; any mismatch is counted (`verification_mismatches`) and
     converted to stay as in v1.0.0. Tests require 0 (I4).
  T7 Budgets (I7): the v1.0.0 node and CPU budgets; every search spends at
     least one node; push depth <= push_depth_limit. A budget-limited person
     never pushes and never vacates (reservation-consistent greedy, labeled).
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any

from .interface import ReferencePolicy
from .t4 import _DESCRIPTIVE, _FROZEN, _NUMERIC, YIELD_EFFECTS, _PriorityPlanner, _T4Base
from .t4_risk import COMPONENTS

PLANNER_VERSION_V11 = "t4-priority-planner@1.1.0"

_DESCRIPTIVE_V11 = {
    **_DESCRIPTIVE,
    "planner": PLANNER_VERSION_V11,
    "yield_rule": "later_planner_vacates_cell_claimed_at_tau1_by_earlier_root_or_its_dependent_if_full_plan_exists",
    "repair": "transactional_push_with_chain_rollback_no_replanning_of_committed_people",
    "reservation_ownership": "no_overwrite_conflicts_resolved_by_push_or_rollback",
    "cycle_rule": "tau1_follow_move_rejected_if_it_closes_a_rotation_cycle",
    "unrepaired_blocked_moves": "stay_counted_as_verification_mismatch",
}
# v1.0.0 numbers, minus repair_rounds (no repair loop), plus the push depth bound (resource limit).
_FROZEN_V11 = {**{k: v for k, v in _FROZEN.items() if k != "repair_rounds"}, "push_depth_limit": 256}
_NUMERIC_V11 = {**{k: v for k, v in _NUMERIC.items() if k != "repair_rounds"}, "push_depth_limit": (int, 0, 512)}


class _T4V11Base(_T4Base):
    version = "1.1.0"

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return {**_DESCRIPTIVE_V11, "priority_scheme": cls.scheme, **_FROZEN_V11}

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        ReferencePolicy.__init__(self, config, seed=None)
        cfg = self.identity.config
        for key, value in {**_DESCRIPTIVE_V11, "priority_scheme": self.scheme}.items():
            if cfg[key] != value:
                raise ValueError(f"{self.name}: {key} is not configurable")
        for key, (typ, lo, hi) in _NUMERIC_V11.items():
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
        return TransactionalPlanner(self.identity.config, obs, self.scheme).run()


class T4RiskPriorityV11(_T4V11Base):
    """A2 v1.1.0: risk-based urgency priority, transactional reservations."""
    name = "T4"
    policy = "risk_based_priority"
    scheme = "risk"


class T4NoPriorityV11(_T4V11Base):
    """A0 v1.1.0: ascending person ID order, transactional reservations."""
    name = "T4-A0"
    policy = "risk_priority_ablation_no_priority"
    scheme = "none"


class T4DistancePriorityV11(_T4V11Base):
    """A1 v1.1.0: fire-distance order, transactional reservations."""
    name = "T4-A1"
    policy = "risk_priority_ablation_fire_distance"
    scheme = "fire_distance"


ABLATIONS_V11 = {"A0": T4NoPriorityV11, "A1": T4DistancePriorityV11, "A2": T4RiskPriorityV11}


class ReservationConflict(AssertionError):
    """A reservation would overwrite another owner's key (invariant I2)."""


class TransactionalPlanner(_PriorityPlanner):
    def __init__(self, cfg: dict[str, Any], obs: dict[str, Any], scheme: str) -> None:
        super().__init__(cfg, obs, scheme)
        self.occ = {i: pid for pid, i in self.pos.items()}
        self.origin: dict[str, str] = {}          # planned person -> root of its transaction
        self.pushed_by: dict[str, str] = {}       # dependent -> direct requester
        self._may_push = True
        self.tx: Counter = Counter()
        self.precedence_audit: list[dict[str, Any]] = []

    # ----- reservations ------------------------------------------------------------------------
    def reserve(self, pid: str, path: list[int], hold: bool) -> None:
        W1, a = self.W + 1, self.g.area
        for tau in range(1, len(path)):          # check every key before writing any (all or nothing)
            c, n = path[tau - 1], path[tau]
            owner = self.vres.get(n * W1 + tau)
            if owner is not None and owner != pid:
                raise ReservationConflict(f"{pid} would overwrite {owner} at cell {self.g.rc(n)} tau {tau}")
            if n != c:
                owner = self.eres.get((c * a + n) * W1 + tau)
                if owner is not None and owner != pid:
                    raise ReservationConflict(f"{pid} would overwrite edge of {owner} at tau {tau}")
        super().reserve(pid, path, hold)

    def _closes_cycle(self, s: int, n: int) -> bool:
        cur, steps = self.occ.get(n), 0
        while cur is not None and cur in self.plans and steps <= len(self.plans):
            nxt = self.plans[cur][1]
            if nxt == self.pos[cur]:
                return False
            if nxt == s:
                return True
            cur, steps = self.occ.get(nxt), steps + 1
        return False

    def _blocked(self, pid: str, s: int, c: int, n: int, t1: int, h: list[int]) -> bool:
        W1, a = self.W + 1, self.g.area
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
            if n != s:
                o = self.occ.get(n)
                if o is not None:
                    if o in self.plans:
                        if self._closes_cycle(s, n):
                            return True
                    elif not self._may_push:
                        return True
        return False

    # ----- transactional planning ----------------------------------------------------------------
    def _limits(self) -> None:
        if self.limit_hit is None and self._out_of_cpu():
            self.limit_hit = "cpu_budget_ms"
        if self.limit_hit is None and self.budget_left <= 0:
            self.limit_hit = "node_budget_total"

    def _safe_least_hazard(self, pid: str, s: int, h: list[int]) -> list[int]:
        self._may_push = False
        try:
            opts = [n for _, n in self.g.succ[s] if not self._blocked(pid, s, s, n, 1, h)]
        finally:
            self._may_push = True
        rank = {n: i for i, (_, n) in enumerate(self.g.succ[s])}
        return [s, min(((self.hazard(n, 1), rank.get(n, 99), n) for n in [*opts, s]))[2]]

    def _fallback(self, pid: str, s: int, h: list[int]) -> tuple[list[int], str]:
        self._may_push = False
        try:
            path = self.greedy(pid, s, h)
        finally:
            self._may_push = True
        if path is not None:
            return path, ("cpu_budget_fallback" if self.limit_hit == "cpu_budget_ms" else
                          "node_budget_fallback" if self.limit_hit == "node_budget_total" else "no_successor_greedy")
        return self._safe_least_hazard(pid, s, h), "no_successor_least_hazard"

    def _audit(self, pid: str) -> None:
        r = self.rank[pid]
        # Every reservation owner is a planned person (tests check owners <= plans), so O(N) per root suffices.
        bad = sorted(o for o in self.plans if self.rank[self.origin[o]] >= r)
        if bad:
            self.precedence_audit.append({"root": pid, "later_origin_owners": bad})

    def plan_tx(self, pid: str, requester: str | None, root: str, depth: int) -> bool:
        """Plan `pid`. Requester None: `pid` is a root and always gets a plan. Else `pid` must vacate its cell
        for `requester`; on failure nothing of `pid` or its sub-chain stays reserved and False is returned."""
        s, info, h, target = self.pos[pid], self.info[pid], self.hfield[pid], self.targets[pid]
        forbid = self.forbid.setdefault(pid, set())
        saved_forbid = set(forbid)
        info["vacate_requested_by"], info["vacated"], info["origin"] = requester, False, root
        if requester is None:
            self._audit(pid)
        else:
            self.tx["vacate_requests"] += 1
            self.tx["max_push_depth"] = max(self.tx["max_push_depth"], depth)
        while True:
            self._limits()
            if requester is not None and (self.limit_hit is not None or depth > self.cfg["push_depth_limit"]):
                self.tx["vacate_failed_" + ("limit" if self.limit_hit is not None else "depth")] += 1
                break
            if self.limit_hit is not None:
                path, status = self._fallback(pid, s, h)
                hold = False
            else:
                self._progress_only = h if requester is not None and self.protected[pid] else None
                try:
                    path, status, exp = self.search(pid, s, h, target)
                finally:
                    self._progress_only = None
                info["expansions"] = info.get("expansions", 0) + exp
                hold = True
                if requester is not None:
                    if not (status in ("goal_exit", "window") and path is not None and len(path) > 1
                            and path[1] != s):
                        self.tx["vacate_failed_no_full_plan"] += 1
                        break
                elif path is None:
                    path, status = self._fallback(pid, s, h)
                    hold = False
            n = path[1] if len(path) > 1 else s
            o = self.occ.get(n) if n != s else None
            self.plans[pid] = path
            self.origin[pid] = root
            self.reserve(pid, path, hold)
            if o is not None and o not in self.plans:
                if self.plan_tx(o, pid, root, depth + 1):
                    self.pushed_by[o] = pid
                else:                              # roll back this attempt, forbid the cell, plan again
                    self.unreserve(pid)
                    del self.plans[pid], self.origin[pid]
                    forbid.add(n)
                    self.tx["push_rollbacks"] += 1
                    continue
            info["plan_status"] = status
            if requester is not None:
                info["vacated"] = True
                self.tx["vacate_success"] += 1
            return True
        # vacate failed: nothing of pid is reserved (any partial attempt was rolled back above)
        forbid.clear()
        forbid.update(saved_forbid)
        info["vacate_requested_by"], info["origin"] = None, None
        info["push_refusals"] = info.get("push_refusals", 0) + 1
        return False

    def run(self) -> tuple[dict[str, str], dict[str, Any]]:
        self.assign_exits()
        fields, exit_idx = self.ra.fields, dict(self.g.safe_exits)
        self.hfield, self.targets = {}, {}
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
            if pid not in self.plans:
                self.plan_tx(pid, None, pid, 0)
        acts = self.actions()
        res = self.resolve(acts)
        converted = sorted(pid for pid, a in acts.items() if a != "stay" and res[pid][0] != "moved")
        if converted:
            final = {**acts, **{pid: "stay" for pid in converted}}
            res2 = self.resolve(final)
            if any(res2[p][1] != res[p][1] for p in res):
                raise AssertionError("converting blocked moves to stay changed end positions")
            acts, res = final, res2
        return acts, self.diagnostics(acts, res, converted, 0, Counter())

    def _chain_taker(self, pid: str, s: int, intended: int) -> str | None:
        """Owner of the person's intended tau = 1 cell (or of the reverse edge) whose ORIGIN precedes the person.
        v1.0.0 accounting only credits owners ranked before the person; a pushed dependent ranked after it but
        planned for an earlier root is credited here (chain yield)."""
        W1, a = self.W + 1, self.g.area
        for o in (self.vres.get(intended * W1 + 1), self.eres.get((intended * a + s) * W1 + 1)):
            if o is not None and o != pid and self.rank[self.origin[o]] < self.rank[pid]:
                return o
        return None

    def diagnostics(self, acts, res, converted, rounds, repaired) -> dict[str, Any]:
        out = super().diagnostics(acts, res, converted, rounds, repaired)
        people = out["people"]
        g = self.g
        for pid, p in people.items():           # chain attribution of waits and reroutes (see _chain_taker)
            if p["priority_effect"] not in ("waited", "moved_alternative"):
                continue
            s = self.pos[pid]
            intended = self._intended(pid)
            o = self._chain_taker(pid, s, intended) if intended != s else None
            if o is not None:
                p["priority_effect"] = "yielded_wait" if p["priority_effect"] == "waited" else "yielded_reroute"
                p["yielded_to"], p["contested_cell"] = o, list(g.rc(intended))
                p["yield_over_id_rule"] = pid < o
        if any(p["yielded_to"] is not None for p in people.values()):
            for pid, p in people.items():
                p["beneficiary_of"] = sorted(q for q, r in people.items() if r["yielded_to"] == pid)
            yields = [p for p in people.values() if p["priority_effect"] in YIELD_EFFECTS]
            eff = [p for p in yields if people[p["yielded_to"]]["predicted_outcome"] == "moved"
                   and list(g.rc(g.idx(res[p["yielded_to"]][1]))) == p["contested_cell"]]
            pr = out["priority"]
            pr["effects"] = dict(sorted(Counter(p["priority_effect"] for p in people.values()).items()))
            pr["yields"] = len(yields)
            pr["yields_over_id_rule"] = sum(p["yield_over_id_rule"] for p in yields)
            pr["effective_yields"] = len(eff)
            pr["effective_yields_over_id_rule"] = sum(p["yield_over_id_rule"] for p in eff)
        inversions = chain = 0
        for pid, p in people.items():
            p["origin"] = self.origin.get(pid)
            p["pushed_by"] = self.pushed_by.get(pid)
            p.setdefault("push_refusals", 0)
            b = p["yielded_to"]
            if p["priority_effect"] in YIELD_EFFECTS and b is not None:
                org = self.origin.get(b, b)
                p["yield_origin"] = org
                p["precedence_inversion"] = self.rank[pid] < self.rank[org]
                p["yield_kind"] = "direct" if self.rank[b] < self.rank[pid] else "chain"
                inversions += p["precedence_inversion"]
                chain += p["yield_kind"] == "chain"
            else:
                p["yield_origin"], p["precedence_inversion"], p["yield_kind"] = None, False, None
        out["priority"]["precedence_inversions"] = inversions
        out["priority"]["chain_yields"] = chain
        out["priority"]["chain_yields_over_id_rule"] = sum(
            p["yield_kind"] == "chain" and p["yield_over_id_rule"] for p in people.values())
        out["transaction"] = {**{k: int(v) for k, v in sorted(self.tx.items())},
                              "precedence_audit_violations": len(self.precedence_audit),
                              "verification_mismatches": len(converted)}
        return out
