"""Exploratory diagnosis of the T4 v1.0.0 repair defect (overnight P1; docs/overnight/T4_V1_FAILURE_ANALYSIS.md).

    python -m ember_v1.teacher.t4_v1_diag [--out PATH]

Read-only. `TracedPlanner` subclasses the frozen v1.0.0 `_PriorityPlanner`. It
records reservation events and a counterfactual per plan call, and it does NOT
change any decision: the run re-checks every decision against the frozen
pilot rows (`data/baselines/p1-08-t4-pilot.json`) and stops on a mismatch.

Traces (per decision):
  overwrite        `reserve()` writes a vertex or edge key that another person owns;
                   the earlier owner silently loses it (I2 violation).
  plan call        person, phase (initial / repair round r), whether a vacate was
                   requested and by whom, the tau = 1 cell chosen, and the tau = 1
                   cell chosen with every reservation of a LATER-ranked person
                   removed (counterfactual). A difference means later-ranked
                   reservations constrained the planner (I1 violation).
  resolver calls   to attribute moves converted to stay.

Classification of one "yield over the ID rule" (yielder Y has the lower ID
than the beneficiary B; A0 rank order = ID order):
  label_only         Y's tau = 1 cell equals the counterfactual: later-ranked
                     reservations did not change Y's action. The label records a
                     reservation, not a priority effect.
  priority_cascade   B's reservation of Y's cell came from a B plan call that was
                     itself a vacate requested by a person ranked before Y. The
                     chain starts at an earlier planner (legitimate precedence).
  genuine_inversion  otherwise: a later-ranked reservation changed the action of
                     an earlier-ranked replanner.

These definitions are fixed here. They are not the definitions of the earlier
exploratory notes (118 / 18 / 102); the counts can differ and both are reported.
This is observational analysis of policy-generated traces, not confirmation.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from ..sim.simulator import Simulator
from . import evaluate as ev
from .t4 import ABLATIONS, YIELD_EFFECTS, _PriorityPlanner
from .t4_pilot import VARIANTS, pilot_cells, pilot_scenario
from .t4_risk import assess

DIAG_VERSION = "t4-v1-diagnosis@1.0.0"
DEFAULT_OUT = ev.REPO / "data/baselines/p1-08-t4-v1-diagnosis.json"
PILOT_REPORT = ev.REPO / "data/baselines/p1-08-t4-pilot.json"


class TracedPlanner(_PriorityPlanner):
    """Frozen v1.0.0 planner with passive tracing. Decisions are unchanged."""

    def __init__(self, *a, **kw) -> None:
        self.t_overwrites: list[dict[str, Any]] = []
        self.t_calls: list[dict[str, Any]] = []
        self.t_resolves: list[tuple[dict[str, str], dict[str, Any]]] = []
        self.t_round = 0
        self.t_quiet = False
        self.t_origin: dict[int, tuple[str, str | None]] = {}   # vertex key -> (owner, owner's vacate requester)
        super().__init__(*a, **kw)

    def reserve(self, pid: str, path: list[int], hold: bool) -> None:
        if not self.t_quiet:
            W1, a = self.W + 1, self.g.area
            req = self.info[pid].get("vacate_requested_by")
            for tau in range(1, len(path)):
                c, n = path[tau - 1], path[tau]
                k = n * W1 + tau
                old = self.vres.get(k)
                if old is not None and old != pid:
                    self.t_overwrites.append({"kind": "vertex", "tau": tau, "cell": list(self.g.rc(n)),
                                              "new_owner": pid, "lost_owner": old, "round": self.t_round,
                                              "stay": n == c})
                self.t_origin[k] = (pid, req)
                if n != c:
                    e = (c * a + n) * W1 + tau
                    old = self.eres.get(e)
                    if old is not None and old != pid:
                        self.t_overwrites.append({"kind": "edge", "tau": tau, "cell": list(self.g.rc(n)),
                                                  "new_owner": pid, "lost_owner": old, "round": self.t_round,
                                                  "stay": False})
        super().reserve(pid, path, hold)

    def _counterfactual(self, pid: str) -> int:
        r = self.rank[pid]
        saved = (self.vres, self.eres, self.budget_left, self.limit_hit, dict(self.info[pid]),
                 self.plans.get(pid), list(self.keys.get(pid, [])), pid in self.keys)
        self.vres = {k: v for k, v in self.vres.items() if self.rank[v] <= r}
        self.eres = {k: v for k, v in self.eres.items() if self.rank[v] <= r}
        self.t_quiet = True
        try:
            _PriorityPlanner.plan_one(self, pid)
            nxt = self.plans[pid][1]
        finally:
            self.t_quiet = False
            (self.vres, self.eres, self.budget_left, self.limit_hit, info, plan, keys, had) = saved
            self.info[pid].clear()
            self.info[pid].update(info)
            if plan is None:
                self.plans.pop(pid, None)
            else:
                self.plans[pid] = plan
            if had:
                self.keys[pid] = keys
            else:
                self.keys.pop(pid, None)
        return nxt

    def plan_one(self, pid: str) -> None:
        if self.t_quiet:
            return super().plan_one(pid)
        r = self.rank[pid]
        later = sorted({v for v in (*self.vres.values(), *self.eres.values()) if self.rank[v] > r})
        cf = self._counterfactual(pid)
        super().plan_one(pid)
        self.t_calls.append({"pid": pid, "round": self.t_round, "replan": self.t_round > 0,
                             "vacate_requested_by": self.info[pid]["vacate_requested_by"],
                             "vacated": self.info[pid]["vacated"], "next": self.plans[pid][1],
                             "counterfactual_next": cf, "later_owners_present": later})

    def repair_roots(self, actions, res):
        roots = super().repair_roots(actions, res)
        if roots:
            self.t_round += 1
        return roots

    def resolve(self, actions):
        out = super().resolve(actions)
        self.t_resolves.append((dict(actions), out))
        return out


class _Traced:
    """Policy wrapper: v1.0.0 identity and config; planner replaced by `TracedPlanner`."""

    def __init__(self, variant: str) -> None:
        self.pol = ABLATIONS[variant]()
        self.last: TracedPlanner | None = None

    def decide(self, obs: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
        from ..contract import validate_actions, validate_observation
        o = validate_observation(obs)
        tp = TracedPlanner(self.pol.identity.config, o, self.pol.scheme)
        acts, dg = tp.run()
        self.last = tp
        return validate_actions(o, acts), dg


def classify(tp: TracedPlanner, dg: dict[str, Any]) -> list[dict[str, Any]]:
    """Classify each yield over the ID rule in one decision."""
    last_call = {}
    for c in tp.t_calls:
        last_call[c["pid"]] = c
    req_of_key = tp.t_origin
    W1 = tp.W + 1
    out = []
    for y, p in dg["people"].items():
        if p["priority_effect"] not in YIELD_EFFECTS or not p["yield_over_id_rule"]:
            continue
        b = p["yielded_to"]
        call = last_call[y]
        if call["next"] == call["counterfactual_next"]:
            kind = "label_only"
        else:
            owner, req = req_of_key.get(tp.pos[y] * W1 + 1, (None, None))
            kind = ("priority_cascade" if owner == b and req is not None and tp.rank[req] < tp.rank[y]
                    else "genuine_inversion")
        out.append({"yielder": y, "beneficiary": b, "effect": p["priority_effect"], "class": kind,
                    "yielder_repairs": p["repairs"], "beneficiary_moved_into_cell":
                    dg["people"][b]["predicted_outcome"] == "moved"})
    return out


def converted_attribution(tp: TracedPlanner, dg: dict[str, Any]) -> Counter:
    """Why each move converted to stay was blocked in the last pre-conversion resolver call."""
    out: Counter = Counter()
    conv = dg["converted_to_stay"]
    if not conv:
        return out
    acts, res = next((a, r) for a, r in reversed(tp.t_resolves) if any(a[p] != "stay" for p in conv))
    occupant = {i: pid for pid, i in tp.pos.items()}
    for pid in conv:
        o = res[pid][0]
        why = o
        if o == "blocked_occupied":
            cur, seen = pid, set()
            while cur is not None and cur not in seen and acts.get(cur, "stay") != "stay":
                seen.add(cur)
                cur = occupant.get(tp.plans[cur][1])
            end = dg["people"].get(cur) if cur is not None else None
            if end is not None:
                why += "/chain_end_" + ("vacate_refused" if end["priority_effect"] == "vacate_refused" else
                                        "converted" if end["priority_effect"] == "converted_to_stay" else
                                        "chain_member" if cur in seen else end["priority_effect"])
        out[why + ("/repaired" if dg["people"][pid]["repairs"] else "/never_repaired")] += 1
    return out


def queue_order(tp: TracedPlanner) -> Counter:
    """Leader-follower pairs (B's static next cell is A's cell): who is planned first."""
    out: Counter = Counter()
    occupant = {i: pid for pid, i in tp.pos.items()}
    for pid in tp.order:
        if not tp.info[pid]["route_found"]:
            continue
        nxt = tp._intended(pid)
        lead = occupant.get(nxt) if nxt != tp.pos[pid] else None
        if lead is not None:
            out["pairs"] += 1
            out["follower_planned_first"] += tp.rank[pid] < tp.rank[lead]
    return out


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    def ranks(v):
        order = sorted(range(len(v)), key=v.__getitem__)
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    if len(xs) < 3:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    sx = sum((a - mx) ** 2 for a in rx) ** 0.5
    sy = sum((b - my) ** 2 for b in ry) ** 0.5
    return None if sx == 0 or sy == 0 else sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (sx * sy)


def run(out_path: Path = DEFAULT_OUT) -> dict[str, Any]:
    pilot = json.loads(PILOT_REPORT.read_text(encoding="utf-8"))
    rows = pilot["pilot"]
    cursor = {v: iter([r for r in rows if r["variant"] == v]) for v in VARIANTS}
    agg: dict[str, Any] = {}
    cells_out: list[dict[str, Any]] = []
    examples: dict[str, Any] = {}
    a2cfg = dict(ABLATIONS["A2"]().identity.config)
    for v in VARIANTS:
        a: Counter = Counter()
        conv: Counter = Counter()
        qo: Counter = Counter()
        rho: list[float] = []
        effects_by_stratum: Counter = Counter()
        for cell in pilot_cells():
            s = pilot_scenario(cell)
            sim = Simulator(s)
            tr = _Traced(v)
            ben_events: list[str] = []
            yld_events: list[str] = []
            c: Counter = Counter()
            n_dec = 0
            init_stratum: dict[str, str] = {}
            while not sim.terminated and (cell["mode"] == "full_episode" or n_dec < 12):
                obs = sim.observe()
                acts, dg = tr.decide(obs)
                ref = next(cursor[v])
                pr = dg["priority"]
                mine = (pr["yields_over_id_rule"], len(dg["converted_to_stay"]), dg["repair_rounds"],
                        pr["vacate_refused"], obs["step"])
                theirs = (ref["yields_over_id_rule"], ref["converted_to_stay"], ref["repair_rounds"],
                          ref["vacate_refused"], ref["step"])
                if mine != theirs:
                    raise AssertionError(f"traced planner diverged from the frozen pilot: {cell} {v} {mine} {theirs}")
                tp = tr.last
                n_dec += 1
                ev_rows = classify(tp, dg)
                for e in ev_rows:
                    c["over_id/" + e["class"]] += 1
                    if e["class"] not in examples.get(v, {}):
                        examples.setdefault(v, {})[e["class"]] = {"cell": cell, "step": obs["step"], **e}
                c["over_id"] += pr["yields_over_id_rule"]
                c["overwrites"] += len(tp.t_overwrites)
                c["overwrites_stay"] += sum(o["stay"] for o in tp.t_overwrites)
                c["overwrites_in_repair"] += sum(o["round"] > 0 for o in tp.t_overwrites)
                calls = tp.t_calls
                c["plan_calls"] += len(calls)
                c["replans"] += sum(x["replan"] for x in calls)
                c["calls_constrained_by_later"] += sum(x["next"] != x["counterfactual_next"] for x in calls)
                c["replans_constrained_by_later"] += sum(x["replan"] and x["next"] != x["counterfactual_next"]
                                                         for x in calls)
                c["repair_rounds"] += dg["repair_rounds"]
                c["decisions_with_repair"] += dg["repair_rounds"] > 0
                c["decisions_at_round_limit"] += dg["repair_rounds"] == tp.cfg["repair_rounds"]
                c["converted_to_stay"] += len(dg["converted_to_stay"])
                c["vacate_refused"] += pr["vacate_refused"]
                c["yields"] += pr["yields"]
                c["effective_yields_over_id_rule"] += pr["effective_yields_over_id_rule"]
                conv.update(converted_attribution(tp, dg))
                qo.update(queue_order(tp))
                d_rank = [(tp.rank[p], tp.info[p]["static_distance_target"]) for p in tp.order
                          if tp.info[p]["route_found"]]
                r = _spearman([x for x, _ in d_rank], [y for _, y in d_rank])
                if r is not None:
                    rho.append(r)
                strat = {pid: r["stratum"] for pid, r in assess(tp.pg, obs, a2cfg).people.items()}
                if n_dec == 1:
                    init_stratum = dict(strat)
                for pid, p in dg["people"].items():
                    effects_by_stratum[f"{strat[pid]}/{p['priority_effect']}"] += 1
                    if p["priority_effect"] in YIELD_EFFECTS and p["yield_over_id_rule"] and \
                            dg["people"][p["yielded_to"]]["predicted_outcome"] == "moved":
                        ben_events.append(p["yielded_to"])
                        yld_events.append(pid)
                sim.step(acts)
            a.update(c)
            if cell["mode"] == "full_episode":
                fin = {p["id"]: p["status"] for p in sim.private_state()["people"]}
                for pid, st in fin.items():
                    a[f"final_by_initial_stratum/{init_stratum.get(pid, 'n/a')}/{st}"] += 1
                bf = Counter(fin[b] for b in ben_events)
                yf = Counter(fin[y] for y in yld_events)
                for k, n in bf.items():
                    a[f"effective_over_id_beneficiary_final/{k}"] += n
                for k, n in yf.items():
                    a[f"effective_over_id_yielder_final/{k}"] += n
            cells_out.append({"variant": v, "kind": cell["kind"], "topology": cell["topology"],
                              "grid": list(cell["grid"]), "n": cell["n"], "model": cell["model"],
                              "exits": cell["exits"], "decisions": n_dec, **dict(sorted(c.items()))})
        agg[v] = {"totals": dict(sorted(a.items())), "converted_to_stay_attribution": dict(sorted(conv.items())),
                  "queue_leader_follower": dict(qo),
                  "rank_vs_target_distance_spearman_mean": statistics.fmean(rho) if rho else None,
                  "decisions_with_rho": len(rho),
                  "effects_by_risk_stratum": dict(sorted(effects_by_stratum.items()))}
    for v in VARIANTS:
        if next(cursor[v], None) is not None:
            raise AssertionError(f"{v}: fewer traced decisions than pilot rows")
    report = {"diag_version": DIAG_VERSION, "source_pilot": str(PILOT_REPORT.relative_to(ev.REPO)),
              "teachers": {v: cls().identity.config_hash for v, cls in ABLATIONS.items()},
              "reproduces_frozen_pilot_rows": True, "variants": agg, "cells": cells_out, "examples": examples,
              "environment": ev._revision(),
              "note": "observational traces generated by the policies themselves; exploratory, not confirmation"}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    rep = run(args.out)
    print(json.dumps({v: rep["variants"][v]["totals"] for v in VARIANTS}, indent=1))


if __name__ == "__main__":
    main()
