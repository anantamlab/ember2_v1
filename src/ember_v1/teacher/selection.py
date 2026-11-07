"""Five-teacher selection preparation (issue #25). It never selects a teacher.

1 `evidence_inventory` reads the committed DEV reports of P1-03..P1-06 and
  lists, per candidate T0..T4: identity hashes, scenarios, casualties,
  evacuated, unresolved and policy CPU. When a teacher appears in several
  reports, every final transition hash must agree (cross-report
  reproducibility). A candidate without DEV evidence is `missing`.
2 `freeze_protocol` validates a selection protocol (the required fields of
  docs/P1_TEACHER_SELECTION_PREP.md section 4) and adds `protocol_hash`.
3 `selection_gate` returns `NO_SELECTION` with every reason until a frozen
  protocol, committed before its evidence, covers all five candidates and the
  evidence is complete and consistent. Even then it returns only
  `READY_FOR_PREREGISTERED_COMPARISON`: the comparison itself runs under the
  frozen protocol, not here.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..sim.canonical import digest
from . import evaluate as ev

SELECTION_PREP_VERSION = "teacher-selection-prep@1.0.0"
CANDIDATES = ("T0", "T1", "T2", "T3", "T4")
DEFAULT_REPORTS = tuple(ev.REPO / f"data/baselines/{n}" for n in (
    "p1-03-t0-dev-v1.json", "p1-04-t1-dev-v1.json", "p1-05-t2-dev-v1.json", "p1-06-t3-dev-v1.json"))
PROTOCOL_FIELDS = ("protocol_id", "candidates", "data", "primary_endpoint", "secondary_endpoints", "comparisons",
                   "decision_rule", "simplicity_rule", "arbiter", "compute_budget", "stop_rules",
                   "excluded_data", "frozen")
CPU_CAP_USD = 15.0          # existing P1-P2 CPU cap (docs/P1_TEACHER_DIRECTION.md section 8)


def evidence_inventory(paths: Sequence[str | Path] = DEFAULT_REPORTS) -> dict[str, Any]:
    per: dict[str, dict[str, Any]] = {c: {"reports": [], "config_hashes": set(), "rows": {}} for c in CANDIDATES}
    conflicts: list[dict[str, Any]] = []
    dev_hashes: set[str] = set()
    for path in paths:
        rep = json.loads(Path(path).read_text(encoding="utf-8"))
        name = Path(path).name
        dev_hashes.add(digest(rep.get("dev")))
        for t, ident in rep.get("policies", {}).items():
            if t not in per:
                continue
            per[t]["reports"].append(name)
            per[t]["config_hashes"].add(ident["config_hash"])
            for r in rep["rows"]:
                x = r[t]
                cur = {"final_transition_hash": x.get("final_transition_hash"), "casualties": x["casualties"],
                       "evacuated": x["evacuated"], "unresolved": x["unresolved"], "people": x["people"],
                       "failure": x.get("failure"), "family_id": r.get("family_id"),   # absent in P1-03 rows
                       "policy_cpu_ms": (x.get("timing") or {}).get("policy_cpu_ms")}
                old = per[t]["rows"].get(r["position"])
                if old is None:
                    per[t]["rows"][r["position"]] = cur
                elif old["final_transition_hash"] != cur["final_transition_hash"]:
                    conflicts.append({"teacher": t, "position": r["position"], "report": name})
    fam = {pos: r["family_id"] for p in per.values() for pos, r in p["rows"].items() if r["family_id"]}
    out: dict[str, Any] = {}
    for t, p in per.items():
        rows = p["rows"]
        for pos, r in rows.items():
            r["family_id"] = r["family_id"] or fam.get(pos)
        out[t] = {"status": "dev_evidence" if rows else "missing", "reports": p["reports"],
                  "config_hashes": sorted(p["config_hashes"]), "scenarios": len(rows),
                  "families": len({r["family_id"] for r in rows.values()}),
                  "casualties": sum(r["casualties"] for r in rows.values()),
                  "evacuated": sum(r["evacuated"] for r in rows.values()),
                  "unresolved": sum(r["unresolved"] for r in rows.values()),
                  "people": sum(r["people"] for r in rows.values()),
                  "failures": sum(r["failure"] is not None for r in rows.values()),
                  "policy_cpu_s": (None if not rows or any(r["policy_cpu_ms"] is None for r in rows.values())
                                   else sum(r["policy_cpu_ms"] for r in rows.values()) / 1000),
                  "positions": sorted(rows)}
    pos = {t: tuple(v["positions"]) for t, v in out.items() if v["status"] != "missing"}
    return {"version": SELECTION_PREP_VERSION, "reports": [Path(p).name for p in paths], "candidates": out,
            "cross_report_conflicts": conflicts, "same_dev_metadata": len(dev_hashes) == 1,
            "same_positions": len(set(pos.values())) <= 1,
            "note": "descriptive inventory; separate reports are not one paired comparison"}


def freeze_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a selection protocol and add its hash. Raise ValueError on any missing or unsafe field."""
    p = dict(protocol)
    missing = [f for f in PROTOCOL_FIELDS if f not in p]
    if missing:
        raise ValueError(f"protocol misses {missing}")
    if tuple(p["candidates"]) != CANDIDATES:
        raise ValueError("protocol must name all five candidates T0..T4 separately")
    if p["frozen"] is not True:
        raise ValueError("protocol must be frozen")
    excluded = set(p["excluded_data"])
    for must in ("locked_test", "blind_test", "emberbench"):
        if must not in excluded:
            raise ValueError(f"protocol must exclude {must}")
    budget = p["compute_budget"]
    if not isinstance(budget, Mapping) or not 0 < float(budget.get("usd_cap", 0)) <= CPU_CAP_USD:
        raise ValueError(f"compute budget must be a positive cap within the existing ${CPU_CAP_USD:g} CPU cap")
    arb = p["arbiter"]
    if arb is not None and not (isinstance(arb, Mapping) and arb.get("equal_compute_to_best_single") is True
                                and arb.get("no_score_averaging") is True and arb.get("joint_resolver_check") is True):
        raise ValueError("an arbiter needs equal compute, no score averaging and a joint resolver check")
    if not p["stop_rules"]:
        raise ValueError("protocol needs stop rules")
    p.pop("protocol_hash", None)
    return {**p, "protocol_hash": digest(p)}


def selection_gate(protocol: Mapping[str, Any] | None, inventory: Mapping[str, Any], *,
                   protocol_commit: str | None, evidence_after_commit: bool) -> dict[str, Any]:
    """`NO_SELECTION` with reasons, or `READY_FOR_PREREGISTERED_COMPARISON`. Never a selected teacher."""
    reasons = []
    if protocol is None:
        reasons.append("no frozen selection protocol")
    else:
        body = {k: v for k, v in protocol.items() if k != "protocol_hash"}
        try:
            ok = freeze_protocol(body)["protocol_hash"] == protocol.get("protocol_hash")
        except ValueError as e:
            ok = False
            reasons.append(f"protocol invalid: {e}")
        if not ok:
            reasons.append("protocol hash does not match")
        if not protocol_commit:
            reasons.append("protocol commit not recorded")
        if not evidence_after_commit:
            reasons.append("evidence was not produced after the protocol commit")
    for t in CANDIDATES:
        c = inventory["candidates"].get(t, {})
        if c.get("status") != "dev_evidence":
            reasons.append(f"{t}: no DEV evidence")
        elif len(c.get("config_hashes", ())) != 1:
            reasons.append(f"{t}: more than one identity in the evidence")
        elif c.get("failures"):
            reasons.append(f"{t}: failed episodes in the evidence")
    if inventory.get("cross_report_conflicts"):
        reasons.append("cross-report reproducibility conflicts")
    if not inventory.get("same_dev_metadata") or not inventory.get("same_positions"):
        reasons.append("evidence does not cover the same DEV scenarios")
    return {"decision": "NO_SELECTION" if reasons else "READY_FOR_PREREGISTERED_COMPARISON", "reasons": reasons,
            "selected_teacher": None}


def summary_by_teacher(inventory: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    keys = ("status", "scenarios", "casualties", "evacuated", "unresolved", "policy_cpu_s")
    return {t: {k: c[k] for k in keys} for t, c in inventory["candidates"].items()}
