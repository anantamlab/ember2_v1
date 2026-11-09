"""Multi-teacher outcome-aware dataset records and CPU validators (P1-14, issue #47).

DRAFT PROTOCOL, FIXTURE SCOPE ONLY. `export` always refuses: real dataset
generation needs a FROZEN protocol, owner authorization and the P1-12 TRAIN gate.

Evidence classes (`data/protocols/p1-14-multiteacher-dataset.json`):
  A successful demonstration   - teacher trajectory tied best in a complete matched group
  B unsuccessful demonstration - every other complete trajectory; contrastive only
  C counterfactual preference  - verified same-state contrast that passes the thresholds
  D inconclusive intervention  - every other verified contrast; never in a preference loss
  E student-visited state      - reserved for P1-16 (#49); refused here

Not every teacher action is equally good: class comes from matched outcome
evidence, never from the teacher's name. No action probability, confidence or
causal label is ever created. Teacher values appear only where T2 computed them.
Weights are auditable: one budget per scenario, equal split inside it,
uninformative scenarios down-weighted, teacher and family shares reported and capped.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import counterfactual as cf
from . import multiteacher as mt
from .sim import Scenario
from .sim.canonical import digest
from .teacher.registry import VALUE_TEACHERS

RECORD_VERSION = "mt-dataset-record@0.1.0"
REPO = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO / "data" / "protocols" / "p1-14-multiteacher-dataset.json"
CLASSES = ("A", "B", "C", "D", "E")
USE = {"A": "sft_positive", "B": "contrastive_only", "C": "preference_pair", "D": "excluded_inconclusive",
       "E": "reserved_blocked_until_p1-16"}
_COMMON = {"record_version", "class", "use", "scope", "scenario_hash", "family_id", "source", "observation",
           "observation_digest", "weight", "record_hash"}
_KEYS = {
    "A": _COMMON | {"teachers", "actions", "proposal_hashes", "teacher_values", "step", "evidence"},
    "B": _COMMON | {"teachers", "actions", "proposal_hashes", "teacher_values", "step", "evidence"},
    "C": _COMMON | {"chosen", "rejected", "evidence"},
    "D": _COMMON | {"alternatives", "evidence"},
}


class DatasetError(ValueError):
    """Record or dataset breaks the P1-14 protocol."""


def load_protocol(path: str | Path = PROTOCOL_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _seal(rec: dict[str, Any]) -> dict[str, Any]:
    body = {k: v for k, v in rec.items() if k != "record_hash"}
    return {**body, "record_hash": digest(body)}


def _forbidden_paths(x: Any, forbidden: frozenset[str], path: str = "") -> list[str]:
    out: list[str] = []
    if isinstance(x, Mapping):
        for k, v in x.items():
            if k in forbidden:
                out.append(f"{path}/{k}")
            out.extend(_forbidden_paths(v, forbidden, f"{path}/{k}"))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            out.extend(_forbidden_paths(v, forbidden, f"{path}[{i}]"))
    return out


# ----- builders ----------------------------------------------------------------------------------
def _family(report: Mapping[str, Any]) -> str:
    src = report["provenance"]["scenario"]["source"] or {}
    return str(src.get("family_id") or report["scenario_hash"])


def demonstrations(report: Mapping[str, Any], scenario: Scenario,
                   protocol: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Class A/B records from one verified, complete five-teacher group. Unweighted (weight None)."""
    protocol = protocol or load_protocol()
    mt.verify_fixture(report, scenario)
    if not report["group_complete"]:
        raise DatasetError("incomplete five-teacher group: no demonstration records (reported, not used)")
    outs = {n: r["outcome"] for n, r in report["policies"].items()}
    best_p = min(o["casualties_plus_unresolved"] for o in outs.values())
    best_c = min(o["casualties"] for o in outs.values())
    cls = {n: "A" if o["casualties_plus_unresolved"] == best_p and o["casualties"] == best_c else "B"
           for n, o in outs.items()}
    uninformative = len({(o["casualties_plus_unresolved"], o["casualties"]) for o in outs.values()}) == 1
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    for name, roll in report["policies"].items():
        for d in roll["decisions"]:
            prop = d["proposal"]
            key = (cls[name], d["observation_digest"], digest(prop["actions"]))
            rec = merged.get(key)
            if rec is None:
                rec = merged[key] = {
                    "record_version": RECORD_VERSION, "class": cls[name], "use": USE[cls[name]],
                    "scope": report["scope"], "scenario_hash": report["scenario_hash"], "family_id": _family(report),
                    "source": {"kind": "p1-12-fixture", "content_hash": report["content_hash"]},
                    "observation": d["observation"], "observation_digest": d["observation_digest"],
                    "teachers": [], "actions": dict(prop["actions"]), "proposal_hashes": [],
                    "teacher_values": None, "step": d["observation"]["step"],
                    "evidence": {"group_best_primary": best_p, "group_best_casualties": best_c,
                                 "uninformative_scenario": uninformative, "trajectory_outcomes": {}},
                    "weight": None,
                }
            rec["teachers"].append(name)
            rec["proposal_hashes"].append(prop["proposal_hash"])
            rec["evidence"]["trajectory_outcomes"][name] = {
                k: outs[name][k] for k in ("casualties", "unresolved", "evacuated", "casualties_plus_unresolved")}
            if name in VALUE_TEACHERS and prop["values"] is not None:
                rec["teacher_values"] = {name: prop["values"]}       # computed by T2; copied, never created
    return [_seal(r) for r in merged.values()]


def preferences(study: Mapping[str, Any], scenario: Scenario,
                protocol: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Class C/D records from one verified `counterfactual.shared_state_study` result."""
    protocol = protocol or load_protocol()
    if study.get("status") != "contrasted":
        return []
    rep = study["report"]
    cf.verify_counterfactual(rep, scenario)
    rule = protocol["preference_rule"]
    stochastic = rep["fire_model"] != "deterministic"
    min_rep = rule["min_replicates_stochastic"] if stochastic else 1
    min_win = rule["min_strict_improvements_stochastic"] if stochastic else 1
    probe = study["probe"]
    common = {"record_version": RECORD_VERSION, "scope": rep["scope"], "scenario_hash": rep["scenario_hash"],
              "family_id": str((rep["prefix_source"] or {}).get("family_id") or rep["scenario_hash"]),
              "observation_digest": rep["common_observation_digest"], "weight": None}
    observation = cf._replay_prefix(scenario, rep["prefix_records"]).observe()
    if digest(observation) != rep["common_observation_digest"] or probe["observation_digest"] != digest(observation):
        raise DatasetError("study observation does not match its report")
    out = []
    for c in rep["contrasts"]:
        wins = c["summary"]["b_better"] if c["verdict"] == "b_better" else c["summary"]["a_better"]
        evidence = {"report_hash": rep["report_hash"], "verdict": c["verdict"], "reason": c["reason"],
                    "replicates": rep["replicates"], "strict_improvements": wins if c["preference_candidate"] else 0,
                    "horizon": rep["horizon"], "censored": c["censored"], "continuation": rep["continuation"],
                    "per_replicate": c["per_replicate"], "causality_limit": rep["causality_limit"]}
        ok = c["preference_candidate"] and not c["censored"] and rep["replicates"] >= min_rep and wins >= min_win
        src = {"kind": "p1-13-study", "report_hash": rep["report_hash"], "prefix_source": rep["prefix_source"]}
        if ok:
            good, bad = (c["b"], c["a"]) if c["verdict"] == "b_better" else (c["a"], c["b"])
            rec = {**common, "class": "C", "use": USE["C"], "source": src, "observation": observation,
                   "chosen": _alt(rep, good), "rejected": _alt(rep, bad), "evidence": evidence}
        else:
            rec = {**common, "class": "D", "use": USE["D"], "source": src, "observation": observation,
                   "alternatives": [_alt(rep, c["a"]), _alt(rep, c["b"])],
                   "evidence": {**evidence, "below_threshold": c["preference_candidate"]}}
        out.append(_seal(rec))
    return out


def _alt(rep: Mapping[str, Any], name: str) -> dict[str, Any]:
    b = rep["branches"][name]
    return {"name": name, "actions": b["intervention_actions"], "sources": b["sources"],
            "outcome_replicate0": b["outcome"]}


# ----- weighting ---------------------------------------------------------------------------------
def assign_weights(records: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any] | None = None
                   ) -> list[dict[str, Any]]:
    """One budget per (scenario, class), equal split; uninformative scenarios scaled down. Re-sealed."""
    protocol = protocol or load_protocol()
    w = protocol["weighting"]
    by = defaultdict(list)
    for r in records:
        by[(r["scenario_hash"], r["class"])].append(r)
    out = []
    for (_, k), rs in sorted(by.items()):
        factor = w["uninformative_scenario_factor"] if k in ("A", "B") and \
            rs[0]["evidence"].get("uninformative_scenario") else 1.0
        each = 0.0 if k in ("D", "E") else w["scenario_budget"] * factor / len(rs)
        out.extend(_seal({**r, "weight": round(each, 12)}) for r in rs)
    return out


def audit(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Class counts, weight by class, teacher shares of class A weight, family shares."""
    teacher = Counter()
    family = Counter()
    for r in records:
        if r["class"] == "A":
            for t in r["teachers"]:
                teacher[t] += r["weight"] / len(r["teachers"])
        if r["class"] in ("A", "C"):
            family[r["family_id"]] += r["weight"]
    ta, fa = sum(teacher.values()), sum(family.values())
    return {"classes": dict(sorted(Counter(r["class"] for r in records).items())),
            "weight_by_class": {k: round(sum(r["weight"] for r in records if r["class"] == k), 9) for k in CLASSES},
            "scenarios": len({r["scenario_hash"] for r in records}),
            "teacher_share_A": {t: round(v / ta, 9) for t, v in sorted(teacher.items())} if ta else {},
            "family_share": {f: round(v / fa, 9) for f, v in sorted(family.items())} if fa else {}}


# ----- validators --------------------------------------------------------------------------------
def record_problems(r: Mapping[str, Any], protocol: Mapping[str, Any]) -> list[str]:
    errs: list[str] = []
    k = r.get("class")
    if k == "E":
        return ["class E (student-visited) is reserved until P1-16 (#49) is authorized"]
    if k not in _KEYS:
        return [f"unknown class {k!r}"]
    if set(r) != _KEYS[k]:
        errs.append(f"class {k}: fields differ: missing {sorted(_KEYS[k] - set(r))}, extra {sorted(set(r) - _KEYS[k])}")
        return errs
    if r["record_version"] != RECORD_VERSION or r["use"] != USE[k]:
        errs.append(f"class {k}: wrong version or use")
    if r["scope"] not in protocol["scope"]["allowed_now"]:
        errs.append(f"scope {r['scope']!r} is not allowed by a non-frozen protocol")
    if digest({x: v for x, v in r.items() if x != "record_hash"}) != r["record_hash"]:
        errs.append("record hash mismatch")
    if digest(r["observation"]) != r["observation_digest"]:
        errs.append("observation digest mismatch")
    try:
        mt.check_public_observation(r["observation"])
    except mt.MultiTeacherError as e:
        errs.append(str(e))
    bad = _forbidden_paths({x: v for x, v in r.items() if x != "observation"}, frozenset(protocol["forbidden_fields"]))
    if bad:
        errs.append(f"fabricated or forbidden fields: {bad[:5]}")
    if not isinstance(r["weight"], (int, float)) or isinstance(r["weight"], bool) or r["weight"] < 0:
        errs.append("weight must be assigned and non-negative")
    elif k == "D" and r["weight"] != 0:
        errs.append("class D must have zero training weight")
    active = {p["id"] for p in r["observation"]["people"] if p["status"] == "active"}
    if k in ("A", "B"):
        if set(r["actions"]) != active:
            errs.append("demonstration joint action is not complete")
        if not r["teachers"] or len(r["teachers"]) != len(r["proposal_hashes"]):
            errs.append("each proposing teacher needs its proposal hash")
        tv = r["teacher_values"]
        if tv is not None and (set(tv) - VALUE_TEACHERS or not set(tv) <= set(r["teachers"])):
            errs.append("teacher values from a teacher that does not compute them")
        ev = r["evidence"]
        outs = ev.get("trajectory_outcomes", {})
        for t in r["teachers"]:
            o = outs.get(t)
            if o is None:
                errs.append(f"no outcome evidence for {t}")
                continue
            best = o["casualties_plus_unresolved"] == ev["group_best_primary"] and \
                o["casualties"] == ev["group_best_casualties"]
            if (k == "A") is not best:
                errs.append(f"class {k} does not follow the outcome rule for {t}")
    if k == "C":
        rule = protocol["preference_rule"]
        ev = r["evidence"]
        need_rep = rule["min_replicates_stochastic"] if ev["replicates"] > 1 else 1
        need_win = rule["min_strict_improvements_stochastic"] if ev["replicates"] > 1 else 1
        if ev["verdict"] not in ("a_better", "b_better") or ev["censored"]:
            errs.append("class C needs an uncensored a_better/b_better verdict")
        if ev["replicates"] < need_rep or ev["strict_improvements"] < need_win:
            errs.append("class C is below the replicate or strict-improvement threshold")
        for side in ("chosen", "rejected"):
            if set(r[side]["actions"]) != active:
                errs.append(f"{side} joint action is not complete")
        if r["chosen"]["actions"] == r["rejected"]["actions"]:
            errs.append("chosen and rejected are identical")
    if k == "D" and r["evidence"]["verdict"] in ("a_better", "b_better") and not r["evidence"].get("below_threshold"):
        errs.append("a class D record with a passing verdict must be class C")
    return errs


def validate_dataset(records: Iterable[Mapping[str, Any]], protocol: Mapping[str, Any] | None = None
                     ) -> dict[str, Any]:
    protocol = protocol or load_protocol()
    records = list(records)
    problems = [f"{i}: {e}" for i, r in enumerate(records) for e in record_problems(r, protocol)]
    def content_key(r: Mapping[str, Any]) -> tuple:
        k = r.get("class")
        what = {"A": ("actions",), "B": ("actions",), "C": ("chosen", "rejected"), "D": ("alternatives",)}
        sides = [r.get(f) for f in what.get(k, ())]
        acts = [s["actions"] if isinstance(s, Mapping) and "actions" in s else
                [x.get("actions") for x in s] if isinstance(s, list) else s for s in sides]
        return k, r.get("observation_digest"), digest(acts)
    seen = Counter(content_key(r) for r in records)
    problems += [f"duplicate record {k}" for k, n in seen.items() if n > 1]
    hashes = Counter(r.get("record_hash") for r in records)
    problems += [f"duplicate record hash {h}" for h, n in hashes.items() if n > 1]
    if problems:
        return {"ok": False, "problems": problems}
    a = audit(records)
    w = protocol["weighting"]
    if a["scenarios"] >= w["min_scenarios_for_share_caps"]:
        problems += [f"teacher {t} share {s} > {w['max_teacher_share_A']}"
                     for t, s in a["teacher_share_A"].items() if s > w["max_teacher_share_A"]]
        problems += [f"family {f} share {s} > {w['max_family_share']}"
                     for f, s in a["family_share"].items() if s > w["max_family_share"]]
    return {"ok": not problems, "problems": problems, "audit": a}


def export(*_: Any, protocol: Mapping[str, Any] | None = None, authorization: str | None = None) -> None:
    """Real dataset generation. Refused: protocol not frozen, no authorization, no TRAIN gate."""
    protocol = protocol or load_protocol()
    reasons = []
    if protocol.get("status") != "FROZEN":
        reasons.append("protocol is not FROZEN")
    if not authorization:
        reasons.append("no owner authorization")
    reasons.append("P1-12 TRAIN collection gate is not implemented or approved")
    raise DatasetError("DATASET_GENERATION_BLOCKED: " + "; ".join(reasons))
