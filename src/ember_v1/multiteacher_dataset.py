"""Multi-teacher outcome-aware dataset records and CPU validators (P1-14, issue #47).

Protocol 1.0.0 (FROZEN, run 17): scopes `synthetic_fixture` and `p1-17-train-full@1`. `export` is the gate
for real records: FROZEN protocol, owner authorization and the collection gate (verified 189/189 collection).
`train_dataset` builds the real records outside git and writes them only if `validate_dataset` passes.
T0 is diagnostic (never class A); maps with a move into a burning cell are never class A; weights are per
family; preference approval stays disabled (every class C is weight 0).

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

import copy
import json
from fractions import Fraction
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import counterfactual as cf
from . import multiteacher as mt
from .sim import Scenario
from .sim.canonical import digest
from .teacher.registry import VALUE_TEACHERS

RECORD_VERSION = "mt-dataset-record@1.0.0"
REPO = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO / "data" / "protocols" / "p1-14-multiteacher-dataset.json"
CLASSES = ("A", "B", "C", "D", "E")
USE = {"A": "sft_positive", "B": "contrastive_only", "C": "preference_candidate", "D": "excluded_inconclusive",
       "E": "reserved_blocked_until_p1-16"}
JOINT_TARGET_RULE = "p1-14-whole-joint-target@0.2.0"
JOINT_TARGET_MODES = ("select_one", "sample_whole_maps")      # select_one = Stage A initial approach
EXCLUSIONS = {
    "A": "competing_class_A_joint_map_at_same_state",
    "B": "contrast_only_not_a_positive_target",
    "C": "preference_candidate_not_approved",
    "D": "inconclusive_comparison",
}
APPROVAL_CHECKS = ("shared_initial_state", "controlled_continuation", "matched_exogenous_fire", "uncensored",
                   "sufficient_replicates", "sufficient_strict_improvements", "no_replicate_worse",
                   "no_unresolved_increase")
_COMMON = {"record_version", "class", "use", "scope", "scenario_hash", "family_id", "lineage_id", "source",
           "provenance", "observation", "observation_digest", "selection", "weight", "exclusion_reason",
           "record_hash"}
_KEYS = {
    "A": _COMMON | {"teachers", "actions", "proposal_hashes", "teacher_values", "step", "evidence", "replay",
                    "step_diagnostics"},
    "B": _COMMON | {"teachers", "actions", "proposal_hashes", "teacher_values", "step", "evidence", "replay",
                    "step_diagnostics"},
    "C": _COMMON | {"chosen", "rejected", "evidence", "approval"},
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


def _lineage(source: Mapping[str, Any] | None, fallback: str) -> str:
    src = source or {}
    return str(src.get("lineage_id") or src.get("family_id") or fallback)


_TRAJECTORY_KEYS = ("casualties", "unresolved", "evacuated", "casualties_plus_unresolved", "blocked_moves",
                    "contested_targets", "peak_target_contention", "steps", "termination_reason")


def diagnostic_teachers(protocol: Mapping[str, Any]) -> frozenset[str]:
    return frozenset((protocol.get("demonstration_rule") or {}).get("diagnostic_teachers") or ())


def class_reason(outcome: Mapping[str, Any], teacher: str, fire_entries: Sequence[str], best_primary: int,
                 best_casualties: int, diagnostic: frozenset[str]) -> str:
    """Preregistered class rule for one teacher decision (protocol demonstration_rule.class_reasons)."""
    if not (outcome["casualties_plus_unresolved"] == best_primary and outcome["casualties"] == best_casualties):
        return "outcome_not_best"
    if teacher in diagnostic:
        return "diagnostic_teacher"
    if fire_entries:
        return "unsafe_fire_entry"
    return "best_outcome"


def demonstrations(report: Mapping[str, Any], scenario: Scenario,
                   protocol: Mapping[str, Any] | None = None, *, family_id: str | None = None,
                   teacher_dev_similar: bool = False) -> list[dict[str, Any]]:
    """Class A/B records from one verified, complete five-teacher group. Unweighted (weight None).

    Every record holds one COMPLETE joint map that a teacher proposed at that exact state (merged when several
    teachers proposed the identical map). Class A records still need `select_joint_targets`.
    Class A needs the group-best outcome (over all five teachers), a non-diagnostic teacher (T0 is diagnostic)
    and a map with no move into a burning cell. `family_id` overrides the scenario source (real TRAIN groups
    carry the family in the collection plan, not in the scenario source)."""
    from .train_collect import fire_entry_moves
    protocol = protocol or load_protocol()
    mt.verify_fixture(report, scenario)
    if not report["group_complete"]:
        raise DatasetError("incomplete five-teacher group: no demonstration records (reported, not used)")
    diag = diagnostic_teachers(protocol)
    outs = {n: r["outcome"] for n, r in report["policies"].items()}
    best_p = min(o["casualties_plus_unresolved"] for o in outs.values())
    best_c = min(o["casualties"] for o in outs.values())
    uninformative = len({(o["casualties_plus_unresolved"], o["casualties"]) for o in outs.values()}) == 1
    prov = report["provenance"]
    family = family_id or _family(report)
    kind = "p1-12-fixture" if report["scope"] == mt.SCOPE_FIXTURE else "matched-group"
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    for name, roll in report["policies"].items():
        for d in roll["decisions"]:
            prop = d["proposal"]
            fire = fire_entry_moves(d)
            why = class_reason(outs[name], name, fire, best_p, best_c, diag)
            k = "A" if why == "best_outcome" else "B"
            key = (k, d["observation_digest"], digest(prop["actions"]))
            rec = merged.get(key)
            if rec is None:
                rec = merged[key] = {
                    "record_version": RECORD_VERSION, "class": k, "use": USE[k],
                    "scope": report["scope"], "scenario_hash": report["scenario_hash"], "family_id": family,
                    "lineage_id": _lineage(prov["scenario"]["source"], family),
                    "source": {"kind": kind, "content_hash": report["content_hash"]},
                    "provenance": {"report_content_hash": report["content_hash"], "simulator": prov["simulator"],
                                   "collector": prov["collector"], "scenario": prov["scenario"], "teachers": {}},
                    "observation": d["observation"], "observation_digest": d["observation_digest"],
                    "teachers": [], "actions": dict(prop["actions"]), "proposal_hashes": [],
                    "teacher_values": None, "step": d["observation"]["step"],
                    "evidence": {"group_best_primary": best_p, "group_best_casualties": best_c,
                                 "uninformative_scenario": uninformative, "trajectory_outcomes": {},
                                 "class_reasons": {}, "fire_entry_moves": fire,
                                 "teacher_dev_similar_family": bool(teacher_dev_similar)},
                    "replay": {}, "step_diagnostics": d["diagnostics"],
                    "selection": None if k == "A" else
                    {"rule": JOINT_TARGET_RULE, "selected": False, "reason": "class B: contrast and probe queue only"},
                    "weight": None, "exclusion_reason": None,
                }
            rec["teachers"].append(name)
            rec["proposal_hashes"].append(prop["proposal_hash"])
            rec["provenance"]["teachers"][name] = prov["teachers"][name]
            rec["replay"][name] = {**roll["replay"], "final_transition_hash": roll["final_transition_hash"]}
            rec["evidence"]["trajectory_outcomes"][name] = {x: outs[name][x] for x in _TRAJECTORY_KEYS}
            rec["evidence"]["class_reasons"][name] = why
            if name in VALUE_TEACHERS and prop["values"] is not None:
                rec["teacher_values"] = {name: prop["values"]}       # computed by T2; copied, never created
    return [_seal(r) for r in merged.values()]


def _mean(xs: Sequence[int]) -> Fraction:
    return Fraction(sum(xs), len(xs))


def _target_key(r: Mapping[str, Any], wins: Mapping[str, int]) -> tuple:
    """Preregistered order (lower is better). Every criterion is training evidence; no teacher name is used.

    Criteria 3 and 4 use the MEAN over the proposing teachers' trajectories (exact fractions). A minimum would
    favour a map that more teachers proposed (more draws, lower minimum), so a common T0/T1 map would beat a
    rare T2/T3/T4 map by count alone. Neither the number of proposers nor teacher CPU cost is a criterion."""
    outs = r["evidence"]["trajectory_outcomes"]
    return (-wins.get(digest(r["actions"]), 0),                                 # 1 approved same-state preference
            r["step_diagnostics"]["blocked_moves"],                             # 2 joint-map coherence at this state
            _mean([outs[t]["blocked_moves"] for t in r["teachers"]]),           # 3 trajectory congestion (mean)
            _mean([outs[t]["steps"] for t in r["teachers"]]),                   # 4 earlier completion (mean)
            digest(r["actions"]))                                               # 5 name-free deterministic tie-break


_TARGET_CRITERIA = ("approved_same_state_preference", "blocked_moves_at_state", "mean_trajectory_blocked_moves",
                    "mean_trajectory_steps", "action_digest_tie_break")


def target_mode(protocol: Mapping[str, Any]) -> str:
    mode = (protocol.get("joint_target_rule") or {}).get("mode")
    if mode not in JOINT_TARGET_MODES:
        raise DatasetError(f"protocol joint_target_rule.mode {mode!r} is not one of {JOINT_TARGET_MODES}")
    return mode


def select_joint_targets(records: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any] | None = None
                         ) -> list[dict[str, Any]]:
    """Whole-joint demonstration selection under the protocol's preregistered `joint_target_rule.mode`.

    `select_one` (Stage A initial approach): at each state exactly ONE complete class-A joint map is the SFT
    target. Competing class-A maps at the same state are kept (provenance, weight 0) and never mixed person by
    person. Approved class C preferences between two competing maps decide first.
    `sample_whole_maps` (preregistered, not active for Stage A): every distinct complete class-A map at a state
    is selected as a whole demonstration; `assign_weights` splits the state's mass EQUALLY over the distinct
    maps, so a map is not favoured because more teachers proposed it. The ranking is still recorded."""
    mode = target_mode(protocol or load_protocol())
    recs = [dict(r) for r in records]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        if r["class"] == "A":
            groups[(r["scenario_hash"], r["observation_digest"])].append(r)
    for (sh, od), rs in groups.items():
        maps = {digest(r["actions"]) for r in rs}
        wins: Counter = Counter()
        for c in recs:
            if c["class"] == "C" and c["scenario_hash"] == sh and c["observation_digest"] == od and \
                    (c.get("approval") or {}).get("status") == "approved":
                w, lo = digest(c["chosen"]["actions"]), digest(c["rejected"]["actions"])
                if w in maps and lo in maps:
                    wins[w] += 1
        ranked = sorted(rs, key=lambda r: _target_key(r, wins))
        competitors = sorted(maps)
        if len(ranked) == 1:
            decided = "only_class_A_map_at_state"
        else:
            a, b = _target_key(ranked[0], wins), _target_key(ranked[1], wins)
            decided = next(name for name, x, y in zip(_TARGET_CRITERIA, a, b) if x != y)
        if mode == "sample_whole_maps" and len(ranked) > 1:
            decided = "sample_whole_maps"
        for i, r in enumerate(ranked):
            chosen = i == 0 or mode == "sample_whole_maps"
            if not chosen:
                why = "another complete class-A map was selected at this state; never mixed per person"
            elif mode == "select_one":
                why = "whole joint map selected as the SFT target"
            else:
                why = f"one of {len(maps)} distinct complete class-A maps, sampled as a whole; never mixed per person"
            r["selection"] = {"rule": JOINT_TARGET_RULE, "mode": mode, "selected": chosen, "rank": i,
                              "decided_by": decided, "distinct_maps_at_state": len(maps),
                              "competing_map_digests": [h for h in competitors if h != digest(r["actions"])],
                              "reason": why}
    return [_seal(r) for r in recs]


def preferences(study: Mapping[str, Any], scenario: Scenario, fixture_report: Mapping[str, Any],
                protocol: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Class C/D records from one `counterfactual.shared_state_study` result.

    Complete shared-state verification first (`cf.verify_shared_state_study` against the SOURCE fixture):
    prefix, probe, common observation and alternatives are bound, and every outcome, delta, censoring flag and
    verdict is re-derived from replay. A stored label is never trusted."""
    protocol = protocol or load_protocol()
    cf.verify_shared_state_study(study, scenario, fixture_report)
    if study.get("status") != "contrasted":
        return []
    rep = study["report"]
    rule = protocol["preference_rule"]
    stochastic = rep["fire_model"] != "deterministic"
    min_rep = rule["min_replicates_stochastic"] if stochastic else 1
    min_win = rule["min_strict_improvements_stochastic"] if stochastic else 1
    probe = study["probe"]
    family = str((rep["prefix_source"] or {}).get("family_id") or rep["scenario_hash"])
    common = {"record_version": RECORD_VERSION, "scope": rep["scope"], "scenario_hash": rep["scenario_hash"],
              "family_id": family, "lineage_id": _lineage(rep["prefix_source"], family),
              "observation_digest": rep["common_observation_digest"], "weight": None, "exclusion_reason": None,
              "provenance": {"report_hash": rep["report_hash"], "counterfactual_version": rep["counterfactual_version"],
                             "prefix_hash": rep["prefix_hash"], "prefix_source": rep["prefix_source"],
                             "continuation": rep["continuation"], "fire_model": rep["fire_model"]}}
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
            checks = approval_checks(rep, c, observation, min_rep, min_win)
            rec = {**common, "class": "C", "use": USE["C"], "source": src, "observation": observation,
                   "chosen": _alt(rep, good), "rejected": _alt(rep, bad), "evidence": evidence,
                   "approval": approval(checks, protocol),
                   "selection": {"rule": JOINT_TARGET_RULE, "selected": None,
                                 "reason": "preference candidate; enters a loss only after approval"}}
        else:
            rec = {**common, "class": "D", "use": USE["D"], "source": src, "observation": observation,
                   "alternatives": [_alt(rep, c["a"]), _alt(rep, c["b"])],
                   "evidence": {**evidence, "below_threshold": c["preference_candidate"]},
                   "selection": {"rule": JOINT_TARGET_RULE, "selected": False,
                                 "reason": "inconclusive comparison: kept and counted, never trained on"}}
        out.append(_seal(rec))
    return out


def approval_checks(rep: Mapping[str, Any], c: Mapping[str, Any], observation: Mapping[str, Any],
                    min_rep: int, min_win: int) -> dict[str, bool]:
    """Preregistered requirements of a FUTURE approved preference (all must hold). Signs are from the chosen
    side: for `b_better` every b-minus-a delta must be <= 0, for `a_better` >= 0."""
    sign = -1 if c["verdict"] == "b_better" else 1
    per = c["per_replicate"]
    wins = c["summary"]["b_better"] if c["verdict"] == "b_better" else c["summary"]["a_better"]
    return {
        "shared_initial_state": digest(observation) == rep["common_observation_digest"],
        "controlled_continuation": all(b["continuation"] == rep["continuation"] for b in rep["branches"].values()),
        "matched_exogenous_fire": rep.get("matched_exogenous_fire_prefix") is True,
        "uncensored": not c["censored"] and not any(r["censored"] for r in per),
        "sufficient_replicates": rep["replicates"] >= min_rep,
        "sufficient_strict_improvements": wins >= min_win,
        "no_replicate_worse": all(sign * r["delta_casualties_plus_unresolved"] >= 0 and
                                  sign * r["delta_casualties"] >= 0 for r in per),
        "no_unresolved_increase": all(sign * r["delta_unresolved"] >= 0 for r in per),
    }


def approval(checks: Mapping[str, bool], protocol: Mapping[str, Any]) -> dict[str, Any]:
    """A candidate is APPROVED only when every check holds AND the protocol is FROZEN with approval enabled.
    The draft protocol never approves: every class C record stays a candidate."""
    pa = protocol.get("preference_approval") or {}
    enabled = protocol.get("status") == "FROZEN" and pa.get("approval_enabled") is True
    ok = set(checks) == set(APPROVAL_CHECKS) and all(checks.values())
    return {"status": "approved" if ok and enabled else "candidate_not_approved", "checks": dict(checks),
            "protocol_status": protocol.get("status"),
            "reason": None if ok and enabled else ("protocol not frozen or approval disabled" if ok else
                                                   "failed: " + ",".join(k for k, v in checks.items() if not v))}


def _alt(rep: Mapping[str, Any], name: str) -> dict[str, Any]:
    b = rep["branches"][name]
    return {"name": name, "actions": b["intervention_actions"], "sources": b["sources"],
            "outcome_replicate0": b["outcome"]}


# ----- weighting ---------------------------------------------------------------------------------
def family_scenario_counts(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Number of scenarios per family that have a selected class-A record (the family weighting divisor)."""
    seen: dict[str, set] = defaultdict(set)
    for r in records:
        if r["class"] == "A" and (r.get("selection") or {}).get("selected"):
            seen[r["family_id"]].add(r["scenario_hash"])
    return {f: len(s) for f, s in sorted(seen.items())}


def assign_weights(records: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any] | None = None,
                   *, family_scenarios: Mapping[str, int] | None = None) -> list[dict[str, Any]]:
    """Training weights. Re-sealed. Every record ends with a weight > 0 or a weight 0 and an exclusion reason.

    Unit `family` (protocol weighting.unit): a scenario's budget is scenario_budget / (number of scenarios of its
    family with selected class-A records). `family_scenarios` gives those counts when the dataset is weighted one
    scenario at a time (streamed build); otherwise they come from `records`. Unit `scenario`: no division.
    Class A: one budget per scenario (x0.25 when uninformative), split EQUALLY over the class-A trajectories
    (tied-best teachers). A short tied-best trajectory is not drowned by a long one, and a rare teacher's
    trajectory gets the same total as a common one.
      select_one: each trajectory's share is split equally over ITS selected records; a merged record gets the
      share of every teacher that proposed it. A teacher whose map lost at a shared state keeps its share on its
      other selected records (its budget never moves to another teacher's map).
      sample_whole_maps: each trajectory's share is split equally over the states it visited; a state's mass is
      split EQUALLY over the distinct class-A maps at that state (independent of how many teachers proposed
      each map).
    Class C: one budget per scenario over APPROVED records only (none under the draft protocol).
    B, D, unapproved C and non-selected A: weight 0 with the class's exclusion reason."""
    protocol = protocol or load_protocol()
    w = protocol["weighting"]
    unit = w.get("unit", "scenario")
    if unit not in ("scenario", "family"):
        raise DatasetError(f"unknown weighting unit {unit!r}")
    by = defaultdict(list)
    for r in records:
        by[(r["scenario_hash"], r["class"])].append(r)
    if unit == "family" and family_scenarios is None:
        if any(r["class"] == "A" and r.get("selection") is None for r in records):
            raise DatasetError("class A records need select_joint_targets before weighting")
        family_scenarios = family_scenario_counts(records)
    out = []
    for (_, k), rs in sorted(by.items()):
        if k == "A" and any(r.get("selection") is None for r in rs):
            raise DatasetError("class A records need select_joint_targets before weighting")
        weights = {id(r): 0.0 for r in rs}
        if k == "A":
            factor = w["uninformative_scenario_factor"] if rs[0]["evidence"].get("uninformative_scenario") else 1.0
            budget = w["scenario_budget"] * factor
            if unit == "family" and any(r["selection"]["selected"] for r in rs):
                n = (family_scenarios or {}).get(rs[0]["family_id"], 0)
                if n < 1:
                    raise DatasetError(f"family {rs[0]['family_id']} has no class-A scenario count")
                budget /= n
            modes = {r["selection"].get("mode") for r in rs}
            if len(modes) != 1 or not modes <= set(JOINT_TARGET_MODES):
                raise DatasetError(f"class A records of one scenario need one joint-target mode, got {sorted(map(str, modes))}")
            chosen = [r for r in rs if r["selection"]["selected"]]
            if modes == {"select_one"}:
                per_teacher = defaultdict(list)
                for r in chosen:
                    for t in r["teachers"]:
                        per_teacher[t].append(r)
                for t, mine in per_teacher.items():
                    for r in mine:
                        weights[id(r)] += budget / len(per_teacher) / len(mine)
            else:
                states_of: dict[str, set] = defaultdict(set)
                for r in chosen:
                    for t in r["teachers"]:
                        states_of[t].add(r["observation_digest"])
                mass: dict[str, float] = defaultdict(float)
                for t, states in states_of.items():
                    for od in states:
                        mass[od] += budget / len(states_of) / len(states)
                maps_at = Counter(r["observation_digest"] for r in chosen)
                for r in chosen:
                    weights[id(r)] = mass[r["observation_digest"]] / maps_at[r["observation_digest"]]
        elif k == "C":
            ok = [r for r in rs if r["approval"]["status"] == "approved"]
            for r in ok:
                weights[id(r)] = w["scenario_budget"] / len(ok)
        for r in rs:
            wt = round(weights[id(r)], 12)
            out.append(_seal({**r, "weight": wt, "exclusion_reason": None if wt > 0 else EXCLUSIONS.get(k)}))
    return out


def _strata(r: Mapping[str, Any]) -> str:
    n = sum(p["status"] == "active" for p in r["observation"]["people"])
    band = "n<=50" if n <= 50 else "n<=110" if n <= 110 else "n>110"
    return f"{band}|{r['observation']['rules']['fire_model']}"


def audit(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Class counts, weight by class, teacher shares of class A weight, family shares, strata and selection."""
    teacher = Counter()
    family = Counter()
    strata = Counter()
    for r in records:
        if r["class"] == "A":
            for t in r["teachers"]:
                teacher[t] += r["weight"] / len(r["teachers"])
            strata[_strata(r)] += r["weight"]
        if r["class"] in ("A", "C"):
            family[r["family_id"]] += r["weight"]
    ta, fa, sa = sum(teacher.values()), sum(family.values()), sum(strata.values())
    a = [r for r in records if r["class"] == "A"]
    contested = Counter((r["scenario_hash"], r["observation_digest"]) for r in a)
    return {"classes": dict(sorted(Counter(r["class"] for r in records).items())),
            "weight_by_class": {k: round(sum(r["weight"] for r in records if r["class"] == k), 9) for k in CLASSES},
            "scenarios": len({r["scenario_hash"] for r in records}),
            "families": len({r["family_id"] for r in records}),
            "teacher_share_A": {t: round(v / ta, 9) for t, v in sorted(teacher.items())} if ta else {},
            "family_share": {f: round(v / fa, 9) for f, v in sorted(family.items())} if fa else {},
            "strata_share_A": {s: round(v / sa, 9) for s, v in sorted(strata.items())} if sa else {},
            "states_with_competing_A_maps": sum(n > 1 for n in contested.values()),
            "selection_decided_by": dict(sorted(Counter(r["selection"]["decided_by"] for r in a
                                                        if r["selection"] and r["selection"]["selected"]).items())),
            # review only: class-A maps a teacher proposed that were not selected (rare-strategy loss check)
            "unselected_A_maps_by_teacher": dict(sorted(Counter(t for r in a if r["selection"]
                                                                and not r["selection"]["selected"]
                                                                for t in r["teachers"]).items())),
            "records_with_teacher_values": sum(1 for r in records if r.get("teacher_values")),
            "class_reasons": dict(sorted(Counter(why for r in records if r["class"] in ("A", "B")
                                                 for why in (r["evidence"].get("class_reasons") or {}).values()
                                                 ).items())),
            "fire_entry_records": sum(1 for r in records if r["class"] in ("A", "B")
                                      and r["evidence"].get("fire_entry_moves")),
            "teacher_dev_similar_weight_A": round(sum(r["weight"] for r in records if r["class"] == "A"
                                                      and r["evidence"].get("teacher_dev_similar_family")), 9),
            "exclusions": dict(sorted(Counter(r["exclusion_reason"] for r in records if r.get("exclusion_reason")).items())),
            "preference_candidates": sum(1 for r in records if r["class"] == "C"),
            "approved_preferences": sum(1 for r in records if r["class"] == "C"
                                        and r["approval"]["status"] == "approved")}


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
        errs.append(f"scope {r['scope']!r} is not allowed by the protocol")
    elif r["scope"] != mt.SCOPE_FIXTURE and protocol.get("status") != "FROZEN":
        errs.append(f"real scope {r['scope']!r} needs a FROZEN protocol")
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
    elif r["weight"] == 0 and r["exclusion_reason"] != EXCLUSIONS[k]:
        errs.append(f"class {k}: zero weight needs the exclusion reason {EXCLUSIONS[k]!r}")
    elif r["weight"] > 0 and r["exclusion_reason"] is not None:
        errs.append("a weighted record cannot carry an exclusion reason")
    elif r["weight"] > 0 and k == "B":
        errs.append("class B is contrast only and must have zero training weight")
    for f in ("family_id", "lineage_id"):
        if not isinstance(r[f], str) or not r[f]:
            errs.append(f"{f} must be a non-empty string")
    sel = r["selection"]
    if not isinstance(sel, Mapping) or sel.get("rule") != JOINT_TARGET_RULE or not sel.get("reason"):
        errs.append("selection rationale is missing or not under the preregistered rule")
    elif k == "A" and sel.get("mode") != (protocol.get("joint_target_rule") or {}).get("mode"):
        errs.append(f"class A: selection mode {sel.get('mode')!r} is not the protocol's preregistered mode")
    elif k == "A" and (not isinstance(sel.get("selected"), bool) or
                       (isinstance(r["weight"], (int, float)) and r["weight"] > 0 and not sel["selected"])):
        errs.append("class A: only a selected whole joint map can carry weight")
    active = {p["id"] for p in r["observation"]["people"] if p["status"] == "active"}
    if k in ("A", "B"):
        tp = r["provenance"].get("teachers", {})
        for t in r["teachers"]:
            p = tp.get(t)
            if not isinstance(p, Mapping) or not str(p.get("code_fingerprint", "")).startswith("sha256:") \
                    or (p.get("identity") or {}).get("name") != t:
                errs.append(f"no exact identity and source fingerprint for {t}")
            rp = r["replay"].get(t) or {}
            if rp.get("status") != "complete" or not rp.get("final_transition_hash"):
                errs.append(f"no validated complete replay for {t}")
        if set(r["actions"]) != active:
            errs.append("demonstration joint action is not complete")
        if not r["teachers"] or len(r["teachers"]) != len(r["proposal_hashes"]):
            errs.append("each proposing teacher needs its proposal hash")
        tv = r["teacher_values"]
        if tv is not None and (set(tv) - VALUE_TEACHERS or not set(tv) <= set(r["teachers"])):
            errs.append("teacher values from a teacher that does not compute them")
        ev = r["evidence"]
        outs = ev.get("trajectory_outcomes", {})
        diag = diagnostic_teachers(protocol)
        fire = ev.get("fire_entry_moves")
        if not isinstance(fire, list):
            errs.append("fire_entry_moves evidence is missing")
            fire = []
        for t in r["teachers"]:
            o = outs.get(t)
            if o is None:
                errs.append(f"no outcome evidence for {t}")
                continue
            why = class_reason(o, t, fire, ev["group_best_primary"], ev["group_best_casualties"], diag)
            if (k == "A") is not (why == "best_outcome") or (ev.get("class_reasons") or {}).get(t) != why:
                errs.append(f"class {k} does not follow the outcome rule for {t} ({why})")
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
        ap = r["approval"]
        if not isinstance(ap, Mapping) or set(ap.get("checks") or {}) != set(APPROVAL_CHECKS):
            errs.append("class C needs every preregistered approval check")
        elif ap["status"] == "approved":
            if protocol.get("status") != "FROZEN" or (protocol.get("preference_approval") or {}).get(
                    "approval_enabled") is not True:
                errs.append("an approved preference needs a FROZEN protocol with approval enabled")
            if not all(ap["checks"].values()) or ev["censored"]:
                errs.append("an approved preference must pass every approval check")
        elif ap["status"] != "candidate_not_approved":
            errs.append(f"unknown approval status {ap['status']!r}")
        if ap.get("status") != "approved" and r["weight"] != 0:
            errs.append("an unapproved preference candidate must have zero training weight")
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
    problems += coherence_problems(records) + source_drift_problems(records)
    if problems:
        return {"ok": False, "problems": problems}
    a = audit(records)
    w = protocol["weighting"]
    review: list[str] = []
    if a["scenarios"] >= w["min_scenarios_for_share_caps"]:
        # Report only: capping would move weight from a strictly better teacher to a worse one.
        review += [f"teacher {t} share {s} > {w['teacher_share_A_review']} (review; never auto-reweighted)"
                   for t, s in a["teacher_share_A"].items() if s > w["teacher_share_A_review"]]
        cap = max(w["max_family_share"]["absolute"], w["max_family_share"]["relative_to_uniform"] / a["families"])
        problems += [f"family {f} share {s} > {round(cap, 9)}" for f, s in a["family_share"].items() if s > cap]
    return {"ok": not problems, "problems": problems, "review": review, "audit": a}


def coherence_problems(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """Contradictory demonstrations. select_one: at most one weighted SFT target per state, and it is one
    complete map. sample_whole_maps: every distinct map at a state is selected, with EQUAL weight."""
    out = []
    groups = defaultdict(list)
    for r in records:
        if r["class"] == "A":
            groups[(r["scenario_hash"], r["observation_digest"])].append(r)
    modes = {r["selection"].get("mode") for rs in groups.values() for r in rs}
    if len(modes) > 1:
        out.append(f"class A records mix joint-target modes {sorted(map(str, modes))}")
    for k, rs in groups.items():
        n_sel = sum(bool(r["selection"]["selected"]) for r in rs)
        if rs[0]["selection"].get("mode") == "sample_whole_maps":
            ws = {round(r["weight"], 9) for r in rs}
            if n_sel != len(rs):
                out.append(f"state {k[1][:19]}: sample_whole_maps selects every distinct class-A map, "
                           f"got {n_sel}/{len(rs)}")
            if len(ws) != 1:
                out.append(f"state {k[1][:19]}: sample_whole_maps needs equal weight per distinct map, got {sorted(ws)}")
            continue
        weighted = sum(r["weight"] > 0 for r in rs)
        if weighted > 1:
            out.append(f"contradictory class-A targets at one state {k[1][:19]}: {weighted} weighted joint maps")
        if n_sel != 1:
            out.append(f"state {k[1][:19]}: exactly one class-A map must be selected, got {n_sel}")
    return out


def model_example(r: Mapping[str, Any]) -> dict[str, Any]:
    """The ONLY view a trainer may read from a positive record: input observation, whole target map, weight.
    Teacher names, proposal hashes, provenance, outcomes and selection data are never a model input, so the
    student cannot use teacher identity as an inference-time oracle."""
    if r.get("class") != "A" or not (r.get("selection") or {}).get("selected") or not r.get("weight"):
        raise DatasetError("only a selected, weighted class-A record is an SFT example")
    return {"observation": copy.deepcopy(r["observation"]), "actions": dict(r["actions"]), "weight": r["weight"]}


def preference_example(r: Mapping[str, Any]) -> dict[str, Any]:
    """The ONLY view a preference trainer may read from a class-C record: the common observation and the two
    whole joint maps. Proposer sets (`sources`), outcomes, verdict evidence and provenance never pass, so the
    student cannot learn "prefer T2's map" from identity. Only an APPROVED, weighted record qualifies
    (none under the draft protocol)."""
    if r.get("class") != "C" or (r.get("approval") or {}).get("status") != "approved" or not r.get("weight"):
        raise DatasetError("only an approved, weighted class-C record is a preference example")
    return {"observation": copy.deepcopy(r["observation"]), "chosen_actions": dict(r["chosen"]["actions"]),
            "rejected_actions": dict(r["rejected"]["actions"]), "weight": r["weight"]}


_STEP_BANDS = ((0, 4), (5, 14), (15, 39), (40, 10 ** 9))


def _step_band(step: int) -> str:
    lo, hi = next(b for b in _STEP_BANDS if b[0] <= step <= b[1])
    return f"step {lo}-{hi}" if hi < 10 ** 9 else f"step >={lo}"


def selection_bias_audit(records: Sequence[Mapping[str, Any]], *, flag_ratio: float = 2.0) -> dict[str, Any]:
    """Review-only audit of OUTCOME-BASED selection (class A = tied-best trajectories). Not a gate.

    The pool is every demonstrated state-map record (A and B, unweighted, one count per record); the selected set
    is the class-A training weight. For each dimension (people band x fire model, step band, teacher, scenario)
    it reports pool share, selected share and their ratio. Ratios above `flag_ratio` or below 1/`flag_ratio`
    are flagged: winners end earlier (fewer late states), drop hard scenarios, or favour some teachers. Also
    reports scenarios with no class-A weight and minority evidence kept (B records, unselected A maps)."""
    pool = [r for r in records if r["class"] in ("A", "B")]
    sel = [r for r in records if r["class"] == "A" and (r.get("weight") or 0) > 0]
    total_w = sum(r["weight"] for r in sel)

    def dims(r: Mapping[str, Any]) -> dict[str, list[str]]:
        return {"strata": [_strata(r)], "step_band": [_step_band(r["step"])], "teacher": list(r["teachers"]),
                "scenario": [r["scenario_hash"]]}

    out: dict[str, Any] = {}
    for dim in ("strata", "step_band", "teacher", "scenario"):
        pc, sc = Counter(), Counter()
        for r in pool:
            for v in dims(r)[dim]:
                pc[v] += 1 / len(dims(r)[dim])
        for r in sel:
            for v in dims(r)[dim]:
                sc[v] += r["weight"] / len(dims(r)[dim])
        pn = sum(pc.values())
        rows = {}
        for v in sorted(set(pc) | set(sc)):
            ps, ss = (pc[v] / pn if pn else 0.0), (sc[v] / total_w if total_w else 0.0)
            ratio = None if ps == 0 else round(ss / ps, 6)
            rows[v] = {"pool_share": round(ps, 6), "selected_share": round(ss, 6), "ratio": ratio,
                       "flag": ratio is None or ratio > flag_ratio or ratio < 1 / flag_ratio}
        out[dim] = rows
    scen = {r["scenario_hash"] for r in pool}
    with_a = {r["scenario_hash"] for r in sel}
    return {"kind": "outcome_selection_bias_audit", "gate": False, "flag_ratio": flag_ratio,
            "pool_records": len(pool), "selected_records": len(sel), "by": out,
            "scenarios_without_class_A_weight": sorted(scen - with_a),
            "uninformative_scenarios": sorted({r["scenario_hash"] for r in pool
                                               if r["evidence"].get("uninformative_scenario")}),
            "minority_evidence_kept": {"class_B_records": sum(r["class"] == "B" for r in records),
                                       "unselected_class_A_maps": sum(1 for r in records if r["class"] == "A" and
                                                                      not (r.get("selection") or {}).get("selected")),
                                       "class_D_records": sum(r["class"] == "D" for r in records)},
            "flagged": sorted(f"{d}:{v}" for d, rows in out.items() if d != "scenario"
                              for v, x in rows.items() if x["flag"]),
            "note": "Selection on outcome changes the state distribution; report it with any trained student."}


def check_targets_against_report(records: Sequence[Mapping[str, Any]], report: Mapping[str, Any]) -> list[str]:
    """Every A/B map must equal, as a whole, the proposal of each teacher it names at that exact state. A map
    assembled person by person from several teachers fails here."""
    props = {(n, d["observation_digest"]): d["proposal"] for n, roll in report["policies"].items()
             for d in roll["decisions"]}
    out = []
    for r in records:
        if r["class"] not in ("A", "B") or r["provenance"].get("report_content_hash") != report["content_hash"]:
            continue
        for t, h in zip(r["teachers"], r["proposal_hashes"]):
            p = props.get((t, r["observation_digest"]))
            if p is None or p["proposal_hash"] != h or dict(p["actions"]) != dict(r["actions"]):
                out.append(f"{r['class']} record at {r['observation_digest'][:19]} is not {t}'s whole proposal")
    return out


def source_drift_problems(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """One identity and one code fingerprint per teacher name across the whole dataset."""
    seen: dict[str, set] = defaultdict(set)
    for r in records:
        if r["class"] in ("A", "B"):
            for t, p in r["provenance"]["teachers"].items():
                seen[t].add((digest(p.get("identity")), p.get("code_fingerprint")))
    return [f"teacher {t} source drift: {len(v)} identities or fingerprints" for t, v in sorted(seen.items()) if len(v) > 1]


def collection_gate_problems(protocol: Mapping[str, Any], verification: Mapping[str, Any] | None) -> list[str]:
    """The protocol's collection gate against a collection verification report (train_collect verify)."""
    g = protocol.get("collection_gate")
    if not isinstance(g, Mapping):
        return ["protocol has no collection gate"]
    if not isinstance(verification, Mapping):
        return ["TRAIN collection gate: no verification report"]
    out = []
    body = {k: v for k, v in verification.items() if k != "verification_hash"}
    if digest(body) != verification.get("verification_hash"):
        out.append("verification report hash does not recompute")
    for key in ("plan_hash", "verification_version", "verification_hash"):
        if verification.get(key) != g.get(key):
            out.append(f"verification {key} {verification.get(key)!r} is not the gate's {g.get(key)!r}")
    if verification.get("full_train_verified") is not True:
        out.append("collection is not full_train_verified")
    if verification.get("status_counts") != {"verified_complete": g.get("groups")}:
        out.append(f"status counts {verification.get('status_counts')} are not {g.get('groups')} verified_complete")
    if g.get("scope") not in protocol["scope"]["allowed_now"]:
        out.append(f"gate scope {g.get('scope')!r} is not an allowed scope")
    return out


def export(*_: Any, protocol: Mapping[str, Any] | None = None, authorization: str | None = None,
           verification: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Gate for real dataset generation: FROZEN protocol, owner authorization and the collection gate.
    Raises DATASET_GENERATION_BLOCKED with every reason; returns the gate record when all pass. The builder
    (`train_dataset`) still runs validate_dataset over every record before it writes the dataset."""
    protocol = protocol or load_protocol()
    reasons = []
    if protocol.get("status") != "FROZEN":
        reasons.append("protocol is not FROZEN")
    if not authorization:
        reasons.append("no owner authorization")
    reasons += collection_gate_problems(protocol, verification)
    if reasons:
        raise DatasetError("DATASET_GENERATION_BLOCKED: " + "; ".join(reasons))
    return {"status": "GATE_PASS", "protocol_version": protocol["version"], "authorization": authorization,
            "plan_hash": verification["plan_hash"], "verification_hash": verification["verification_hash"]}
