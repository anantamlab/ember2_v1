"""P1-14 dataset records: classes A-E, outcome rule, preference thresholds, weights, negative tests.

Fixture scope only. Nothing here is real TRAIN data or a training label.
"""
from __future__ import annotations

import copy

import pytest

from ember_v1 import counterfactual as cf
from ember_v1 import multiteacher as mt
from ember_v1 import multiteacher_dataset as ds
from test_multiteacher import CrashPolicy, busy_room, factories, real_factories, room


@pytest.fixture(scope="module")
def built():
    s = busy_room()
    r = mt.collect_fixture(s, real_factories())
    recs = ds.demonstrations(r, s)
    for step in (0, 3):
        study = cf.shared_state_study(s, r, "T1", step, real_factories(), horizon=10, max_horizon=10, replicates=5)
        recs += ds.preferences(study, s)
    return s, r, ds.assign_weights(ds.select_joint_targets(recs))


def _reseal(rec):
    return ds._seal({k: v for k, v in rec.items() if k != "record_hash"})


def _problems(rec):
    return ds.record_problems(rec, ds.load_protocol())


def test_protocol_is_a_draft_and_export_is_always_blocked():
    p = ds.load_protocol()
    assert p["status"] == "DRAFT_NOT_FROZEN" and p["authorizes"].startswith("nothing")
    with pytest.raises(ds.DatasetError, match="DATASET_GENERATION_BLOCKED.*not FROZEN.*no owner authorization"):
        ds.export()
    frozen = {**p, "status": "FROZEN"}
    with pytest.raises(ds.DatasetError, match="TRAIN collection gate"):
        ds.export(protocol=frozen, authorization="owner-says-yes")


def test_fixture_dataset_has_all_four_classes_and_validates(built):
    s, r, recs = built
    v = ds.validate_dataset(recs)
    assert v["ok"], v["problems"]
    a = v["audit"]
    assert set(a["classes"]) == {"A", "B", "C", "D"}
    assert a["weight_by_class"]["A"] == pytest.approx(1.0) and a["weight_by_class"]["D"] == 0
    # class follows matched outcomes, not teacher names
    outs = {n: p["outcome"] for n, p in r["policies"].items()}
    best = min(o["casualties_plus_unresolved"] for o in outs.values())
    a_teachers = {t for x in recs if x["class"] == "A" for t in x["teachers"]}
    assert a_teachers == {n for n, o in outs.items() if o["casualties_plus_unresolved"] == best}
    assert {t for x in recs if x["class"] == "B" for t in x["teachers"]} == set(outs) - a_teachers
    assert set(a["teacher_share_A"]) == a_teachers


def test_teacher_values_only_where_t2_computed_them(built):
    _, r, recs = built
    with_vals = [x for x in recs if x["class"] in ("A", "B") and x["teacher_values"]]
    assert with_vals and all(set(x["teacher_values"]) == {"T2"} and "T2" in x["teachers"] for x in with_vals)
    assert all(x["teacher_values"] is None for x in recs if x["class"] in ("A", "B") and "T2" not in x["teachers"])
    rec = next(x for x in recs if x["class"] in ("A", "B") and "T2" not in x["teachers"])
    forged = _reseal({**rec, "teacher_values": {rec["teachers"][0]: {"north": 1.0}}})
    assert any("does not compute" in e for e in _problems(forged))


@pytest.mark.parametrize("field", ["probability", "confidence", "causal_label", "logprob"])
def test_fabricated_fields_are_refused(built, field):
    rec = next(x for x in built[2] if x["class"] == "A")
    bad = _reseal({**rec, "evidence": {**rec["evidence"], field: 0.9}})
    assert any("forbidden" in e for e in _problems(bad))


def test_relabelling_a_poor_demonstration_as_positive_is_refused(built):
    rec = next(x for x in built[2] if x["class"] == "B")
    bad = _reseal({**rec, "class": "A", "use": "sft_positive"})
    assert any("outcome rule" in e for e in _problems(bad))


def test_preference_thresholds_and_censoring(built):
    rec = next(x for x in built[2] if x["class"] == "C")
    assert not _problems(rec)
    for change, msg in (({"censored": True}, "uncensored"), ({"strict_improvements": 1}, "threshold"),
                        ({"replicates": 2}, "threshold"), ({"verdict": "inconclusive"}, "uncensored")):
        bad = _reseal({**rec, "evidence": {**rec["evidence"], **change}})
        assert any(msg in e for e in _problems(bad)), change
    same = _reseal({**rec, "rejected": rec["chosen"]})
    assert any("identical" in e for e in _problems(same))
    partial = copy.deepcopy(rec)
    partial["chosen"]["actions"].popitem()
    assert any("not complete" in e for e in _problems(_reseal(partial)))


def test_inconclusive_has_no_weight_and_cannot_hide_a_passing_verdict(built):
    rec = next(x for x in built[2] if x["class"] == "D")
    assert rec["weight"] == 0 and rec["use"] == "excluded_inconclusive"
    assert any("zero training weight" in e for e in _problems(_reseal({**rec, "weight": 0.5})))
    hidden = _reseal({**rec, "evidence": {**rec["evidence"], "verdict": "b_better", "below_threshold": False}})
    assert any("must be class C" in e for e in _problems(hidden))


def test_student_visited_class_is_reserved(built):
    rec = {**next(x for x in built[2] if x["class"] == "A"), "class": "E"}
    assert _problems(rec) == ["class E (student-visited) is reserved until P1-16 (#49) is authorized"]


def test_integrity_scope_weight_and_duplicates(built):
    recs = built[2]
    rec = next(x for x in recs if x["class"] == "A")
    assert any("hash mismatch" in e for e in _problems({**rec, "weight": 0.123}))
    assert any("not allowed" in e for e in _problems(_reseal({**rec, "scope": "train"})))
    assert any("weight must be assigned" in e for e in _problems(_reseal({**rec, "weight": None})))
    obs = copy.deepcopy(rec["observation"])
    obs["rules"]["seed"] = 1
    assert any("digest" in e or "private" in e for e in _problems(_reseal({**rec, "observation": obs})))
    v = ds.validate_dataset([*recs, rec])
    assert not v["ok"] and any("duplicate" in e for e in v["problems"])
    partial = copy.deepcopy(rec)
    partial["actions"].popitem()
    assert any("not complete" in e for e in _problems(_reseal(partial)))


def test_family_cap_applies_and_teacher_share_is_review_only(built):
    p = ds.load_protocol()
    tight = {**p, "weighting": {**p["weighting"], "min_scenarios_for_share_caps": 1, "teacher_share_A_review": 0.4,
                                "max_family_share": {"absolute": 0.05, "relative_to_uniform": 0.5}}}
    v = ds.validate_dataset(built[2], tight)
    assert not v["ok"] and any("family" in e and "share" in e for e in v["problems"])
    assert not any("teacher" in e for e in v["problems"]) and any("review" in e for e in v["review"])


def test_incomplete_group_gives_no_demonstrations():
    fs = factories()
    fs["T3"] = lambda: CrashPolicy("T3")
    r = mt.collect_fixture(room(), fs)
    with pytest.raises(ds.DatasetError, match="incomplete five-teacher group"):
        ds.demonstrations(r, room())


def test_uninformative_scenario_is_down_weighted():
    r = mt.collect_fixture(room(), factories())           # five identical policies: no discrimination
    recs = ds.assign_weights(ds.select_joint_targets(ds.demonstrations(r, room())))
    assert {x["class"] for x in recs} == {"A"}
    assert all(x["evidence"]["uninformative_scenario"] for x in recs)
    assert sum(x["weight"] for x in recs) == pytest.approx(0.25)
    assert ds.validate_dataset(recs)["ok"]


def test_no_disagreement_study_gives_no_preference(built):
    s, r, _ = built
    study = cf.shared_state_study(s, r, "T1", 1, real_factories())
    assert study["status"] == "no_disagreement" and ds.preferences(study, s) == []


# ----- run 8: coherent joint targets, approval, provenance and adversarial records --------------------------
def _competing(recs):
    from collections import defaultdict
    g = defaultdict(list)
    for x in recs:
        if x["class"] == "A":
            g[x["observation_digest"]].append(x)
    return next(v for v in g.values() if len(v) > 1)


def test_one_whole_joint_target_per_state(built):
    s, r, recs = built
    a = ds.audit(recs)
    assert a["states_with_competing_A_maps"] >= 1
    comp = _competing(recs)
    sel = [x for x in comp if x["selection"]["selected"]]
    assert len(sel) == 1 and sel[0]["weight"] > 0 and sel[0]["exclusion_reason"] is None
    for x in comp:
        if x is not sel[0]:
            assert x["weight"] == 0 and x["exclusion_reason"] == ds.EXCLUSIONS["A"]
            assert x["selection"]["decided_by"] == sel[0]["selection"]["decided_by"] != "approved_same_state_preference"
    assert ds.check_targets_against_report(recs, r) == []
    assert sum(x["weight"] for x in recs if x["class"] == "A") == pytest.approx(1.0)


def test_contradictory_weighted_demonstrations_are_refused(built):
    recs = list(built[2])
    comp = _competing(recs)
    loser = next(x for x in comp if not x["selection"]["selected"])
    forged = _reseal({**loser, "weight": 0.1, "exclusion_reason": None,
                      "selection": {**loser["selection"], "selected": True}})
    v = ds.validate_dataset([forged if x is loser else x for x in recs])
    assert not v["ok"] and any("contradictory class-A targets" in e for e in v["problems"])


def test_per_person_mixture_of_two_good_teachers_is_detected(built):
    s, r, recs = built
    od = _competing(recs)[0]["observation_digest"]
    maps = [x for x in recs if x["class"] in ("A", "B") and x["observation_digest"] == od]
    a, b = next((x, y) for x in maps for y in maps
                if sum(x["actions"][p] != y["actions"][p] for p in x["actions"]) >= 2)
    diff = [p for p in a["actions"] if a["actions"][p] != b["actions"][p]]
    mixed = {**a["actions"], diff[0]: b["actions"][diff[0]]}      # one person from the other teacher's map
    assert mixed not in [x["actions"] for x in maps]
    bad = _reseal({**a, "actions": mixed})
    assert ds.check_targets_against_report([bad], r)
    assert ds.check_targets_against_report([a, b], r) == []


def test_selection_is_required_before_weighting(built):
    s, r, _ = built
    with pytest.raises(ds.DatasetError, match="select_joint_targets"):
        ds.assign_weights(ds.demonstrations(r, s))


def test_preference_candidates_are_not_approved_under_the_draft(built):
    recs = built[2]
    cs = [x for x in recs if x["class"] == "C"]
    assert cs and all(x["approval"]["status"] == "candidate_not_approved" and x["weight"] == 0
                      and x["exclusion_reason"] == ds.EXCLUSIONS["C"] for x in cs)
    assert set(cs[0]["approval"]["checks"]) == set(ds.APPROVAL_CHECKS)
    assert ds.audit(recs)["approved_preferences"] == 0


def test_approval_needs_every_check_and_a_frozen_enabled_protocol():
    p = ds.load_protocol()
    ok = dict.fromkeys(ds.APPROVAL_CHECKS, True)
    assert ds.approval(ok, p)["status"] == "candidate_not_approved"
    frozen = {**p, "status": "FROZEN", "preference_approval": {**p["preference_approval"], "approval_enabled": True}}
    assert ds.approval(ok, frozen)["status"] == "approved"
    for k in ds.APPROVAL_CHECKS:
        got = ds.approval({**ok, k: False}, frozen)
        assert got["status"] == "candidate_not_approved" and k in got["reason"]
    assert ds.approval({k: True for k in ds.APPROVAL_CHECKS[:-1]}, frozen)["status"] == "candidate_not_approved"


def test_malformed_preference_records_are_refused(built):
    rec = next(x for x in built[2] if x["class"] == "C")
    fake = _reseal({**rec, "approval": {**rec["approval"], "status": "approved"}})
    assert any("FROZEN" in e for e in _problems(fake))
    no_check = copy.deepcopy(rec)
    no_check["approval"]["checks"].pop("no_unresolved_increase")
    assert any("approval check" in e for e in _problems(_reseal(no_check)))
    weighted = _reseal({**rec, "weight": 0.3, "exclusion_reason": None})
    assert any("zero training weight" in e for e in _problems(weighted))
    no_reason = _reseal({**rec, "exclusion_reason": None})
    assert any("exclusion reason" in e for e in _problems(no_reason))


def test_records_carry_provenance_replay_and_congestion(built):
    recs = built[2]
    for x in recs:
        assert x["lineage_id"] and x["family_id"] and x["selection"]["reason"]
        if x["class"] in ("A", "B"):
            assert set(x["provenance"]["teachers"]) == set(x["teachers"]) == set(x["replay"])
            assert all(v["status"] == "complete" for v in x["replay"].values())
            assert "blocked_moves" in x["step_diagnostics"]
            assert all("contested_targets" in o for o in x["evidence"]["trajectory_outcomes"].values())
    b = next(x for x in recs if x["class"] == "B")
    assert b["weight"] == 0 and b["exclusion_reason"] == ds.EXCLUSIONS["B"]
    assert any("contrast only" in e for e in _problems(_reseal({**b, "weight": 0.2, "exclusion_reason": None})))
    unsourced = copy.deepcopy(next(x for x in recs if x["class"] == "A"))
    unsourced["provenance"]["teachers"][unsourced["teachers"][0]]["code_fingerprint"] = None
    assert any("source fingerprint" in e for e in _problems(_reseal(unsourced)))
    unreplayed = copy.deepcopy(next(x for x in recs if x["class"] == "A"))
    unreplayed["replay"][unreplayed["teachers"][0]]["status"] = "partial"
    assert any("replay" in e for e in _problems(_reseal(unreplayed)))


def test_source_drift_is_refused(built):
    recs = list(built[2])
    i = next(i for i, x in enumerate(recs) if x["class"] == "B")
    drift = copy.deepcopy(recs[i])
    t = drift["teachers"][0]
    drift["provenance"]["teachers"][t]["code_fingerprint"] = "sha256:" + "0" * 64
    recs[i] = _reseal(drift)
    v = ds.validate_dataset(recs)
    assert not v["ok"] and any(f"teacher {t} source drift" in e for e in v["problems"])


def test_duplicated_family_copy_is_refused(built):
    recs = list(built[2])
    rec = next(x for x in recs if x["class"] == "A" and x["weight"] > 0)
    copy_other_family = _reseal({**rec, "family_id": "another-family", "lineage_id": "another-family"})
    v = ds.validate_dataset([*recs, copy_other_family])
    assert not v["ok"] and any("duplicate record" in e for e in v["problems"])


# ----- run 9: conflicting valid whole maps, count neutrality, preregistered sampling, no identity oracle ------
def _sample_protocol():
    p = ds.load_protocol()
    return {**p, "joint_target_rule": {**p["joint_target_rule"], "mode": "sample_whole_maps"}}


def _unselected(rec):
    return _reseal({**rec, "selection": None, "weight": None, "exclusion_reason": None})


def _conflict(built, common=("T0", "T1", "T3", "T4"), rare=("T2",), common_blocked=(0, 9, 9, 9), rare_blocked=(1,)):
    """Two valid class-A whole maps at one state: a common map (four teachers) and a rare one (one teacher).
    Equal at criteria 1-2; the common map has one lucky low-congestion trajectory and three bad ones."""
    a, b = (copy.deepcopy(x) for x in _competing(built[2])[:2])
    for rec, names, blocked in ((a, common, common_blocked), (b, rare, rare_blocked)):
        o = next(iter(rec["evidence"]["trajectory_outcomes"].values()))
        rec["teachers"] = list(names)
        rec["proposal_hashes"] = [f"sha256:{n}" for n in names]
        rec["evidence"]["trajectory_outcomes"] = {n: {**o, "blocked_moves": k, "steps": 10}
                                                  for n, k in zip(names, blocked)}
        rec["step_diagnostics"] = {**rec["step_diagnostics"], "blocked_moves": 0}
    assert a["actions"] != b["actions"] and a["observation_digest"] == b["observation_digest"]
    return _unselected(a), _unselected(b)


def test_rare_valid_map_is_not_discarded_because_more_teachers_proposed_the_other(built):
    a, b = _conflict(built)
    sel = {x["teachers"][0]: x for x in ds.select_joint_targets([a, b])}
    assert sel["T2"]["selection"]["selected"] and not sel["T0"]["selection"]["selected"]
    assert sel["T2"]["selection"]["decided_by"] == "mean_trajectory_blocked_moves"
    # the replaced minimum rule would have chosen the common map by its single lucky draw
    assert min(a["evidence"]["trajectory_outcomes"][t]["blocked_moves"] for t in a["teachers"]) < \
        b["evidence"]["trajectory_outcomes"]["T2"]["blocked_moves"]
    # both maps keep complete provenance and stay whole; the loser is kept with weight 0
    w = {x["teachers"][0]: x for x in ds.assign_weights(ds.select_joint_targets([a, b]))}
    assert w["T0"]["weight"] == 0 and w["T0"]["exclusion_reason"] == ds.EXCLUSIONS["A"]
    assert w["T0"]["teachers"] == ["T0", "T1", "T3", "T4"] and w["T0"]["actions"] == a["actions"]
    assert w["T2"]["weight"] > 0 and w["T2"]["actions"] == b["actions"]


def test_selection_and_weights_do_not_depend_on_teacher_names(built):
    def by_map(recs):
        return {ds.digest(x["actions"]): (x["selection"]["selected"], x["weight"]) for x in recs}
    a, b = _conflict(built)
    base = by_map(ds.assign_weights(ds.select_joint_targets([a, b])))
    swap = {"T2": "T1", "T1": "T2"}
    renamed = []
    for x in (a, b):
        y = copy.deepcopy(x)
        y["teachers"] = [swap.get(t, t) for t in y["teachers"]]
        y["evidence"]["trajectory_outcomes"] = {swap.get(t, t): o for t, o in y["evidence"]["trajectory_outcomes"].items()}
        renamed.append(_reseal(y))
    assert by_map(ds.assign_weights(ds.select_joint_targets(renamed))) == base
    # count alone never decides: equal mean, four proposers vs one -> decided by the name-free digest
    a2, b2 = _conflict(built, common_blocked=(1, 1, 1, 1), rare_blocked=(1,))
    out = ds.select_joint_targets([a2, b2])
    assert {x["selection"]["decided_by"] for x in out} == {"action_digest_tie_break"}
    winner = min((a2, b2), key=lambda x: ds.digest(x["actions"]))
    assert next(x for x in out if x["selection"]["selected"])["actions"] == winner["actions"]


def test_preregistered_sampling_keeps_every_valid_whole_map_with_equal_weight(built):
    s, r, recs = built
    p = _sample_protocol()
    raw = [_unselected(x) if x["class"] == "A" else x for x in recs]
    out = ds.assign_weights(ds.select_joint_targets(raw, p), p)
    v = ds.validate_dataset(out, p)
    assert v["ok"], v["problems"]
    comp = [x for x in out if x["class"] == "A" and x["observation_digest"] == _competing(recs)[0]["observation_digest"]]
    assert len(comp) >= 2 and all(x["selection"]["selected"] and x["selection"]["mode"] == "sample_whole_maps"
                                  for x in comp)
    assert len({x["weight"] for x in comp}) == 1 and comp[0]["weight"] > 0
    assert sum(x["weight"] for x in out if x["class"] == "A") == pytest.approx(1.0)
    assert ds.check_targets_against_report(out, r) == []                 # every weighted map is a whole proposal
    # the conflict pair: equal weight regardless of four proposers vs one
    a, b = _conflict(built)
    w = ds.assign_weights(ds.select_joint_targets([a, b], p), p)
    assert w[0]["weight"] == w[1]["weight"] > 0
    # a select_one dataset is refused under the sampling protocol and vice versa (no silent mode switch)
    assert any("preregistered mode" in e for e in ds.record_problems(next(x for x in recs if x["class"] == "A"), p))
    assert any("preregistered mode" in e for e in _problems(next(x for x in out if x["class"] == "A")))
    unequal = [(_reseal({**x, "weight": x["weight"] * 1.5}) if x is comp[0] else x) for x in out]
    v = ds.validate_dataset(unequal, p)
    assert not v["ok"] and any("equal weight per distinct map" in e for e in v["problems"])
    with pytest.raises(ds.DatasetError, match="mode"):
        ds.select_joint_targets(raw, {**p, "joint_target_rule": {**p["joint_target_rule"], "mode": "mix_per_person"}})


def test_stage_a_mode_is_select_one_and_sampling_is_preregistered_but_inactive():
    rule = ds.load_protocol()["joint_target_rule"]
    assert rule["id"] == ds.JOINT_TARGET_RULE and rule["mode"] == "select_one"
    assert set(rule["modes"]) == set(ds.JOINT_TARGET_MODES)
    assert "NOT active" in rule["modes"]["sample_whole_maps"]["status"]
    assert any("MIXTURE" in x for x in rule["modes"]["sample_whole_maps"]["admissible_only_if"])


def test_teacher_identity_is_never_a_model_input(built):
    import json
    recs = built[2]
    sel = [x for x in recs if x["class"] == "A" and x["weight"] > 0]
    ex = [ds.model_example(x) for x in sel]
    assert all(set(e) == {"observation", "actions", "weight"} for e in ex)
    text = json.dumps(ex)
    for name in mt.CANDIDATES:
        assert f'"{name}"' not in text
    for field in ("teachers", "proposal_hashes", "provenance", "teacher_values", "trajectory_outcomes", "selection"):
        assert f'"{field}"' not in text
    for bad in (next(x for x in recs if x["class"] == "B"),
                next(x for x in recs if x["class"] == "A" and x["weight"] == 0),
                next(x for x in recs if x["class"] == "C")):
        with pytest.raises(ds.DatasetError, match="SFT example"):
            ds.model_example(bad)
