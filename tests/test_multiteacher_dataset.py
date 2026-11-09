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
    return s, r, ds.assign_weights(recs)


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


def test_share_caps_apply_once_enough_scenarios(built):
    p = ds.load_protocol()
    tight = {**p, "weighting": {**p["weighting"], "min_scenarios_for_share_caps": 1, "max_teacher_share_A": 0.4,
                                "max_family_share": 1.0}}
    v = ds.validate_dataset(built[2], tight)
    assert not v["ok"] and any("share" in e for e in v["problems"])


def test_incomplete_group_gives_no_demonstrations():
    fs = factories()
    fs["T3"] = lambda: CrashPolicy("T3")
    r = mt.collect_fixture(room(), fs)
    with pytest.raises(ds.DatasetError, match="incomplete five-teacher group"):
        ds.demonstrations(r, room())


def test_uninformative_scenario_is_down_weighted():
    r = mt.collect_fixture(room(), factories())           # five identical policies: no discrimination
    recs = ds.assign_weights(ds.demonstrations(r, room()))
    assert {x["class"] for x in recs} == {"A"}
    assert all(x["evidence"]["uninformative_scenario"] for x in recs)
    assert sum(x["weight"] for x in recs) == pytest.approx(0.25)
    assert ds.validate_dataset(recs)["ok"]


def test_no_disagreement_study_gives_no_preference(built):
    s, r, _ = built
    study = cf.shared_state_study(s, r, "T1", 1, real_factories())
    assert study["status"] == "no_disagreement" and ds.preferences(study, s) == []
