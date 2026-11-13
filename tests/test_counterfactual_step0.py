"""Real-scope step-0 counterfactual studies (P1-17 Phase 7) on tiny fixtures assembled in the real scope."""
from __future__ import annotations

import copy

import pytest

from ember_v1 import counterfactual as cf
from ember_v1 import multiteacher as mt
from ember_v1 import multiteacher_dataset as ds
from ember_v1.teacher import registry as reg
from ember_v1.teacher.registry import CANDIDATES
from test_multiteacher import busy_room, real_factories

SCOPE = "p1-17-train-full@1"


def _group(s, scope=SCOPE):
    pols = {n: mt._new_policy(f, n) for n, f in real_factories().items()}
    res = {n: mt._rollout(s, pols[n], cpu_ms_cap=30000) for n in CANDIDATES}
    return mt.assemble_group(s, res, mt.provenance(s, pols, None), scope=scope)


def _t(name):
    cls = reg.teacher_class(name)
    return lambda: cls()


@pytest.fixture(scope="module")
def study():
    s = busy_room()
    g = _group(s)
    reps = 1 if s.fire_model == "deterministic" else 3
    st = cf.step0_study(s, g, family_id="fam-x", lineage_id="lin-x", horizon=s.max_steps,
                        max_horizon=s.max_steps, replicates=reps, continuation_factory=_t("T1"))
    return s, g, st


def test_step0_study_uses_stored_proposals_and_verifies(study):
    s, g, st = study
    assert st["status"] == "contrasted" and st["report"]["scope"] == SCOPE and st["report"]["prefix_records"] == []
    for n in CANDIDATES:
        assert st["probe"]["proposals"][n] == g["policies"][n]["decisions"][0]["proposal"]
    v = cf.verify_shared_state_study(st, s, g, continuation_factory=_t("T1"))
    assert v["verified"] and v["status"] == "contrasted" and v["teachers_reasked"] is False
    src = st["report"]["prefix_source"]
    assert src["kind"] == cf.STEP0_SOURCE and src["family_id"] == "fam-x"


def test_step0_records_are_class_c_or_d_with_zero_weight(study):
    s, g, st = study
    recs = ds.preferences(st, s, g)
    assert recs and {r["class"] for r in recs} <= {"C", "D"} and all(r["scope"] == SCOPE for r in recs)
    assert all(r["family_id"] == "fam-x" for r in recs)
    w = ds.assign_weights(recs)
    assert all(r["weight"] == 0 for r in w)
    assert all(r["approval"]["status"] == "candidate_not_approved" for r in w if r["class"] == "C")
    v = ds.validate_dataset(w)
    assert v["ok"], v["problems"]


def test_same_pair_under_two_continuations_is_not_a_duplicate(study):
    s, g, st = study
    reps = 1 if s.fire_model == "deterministic" else 3
    st4 = cf.step0_study(s, g, family_id="fam-x", lineage_id="lin-x", horizon=s.max_steps,
                         max_horizon=s.max_steps, replicates=reps, continuation_factory=_t("T4"))
    recs = ds.assign_weights(ds.preferences(st, s, g) + ds.preferences(st4, s, g))
    v = ds.validate_dataset(recs)
    assert v["ok"], v["problems"]
    one = ds.assign_weights(ds.preferences(st, s, g))
    twice = ds.validate_dataset(one + [one[0]])
    assert not twice["ok"] and any("duplicate" in p for p in twice["problems"])


def test_tampered_probe_or_group_is_refused(study):
    s, g, st = study
    bad = copy.deepcopy(st)
    t0 = bad["probe"]["proposals"]["T0"]
    pid = next(iter(t0["actions"]))
    t0["actions"][pid] = "stay" if t0["actions"][pid] != "stay" else "north"
    body = {k: v for k, v in bad["probe"].items() if k != "probe_hash"}
    bad["probe"]["probe_hash"] = cf.digest(mt.deterministic_view(body))
    with pytest.raises(cf.CounterfactualError):
        cf.verify_shared_state_study(bad, s, g)
    with pytest.raises(cf.CounterfactualError, match="not re-asked"):
        cf.verify_shared_state_study(st, s, g, factories=real_factories())


def test_fixture_scope_group_and_horizon_caps():
    s = busy_room()
    with pytest.raises(cf.CounterfactualError, match="authorized real scope"):
        cf.step0_study(s, _group(s, mt.SCOPE_FIXTURE), family_id="f", lineage_id="l", horizon=4)
    assert cf.horizon_cap(SCOPE) == 300 and cf.horizon_cap(cf.SCOPE) == cf.HARD_MAX_HORIZON == 64
    with pytest.raises(cf.CounterfactualError, match="authorization"):
        cf.counterfactual_fixture(s, [], {"a": {}, "b": {}}, scope="p1-12-train-pilot@1")
