"""P1-13: controlled, bounded joint-action intervention on ONE replayed public state."""
from __future__ import annotations

import copy

import pytest

from ember_v1 import counterfactual as cf
from ember_v1.sim import Simulator
from ember_v1.sim.canonical import digest
from sim_helpers import scenario


def room():
    return scenario(["E.....", "......", "..F...", "......"],
                    {"P1": (1, 1), "P2": (1, 4)}, fire="stochastic",
                    p=0.35, interval=2, seed=171, max_steps=5,
                    enter_fire="blocked", sid="p1-13-common-fixture")


def alternatives():
    return {"a_stay": {"P1": "stay", "P2": "stay"},
            "b_depart": {"P1": "north", "P2": "west"}}


def test_two_branches_have_same_common_observation_and_fire():
    report = cf.counterfactual_fixture(room(), [], alternatives(), horizon=3)
    assert report["scope"] == "synthetic_fixture" and not report["training_eligible"]
    assert len(report["branches"]) == 2 and report["matched_exogenous_fire_prefix"]
    assert report["report_hash"] == digest({k: v for k, v in report.items() if k != "report_hash"})
    a, b = report["branches"]["a_stay"], report["branches"]["b_depart"]
    assert a["intervention_actions"] != b["intervention_actions"]
    assert a["intervention_transition"]["ignitions"] == b["intervention_transition"]["ignitions"]
    assert a["intervention_transition"]["fire_draws"] == b["intervention_transition"]["fire_draws"]
    assert a["continuation"]["name"] == b["continuation"]["name"] == "T1"
    assert all(x["training_preference_allowed"] is False for x in report["contrasts"])
    assert "individual-action" in report["causality_limit"]


def test_valid_prefix_is_replayed_before_joint_interventions():
    s = room()
    sim = Simulator(s)
    prefix = [sim.step({"P1": "stay", "P2": "stay"})]
    rep = cf.counterfactual_fixture(s, prefix, alternatives(), horizon=2)
    assert rep["prefix_steps"] == 1
    assert rep["common_observation_digest"] == digest(sim.observe())
    assert rep["prefix_hash"] == digest(prefix)
    for b in rep["branches"].values():
        assert b["replay"] in ("partial", "complete")
        assert b["intervention_transition"]["step"] == 2


def test_corrupted_prefix_cannot_be_used_for_counterfactual_label():
    s = room()
    sim = Simulator(s)
    prefix = [sim.step({"P1": "stay", "P2": "stay"})]
    broken = copy.deepcopy(prefix)
    broken[0]["state_hash"] = "sha256:" + "0" * 64
    with pytest.raises(cf.CounterfactualError, match="prefix diverged"):
        cf.counterfactual_fixture(s, broken, alternatives(), horizon=2)


def test_censoring_is_explicit_and_never_produces_training_preference():
    rep = cf.counterfactual_fixture(room(), [], alternatives(), horizon=1)
    assert all(x["outcome"]["censored"] for x in rep["branches"].values())
    assert rep["contrasts"][0]["censored"]
    assert not rep["contrasts"][0]["training_preference_allowed"]


def test_partial_joint_map_or_identical_alternatives_rejected():
    bad = alternatives()
    bad["b_depart"].pop("P1")
    with pytest.raises(cf.CounterfactualError, match="incomplete or invalid"):
        cf.counterfactual_fixture(room(), [], bad)
    same = {"one": alternatives()["a_stay"], "two": alternatives()["a_stay"]}
    with pytest.raises(cf.CounterfactualError, match="identical"):
        cf.counterfactual_fixture(room(), [], same)


def test_only_fixture_scope_and_bounded_horizon_allowed():
    with pytest.raises(cf.CounterfactualError, match="authorization"):
        cf.counterfactual_fixture(room(), [], alternatives(), scope="train")
    with pytest.raises(cf.CounterfactualError, match="horizon"):
        cf.counterfactual_fixture(room(), [], alternatives(), horizon=20)
    with pytest.raises(cf.CounterfactualError, match="CPU cap"):
        cf.counterfactual_fixture(room(), [], alternatives(), cpu_ms_per_branch_cap=-1)


def test_prefix_already_terminal_is_refused():
    s = scenario(["E...", "....", "..F.", "...."], {"P1": (0, 1)},
                 max_steps=1, sid="terminal-fixture")
    sim = Simulator(s)
    prefix = [sim.step({"P1": "stay"})]
    with pytest.raises(cf.CounterfactualError, match="terminal"):
        cf.counterfactual_fixture(s, prefix, {"one": {"P1": "stay"}, "two": {"P1": "west"}})
