"""P1-12 fixture contracts: five matched rollouts, replay, fire draws, same-state probes.

These tests are deliberately NOT real TRAIN source generation or policy scoring.
"""
from __future__ import annotations

import copy

import pytest

from ember_v1 import multiteacher as mt
from ember_v1.sim import Simulator
from ember_v1.sim.canonical import digest
from ember_v1.teacher.interface import ReferencePolicy
from ember_v1.teacher.registry import CANDIDATES
from sim_helpers import scenario


class FixturePolicy(ReferencePolicy):
    name = "T0"
    policy = "synthetic_fixture"
    version = "0.0.0"

    def __init__(self, name: str, action: str = "stay"):
        self.name = name
        self.action = action
        super().__init__()

    def _plan(self, obs):
        return ({p["id"]: self.action for p in obs["people"] if p["status"] == "active"},
                {"fixture": True})


def factories(actions=None):
    acts = actions or {}
    return {name: (lambda n=name: FixturePolicy(n, acts.get(n, "stay"))) for name in CANDIDATES}


def room():
    return scenario(["E.....", "......", "..F...", "......"],
                    {"P1": (1, 1), "P2": (1, 4)}, fire="stochastic",
                    p=0.4, interval=1, seed=93, max_steps=4, enter_fire="blocked",
                    sid="p1-12-independent-fixture")


def test_five_full_trajectories_replay_and_pair_fire():
    s = room()
    r = mt.collect_fixture(s, factories({"T0": "north", "T3": "east"}))
    assert r["scope"] == "synthetic_fixture" and not r["training_eligible"]
    assert r["teacher_order"] == list(CANDIDATES)
    assert set(r["policies"]) == set(CANDIDATES)
    assert r["matched_exogenous_fire_prefix"] is True
    assert r["evidence_hash"] == digest({k: v for k, v in r.items() if k != "evidence_hash"})
    for p in r["policies"].values():
        assert p["replay"]["status"] == "complete"
        assert p["outcome"]["people"] == 2
        assert p["outcome"]["casualties"] + p["outcome"]["evacuated"] + p["outcome"]["unresolved"] == 2
        assert p["decisions"]
        assert all("rng" not in step["observation"] and "trapped" not in step["observation"]
                   and "seed" not in step["observation"] for step in p["decisions"])
        assert all(set(step["proposal"]["actions"]) ==
                   {q["id"] for q in step["observation"]["people"] if q["status"] == "active"}
                   for step in p["decisions"])
    assert r["policies"]["T0"]["decisions"][0]["proposal"]["actions"] != \
           r["policies"]["T1"]["decisions"][0]["proposal"]["actions"]


def test_rollouts_identical_when_all_policies_choose_same_actions():
    a = mt.collect_fixture(room(), factories())
    b = mt.collect_fixture(room(), factories())
    # Runtime metadata is intentionally in the evidence hash and varies by host.
    for report in (a, b):
        assert report["evidence_hash"] == digest({k: v for k, v in report.items() if k != "evidence_hash"})
    assert {p["final_transition_hash"] for p in a["policies"].values()} == \
           {p["final_transition_hash"] for p in b["policies"].values()}
    hashes = {p["final_transition_hash"] for p in a["policies"].values()}
    assert len(hashes) == 1


def test_shared_observation_probe_has_ten_pairs_and_no_quality_label():
    obs = Simulator(room()).observe()
    p = mt.probe_shared_state(obs, factories({"T0": "north", "T4": "west"}))
    assert set(p["proposals"]) == set(CANDIDATES)
    assert len(p["disagreement"]["pairwise_agreement"]) == 10
    assert p["disagreement"]["unanimous_fraction"] < 1
    assert not p["training_eligible"]
    assert "causal" in p["note"]
    assert p["probe_hash"] == digest({k: v for k, v in p.items() if k != "probe_hash"})


def test_only_fixture_scope_allowed_and_no_untrusted_real_split():
    with pytest.raises(mt.MultiTeacherError, match="reviewed"):
        mt.collect_fixture(room(), factories(), scope="train")
    with pytest.raises(mt.MultiTeacherError, match="reviewed"):
        mt.collect_fixture(room(), factories(), scope="dev")
    with pytest.raises(mt.MultiTeacherError, match="reviewed"):
        mt.collect_fixture(room(), factories(), scope="selection_confirmation")


def test_no_missing_policy_or_identity_substitution():
    fs = factories()
    fs.pop("T4")
    with pytest.raises(mt.MultiTeacherError, match="exactly five"):
        mt.collect_fixture(room(), fs)
    fs = factories()
    fs["T4"] = lambda: FixturePolicy("T0")
    with pytest.raises(mt.MultiTeacherError, match="expected policy T4"):
        mt.collect_fixture(room(), fs)


def test_bounded_cpu_and_steps_fail_closed():
    with pytest.raises(mt.MultiTeacherError, match="bounds"):
        mt.collect_fixture(room(), factories(), max_steps=3)
    with pytest.raises(mt.MultiTeacherError, match="CPU cap"):
        mt.collect_fixture(room(), factories(), cpu_ms_per_teacher_cap=-1)
    with pytest.raises(mt.MultiTeacherError, match="bounds"):
        mt.collect_fixture(room(), factories(), max_people=1)


def test_observation_mismatch_proposal_refused():
    obs = Simulator(room()).observe()
    p = mt.probe_shared_state(obs, factories())
    altered = copy.deepcopy(obs)
    altered["step"] += 1
    from ember_v1.teacher.proposals import ProposalError, validate_proposal
    with pytest.raises(ProposalError, match="another observation"):
        validate_proposal(altered, p["proposals"]["T0"])
