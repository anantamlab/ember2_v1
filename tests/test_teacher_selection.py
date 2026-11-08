"""Teacher proposals (issue #25): common record, fabricated values, joint resolver checks, disagreement."""
from __future__ import annotations

import random

import pytest

from ember_v1.sim import Simulator
from ember_v1.sim.canonical import digest
from ember_v1.teacher import proposals as pr
from ember_v1.teacher.t0 import T0ShortestPath
from ember_v1.teacher.t1 import T1CongestionAware
from ember_v1.teacher.t2 import T2StochasticRollout
from ember_v1.teacher.t3 import T3DynamicACO
from gen_helpers import random_scenario
from test_teacher_t0 import ascii_scenario


@pytest.fixture(scope="module")
def obs():
    return Simulator(random_scenario(random.Random(31))).observe()


@pytest.fixture(scope="module")
def props(obs):
    return [pr.proposal(p.decide(obs)) for p in (T0ShortestPath(), T1CongestionAware(), T2StochasticRollout(),
                                                 T3DynamicACO())]


def _rehash(p, **ch):
    q = {**p, **ch}
    body = {k: v for k, v in q.items() if k not in ("compute", "proposal_hash")}
    return {**q, "proposal_hash": digest(body)}


def test_proposals_from_all_registered_teachers_validate(obs, props):
    for p in props:
        assert pr.validate_proposal(obs, p)["teacher"]["name"] in ("T0", "T1", "T2", "T3")
        assert (p["values"] is not None) == (p["teacher"]["name"] == "T2")
    assert props[0]["proposal_hash"] == pr.proposal(T0ShortestPath().decide(obs))["proposal_hash"]


def test_invalid_partial_and_fabricated_proposals_fail(obs, props):
    t0 = props[0]
    pid = sorted(t0["actions"])[0]
    with pytest.raises(pr.ProposalError, match="hash"):
        pr.validate_proposal(obs, {**t0, "actions": {**t0["actions"], pid: "stay"}} if t0["actions"][pid] != "stay"
                             else {**t0, "actions": {**t0["actions"], pid: "north"}})
    with pytest.raises(pr.ProposalError, match="partial"):
        pr.validate_proposal(obs, _rehash(t0, actions={k: v for k, v in t0["actions"].items() if k != pid}))
    with pytest.raises(pr.ProposalError, match="partial"):
        pr.validate_proposal(obs, _rehash(t0, actions={**t0["actions"], "P999": "stay"}))
    with pytest.raises(pr.ProposalError, match="invalid action"):
        pr.validate_proposal(obs, _rehash(t0, actions={**t0["actions"], pid: "jump"}))
    with pytest.raises(pr.ProposalError, match="fabricated"):
        pr.validate_proposal(obs, _rehash(t0, values={pid: {"stay": 0.0}}))
    t2 = props[2]
    if t2["values"]:
        with pytest.raises(pr.ProposalError, match="allowed actions"):
            pr.validate_proposal(obs, _rehash(t2, values={pid: {"teleport": 1.0}}))
    with pytest.raises(pr.ProposalError, match="another observation"):
        pr.validate_proposal(Simulator(random_scenario(random.Random(32))).observe(), t0)
    with pytest.raises(pr.ProposalError, match="missing or extra"):
        pr.validate_proposal(obs, {**t0, "score": 1})


def test_joint_check_detects_a_harmful_combination():
    # Two people face each other in a corridor; P1 swaps forward into P0's cell while P0 moves into P1's cell.
    s = ascii_scenario(["0.ab.1"])
    o = Simulator(s).observe()
    ids = sorted(p["id"] for p in o["people"])
    pos = {p["id"]: tuple(p["position"]) for p in o["people"]}
    left, right = sorted(ids, key=lambda i: pos[i][1])
    good = pr.joint_check(o, {left: "west", right: "east"})
    assert good["valid_joint_plan"] and good["blocked"] == []
    bad = pr.joint_check(o, {left: "east", right: "west"})      # head-on swap is blocked by the resolver
    assert not bad["valid_joint_plan"] and set(bad["blocked"]) == {left, right}


def test_combine_records_sources_and_checks_the_joint_plan():
    obs = Simulator(ascii_scenario(["0.abc..1", "........"])).observe()
    props = [pr.proposal(p.decide(obs)) for p in (T0ShortestPath(), T1CongestionAware())]
    active = sorted(props[0]["actions"])
    assert len(active) == 3
    choice = {pid: props[i % 2] for i, pid in enumerate(active)}
    c = pr.combine(obs, choice)
    assert set(c["actions"]) == set(active) and set(c["sources"].values()) <= {"T0", "T1"}
    assert "valid_joint_plan" in c["joint"]
    with pytest.raises(pr.ProposalError, match="every active person"):
        pr.combine(obs, {active[0]: props[0]})


def test_disagreement_is_descriptive_only(obs, props):
    d = pr.disagreement(obs, props)
    assert d["teachers"] == ["T0", "T1", "T2", "T3"] and len(d["pairwise_agreement"]) == 6
    assert "not evidence" in d["note"]
    with pytest.raises(pr.ProposalError, match="one proposal per teacher"):
        pr.disagreement(obs, [props[0], props[0]])


# Inventory, protocol, prospective gate and arbiter: tests/test_teacher_selection_gate.py
