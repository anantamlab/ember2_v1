"""Five-teacher selection prep (issue #25): proposals, joint checks, evidence inventory, protocol and gate."""
from __future__ import annotations

import json
import random

import pytest

from ember_v1.sim import Simulator
from ember_v1.sim.canonical import digest
from ember_v1.teacher import proposals as pr
from ember_v1.teacher import selection as sel
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


# ----- evidence inventory and gate ---------------------------------------------------------------------------
def test_inventory_of_committed_dev_reports():
    inv = sel.evidence_inventory()
    c = inv["candidates"]
    assert [t for t in sel.CANDIDATES if c[t]["status"] == "dev_evidence"] == ["T0", "T1", "T2", "T3"]
    assert c["T4"]["status"] == "missing"
    assert inv["cross_report_conflicts"] == [] and inv["same_dev_metadata"] and inv["same_positions"]
    for t in ("T0", "T1", "T2", "T3"):
        assert c[t]["scenarios"] == 48 and len(c[t]["config_hashes"]) == 1 and c[t]["failures"] == 0
    assert len(c["T1"]["reports"]) == 3 and len(c["T0"]["reports"]) == 3


def test_inventory_detects_cross_report_conflicts(tmp_path):
    a = json.loads(sel.DEFAULT_REPORTS[1].read_text(encoding="utf-8"))
    b = json.loads(json.dumps(a))
    b["rows"][0]["T1"]["final_transition_hash"] = "sha256:" + "0" * 64
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a), encoding="utf-8")
    pb.write_text(json.dumps(b), encoding="utf-8")
    inv = sel.evidence_inventory([pa, pb])
    assert {x["teacher"] for x in inv["cross_report_conflicts"]} == {"T1"}
    assert "cross-report reproducibility conflicts" in sel.selection_gate(
        None, inv, protocol_commit=None, evidence_after_commit=False)["reasons"]


def _protocol(**over):
    p = {"protocol_id": "test", "candidates": list(sel.CANDIDATES), "data": "dev p1-02-v1",
         "primary_endpoint": "casualties", "secondary_endpoints": [], "comparisons": [],
         "decision_rule": "ci_and_sign_test", "simplicity_rule": "prefer simpler on no evidence",
         "arbiter": None, "compute_budget": {"usd_cap": 15.0}, "stop_rules": ["stop on failed gate"],
         "excluded_data": ["locked_test", "blind_test", "emberbench"], "frozen": True}
    p.update(over)
    return p


@pytest.mark.parametrize("over,match", [
    ({"candidates": ["T0", "T1"]}, "all five"),
    ({"frozen": False}, "frozen"),
    ({"excluded_data": ["locked_test"]}, "exclude"),
    ({"compute_budget": {"usd_cap": 50}}, "budget"),
    ({"compute_budget": {}}, "budget"),
    ({"arbiter": {"equal_compute_to_best_single": True}}, "arbiter"),
    ({"stop_rules": []}, "stop rules"),
])
def test_unsafe_protocols_are_refused(over, match):
    with pytest.raises(ValueError, match=match):
        sel.freeze_protocol(_protocol(**over))
    with pytest.raises(ValueError, match="misses"):
        sel.freeze_protocol({k: v for k, v in _protocol().items() if k != "stop_rules"})


def test_gate_never_selects_and_blocks_now():
    inv = sel.evidence_inventory()
    now = sel.selection_gate(None, inv, protocol_commit=None, evidence_after_commit=False)
    assert now["decision"] == "NO_SELECTION" and now["selected_teacher"] is None
    assert "T4: no DEV evidence" in now["reasons"]
    frozen = sel.freeze_protocol(_protocol())
    g = sel.selection_gate(frozen, inv, protocol_commit="abc", evidence_after_commit=True)
    assert g["reasons"] == ["T4: no DEV evidence"]
    full = json.loads(json.dumps(inv))
    full["candidates"]["T4"] = {**full["candidates"]["T1"], "config_hashes": ["sha256:x"]}
    ok = sel.selection_gate(frozen, full, protocol_commit="abc", evidence_after_commit=True)
    assert ok == {"decision": "READY_FOR_PREREGISTERED_COMPARISON", "reasons": [], "selected_teacher": None}
    for kw in ({"protocol_commit": None, "evidence_after_commit": True},
               {"protocol_commit": "abc", "evidence_after_commit": False}):
        assert sel.selection_gate(frozen, full, **kw)["decision"] == "NO_SELECTION"
    tampered = {**frozen, "primary_endpoint": "evacuated"}
    assert "protocol hash does not match" in sel.selection_gate(
        tampered, full, protocol_commit="abc", evidence_after_commit=True)["reasons"]
