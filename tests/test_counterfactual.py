"""P1-13: controlled, bounded joint-action intervention on ONE replayed public state."""
from __future__ import annotations

import copy

import pytest

from ember_v1 import counterfactual as cf
from ember_v1 import multiteacher as mt
from ember_v1.sim import Simulator
from ember_v1.sim.canonical import digest
from ember_v1.sim.simulator import SimulationError
from ember_v1.teacher.t4_v12 import T4RiskPriorityV12
from sim_helpers import scenario
from test_multiteacher import FixturePolicy, busy_room, factories, real_factories


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
    assert report["report_hash"] == digest(mt.deterministic_view(
        {k: v for k, v in report.items() if k != "report_hash"}))
    assert cf.verify_counterfactual(report, room())["verified"]
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


# ----- v0.2.0: matched future-fire replicates, verdicts, verification, five-teacher study -----------
def stay(sim):
    return {pid: "stay" for pid in sim.active_ids()}


def test_fire_fork_changes_only_future_draws_and_is_bound_into_the_chain():
    s = busy_room()
    a, b, c = Simulator(s), Simulator(s), Simulator(s)
    for x in (a, b, c):
        x.step(stay(x))
    state = a.private_state()
    b.fork_fire(7)
    c.fork_fire(7)
    assert {k: v for k, v in b.private_state().items() if k != "rng"} == \
           {k: v for k, v in state.items() if k != "rng"}
    assert b.chain_hash == c.chain_hash != a.chain_hash
    rb, rc = [], []
    for _ in range(6):
        if b.terminated:
            break
        rb.append(b.step(stay(b))["ignitions"])
        rc.append(c.step(stay(c))["ignitions"])
    assert rb == rc                                  # same fork seed: same future
    with pytest.raises(ValueError):
        Simulator(s).fork_fire(-1)
    t = Simulator(scenario(["E.", ".."], {"P1": (0, 1)}, max_steps=1, sid="t"))
    t.step({"P1": "west"})
    with pytest.raises(SimulationError, match="terminated"):
        t.fork_fire(1)


def test_replicates_share_fire_across_alternatives_and_differ_between_replicates():
    rep = cf.counterfactual_fixture(busy_room(), [], {"stay": stay(Simulator(busy_room())),
                                                       "north": {p: "north" for p in Simulator(busy_room()).active_ids()}},
                                    horizon=6, replicates=4)
    a, b = rep["branches"]["north"]["replicates"], rep["branches"]["stay"]["replicates"]
    assert [r["fork_seed_evaluator_only"] for r in a] == [r["fork_seed_evaluator_only"] for r in b]
    assert a[0]["fork_seed_evaluator_only"] is None and all(isinstance(r["fork_seed_evaluator_only"], int) for r in a[1:])
    for ra, rb in zip(a, b):
        n = min(len(ra["records"]), len(rb["records"]))
        assert [x["ignitions"] for x in ra["records"][:n]] == [x["ignitions"] for x in rb["records"][:n]]
    fires = {digest([x["ignitions"] for x in r["records"]]) for r in a}
    assert len(fires) > 1                            # replicates really sample different futures
    assert rep["branches"]["north"]["horizon_records"] == a[0]["records"]   # replicate 0 = factual stream
    assert cf.verify_counterfactual(rep, busy_room())["verified"]


@pytest.mark.parametrize("deltas,censored,stochastic,verdict,reason", [
    ([(-1, -1), (0, 0), (0, 0)], False, True, "b_better", None),
    ([(1, 0), (0, 0), (1, 1)], False, True, "a_better", None),
    ([(-1, -1), (1, 1), (0, 0)], False, True, "inconclusive", "mixed_sign"),
    ([(-1, 0), (-1, 1), (-1, 0)], False, True, "inconclusive", "mixed_sign"),   # guard (casualties) worse
    ([(-1, -1), (-1, -1)], False, True, "inconclusive", "insufficient_replicates"),
    ([(-1, -1)], False, False, "b_better", None),                               # deterministic: exact
    ([(0, 0), (0, 0), (0, 0)], False, True, "no_difference", None),
    ([(-1, -1), (-1, -1), (-1, -1)], True, True, "inconclusive", "censored"),
])
def test_conservative_verdict_rule(deltas, censored, stochastic, verdict, reason):
    per = [{"replicate": k, "censored": censored and k == 1, "delta_casualties_plus_unresolved": p,
            "delta_casualties": g} for k, (p, g) in enumerate(deltas)]
    got = cf.classify(per, stochastic=stochastic)
    assert got["verdict"] == verdict and (reason is None or got["reason"] == reason)


def test_sign_test_is_exact():
    assert cf.sign_test_p(0, 0) is None
    assert cf.sign_test_p(5, 0) == pytest.approx(0.0625)
    assert cf.sign_test_p(2, 2) == 1.0


def test_replicate_count_rules():
    det = scenario(["E...", "....", "..F."], {"P1": (1, 1), "P2": (1, 3)}, max_steps=6, sid="det")
    alts = {"s": {"P1": "stay", "P2": "stay"}, "n": {"P1": "north", "P2": "west"}}
    with pytest.raises(cf.CounterfactualError, match="deterministic fire"):
        cf.counterfactual_fixture(det, [], alts, replicates=2)
    with pytest.raises(cf.CounterfactualError, match="replicate count"):
        cf.counterfactual_fixture(room(), [], alternatives(), replicates=0)
    with pytest.raises(cf.CounterfactualError, match="replicate count"):
        cf.counterfactual_fixture(room(), [], alternatives(), replicates=cf.MAX_REPLICATES + 1)


def test_continuation_must_be_a_pinned_retained_teacher():
    with pytest.raises(cf.CounterfactualError, match="registry-pinned"):
        cf.counterfactual_fixture(room(), [], alternatives(), continuation_factory=lambda: FixturePolicy("T1"))
    rep = cf.counterfactual_fixture(room(), [], alternatives(), continuation_factory=T4RiskPriorityV12)
    assert rep["continuation"]["name"] == "T4" and rep["continuation"]["version"] == "1.2.0"


def test_tampered_counterfactual_fails_verification():
    s = busy_room()
    alts = {"stay": stay(Simulator(s)), "north": {p: "north" for p in Simulator(s).active_ids()}}
    rep = cf.counterfactual_fixture(s, [], alts, horizon=6, replicates=3)
    reseal = lambda r: {**r, "report_hash": digest(mt.deterministic_view(  # noqa: E731
        {k: v for k, v in r.items() if k != "report_hash"}))}
    bad = copy.deepcopy(rep)
    bad["contrasts"][0]["verdict"] = "a_better" if rep["contrasts"][0]["verdict"] != "a_better" else "b_better"
    with pytest.raises(cf.CounterfactualError, match="hash mismatch"):
        cf.verify_counterfactual(bad, s)
    with pytest.raises(cf.CounterfactualError, match="verdict does not follow"):
        cf.verify_counterfactual(reseal(bad), s)
    bad = copy.deepcopy(rep)
    bad["branches"]["stay"]["replicates"][2]["fork_seed_evaluator_only"] = 12345
    with pytest.raises(cf.CounterfactualError, match="fork seed"):
        cf.verify_counterfactual(reseal(bad), s)
    bad = copy.deepcopy(rep)
    bad["branches"]["north"]["replicates"][1]["records"][-1]["casualties"] = {"P9": "fire_spread"}
    with pytest.raises(cf.CounterfactualError, match="regenerate"):
        cf.verify_counterfactual(reseal(bad), s)
    bad = copy.deepcopy(rep)
    bad["branches"]["north"]["replicates"][0]["records"][0]["actions"] = alts["stay"]
    with pytest.raises(cf.CounterfactualError, match="did not apply the intervention"):
        cf.verify_counterfactual(reseal(bad), s)


@pytest.fixture(scope="module")
def study():
    s = busy_room()
    r = mt.collect_fixture(s, real_factories())
    return s, r


def test_five_teacher_shared_state_study_end_to_end(study):
    s, r = study
    out = cf.shared_state_study(s, r, "T1", 3, real_factories(), horizon=10, max_horizon=10, replicates=5)
    assert out["status"] == "contrasted" and out["training_eligible"] is False
    rep = out["report"]
    assert rep["prefix_steps"] == 3 and rep["prefix_source"]["content_hash"] == r["content_hash"]
    assert rep["prefix_source"]["probe_hash"] == out["probe"]["probe_hash"]
    assert rep["common_observation_digest"] == r["policies"]["T1"]["decisions"][3]["observation_digest"]
    # every teacher is represented exactly once across the distinct alternatives
    proposers = sorted(t for b in rep["branches"].values() for t in b["sources"]["proposed_by"])
    assert proposers == ["T0", "T1", "T2", "T3", "T4"] and len(rep["branches"]) >= 2
    verdicts = {c["verdict"] for c in rep["contrasts"]}
    assert verdicts <= set(cf.VERDICTS) and verdicts & {"a_better", "b_better"}
    assert all(c["training_preference_allowed"] is False for c in rep["contrasts"])
    assert all(c["preference_candidate"] == (c["verdict"] in ("a_better", "b_better")) for c in rep["contrasts"])
    assert cf.verify_counterfactual(rep, s)["verified"]


def test_no_disagreement_state_makes_no_contrast(study):
    s, r = study
    out = cf.shared_state_study(s, r, "T1", 1, real_factories())
    assert out["status"] == "no_disagreement" and "report" not in out
    with pytest.raises(cf.CounterfactualError, match="outside"):
        cf.shared_state_study(s, r, "T1", 99, real_factories())
    bad = copy.deepcopy(r)
    bad["policies"]["T1"]["outcome"]["casualties"] += 1
    with pytest.raises(mt.MultiTeacherError, match="hash mismatch"):
        cf.shared_state_study(s, bad, "T1", 0, real_factories())


def test_composite_requires_a_whole_unblocked_joint_map():
    obs = Simulator(room()).observe()
    probe = mt.probe_shared_state(obs, factories({"T0": "north", "T1": "west", "T2": "east"}))
    acts, src = cf.composite_alternative(obs, probe, {"P1": "T0", "P2": "T1"})
    assert acts == {"P1": "north", "P2": "west"} and src["per_person_source"] == {"P1": "T0", "P2": "T1"}
    assert src["joint"]["valid_joint_plan"]
    with pytest.raises(cf.CounterfactualError, match="complete"):
        cf.composite_alternative(obs, probe, {"P1": "T0"})
    # P1 (1,1) east and P2 (1,4) west do not collide; force a collision on a 2-person line instead
    line = scenario(["E...."], {"P1": (0, 1), "P2": (0, 3)}, max_steps=4, sid="line")
    lo = Simulator(line).observe()
    lp = mt.probe_shared_state(lo, factories({"T0": "east", "T1": "west"}))
    with pytest.raises(cf.CounterfactualError, match="blocks"):
        cf.composite_alternative(lo, lp, {"P1": "T0", "P2": "T1"})
    alts, sources = cf.alternatives_from_probe(probe)
    assert set(alts) == {"T0", "T1", "T2", "T3+T4"}
    assert sources["T3+T4"]["proposed_by"] == ["T3", "T4"]
