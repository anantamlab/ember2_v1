"""P1-12 fixture contracts: five matched rollouts, replay, fire draws, same-state probes.

These tests are deliberately NOT real TRAIN source generation or policy scoring.
"""
from __future__ import annotations

import copy
import dataclasses
import time

import pytest

from ember_v1 import multiteacher as mt
from ember_v1.sim import Simulator
from ember_v1.sim.canonical import digest
from ember_v1.teacher import registry as reg
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
    assert r["group_complete"] and r["comparison_eligible"] and r["failed_teachers"] == []
    assert r["evidence_hash"] == digest({k: v for k, v in r.items() if k != "evidence_hash"})
    assert r["content_hash"] == digest(mt.deterministic_view(
        {k: v for k, v in r.items() if k not in ("content_hash", "evidence_hash")}))
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
    # Runtime metadata is in the evidence hash and varies by host; the content hash must not.
    for report in (a, b):
        assert report["evidence_hash"] == digest({k: v for k, v in report.items() if k != "evidence_hash"})
    assert a["content_hash"] == b["content_hash"]
    assert {p["final_transition_hash"] for p in a["policies"].values()} == \
           {p["final_transition_hash"] for p in b["policies"].values()}
    hashes = {p["final_transition_hash"] for p in a["policies"].values()}
    assert len(hashes) == 1
    # identical play: every step is a genuinely common state
    assert a["common_state_steps"] == list(range(len(a["policies"]["T0"]["decisions"])))


def test_shared_observation_probe_has_ten_pairs_and_no_quality_label():
    obs = Simulator(room()).observe()
    p = mt.probe_shared_state(obs, factories({"T0": "north", "T4": "west"}))
    assert set(p["proposals"]) == set(CANDIDATES)
    assert len(p["disagreement"]["pairwise_agreement"]) == 10
    assert p["disagreement"]["unanimous_fraction"] < 1
    assert not p["training_eligible"]
    assert "causal" in p["note"]
    assert p["distinct_joint_actions"] == 3
    assert p["probe_hash"] == digest(mt.deterministic_view({k: v for k, v in p.items() if k != "probe_hash"}))


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


# ----- v0.2.0: real teachers, provenance, replay/regeneration, leakage, failures -----------------------
def real_factories():
    classes = reg.retained_classes()
    return {n: (lambda n=n: classes[n]()) for n in CANDIDATES}


def busy_room():
    # Hand-built: wall, two exits, five people. Real teachers diverge here (five distinct
    # trajectories, different casualty counts), so the matched comparison is not vacuous.
    return scenario(["E...#...", "....#...", "........", "..F.....", "......E."],
                    {"P1": (2, 1), "P2": (2, 2), "P3": (2, 3), "P4": (1, 3), "P5": (4, 5)},
                    fire="stochastic", p=0.5, interval=1, seed=5, max_steps=10, enter_fire="blocked",
                    sid="p1-12-real-teacher-fixture")


@pytest.fixture(scope="module")
def real_report():
    return mt.collect_fixture(busy_room(), real_factories())


def test_real_five_teachers_complete_pinned_and_verified(real_report):
    r = real_report
    assert r["group_complete"] and r["teacher_order"] == list(CANDIDATES)
    prov = r["provenance"]
    assert prov["simulator"]["module_fingerprint"].startswith("sha256:")
    assert prov["collector"]["schema_version"] == mt.SCHEMA_VERSION
    for n in CANDIDATES:
        t = prov["teachers"][n]
        assert t["registry_pinned"] is True and t["registry_status"] == "retained"
        assert t["code_fingerprint"].startswith("sha256:")
        o = r["policies"][n]["outcome"]
        assert o["casualties"] + o["evacuated"] + o["unresolved"] == o["people"] == 5
        assert o["casualties_plus_unresolved"] == o["casualties"] + o["unresolved"]
        rt = r["policies"][n]["runtime"]
        assert len(rt["collector_cpu_ms"]) == len(r["policies"][n]["decisions"]) == o["steps"]
    assert prov["teachers"]["T4"]["identity"]["version"] == "1.2.0"
    assert mt.verify_fixture(r, busy_room())["verified"]


def test_real_report_regenerates_and_teachers_are_markov_at_trajectory_states(real_report):
    assert mt.regenerate(real_report, busy_room(), real_factories())["regenerated"]
    # A fresh teacher at its OWN trajectory state reproduces its recorded proposal (no hidden
    # history), so a fresh-instance probe at a replayed common state is that teacher's decision.
    for n in CANDIDATES:
        steps = real_report["policies"][n]["decisions"]
        k = min(2, len(steps) - 1)
        probe = mt.probe_shared_state(steps[k]["observation"], real_factories())
        assert probe["proposals"][n]["proposal_hash"] == steps[k]["proposal"]["proposal_hash"]


def test_same_step_index_is_not_a_common_state(real_report):
    r = real_report
    assert r["common_state_steps"] == [0]           # only the initial state is shared
    assert len({p["final_transition_hash"] for p in r["policies"].values()}) == 5
    assert len({p["outcome"]["casualties"] for p in r["policies"].values()}) > 1
    digests = {n: [d["observation_digest"] for d in p["decisions"]] for n, p in r["policies"].items()}
    for i in range(1, min(len(v) for v in digests.values())):
        same = len({v[i] for v in digests.values()}) == 1
        assert (i in r["common_state_steps"]) is same


def test_tampered_report_fails_verification(real_report):
    s = busy_room()
    bad = copy.deepcopy(real_report)
    bad["policies"]["T1"]["outcome"]["casualties"] += 1
    with pytest.raises(mt.MultiTeacherError, match="hash mismatch"):
        mt.verify_fixture(bad, s)
    resealed = mt.seal(bad)                         # attacker recomputes hashes: regeneration still fails
    with pytest.raises(mt.MultiTeacherError, match="outcome does not regenerate"):
        mt.verify_fixture(resealed, s)
    bad = copy.deepcopy(real_report)
    bad["policies"]["T3"]["decisions"][0]["observation"]["burning"] = []
    with pytest.raises(mt.MultiTeacherError, match="not the regenerated state"):
        mt.verify_fixture(mt.seal(bad), s)
    bad = copy.deepcopy(real_report)
    bad["group_complete"] = False
    with pytest.raises(mt.MultiTeacherError, match="completeness"):
        mt.verify_fixture(mt.seal(bad), s)
    bad = copy.deepcopy(real_report)
    bad["training_eligible"] = True
    with pytest.raises(mt.MultiTeacherError, match="eligibility"):
        mt.verify_fixture(mt.seal(bad), s)
    with pytest.raises(mt.MultiTeacherError, match="another scenario"):
        mt.verify_fixture(real_report, room())


def test_unmatched_fire_between_trajectories_raises(real_report):
    pol = copy.deepcopy(real_report["policies"])
    pol["T4"]["decisions"][0]["record"]["fire_draws"] += 1
    with pytest.raises(mt.MultiTeacherError, match="unmatched exogenous fire for T4"):
        mt.check_matched_fire(pol)


class SpyPolicy(FixturePolicy):
    seen: list = []

    def decide(self, observation):
        SpyPolicy.seen.append(copy.deepcopy(observation))
        clean = copy.deepcopy(observation)
        observation["people"].clear()               # hostile mutation of its own input
        observation["burning"].append([0, 0])
        return super().decide(clean)


def test_policy_gets_only_a_public_copy_and_cannot_alter_the_record():
    SpyPolicy.seen = []
    fs = factories()
    fs["T2"] = lambda: SpyPolicy("T2")
    r = mt.collect_fixture(room(), fs)
    assert SpyPolicy.seen and all(set(o) == mt.PUBLIC_OBSERVATION_KEYS for o in SpyPolicy.seen)
    assert all(not mt._private_key_paths(o) for o in SpyPolicy.seen)
    assert mt.verify_fixture(r, room())["verified"]       # stored observations are untouched
    with pytest.raises(mt.MultiTeacherError, match="not exactly the public"):
        mt.check_public_observation(Simulator(room()).private_state())
    leak = Simulator(room()).observe()
    leak["rules"]["seed"] = 93
    with pytest.raises(mt.MultiTeacherError, match="private simulator fields"):
        mt.check_public_observation(leak)
    with pytest.raises(mt.MultiTeacherError, match="private simulator fields"):
        mt.probe_shared_state(leak, factories())


class CrashPolicy(FixturePolicy):
    def _plan(self, obs):
        if obs["step"] >= 1:
            raise RuntimeError("synthetic teacher bug")
        return super()._plan(obs)


class ForgedPolicy(FixturePolicy):
    def decide(self, observation):
        d = super().decide(observation)
        return dataclasses.replace(d, reproducibility={**d.reproducibility,
                                                       "observation_digest": "sha256:" + "1" * 64})


class SlowPolicy(FixturePolicy):
    def _plan(self, obs):
        t0 = time.thread_time()
        while time.thread_time() - t0 < 0.3:
            pass
        return super()._plan(obs)


@pytest.mark.parametrize("cls,kind,kw", [(CrashPolicy, "teacher_exception", {}),
                                         (ForgedPolicy, "invalid_proposal", {}),
                                         (SlowPolicy, "cpu_cap", {"cpu_ms_per_teacher_cap": 100})])
def test_teacher_failure_is_recorded_and_group_is_not_comparable(cls, kind, kw):
    fs = factories({"T1": "north"})
    fs["T3"] = lambda: cls("T3")
    r = mt.collect_fixture(room(), fs, **kw)
    t3 = r["policies"]["T3"]
    assert t3["status"] == "failed" and t3["failure"]["kind"] == kind and t3["outcome"] is None
    assert t3["replay"]["status"] == "partial"
    assert not r["group_complete"] and not r["comparison_eligible"] and r["failed_teachers"] == ["T3"]
    assert mt.outcome_table(r) == {"descriptive_only": True, "comparable": False,
                                   "reason": "incomplete five-teacher group", "failed_teachers": ["T3"]}
    assert mt.verify_fixture(r, room())["verified"]


def test_train_protocol_draft_is_not_frozen_and_cites_real_numbers():
    import json
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    doc = (repo / "docs/P1_12_TRAIN_COLLECTION_PROTOCOL.md").read_text(encoding="utf-8")
    assert "**Status:** `DRAFT_NOT_FROZEN`" in doc and "Authorizes nothing" in doc
    train = json.loads((repo / "data/splits/p1-02-v1/train.json").read_text(encoding="utf-8"))
    assert train["manifest_hash"].startswith("sha256:96a56481") and len(train["scenarios"]) == 189
    assert len({e["family_id"] for e in train["scenarios"]}) == 64
    gate = (repo / "data/selection/p1-09-selection@1/outcome/selection_gate.json").read_text(encoding="utf-8")
    assert '"cpu_s": 7555.15625' in gate
    with pytest.raises(mt.MultiTeacherError, match="reviewed"):
        mt.collect_fixture(room(), real_factories(), scope="p1-12-matched-train@0.1.0-draft")


class _Pair(FixturePolicy):
    def _plan(self, obs):
        act = {"P1": "east", "P2": "west"}
        return ({p["id"]: act[p["id"]] for p in obs["people"] if p["status"] == "active"}, {})


def test_public_congestion_diagnostics_count_contested_targets():
    s = scenario(["E....", ".....", "....F"], {"P1": (1, 0), "P2": (1, 2)}, max_steps=3, sid="contest")
    fs = factories()
    fs["T0"] = lambda: _Pair("T0")
    r = mt.collect_fixture(s, fs)
    d0 = r["policies"]["T0"]["decisions"][0]["diagnostics"]
    assert d0["contested_targets"] == 1 and d0["max_target_contention"] == 2
    assert d0["blocked_moves"] == 1 and d0["movement_outcomes"]["blocked_conflict"] == 1
    assert r["policies"]["T0"]["outcome"]["contested_targets"] >= 1
    table = mt.outcome_table(r)
    assert table["descriptive_only"] and table["rows"]["T0"]["blocked_moves"] >= 1


def _stay(sim):
    return {pid: "stay" for pid in sim.active_ids()}


def test_fire_fork_changes_only_future_draws_and_is_bound_into_the_chain():
    """Evaluator-only fork used by P1-13 (#51). Lives here so the simulator identity is final before freeze."""
    from ember_v1.sim.simulator import SimulationError
    s = busy_room()
    a, b, c = Simulator(s), Simulator(s), Simulator(s)
    for x in (a, b, c):
        x.step(_stay(x))
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
        rb.append(b.step(_stay(b))["ignitions"])
        rc.append(c.step(_stay(c))["ignitions"])
    assert rb == rc
    with pytest.raises(ValueError):
        Simulator(s).fork_fire(-1)
    t = Simulator(scenario(["E.", ".."], {"P1": (0, 1)}, max_steps=1, sid="t"))
    t.step({"P1": "west"})
    with pytest.raises(SimulationError, match="terminated"):
        t.fork_fire(1)
