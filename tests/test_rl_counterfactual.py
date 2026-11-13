import copy

import pytest

from ember_v1.rl_counterfactual import snapshot, restore, compare, verify_study, curriculum_stage
from ember_v1.rl_env import fresh_scenario
from ember_v1.sim import Simulator


def test_private_snapshot_restores_exact_next_transition_and_does_not_alias():
    sim = Simulator(fresh_scenario("cf-tests", "train", 0, 10))
    sim.step({p: "stay" for p in sim.active_ids()})
    saved = snapshot(sim)
    a = {p: "east" for p in sim.active_ids()}
    restored = restore(saved)
    assert restored.observe() == sim.observe()
    assert restored.private_state() == sim.private_state()
    assert restored.chain_hash == sim.chain_hash
    assert restored.step(a) == sim.step(a)
    assert saved == snapshot(restore(saved))


def test_matched_branches_are_public_only_and_replay_exactly():
    sim = Simulator(fresh_scenario("cf-tests", "train", 1, 8))
    calls = []
    def continuation(obs):
        assert set(obs) == set(sim.observe())
        assert not {"rng", "seed", "trapped", "scenario_id", "chain"} & set(obs)
        calls.append(copy.deepcopy(obs))
        return {p["id"]: "stay" for p in obs["people"] if p["status"] == "active"}
    actions = {p: "stay" for p in sim.active_ids()}
    alternative = {p: "east" for p in sim.active_ids()}
    study = compare(snapshot(sim), {"student": actions, "variation": alternative},
                    continuation=continuation, continuation_identity={"name": "test-stay"},
                    fork_seeds=[None, 7, 11, 13], source={"split": "train", "policy": "fixture"})
    assert calls and study["matched_future_fire"]
    assert verify_study(study)["status"] == "PASS"
    corrupted = copy.deepcopy(study)
    corrupted["branches"]["variation"][0]["records"][0]["fire_draws"] += 1
    with pytest.raises(ValueError):
        verify_study(corrupted)
    with pytest.raises(ValueError, match="TRAIN"):
        compare(snapshot(sim), {"student": actions, "variation": alternative},
                continuation=continuation, continuation_identity={}, fork_seeds=[None], source={"split": "dev"})


def test_curriculum_progress_depends_only_on_train_performance():
    rows = [{"evacuated": 7, "casualties": 1, "people": 10} for _ in range(16)]
    assert curriculum_stage(0, rows, 16) == 1
    assert curriculum_stage(0, rows[:8], 8) == 0
    assert curriculum_stage(2, [{"evacuated": 0, "casualties": 10, "people": 10}]*64, 64) == 3


def test_curriculum_scenarios_cover_stages_and_have_disjoint_topologies():
    from ember_v1.rl_cf_scenarios import scenarios
    from ember_v1.gen.analysis import topology_hash
    hashes = []
    for split, n in (("train", 256), ("select", 32), ("report", 32)):
        rows = scenarios("ember-r22-counterfactual-20261113-v1", split, n)
        assert {m["stage"] for _, m in rows} == set(range(5))
        assert max(len(s.people) for s, _ in rows) == 20
        for s, _ in rows:
            hashes.append(topology_hash(s.height, s.width, s.obstacles, [p for _, p in s.exits]))
    assert len(set(hashes)) == len(hashes)


def test_unfinished_phase2_cannot_be_used_as_counterfactual_parent(tmp_path):
    from ember_v1.rl_cf_training import verify_parent
    with pytest.raises(ValueError, match="not complete"):
        verify_parent(tmp_path)
