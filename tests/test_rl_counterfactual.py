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
    from ember_v1.sim.canonical import digest
    corrupted["study_hash"] = digest({k: v for k, v in corrupted.items() if k != "study_hash"})
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
    sets = [[r for stage in range(5) for r in scenarios("ember-r22-counterfactual-20261113-v1", "train", 256, stage=stage)]]
    sets += [scenarios("ember-r22-counterfactual-20261113-v1", split, 32) for split in ("select", "report")]
    for rows in sets:
        assert {m["stage"] for _, m in rows} == set(range(5))
        assert max(len(s.people) for s, _ in rows) == 20
        for s, _ in rows:
            hashes.append(topology_hash(s.height, s.width, s.obstacles, [p for _, p in s.exits]))
    assert len(set(hashes)) == len(hashes)


def test_unfinished_phase2_cannot_be_used_as_counterfactual_parent(tmp_path):
    from ember_v1.rl_cf_training import verify_parent
    with pytest.raises(ValueError, match="not complete"):
        verify_parent(tmp_path)


def test_both_arms_use_identical_train_controlled_curriculum():
    from ember_v1.rl_cf_training import advance_shared_curriculum
    bad = [{"people": 10, "evacuated": 1, "casualties": 5}]*16
    good = [{"people": 10, "evacuated": 8, "casualties": 1}]*16
    states = {"ppo": {"stage": 0, "episodes_in_stage": 16, "episodes": bad},
              "cf": {"stage": 0, "episodes_in_stage": 16, "episodes": good}}
    advance_shared_curriculum(states)
    assert states["cf"]["stage"] == states["ppo"]["stage"] == 0
    states["ppo"]["episodes"] = good
    advance_shared_curriculum(states)
    assert states["cf"]["stage"] == states["ppo"]["stage"] == 1
    assert states["cf"]["episodes_in_stage"] == states["ppo"]["episodes_in_stage"] == 0


def test_full_teacher_candidate_job_on_hand_built_fire_chain():
    import math
    from ember_v1.sim import Scenario
    from ember_v1.rl_counterfactual import evaluate_job, fixed_t1
    from ember_v1.rl_cf_training import CounterfactualObjective, jobs_from_visits
    scenario = Scenario.from_dict({"schema_version": "1.0", "scenario_id": "cf-hand-fixture",
        "grid": {"height": 3, "width": 6}, "obstacles": [],
        "exits": [{"id": "E0", "position": [1, 5]}],
        "people": [{"id": "P0", "position": [1, 4]}, {"id": "P1", "position": [1, 3]}],
        "ignitions": [[1, 2]], "fire": {"model": "deterministic", "spread_interval": 1, "exits_flammable": True},
        "movement": {"conflict_resolution": "lowest_id_wins", "enter_fire": "blocked"},
        "termination": {"max_steps": 8}, "seed": 17})
    sim = Simulator(scenario)
    obs = sim.observe()
    actions = {p: "stay" for p in sim.active_ids()}
    visit = {"label": "train-0", "scenario_id": scenario.scenario_id, "observation": obs,
             "sampled_actions": actions, "executed_actions": actions, "record": sim.step(actions),
             "person_logprobs": [[math.log(.2)]*5 for _ in actions]}
    cfg = {"seed": 17, "cf_states_per_update": 1, "cf_fire_replicates": 8,
           "teacher_candidate_limits": {"T2": {"cpu_budget_ms": 500., "max_rollouts": 24},
                                        "T3": {"cpu_budget_ms": 200., "max_ant_steps": 20000}}}
    job = jobs_from_visits([visit], {scenario.scenario_id: scenario}, cfg, 0, "synthetic-fixture-not-a-model")[0]
    study = evaluate_job(job)
    assert set(study["teacher_diagnostics"]) == {"T0", "T1", "T2", "T3", "T4"}
    assert all(v["status"] != "failed" for v in study["teacher_diagnostics"].values())
    assert verify_study(study, continuation=fixed_t1)["status"] == "PASS"
    assert CounterfactualObjective([study]).pairs
