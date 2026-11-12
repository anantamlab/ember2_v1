import math

import pytest

from ember_v1.rl_env import EvacuationEnv, RewardConfig, fresh_scenario, gae, potential, state_features
from ember_v1.sim import Scenario, Simulator


def tiny(max_steps=2):
    return Scenario.from_dict({
        "schema_version": "1.0", "scenario_id": "rl-test", "grid": {"height": 3, "width": 4},
        "obstacles": [], "exits": [{"id": "E0", "position": [1, 3]}],
        "people": [{"id": "P0", "position": [1, 2]}, {"id": "P1", "position": [1, 1]}],
        "ignitions": [[0, 0]], "fire": {"model": "deterministic", "spread_interval": 10,
                                              "exits_flammable": True},
        "movement": {"enter_fire": "blocked", "conflict_resolution": "lowest_id_wins"},
        "termination": {"max_steps": max_steps}, "seed": 123})


def test_wrapper_preserves_observation_and_joint_resolution():
    scenario = tiny()
    env = EvacuationEnv(scenario)
    obs, _ = env.reset()
    sim = Simulator(scenario)
    assert obs == sim.observe()
    # A chain into the exiting person's previous position is resolved simultaneously.
    for actions in ({"P0": "east", "P1": "east"}, {"P1": "east"}):
        expected = sim.step(actions)
        obs, r, done, truncated, info = env.step(actions)
        assert info["record"] == expected
        assert obs == sim.observe()
        assert not truncated
    assert done and r == 1.0
    with pytest.raises(RuntimeError):
        env.step({})


def test_terminal_unresolved_and_stays_are_not_penalized_per_action():
    env = EvacuationEnv(tiny())
    env.reset()
    assert env.step({"P0": "stay", "P1": "stay"})[1] == 0
    _, reward, done, _, info = env.step({"P0": "east", "P1": "stay"})
    assert done and reward == (1 - 0.75)/2
    assert info["terminal_reward"] == reward


def test_shaping_discounted_telescoping_and_terminal_zero():
    gamma, eta = 0.9, 0.3
    env = EvacuationEnv(tiny(), RewardConfig(gamma=gamma, eta=eta))
    obs, _ = env.reset()
    phi0 = potential(obs, 2)
    a = env.step({"P0": "stay", "P1": "stay"})
    b = env.step({"P0": "east", "P1": "stay"})
    assert a[4]["shaping_reward"] + gamma*b[4]["shaping_reward"] == pytest.approx(-eta*phi0)
    assert b[4]["terminal_potential"] == 0
    assert potential(obs, 2, terminal=True) == 0


def test_gae_terminal_vs_bootstrap():
    adv, ret = gae([0, 1], [0.2, 0.3], [False, True], gamma=0.9, lam=1, bootstrap=999)
    assert ret == pytest.approx([0.9, 1])
    assert adv == pytest.approx([0.7, 0.7])
    assert gae([1], [0], [False], gamma=0.9, bootstrap=2)[1] == pytest.approx([2.8])


def test_fresh_splits_reproduce_and_have_independent_families():
    from ember_v1.gen.analysis import topology_hash
    scenarios = [fresh_scenario("unit-test-fresh", split, i) for split in ("train", "dev") for i in range(16)]
    hashes = [topology_hash(s.height, s.width, s.obstacles, [p for _, p in s.exits]) for s in scenarios]
    assert len(set(hashes)) == len(hashes)
    assert scenarios[0] == fresh_scenario("unit-test-fresh", "train", 0)
    for scenario in scenarios:
        obs, _ = EvacuationEnv(scenario).reset()
        features = state_features(obs)
        assert len(features) == 6*16*16+12 and all(math.isfinite(x) for x in features)
        assert "seed" not in obs and "scenario_id" not in obs


def test_parallel_cpu_transition_matches_sequential():
    from concurrent.futures import ProcessPoolExecutor
    from ember_v1.rl_run import _step
    a, b = EvacuationEnv(tiny()), EvacuationEnv(tiny())
    a.reset()
    b.reset()
    payload = (a, {"P0": "east", "P1": "east"})
    with ProcessPoolExecutor(max_workers=2) as pool:
        parallel = pool.submit(_step, payload).result()
    serial = _step((b, payload[1]))
    assert parallel[1] == serial[1]
    assert parallel[0].sim.observe() == serial[0].sim.observe()


def test_rollout_interface_and_invalid_partial_joint_action():
    env = EvacuationEnv(tiny())
    trajectory = list(env.rollout(lambda obs: {p["id"]: "east" for p in obs["people"] if p["status"] == "active"}))
    assert len(trajectory) == 2 and trajectory[-1]["done"]
    env.reset()
    with pytest.raises(ValueError):
        env.step({"P0": "east"})
