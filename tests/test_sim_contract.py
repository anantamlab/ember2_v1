"""Observation boundary, deterministic replay, invalid inputs and independence of `ember_v1.sim`."""
from __future__ import annotations

import ast
import copy
import json
import random
import sys
from pathlib import Path

import pytest

from ember_v1.contract import validate_observation
from ember_v1.sim import (InvalidActionError, Scenario, SimulationError, Simulator, first_divergence,
                          replay, run_episode, verify_replay)
from ember_v1.sim.canonical import canonical_json
from sim_helpers import scenario
from test_sim_properties import random_policy, random_scenario

SIM_DIR = Path(__file__).resolve().parents[1] / "src" / "ember_v1" / "sim"
ROOT_KEYS = {"observation_version", "step", "map", "rules", "people", "burning", "allowed_actions"}
SECRET_SEED = 9876543210123456789


def _stochastic(seed=SECRET_SEED, sid="secret-scenario-id", description="hidden note"):
    return scenario(["F.....", "......", ".#..E.", "......"], {"P2": (3, 0), "P10": (1, 5), "P1": (3, 5)},
                    fire="stochastic", p=0.4, interval=1, seed=seed, sid=sid, description=description)


# ----- observation boundary --------------------------------------------------------------

def test_observation_is_whitelisted_and_already_canonical():
    sim = Simulator(_stochastic())
    policy = random_policy(3)
    while True:
        obs = sim.observe()
        assert set(obs) == ROOT_KEYS
        assert validate_observation(obs) == obs          # ember_v1 whitelist accepts it unchanged
        if sim.terminated:
            break
        sim.step(policy(obs))


def test_observation_holds_no_private_data():
    sim = Simulator(_stochastic())
    sim.step({"P1": "stay", "P10": "west", "P2": "north"})
    text = canonical_json(sim.observe())
    private = sim.private_state()
    for needle in (str(SECRET_SEED), "secret-scenario-id", "hidden note", "trapped", "rng", "seed",
                   "scenario", "hash", str(private["rng"]["fire"])):
        assert needle not in text


def test_observation_does_not_depend_on_seed_id_or_description():
    a = Simulator(_stochastic(seed=1, sid="a", description="x")).observe()
    b = Simulator(_stochastic(seed=2, sid="b", description=None)).observe()
    assert a == b
    det = lambda s: scenario(["F...E", "....."], {"P1": (1, 0)}, seed=s, sid=f"s{s}")  # noqa: E731
    runs = []
    for seed in (5, 6):
        sim = Simulator(det(seed))
        obs = [sim.observe()]
        while not sim.terminated:
            sim.step({p: "east" for p in sim.active_ids()})
            obs.append(sim.observe())
        runs.append(obs)
    assert runs[0] == runs[1]


def test_rules_view_copies_scenario_config():
    sc = _stochastic()
    rules = Simulator(sc).observe()["rules"]
    assert rules == {"fire_model": "stochastic", "spread_interval": 1, "spread_probability": 0.4,
                     "exits_flammable": True, "enter_fire": "casualty",
                     "conflict_resolution": "lowest_id_wins", "max_steps": 50}
    assert Simulator(scenario(["E."], {"P1": (0, 1)})).observe()["rules"]["spread_probability"] is None


def test_mutating_an_observation_cannot_change_the_simulator():
    sim = Simulator(_stochastic())
    before = sim.private_state()
    obs = sim.observe()
    obs["people"][0]["position"] = [0, 0]
    obs["burning"].clear()
    obs["map"]["obstacles"].append([0, 1])
    assert sim.private_state() == before and sim.observe() != obs


def test_policy_receives_only_the_observation():
    seen = []

    def policy(obs):
        seen.append(obs)
        return {p["id"]: "stay" for p in obs["people"] if p["status"] == "active"}

    run_episode(_stochastic(), policy)
    assert seen and all(type(o) is dict and set(o) == ROOT_KEYS for o in seen)


# ----- deterministic replay and hashing --------------------------------------------------------

def test_same_seed_same_records_and_chain():
    rnd = random.Random(8)
    for i in range(40):
        sc = random_scenario(rnd)
        a = run_episode(sc, random_policy(i))
        b = run_episode(sc, random_policy(i))
        assert a == b
        assert replay(sc, None, [r["actions"] for r in a]) == a
        assert first_divergence(sc, None, a) is None


def test_records_are_json_canonical_and_hash_chained():
    recs = run_episode(_stochastic(), random_policy(1))
    hashes = [r["transition_hash"] for r in recs]
    assert len(set(hashes)) == len(hashes) and all(h.startswith("sha256:") for h in hashes)
    for r in recs:
        assert json.loads(canonical_json(r)) == r


def test_seed_changes_stochastic_fire_and_hashes():
    sc = _stochastic(seed=1)
    stay = lambda o: {p["id"]: "stay" for p in o["people"] if p["status"] == "active"}  # noqa: E731
    chains = {run_episode(sc, stay, seed=s)[-1]["transition_hash"] for s in range(8)}
    fires = {json.dumps([r["ignitions"] for r in run_episode(sc, stay, seed=s)]) for s in range(8)}
    assert len(chains) == 8 and len(fires) > 1


def test_first_divergence_reports_tampered_step():
    sc = _stochastic()
    recs = run_episode(sc, random_policy(4))
    assert len(recs) >= 3
    bad = copy.deepcopy(recs)
    bad[2]["ignitions"] = bad[2]["ignitions"] + [[3, 3]]
    assert first_divergence(sc, None, bad) == 3
    assert first_divergence(sc, SECRET_SEED + 1, recs) is not None   # wrong seed diverges
    wrong_actions = copy.deepcopy(recs)
    wrong_actions[1]["actions"] = {}
    assert first_divergence(sc, None, wrong_actions) == 2


# ----- full-episode replay verification -----------------------------------------------------------

def _episode():
    sc = _stochastic()
    recs = run_episode(sc, random_policy(4))
    assert len(recs) >= 3 and recs[-1]["termination_reason"] is not None
    assert all(r["termination_reason"] is None for r in recs[:-1])
    return sc, recs


def test_verify_replay_accepts_complete_episode():
    sc, recs = _episode()
    v = verify_replay(sc, None, recs)
    assert v.status == "complete" and v.complete and v.steps_verified == len(recs)
    assert v.step is None and v.reason is None
    assert v.termination_reason == recs[-1]["termination_reason"]
    assert v.final_transition_hash == recs[-1]["transition_hash"]


def test_verify_replay_rejects_missing_final_step_that_prefix_check_accepts():
    sc, recs = _episode()
    v = verify_replay(sc, None, recs[:-1])
    assert (v.status, v.reason, v.step, v.steps_verified) == ("diverged", "incomplete", len(recs), len(recs) - 1)
    assert not v.complete
    assert first_divergence(sc, None, recs[:-1]) is None        # prefix check alone cannot prove completeness
    empty = verify_replay(sc, None, [])
    assert (empty.status, empty.reason, empty.step) == ("diverged", "incomplete", 1)


def test_verify_replay_rejects_step_after_termination():
    sc, recs = _episode()
    extra = recs + [dict(copy.deepcopy(recs[-1]), step=len(recs) + 1)]
    v = verify_replay(sc, None, extra)
    assert (v.status, v.reason, v.step, v.steps_verified) == ("diverged", "step_after_termination", len(recs) + 1, len(recs))
    assert verify_replay(sc, None, extra, partial=True).status == "diverged"
    assert first_divergence(sc, None, extra) == len(recs) + 1


@pytest.mark.parametrize("field,value", [("movements", {}), ("ignitions", [[0, 0], [3, 3]]),
                                         ("evacuations", {"P1": "E1"}), ("state_hash", "sha256:" + "0" * 64)])
def test_verify_replay_rejects_modified_intermediate_step(field, value):
    sc, recs = _episode()
    bad = copy.deepcopy(recs)
    bad[1][field] = value
    v = verify_replay(sc, None, bad)
    assert (v.status, v.reason, v.step, v.steps_verified) == ("diverged", "record_mismatch", 2, 1)


def test_verify_replay_rejects_wrong_termination_reason():
    sc, recs = _episode()
    bad = copy.deepcopy(recs)
    real = bad[-1]["termination_reason"]
    bad[-1]["termination_reason"] = "max_steps" if real == "all_resolved" else "all_resolved"
    v = verify_replay(sc, None, bad)
    assert (v.status, v.reason, v.step) == ("diverged", "record_mismatch", len(recs))
    early = copy.deepcopy(recs)
    early[0]["termination_reason"] = "all_resolved"              # claims termination too early
    assert verify_replay(sc, None, early).step == 1


def test_verify_replay_rejects_incorrect_transition_hash():
    sc, recs = _episode()
    bad = copy.deepcopy(recs)
    bad[1]["transition_hash"] = "sha256:" + "f" * 64
    assert (verify_replay(sc, None, bad).reason, verify_replay(sc, None, bad).step) == ("record_mismatch", 2)
    last = copy.deepcopy(recs)
    last[-1]["transition_hash"] = recs[-2]["transition_hash"]
    assert verify_replay(sc, None, last).step == len(recs)


def test_verify_replay_is_type_strict_and_rejects_malformed_records():
    sc, recs = _episode()
    bad = copy.deepcopy(recs)
    bad[0]["fire_draws"] = float(bad[0]["fire_draws"])           # equal under ==, different JSON
    assert (verify_replay(sc, None, bad).reason, verify_replay(sc, None, bad).step) == ("record_mismatch", 1)
    for broken in ("not a record", {"step": 1}):
        v = verify_replay(sc, None, [broken] + recs[1:])
        assert (v.reason, v.step) == ("malformed_record", 1)
    wrong = copy.deepcopy(recs)
    wrong[1]["actions"] = {}
    assert (verify_replay(sc, None, wrong).reason, verify_replay(sc, None, wrong).step) == ("invalid_actions", 2)


def test_verify_replay_partial_trace_is_explicitly_partial():
    sc, recs = _episode()
    v = verify_replay(sc, None, recs[:2], partial=True)
    assert (v.status, v.steps_verified, v.step, v.reason) == ("partial", 2, None, None)
    assert not v.complete and v.final_transition_hash == recs[1]["transition_hash"]
    assert verify_replay(sc, None, recs[:2]).status == "diverged"            # not accepted without the flag
    assert verify_replay(sc, None, [], partial=True).status == "partial"
    assert verify_replay(sc, None, recs, partial=True).status == "complete"  # a full trace is still complete
    bad = copy.deepcopy(recs[:2])
    bad[1]["ignitions"] = [[3, 3]]
    assert verify_replay(sc, None, bad, partial=True).status == "diverged"


# ----- invalid inputs ---------------------------------------------------------------------------

@pytest.mark.parametrize("actions", [
    {"P1": "stay", "P10": "stay"},                              # missing P2
    {"P1": "stay", "P10": "stay", "P2": "stay", "P3": "stay"},  # unknown id
    {"P1": "stay", "P10": "stay", "P2": "jump"},                # unsupported value
    {"P1": "stay", "P10": "stay", "P2": None},                  # non-string
    {"P1": "stay", "P10": "stay", "P2": "STAY"},                # no case repair
    [("P1", "stay")],                                           # not a mapping
])
def test_invalid_actions_raise_and_leave_state_unchanged(actions):
    sim = Simulator(_stochastic())
    state, chain = sim.private_state(), sim.chain_hash
    with pytest.raises(InvalidActionError):
        sim.step(actions)
    assert sim.private_state() == state and sim.chain_hash == chain


def test_actions_for_resolved_people_are_rejected():
    sim = Simulator(scenario([".E", ".."], {"P1": (0, 0), "P2": (1, 0)}))
    sim.step({"P1": "east", "P2": "stay"})
    with pytest.raises(InvalidActionError):
        sim.step({"P1": "stay", "P2": "stay"})
    sim.step({"P2": "north"})


def test_terminated_simulator_rejects_steps():
    sim = Simulator(scenario(["E."], {"P1": (0, 1)}, max_steps=1))
    sim.step({"P1": "stay"})
    with pytest.raises(SimulationError):
        sim.step({"P1": "stay"})


def _base():
    return json.loads(_stochastic().to_json())


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(extra=1),
    lambda d: d.pop("seed"),
    lambda d: d.update(schema_version="2.0"),
    lambda d: d.update(scenario_id="bad id"),
    lambda d: d["grid"].update(width=0),
    lambda d: d["grid"].update(width=1025),
    lambda d: d["grid"].update(height=True),
    lambda d: d["grid"].update(height=4.0),
    lambda d: d["obstacles"].append([9, 9]),
    lambda d: d["obstacles"].append(d["obstacles"][0]),
    lambda d: d.update(exits=[]),
    lambda d: d["exits"].append(dict(d["exits"][0])),
    lambda d: d["exits"].append({"id": "E9", "position": d["obstacles"][0]}),
    lambda d: d.update(people=[]),
    lambda d: d["people"].append({"id": "P1", "position": [0, 3]}),
    lambda d: d["people"].append({"id": "P5", "position": d["exits"][0]["position"]}),
    lambda d: d["people"].append({"id": "P5", "position": d["ignitions"][0]}),
    lambda d: d["people"].append({"id": "P5", "position": d["obstacles"][0]}),
    lambda d: d["people"].append({"id": "P 5", "position": [0, 3]}),
    lambda d: d["ignitions"].append(d["obstacles"][0]),
    lambda d: (d["fire"].update(exits_flammable=False), d["ignitions"].append(d["exits"][0]["position"])),
    lambda d: d["fire"].update(model="wind"),
    lambda d: d["fire"].pop("spread_probability"),
    lambda d: d["fire"].update(spread_probability=1.5),
    lambda d: d["fire"].update(spread_probability=float("nan")),
    lambda d: d["fire"].update(spread_probability=True),
    lambda d: d["fire"].update(spread_interval=0),
    lambda d: d["fire"].update(exits_flammable=1),
    lambda d: d.update(fire={"model": "deterministic", "spread_interval": 1, "exits_flammable": True,
                             "spread_probability": 0.5}),
    lambda d: d["movement"].update(conflict_resolution="random"),
    lambda d: d["movement"].update(enter_fire="maybe"),
    lambda d: d["termination"].update(max_steps=0),
    lambda d: d["termination"].update(max_steps=100001),
    lambda d: d.update(seed=-1),
    lambda d: d.update(seed=2 ** 64),
    lambda d: d.update(description=5),
])
def test_scenario_rejects_invalid_input(mutate):
    d = _base()
    mutate(d)
    with pytest.raises(ValueError):
        Scenario.from_dict(d)


def test_scenario_canonical_form_ignores_input_order():
    d = _base()
    d2 = copy.deepcopy(d)
    for key in ("obstacles", "exits", "people", "ignitions"):
        d2[key] = list(reversed(d2[key]))
    a, b = Scenario.from_dict(d), Scenario.from_dict(d2)
    assert a == b and a.content_hash() == b.content_hash()
    assert Scenario.from_dict(json.loads(a.to_json())) == a


def test_simulator_rejects_non_scenario():
    with pytest.raises(TypeError):
        Simulator(_base())


# ----- independence -------------------------------------------------------------------------------

def test_sim_package_imports_only_stdlib_and_itself():
    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    for path in SIM_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:          # relative import inside ember_v1.sim
                    assert node.level == 1, f"{path.name}: parent-relative import"
                    continue
                names = [node.module.split(".")[0]]
            else:
                continue
            for name in names:
                assert name in stdlib, f"{path.name} imports {name}"


def test_sim_source_names_no_model_modal_or_benchmark_package():
    for path in SIM_DIR.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for banned in ("import modal", "unsloth", "torch", "transformers", "import emberbench",
                       "from emberbench"):
            assert banned not in text, f"{path.name}: {banned}"
