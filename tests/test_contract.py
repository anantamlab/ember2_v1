import copy
import json
from pathlib import Path

import pytest

from ember_v1.contract import validate_actions, validate_observation
from ember_v1.dataset import build_decision_row
from ember_v1.diagnostics import raw_action_diagnostics
from ember_v1.encoding import encode_observation

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "examples" / "observation.json").read_text())


def test_model_visible_input_canonical_reorders_people_and_walls():
    a = validate_observation(FIXTURE)
    shuffled = copy.deepcopy(FIXTURE)
    shuffled["people"] = list(reversed(shuffled["people"]))
    shuffled["map"]["obstacles"] = list(reversed(shuffled["map"]["obstacles"]))
    assert validate_observation(shuffled) == a
    assert encode_observation(shuffled) == encode_observation(FIXTURE)
    assert "P10" in encode_observation(FIXTURE)
    assert "P2" in encode_observation(FIXTURE)


@pytest.mark.parametrize("bad", ["seed", "rng_state", "scenario_id", "trapped", "future_fire", "manifest_hash"])
def test_privileged_field_fails_closed(bad):
    obs = copy.deepcopy(FIXTURE)
    obs[bad] = "DO NOT USE"
    with pytest.raises(ValueError):
        validate_observation(obs)


def test_private_field_within_person_fails_closed():
    obs = copy.deepcopy(FIXTURE)
    obs["people"][0]["rng_state"] = 9
    with pytest.raises(ValueError):
        validate_observation(obs)


def test_inactive_person_does_not_get_question():
    row = build_decision_row(FIXTURE, {"P10": "north", "P2": "east"}, group_id="family-A", source="teacher-v1")
    assert set(row["gold"]) == {"P10", "P2"}
    assert set(row["questions"]) == {"P10", "P2"}
    assert set(row["questions"]["P2"]["criteria"]) == {"stay", "north", "south", "east", "west"}
    assert row["group_id"] == "family-A"


def test_wrong_action_set_must_not_be_repaired():
    with pytest.raises(ValueError):
        validate_actions(FIXTURE, {"P2": "north"})
    with pytest.raises(ValueError):
        validate_actions(FIXTURE, {"P2": "north", "P10": "west", "P3": "stay"})
    with pytest.raises(ValueError):
        validate_actions(FIXTURE, {"P2": "NORTH", "P10": "stay"})


def test_ascii_grid_can_represent_people_and_fire_and_rules():
    text = encode_observation(FIXTURE, mode="ascii", local_neighbors=True)
    assert "GRID" in text
    assert "PERSON P10 @ 2,3" in text
    assert "P2 @ 2,4" in text
    assert "ADJ=" in text
    assert "spread_probability" in text
    assert "F" in text
    assert "rng_state" not in text


def test_coords_and_ascii_have_identical_decision_id_set():
    a = build_decision_row(FIXTURE, {"P10": "north", "P2": "east"}, group_id="g", source="t", encoding="coords")
    b = build_decision_row(FIXTURE, {"P10": "north", "P2": "east"}, group_id="g", source="t", encoding="ascii")
    assert a["gold"] == b["gold"]
    assert a["questions"] == b["questions"]
    assert a["state"] != b["state"]


def test_diags_are_not_movement_solver():
    obs = copy.deepcopy(FIXTURE)
    # Two people both move to (2,3): conflict intent with lower ID winner.
    obs["people"][0]["position"] = [2, 2]  # P10
    obs["people"][1]["position"] = [2, 4]  # P2
    d = raw_action_diagnostics(obs, {"P10": "east", "P2": "west"})
    assert d["contested_destination_extra_intents"] == 1
    assert d["moving_intents"] == 2


def test_following_chain_not_mistaken_for_same_cell_conflict():
    obs = copy.deepcopy(FIXTURE)
    obs["people"][0]["position"] = [2, 2]
    obs["people"][1]["position"] = [2, 3]
    d = raw_action_diagnostics(obs, {"P10": "east", "P2": "east"})
    assert d["contested_destination_extra_intents"] == 0
    assert d["occupied_destination_intents"] == 1


def test_bad_coordinates_rejected():
    obs = copy.deepcopy(FIXTURE)
    obs["people"][0]["position"] = [-1, 100]
    with pytest.raises(ValueError):
        validate_observation(obs)


def test_decision_agent_with_fake_head():
    from ember_v1.agent import DecisionAgent

    def fake(state, questions):
        assert "FIRE" not in state or "GRID" in state
        return {pid: {"answer": "stay", "probabilities": {"stay": 1.0}} for pid in questions}

    result = DecisionAgent(fake).act(FIXTURE)
    assert result == {"P10": "stay", "P2": "stay"}


def test_decision_agent_rejects_partial_prediction():
    from ember_v1.agent import DecisionAgent
    with pytest.raises(ValueError):
        DecisionAgent(lambda state, q: {"P2": {"answer": "stay"}}).act(FIXTURE)


@pytest.mark.parametrize("n", [1, 10, 50, 100, 200])
def test_synthetic_shapes(n):
    from ember_v1.synthetic import make_synthetic_observation
    obs = make_synthetic_observation(n)
    normalized = validate_observation(obs)
    assert len(normalized["people"]) == n
    assert len(build_decision_row(obs, {f"P{i}": "stay" for i in range(n)}, group_id="synthetic", source="fixture")["questions"]) == n
