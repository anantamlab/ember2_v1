"""Shield, compact questions and shielded agent: CPU-only logic tests."""
import json

import pytest

from ember_v1.agent import DecisionAgent
from ember_v1.dataset import build_decision_row, decision_questions
from ember_v1.shield import admissible_actions, shielded_choice
from ember_v1.synthetic import make_synthetic_observation


def obs_fixture():
    # 3x3 grid: wall west of P1, fire south of P1, P2 east of P1 (occupied is kept).
    return {
        "observation_version": "1.0", "step": 0,
        "map": {"width": 3, "height": 3, "obstacles": [[0, 0]], "exits": [{"id": "E1", "position": [2, 2]}]},
        "rules": {"fire_model": "deterministic", "spread_interval": 2, "spread_probability": None,
                  "exits_flammable": False, "enter_fire": "casualty",
                  "conflict_resolution": "lowest_id_wins", "max_steps": 10},
        "people": [{"id": "P1", "position": [0, 1], "status": "active"},
                   {"id": "P2", "position": [0, 2], "status": "active"}],
        "burning": [[1, 1]], "allowed_actions": ["stay", "north", "south", "east", "west"],
    }


def test_admissible_removes_boundary_wall_fire_but_keeps_occupied():
    adm = admissible_actions(obs_fixture())
    assert adm["P1"] == ("stay", "east")      # north=boundary, south=fire, west=wall
    assert adm["P2"] == ("stay", "south", "west")  # west is occupied by P1: kept


def test_shielded_choice_is_deterministic():
    probs = dict(stay=.1, north=.5, south=.1, east=.15, west=.15)
    assert shielded_choice(probs, ("stay", "east", "west")) == "east"  # tie -> alphabetical
    with pytest.raises(ValueError):
        shielded_choice({"stay": 1.0}, ("stay",))


def test_shielded_agent_overrides_and_counts():
    def fake(state, questions):
        p = dict(stay=.1, north=.1, south=.6, east=.1, west=.1)
        return {pid: {"answer": "south", "probabilities": p} for pid in questions}

    raw = DecisionAgent(fake).act(obs_fixture())
    assert raw == {"P1": "south", "P2": "south"}
    agent = DecisionAgent(fake, shield=True)
    assert agent.act(obs_fixture()) == {"P1": "east", "P2": "south"}
    assert agent.stats == {"steps": 1, "decisions": 2, "shield_overrides": 1}


def test_shield_fails_closed_without_probabilities():
    agent = DecisionAgent(lambda s, q: {pid: {"answer": "stay"} for pid in q}, shield=True)
    with pytest.raises(ValueError, match="probabilities"):
        agent.act(obs_fixture())


def test_compact_questions_keep_ids_options_and_shrink_payload():
    obs = make_synthetic_observation(200)
    verbose = decision_questions(obs)
    compact = decision_questions(obs, style="compact")
    assert set(verbose) == set(compact)
    assert all(set(q["criteria"]) == set(verbose[pid]["criteria"]) for pid, q in compact.items())
    def text(qs):  # instruction + option-description text, without JSON syntax
        return sum(len(q["instructions"]) + sum(map(len, q["criteria"].values())) for q in qs.values())

    assert text(compact) < text(verbose) / 3
    assert len(json.dumps(compact)) < 0.6 * len(json.dumps(verbose))
    row = build_decision_row(obs, {pid: "stay" for pid in compact}, group_id="g", source="s",
                             question_style="compact")
    assert row["questions"] == compact
    with pytest.raises(ValueError):
        decision_questions(obs, style="other")
