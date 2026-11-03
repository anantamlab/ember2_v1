"""Convert trusted teacher examples into Unsloth typed-decision rows.

This is SFT *data preparation*, not a teacher, simulator, verifier, or trainer.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .contract import ACTIONS, validate_actions, validate_observation
from .encoding import encode_observation

_OPTION_DESCRIPTIONS = {
    "stay": "hold position for this timestep",
    "north": "move one row toward the top",
    "south": "move one row toward the bottom",
    "east": "move one column right",
    "west": "move one column left",
}


# The option set is repeated once per person, so its text dominates the encoded
# record for large crowds (about 65k chars of questions at N=200 in "verbose").
# "compact" keeps the ID, the position and an unambiguous grid delta per option.
_COMPACT_DESCRIPTIONS = {
    "stay": "hold",
    "north": "row-1",
    "south": "row+1",
    "east": "col+1",
    "west": "col-1",
}
QUESTION_STYLES = ("verbose", "compact")


def decision_questions(observation: Mapping[str, Any], *, style: str = "verbose") -> dict[str, Any]:
    if style not in QUESTION_STYLES:
        raise ValueError(f"unsupported question style: {style}")
    obs = validate_observation(observation)
    questions = {}
    for p in obs["people"]:
        if p["status"] != "active":
            continue
        r, c = p["position"]
        if style == "verbose":
            instructions = f"Choose the safest coordinated next move for active person {p['id']} at ({r},{c})."
            criteria = dict(_OPTION_DESCRIPTIONS)
        else:
            instructions = f"Move for {p['id']} at {r},{c}"
            criteria = dict(_COMPACT_DESCRIPTIONS)
        questions[p["id"]] = {"type": "choice", "instructions": instructions, "criteria": criteria}
    return questions


def build_decision_row(
    observation: Mapping[str, Any],
    teacher_actions: Mapping[str, str],
    *, group_id: str,
    source: str,
    encoding: str = "ascii",
    local_neighbors: bool = False,
    question_style: str = "verbose",
) -> dict[str, Any]:
    """Build a `state`, `questions`, `gold` record consumed by FastDecisionModel.

    Split IDs must be assigned by scenario/topology lineage in a separate
    preprocessing step; never split individual timesteps at random.
    """
    if not isinstance(group_id, str) or not group_id:
        raise ValueError("nonempty group_id (scenario family lineage) required")
    if not isinstance(source, str) or not source:
        raise ValueError("teacher source/version required")
    obs = validate_observation(observation)
    gold = validate_actions(obs, teacher_actions)
    return {
        "state": encode_observation(obs, mode=encoding, local_neighbors=local_neighbors),
        "questions": decision_questions(obs, style=question_style),
        "gold": gold,
        "group_id": group_id,
        "source": source,
    }
