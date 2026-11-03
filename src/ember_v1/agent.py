"""Non-generative decision head -> exact `Agent.act` action mapping.

The converter is intentionally fail-closed and cannot silently invoke a
planner/fallback after a missing or unrecognized action. The optional shield
(see `shield.py`) is a disclosed ablation arm, off by default.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .contract import ACTIONS, validate_actions, validate_observation
from .dataset import decision_questions
from .encoding import encode_observation
from .shield import admissible_actions, shielded_choice


class DecisionAgent:
    """Build fresh per episode. predictor(state, questions) -> Unsloth answers.

    The predictor must return {person_id: {'answer': 'north', 'probabilities': {...}}}.
    The model is a shared checkpoint; the agent's lifecycle is per-episode.
    `stats` counts shield overrides so raw and shielded runs can be reported apart.
    """
    def __init__(self, predictor: Callable[[str, dict], Mapping[str, Any]], *, mode: str = "ascii",
                 neighbors: bool = False, question_style: str = "verbose", shield: bool = False):
        self.predictor = predictor
        self.mode = mode
        self.neighbors = neighbors
        self.question_style = question_style
        self.shield = shield
        self.stats = {"steps": 0, "decisions": 0, "shield_overrides": 0}

    def act(self, observation: Any) -> dict[str, str]:
        raw = observation.to_dict() if hasattr(observation, "to_dict") else observation
        obs = validate_observation(raw)
        questions = decision_questions(obs, style=self.question_style)
        if not questions:
            return {}
        state = encode_observation(obs, mode=self.mode, local_neighbors=self.neighbors)
        predicted = self.predictor(state, questions)
        if not isinstance(predicted, Mapping) or set(predicted) != set(questions):
            raise ValueError("decision model must return exactly one prediction for every active person")
        admissible = admissible_actions(obs) if self.shield else {}
        actions = {}
        for pid in sorted(questions):
            result = predicted[pid]
            if not isinstance(result, Mapping) or not isinstance(result.get("answer"), str):
                raise ValueError(f"missing decision answer for {pid}")
            answer = result["answer"]
            if self.shield:
                probs = result.get("probabilities")
                if not isinstance(probs, Mapping) or set(probs) != set(ACTIONS):
                    raise ValueError(f"shield needs all five action probabilities for {pid}")
                shielded = shielded_choice(probs, admissible[pid])
                self.stats["shield_overrides"] += shielded != answer
                answer = shielded
            actions[pid] = answer
        self.stats["steps"] += 1
        self.stats["decisions"] += len(actions)
        return validate_actions(obs, actions)


def unsloth_predictor(model: Any, tokenizer: Any) -> Callable[[str, dict], Mapping[str, Any]]:
    """Bind the existing Unsloth decision API; no bespoke neural network."""
    from unsloth import FastDecisionModel  # requires a configured compatible GPU environment

    FastDecisionModel.for_inference(model)

    def predict(state: str, questions: dict) -> Mapping[str, Any]:
        return FastDecisionModel.predict(model, tokenizer, state, questions)

    return predict


def modal_predictor(app_name: str = "ember-v1", cls_name: str = "DecisionPolicy",
                    **params: Any) -> Callable[[str, dict], Mapping[str, Any]]:
    """Call the deployed Modal GPU policy (see `infra/modal_app.py`).

    The simulator and this agent run on the CPU side; only inference is remote.
    Network errors raise: there is no silent fallback action.
    """
    import modal

    policy = modal.Cls.from_name(app_name, cls_name)(**params)

    def predict(state: str, questions: dict) -> Mapping[str, Any]:
        return policy.decide.remote({"state": state, "questions": questions})["answers"]

    return predict
