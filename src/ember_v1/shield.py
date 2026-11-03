"""Disclosed, deterministic single-person action shield (an ablation arm).

The shield removes only options that, for that person alone, cannot be better
than "stay": a move off the grid, into a wall, or into a burning cell. Under
`enter_fire="blocked"` a fire move is a wasted step; under "casualty" it is fatal.
It never removes a move into an occupied cell (following chains are legal) and
never plans a route. Always report raw and shielded results separately.

ASSUMPTION to verify against EMBERBench `movement.py` before use: off-grid and
wall moves resolve as a blocked move (the person stays), not as an error.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .contract import ACTIONS, validate_observation
from .diagnostics import DELTAS


def admissible_actions(observation: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """Return the admissible options per active person ID ("stay" is always kept)."""
    obs = validate_observation(observation)
    m = obs["map"]
    walls = {tuple(x) for x in m["obstacles"]}
    burning = {tuple(x) for x in obs["burning"]}
    result = {}
    for p in obs["people"]:
        if p["status"] != "active":
            continue
        r, c = p["position"]
        keep = []
        for action in ACTIONS:
            dr, dc = DELTAS[action]
            cell = (r + dr, c + dc)
            if action != "stay" and (
                not (0 <= cell[0] < m["height"] and 0 <= cell[1] < m["width"])
                or cell in walls or cell in burning
            ):
                continue
            keep.append(action)
        result[p["id"]] = tuple(keep)
    return result


def shielded_choice(probabilities: Mapping[str, float], admissible: tuple[str, ...]) -> str:
    """Highest-probability admissible option; ties break alphabetically (deterministic)."""
    if set(probabilities) != set(ACTIONS):
        raise ValueError("shield needs probabilities for all five actions")
    if "stay" not in admissible:
        raise ValueError("stay must stay admissible")
    return min(admissible, key=lambda a: (-probabilities[a], a))
