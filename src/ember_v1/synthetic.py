"""Synthetic observation CONTRACT fixtures, NOT training scenarios.

These do not come from EMBERBench fixed or hidden evaluation suites. They are
not intended for performance scoring or safety claims: merely token/shape tests.
"""
from __future__ import annotations

from .contract import ACTIONS


def make_synthetic_observation(n: int, side: int = 24) -> dict:
    if type(n) is not int or n < 0 or side < 3 or n > side * side - 4:
        raise ValueError("invalid synthetic crowd capacity")
    excluded = {(0, 0), (0, side - 1), (side - 1, side - 1), (1, 1)}
    cells = [(r, c) for r in range(side) for c in range(side) if (r, c) not in excluded]
    return {
        "observation_version": "1.0", "step": 8,
        "map": {"width": side, "height": side,
                "obstacles": [[1, 1]],
                "exits": [{"id": "E1", "position": [0, side - 1]}, {"id": "E2", "position": [side - 1, side - 1]}]},
        "rules": {"fire_model": "stochastic", "spread_interval": 3,
                  "spread_probability": 0.2, "exits_flammable": True,
                  "enter_fire": "blocked", "conflict_resolution": "lowest_id_wins", "max_steps": 200},
        "people": [{"id": f"P{i}", "position": list(cells[i]), "status": "active"} for i in range(n)],
        "burning": [[0, 0]], "allowed_actions": list(ACTIONS),
    }
