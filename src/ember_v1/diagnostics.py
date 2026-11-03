"""Non-mutating diagnostics for raw per-person actions (not a movement solver)."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from .contract import validate_actions, validate_observation

DELTAS = {"stay": (0, 0), "north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1)}


def raw_action_diagnostics(observation: Mapping[str, Any], actions: Mapping[str, str]) -> dict[str, int]:
    obs = validate_observation(observation)
    act = validate_actions(obs, actions)
    positions = {p["id"]: tuple(p["position"]) for p in obs["people"] if p["status"] == "active"}
    targets = {pid: (positions[pid][0] + DELTAS[a][0], positions[pid][1] + DELTAS[a][1]) for pid, a in act.items() if a != "stay"}
    # Only intent to the same destination; movement resolver additionally handles
    # ID-priority, chains, cycles, boundary/wall/fire rules.
    counts = Counter(targets.values())
    conflicts = sum(n - 1 for n in counts.values() if n > 1)
    swaps = sum(targets.get(a) == positions[b] and targets.get(b) == positions[a]
                for a in positions for b in positions if a < b)
    occupied = set(positions.values())
    w, h = obs["map"]["width"], obs["map"]["height"]
    walls = {tuple(x) for x in obs["map"]["obstacles"]}
    burning = {tuple(x) for x in obs["burning"]}
    return {
        "active": len(positions),
        "moving_intents": len(targets),
        "contested_destination_extra_intents": conflicts,
        "direct_swaps": swaps,
        "boundary_intents": sum(not (0 <= r < h and 0 <= c < w) for r, c in targets.values()),
        "wall_intents": sum(x in walls for x in targets.values()),
        "fire_intents": sum(x in burning for x in targets.values()),
        "occupied_destination_intents": sum(x in occupied for x in targets.values()),
    }
