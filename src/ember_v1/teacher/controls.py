"""Simple baseline controls for the T0 diagnostic comparison (issue #7).

They are controls, not teachers. Each uses only the public observation.
  C-stay    every active person stays.
  C-random  uniform choice from `allowed_actions` (all five), seeded. The RNG
            is a function of (policy seed, observation step) only, so equal
            observations give equal actions.
  C-greedy  myopic nearest-exit greedy: target = nearest safe (not burning)
            exit by Manhattan distance (ties: lowest exit ID); take the first
            Manhattan-reducing move (larger axis gap first, then the vertical
            one) whose cell is in the grid, not a wall and not burning now;
            otherwise stay. No path search: it can stall behind walls.
"""
from __future__ import annotations

import random
from typing import Any

from .grid import PublicGrid
from .interface import ReferencePolicy


class StayControl(ReferencePolicy):
    name = "C-stay"
    policy = "stay"
    version = "1.0.0"

    def _plan(self, obs: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
        ids = [p["id"] for p in obs["people"] if p["status"] == "active"]
        return {pid: "stay" for pid in ids}, {"active": len(ids)}


class RandomControl(ReferencePolicy):
    name = "C-random"
    policy = "uniform_random_allowed_action"
    version = "1.0.0"

    def __init__(self, seed: int) -> None:
        if seed is None:
            raise ValueError("RandomControl needs an explicit seed")
        super().__init__(None, seed=seed)

    def _plan(self, obs: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
        rng = random.Random(f"{self.name}/{self.version}/{self.seed}/{obs['step']}")
        allowed = list(obs["allowed_actions"])
        ids = sorted(p["id"] for p in obs["people"] if p["status"] == "active")
        return {pid: rng.choice(allowed) for pid in ids}, {"active": len(ids)}


_DELTA = {"north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1)}


class GreedyNearestExitControl(ReferencePolicy):
    name = "C-greedy"
    policy = "myopic_nearest_exit_greedy"
    version = "1.0.0"

    def _plan(self, obs: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
        grid = PublicGrid.from_observation(obs)
        exits = grid.safe_exits()
        actions: dict[str, str] = {}
        stalled = 0
        for pid, (r, c) in grid.active:
            if not exits:
                actions[pid] = "stay"
                stalled += 1
                continue
            _, _, (er, ec) = min((abs(er - r) + abs(ec - c), eid, (er, ec)) for eid, (er, ec) in exits)
            vert = "south" if er > r else "north" if er < r else None
            horiz = "east" if ec > c else "west" if ec < c else None
            order = [vert, horiz] if abs(er - r) >= abs(ec - c) else [horiz, vert]
            action = "stay"
            for a in order:
                if a is None:
                    continue
                dr, dc = _DELTA[a]
                if grid.passable((r + dr, c + dc)):
                    action = a
                    break
            stalled += action == "stay"
            actions[pid] = action
        return actions, {"active": len(actions), "stalled": stalled}
