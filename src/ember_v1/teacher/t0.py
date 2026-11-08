"""T0: observation-only, fire-aware shortest-path reference policy (issue #7).

Algorithm (deterministic, CPU only, no randomness):
  1 Validate the public observation (strict whitelist, see interface.py).
  2 Safe exits = exits that are not burning now.
  3 BFS distance field from all safe exits over 4-neighbour cells that are in
    the grid, not walls and not burning now (grid.exit_distance_field).
  4 Each active person independently:
      - target exit = nearest safe exit; equal distance -> lowest exit ID;
      - action = first move in the order north, south, east, west whose cell is
        one step closer to that target exit;
      - no path to any safe exit -> "stay" (documented safe default), with
        route_found = False and reason "no_safe_exit_reachable".
  5 Predict the joint result of the full action map with the exact public
    resolver (`sim.movement.resolve_moves`). This is a diagnostic only: T0 does
    not change any action from it.

What T0 does NOT do (by design; these belong to T1-T4):
  - no reservation, congestion or exit-load planning; other people are not
    obstacles for the BFS, so several people may target the same cell;
  - no fire forecast: only cells that burn now are avoided (the optional
    spread-risk estimate of issue #7 is not in T0 v1);
  - no urgency ordering, no stochastic rollouts, no ACO.

Joint-movement facts that follow from the construction:
  - every intended move goes to a cell with a strictly smaller distance, so the
    intended-move graph has no cycles: T0 never causes blocked_cycle;
  - every move target is in the grid, not a wall and not burning: T0 never
    causes blocked_boundary, blocked_wall, blocked_fire or entered_fire;
  - blocked_conflict and blocked_occupied can happen (group limitation).
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from ..sim.movement import resolve_moves
from .grid import MOVES, PublicGrid, exit_distance_field
from .interface import ReferencePolicy

NO_ROUTE_ACTION = "stay"


class T0ShortestPath(ReferencePolicy):
    name = "T0"
    policy = "fire_aware_shortest_path"
    version = "1.0.0"

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        # Fixed, descriptive configuration. T0 has no tunable hyperparameters.
        return {"neighbourhood": "4", "avoid": ["boundary", "wall", "burning_now"],
                "exit_tie_break": "lowest_exit_id", "move_tie_break": ["north", "south", "east", "west"],
                "no_route_action": NO_ROUTE_ACTION, "fire_forecast": "none", "coordination": "none"}

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if config:
            raise ValueError("T0 configuration is fixed; use default_config()")
        super().__init__(None, seed=None)

    def _plan(self, obs: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
        grid = PublicGrid.from_observation(obs)
        field = exit_distance_field(grid)
        actions: dict[str, str] = {}
        people: dict[str, dict[str, Any]] = {}
        for pid, pos in grid.active:
            entry = field.get(pos)
            if entry is None:
                actions[pid] = NO_ROUTE_ACTION
                people[pid] = {"action": NO_ROUTE_ACTION, "route_found": False, "reason": "no_safe_exit_reachable",
                               "target_exit": None, "distance": None, "equal_first_moves": 0}
                continue
            dist, exit_id = entry
            if dist == 0:  # cannot happen for an active person (they evacuate on arrival); kept fail-safe
                actions[pid] = "stay"
                people[pid] = {"action": "stay", "route_found": True, "reason": "on_safe_exit",
                               "target_exit": exit_id, "distance": 0, "equal_first_moves": 0}
                continue
            choices = [a for a, (dr, dc) in MOVES if field.get((pos[0] + dr, pos[1] + dc)) == (dist - 1, exit_id)]
            if not choices:
                raise AssertionError("distance field has no descending neighbour")  # construction invariant
            actions[pid] = choices[0]
            people[pid] = {"action": choices[0], "route_found": True, "reason": "shortest_path",
                           "target_exit": exit_id, "distance": dist, "equal_first_moves": len(choices)}

        resolved = resolve_moves(height=grid.height, width=grid.width, walls=grid.walls, burning=grid.burning,
                                 enter_fire=grid.enter_fire, positions=dict(grid.active), actions=actions)
        for pid, (outcome, _) in resolved.items():
            people[pid]["predicted_outcome"] = outcome
        outcomes = Counter(o for o, _ in resolved.values())
        targets = Counter(d["target_exit"] for d in people.values() if d["route_found"])
        diagnostics = {
            "active": len(actions),
            "safe_exits": [i for i, _ in grid.safe_exits()],
            "no_route": sorted(pid for pid, d in people.items() if not d["route_found"]),
            "target_exit_load": dict(sorted(targets.items())),
            "predicted_outcomes": dict(sorted(outcomes.items())),
            "people": dict(sorted(people.items())),
        }
        return actions, diagnostics
