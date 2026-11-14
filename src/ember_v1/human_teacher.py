"""Belief teacher: an automatic expert that decides from a person's own view only.

It replaces human feedback. Because it uses nothing that the student cannot see, every label
it gives is one the student can learn (no "imitation gap" from privileged information).

The full-map teachers T0-T4 can still supply labels (`label_mode`), for an ablation:
  belief      the belief teacher's action (default; realizable).
  privileged  a full-map teacher's action, even where the person could not know it.
  consistent  the full-map action when the person's own estimate does not contradict it,
              else the belief action.
"""
from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from typing import Any

from .contract import ACTIONS
from .human_view import MOVES, SOCIAL_BLOCKS, admissible

LABEL_MODES = ("belief", "privileged", "consistent")
Policy = Callable[[Mapping[str, Mapping[str, Any]]], dict[str, str]]


def belief_action(view: Mapping[str, Any], route: Mapping[str, Any] | None = None) -> str:
    """One person's move from their own route estimate.

    Rules, in order: take the action with the fewest steps left; move only for a real gain;
    on a tie keep the last direction, then prefer a free cell; after a block by another person,
    take an equally good free cell, or step aside after a head-on block.
    `route` supplies `goal/steps` when the view was built without hints.
    """
    steps = (route or view)["steps"]
    if steps is None:
        raise ValueError("the belief teacher needs a route estimate (hints or `route`)")
    legal = [a for a in admissible(view) if steps[a] is not None]
    moves = [a for a in legal if a != "stay"]
    if not moves:
        return "stay"
    here = steps["stay"]

    def rank(a: str):
        return (steps[a], a != view["last_action"], view["moves"][a] == "person", MOVES.index(a))

    best = min(moves, key=rank)
    if here is not None and steps[best] >= here:
        return "stay"
    if view["last_outcome"] in SOCIAL_BLOCKS and best == view["last_action"]:
        free = [a for a in moves if a != best and view["moves"][a] != "person"]
        same = [a for a in free if steps[a] <= steps[best]]
        if same:
            return min(same, key=rank)
        if view["last_outcome"] == "blocked_cycle":
            aside = [a for a in free if here is None or steps[a] <= here + 1]
            if aside:
                return min(aside, key=rank)
    return best


def belief_policy(views: Mapping[str, Mapping[str, Any]], routes: Mapping[str, Any] | None = None
                  ) -> dict[str, str]:
    return {pid: belief_action(v, (routes or {}).get(pid)) for pid, v in sorted(views.items())}


def random_policy(seed: int, masked: bool = True) -> Policy:
    """Uniform random baseline (masked = only actions the person can see are not wasted)."""
    rng = random.Random(seed)

    def act(views, routes=None):
        return {pid: rng.choice(admissible(v) if masked else ACTIONS) for pid, v in sorted(views.items())}
    return act


def labels(views: Mapping[str, Mapping[str, Any]], mode: str = "belief", *,
           routes: Mapping[str, Any] | None = None, privileged: Mapping[str, str] | None = None
           ) -> dict[str, str]:
    """Teacher labels for every viewed person. `privileged` = a full-map teacher's joint action."""
    if mode not in LABEL_MODES:
        raise ValueError(f"unknown label mode {mode!r}")
    own = belief_policy(views, routes)
    if mode == "belief":
        return own
    if privileged is None or set(privileged) != set(views) or set(privileged.values()) - set(ACTIONS):
        raise ValueError("privileged labels must cover exactly the viewed people")
    if mode == "privileged":
        return {pid: privileged[pid] for pid in sorted(views)}
    out = {}
    for pid, view in sorted(views.items()):
        steps, a = ((routes or {}).get(pid) or view)["steps"], privileged[pid]
        here = steps["stay"]
        agrees = a in admissible(view) and steps[a] is not None and (here is None or steps[a] <= here)
        out[pid] = a if agrees else own[pid]
    return out
