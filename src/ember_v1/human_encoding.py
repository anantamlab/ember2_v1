"""Serializations of one human view: text for the language model, numbers for a small network.

Both read the view only. Nothing here can add information that the view does not hold.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .contract import ACTIONS
from .human_view import GOALS, MOVES, SOCIAL_BLOCKS, validate_view

QUESTION_ID = "you"     # one question per row; the same key for every person
_CRITERIA = {"stay": "hold", "north": "row-1", "south": "row+1", "east": "col+1", "west": "col-1"}
_CHANNELS = "#.EFP?"
_STEP_SCALE = 32.0


def encode_view(view: Mapping[str, Any]) -> str:
    """Egocentric text. Row 1 of the window is north; the person is `@` at the centre."""
    validate_view(view)
    rules = json.dumps(dict(view["rules"]), sort_keys=True, separators=(",", ":"))
    lines = [f"STEP={view['step']}", f"RULES={rules}",
             f"LAST action={view['last_action'] or 'none'} result={view['last_outcome'] or 'none'}",
             f"VIEW radius={view['radius']} @=you #=wall F=fire P=person E=exit .=floor ?=unseen",
             *view["window"],
             "NEXT " + " ".join(f"{a}={view['moves'][a]}" for a in MOVES),
             "EXITS " + (";".join(f"{dr:+d},{dc:+d}" for dr, dc in view["exits"]) or "none")]
    if view["goal"] is not None:
        steps = " ".join(f"{a}={'x' if view['steps'][a] is None else view['steps'][a]}" for a in ACTIONS)
        lines.append(f"ROUTE goal={view['goal']} steps_left {steps}")
    return "\n".join(lines)


def view_question() -> dict[str, Any]:
    return {"type": "choice", "instructions": "Your next move", "criteria": dict(_CRITERIA)}


def view_row(view: Mapping[str, Any], gold: str) -> dict[str, Any]:
    """One Clef decision row: the same `state/questions/gold` shape the V1 trainer adapters take."""
    if gold not in ACTIONS:
        raise ValueError(f"unsupported label {gold!r}")
    return {"state": encode_view(view), "questions": {QUESTION_ID: view_question()}, "gold": {QUESTION_ID: gold}}


def feature_size(radius: int) -> int:
    return len(_CHANNELS) * (2 * radius + 1) ** 2 + 6 + 2 + len(GOALS) + 1 + 2 * len(ACTIONS) + 1


def view_features(view: Mapping[str, Any]) -> list[float]:
    """Fixed-length numbers: six window channels, last action, blocked flags, route estimate, time."""
    validate_view(view)
    out = [float(ch == k) for k in _CHANNELS for row in view["window"] for ch in row]
    out += [float(view["last_action"] == a) for a in (None, *ACTIONS)]
    outcome = view["last_outcome"] or ""
    out += [float(outcome.startswith("blocked")), float(outcome in SOCIAL_BLOCKS)]
    out += [float(view["goal"] == g) for g in GOALS] + [float(view["goal"] is None)]
    steps = view["steps"] or {a: None for a in ACTIONS}
    known = [v for v in steps.values() if v is not None]
    base = min(known) if known else 0
    for a in ACTIONS:       # steps relative to the best action; the scale keeps values small
        out += [0.0, 0.0] if steps[a] is None else [1.0, min(1.0, (steps[a] - base) / _STEP_SCALE)]
    out.append(min(1.0, view["step"] / max(1, view["rules"]["max_steps"])))
    return out
