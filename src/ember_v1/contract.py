"""Strict whitelist of EMBERBench V1 model-visible observation fields.

No evaluator-private Scenario, RunRecord, SimState or RNG data crosses this
boundary. The simulator's interface validates action schema separately; this
module mirrors its public data contract for standalone data preparation.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

ACTIONS = ("stay", "north", "south", "east", "west")
_ROOT = {"observation_version", "step", "map", "rules", "people", "burning", "allowed_actions"}
_MAP = {"width", "height", "obstacles", "exits"}
_RULES = {"fire_model", "spread_interval", "spread_probability", "exits_flammable", "enter_fire", "conflict_resolution", "max_steps"}
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _keys(data: Any, fields: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping) or set(data) != fields:
        raise ValueError(f"{name} must contain exactly {sorted(fields)}")
    return data


def _pos(pos: Any, w: int, h: int, name: str) -> tuple[int, int]:
    if not isinstance(pos, (list, tuple)) or len(pos) != 2:
        raise ValueError(f"{name}: expected [row, col]")
    r, c = pos
    if type(r) is not int or type(c) is not int or not (0 <= r < h and 0 <= c < w):
        raise ValueError(f"{name}: out-of-bounds/non-integer position {pos!r}")
    return r, c


def _id(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise ValueError(f"{name}: invalid id {value!r}")
    return value


def validate_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Accept only Observation.to_dict()-shaped data; copy into canonical order.

    Reject any unknown field including seed, rng_state, scenario_id, trapped.
    This is a data boundary, not a sandbox for arbitrary untrusted Python.
    """
    d = _keys(observation, _ROOT, "observation")
    if d["observation_version"] != "1.0":
        raise ValueError("unsupported observation_version")
    if type(d["step"]) is not int or d["step"] < 0:
        raise ValueError("step must be a nonnegative integer")
    m = _keys(d["map"], _MAP, "map")
    w, h = m["width"], m["height"]
    if type(w) is not int or type(h) is not int or w < 1 or h < 1:
        raise ValueError("grid dimensions must be positive integers")
    walls = sorted(_pos(p, w, h, "obstacle") for p in m["obstacles"])
    if len(walls) != len(set(walls)):
        raise ValueError("duplicate walls")
    exits = []
    for x in m["exits"]:
        v = _keys(x, {"id", "position"}, "exit")
        exits.append({"id": _id(v["id"], "exit"), "position": list(_pos(v["position"], w, h, "exit"))})
    if len({x["id"] for x in exits}) != len(exits):
        raise ValueError("duplicate exit ids")
    if len({tuple(x["position"]) for x in exits}) != len(exits):
        raise ValueError("duplicate exit positions")
    if any(tuple(e["position"]) in walls for e in exits):
        raise ValueError("exit overlaps wall")
    rules = _keys(d["rules"], _RULES, "rules")
    if rules["fire_model"] not in ("deterministic", "stochastic"):
        raise ValueError("unknown fire model")
    if type(rules["spread_interval"]) is not int or rules["spread_interval"] < 1:
        raise ValueError("invalid spread interval")
    prob = rules["spread_probability"]
    if rules["fire_model"] == "deterministic" and prob is not None:
        raise ValueError("deterministic spread_probability must be null")
    if rules["fire_model"] == "stochastic" and (type(prob) not in (int, float) or not 0 <= prob <= 1):
        raise ValueError("invalid stochastic spread_probability")
    if type(rules["exits_flammable"]) is not bool:
        raise ValueError("exits_flammable must be bool")
    if rules["enter_fire"] not in ("blocked", "casualty"):
        raise ValueError("invalid enter_fire")
    if rules["conflict_resolution"] != "lowest_id_wins":
        raise ValueError("unsupported conflict_resolution")
    if type(rules["max_steps"]) is not int or rules["max_steps"] < 1:
        raise ValueError("invalid max_steps")
    people = []
    for p in d["people"]:
        v = _keys(p, {"id", "position", "status"}, "person")
        if v["status"] not in ("active", "evacuated", "casualty"):
            raise ValueError("invalid person status")
        people.append({"id": _id(v["id"], "person"), "position": list(_pos(v["position"], w, h, "person")), "status": v["status"]})
    if len({p["id"] for p in people}) != len(people):
        raise ValueError("duplicate person ids")
    active = [tuple(p["position"]) for p in people if p["status"] == "active"]
    if len(active) != len(set(active)):
        raise ValueError("multiple active people in one cell")
    if any(x in walls for x in active):
        raise ValueError("active person on wall")
    burning = sorted(_pos(p, w, h, "burning") for p in d["burning"])
    if len(burning) != len(set(burning)):
        raise ValueError("duplicate burning cells")
    if any(x in walls for x in burning):
        raise ValueError("burning cell is wall")
    allowed = d["allowed_actions"]
    if not isinstance(allowed, (list, tuple)) or list(allowed) != list(ACTIONS):
        raise ValueError("unexpected allowed action order or values")
    return {
        "observation_version": "1.0", "step": d["step"],
        "map": {"width": w, "height": h, "obstacles": [list(x) for x in walls], "exits": sorted(exits, key=lambda x: x["id"])},
        "rules": dict(rules), "people": sorted(people, key=lambda x: x["id"]),
        "burning": [list(x) for x in burning], "allowed_actions": list(ACTIONS),
    }


def validate_actions(observation: Mapping[str, Any], actions: Mapping[str, str]) -> dict[str, str]:
    """Fail closed on invalid model actions (no silent STAY or planner fallback)."""
    obs = validate_observation(observation)
    expected = {p["id"] for p in obs["people"] if p["status"] == "active"}
    if not isinstance(actions, Mapping) or set(actions) != expected:
        raise ValueError(f"actions must contain exactly active ids {sorted(expected)}")
    if any(type(v) is not str or v not in ACTIONS for v in actions.values()):
        raise ValueError("unsupported action; do not repair automatically")
    return {pid: actions[pid] for pid in sorted(expected)}
