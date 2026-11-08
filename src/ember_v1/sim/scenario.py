"""Scenario input, schema 1.0 (public EMBERBench `docs/scenario-schema.md`, commit 423143a).

A scenario is simulator-private: it holds the seed and the scenario ID. Only
`Simulator.observe()` output may reach a policy or a teacher.

Fail-closed choices that are stricter than the documented schema (see
docs/P1_SIMULATOR_PARITY.md, U-list): person and exit IDs must match the same
pattern as `scenario_id`.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .canonical import canonical_json, digest
from .rng import MASK64, check_probability

Pos = tuple[int, int]

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_ROOT = {"schema_version", "scenario_id", "grid", "obstacles", "exits", "people", "ignitions",
         "fire", "movement", "termination", "seed"}
_FIRE_MODELS = ("deterministic", "stochastic")
_ENTER_FIRE = ("blocked", "casualty")


def _int(value: Any, name: str, lo: int, hi: int) -> int:
    if type(value) is not int or not lo <= value <= hi:
        raise ValueError(f"{name} must be an integer in [{lo}, {hi}]")
    return value


def _keys(data: Any, required: set[str], name: str, optional: frozenset[str] = frozenset()) -> Mapping[str, Any]:
    if not isinstance(data, Mapping) or not required <= set(data) <= required | optional:
        raise ValueError(f"{name} must contain exactly {sorted(required)} (optional {sorted(optional)})")
    return data


def _id(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise ValueError(f"{name}: invalid id {value!r}")
    return value


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    width: int
    height: int
    obstacles: tuple[Pos, ...]
    exits: tuple[tuple[str, Pos], ...]
    people: tuple[tuple[str, Pos], ...]
    ignitions: tuple[Pos, ...]
    fire_model: str
    spread_interval: int
    exits_flammable: bool
    spread_probability: float | None
    enter_fire: str
    max_steps: int
    seed: int
    description: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Scenario":
        d = _keys(data, _ROOT, "scenario", frozenset({"description"}))
        if d["schema_version"] != "1.0":
            raise ValueError("unsupported schema_version")
        sid = _id(d["scenario_id"], "scenario_id")
        desc = d.get("description")
        if desc is not None and not isinstance(desc, str):
            raise ValueError("description must be a string")
        grid = _keys(d["grid"], {"width", "height"}, "grid")
        w = _int(grid["width"], "grid.width", 1, 1024)
        h = _int(grid["height"], "grid.height", 1, 1024)

        def pos(p: Any, name: str) -> Pos:
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                raise ValueError(f"{name}: expected [row, col]")
            return _int(p[0], f"{name} row", 0, h - 1), _int(p[1], f"{name} col", 0, w - 1)

        def cells(items: Any, name: str) -> tuple[Pos, ...]:
            if not isinstance(items, (list, tuple)):
                raise ValueError(f"{name} must be a list")
            out = [pos(p, name) for p in items]
            if len(out) != len(set(out)):
                raise ValueError(f"duplicate {name}")
            return tuple(sorted(out))

        def named(items: Any, name: str) -> tuple[tuple[str, Pos], ...]:
            if not isinstance(items, (list, tuple)) or not items:
                raise ValueError(f"{name} must be a non-empty list")
            out = []
            for x in items:
                v = _keys(x, {"id", "position"}, name)
                out.append((_id(v["id"], name), pos(v["position"], name)))
            if len({i for i, _ in out}) != len(out) or len({p for _, p in out}) != len(out):
                raise ValueError(f"duplicate {name} ids or positions")
            return tuple(sorted(out))

        walls = cells(d["obstacles"], "obstacles")
        exits = named(d["exits"], "exits")
        people = named(d["people"], "people")
        ignitions = cells(d["ignitions"], "ignitions")
        fire = d["fire"]
        if not isinstance(fire, Mapping) or fire.get("model") not in _FIRE_MODELS:
            raise ValueError("fire.model must be deterministic or stochastic")
        stochastic = fire["model"] == "stochastic"
        base = {"model", "spread_interval", "exits_flammable"}
        _keys(fire, base | {"spread_probability"} if stochastic else base, "fire")
        interval = _int(fire["spread_interval"], "fire.spread_interval", 1, MASK64)
        if type(fire["exits_flammable"]) is not bool:
            raise ValueError("fire.exits_flammable must be bool")
        prob = None
        if stochastic:
            p = fire["spread_probability"]
            if type(p) is float and not math.isfinite(p):
                raise ValueError("fire.spread_probability must be finite")
            prob = check_probability(p)
        movement = _keys(d["movement"], {"conflict_resolution", "enter_fire"}, "movement")
        if movement["conflict_resolution"] != "lowest_id_wins":
            raise ValueError("unsupported conflict_resolution")
        if movement["enter_fire"] not in _ENTER_FIRE:
            raise ValueError("invalid movement.enter_fire")
        term = _keys(d["termination"], {"max_steps"}, "termination")
        max_steps = _int(term["max_steps"], "termination.max_steps", 1, 100000)
        seed = _int(d["seed"], "seed", 0, MASK64)

        wall_set, exit_cells = set(walls), {p for _, p in exits}
        if exit_cells & wall_set:
            raise ValueError("exit on an obstacle")
        if set(ignitions) & wall_set:
            raise ValueError("ignition on an obstacle")
        if not fire["exits_flammable"] and set(ignitions) & exit_cells:
            raise ValueError("ignition on a non-flammable exit")
        if {p for _, p in people} & (wall_set | exit_cells | set(ignitions)):
            raise ValueError("person on an obstacle, exit or ignition")
        return cls(sid, w, h, walls, exits, people, ignitions, fire["model"], interval,
                   fire["exits_flammable"], prob, movement["enter_fire"], max_steps, seed, desc)

    def to_dict(self) -> dict[str, Any]:
        fire: dict[str, Any] = {"model": self.fire_model, "spread_interval": self.spread_interval,
                                "exits_flammable": self.exits_flammable}
        if self.fire_model == "stochastic":
            fire["spread_probability"] = self.spread_probability
        out = {
            "schema_version": "1.0", "scenario_id": self.scenario_id,
            "grid": {"width": self.width, "height": self.height},
            "obstacles": [list(p) for p in self.obstacles],
            "exits": [{"id": i, "position": list(p)} for i, p in self.exits],
            "people": [{"id": i, "position": list(p)} for i, p in self.people],
            "ignitions": [list(p) for p in self.ignitions], "fire": fire,
            "movement": {"conflict_resolution": "lowest_id_wins", "enter_fire": self.enter_fire},
            "termination": {"max_steps": self.max_steps}, "seed": self.seed,
        }
        if self.description is not None:
            out["description"] = self.description
        return out

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    def content_hash(self) -> str:
        """Identity of this simulator's canonical form. Not claimed equal to EMBERBench's hash."""
        return digest(self.to_dict())
