"""Generator specification: independent scenario dimensions with strict validation.

Every dimension is set on its own. A range is `[lo, hi]` (inclusive integers);
a mix is `{option: integer weight}`; a percentage is an integer in [0, 100].
Integers only, so the canonical JSON and the spec hash are stable.

The default ranges come from general evacuation factors (room scale, exit
count, crowd density, fire speed). They are not fitted to any benchmark's
scenario families, results or coverage gaps.
"""
from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ember_v1.sim.canonical import canonical_json, digest

GENERATOR_REVISION = "gen-1.0.0"

LAYOUTS = ("open_room", "corridor", "multi_room", "irregular")
CONNECTIVITY = ("preserve", "free")
EXIT_PLACEMENTS = ("random", "spread", "clustered")
CROWD_PLACEMENTS = ("uniform", "clustered", "far_from_exits", "near_exits")
IGNITION_PLACEMENTS = ("random", "near_crowd", "near_exit", "bottleneck", "far_from_crowd")
FIRE_MODELS = ("deterministic", "stochastic")
ENTER_FIRE = ("blocked", "casualty")
TARGETS = ("feasible", "difficult", "unreachable")

# kind, then (min, max) for int/range, or the allowed options for a mix.
_SCHEMA: dict[str, dict[str, tuple]] = {
    "grid": {"short_side": ("range", 3, 256), "aspect_pct": ("range", 100, 800)},
    "layout": {"mix": ("mix", LAYOUTS), "corridor_width": ("range", 1, 4), "door_width": ("range", 1, 4),
               "min_room_side": ("range", 2, 32), "extra_door_pct": ("pct",), "sealed_room_pct": ("pct",),
               "cave_fill_pct": ("range", 30, 60), "cave_smoothing": ("range", 0, 8)},
    "obstacles": {"density_pct": ("range", 0, 40), "connectivity": ("mix", CONNECTIVITY)},
    "exits": {"count": ("range", 1, 16), "placement": ("mix", EXIT_PLACEMENTS), "interior_pct": ("pct",)},
    "crowd": {"density_permille": ("range", 1, 500), "count": ("range", 1, 2000),
              "placement": ("mix", CROWD_PLACEMENTS), "clusters": ("range", 1, 8)},
    "ignition": {"count": ("range", 1, 16), "placement": ("mix", IGNITION_PLACEMENTS)},
    "fire": {"model": ("mix", FIRE_MODELS), "spread_interval": ("range", 1, 16),
             "spread_probability_pct": ("range", 0, 100), "exits_flammable_pct": ("pct",),
             "enter_fire": ("mix", ENTER_FIRE)},
    "horizon": {"steps_per_side_pct": ("range", 50, 1000), "max_steps": ("range", 1, 100000)},
    "targets": {"mix": ("mix", TARGETS)},
    "attempts": {"family": ("int", 1, 1000), "lineage": ("int", 1, 1000)},
}

DEFAULT_SPEC: dict[str, Any] = {
    "grid": {"short_side": [8, 30], "aspect_pct": [100, 250]},
    "layout": {"mix": {"open_room": 1, "corridor": 1, "multi_room": 1, "irregular": 1},
               "corridor_width": [1, 3], "door_width": [1, 3], "min_room_side": [3, 6],
               "extra_door_pct": 35, "sealed_room_pct": 15, "cave_fill_pct": [38, 46], "cave_smoothing": [2, 5]},
    "obstacles": {"density_pct": [0, 20], "connectivity": {"preserve": 3, "free": 1}},
    "exits": {"count": [1, 4], "placement": {"random": 1, "spread": 1, "clustered": 1}, "interior_pct": 10},
    "crowd": {"density_permille": [10, 200], "count": [1, 120],
              "placement": {"uniform": 2, "clustered": 2, "far_from_exits": 1, "near_exits": 1},
              "clusters": [1, 3]},
    "ignition": {"count": [1, 3],
                 "placement": {"random": 2, "near_crowd": 1, "near_exit": 1, "bottleneck": 1, "far_from_crowd": 1}},
    "fire": {"model": {"deterministic": 1, "stochastic": 1}, "spread_interval": [1, 4],
             "spread_probability_pct": [10, 90], "exits_flammable_pct": 50,
             "enter_fire": {"blocked": 1, "casualty": 1}},
    "horizon": {"steps_per_side_pct": [150, 400], "max_steps": [20, 300]},
    "targets": {"mix": {"feasible": 6, "difficult": 3, "unreachable": 1}},
    "attempts": {"family": 40, "lineage": 60},
}


def _check(value: Any, kind: tuple, name: str) -> Any:
    tag = kind[0]
    if tag == "pct":
        if type(value) is not int or not 0 <= value <= 100:
            raise ValueError(f"{name} must be an integer percentage in [0, 100]")
        return value
    if tag == "int":
        if type(value) is not int or not kind[1] <= value <= kind[2]:
            raise ValueError(f"{name} must be an integer in [{kind[1]}, {kind[2]}]")
        return value
    if tag == "range":
        lo_min, hi_max = kind[1], kind[2]
        if (not isinstance(value, (list, tuple)) or len(value) != 2
                or any(type(v) is not int for v in value) or not lo_min <= value[0] <= value[1] <= hi_max):
            raise ValueError(f"{name} must be [lo, hi] integers with {lo_min} <= lo <= hi <= {hi_max}")
        return [value[0], value[1]]
    if tag == "mix":
        allowed = kind[1]
        if not isinstance(value, Mapping) or not value or not set(value) <= set(allowed):
            raise ValueError(f"{name} must map a subset of {list(allowed)} to weights")
        if any(type(w) is not int or w < 0 for w in value.values()) or sum(value.values()) < 1:
            raise ValueError(f"{name} weights must be non-negative integers with a positive sum")
        return {k: value[k] for k in sorted(value)}
    raise AssertionError(tag)


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(dict(base))
    for k, v in override.items():
        if k not in out:
            raise ValueError(f"unknown spec section {k!r}")
        if not isinstance(v, Mapping):
            raise ValueError(f"spec section {k!r} must be a mapping")
        for kk, vv in v.items():
            if kk not in out[k]:
                raise ValueError(f"unknown spec field {k}.{kk}")
            out[k][kk] = copy.deepcopy(vv)
    return out


@dataclass(frozen=True)
class GeneratorSpec:
    """Validated, canonical spec. Build with `from_dict` (strict) or `with_overrides`."""

    canonical: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GeneratorSpec":
        if not isinstance(data, Mapping) or set(data) != set(_SCHEMA):
            raise ValueError(f"spec must contain exactly {sorted(_SCHEMA)}")
        out: dict[str, Any] = {}
        for section, fields in _SCHEMA.items():
            sec = data[section]
            if not isinstance(sec, Mapping) or set(sec) != set(fields):
                raise ValueError(f"spec.{section} must contain exactly {sorted(fields)}")
            out[section] = {f: _check(sec[f], kind, f"{section}.{f}") for f, kind in fields.items()}
        if out["horizon"]["max_steps"][1] > 100000:
            raise ValueError("horizon.max_steps above the schema limit")
        return cls(canonical_json(out))

    @classmethod
    def default(cls) -> "GeneratorSpec":
        return cls.from_dict(DEFAULT_SPEC)

    @classmethod
    def with_overrides(cls, overrides: Mapping[str, Any]) -> "GeneratorSpec":
        return cls.from_dict(_merge(DEFAULT_SPEC, overrides))

    def __post_init__(self) -> None:
        object.__setattr__(self, "_data", json.loads(self.canonical))

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)  # type: ignore[attr-defined]

    def spec_hash(self) -> str:
        return digest({"generator_revision": GENERATOR_REVISION, "spec": self._data})  # type: ignore[attr-defined]

    def get(self, section: str, field: str) -> Any:
        return copy.deepcopy(self._data[section][field])  # type: ignore[attr-defined]
