"""Deterministic, observation-only serializations for representation ablations."""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .contract import validate_observation


def _rule_string(rules: Mapping[str, Any]) -> str:
    return json.dumps(dict(rules), sort_keys=True, separators=(",", ":"))


def _grid(obs: Mapping[str, Any]) -> str:
    m = obs["map"]
    rows = [["." for _ in range(m["width"])] for _ in range(m["height"])]
    for r, c in m["obstacles"]:
        rows[r][c] = "#"
    for e in m["exits"]:
        r, c = e["position"]
        rows[r][c] = "E"
    for p in obs["people"]:
        if p["status"] == "active":
            r, c = p["position"]
            rows[r][c] = "P"
    for r, c in obs["burning"]:
        rows[r][c] = "F"  # same precedence as benchmark's render()
    return "\n".join("".join(row) for row in rows)


def _nearby(obs: Mapping[str, Any], p: Mapping[str, Any]) -> str:
    m = obs["map"]
    walls = {tuple(x) for x in m["obstacles"]}
    burning = {tuple(x) for x in obs["burning"]}
    exits = {tuple(e["position"]) for e in m["exits"]}
    occupied = {tuple(x["position"]) for x in obs["people"] if x["status"] == "active"}
    r, c = p["position"]
    kinds = []
    for k, dr, dc in (("N", -1, 0), ("S", 1, 0), ("E", 0, 1), ("W", 0, -1)):
        cell = (r + dr, c + dc)
        if not (0 <= cell[0] < m["height"] and 0 <= cell[1] < m["width"]):
            t = "boundary"
        elif cell in walls:
            t = "wall"
        elif cell in burning:
            t = "fire"
        elif cell in occupied:
            t = "person"
        elif cell in exits:
            t = "exit"
        else:
            t = "empty"
        kinds.append(f"{k}={t}")
    return ",".join(kinds)


def encode_observation(observation: Mapping[str, Any], *, mode: str = "ascii", local_neighbors: bool = False) -> str:
    """Encode the permitted observation with no hidden simulator metadata.

    Valid modes: ascii (2D grid + stable-ID people mapping), coords (explicit
    coordinates); they are hypotheses for a measured representation ablation.
    """
    obs = validate_observation(observation)
    m = obs["map"]
    lines = [f"STEP={obs['step']} SIZE={m['height']}x{m['width']}", f"RULES={_rule_string(obs['rules'])}"]
    lines.append("EXITS=" + ";".join(f"{e['id']}@{e['position'][0]},{e['position'][1]}" for e in m["exits"]))
    if mode == "ascii":
        lines += ["GRID #=wall F=fire P=active E=exit .=empty", _grid(obs)]
    elif mode == "coords":
        lines += ["WALLS=" + ";".join(f"{r},{c}" for r, c in m["obstacles"])]
        lines += ["FIRE=" + ";".join(f"{r},{c}" for r, c in obs["burning"])]
    else:
        raise ValueError(f"unsupported encoding mode: {mode}")
    lines.append("ACTIVE_PEOPLE=" + str(sum(p["status"] == "active" for p in obs["people"])))
    for p in obs["people"]:
        if p["status"] != "active":
            continue
        row, col = p["position"]
        line = f"PERSON {p['id']} @ {row},{col}"
        if local_neighbors:
            line += " ADJ=" + _nearby(obs, p)
        lines.append(line)
    return "\n".join(lines)
