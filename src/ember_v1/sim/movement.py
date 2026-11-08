"""Simultaneous movement resolution (phases 3-5 of the public EMBERBench V1 step).

Source: EMBERBench `docs/simulation-contract.md` "Phase order" and
"Collisions", `docs/v1-fairness-spec.md` R1-R4 and R6, commit 423143a.

Order of rules (each looks only at movers that are still unresolved):
  phase 3  stay -> stayed; off grid -> blocked_boundary; wall -> blocked_wall;
           burning target with enter_fire="blocked" -> blocked_fire.
  rule 1   target holds a non-mover (stay or blocked in phase 3) -> blocked_occupied.
  rule 2   same target: lowest ID (plain string order) wins, others -> blocked_conflict.
           Decided on intent; a loser is never re-awarded the cell.
  rule 3   a winner moves if its target is empty or its occupant moves out; a chain
           that ends at a person who does not move -> blocked_occupied.
  rule 4   closed cycles (swaps, rotations) -> blocked_cycle. Every cycle cell is the
           target of a cycle member, so an outside mover meets a cycle only through a
           rule-2 conflict (R3). On a 4-connected grid every cycle has even length >= 2.

Rule 1 is one pass over phase-3 non-movers (literal reading, U2). Final positions
do not depend on that reading; tests check this.
"""
from __future__ import annotations

from collections.abc import Mapping

Pos = tuple[int, int]

ACTIONS = ("stay", "north", "south", "east", "west")
DELTAS = {"stay": (0, 0), "north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1)}
OUTCOMES = ("stayed", "moved", "blocked_boundary", "blocked_wall", "blocked_fire",
            "blocked_occupied", "blocked_conflict", "blocked_cycle")


def resolve_moves(*, height: int, width: int, walls: frozenset[Pos], burning: frozenset[Pos],
                  enter_fire: str, positions: Mapping[str, Pos], actions: Mapping[str, str]
                  ) -> dict[str, tuple[str, Pos]]:
    """Return `{id: (outcome, end_position)}` for every active person.

    `positions` holds only active people. The caller validates the action set.
    """
    if enter_fire not in ("blocked", "casualty"):
        raise ValueError("invalid enter_fire")
    if set(actions) != set(positions):
        raise ValueError("actions must cover exactly the active people")
    outcome: dict[str, str] = {}
    target: dict[str, Pos] = {}
    for pid in sorted(positions):
        action = actions[pid]
        if action not in DELTAS:
            raise ValueError(f"unsupported action {action!r}")
        if action == "stay":
            outcome[pid] = "stayed"
            continue
        r, c = positions[pid]
        dr, dc = DELTAS[action]
        t = (r + dr, c + dc)
        if not (0 <= t[0] < height and 0 <= t[1] < width):
            outcome[pid] = "blocked_boundary"
        elif t in walls:
            outcome[pid] = "blocked_wall"
        elif t in burning and enter_fire == "blocked":
            outcome[pid] = "blocked_fire"
        else:
            target[pid] = t

    occupant = {pos: pid for pid, pos in positions.items()}
    if len(occupant) != len(positions):
        raise ValueError("two active people share a cell")

    # Rule 1: target holds a phase-3 non-mover.
    for pid, t in target.items():
        occ = occupant.get(t)
        if occ is not None and occ not in target:
            outcome[pid] = "blocked_occupied"

    # Rule 2: lowest ID wins each contested target (intent only).
    claims: dict[Pos, list[str]] = {}
    for pid, t in target.items():
        if pid not in outcome:
            claims.setdefault(t, []).append(pid)
    winners: dict[str, Pos] = {}
    for t, pids in claims.items():
        best = min(pids)
        winners[best] = t
        for pid in pids:
            if pid != best:
                outcome[pid] = "blocked_conflict"

    # Rules 3 and 4: follow each winner's chain; in-degree is at most one.
    for start in sorted(winners):
        if start in outcome:
            continue
        path: list[str] = []
        seen: dict[str, int] = {}
        cur = start
        while True:
            if cur in outcome:
                moves = outcome[cur] == "moved"
                break
            if cur in seen:
                for member in path[seen[cur]:]:
                    outcome[member] = "blocked_cycle"
                path = path[:seen[cur]]
                moves = False
                break
            seen[cur] = len(path)
            path.append(cur)
            occ = occupant.get(winners[cur])
            if occ is None:
                moves = True
                break
            if occ not in winners:
                moves = False
                break
            cur = occ
        for member in path:
            outcome[member] = "moved" if moves else "blocked_occupied"

    return {pid: (outcome[pid], winners[pid] if outcome[pid] == "moved" else positions[pid])
            for pid in sorted(positions)}
