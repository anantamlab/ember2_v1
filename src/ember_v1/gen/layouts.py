"""Static layout builders: walls only. Exits, people and fire are placed later.

Every builder returns a frozenset of wall cells `(row, col)` on an `h x w`
grid. The grid edge acts as a wall in the simulator, so layouts have no outer
wall ring. Builders are pure functions of their arguments and the `GenRng`.

Archetypes (general building forms, not benchmark families):
  open_room   one hall; structure comes only from clutter.
  corridor    a solid block with carved corridors of width `corridor_width`.
  multi_room  binary space partition into rooms; partition walls have doors
              of width `door_width`; extra doors make loops; a sealed room has
              no door (an isolated area).
  irregular   cellular-automaton cave walls (damaged or non-orthogonal space).
"""
from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Any

from .rng import GenRng

Pos = tuple[int, int]


def neighbours4(p: Pos, h: int, w: int) -> list[Pos]:
    r, c = p
    return [(a, b) for a, b in ((r - 1, c), (r, c - 1), (r, c + 1), (r + 1, c)) if 0 <= a < h and 0 <= b < w]


def components(h: int, w: int, walls: frozenset[Pos] | set[Pos]) -> list[list[Pos]]:
    """4-connected free-cell components, each sorted, largest first (ties: first cell)."""
    seen: set[Pos] = set()
    out = []
    for r in range(h):
        for c in range(w):
            if (r, c) in walls or (r, c) in seen:
                continue
            comp, queue = [(r, c)], deque([(r, c)])
            seen.add((r, c))
            while queue:
                for n in neighbours4(queue.popleft(), h, w):
                    if n not in walls and n not in seen:
                        seen.add(n)
                        comp.append(n)
                        queue.append(n)
            out.append(sorted(comp))
    out.sort(key=lambda comp: (-len(comp), comp[0]))
    return out


# ----- archetypes -----------------------------------------------------------

def open_room(h: int, w: int, rng: GenRng, params: dict[str, Any]) -> frozenset[Pos]:
    return frozenset()


def corridor(h: int, w: int, rng: GenRng, params: dict[str, Any]) -> frozenset[Pos]:
    """Corridor network along the long axis, with cross branches and optional loops."""
    transpose = h > w
    H, W = (w, h) if transpose else (h, w)
    cw = max(1, min(params["corridor_width"], max(1, H // 3)))
    params["corridor_width_used"] = cw
    free: set[Pos] = set()

    def carve_row(r0: int, c0: int, c1: int) -> None:
        for r in range(r0, min(H, r0 + cw)):
            for c in range(max(0, c0), min(W, c1 + 1)):
                free.add((r, c))

    def carve_col(c0: int, r0: int, r1: int) -> None:
        for c in range(c0, min(W, c0 + cw)):
            for r in range(max(0, r0), min(H, r1 + 1)):
                free.add((r, c))

    spine = rng.integer(0, H - cw)
    carve_row(spine, 0, W - 1)
    spines = [spine]
    if rng.chance(params["extra_door_pct"] / 100) and H >= 2 * cw + 3:
        other = rng.integer(0, H - cw)
        if abs(other - spine) > cw:
            carve_row(other, 0, W - 1)
            spines.append(other)
    branches = rng.integer(1, max(1, W // 5))
    for _ in range(branches):
        col = rng.integer(0, W - cw)
        kind = rng.integer(0, 2)  # 0 up, 1 down, 2 through
        s = rng.choice(spines)
        if kind == 0:
            carve_col(col, rng.integer(0, s), s)
        elif kind == 1:
            carve_col(col, s, rng.integer(s, H - 1))
        else:
            carve_col(col, 0, H - 1)
    walls = {(r, c) for r in range(H) for c in range(W)} - free
    if transpose:
        walls = {(c, r) for r, c in walls}
    return frozenset(walls)


def multi_room(h: int, w: int, rng: GenRng, params: dict[str, Any]) -> frozenset[Pos]:
    """Binary space partition. Doors are carved in each partition wall."""
    min_side = params["min_room_side"]
    dw_max = params["door_width"]
    walls: set[Pos] = set()
    doors: list[tuple[list[Pos], bool]] = []
    leaves: list[tuple[int, int, int, int]] = []

    def split(r0: int, c0: int, r1: int, c1: int, depth: int) -> None:
        hh, ww = r1 - r0 + 1, c1 - c0 + 1
        can_h = hh >= 2 * min_side + 1
        can_v = ww >= 2 * min_side + 1
        if not (can_h or can_v) or (depth >= 2 and rng.chance(0.25)):
            leaves.append((r0, c0, r1, c1))
            return
        horizontal = can_h and (not can_v or (hh > ww) or (hh == ww and rng.chance(0.5)))
        if horizontal:
            line = rng.integer(r0 + min_side, r1 - min_side)
            seg = [(line, c) for c in range(c0, c1 + 1)]
        else:
            line = rng.integer(c0 + min_side, c1 - min_side)
            seg = [(r, line) for r in range(r0, r1 + 1)]
        walls.update(seg)
        n_doors = 1 + (1 if rng.chance(params["extra_door_pct"] / 100) and len(seg) >= 6 else 0)
        for _ in range(n_doors):
            dw = min(rng.integer(1, dw_max), max(1, len(seg) - 2))
            start = rng.integer(0, len(seg) - dw)
            doors.append((seg[start:start + dw], horizontal))
        if horizontal:
            split(r0, c0, line - 1, c1, depth + 1)
            split(line + 1, c0, r1, c1, depth + 1)
        else:
            split(r0, c0, r1, line - 1, depth + 1)
            split(r0, line + 1, r1, c1, depth + 1)

    split(0, 0, h - 1, w - 1, 0)
    for door, horizontal in doors:
        walls.difference_update(door)
        # A child partition can end next to a parent door; keep both sides of the door open.
        for r, c in door:
            across = ((r - 1, c), (r + 1, c)) if horizontal else ((r, c - 1), (r, c + 1))
            walls.difference_update(across)
    params["rooms"] = len(leaves)
    sealed = 0
    if len(leaves) >= 3 and rng.chance(params["sealed_room_pct"] / 100):
        r0, c0, r1, c1 = rng.choice(leaves)
        ring = ([(r0 - 1, c) for c in range(c0, c1 + 1)] + [(r1 + 1, c) for c in range(c0, c1 + 1)]
                + [(r, c0 - 1) for r in range(r0, r1 + 1)] + [(r, c1 + 1) for r in range(r0, r1 + 1)])
        walls.update(p for p in ring if 0 <= p[0] < h and 0 <= p[1] < w)
        sealed = 1
    params["sealed_rooms"] = sealed
    return frozenset(walls)


def irregular(h: int, w: int, rng: GenRng, params: dict[str, Any]) -> frozenset[Pos]:
    """Cellular-automaton cave: random fill, then the 4-5 smoothing rule."""
    fill = params["cave_fill_pct"] / 100
    grid = [[rng.chance(fill) for _ in range(w)] for _ in range(h)]
    for _ in range(params["cave_smoothing"]):
        nxt = [[False] * w for _ in range(h)]
        for r in range(h):
            for c in range(w):
                n = 0
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == dc == 0:
                            continue
                        rr, cc = r + dr, c + dc
                        n += 1 if not (0 <= rr < h and 0 <= cc < w) or grid[rr][cc] else 0
                nxt[r][c] = n >= 5 or (grid[r][c] and n == 4)
        grid = nxt
    return frozenset((r, c) for r in range(h) for c in range(w) if grid[r][c])


BUILDERS = {"open_room": open_room, "corridor": corridor, "multi_room": multi_room, "irregular": irregular}


# ----- clutter --------------------------------------------------------------

_RING = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))


def _locally_safe(p: Pos, h: int, w: int, walls: set[Pos]) -> bool:
    """True if walling `p` cannot disconnect its free 4-neighbours.

    Sufficient test: all free 4-neighbours lie on one cyclic run of free cells
    of the 8-ring around `p` (consecutive ring cells are 4-adjacent).
    """
    r, c = p
    ring = [0 <= r + dr < h and 0 <= c + dc < w and (r + dr, c + dc) not in walls for dr, dc in _RING]
    orth = [i for i in (1, 3, 5, 7) if ring[i]]
    if len(orth) <= 1:
        return len(orth) == 1
    if all(ring):
        return True
    start = ring.index(False)
    runs, cur = [], set()
    for k in range(1, 9):
        i = (start + k) % 8
        if ring[i]:
            cur.add(i)
        elif cur:
            runs.append(cur)
            cur = set()
    if cur:
        runs.append(cur)
    return any(set(orth) <= run for run in runs)


def add_clutter(h: int, w: int, walls: frozenset[Pos], density_pct: int, connectivity: str,
                rng: GenRng, keep_free: Iterable[Pos] = ()) -> tuple[frozenset[Pos], int]:
    """Place furniture-like obstacles on `density_pct` % of free cells.

    `preserve`: 1x1 obstacles, each skipped if it might disconnect free space.
    `free`: 1x1 to 2x2 blocks with no connectivity check (may create pockets).
    Returns the new wall set and the number of cells added.
    """
    out = set(walls)
    protected = set(keep_free)
    free = [(r, c) for r in range(h) for c in range(w) if (r, c) not in out and (r, c) not in protected]
    target = (len(free) * density_pct) // 100
    rng.shuffle(free)
    added = 0
    for p in free:
        if added >= target:
            break
        if p in out:
            continue
        if connectivity == "preserve":
            if _locally_safe(p, h, w, out):
                out.add(p)
                added += 1
            continue
        bh, bw = rng.integer(1, 2), rng.integer(1, 2)
        block = [(p[0] + i, p[1] + j) for i in range(bh) for j in range(bw)]
        block = [q for q in block if 0 <= q[0] < h and 0 <= q[1] < w and q not in out and q not in protected]
        for q in block[: target - added]:
            out.add(q)
            added += 1
    return frozenset(out), added
