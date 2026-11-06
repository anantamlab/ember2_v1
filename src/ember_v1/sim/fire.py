"""Fire spread (public EMBERBench `docs/simulation-contract.md` "Fire", commit 423143a).

The fire model sees only static terrain, the burning set, the 1-based step
number and the fire RNG stream. It never sees people or actions, so the fire
trajectory for one seed is the same for every action sequence.
"""
from __future__ import annotations

from dataclasses import dataclass

from .rng import SplitMix64

Pos = tuple[int, int]


@dataclass(frozen=True)
class Terrain:
    height: int
    width: int
    walls: frozenset[Pos]
    exits: frozenset[Pos]
    exits_flammable: bool

    def in_grid(self, p: Pos) -> bool:
        return 0 <= p[0] < self.height and 0 <= p[1] < self.width

    def flammable(self, p: Pos) -> bool:
        return self.in_grid(p) and p not in self.walls and (self.exits_flammable or p not in self.exits)


def neighbours(p: Pos) -> tuple[Pos, ...]:
    """4-neighbours in row-major order (north, west, east, south). May leave the grid."""
    r, c = p
    return (r - 1, c), (r, c - 1), (r, c + 1), (r + 1, c)


def frontier(terrain: Terrain, burning: frozenset[Pos]) -> list[tuple[Pos, tuple[Pos, ...]]]:
    """Candidates in row-major order, each with its burning 4-neighbours in row-major order."""
    candidates = {n for b in burning for n in neighbours(b) if n not in burning and terrain.flammable(n)}
    return [(c, tuple(s for s in neighbours(c) if s in burning)) for c in sorted(candidates)]


def edge_count(terrain: Terrain, burning: frozenset[Pos]) -> int:
    return sum(len(srcs) for _, srcs in frontier(terrain, burning))


def spread(terrain: Terrain, burning: frozenset[Pos], step: int, model: str, interval: int,
           probability: float | None, rng: SplitMix64) -> frozenset[Pos]:
    """Cells that ignite in step `step` (1-based). Spread only when `step % interval == 0`."""
    if type(step) is not int or step < 1:
        raise ValueError("step is 1-based")
    if step % interval:
        return frozenset()
    front = frontier(terrain, burning)
    if model == "deterministic":
        return frozenset(c for c, _ in front)
    if model != "stochastic" or probability is None:
        raise ValueError("unknown fire model")
    ignited = set()
    for cell, sources in front:
        hits = [rng.bernoulli(probability) for _ in sources]  # every edge draws, no short-circuit
        if any(hits):
            ignited.add(cell)
    return frozenset(ignited)
