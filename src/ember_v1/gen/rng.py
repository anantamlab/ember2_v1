"""Generator-private randomness, independent of Python's `random` module.

`random.Random` methods such as `randrange`, `choice` and `sample` changed
between Python releases. The generator must give the same scenarios on every
supported Python, so it uses its own SplitMix64 stream and its own integer
helpers. Seeds come from SHA-256 over a label path, with a prefix that differs
from the simulator's fire-stream prefix (`emberbench/v1/...`), so generator
draws never share a stream with simulator draws.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import TypeVar

from ember_v1.sim.rng import MASK64, SplitMix64

T = TypeVar("T")
_PREFIX = "ember_v1/gen"


def derive_seed(*parts: object) -> int:
    """64-bit seed from an ordered label path. Parts are ints or strings."""
    if not parts:
        raise ValueError("derive_seed needs at least one part")
    text = []
    for p in parts:
        if type(p) is int:
            if not 0 <= p <= MASK64:
                raise ValueError("integer seed parts must be in [0, 2**64 - 1]")
            text.append(f"i{p}")
        elif isinstance(p, str) and p:
            text.append(f"s{p}")
        else:
            raise ValueError(f"unsupported seed part {p!r}")
    raw = hashlib.sha256((_PREFIX + "/" + "/".join(text)).encode("utf-8")).digest()
    return int.from_bytes(raw[:8], "big")


class GenRng:
    """Deterministic draws for the generator. Every method is version independent."""

    __slots__ = ("_sm",)

    def __init__(self, seed: int) -> None:
        self._sm = SplitMix64(seed)

    def u64(self) -> int:
        return self._sm.next_u64()

    def below(self, n: int) -> int:
        """Uniform integer in [0, n) by rejection (no modulo bias)."""
        if type(n) is not int or n < 1:
            raise ValueError("n must be a positive integer")
        limit = (1 << 64) - ((1 << 64) % n)
        while True:
            x = self.u64()
            if x < limit:
                return x % n

    def integer(self, lo: int, hi: int) -> int:
        """Uniform integer in [lo, hi] (both inclusive)."""
        if hi < lo:
            raise ValueError("empty integer range")
        return lo + self.below(hi - lo + 1)

    def unit(self) -> float:
        """Uniform float in [0, 1) with 53 exact bits."""
        return (self.u64() >> 11) / float(1 << 53)

    def chance(self, p: float) -> bool:
        return self.unit() < p

    def choice(self, items: Sequence[T]) -> T:
        if not items:
            raise ValueError("choice from an empty sequence")
        return items[self.below(len(items))]

    def weighted(self, weights: Sequence[tuple[T, int]]) -> T:
        """Choice with integer weights; zero weights are never chosen."""
        total = sum(w for _, w in weights)
        if total < 1:
            raise ValueError("weights must have a positive sum")
        x = self.below(total)
        for item, w in weights:
            if x < w:
                return item
            x -= w
        raise AssertionError("unreachable")

    def shuffle(self, items: list[T]) -> None:
        """Fisher-Yates in place."""
        for i in range(len(items) - 1, 0, -1):
            j = self.below(i + 1)
            items[i], items[j] = items[j], items[i]

    def sample(self, items: Sequence[T], k: int) -> list[T]:
        if not 0 <= k <= len(items):
            raise ValueError("sample size out of range")
        pool = list(items)
        for i in range(k):
            j = i + self.below(len(pool) - i)
            pool[i], pool[j] = pool[j], pool[i]
        return pool[:k]
