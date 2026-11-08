"""SplitMix64 stream and stream-seed derivation (public EMBERBench V1 contract).

Source: EMBERBench `docs/simulation-contract.md` "Randomness" and
"Stochastic fire (frozen V1 contract)", commit 423143a. Written from that text
and the public SplitMix64 reference algorithm (Steele, Lea and Flood 2014;
Vigna's C reference), not from EMBERBench code.
"""
from __future__ import annotations

import hashlib

MASK64 = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15
_GAMMA_INV = pow(GAMMA, -1, 1 << 64)
_TWO53 = float(1 << 53)


def stream_seed(seed: int, label: str) -> int:
    """`sha256("emberbench/v1/{seed}/{label}")[:8]`, big-endian (seed in decimal)."""
    if type(seed) is not int or not 0 <= seed <= MASK64:
        raise ValueError("seed must be an integer in [0, 2**64 - 1]")
    if not isinstance(label, str) or not label:
        raise ValueError("label must be a non-empty string")
    digest = hashlib.sha256(f"emberbench/v1/{seed}/{label}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def draws_between(state_before: int, state_after: int) -> int:
    """Exact draw count between two states: each draw adds GAMMA mod 2**64."""
    return ((state_after - state_before) * _GAMMA_INV) & MASK64


def check_probability(p: object) -> float:
    if type(p) not in (int, float) or not 0 <= p <= 1:  # bool is rejected: type(True) is bool
        raise ValueError("probability must be a finite number in [0, 1]")
    return float(p)


class SplitMix64:
    """64-bit SplitMix: add GAMMA, then mix. One `next_u64()` is one draw."""

    __slots__ = ("state",)

    def __init__(self, state: int) -> None:
        if type(state) is not int or not 0 <= state <= MASK64:
            raise ValueError("state must be an integer in [0, 2**64 - 1]")
        self.state = state

    def next_u64(self) -> int:
        self.state = (self.state + GAMMA) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def bernoulli(self, p: float) -> bool:
        """Exactly one draw: `(u >> 11) / 2**53 < p`. The quotient is an exact dyadic float."""
        p = check_probability(p)
        return (self.next_u64() >> 11) / _TWO53 < p
