"""Versioned reference-policy interface shared by T0-T4 and the baseline controls.

A policy sees only `Observation.to_dict()`. `decide` validates the observation
with the strict public whitelist (`contract.validate_observation`): any extra
field such as a seed, RNG state, scenario ID, trapped set, difficulty label or
lineage metadata is rejected, not ignored. The returned action map is checked
with `contract.validate_actions` (exact active IDs, allowed action names) and is
never repaired.

Each policy has its own name, version and configuration. The configuration hash
covers name, version and configuration, so two policies never share an identity.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ..contract import validate_actions, validate_observation
from ..sim.canonical import digest
from .grid import GRID_UTILS_VERSION

POLICY_INTERFACE_VERSION = "policy-interface@1.0.0"


@dataclass(frozen=True)
class PolicyIdentity:
    name: str            # short teacher name, e.g. "T0"
    policy: str          # descriptive policy ID, e.g. "fire_aware_shortest_path"
    version: str         # policy code version, e.g. "1.0.0"
    config: Mapping[str, Any]
    config_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "policy": self.policy, "version": self.version,
                "config": dict(self.config), "config_hash": self.config_hash}


@dataclass(frozen=True)
class PolicyDecision:
    """One decision. Everything except `runtime` is a deterministic function of
    (identity, policy seed, observation)."""
    actions: dict[str, str]
    teacher: dict[str, Any]
    diagnostics: dict[str, Any]
    reproducibility: dict[str, Any]
    runtime: dict[str, float] = field(default_factory=dict)

    def deterministic_dict(self) -> dict[str, Any]:
        return {"actions": self.actions, "teacher": self.teacher, "diagnostics": self.diagnostics,
                "reproducibility": self.reproducibility}

    def to_dict(self) -> dict[str, Any]:
        return {**self.deterministic_dict(), "runtime": dict(self.runtime)}


class ReferencePolicy(ABC):
    """Base class. Subclasses set the class attributes and implement `_plan`."""
    name: ClassVar[str]
    policy: ClassVar[str]
    version: ClassVar[str]

    def __init__(self, config: Mapping[str, Any] | None = None, *, seed: int | None = None) -> None:
        self._config = dict(self.default_config())
        if config:
            unknown = set(config) - set(self._config)
            if unknown:
                raise ValueError(f"{self.name}: unknown config keys {sorted(unknown)}")
            self._config.update(config)
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2 ** 64):
            raise ValueError("policy seed must be an integer in [0, 2**64)")
        self._seed = seed
        cfg = dict(sorted(self._config.items()))
        self._identity = PolicyIdentity(self.name, self.policy, self.version, cfg,
                                        digest({"name": self.name, "policy": self.policy,
                                                "version": self.version, "config": cfg}))

    @classmethod
    def default_config(cls) -> Mapping[str, Any]:
        return {}

    @property
    def identity(self) -> PolicyIdentity:
        return self._identity

    @property
    def seed(self) -> int | None:
        return self._seed

    def decide(self, observation: Mapping[str, Any]) -> PolicyDecision:
        wall0, cpu0 = time.perf_counter_ns(), time.thread_time_ns()
        obs = validate_observation(observation)
        actions, diagnostics = self._plan(obs)
        actions = validate_actions(obs, actions)
        cpu1, wall1 = time.thread_time_ns(), time.perf_counter_ns()
        return PolicyDecision(
            actions=actions,
            teacher=self._identity.to_dict(),
            diagnostics=diagnostics,
            reproducibility={"interface": POLICY_INTERFACE_VERSION, "utilities": {"grid": GRID_UTILS_VERSION},
                             "step": obs["step"], "observation_digest": digest(obs), "policy_seed": self._seed,
                             "deterministic": True},
            runtime={"wall_ms": (wall1 - wall0) / 1e6, "cpu_ms": (cpu1 - cpu0) / 1e6},
        )

    def act(self, observation: Mapping[str, Any]) -> dict[str, str]:
        """`Agent.act`-style call: the action map only (usable as `run_episode` policy)."""
        return self.decide(observation).actions

    __call__ = act

    @abstractmethod
    def _plan(self, obs: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
        """Return (ID -> action for every active person, diagnostics). `obs` is validated."""
