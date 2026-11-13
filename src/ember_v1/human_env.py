"""Decentralized evacuation environment: every person acts from their own human view.

The simulator, its rules and its joint step are unchanged (`rl_env.EvacuationEnv`). This wrapper
changes only what the policy receives (per-person views) and how reward is credited (per person).
Rewards may use the full public state; they are a training signal and never enter a view.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .human_view import HumanObserver, ViewConfig
from .rl_env import EvacuationEnv, outcomes, safe_distances
from .sim import Scenario
from .sim.movement import OUTCOMES


@dataclass(frozen=True)
class HumanRewardConfig:
    """Per-person reward. Own result when the person is resolved (the episode cap counts as
    unresolved), plus potential shaping on the true safe distance to an exit, plus an optional
    cost per blocked move. `team` mixes in the mean reward of everyone who acted (0 = selfish,
    1 = fully shared)."""
    gamma: float = 1.0
    eta: float = 0.1
    evacuated: float = 1.0
    casualty: float = -2.0
    unresolved: float = -0.75
    blocked: float = 0.0
    team: float = 0.0

    def __post_init__(self) -> None:
        values = (self.gamma, self.eta, self.evacuated, self.casualty, self.unresolved, self.blocked, self.team)
        if not all(math.isfinite(x) for x in values):
            raise ValueError("reward coefficients must be finite")
        if not 0 < self.gamma <= 1 or self.eta < 0 or self.blocked > 0 or not 0 <= self.team <= 1:
            raise ValueError("invalid gamma/eta/blocked/team")


class HumanEvacuationEnv:
    def __init__(self, scenario: Scenario, view: ViewConfig = ViewConfig(),
                 reward: HumanRewardConfig = HumanRewardConfig()) -> None:
        self.scenario, self.reward_config = scenario, reward
        self.env = EvacuationEnv(scenario)
        self.observer = HumanObserver(view)
        self.observation: dict[str, Any] | None = None

    def reset(self, *, seed: int | None = None):
        self.observation, info = self.env.reset(seed=seed)
        self.observer.reset()
        return self.observer.observe(self.observation), {**info, "observation": self.observation}

    def _potentials(self, obs: Mapping[str, Any]) -> dict[str, float]:
        dist, cap = safe_distances(obs), obs["map"]["height"] * obs["map"]["width"]
        return {p["id"]: -dist.get(tuple(p["position"]), cap) / cap
                for p in obs["people"] if p["status"] == "active"}

    def step(self, actions: Mapping[str, str]):
        """Returns (views, rewards, done, info). `rewards` covers everyone who acted in this step;
        `views` covers everyone still active (empty when the episode is over)."""
        if self.observation is None:
            raise RuntimeError("reset before step")
        cfg = self.reward_config
        before = self._potentials(self.observation) if cfg.eta else {}
        obs, _, done, _, info = self.env.step(actions)
        record = info["record"]
        after = self._potentials(obs) if cfg.eta and not done else {}
        own = {}
        for pid in record["actions"]:
            if pid in record["evacuations"]:
                r = cfg.evacuated
            elif pid in record["casualties"]:
                r = cfg.casualty
            else:
                r = cfg.unresolved if done else 0.0
            if cfg.eta:
                r += cfg.eta * (cfg.gamma * after.get(pid, 0.0) - before[pid])
            if record["movements"][pid]["outcome"].startswith("blocked"):
                r += cfg.blocked
            own[pid] = r
        mean = sum(own.values()) / len(own)
        rewards = {pid: (1 - cfg.team) * r + cfg.team * mean for pid, r in own.items()}
        self.observation = obs
        views = {} if done else self.observer.observe(obs, record)
        return views, rewards, done, {"record": record, "observation": obs, **outcomes(obs)}


class EpisodeStats:
    """Movement accounting for one episode, fed with the simulator's step records."""

    def __init__(self, scenario_id: str) -> None:
        self.scenario_id, self.moves, self.last = scenario_id, Counter(), {}
        self.repeated = self.steps = 0

    def add(self, record: Mapping[str, Any]) -> None:
        for pid, mv in record["movements"].items():
            key = (record["actions"][pid], mv["outcome"])
            self.moves[mv["outcome"]] += 1
            # the same blocked action twice in a row by one person
            self.repeated += mv["outcome"].startswith("blocked") and self.last.get(pid) == key
            self.last[pid] = key
        self.steps += 1

    def result(self, counts: Mapping[str, Any]) -> dict[str, Any]:
        blocked = sum(n for k, n in self.moves.items() if k.startswith("blocked"))
        return {"scenario_id": self.scenario_id, "steps": self.steps,
                **{k: counts[k] for k in ("people", "evacuated", "casualties", "unresolved")},
                "attempted": blocked + self.moves["moved"], "blocked": blocked, "stayed": self.moves["stayed"],
                "repeated_ineffective": self.repeated, "movements": {k: self.moves[k] for k in OUTCOMES}}


def run_episode(scenario: Scenario, policy: Callable, view: ViewConfig = ViewConfig(), *,
                seed: int | None = None) -> dict[str, Any]:
    """Roll one episode. `policy(views, routes) -> {pid: action}`; `routes` is each person's own
    route estimate (the same data a view with hints holds). Returns outcome and movement counts."""
    env = HumanEvacuationEnv(scenario, view, HumanRewardConfig(eta=0.0))
    views, info = env.reset(seed=seed)
    stats, done = EpisodeStats(scenario.scenario_id), False
    while not done:
        views, _, done, info = env.step(policy(views, env.observer.plans()))
        stats.add(info["record"])
    return stats.result(info)


def summarize(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Totals and rates over episodes; `utility` is the project's primary (E - 2C - 0.75U)/N."""
    if not episodes:
        raise ValueError("no episodes")
    keys = ("people", "evacuated", "casualties", "unresolved", "attempted", "blocked", "stayed",
            "repeated_ineffective")
    t = {k: sum(e[k] for e in episodes) for k in keys}
    n, decisions = t["people"], t["attempted"] + t["stayed"]
    return {"episodes": len(episodes), **t,
            "movements": {k: sum(e["movements"][k] for e in episodes) for k in OUTCOMES},
            "evacuated_rate": t["evacuated"] / n, "casualty_rate": t["casualties"] / n,
            "unresolved_rate": t["unresolved"] / n,
            "utility": (t["evacuated"] - 2 * t["casualties"] - 0.75 * t["unresolved"]) / n,
            "blocked_per_attempt": t["blocked"] / t["attempted"] if t["attempted"] else 0.0,
            "stay_rate": t["stayed"] / decisions if decisions else 0.0}
