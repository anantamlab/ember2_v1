"""Public-contract RL wrapper around our independent simulator; no benchmark imports."""
from __future__ import annotations

import hashlib
import random
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .sim import Scenario, Simulator
from .sim.movement import DELTAS


def safe_distances(obs: Mapping[str, Any]) -> dict[tuple[int, int], int]:
    """BFS using current PUBLIC walls, burning cells and safe exits, ignoring occupancy."""
    m = obs["map"]
    blocked = {tuple(p) for p in m["obstacles"] + obs["burning"]}
    dist = {tuple(e["position"]): 0 for e in m["exits"] if tuple(e["position"]) not in blocked}
    todo = deque(dist)
    while todo:
        r, c = todo.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            p = r + dr, c + dc
            if 0 <= p[0] < m["height"] and 0 <= p[1] < m["width"] and p not in blocked and p not in dist:
                dist[p] = dist[r, c] + 1
                todo.append(p)
    return dist


def potential(obs: Mapping[str, Any], initial_population: int, terminal: bool = False) -> float:
    if terminal:
        return 0.0
    dist = safe_distances(obs)
    cap = obs["map"]["height"] * obs["map"]["width"]
    return -sum(dist.get(tuple(p["position"]), cap) / cap
                for p in obs["people"] if p["status"] == "active") / initial_population


def outcomes(obs: Mapping[str, Any]) -> dict[str, int]:
    counts = Counter(p["status"] for p in obs["people"])
    return dict(people=len(obs["people"]), evacuated=counts["evacuated"],
                casualties=counts["casualty"], unresolved=counts["active"])


@dataclass(frozen=True)
class RewardConfig:
    gamma: float = 0.99
    eta: float = 0.0
    evacuated: float = 1.0
    casualty: float = -2.0
    unresolved: float = -0.75

    def __post_init__(self):
        import math
        if not all(math.isfinite(x) for x in (self.gamma, self.eta, self.evacuated, self.casualty, self.unresolved)):
            raise ValueError("reward coefficients must be finite")
        if not 0 < self.gamma <= 1 or self.eta < 0:
            raise ValueError("invalid gamma/eta")


class EvacuationEnv:
    def __init__(self, scenario: Scenario, reward: RewardConfig = RewardConfig()):
        self.scenario, self.reward_config = scenario, reward
        self.sim: Simulator | None = None

    def reset(self, *, seed: int | None = None):
        self.sim = Simulator(self.scenario, seed=seed)
        return self.sim.observe(), {"initial_population": len(self.scenario.people)}

    def step(self, actions: Mapping[str, str]):
        if self.sim is None:
            raise RuntimeError("reset before step")
        cfg, n = self.reward_config, len(self.scenario.people)
        phi = potential(self.sim.observe(), n) if cfg.eta else 0.0
        record = self.sim.step(actions)  # ONE simultaneous map, unchanged
        obs, done = self.sim.observe(), self.sim.terminated
        stats = outcomes(obs)
        terminal = ((cfg.evacuated * stats["evacuated"] + cfg.casualty * stats["casualties"]
                     + cfg.unresolved * stats["unresolved"]) / n) if done else 0.0
        shape = cfg.eta * (cfg.gamma * potential(obs, n, done) - phi) if cfg.eta else 0.0
        return obs, terminal + shape, done, False, {
            "record": record, "terminal_reward": terminal, "shaping_reward": shape,
            "terminal_potential": potential(obs, n, True) if done else None, **stats}

    def rollout(self, policy: Callable):
        obs, _ = self.reset()
        while True:
            actions = policy(obs)
            nxt, reward, done, _, info = self.step(actions)
            yield {"observation": obs, "actions": dict(actions), "next_observation": nxt,
                   "reward": reward, "done": done, "info": info}
            obs = nxt
            if done:
                return


def fresh_scenario(namespace: str, split: str, index: int, max_steps: int = 40) -> Scenario:
    """New TRAIN/DEV-only maps, independent of all historical collections and labels.

    No outcome-based rejection: choose reachable initial cells from public geometry only.
    Irregular rooms, optional dividing wall/door, varied exit orientation, stochastic fire.
    """
    if split not in {"train", "dev"} or index < 0 or max_steps < 1:
        raise ValueError("invalid fresh-scenario request")
    key = hashlib.sha256(f"{namespace}/{split}/{index}".encode()).hexdigest()
    rng = random.Random(int(key, 16))
    h, w = rng.randint(8, 12), rng.randint(8, 12)
    cells = [(r, c) for r in range(h) for c in range(w)]
    boundary = [p for p in cells if p[0] in {0, h-1} or p[1] in {0, w-1}]
    exits = rng.sample(boundary, rng.randint(1, 3))
    walls = {p for p in cells if rng.random() < 0.08 and p not in exits}
    if rng.random() < 0.5:
        col, door = rng.randrange(2, w-2), rng.randrange(1, h-1)
        walls.update((r, col) for r in range(h) if r != door and (r, col) not in exits)
    # Use the same public BFS as shaping solely to find initially reachable cells.
    stub = {"map": {"height": h, "width": w, "obstacles": [list(p) for p in walls],
                    "exits": [{"position": list(p)} for p in exits]}, "burning": []}
    reachable = sorted(set(safe_distances(stub)) - set(exits))
    rng.shuffle(reachable)
    n = min(rng.randint(4, 12), len(reachable)-1)
    if n < 1:
        raise RuntimeError("fresh map lacks capacity")
    people, ignition = reachable[:n], reachable[n]
    return Scenario.from_dict({
        "schema_version": "1.0", "scenario_id": f"rl-{split}-{key[:24]}",
        "grid": {"height": h, "width": w}, "obstacles": [list(p) for p in sorted(walls)],
        "exits": [{"id": f"E{i}", "position": list(p)} for i, p in enumerate(exits)],
        "people": [{"id": f"P{i}", "position": list(p)} for i, p in enumerate(people)],
        "ignitions": [list(ignition)],
        "fire": {"model": "stochastic", "spread_interval": rng.randint(2, 4),
                 "spread_probability": rng.choice([0.25, 0.45, 0.65]), "exits_flammable": True},
        "movement": {"conflict_resolution": "lowest_id_wins", "enter_fire": "blocked"},
        "termination": {"max_steps": max_steps}, "seed": int(key[:16], 16)})


def state_features(obs: Mapping[str, Any]) -> list[float]:
    """Centralized critic: all public map/people/fire channels on a 16x16 spatial grid.

    Coarse binning supports arbitrary map sizes; actor still sees the exact original state.
    No simulator-private information, sampled actions or future outcomes enter the value head.
    """
    m, side = obs["map"], 16
    channels = [[0.0] * (side * side) for _ in range(6)]
    def add(channel, pos):
        r, c = pos
        channels[channel][min(side-1, r*side//m["height"])*side + min(side-1, c*side//m["width"])] += 1.0
    for p in m["obstacles"]:
        add(0, p)
    for e in m["exits"]:
        add(1, e["position"])
    for p in obs["burning"]:
        add(2, p)
    for p in obs["people"]:
        add({"active": 3, "evacuated": 4, "casualty": 5}[p["status"]], p["position"])
    rules, counts = obs["rules"], outcomes(obs)
    n = counts["people"]
    return [v for ch in channels for v in ch] + [
        obs["step"]/rules["max_steps"], m["height"]/100, m["width"]/100, n/200,
        counts["evacuated"]/n, counts["casualties"]/n, counts["unresolved"]/n,
        rules["spread_interval"]/16, rules["spread_probability"] or 0.0,
        float(rules["fire_model"] == "stochastic"), float(rules["exits_flammable"]),
        float(rules["enter_fire"] == "casualty")]


def gae(rewards, values, dones, gamma=0.99, lam=0.95, bootstrap=0.0):
    if not len(rewards) == len(values) == len(dones):
        raise ValueError("GAE lengths differ")
    advantages, carry, nxt = [0.0] * len(rewards), 0.0, bootstrap
    for i in reversed(range(len(rewards))):
        live = 1.0 - float(dones[i])
        delta = rewards[i] + gamma * live * nxt - values[i]
        carry = delta + gamma * lam * live * carry
        advantages[i], nxt = carry, values[i]
    return advantages, [a+v for a, v in zip(advantages, values)]
