"""T2: stochastic fire-forecasting rollout reference agent (issue #9).

CPU only, observation only. T2 is a standalone policy with its own name,
version, configuration, seed and evaluation. It uses the frozen T1 policy
(`T1CongestionAware`, config hash `T1_FROZEN_CONFIG_HASH`) only as the
proposal and continuation planner inside its own rollouts.

One decision (all stages are functions of the validated observation, the
public rules, the frozen configuration and the teacher seed):

  1 T1 proposal. Frozen T1 gives a complete joint action map A on the real
    observation. If T1 fails, T2 fails closed (the error propagates).

  2 Fire futures. K futures of H steps are sampled from the visible burning set
    with the public spread rule: spread in step t only if t % spread_interval
    == 0; frontier = flammable cells (not wall; exits only if exits_flammable)
    with a burning 4-neighbour, row-major; deterministic: every frontier cell
    ignites; stochastic: one Bernoulli(p) draw per (cell, burning neighbour)
    edge, every edge draws, the cell ignites if any draw hits. The draws come
    from the TEACHER's own stream: `random.Random(seed_k)` with seed_k =
    sha256("ember_v1/t2/fire-futures@1/<teacher seed>/<public fire digest>/<k>").
    The public fire digest covers only the map, the rules, the step and the
    burning set. Deterministic fire has one exact future (K = 1).
    The futures never depend on actions (the public fire rule ignores people),
    so the same K futures are common random numbers for every action.

  3 Rollout model. Phase order of the public step: movement with the public
    resolver (`sim.movement.resolve_moves`), evacuation on exits not burning at
    the start of the step, spread, casualties (entered_fire / fire_spread),
    termination (all resolved, or the public max_steps). Step 1 uses the
    evaluated joint map; steps 2..H use frozen T1 on an observation rebuilt
    from the rollout state. Return = sum over all people of
      evacuated at relative step tau:   evac_reward - time_cost * tau
      casualty:                          -casualty_penalty
      unresolved when max_steps ends the rollout:  unresolved_at_limit_value
      unresolved at the horizon, a safe exit reachable in that future, arrival
        tau_end + d <= max_steps:        evac_reward - time_cost*(tau_end + d)
                                          - unresolved_route_penalty
      unresolved at the horizon, otherwise (no safe exit reachable in that future,
        or arrival after max_steps): -trapped_terminal_penalty if not reachable,
                                          unresolved_at_limit_value if too late.

  4 Per-action scoring. People are selected in ascending (fire BFS distance,
    person ID) up to `max_scored_people`. For each selected person i and each
    action a in (stay, north, south, east, west): joint map A with i -> a.
    If the step-1 end positions equal those of A or of an action already
    evaluated for i, the rollouts are identical (T1 is deterministic and
    stateless, the futures are common) and the existing values are reused
    (`shared_with`). Otherwise K new rollouts. Values: mean, sample variance,
    standard error, paired difference to A's action with its standard error.
    A person whose rollouts do not all finish inside the budget gets NO values.

  5 Joint action. Best action per scored person (ties within `tie_tolerance`:
    T1's action, else the public action order). A deviation from A is a
    candidate only if gain > min_gain and gain > deviation_z * SE(gain).
    Candidates are added greedily (gain descending, person ID) to J = A only if
    the public resolver gives exactly the step-1 end positions that were
    evaluated (A's ends with the deviation's own changes); otherwise the
    deviation is rejected (`joint_conflict`). If more than one deviation is
    accepted, J is evaluated on the same K futures; if mean V(J) - V(A) <=
    min_gain, T2 uses A plus only the best single deviation (whose rollouts
    already are the evaluation of that exact map). Moves still predicted
    blocked become `stay` (end positions are asserted unchanged).

  6 Budgets: `max_rollouts` (deterministic count), `cpu_budget_ms` (thread CPU,
    checked after every rollout step). If the base rollouts of A do not finish,
    T2 returns A, status `budget_fallback_t1`. Missing values stay missing.

What T2 does NOT do: read any seed, RNG state, future fire, trapped set,
scenario ID, label or split; ACO (T3); risk-based priority (T4); learning.
Per-action values are conditional single-person values with the other people
held to T1; they are not joint-optimal or causal values.
"""
from __future__ import annotations

import hashlib
import math
import random
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

from ..contract import validate_actions, validate_observation
from ..sim.canonical import digest
from ..sim.movement import ACTIONS, resolve_moves
from .interface import ReferencePolicy
from .t1 import T1CongestionAware

Pos = tuple[int, int]
T1_FROZEN_CONFIG_HASH = "sha256:5047e942db3326706b6184c9f29e0b992078920f03c8a53adc8cc24258345361"
T1_VERSION = "1.0.0"
T2_DEFAULT_SEED = 20261107
FUTURES_VERSION = "t2-fire-futures@1"
PREFERENCE = ACTIONS                     # tie order after T1's own action: stay, north, south, east, west

_DESCRIPTIVE = {
    "continuation_policy": "T1/congestion_aware_cooperative_planner@1.0.0",
    "continuation_config_hash": T1_FROZEN_CONFIG_HASH,
    "proposal_policy": "frozen_T1_joint_map",
    "fire_sampler": "public_rule; per_edge_bernoulli; teacher_stream_random.Random(sha256(seed,public_fire_digest,k))",
    "deterministic_fire_samples": "1_exact",
    "common_random_numbers": "same_K_futures_for_every_action_and_person",
    "value": "sum_over_people(evac: evac_reward - time_cost*tau; casualty: -casualty_penalty; "
             "unresolved: terminal)",
    "terminal": "route_and_in_time: evac_reward - time_cost*(tau_end+d) - unresolved_route_penalty; "
                "route_too_late_or_limit: unresolved_at_limit_value; no_route: -trapped_terminal_penalty",
    "person_selection": "ascending_fire_bfs_distance_then_person_id",
    "action_dedup": "identical_step1_end_positions_share_rollouts",
    "deviation_rule": "gain > min_gain and gain > deviation_z * se_paired",
    "tie_break": "t1_action_then_stay_north_south_east_west",
    "joint_construction": "greedy_gain_desc_accept_if_resolver_ends_match_evaluated; "
                          "multi_deviation_validated_on_common_futures_else_best_single",
    "budget_fallback": "frozen_T1_joint_map",
    "unscored_people": "t1_action_no_values",
    "unrepaired_blocked_moves": "stay",
}
_NUMERIC = {
    "horizon": (int, 1, 64),
    "samples_stochastic": (int, 1, 256),
    "max_scored_people": (int, 0, 10 ** 4),
    "max_rollouts": (int, 0, 10 ** 7),
    "cpu_budget_ms": (float, 0.0, 10 ** 8),
    "evac_reward": (float, 0.0, 1000.0),
    "time_cost": (float, 0.0, 100.0),
    "casualty_penalty": (float, 0.0, 10000.0),
    "unresolved_route_penalty": (float, 0.0, 1000.0),
    "trapped_terminal_penalty": (float, 0.0, 10000.0),
    "unresolved_at_limit_value": (float, -10000.0, 1000.0),
    "deviation_z": (float, 0.0, 100.0),
    "min_gain": (float, 0.0, 1000.0),
    "tie_tolerance": (float, 0.0, 1.0),
}


def t1_source_digest() -> str:
    """sha256 of t1.py with normalized line endings: the continuation code identity."""
    text = (Path(__file__).resolve().parent / "t1.py").read_text(encoding="utf-8").replace("\r\n", "\n")
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


class BudgetExceeded(Exception):
    """Internal: a rollout could not finish inside the decision budget."""


# ----- public rules: static world and fire futures -----------------------------
class World:
    """Static map and public rules from a validated observation."""

    def __init__(self, obs: dict[str, Any]) -> None:
        m, r = obs["map"], obs["rules"]
        self.obs = obs
        self.height, self.width = m["height"], m["width"]
        self.walls = frozenset(tuple(p) for p in m["obstacles"])
        self.exits = {tuple(x["position"]): x["id"] for x in m["exits"]}
        self.model = r["fire_model"]
        self.interval = r["spread_interval"]
        self.p = r["spread_probability"]
        self.exits_flammable = r["exits_flammable"]
        self.enter_fire = r["enter_fire"]
        self.max_steps = r["max_steps"]
        self.t0 = obs["step"]
        self.burning0 = frozenset(tuple(p) for p in obs["burning"])

    def in_grid(self, p: Pos) -> bool:
        return 0 <= p[0] < self.height and 0 <= p[1] < self.width

    def flammable(self, p: Pos) -> bool:
        return self.in_grid(p) and p not in self.walls and (self.exits_flammable or p not in self.exits)


def _nbrs(p: Pos) -> tuple[Pos, ...]:
    r, c = p
    return (r - 1, c), (r, c - 1), (r, c + 1), (r + 1, c)          # row-major: north, west, east, south


def public_fire_digest(obs: dict[str, Any]) -> str:
    """Digest of the fire-relevant public observation only (map, rules, step, burning)."""
    return digest({"map": obs["map"], "rules": obs["rules"], "step": obs["step"], "burning": obs["burning"]})


def sample_seed(teacher_seed: int, fire_digest: str, k: int) -> int:
    h = hashlib.sha256(f"ember_v1/t2/fire-futures@1/{teacher_seed}/{fire_digest}/{k}".encode("ascii")).digest()
    return int.from_bytes(h[:8], "big")


def spread_step(world: World, burning: frozenset[Pos], t: int, uniform) -> frozenset[Pos]:
    """Cells that ignite in 1-based step t under the public rule. `uniform()` -> float in [0, 1)."""
    if t % world.interval:
        return frozenset()
    cand = {n for b in burning for n in _nbrs(b) if n not in burning and world.flammable(n)}
    if world.model == "deterministic":
        return frozenset(cand)
    p = world.p
    out = set()
    for c in sorted(cand):
        hit = False
        for s in _nbrs(c):
            if s in burning and uniform() < p:     # every edge draws (no short-circuit on `hit`)
                hit = True
        if hit:
            out.add(c)
    return frozenset(out)


def sample_futures(world: World, horizon: int, k_samples: int, teacher_seed: int,
                   fire_digest: str) -> tuple[list[list[frozenset[Pos]]], list[int]]:
    """K futures; future[k][tau] = burning set after tau more steps (tau = 0 is now)."""
    n_steps = max(0, min(horizon, world.max_steps - world.t0))
    k_eff = 1 if world.model == "deterministic" else k_samples
    futures, seeds = [], []
    for k in range(k_eff):
        seed = sample_seed(teacher_seed, fire_digest, k)
        rng = random.Random(seed)
        b = world.burning0
        traj = [b]
        for tau in range(1, n_steps + 1):
            b = b | spread_step(world, b, world.t0 + tau, rng.random)
            traj.append(b)
        futures.append(traj)
        seeds.append(seed)
    return futures, seeds


def fire_distance(world: World) -> dict[Pos, int]:
    """BFS distance from the burning set over non-wall cells (fire proximity, for person selection)."""
    dist = {b: 0 for b in world.burning0}
    q = deque(sorted(world.burning0))
    while q:
        p = q.popleft()
        for n in _nbrs(p):
            if n not in dist and world.in_grid(n) and n not in world.walls:
                dist[n] = dist[p] + 1
                q.append(n)
    return dist


def exit_distance(world: World, burning: frozenset[Pos]) -> dict[Pos, int]:
    """BFS from exits not in `burning` over cells that are not wall and not in `burning`."""
    start = sorted(p for p in world.exits if p not in burning)
    dist = {p: 0 for p in start}
    q = deque(start)
    while q:
        p = q.popleft()
        for n in _nbrs(p):
            if n not in dist and world.in_grid(n) and n not in world.walls and n not in burning:
                dist[n] = dist[p] + 1
                q.append(n)
    return dist


# ----- policy --------------------------------------------------------------------
class T2StochasticRollout(ReferencePolicy):
    name = "T2"
    policy = "stochastic_fire_forecast_rollout"
    version = "1.0.0"

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        # Frozen preregistered configuration (docs/P1_T2_ROLLOUT.md section 3).
        return {**_DESCRIPTIVE, "horizon": 8, "samples_stochastic": 8, "max_scored_people": 4,
                "max_rollouts": 144, "cpu_budget_ms": 20000.0,
                "evac_reward": 1.0, "time_cost": 0.01, "casualty_penalty": 5.0,
                "unresolved_route_penalty": 0.25, "trapped_terminal_penalty": 2.5,
                "unresolved_at_limit_value": 0.0, "deviation_z": 1.0, "min_gain": 0.005,
                "tie_tolerance": 1e-9}

    def __init__(self, config: dict[str, Any] | None = None, *, seed: int = T2_DEFAULT_SEED) -> None:
        super().__init__(config, seed=seed)
        cfg = self.identity.config
        for key, value in _DESCRIPTIVE.items():
            if cfg[key] != value:
                raise ValueError(f"T2: {key} is not configurable")
        for key, (typ, lo, hi) in _NUMERIC.items():
            v = cfg[key]
            ok = (type(v) is int) if typ is int else (type(v) in (int, float) and math.isfinite(v))
            if not ok or not lo <= v <= hi:
                raise ValueError(f"T2: {key} must be {typ.__name__} in [{lo}, {hi}]")
        self.t1 = T1CongestionAware()
        if self.t1.identity.config_hash != T1_FROZEN_CONFIG_HASH or self.t1.identity.version != T1_VERSION:
            raise RuntimeError("T2: the continuation is not the frozen T1")
        self.continuation = {"name": "T1", "version": self.t1.identity.version,
                             "config_hash": self.t1.identity.config_hash, "source_sha256": t1_source_digest()}

    def t1_actions(self, obs: dict[str, Any]) -> dict[str, str]:
        """Frozen T1 action map: the same checks and planner as `T1CongestionAware.decide`, no trace digest."""
        o = validate_observation(obs)
        acts, _ = self.t1._plan(o)
        return validate_actions(o, acts)

    def _plan(self, obs: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
        return _Decision(self, obs).run()


T1_CACHE_MAX = 4096                     # continuation-cache entries per decision (memory bound)


class _Decision:
    def __init__(self, pol: T2StochasticRollout, obs: dict[str, Any]) -> None:
        self.pol = pol
        self.cfg = cfg = pol.identity.config
        self.obs = obs
        self.world = World(obs)
        self.H = cfg["horizon"]
        self.cpu0 = time.thread_time_ns()
        self.cpu_cap_ns = int(float(cfg["cpu_budget_ms"]) * 1e6)
        self.rollouts = self.rollouts_nominal = self.rollout_steps = self.t1_calls = self.t1_cache_hits = 0
        self.t1_cache: dict[tuple, dict[str, str]] = {}
        self.limit: str | None = None
        self.people_all = [dict(p) for p in obs["people"]]
        self.active = {p["id"]: tuple(p["position"]) for p in obs["people"] if p["status"] == "active"}
        self._exit_dist: list[dict[int, dict[Pos, int]]] = []

    # ----- budget ------------------------------------------------------------------
    def _check_cpu(self) -> None:
        if time.thread_time_ns() - self.cpu0 > self.cpu_cap_ns:
            self.limit = "cpu_budget_ms"
            raise BudgetExceeded

    def _reserve(self, n: int) -> None:
        if self.rollouts + n > self.cfg["max_rollouts"]:
            self.limit = "max_rollouts"
            raise BudgetExceeded

    # ----- rollout model -----------------------------------------------------------
    def resolve(self, pos: dict[str, Pos], acts: dict[str, str], burning: frozenset[Pos]):
        w = self.world
        return resolve_moves(height=w.height, width=w.width, walls=w.walls, burning=burning,
                             enter_fire=w.enter_fire, positions=pos, actions=acts)

    def _obs_at(self, step: int, pos: dict[str, Pos], status: dict[str, str], burning: frozenset[Pos]) -> dict:
        people = [{"id": p["id"], "position": list(pos.get(p["id"], p["position"])),
                   "status": status.get(p["id"], p["status"])} for p in self.people_all]
        return {"observation_version": self.obs["observation_version"], "step": step, "map": self.obs["map"],
                "rules": self.obs["rules"], "people": people, "burning": [list(b) for b in sorted(burning)],
                "allowed_actions": list(self.obs["allowed_actions"])}

    def _terminal(self, k: int, tau_end: int, fut: list[frozenset[Pos]], pos: dict[str, Pos]) -> float:
        cfg, w = self.cfg, self.world
        cache = self._exit_dist[k]
        if tau_end not in cache:
            cache[tau_end] = exit_distance(w, fut[tau_end])
        dist = cache[tau_end]
        v = 0.0
        for p in pos.values():
            d = dist.get(p)
            if d is None:
                v -= cfg["trapped_terminal_penalty"]
            elif w.t0 + tau_end + d > w.max_steps:
                v += cfg["unresolved_at_limit_value"]
            else:
                v += cfg["evac_reward"] - cfg["time_cost"] * (tau_end + d) - cfg["unresolved_route_penalty"]
        return v

    def rollout(self, first: dict[str, str], k: int) -> float:
        """Return of one rollout: `first` at step 1, frozen T1 afterwards, fire future k."""
        cfg, w = self.cfg, self.world
        fut = self.futures[k]
        n_steps = len(fut) - 1
        pos = dict(self.active)
        status: dict[str, str] = {}
        value = 0.0
        tau = 0
        while pos and tau < n_steps:
            tau += 1
            b0, b1 = fut[tau - 1], fut[tau]
            if tau == 1:
                acts = first
            else:
                # T1 reads only the active people, the burning set, the rules and the step (PublicGrid,
                # _Fire), so this key fixes its output exactly; the cache only saves repeated calls.
                key = (tau, tuple(sorted(pos.items())), b0)
                acts = self.t1_cache.get(key)
                if acts is None:
                    acts = self.pol.t1_actions(self._obs_at(w.t0 + tau - 1, pos, status, b0))
                    self.t1_calls += 1
                    if len(self.t1_cache) >= T1_CACHE_MAX:
                        self.t1_cache.clear()
                    self.t1_cache[key] = acts
                else:
                    self.t1_cache_hits += 1
            moves = self.resolve(pos, acts, b0)
            for pid, (_, end) in moves.items():
                pos[pid] = end
            for pid in sorted(pos):
                p = pos[pid]
                if p in w.exits and p not in b0:
                    value += cfg["evac_reward"] - cfg["time_cost"] * tau
                    status[pid] = "evacuated"
                elif p in b1:
                    value -= cfg["casualty_penalty"]
                    status[pid] = "casualty"
            for pid in status:
                pos.pop(pid, None)
            self.rollout_steps += 1
            self._check_cpu()
        if pos:
            if w.t0 + tau >= w.max_steps:                 # the public step limit ends the episode
                value += cfg["unresolved_at_limit_value"] * len(pos)
            else:
                value += self._terminal(k, tau, fut, pos)
        return value

    def evaluate(self, joint: dict[str, str]) -> list[float]:
        """K rollouts of one joint map. Either all K finish or BudgetExceeded is raised."""
        reps = sorted(set(self.rep))
        self._reserve(len(reps))
        done = {k: self.rollout(joint, k) for k in reps}
        self.rollouts += len(reps)
        self.rollouts_nominal += len(self.futures)
        return [done[self.rep[k]] for k in range(len(self.futures))]

    # ----- statistics ----------------------------------------------------------------
    @staticmethod
    def stats(xs: list[float]) -> dict[str, Any]:
        n = len(xs)
        mean = math.fsum(xs) / n
        var = math.fsum((x - mean) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
        return {"mean": mean, "variance": var, "stderr": math.sqrt(var / n), "n": n}

    # ----- decision --------------------------------------------------------------------
    def run(self) -> tuple[dict[str, str], dict[str, Any]]:
        cfg, w, pol = self.cfg, self.world, self.pol
        t1_decision = pol.t1._plan(self.obs)               # obs already validated by decide()
        base = validate_actions(self.obs, t1_decision[0])
        t1_people = t1_decision[1]["people"]
        fdig = public_fire_digest(self.obs)
        self.futures, seeds = sample_futures(w, self.H, cfg["samples_stochastic"], pol.seed, fdig)
        self._exit_dist = [{} for _ in self.futures]
        # Identical sampled futures give identical rollouts: each sample k is evaluated by its first equal sample.
        self.rep = [next(j for j in range(k + 1) if self.futures[j] == self.futures[k]) for k in range(len(self.futures))]
        k_eff = len(self.futures)
        fd = fire_distance(w)
        far = w.height * w.width + 1
        order = sorted(self.active, key=lambda pid: (fd.get(self.active[pid], far), pid))
        selected = order[:cfg["max_scored_people"]]
        base_res = self.resolve(self.active, base, w.burning0)
        base_ends = tuple(sorted((pid, e) for pid, (_, e) in base_res.items()))
        people: dict[str, dict[str, Any]] = {}
        status = "scored"
        error = None
        base_vals: list[float] | None = None
        cands: list[tuple[float, str, str, dict[str, Pos], list[float]]] = []
        reached = 0
        try:
            try:
                base_vals = self.evaluate(base)
            except BudgetExceeded:
                status = "budget_fallback_t1"
            if base_vals is not None:
                for pid in selected:
                    try:
                        rec, cand = self._score_person(pid, base, base_res, base_ends, base_vals)
                    except BudgetExceeded:
                        status = "budget_limited"
                        break
                    reached += 1
                    people[pid] = rec
                    if cand is not None:
                        cands.append(cand)
            joint, jinfo = self._construct(base, base_res, base_vals, cands)
        except BudgetExceeded:
            raise
        except Exception as exc:     # a continuation call failed: no value is trusted, use the valid T1 map
            status, error = "continuation_error_fallback_t1", type(exc).__name__
            people, cands, base_vals, reached = {}, [], None, 0
            joint, jinfo = dict(base), {"candidates": 0, "accepted": [], "rejected_conflict": [], "validation": None}
        res = self.resolve(self.active, joint, w.burning0)
        converted = sorted(pid for pid, a in joint.items() if a != "stay" and res[pid][0] != "moved")
        if converted:
            final = {**joint, **{pid: "stay" for pid in converted}}
            res2 = self.resolve(self.active, final, w.burning0)
            if any(res2[p][1] != res[p][1] for p in res):
                raise AssertionError("converting blocked moves to stay changed end positions")
            joint, res = final, res2
        for pid in sorted(self.active):
            if pid not in people:
                reason = ("not_selected" if pid not in selected else
                          "budget_limited" if base_vals is not None else status)
                people[pid] = {"scored": False, "reason": reason, "values": None}
            people[pid].update({"t1_action": base[pid], "action": joint[pid], "predicted_outcome": res[pid][0],
                                "route_found": t1_people[pid]["route_found"]})
        diagnostics = {
            "status": status,
            "error": error,
            "active": len(self.active),
            "horizon": self.H,
            "samples": k_eff,
            "fire_model": w.model,
            "continuation": dict(pol.continuation),
            "futures": FUTURES_VERSION,
            "provenance": {"teacher_seed": pol.seed, "public_fire_digest": fdig, "sample_seeds": seeds},
            "selection_order": selected,
            "scored": sorted(pid for pid, p in people.items() if p["scored"]),
            "base_value": self.stats(base_vals) if base_vals is not None else None,
            "joint": jinfo,
            "converted_to_stay": converted,
            "no_route": sorted(pid for pid in self.active if not t1_people[pid]["route_found"]),
            "deviations": sorted(pid for pid in joint if joint[pid] != base[pid]),
            "distinct_futures": len(set(self.rep)),
            "budget": {"limit_hit": self.limit, "rollouts": self.rollouts, "rollouts_nominal": self.rollouts_nominal,
                       "rollout_steps": self.rollout_steps, "t1_continuation_calls": self.t1_calls,
                       "t1_cache_hits": self.t1_cache_hits, "max_rollouts": cfg["max_rollouts"],
                       "people_reached": reached},
            "predicted_outcomes": dict(sorted(Counter(o for o, _ in res.values()).items())),
            "people": dict(sorted(people.items())),
        }
        return joint, diagnostics

    def _score_person(self, pid: str, base: dict[str, str], base_res, base_ends, base_vals):
        cfg, w = self.cfg, self.world
        by_ends: dict[tuple, str] = {base_ends: base[pid]}
        vals: dict[str, list[float]] = {base[pid]: base_vals}
        shared: dict[str, str | None] = {base[pid]: "t1_joint_map"}
        outcome: dict[str, str] = {base[pid]: base_res[pid][0]}
        deltas: dict[str, dict[str, Pos]] = {base[pid]: {}}
        for a in ACTIONS:
            if a == base[pid]:
                continue
            joint = {**base, pid: a}
            res = self.resolve(self.active, joint, w.burning0)
            ends = tuple(sorted((q, e) for q, (_, e) in res.items()))
            outcome[a] = res[pid][0]
            deltas[a] = {q: e for q, (_, e) in res.items() if e != base_res[q][1]}
            if ends in by_ends:
                src = by_ends[ends]
                vals[a] = vals[src]
                shared[a] = src
            else:
                vals[a] = self.evaluate(joint)
                by_ends[ends] = a
                shared[a] = None
        st = {a: self.stats(vals[a]) for a in ACTIONS}
        best = max(st[a]["mean"] for a in ACTIONS)
        ties = [a for a in ACTIONS if best - st[a]["mean"] <= cfg["tie_tolerance"]]
        chosen = base[pid] if base[pid] in ties else next(a for a in PREFERENCE if a in ties)
        diffs = [x - y for x, y in zip(vals[chosen], base_vals)]
        dst = self.stats(diffs)
        gain, se = dst["mean"], dst["stderr"]
        significant = chosen != base[pid] and gain > cfg["min_gain"] and gain > cfg["deviation_z"] * se
        rec = {"scored": True, "reason": None,
               "values": {a: st[a]["mean"] for a in ACTIONS},
               "variance": {a: st[a]["variance"] for a in ACTIONS},
               "stderr": {a: st[a]["stderr"] for a in ACTIONS},
               "samples": {a: st[a]["n"] for a in ACTIONS},
               "shared_with": shared, "step1_outcome": outcome, "ties": ties, "best_action": chosen,
               "gain_vs_t1": gain, "gain_stderr": se,
               "deviation": "candidate" if significant else ("not_significant" if chosen != base[pid] else None)}
        cand = (gain, pid, chosen, deltas[chosen], vals[chosen]) if significant else None
        return rec, cand

    def _construct(self, base, base_res, base_vals, cands):
        info: dict[str, Any] = {"candidates": len(cands), "accepted": [], "rejected_conflict": [],
                                "validation": None}
        if not cands:
            return dict(base), info
        cands = sorted(cands, key=lambda c: (-c[0], c[1]))
        joint = dict(base)
        expect = {q: e for q, (_, e) in base_res.items()}
        accepted = []
        touched: set[str] = set()            # people whose step-1 end an accepted deviation changed
        for gain, pid, a, delta, _ in cands:
            trial = {**joint, pid: a}
            res = self.resolve(self.active, trial, self.world.burning0)
            want = {**expect, **delta}
            if touched & set(delta) or any(res[q][1] != want[q] for q in res):
                info["rejected_conflict"].append(pid)
                continue
            joint, expect = trial, want
            touched |= set(delta)
            accepted.append((gain, pid, a, delta))
        info["accepted"] = [c[1] for c in accepted]
        if len(accepted) <= 1:
            return joint, info
        single = {**base, cands[0][1]: cands[0][2]}
        single_vals = cands[0][4]
        try:
            jvals = self.evaluate(joint)
        except BudgetExceeded:
            info["validation"] = {"result": "budget_limited_best_single"}
            info["accepted"] = [cands[0][1]]
            return single, info
        d = self.stats([x - y for x, y in zip(jvals, base_vals)])
        ds = self.stats([x - y for x, y in zip(single_vals, base_vals)])
        ok = d["mean"] > self.cfg["min_gain"] and d["mean"] >= ds["mean"]
        info["validation"] = {"result": "joint_accepted" if ok else "best_single", "joint_gain": d["mean"],
                              "joint_gain_stderr": d["stderr"], "best_single_gain": ds["mean"]}
        if not ok:
            info["accepted"] = [cands[0][1]]
            return single, info
        return joint, info
