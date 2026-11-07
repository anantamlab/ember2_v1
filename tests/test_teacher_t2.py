"""T2 stochastic fire-forecasting rollout agent (issue #9).

Covers: identity and frozen continuation, observation-consistent fire futures
(public rule, teacher stream, no private state), common random numbers,
per-action values and uncertainty, joint-action construction and validity,
budgets and fallbacks, fail-closed behaviour, and cases where T2 is worse than
frozen T1. Predicted-vs-actual movement checks use the same public resolver as
the simulator, so they are internal consistency checks only.
"""
from __future__ import annotations

import ast
import copy
import math
import random
from collections import Counter
from pathlib import Path

import pytest

from ember_v1.contract import validate_observation
from ember_v1.sim import Scenario, Simulator
from ember_v1.sim.bench import room
from ember_v1.sim.fire import Terrain, spread
from ember_v1.sim.movement import ACTIONS
from ember_v1.sim.rng import SplitMix64
from ember_v1.teacher import t2 as t2mod
from ember_v1.teacher.t1 import T1CongestionAware
from ember_v1.teacher.t2 import (T1_FROZEN_CONFIG_HASH, T2StochasticRollout, World, _Decision, exit_distance,
                                 public_fire_digest, sample_futures, spread_step)
from gen_helpers import random_scenario
from test_teacher_t0 import PRIVATE_KEYS, ascii_scenario

BLOCKED = {"blocked_occupied", "blocked_conflict", "blocked_cycle", "blocked_wall", "blocked_boundary", "blocked_fire"}
SMALL = {"horizon": 3, "samples_stochastic": 2, "max_scored_people": 2, "max_rollouts": 2 * (1 + 4 * 2) + 2}
# Corridor: E0 at col 0, E1 at col 14, Pa at col 9; fire 3 cells below E1 (layer 3), p = 0.3.
FORECAST_ROWS = ["0........a....1", "##############.", "##############.", "##############*"]


def stochastic(rows, p=0.5, **kw):
    d = ascii_scenario(rows, model="stochastic", **kw).to_dict()
    d["fire"]["spread_probability"] = p
    return Scenario.from_dict(d)


def obs_of(s: Scenario, steps: int = 0, policy=None) -> dict:
    sim = Simulator(s)
    pol = policy or T1CongestionAware()
    for _ in range(steps):
        if sim.terminated:
            break
        sim.step(pol.act(sim.observe()))
    return sim.observe()


def closed_loop(s: Scenario, policy) -> tuple[Counter, int, list]:
    sim = Simulator(s)
    decisions = []
    while not sim.terminated:
        d = policy.decide(sim.observe())
        decisions.append(d)
        rec = sim.step(d.actions)
        for pid, mv in rec["movements"].items():
            assert mv["outcome"] not in BLOCKED, (pid, mv)
            assert d.diagnostics["people"][pid]["predicted_outcome"] == mv["outcome"]
    return Counter(p["status"] for p in sim.private_state()["people"]), sim.private_state()["step"], decisions


# ----- identity -----------------------------------------------------------------------
def test_identity_and_frozen_continuation():
    t2 = T2StochasticRollout()
    ident = t2.identity
    assert (ident.name, ident.policy, ident.version) == ("T2", "stochastic_fire_forecast_rollout", "1.0.0")
    assert ident.config_hash not in (T1CongestionAware().identity.config_hash,)
    assert ident.config["continuation_config_hash"] == T1_FROZEN_CONFIG_HASH == T1CongestionAware().identity.config_hash
    assert t2.continuation["name"] == "T1" and t2.continuation["config_hash"] == T1_FROZEN_CONFIG_HASH
    assert T2StochasticRollout({"horizon": 4}).identity.config_hash != ident.config_hash
    d = t2.decide(obs_of(ascii_scenario(["0.a"])))
    assert d.teacher["name"] == "T2" and d.reproducibility["policy_seed"] == t2mod.T2_DEFAULT_SEED
    assert d.diagnostics["continuation"] == t2.continuation


T2_FROZEN_CONFIG_HASH = "sha256:302139e18ba05fdcdc5561b7390b95f725600ec8791e336d065f77cea0c507e5"
T1_SOURCE_SHA256 = "sha256:0365af65c598ce56ef64a6bd4032cb8ef4fefa5f912b004cdc023993e80d4dfa"


def test_frozen_configuration_and_continuation_code_are_pinned():
    """Preregistered T2 identity (docs/P1_T2_ROLLOUT.md section 3) and the unchanged frozen T1 source."""
    t2 = T2StochasticRollout()
    assert t2.identity.config_hash == T2_FROZEN_CONFIG_HASH
    assert t2.continuation["source_sha256"] == T1_SOURCE_SHA256
    doc = (Path(__file__).resolve().parents[1] / "docs/P1_T2_ROLLOUT.md").read_text(encoding="utf-8")
    assert T2_FROZEN_CONFIG_HASH in doc and T1_SOURCE_SHA256 in doc and T1_FROZEN_CONFIG_HASH in doc
    for key, value in t2.identity.config.items():
        if not isinstance(value, str):
            assert f"`{key}`" in doc, key


@pytest.mark.parametrize("cfg", [{"fire_sampler": "x"}, {"continuation_policy": "T0"}, {"horizon": 0},
                                 {"samples_stochastic": 1.5}, {"cpu_budget_ms": float("nan")}, {"nope": 1}])
def test_invalid_or_descriptive_config_is_rejected(cfg):
    with pytest.raises(ValueError):
        T2StochasticRollout(cfg)


def test_t2_module_does_not_import_simulator_internals_or_generator():
    forbidden = ("ember_v1.sim.simulator", "ember_v1.sim.scenario", "ember_v1.sim.rng", "ember_v1.sim.fire",
                 "ember_v1.gen", "ember_v1.teacher.t0", "ember_v1.teacher.evaluate", "ember_v1.teacher.t1_eval",
                 "ember_v1.teacher.t2_eval", "ember_v1.teacher.t2_pilot")
    src = (Path(__file__).resolve().parents[1] / "src/ember_v1/teacher/t2.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom):
            full = {1: f"ember_v1.teacher.{node.module}", 2: f"ember_v1.{node.module}"}.get(node.level, node.module)
            assert not str(full).startswith(forbidden), full
        elif isinstance(node, ast.Import):
            assert not any(a.name.startswith(forbidden) for a in node.names)
    assert "emberbench/v1" not in src and "stream_seed" not in src and "private_state" not in src


# ----- observation-consistent futures -----------------------------------------------------
def _sim_burning(s: Scenario, steps: int, seed: int | None = None) -> list[frozenset]:
    sim = Simulator(s, seed)
    out = [frozenset(map(tuple, sim.observe()["burning"]))]
    while not sim.terminated and len(out) <= steps:
        sim.step({pid: "stay" for pid in sim.active_ids()})
        out.append(frozenset(map(tuple, sim.observe()["burning"])))
    return out


def test_deterministic_future_equals_the_public_rule_trajectory():
    rnd = random.Random(5)
    checked = 0
    for _ in range(40):
        s = random_scenario(rnd, model="deterministic", max_steps=30)
        vo = validate_observation(Simulator(s).observe())
        fut, seeds = sample_futures(World(vo), 8, 16, 1, public_fire_digest(vo))
        assert len(fut) == 1 and len(seeds) == 1                      # one exact future
        real = _sim_burning(s, 8)
        n = min(len(real), len(fut[0]))
        assert fut[0][:n] == real[:n]
        checked += n
    assert checked > 100


class _SplitMixUniform:
    def __init__(self, state):
        self.rng = SplitMix64(state)

    def __call__(self):
        return (self.rng.next_u64() >> 11) / float(1 << 53)


def test_stochastic_sampler_draws_edges_exactly_like_the_public_rule():
    """Fed the same uniform stream, the teacher sampler and the rule implementation ignite the same cells."""
    rnd = random.Random(11)
    for i in range(60):
        s = random_scenario(rnd, model="stochastic", max_steps=30)
        if s.spread_probability in (0.0, 1.0):
            continue
        vo = validate_observation(Simulator(s).observe())
        w = World(vo)
        terrain = Terrain(s.height, s.width, frozenset(s.obstacles), frozenset(p for _, p in s.exits),
                          s.exits_flammable)
        b = frozenset(s.ignitions)
        for t in range(1, 7):
            ours = spread_step(w, b, t, _SplitMixUniform(1000 + i))
            ref = spread(terrain, b, t, "stochastic", s.spread_interval, s.spread_probability, SplitMix64(1000 + i))
            assert ours == ref
            b = b | ref


def test_stochastic_frontier_frequency_matches_edge_rule():
    s = stochastic(["*.*", "...", "0.a"], p=0.3, interval=1)
    vo = validate_observation(Simulator(s).observe())
    w = World(vo)
    rng = random.Random(3)
    hits = Counter()
    n = 4000
    for _ in range(n):
        hits.update(spread_step(w, w.burning0, 1, rng.random))
    assert abs(hits[(0, 1)] / n - (1 - 0.7 ** 2)) < 0.03          # two burning neighbours
    assert abs(hits[(1, 0)] / n - 0.3) < 0.03                       # one burning neighbour
    assert hits[(2, 0)] == 0                                         # not on the frontier


def test_futures_depend_only_on_public_fire_state_and_teacher_seed():
    s = stochastic(FORECAST_ROWS, p=0.3, interval=1)
    vo = validate_observation(Simulator(s).observe())
    f1, s1 = sample_futures(World(vo), 8, 8, 7, public_fire_digest(vo))
    moved = copy.deepcopy(vo)
    moved["people"][0]["position"] = [0, 5]                         # people do not enter the fire digest
    f2, s2 = sample_futures(World(moved), 8, 8, 7, public_fire_digest(moved))
    assert (f1, s1) == (f2, s2)
    f3, s3 = sample_futures(World(vo), 8, 8, 8, public_fire_digest(vo))
    assert s3 != s1
    for traj in f1:                                                  # every future follows the public rule
        for tau in range(1, len(traj)):
            new = traj[tau] - traj[tau - 1]
            assert traj[tau - 1] <= traj[tau]
            assert all(any(nb in traj[tau - 1] for nb in t2mod._nbrs(c)) and World(vo).flammable(c) for c in new)


def test_evaluator_seed_and_scenario_id_do_not_change_the_decision_or_futures():
    base = stochastic(FORECAST_ROWS, p=0.3, interval=1).to_dict()
    decisions, realized = [], set()
    for seed, sid in ((1, "a"), (2, "b"), (987654321, "c"), (2 ** 63, "d")):
        d = copy.deepcopy(base)
        d["seed"], d["scenario_id"] = seed, sid
        s = Scenario.from_dict(d)
        obs = Simulator(s).observe()
        decisions.append(T2StochasticRollout().decide(obs).deterministic_dict())
        realized.add(tuple(_sim_burning(s, 8)))
    assert all(x == decisions[0] for x in decisions)
    assert len(realized) > 1          # the evaluator futures differ; T2's samples do not follow them


@pytest.mark.parametrize("key", PRIVATE_KEYS)
def test_private_fields_are_rejected_not_ignored(key):
    obs = Simulator(ascii_scenario(["0.a"])).observe()
    for where in ("root", "map", "rules", "person"):
        bad = copy.deepcopy(obs)
        {"root": bad, "map": bad["map"], "rules": bad["rules"], "person": bad["people"][0]}[where][key] = 1
        with pytest.raises(ValueError):
            T2StochasticRollout().decide(bad)


# ----- rollout model ---------------------------------------------------------------------
def _sim_value(s: Scenario, horizon: int, cfg: dict, steps_before: int = 0) -> tuple[float, list[frozenset]]:
    """Return of frozen T1 for `horizon` steps on the real simulator, scored with T2's preregistered value."""
    sim = Simulator(s)
    t1 = T1CongestionAware()
    for _ in range(steps_before):
        sim.step(t1.act(sim.observe()))
    obs0 = sim.observe()
    t0, max_steps = obs0["step"], obs0["rules"]["max_steps"]
    traj = [frozenset(map(tuple, obs0["burning"]))]
    value, tau = 0.0, 0
    while not sim.terminated and tau < horizon:
        tau += 1
        rec = sim.step(t1.act(sim.observe()))
        traj.append(frozenset(map(tuple, sim.observe()["burning"])))
        value += sum(cfg["evac_reward"] - cfg["time_cost"] * tau for _ in rec["evacuations"])
        value -= cfg["casualty_penalty"] * len(rec["casualties"])
    obs = sim.observe()
    active = [tuple(p["position"]) for p in obs["people"] if p["status"] == "active"]
    if active:
        if t0 + tau >= max_steps:
            value += cfg["unresolved_at_limit_value"] * len(active)
        else:
            dist = exit_distance(World(validate_observation(obs)), traj[tau])
            for p in active:
                d = dist.get(p)
                if d is None:
                    value -= cfg["trapped_terminal_penalty"]
                elif t0 + tau + d > max_steps:
                    value += cfg["unresolved_at_limit_value"]
                else:
                    value += cfg["evac_reward"] - cfg["time_cost"] * (tau + d) - cfg["unresolved_route_penalty"]
    return value, traj


def test_rollout_model_matches_the_simulator_on_the_realized_future():
    """T2's planning model (phase order, evacuation, casualties, termination) equals the simulator when fed
    the simulator's own realized fire trajectory (test-side only; T2 never gets it)."""
    rnd = random.Random(23)
    pol = T2StochasticRollout()
    cfg = pol.identity.config
    for i in range(80):
        s = random_scenario(rnd, max_steps=rnd.randint(3, 20))
        want, traj = _sim_value(s, cfg["horizon"], cfg)
        dec = _Decision(pol, validate_observation(Simulator(s).observe()))
        dec.futures, dec.rep, dec._exit_dist = [traj], [0], [{}]
        base = pol.t1_actions(Simulator(s).observe())
        assert math.isclose(dec.rollout(base, 0), want, abs_tol=1e-9), i


def test_t1_cache_and_continuation_equal_fresh_t1_calls():
    s = room(16, 30, 2, "stochastic")
    obs = obs_of(s, 2)
    pol = T2StochasticRollout(SMALL)
    d = pol.decide(obs)
    assert d.diagnostics["budget"]["t1_cache_hits"] >= 0
    assert pol.t1_actions(obs) == T1CongestionAware().act(obs)


# ----- per-action values -----------------------------------------------------------------
def test_scored_people_have_five_real_values_unscored_have_none():
    obs = obs_of(room(16, 20, 3, "stochastic"), 2)
    d = T2StochasticRollout(SMALL).decide(obs)
    dg = d.diagnostics
    assert dg["status"] == "scored" and len(dg["scored"]) == 2
    k = dg["samples"]
    for pid, p in dg["people"].items():
        if pid in dg["scored"]:
            assert set(p["values"]) == set(ACTIONS) and all(math.isfinite(v) for v in p["values"].values())
            assert set(p["samples"].values()) == {k}
            assert all(v >= 0 for v in p["variance"].values())
            assert all(math.isclose(p["stderr"][a], math.sqrt(p["variance"][a] / k)) for a in ACTIONS)
            assert p["best_action"] in p["ties"]
        else:
            assert p["values"] is None and "variance" not in p and p["reason"] == "not_selected"
            assert p["action"] == p["t1_action"] or pid in dg["converted_to_stay"]


def test_same_observation_and_seed_give_identical_decisions_and_scores():
    obs = obs_of(room(16, 20, 4, "stochastic"), 3)
    a = T2StochasticRollout(SMALL, seed=5).decide(obs).deterministic_dict()
    b = T2StochasticRollout(SMALL, seed=5).decide(copy.deepcopy(obs)).deterministic_dict()
    assert a == b
    c = T2StochasticRollout(SMALL, seed=6).decide(obs).deterministic_dict()
    assert c["diagnostics"]["provenance"]["sample_seeds"] != a["diagnostics"]["provenance"]["sample_seeds"]


def test_common_random_numbers_are_reused_across_actions(monkeypatch):
    calls = {"futures": 0}
    seen: dict[str, list] = {}
    real_sample, real_rollout = t2mod.sample_futures, _Decision.rollout

    def counting_sample(*a, **kw):
        calls["futures"] += 1
        return real_sample(*a, **kw)

    def recording_rollout(self, joint, k):
        seen.setdefault(repr(sorted(joint.items())), []).append((k, id(self.futures[k])))
        return real_rollout(self, joint, k)

    monkeypatch.setattr(t2mod, "sample_futures", counting_sample)
    monkeypatch.setattr(_Decision, "rollout", recording_rollout)
    obs = obs_of(room(16, 10, 5, "stochastic"), 2)
    d = T2StochasticRollout(SMALL).decide(obs)
    assert calls["futures"] == 1
    used = {tuple(v) for v in seen.values()}
    assert len(used) == 1                         # every evaluated joint map used the same futures objects
    assert len(seen) == d.diagnostics["budget"]["rollouts"] // len(next(iter(used)))


def test_actions_with_identical_step1_state_share_rollouts():
    obs = obs_of(ascii_scenario(["#####", "0.a.1", "#####"]))  # north/south are walls -> same state as stay
    d = T2StochasticRollout().decide(obs)
    p = d.diagnostics["people"]["Pa"]
    assert p["step1_outcome"]["north"] == "blocked_wall" and p["values"]["north"] == p["values"]["stay"]
    assert p["shared_with"]["stay"] is None and p["shared_with"]["north"] == p["shared_with"]["south"] == "stay"
    assert d.actions["Pa"] in ("east", "west")


def test_sampled_futures_change_the_preferred_action_and_exit_blockage_reroutes():
    """K = 1: a future in which E1 burns before Pa can arrive (step 5) makes T2 leave T1's route (east)."""
    s = stochastic(FORECAST_ROWS, p=0.3, interval=1)
    obs = obs_of(s)
    vo = validate_observation(obs)
    assert T1CongestionAware().act(obs) == {"Pa": "east"}
    blocked, clear = [], []
    for seed in range(40):
        fut, _ = sample_futures(World(vo), 8, 1, seed, public_fire_digest(vo))
        d = T2StochasticRollout({"samples_stochastic": 1}, seed=seed).decide(obs)
        p = d.diagnostics["people"]["Pa"]
        if (0, 14) in fut[0][4]:
            blocked.append(seed)
            assert d.actions["Pa"] == "west" and p["values"]["west"] > p["values"]["east"]
        elif all((0, 14) not in b for b in fut[0]):
            clear.append(seed)
            assert d.actions["Pa"] == "east"
    assert blocked and clear
    default = T2StochasticRollout().decide(obs)                       # K = 8: blockage is rare, keep east
    assert default.actions == {"Pa": "east"} and default.diagnostics["people"]["Pa"]["gain_vs_t1"] == 0.0


def test_deterministic_fire_uses_one_exact_future_with_zero_variance():
    s = ascii_scenario(["0...a..*"], model="deterministic", interval=1)
    d = T2StochasticRollout().decide(obs_of(s))
    p = d.diagnostics["people"]["Pa"]
    assert d.diagnostics["samples"] == 1 and set(p["variance"].values()) == {0.0}


# ----- joint construction and validity ----------------------------------------------------
def test_conflicting_individual_best_actions_are_not_combined():
    obs = validate_observation(obs_of(ascii_scenario(["a.b", "...", "0.."])))
    pol = T2StochasticRollout()
    dec = _Decision(pol, obs)
    base = {"Pa": "stay", "Pb": "stay"}
    base_res = dec.resolve(dec.active, base, dec.world.burning0)
    cands = [(0.5, "Pa", "east", {"Pa": (0, 1)}, [1.0]), (0.4, "Pb", "west", {"Pb": (0, 1)}, [0.9])]
    joint, info = dec._construct(base, base_res, [0.5], cands)
    assert joint == {"Pa": "east", "Pb": "stay"}
    assert info["accepted"] == ["Pa"] and info["rejected_conflict"] == ["Pb"]


@pytest.mark.parametrize("n", [1, 10, 50, 100, 200])
def test_crowd_sizes_full_valid_maps_and_repeatable(n):
    obs = obs_of(room(32, n, 1, "stochastic"), 2)
    a = T2StochasticRollout(SMALL).decide(obs)
    active = {p["id"] for p in obs["people"] if p["status"] == "active"}
    assert set(a.actions) == active and set(a.actions.values()) <= set(ACTIONS)
    assert not set(a.diagnostics["predicted_outcomes"]) & BLOCKED
    assert a.deterministic_dict() == T2StochasticRollout(SMALL).decide(obs).deterministic_dict()


def test_closed_loop_random_episodes_valid_joint_maps():
    rnd = random.Random(2)
    steps = deviations = 0
    for _ in range(60):
        st, n, decisions = closed_loop(random_scenario(rnd), T2StochasticRollout())
        steps += n
        deviations += sum(len(d.diagnostics["deviations"]) for d in decisions)
        assert all(d.diagnostics["status"] == "scored" for d in decisions)
    assert steps > 300 and deviations > 0


# ----- budgets and fail-closed --------------------------------------------------------------
def test_zero_cpu_budget_falls_back_to_labeled_t1_map_without_values():
    obs = obs_of(room(16, 20, 1, "stochastic"), 1)
    d = T2StochasticRollout({"cpu_budget_ms": 0.0}).decide(obs)
    assert d.diagnostics["status"] == "budget_fallback_t1" and d.diagnostics["budget"]["limit_hit"] == "cpu_budget_ms"
    assert d.actions == T1CongestionAware().act(obs)
    assert all(p["values"] is None for p in d.diagnostics["people"].values())


def test_rollout_cap_leaves_missing_values_missing():
    obs = obs_of(room(16, 20, 1, "stochastic"), 1)
    d = T2StochasticRollout({**SMALL, "max_rollouts": 2}).decide(obs)
    dg = d.diagnostics
    assert dg["status"] == "budget_limited" and dg["budget"]["limit_hit"] == "max_rollouts"
    assert dg["base_value"] is not None and dg["scored"] == []
    for pid in dg["selection_order"]:
        assert dg["people"][pid]["values"] is None and dg["people"][pid]["reason"] == "budget_limited"
    assert d.actions == T1CongestionAware().act(obs)


def test_cpu_budget_stops_expensive_rollouts():
    obs = obs_of(room(32, 100, 1, "stochastic"), 2)
    d = T2StochasticRollout({"cpu_budget_ms": 200.0, "max_rollouts": 10 ** 6, "max_scored_people": 100}).decide(obs)
    assert d.diagnostics["status"] in ("budget_limited", "budget_fallback_t1")
    assert d.runtime["cpu_ms"] < 200.0 + 1500.0                     # one T1 proposal + one rollout step overrun
    assert set(d.actions) == {p["id"] for p in obs["people"] if p["status"] == "active"}


def test_t1_proposal_failure_fails_closed(monkeypatch):
    pol = T2StochasticRollout()
    monkeypatch.setattr(pol.t1, "_plan", lambda obs: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        pol.decide(obs_of(ascii_scenario(["0.a"])))


def test_incomplete_t1_proposal_fails_closed(monkeypatch):
    pol = T2StochasticRollout()
    monkeypatch.setattr(pol.t1, "_plan", lambda obs: ({}, {"people": {}}))
    with pytest.raises(ValueError):
        pol.decide(obs_of(ascii_scenario(["0.ab"])))


def test_continuation_failure_uses_labeled_valid_t1_fallback(monkeypatch):
    pol = T2StochasticRollout()
    monkeypatch.setattr(pol, "t1_actions", lambda obs: (_ for _ in ()).throw(ValueError("invalid continuation map")))
    obs = obs_of(ascii_scenario(["0....ab"]))
    d = pol.decide(obs)
    assert d.diagnostics["status"] == "continuation_error_fallback_t1" and d.diagnostics["error"] == "ValueError"
    assert d.actions == T1CongestionAware().act(obs)
    assert all(p["values"] is None for p in d.diagnostics["people"].values())


# ----- T2 is not always better than T1 --------------------------------------------------------
def _case(index: int) -> Scenario:
    rnd = random.Random(1)
    for i in range(index + 1):
        s = random_scenario(rnd, model="stochastic")
    return s


def test_case_where_t2_is_worse_than_frozen_t1():
    s = _case(63)
    st1, _, _ = closed_loop(s, T1CongestionAware())
    st2, _, _ = closed_loop(s, T2StochasticRollout())
    assert st2["casualty"] > st1["casualty"]


def test_case_where_t2_is_better_than_frozen_t1():
    s = _case(99)
    st1, _, _ = closed_loop(s, T1CongestionAware())
    st2, _, _ = closed_loop(s, T2StochasticRollout())
    assert st2["casualty"] < st1["casualty"]
