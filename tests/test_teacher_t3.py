"""T3 dynamic ACO reference agent (issue #23).

Covers: identity and independence from T0-T2, observation-only inputs, the
teacher-owned ACO stream, ordering invariance, the public fire-risk model,
pheromone evaporation/deposit and the exploration rule, dynamic replanning
(fire spread, exit loss, forecast blockage, congestion), joint-action
coordination (lowest-ID precedence, follow chains, swaps, rotations, narrow
corridors, no route), budgets and labeled fallbacks. Predicted-vs-actual
movement checks use the same public resolver as the simulator, so they are
internal consistency checks only.
"""
from __future__ import annotations

import ast
import copy
import random
from collections import Counter
from pathlib import Path

import pytest

from ember_v1.contract import validate_observation
from ember_v1.sim import Scenario, Simulator
from ember_v1.sim.bench import room
from ember_v1.sim.fire import Terrain, spread
from ember_v1.sim.rng import SplitMix64
from ember_v1.teacher.grid import MOVES, PublicGrid, exit_distance_field
from ember_v1.teacher.t0 import T0ShortestPath
from ember_v1.teacher.t1 import T1CongestionAware
from ember_v1.teacher.t2 import T2StochasticRollout
from ember_v1.teacher.t3 import (ACO_STREAM, T3_DEFAULT_SEED, FireRisk, T3DynamicACO, _Colony, aco_seed,
                                 canonical_public_digest)
from gen_helpers import random_scenario
from test_teacher_t0 import PRIVATE_KEYS, ascii_scenario

T3_FROZEN_CONFIG_HASH = "sha256:5a8d4985b1f65d5de18a06c86d8dd522e2ca95e2accd583165b7e7d7ea1d02e0"
BLOCKED = ("blocked_occupied", "blocked_conflict", "blocked_cycle", "blocked_wall", "blocked_boundary", "blocked_fire")
# E0 at col 0, E1 at col 14, Pa at col 9; fire 3 cells below E1 (fire layer 3 at E1).
FORECAST_ROWS = ["0........a....1", "##############.", "##############.", "##############*"]


def stochastic(rows, p=0.5, **kw):
    d = ascii_scenario(rows, model="stochastic", **kw).to_dict()
    d["fire"]["spread_probability"] = p
    return Scenario.from_dict(d)


def obs_of(s: Scenario) -> dict:
    return Simulator(s).observe()


def decide(s: Scenario, pol: T3DynamicACO | None = None):
    return (pol or T3DynamicACO()).decide(obs_of(s))


def no_blocked(d) -> bool:
    return not any(o.startswith("blocked_") for o in d.diagnostics["predicted_outcomes"])


def colony_of(obs: dict, cfg: dict | None = None, seed: int = T3_DEFAULT_SEED) -> _Colony:
    return _Colony(T3DynamicACO(cfg, seed=seed), validate_observation(obs))


# ----- identity and independence ------------------------------------------------------------
def test_identity_and_frozen_config_hash():
    t3 = T3DynamicACO()
    ident = t3.identity
    assert (ident.name, ident.policy, ident.version) == ("T3", "dynamic_ant_colony_optimization", "1.0.0")
    assert ident.config_hash == T3_FROZEN_CONFIG_HASH
    assert t3.seed == T3_DEFAULT_SEED == 20261123
    others = {T0ShortestPath().identity.config_hash, T1CongestionAware().identity.config_hash,
              T2StochasticRollout().identity.config_hash}
    assert ident.config_hash not in others and len(others) == 3
    assert T3DynamicACO({"iterations": 11}).identity.config_hash != ident.config_hash


@pytest.mark.parametrize("cfg", [{"graph": "other"}, {"memory": "cross_step"}, {"iterations": 0},
                                 {"ants_per_person": 1.5}, {"q0": 1.5}, {"evaporation": -0.1},
                                 {"tau_min": 2.0}, {"cpu_budget_ms": float("nan")}, {"nope": 1}])
def test_invalid_or_descriptive_config_is_rejected(cfg):
    with pytest.raises(ValueError):
        T3DynamicACO(cfg)


def test_teacher_seed_is_required_and_bounded():
    with pytest.raises(ValueError):
        T3DynamicACO(seed=None)
    with pytest.raises(ValueError):
        T3DynamicACO(seed=-1)


def test_t3_module_is_independent_of_t0_t1_t2_and_private_state():
    forbidden = ("ember_v1.sim.simulator", "ember_v1.sim.scenario", "ember_v1.sim.rng", "ember_v1.sim.fire",
                 "ember_v1.gen", "ember_v1.teacher.t0", "ember_v1.teacher.t1", "ember_v1.teacher.t2",
                 "ember_v1.teacher.evaluate", "ember_v1.teacher.controls", "ember_v1.teacher.t3_eval",
                 "ember_v1.teacher.t3_pilot")
    allowed = {"ember_v1.sim.canonical", "ember_v1.sim.movement", "ember_v1.teacher.grid",
               "ember_v1.teacher.interface", "__future__", "hashlib", "math", "random", "time", "bisect",
               "collections", "typing"}
    src = (Path(__file__).resolve().parents[1] / "src/ember_v1/teacher/t3.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom):
            full = {1: f"ember_v1.teacher.{node.module}", 2: f"ember_v1.{node.module}"}.get(node.level, node.module)
            assert not str(full).startswith(forbidden), full
            assert full in allowed, full
        elif isinstance(node, ast.Import):
            assert all(a.name in allowed for a in node.names)
    assert "emberbench/v1" not in src and "stream_seed" not in src and "private_state" not in src


@pytest.mark.parametrize("key", PRIVATE_KEYS)
def test_private_fields_are_rejected_not_ignored(key):
    obs = obs_of(ascii_scenario(["0.a"]))
    for where in ("root", "map", "rules", "person"):
        bad = copy.deepcopy(obs)
        {"root": bad, "map": bad["map"], "rules": bad["rules"], "person": bad["people"][0]}[where][key] = 1
        with pytest.raises(ValueError):
            T3DynamicACO().decide(bad)


# ----- reproducibility and observation-only evidence -------------------------------------------
def test_repeated_decisions_are_identical_and_stateless():
    s = room(24, 30, 3, "stochastic")
    pol = T3DynamicACO()
    a = pol.decide(obs_of(s)).deterministic_dict()
    other = obs_of(room(24, 10, 4, "deterministic"))
    pol.decide(other)                                   # no memory between decisions
    b = pol.decide(obs_of(s)).deterministic_dict()
    c = T3DynamicACO().decide(obs_of(s)).deterministic_dict()
    assert a == b == c
    prov = a["diagnostics"]["provenance"]
    vo = validate_observation(obs_of(s))
    assert prov == {"stream": ACO_STREAM, "teacher_seed": T3_DEFAULT_SEED,
                    "aco_seed": aco_seed(T3_DEFAULT_SEED, canonical_public_digest(vo)),
                    "public_observation_digest": canonical_public_digest(vo)}


def test_teacher_seed_controls_the_ant_stream():
    s = room(24, 30, 3, "stochastic")
    d1 = T3DynamicACO(seed=1).decide(obs_of(s)).diagnostics
    d2 = T3DynamicACO(seed=2).decide(obs_of(s)).diagnostics
    assert d1["provenance"]["aco_seed"] != d2["provenance"]["aco_seed"]
    assert d1["colony"] != d2["colony"]                # different ant samples


def test_exploitation_only_makes_no_random_draws():
    s = room(24, 30, 3, "stochastic")
    cfg = {"q0": 1.0}
    d1 = T3DynamicACO(cfg, seed=1).decide(obs_of(s)).deterministic_dict()
    d2 = T3DynamicACO(cfg, seed=2).decide(obs_of(s)).deterministic_dict()
    assert d1["actions"] == d2["actions"]
    assert d1["diagnostics"]["colony"] == d2["diagnostics"]["colony"]
    assert d1["diagnostics"]["provenance"]["aco_seed"] != d2["diagnostics"]["provenance"]["aco_seed"]


def test_people_and_exit_order_do_not_change_the_decision():
    s = room(24, 30, 3, "stochastic")
    obs = obs_of(s)
    for k in range(3):
        perm = copy.deepcopy(obs)
        random.Random(k).shuffle(perm["people"])
        perm["map"]["exits"] = list(reversed(perm["map"]["exits"]))
        random.Random(k + 10).shuffle(perm["map"]["obstacles"])
        a, b = T3DynamicACO().decide(obs), T3DynamicACO().decide(perm)
        assert a.actions == b.actions and a.diagnostics == b.diagnostics


def test_evaluator_seed_and_scenario_id_do_not_change_the_decision():
    base = stochastic(FORECAST_ROWS, p=0.5, interval=1).to_dict()
    decisions, realized = [], set()
    for seed, sid in ((1, "a"), (2, "b"), (987654321, "c"), (2 ** 63, "d")):
        d = copy.deepcopy(base)
        d["seed"], d["scenario_id"] = seed, sid
        s = Scenario.from_dict(d)
        decisions.append(T3DynamicACO().decide(obs_of(s)).deterministic_dict())
        sim = Simulator(s)
        traj = []
        for _ in range(6):
            sim.step({p: "stay" for p in sim.active_ids()})
            traj.append(tuple(map(tuple, sim.observe()["burning"])))
            if sim.terminated:
                break
        realized.add(tuple(traj))
    assert all(x == decisions[0] for x in decisions)
    assert len(realized) > 1                            # the evaluator futures differ; T3 does not follow them


# ----- public fire-risk model --------------------------------------------------------------------
def _stay_trajectory(s: Scenario, k: int) -> list[frozenset]:
    sim = Simulator(s)
    out = []
    for _ in range(k):
        if sim.terminated:
            break
        sim.step({p: "stay" for p in sim.active_ids()})
        out.append(frozenset(map(tuple, sim.observe()["burning"])))
    return out


def test_deterministic_hazard_equals_the_public_rule():
    rnd = random.Random(23)
    checks = 0
    for _ in range(40):
        s = random_scenario(rnd, model="deterministic", max_steps=30)
        obs = validate_observation(obs_of(s))
        risk = FireRisk(obs, PublicGrid.from_observation(obs))
        traj = _stay_trajectory(s, 8)
        walls = set(map(tuple, obs["map"]["obstacles"]))
        burning0 = set(map(tuple, obs["burning"]))
        for k, burning in enumerate(traj, start=1):
            for r in range(obs["map"]["height"]):
                for c in range(obs["map"]["width"]):
                    if (r, c) in walls or (r, c) in burning0:
                        continue
                    assert risk.hazard((r, c), k) == (1.0 if (r, c) in burning else 0.0)
                    checks += 1
    assert checks > 1000


def test_stochastic_hazard_bounds_monotone_and_layer1_frequency():
    s = stochastic(["...a", "....", ".*..", "...0"], p=0.3, interval=1)
    obs = validate_observation(obs_of(s))
    risk = FireRisk(obs, PublicGrid.from_observation(obs))
    for cell in [(r, c) for r in range(4) for c in range(4) if (r, c) != (2, 1)]:
        hs = [risk.hazard(cell, k) for k in range(0, 9)]
        assert hs[0] == 0.0 and all(0.0 <= x <= 1.0 for x in hs) and hs == sorted(hs)
    assert risk.hazard((1, 1), 1) == pytest.approx(0.3)                       # one burning neighbour
    assert risk.hazard((1, 1), 2) == pytest.approx(1 - 0.7 ** 2)
    # Monte Carlo with the simulator's own spread rule: layer-1 exact, layer-2 a lower bound.
    terrain = Terrain(4, 4, frozenset(), frozenset(), True)
    rng = SplitMix64(99)
    hits1 = hits2 = 0
    trials = 4000
    for _ in range(trials):
        b = frozenset({(2, 1)})
        for t in (1, 2):
            b = b | spread(terrain, b, t, "stochastic", 1, 0.3, rng)
        hits1 += (1, 1) in b
        hits2 += (0, 1) in b
    assert abs(hits1 / trials - risk.hazard((1, 1), 2)) < 0.03
    assert hits2 / trials >= risk.hazard((0, 1), 2) - 0.03


def test_hazard_respects_spread_interval_and_exit_phase_order():
    s = ascii_scenario(["0.*", "a.."], interval=2)
    obs = validate_observation(obs_of(s))
    risk = FireRisk(obs, PublicGrid.from_observation(obs))
    assert risk.spreads(1) == 0 and risk.spreads(2) == 1 and risk.spreads(4) == 2
    assert risk.hazard((0, 1), 1) == 0.0 and risk.hazard((0, 1), 2) == 1.0
    col = colony_of(obs_of(s))
    # the exit (layer 2) burns at the end of step 4; evacuation in step 4 still works (k - 1 rule)
    assert col.hazard_at((0, 0), 4) == 0.0 and col.hazard_at((0, 0), 5) == 1.0


# ----- pheromone mechanics ------------------------------------------------------------------------
def test_evaporation_without_deposits_decays_every_edge():
    cfg = {"deposit_q": 0.0, "elite_weight": 0.0, "iterations": 5}
    col = colony_of(obs_of(room(16, 8, 2, "stochastic")), cfg)
    col.colony()
    expected = max(0.1, 1.0 * (1 - 0.2) ** 5)
    assert col.unseen == pytest.approx(expected)
    assert col.pher and all(v == pytest.approx(expected) for v in col.pher.values())


def test_deposits_are_bounded_and_reinforce_found_routes():
    col = colony_of(obs_of(room(16, 8, 2, "stochastic")))
    stats = col.colony()
    cfg = col.cfg
    assert all(cfg["tau_min"] - 1e-12 <= v <= cfg["tau_max"] + 1e-12 for v in col.pher.values())
    assert stats["iterations_run"] == cfg["iterations"] and stats["ants"] > 0
    for pid, b in col.best.items():
        if b is None:
            continue
        edges = [(b.path[i], b.path[i + 1]) for i in range(len(b.path) - 1)]
        assert all(col.pher[e] > col.unseen for e in edges)       # best routes carry pheromone
    col.deposit(((0, 0), (0, 1)), 1e9)
    assert col.pher[((0, 0), (0, 1))] == cfg["tau_max"]


def test_route_costs_are_recomputed_from_current_tables():
    col = colony_of(obs_of(room(16, 20, 2, "stochastic")))
    col.colony()
    for pid, b in col.best.items():
        if b is not None:
            cost, hmax = col.score(pid, b.path)
            assert cost == pytest.approx(b.cost) and hmax == pytest.approx(b.hmax)
            assert len(b.path) - 1 <= col.lmax[pid]


# ----- dynamic replanning -------------------------------------------------------------------------
def _person(d, pid="Pa"):
    return d.diagnostics["people"][pid]


def test_burning_cell_on_the_nearest_route_changes_the_exit():
    free = decide(ascii_scenario(["0.....a.1"], interval=100))
    blocked = decide(ascii_scenario(["0.....a*1"], interval=100))
    assert (free.actions["Pa"], _person(free)["target_exit"]) == ("east", "E1")
    assert (blocked.actions["Pa"], _person(blocked)["target_exit"]) == ("west", "E0")


def test_burning_exit_is_not_a_destination():
    d = decide(ascii_scenario(["0.....a.1"], ignitions=((0, 8),), interval=100))
    assert d.actions["Pa"] == "west" and _person(d)["target_exit"] == "E0"
    assert d.diagnostics["safe_exits"] == ["E0"]


def test_certain_forecast_blockage_is_avoided_deterministic():
    s = ascii_scenario(FORECAST_ROWS, interval=1)
    assert T0ShortestPath().act(obs_of(s)) == {"Pa": "east"}   # nearest exit E1 burns before arrival
    d = decide(s)
    assert d.actions["Pa"] == "west" and _person(d)["target_exit"] == "E0"
    assert _person(d)["reason"] == "aco_route" and _person(d)["route_hazard_max"] == 0.0


def test_stochastic_risk_is_a_soft_cost():
    low = decide(stochastic(FORECAST_ROWS, p=0.3, interval=1))
    high = decide(stochastic(FORECAST_ROWS, p=0.9, interval=1))
    assert low.actions["Pa"] == "east" and _person(low)["target_exit"] == "E1"
    assert not _person(low)["route_unsafe"] and _person(low)["route_hazard_max"] > 0     # risky, still chosen
    assert high.actions["Pa"] == "west" and _person(high)["target_exit"] == "E0"


def test_queue_at_the_nearest_exit_spreads_people_over_exits():
    rows = ["...1...", ".......", "abcdefg", "hijklmn", ".......", ".......", "...2..."]
    s = ascii_scenario(rows, interval=100)
    t0_load = Counter(T0ShortestPath().decide(obs_of(s)).diagnostics["target_exit_load"])
    d = decide(s)
    load = d.diagnostics["target_exit_load"]
    assert set(t0_load) == {"E1"} and load.get("E2", 0) > 0 and load.get("E1", 0) > 0
    assert no_blocked(d)


def test_replans_each_step_as_people_and_fire_change():
    s = stochastic(["0.........", "..abcd....", "..efgh....", "......*...", ".........1"], p=0.8, interval=1)
    sim = Simulator(s)
    pol = T3DynamicACO()
    active = []
    while not sim.terminated:
        obs = sim.observe()
        d = pol.decide(obs)
        assert set(d.actions) == set(sim.active_ids())
        assert d.diagnostics == T3DynamicACO().decide(obs).diagnostics      # fresh policy, same decision
        active.append(len(d.actions))
        rec = sim.step(d.actions)
        for pid, mv in rec["movements"].items():
            assert d.diagnostics["people"][pid]["predicted_outcome"] == mv["outcome"]
            assert mv["outcome"] not in BLOCKED
    assert active == sorted(active, reverse=True) and active[-1] < active[0]


def test_realized_fire_spread_triggers_a_switch_to_another_exit():
    """Pa first heads to the near exit E1; when the realized fire makes E1 unsafe, T3 replans to E0."""
    base = stochastic(FORECAST_ROWS, p=0.3, interval=1).to_dict()
    switched = kept = 0
    for seed in range(1, 41):
        d = copy.deepcopy(base)
        d["seed"] = seed
        sim = Simulator(Scenario.from_dict(d))
        targets = []
        while not sim.terminated:
            dec = T3DynamicACO().decide(sim.observe())
            targets.append(dec.diagnostics["people"]["Pa"]["target_exit"])
            sim.step(dec.actions)
        assert targets[0] == "E1"
        switched += "E0" in targets
        kept += "E0" not in targets
    assert switched and kept                     # depends on the realized fire, not on a fixed plan


# ----- joint action coordination ---------------------------------------------------------------------
def _coordinate(rows, people_opts, enter_fire="blocked"):
    s = ascii_scenario(rows, interval=100, enter_fire=enter_fire)
    col = colony_of(obs_of(s))
    people = {pid: {"route_found": True, "route_length": 1} for pid in col.order}
    acts, coord = col.coordinate(people, {pid: people_opts[pid] for pid in col.order})
    return col, acts, coord, col.resolve(acts)


def test_swap_is_never_attempted():
    col, acts, _, res = _coordinate(["0ab.1"], {"Pa": ["east", "stay"], "Pb": ["west", "stay"]})
    assert not (acts["Pa"] == "east" and acts["Pb"] == "west")
    assert all(o in ("moved", "stayed") for o, _ in res.values())


def test_rotation_cycle_is_repaired_to_legal_moves():
    rows = ["ab.", "dc.", "..0"]
    opts = {"Pa": ["east", "stay"], "Pb": ["south", "stay"], "Pc": ["west", "stay"], "Pd": ["north", "stay"]}
    col, acts, coord, res = _coordinate(rows, opts)
    assert all(o in ("moved", "stayed") for o, _ in res.values())
    assert coord["repair_rounds"] >= 1


def test_lowest_id_precedence_on_a_contested_exit():
    d = decide(ascii_scenario(["#a#", "b0."], interval=100))
    assert d.actions == {"Pa": "south", "Pb": "stay"}
    assert d.diagnostics["predicted_outcomes"] == {"moved": 1, "stayed": 1}


def test_follow_chain_moves_a_queue_in_any_id_order():
    d = decide(ascii_scenario(["0dcba....."], interval=100))
    assert set(d.actions.values()) == {"west"}
    assert d.diagnostics["predicted_outcomes"] == {"moved": 4}


def test_narrow_corridor_with_many_people_has_no_blocked_moves():
    s = ascii_scenario(["#########", "0abcdefg1", "#########"], interval=100)
    sim = Simulator(s)
    while not sim.terminated:
        d = T3DynamicACO().decide(sim.observe())
        assert no_blocked(d)
        sim.step(d.actions)
    assert Counter(p["status"] for p in sim.private_state()["people"]) == {"evacuated": 7}


def test_people_with_no_viable_route_shelter_and_are_labeled():
    d = decide(ascii_scenario(["a#0", ".#."], interval=100))
    p = _person(d)
    assert p["route_found"] is False and p["reason"] == "no_route_shelter" and d.diagnostics["no_route"] == ["Pa"]
    s = ascii_scenario(["a...0", "####.", "####*"], interval=1)    # exit certain to burn before arrival
    d2 = decide(s)
    p2 = _person(d2)
    assert p2["route_found"] is False and p2["reason"] == "no_feasible_route_shelter" and not p2["fallback"]
    assert no_blocked(d) and no_blocked(d2)


@pytest.mark.parametrize("n,side", [(1, 24), (10, 24), (50, 32), (100, 32), (200, 48)])
def test_complete_valid_maps_for_n_1_to_200(n, side):
    s = room(side, n, 5, "stochastic")
    sim = Simulator(s)
    for _ in range(2):
        sim.step(T3DynamicACO().act(sim.observe()))
    obs = sim.observe()
    d = T3DynamicACO().decide(obs)
    assert set(d.actions) == set(sim.active_ids()) and no_blocked(d)
    assert d.deterministic_dict() == T3DynamicACO().decide(obs).deterministic_dict()
    assert d.diagnostics["status"] == "complete"


def test_closed_loop_random_episodes_are_legal_and_predicted():
    rnd = random.Random(7)
    steps = 0
    reasons: Counter = Counter()
    for _ in range(60):
        s = random_scenario(rnd)
        sim = Simulator(s)
        pol = T3DynamicACO()
        while not sim.terminated:
            d = pol.decide(sim.observe())
            assert set(d.actions) == set(sim.active_ids())
            reasons.update(p["reason"] for p in d.diagnostics["people"].values())
            rec = sim.step(d.actions)
            steps += 1
            for pid, mv in rec["movements"].items():
                assert mv["outcome"] in ("moved", "stayed")
                assert d.diagnostics["people"][pid]["predicted_outcome"] == mv["outcome"]
    assert steps > 300 and reasons["aco_route"] > 0.9 * sum(reasons.values()) - reasons["no_route_shelter"]


# ----- budgets and labeled fallback --------------------------------------------------------------------
def _bfs_first_move(obs: dict, pid: str) -> str:
    g = PublicGrid.from_observation(validate_observation(obs))
    field = exit_distance_field(g)
    pos = dict(g.active)[pid]
    d = field[pos][0]
    return next(a for a, (dr, dc) in MOVES if field.get((pos[0] + dr, pos[1] + dc), (None,))[0] == d - 1)


def test_zero_ant_budget_falls_back_to_labeled_bfs_descent():
    s = room(20, 12, 4, "stochastic")
    obs = obs_of(s)
    d = T3DynamicACO({"max_ant_steps": 0}).decide(obs)
    dg = d.diagnostics
    assert dg["status"] == "budget_limited" and dg["budget"]["limit_hit"] == "max_ant_steps"
    assert dg["colony"]["ants"] == 0 and dg["aco_route_people"] == 0
    routed = [pid for pid, p in dg["people"].items() if p["route_found"]]
    assert dg["fallback_people"] == sorted(routed) and routed
    for pid in routed:
        p = dg["people"][pid]
        assert p["reason"] == "fallback_bfs_descent" and p["fallback"] is True
        assert p["planned_first_move"] == _bfs_first_move(obs, pid)
    assert no_blocked(d)
    assert d.deterministic_dict() == T3DynamicACO({"max_ant_steps": 0}).decide(obs).deterministic_dict()


def test_cpu_budget_stops_the_colony_and_labels_fallbacks():
    d = T3DynamicACO({"cpu_budget_ms": 0.0}).decide(obs_of(room(32, 100, 4, "stochastic")))
    dg = d.diagnostics
    assert dg["status"] == "budget_limited" and dg["budget"]["limit_hit"] == "cpu_budget_ms"
    assert dg["colony"]["iterations_run"] == 1 and dg["fallback_people"]
    assert all(dg["people"][p]["reason"] == "fallback_bfs_descent" for p in dg["fallback_people"])
    assert no_blocked(d)


def test_t3_is_sometimes_worse_and_sometimes_better_than_t1_on_synthetic():
    """Closed-loop casualties on random synthetic scenarios: T3 is not uniformly better (or worse)."""
    rnd = random.Random(31)
    worse = better = 0
    for _ in range(150):
        s = random_scenario(rnd, model="stochastic")
        cas = []
        for pol in (T1CongestionAware(), T3DynamicACO()):
            sim = Simulator(s)
            while not sim.terminated:
                sim.step(pol.act(sim.observe()))
            cas.append(sum(p["status"] == "casualty" for p in sim.private_state()["people"]))
        worse += cas[1] > cas[0]
        better += cas[1] < cas[0]
        if worse and better:
            break
    assert worse and better
