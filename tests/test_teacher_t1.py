"""T1 congestion-aware cooperative planner (issue #8): joint movement, congestion, fire, leakage, determinism.

Predicted-vs-actual checks compare T1's resolver prediction with the simulator's
movement outcome. Both use the same public resolver (`sim.movement.resolve_moves`),
so these are internal consistency checks, not independent evidence.
"""
from __future__ import annotations

import ast
import copy
import random
from collections import Counter
from pathlib import Path

import pytest

from ember_v1.contract import validate_observation
from ember_v1.sim import Simulator
from ember_v1.sim.bench import room
from ember_v1.teacher import T0ShortestPath
from ember_v1.teacher.t1 import INF, T1CongestionAware, _Decision
from gen_helpers import random_scenario
from test_teacher_t0 import PRIVATE_KEYS, ascii_scenario

BLOCKED = {"blocked_occupied", "blocked_conflict", "blocked_cycle", "blocked_wall", "blocked_boundary", "blocked_fire"}


def first(rows, **kw):
    sim = Simulator(ascii_scenario(rows, **kw))
    return sim, T1CongestionAware().decide(sim.observe())


def closed_loop(rows, policy, **kw):
    """Run to termination; check T1's per-step prediction against the simulator. Returns (statuses, steps, outcomes)."""
    sim = Simulator(ascii_scenario(rows, **kw))
    outcomes: Counter = Counter()
    while not sim.terminated:
        d = policy.decide(sim.observe())
        rec = sim.step(d.actions)
        for pid, mv in rec["movements"].items():
            outcomes[mv["outcome"]] += 1
            assert d.diagnostics["people"][pid]["predicted_outcome"] == mv["outcome"]
    st = Counter(p["status"] for p in sim.private_state()["people"])
    return st, sim.private_state()["step"], outcomes


# ----- identity and configuration -------------------------------------------
def test_identity_version_and_config_hash():
    t1 = T1CongestionAware()
    ident = t1.identity
    assert (ident.name, ident.policy, ident.version) == ("T1", "congestion_aware_cooperative_planner", "1.0.0")
    assert ident.config_hash == T1CongestionAware().identity.config_hash
    assert ident.config_hash != T0ShortestPath().identity.config_hash
    assert ident.config["planning_order"] == "ascending_person_id"
    assert T1CongestionAware({"reservation_horizon": 4}).identity.config_hash != ident.config_hash
    d = t1.decide(Simulator(ascii_scenario(["0.a"])).observe())
    assert d.teacher["name"] == "T1" and d.teacher["config_hash"] == ident.config_hash
    assert d.reproducibility["policy_seed"] is None and d.reproducibility["deterministic"] is True


@pytest.mark.parametrize("cfg", [{"planning_order": "random"}, {"no_route_mode": "stay"}, {"unknown": 1},
                                 {"reservation_horizon": 0}, {"reservation_horizon": 2.0}, {"risk_hard": 1.5},
                                 {"move_extra_cost": 1.0}, {"risk_weight": float("nan")}])
def test_invalid_or_descriptive_config_is_rejected(cfg):
    with pytest.raises(ValueError):
        T1CongestionAware(cfg)


# ----- joint movement ---------------------------------------------------------
def test_two_people_competing_for_one_cell_lowest_id_moves_other_waits():
    sim, d = first(["#0#", "a.b"])
    assert d.actions == {"Pa": "east", "Pb": "stay"}
    rec = sim.step(d.actions)
    assert {p: m["outcome"] for p, m in rec["movements"].items()} == {"Pa": "moved", "Pb": "stayed"}
    t0 = T0ShortestPath().decide(Simulator(ascii_scenario(["#0#", "a.b"])).observe())
    assert t0.diagnostics["people"]["Pb"]["predicted_outcome"] == "blocked_conflict"


def test_exit_capacity_is_one_person_per_exit_per_step():
    sim, d = first(["a0b"])
    assert d.actions == {"Pa": "east", "Pb": "stay"}
    st, steps, out = closed_loop(["a0b"], T1CongestionAware())
    assert st["evacuated"] == 2 and steps == 2 and not set(out) & BLOCKED


@pytest.mark.parametrize("row", ["0abcdefghij", "0jihgfedcba", "0aibhcgdfej"])
def test_long_follow_chains_move_in_any_id_order(row):
    sim, d = first([row])
    assert set(d.actions.values()) == {"west"}
    rec = sim.step(d.actions)
    assert all(m["outcome"] == "moved" for m in rec["movements"].values())
    st, steps, out = closed_loop([row], T1CongestionAware())
    assert st["evacuated"] == 10 and steps == 10 and not set(out) & BLOCKED


def test_chain_behind_a_waiting_person_waits_instead_of_bumping():
    # Pa (lowest ID) holds the exit approach this step; the chain behind Pb must not attempt blocked moves
    sim, d = first(["#####", "a0bcd", "#####"])
    rec = sim.step(d.actions)
    assert d.actions["Pa"] == "east" and d.actions["Pb"] == "stay"
    assert not {m["outcome"] for m in rec["movements"].values()} & BLOCKED


def test_repair_roots_swap_rotation_and_stayer():
    obs = validate_observation(Simulator(ascii_scenario(["0...", ".ab.", ".dc.", "...."])).observe())
    dec = _Decision(T1CongestionAware().identity.config, obs)
    w = dec.g.width
    at = {pid: i for pid, i in dec.pos.items()}
    # 4-rotation: a -> b's cell -> c's cell -> d's cell -> a's cell
    dec.plans = {"Pa": [at["Pa"], at["Pb"]], "Pb": [at["Pb"], at["Pc"]], "Pc": [at["Pc"], at["Pd"]],
                 "Pd": [at["Pd"], at["Pa"]]}
    acts = dec.actions()
    res = dec.resolve(acts)
    assert {o for o, _ in res.values()} == {"blocked_cycle"}
    assert dec.repair_roots(acts, res) == ["Pd"]          # highest ID of the cycle is re-planned
    # direct swap a <-> b, and c queued behind a stayer d
    dec.plans = {"Pa": [at["Pa"], at["Pb"]], "Pb": [at["Pb"], at["Pa"]], "Pc": [at["Pc"], at["Pd"]],
                 "Pd": [at["Pd"], at["Pd"]]}
    acts = dec.actions()
    res = dec.resolve(acts)
    assert res["Pa"][0] == res["Pb"][0] == "blocked_cycle" and res["Pc"][0] == "blocked_occupied"
    assert dec.repair_roots(acts, res) == ["Pb", "Pc"]
    assert w == 4


class _ForcedTargets(_Decision):
    """Test-only: force each person's target exit (opposing flows that nearest-exit routing never creates)."""
    forced: dict[str, str] = {}

    def assign_exits(self, fields):
        super().assign_exits(fields)
        for pid, eid in self.forced.items():
            if pid in self.info and self.info[pid]["route_found"]:
                self.info[pid]["target_exit"] = eid


class _ForcedT1(T1CongestionAware):
    """Forced targets plus a prohibitive margin, so nobody leaves through the other exit instead."""

    def __init__(self, forced):
        super().__init__({"alt_exit_margin": 1000.0})
        self.forced = forced

    def _plan(self, obs):
        dec = _ForcedTargets(self.identity.config, obs)
        dec.forced = self.forced
        return dec.run()


@pytest.mark.parametrize("bay", [True, False])
def test_narrow_corridor_opposing_movement_never_swaps_or_livelocks(bay):
    rows = ["###.###" if bay else "#######", "0..b.a1", "#######"]
    pol = _ForcedT1({"Pa": "E0", "Pb": "E1"})               # Pa must pass Pb in a one-wide corridor
    sim = Simulator(ascii_scenario(rows, max_steps=30))
    pa_moves, evac, out = [], {}, Counter()
    while not sim.terminated:
        d = pol.decide(sim.observe())
        rec = sim.step(d.actions)
        out.update(m["outcome"] for m in rec["movements"].values())
        if "Pa" in d.actions:
            pa_moves.append(d.actions["Pa"])
        evac.update(rec["evacuations"])
    assert not set(out) & BLOCKED                            # no swap attempt, no cycle, no conflict
    assert set(pa_moves) == {"west"}                         # lower ID keeps precedence, no oscillation
    if bay:
        assert evac == {"Pa": "E0", "Pb": "E1"} and sim.private_state()["step"] <= 8   # Pb waits in the bay
    else:
        assert evac == {"Pa": "E0", "Pb": "E0"}               # no bay: Pb yields back out through E0


def test_resolver_never_reports_a_blocked_t1_move_on_random_closed_loop_episodes():
    """Full action-set validity on random episodes; predicted == simulator outcome (internal consistency)."""
    rnd = random.Random(20261107)
    t1 = T1CongestionAware()
    steps = 0
    for _ in range(250):
        s = random_scenario(rnd)
        sim = Simulator(s)
        while not sim.terminated:
            obs = sim.observe()
            d = t1.decide(obs)
            active = {p["id"] for p in obs["people"] if p["status"] == "active"}
            assert set(d.actions) == active and set(d.actions.values()) <= set(obs["allowed_actions"])
            assert set(d.diagnostics["people"]) == active
            assert set(d.diagnostics["no_route"]) == set(sim.private_state()["trapped"])   # evaluator-side oracle
            rec = sim.step(d.actions)
            for pid, mv in rec["movements"].items():
                assert mv["outcome"] == d.diagnostics["people"][pid]["predicted_outcome"]
                assert mv["outcome"] in ("moved", "stayed")
            assert "entered_fire" not in rec["casualties"].values()
            steps += 1
    assert steps > 1000


# ----- congestion, exits and fire ---------------------------------------------
def test_multiple_exits_with_uneven_queues_congestion_rerouting():
    rows = ["####0####", "#abcdefg#", "#hijklmn#", "#.......#", "####1####"]
    _, d = first(rows)
    people = d.diagnostics["people"]
    assert all(p["nearest_exit"] == "E0" for p in people.values())
    rerouted = sorted(pid for pid, p in people.items() if p["target_exit"] == "E1")
    assert rerouted and all(people[pid]["nominal_delay"] == 0 or people[pid]["exit_cost"] > 0 for pid in rerouted)
    assert d.diagnostics["target_exit_load"]["E1"] == len(rerouted) >= 3
    st1, steps1, out1 = closed_loop(rows, T1CongestionAware())
    st0, steps0, out0 = closed_loop(rows, T0ShortestPath())
    assert st1["evacuated"] == st0["evacuated"] == 14
    assert steps1 < steps0 and not set(out1) & BLOCKED and out0["blocked_conflict"] > 0


def test_exit_that_becomes_unsafe_before_arrival_is_not_chosen():
    rows = ["0...a......1", "..##########", "*.##########"]      # fire reaches E0 at step 2, Pa needs 4
    _, d = first(rows, interval=1)
    p = d.diagnostics["people"]["Pa"]
    assert (p["nearest_exit"], p["target_exit"], p["fire_feasible"]) == ("E0", "E1", True)
    assert d.actions["Pa"] == "east"
    st1, steps1, _ = closed_loop(rows, T1CongestionAware(), interval=1)
    st0, steps0, _ = closed_loop(rows, T0ShortestPath(), interval=1)
    assert st1["evacuated"] == st0["evacuated"] == 1 and steps1 == 7 < steps0


def test_burning_exit_is_not_a_target_and_cells_about_to_ignite_are_avoided():
    _, d = first(["0#....", "...a.1", "*....."], interval=1)  # E0 path runs next to the fire
    assert d.diagnostics["people"]["Pa"]["target_exit"] == "E1"
    sim, d = first(["0a*"], interval=1, flammable=True, ignitions=((0, 0),))
    assert d.diagnostics["safe_exits"] == [] and d.diagnostics["no_route"] == ["Pa"]


def test_deterministic_forecast_is_exact_for_the_public_rule():
    rnd = random.Random(31)
    cfg = T1CongestionAware().identity.config
    checked = 0
    for _ in range(80):
        s = random_scenario(rnd, model="deterministic", max_steps=30)
        sim = Simulator(s)
        dec = _Decision(cfg, validate_observation(sim.observe()))
        w = s.width
        futures = []
        while not sim.terminated and len(futures) < 6:
            sim.step({pid: "stay" for pid in sim.active_ids()})
            futures.append({tuple(b) for b in sim.observe()["burning"]})
        for tau, burning in enumerate(futures, start=1):
            for i in range(dec.g.area):
                if dec.g.open_or_burning[i]:
                    assert dec.fire.prob_burning(i, tau) == float(divmod(i, w) in burning)
                    checked += 1
    assert checked > 1000


def test_stochastic_frontier_probability_uses_burning_neighbour_count():
    obs = validate_observation(Simulator(ascii_scenario(["*.*", "...", "0.a"], model="stochastic",
                                                        interval=1)).observe())
    dec = _Decision(T1CongestionAware().identity.config, obs)
    top_mid = 1                                            # (0,1) touches two burning cells
    assert dec.fire.prob_burning(top_mid, 1) == pytest.approx(1 - 0.5 ** 2)
    assert dec.fire.prob_burning(4, 1) == 0.0             # (1,1) is layer 2: no ignition in one step
    assert dec.fire.prob_burning(4, 2) == pytest.approx(0.25)   # P(Bin(2, .5) >= 2), single-front lower bound


def test_initially_unreachable_person_shelters_away_from_fire_without_a_fake_route():
    rows = ["0#######", "########", "#*...a.#", "########"]
    sim, d = first(rows, interval=2)
    p = d.diagnostics["people"]["Pa"]
    assert (p["route_found"], p["mode"], p["reason"], p["target_exit"]) == (False, "shelter", "no_safe_exit_reachable", None)
    assert d.actions["Pa"] == "east" and d.diagnostics["no_route"] == ["Pa"]
    assert sim.private_state()["trapped"] == ["Pa"]
    assert T0ShortestPath().act(sim.observe()) == {"Pa": "stay"}


def test_unreachable_person_does_not_disturb_others():
    rows = ["0..a..", "######", "b.*..."]
    _, d = first(rows, interval=50)
    assert d.diagnostics["no_route"] == ["Pb"] and d.actions["Pa"] == "west"


# ----- scale, budgets, determinism, leakage -------------------------------------
@pytest.mark.parametrize("n", [1, 10, 50, 100, 200])
def test_crowd_sizes_full_valid_action_maps_and_repeatable(n):
    sim = Simulator(room(32, n, 3))
    t1 = T1CongestionAware()
    for _ in range(3):
        if sim.terminated:
            break
        obs = sim.observe()
        d = t1.decide(obs)
        assert len(d.actions) == sum(p["status"] == "active" for p in obs["people"])
        assert T1CongestionAware().decide(copy.deepcopy(obs)).deterministic_dict() == d.deterministic_dict()
        rec = sim.step(d.actions)
        assert all(m["outcome"] in ("moved", "stayed") for m in rec["movements"].values())


def test_budgets_are_enforced_and_fallbacks_stay_valid():
    sim = Simulator(room(32, 100, 4))
    obs = sim.observe()
    zero = T1CongestionAware({"node_budget_total": 0}).decide(obs)
    assert zero.diagnostics["budget_exhausted"] and zero.diagnostics["expansions"] == 0
    assert set(zero.diagnostics["plan_status"]) <= {"budget_fallback", "no_successor_least_hazard"}
    tiny = T1CongestionAware({"node_budget_per_person": 2}).decide(obs)
    assert tiny.diagnostics["plan_status"].get("budget_partial", 0) > 0
    for d in (zero, tiny):
        rec = Simulator(room(32, 100, 4)).step(d.actions)
        assert all(rec["movements"][p]["outcome"] == d.diagnostics["people"][p]["predicted_outcome"] for p in d.actions)
        assert not {m["outcome"] for m in rec["movements"].values()} & BLOCKED
    full = T1CongestionAware().decide(obs)
    assert full.diagnostics["expansions"] <= full.teacher["config"]["node_budget_total"]


def test_same_public_observation_same_decision_for_any_private_seed_or_lineage():
    rows = ["#########", "0..a..b.#", "#.###.#.#", "#c..*..d1", "#########"]
    logs = []
    for seed, sid in ((1, "lineage-A"), (2 ** 63 + 5, "lineage-B"), (77, "other-family")):
        sim = Simulator(ascii_scenario(rows, interval=2, seed=seed, scenario_id=sid))
        log = []
        while not sim.terminated:
            obs = sim.observe()
            d = T1CongestionAware().decide(obs)
            log.append((obs, d.deterministic_dict()))
            sim.step(d.actions)
        logs.append(log)
    assert logs[0] == logs[1] == logs[2]
    obs = logs[0][0][0]
    shuffled = copy.deepcopy(obs)
    shuffled["people"].reverse()
    shuffled["map"]["exits"].reverse()
    assert T1CongestionAware().decide(shuffled).deterministic_dict() == logs[0][0][1]


@pytest.mark.parametrize("key", PRIVATE_KEYS)
def test_private_fields_are_rejected_not_ignored(key):
    obs = Simulator(ascii_scenario(["0.a"])).observe()
    for where in ("root", "map", "rules", "person"):
        bad = copy.deepcopy(obs)
        {"root": bad, "map": bad["map"], "rules": bad["rules"], "person": bad["people"][0]}[where][key] = 1
        with pytest.raises(ValueError):
            T1CongestionAware().decide(bad)


@pytest.mark.parametrize("mutate", [lambda o: o.update(allowed_actions=["stay"]),
                                    lambda o: o["rules"].update(conflict_resolution="random"),
                                    lambda o: o["people"].append(dict(o["people"][0])),
                                    lambda o: o.update(step=-1)])
def test_invalid_observations_are_rejected(mutate):
    obs = Simulator(ascii_scenario(["0.a"])).observe()
    mutate(obs)
    with pytest.raises(ValueError):
        T1CongestionAware().decide(obs)


FORBIDDEN = ("ember_v1.gen", "ember_v1.sim.simulator", "ember_v1.sim.scenario", "ember_v1.sim.rng", "ember_v1.sim.fire")


def test_t1_module_does_not_import_simulator_internals_or_generator():
    tree = ast.parse((Path(__file__).resolve().parents[1] / "src/ember_v1/teacher/t1.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            full = {1: f"ember_v1.teacher.{node.module}", 2: f"ember_v1.{node.module}"}.get(node.level, node.module)
            assert not str(full).startswith(FORBIDDEN), full
            assert full not in ("ember_v1.teacher.t0", "ember_v1.teacher.evaluate", "ember_v1.teacher.t1_eval"), full
        elif isinstance(node, ast.Import):
            assert not any(a.name.startswith(FORBIDDEN) for a in node.names)


def test_inf_marks_unreachable_cells():
    obs = validate_observation(Simulator(ascii_scenario(["0#a"])).observe())
    dec = _Decision(T1CongestionAware().identity.config, obs)
    assert dec.exit_fields()["E0"][2] == INF
