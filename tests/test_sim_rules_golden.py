"""Golden step-level parity: exits, fire, casualties, termination, RNG.

Expected values are hand-derived from EMBERBench public docs at commit 423143a:
  SC  docs/simulation-contract.md (phase order, exits, fire, randomness, trapping)
  FS  docs/v1-fairness-spec.md (R4-R10)
"""
from __future__ import annotations

import hashlib
import math
from fractions import Fraction

import pytest

from ember_v1.sim import SimulationError, Simulator, SplitMix64, stream_seed
from ember_v1.sim.fire import Terrain, edge_count, frontier, spread
from ember_v1.sim.rng import draws_between
from sim_helpers import scenario


def statuses(sim):
    return {p["id"]: (p["status"], tuple(p["position"])) for p in sim.observe()["people"]}


# ----- exits and evacuation ------------------------------------------------------------

def test_exit_queue_one_person_per_step_no_gaps():  # FS R4, SC exits
    sim = Simulator(scenario(["...E"], {"P3": (0, 0), "P2": (0, 1), "P1": (0, 2)}))
    evac = []
    while not sim.terminated:
        rec = sim.step({pid: "east" for pid in sim.active_ids()})
        evac.append(rec["evacuations"])
    assert evac == [{"P1": "E1"}, {"P2": "E1"}, {"P3": "E1"}]
    assert sim.termination_reason == "all_resolved"
    assert statuses(sim) == {"P1": ("evacuated", (0, 3)), "P2": ("evacuated", (0, 3)),
                             "P3": ("evacuated", (0, 3))}


def test_two_people_cannot_enter_one_exit_in_one_step():  # FS R4
    sim = Simulator(scenario([".E."], {"P1": (0, 0), "P2": (0, 2)}))
    rec = sim.step({"P1": "east", "P2": "west"})
    assert rec["evacuations"] == {"P1": "E1"}
    assert rec["movements"]["P2"]["outcome"] == "blocked_conflict"
    rec = sim.step({"P2": "west"})
    assert rec["evacuations"] == {"P2": "E1"} and sim.termination_reason == "all_resolved"


def test_two_adjacent_exits_pass_two_people_per_step():  # FS R4
    sim = Simulator(scenario(["EE", ".."], {"P1": (1, 0), "P2": (1, 1)}))
    assert sim.step({"P1": "north", "P2": "north"})["evacuations"] == {"P1": "E1", "P2": "E2"}


def test_entering_burning_exit_is_casualty_not_evacuation():  # FS R5, R6
    sim = Simulator(scenario(["..B"], {"P1": (0, 1)}, enter_fire="casualty"))
    rec = sim.step({"P1": "east"})
    assert rec["movements"]["P1"]["outcome"] == "moved"
    assert rec["evacuations"] == {} and rec["casualties"] == {"P1": "entered_fire"}


def test_burning_exit_blocked_policy_then_spread_casualty():  # FS R5, R6, R7
    sim = Simulator(scenario(["..B"], {"P1": (0, 1)}, enter_fire="blocked", interval=2))
    rec = sim.step({"P1": "east"})
    assert rec["movements"]["P1"]["outcome"] == "blocked_fire" and rec["casualties"] == {}
    rec = sim.step({"P1": "stay"})
    assert rec["ignitions"] == [[0, 1]] and rec["casualties"] == {"P1": "fire_spread"}


def test_exit_reached_in_the_step_fire_reaches_it_evacuates():  # SC phase order, FS R5
    sim = Simulator(scenario(["FE."], {"P1": (0, 2)}))
    rec = sim.step({"P1": "west"})
    assert rec["evacuations"] == {"P1": "E1"} and rec["ignitions"] == [[0, 1]]
    assert statuses(sim)["P1"] == ("evacuated", (0, 1))


def test_follow_chain_into_fire_under_casualty_policy():  # SC rule 3, FS R6
    sim = Simulator(scenario(["..F.E"], {"P1": (0, 0), "P2": (0, 1)}, interval=5))
    rec = sim.step({"P1": "east", "P2": "east"})
    assert rec["casualties"] == {"P2": "entered_fire"}
    assert statuses(sim) == {"P1": ("active", (0, 1)), "P2": ("casualty", (0, 2))}


# ----- fire timing and casualties ------------------------------------------------------

def test_stay_on_igniting_cell_dies_mover_survives():  # FS R7, SC phase 8
    sim = Simulator(scenario(["F..E", "F..."], {"P1": (0, 1), "P2": (1, 1)}))
    rec = sim.step({"P1": "stay", "P2": "east"})
    assert rec["ignitions"] == [[0, 1], [1, 1]]
    assert rec["casualties"] == {"P1": "fire_spread"}
    assert statuses(sim)["P2"] == ("active", (1, 2))


def test_spread_interval_and_no_burn_out():  # SC fire, FS R7, R8
    sim = Simulator(scenario(["F....E"], {"P1": (0, 4)}, interval=2, max_steps=4))
    burning = []
    while not sim.terminated:
        sim.step({"P1": "stay"})
        burning.append(sim.observe()["burning"])
    assert burning == [[[0, 0]], [[0, 0], [0, 1]], [[0, 0], [0, 1]], [[0, 0], [0, 1], [0, 2]]]
    assert sim.termination_reason == "max_steps"


def test_deterministic_frontier_walls_and_exit_flammability():  # SC fire
    walls = frozenset({(0, 1)})
    t = Terrain(3, 3, walls, frozenset({(1, 0)}), exits_flammable=False)
    assert frontier(t, frozenset({(0, 0)})) == []
    t2 = Terrain(3, 3, walls, frozenset({(1, 0)}), exits_flammable=True)
    assert frontier(t2, frozenset({(0, 0)})) == [((1, 0), ((0, 0),))]
    open3 = Terrain(3, 3, frozenset(), frozenset(), True)
    assert spread(open3, frozenset({(1, 1)}), 1, "deterministic", 1, None, SplitMix64(0)) == \
        frozenset({(0, 1), (1, 0), (1, 2), (2, 1)})


def test_stochastic_edge_draw_order_and_count():  # SC stochastic fire: frontier, draw order
    t = Terrain(3, 3, frozenset(), frozenset(), True)
    burning = frozenset({(0, 0), (0, 2)})
    # Candidates row-major: (0,1) [sources (0,0),(0,2)], (1,0) [(0,0)], (1,2) [(0,2)].
    assert frontier(t, burning) == [((0, 1), ((0, 0), (0, 2))), ((1, 0), ((0, 0),)), ((1, 2), ((0, 2),))]
    assert edge_count(t, burning) == 4

    class Scripted:
        def __init__(self, outs):
            self.outs, self.calls = list(outs), 0

        def bernoulli(self, p):
            self.calls += 1
            return self.outs.pop(0)

    for outs, expected in (([False, True, False, False], {(0, 1)}),
                           ([True, True, False, False], {(0, 1)}),
                           ([False, False, True, False], {(1, 0)}),
                           ([False, False, False, True], {(1, 2)})):
        rng = Scripted(outs)
        assert spread(t, burning, 1, "stochastic", 1, 0.5, rng) == frozenset(expected)
        assert rng.calls == 4  # every edge draws, also after an earlier edge of a candidate hit


@pytest.mark.parametrize("p,expected", [(0.0, frozenset()), (1.0, frozenset({(0, 1), (1, 0), (1, 2)}))])
def test_stochastic_p0_p1_and_draw_consumption(p, expected):  # SC draw-consumption rule
    t = Terrain(3, 3, frozenset(), frozenset(), True)
    burning = frozenset({(0, 0), (0, 2)})
    rng = SplitMix64(12345)
    assert spread(t, burning, 1, "stochastic", 1, p, rng) == expected
    assert draws_between(12345, rng.state) == 4
    rng2 = SplitMix64(12345)
    assert spread(t, burning, 1, "stochastic", 2, p, rng2) == frozenset() and rng2.state == 12345


def test_deterministic_model_never_draws():
    t = Terrain(3, 3, frozenset(), frozenset(), True)
    rng = SplitMix64(99)
    spread(t, frozenset({(1, 1)}), 1, "deterministic", 1, None, rng)
    assert rng.state == 99


# ----- termination ----------------------------------------------------------------------

def test_all_resolved_wins_over_max_steps():  # FS R10
    sim = Simulator(scenario([".E"], {"P1": (0, 0)}, max_steps=1))
    sim.step({"P1": "east"})
    assert sim.termination_reason == "all_resolved"
    with pytest.raises(SimulationError):
        sim.step({})


def test_max_steps_termination():  # FS R10
    sim = Simulator(scenario(["..E"], {"P1": (0, 0)}, max_steps=2))
    sim.step({"P1": "stay"})
    assert not sim.terminated
    sim.step({"P1": "stay"})
    assert sim.terminated and sim.termination_reason == "max_steps"


def test_trapped_is_private_and_ignores_people():  # FS R9
    sim = Simulator(scenario(["E#.", "#.."], {"P1": (0, 2), "P2": (1, 1)}))
    assert sim.private_state()["trapped"] == ["P1", "P2"]
    assert "trapped" not in sim.observe()
    open_sim = Simulator(scenario(["E..", "..."], {"P1": (0, 2), "P2": (0, 1)}))
    assert open_sim.private_state()["trapped"] == []  # P2 blocks P1's path, but people are ignored


# ----- randomness -------------------------------------------------------------------------

def test_splitmix64_reference_vector():
    # Published SplitMix64 reference outputs for state 1234567 (Vigna's splitmix64.c).
    rng = SplitMix64(1234567)
    assert [rng.next_u64() for _ in range(5)] == [
        6457827717110365317, 3203168211198807973, 9817491932198370423,
        4593380528125082431, 16408922859458223821]


def test_stream_seed_formula():  # SC randomness
    expected = int.from_bytes(hashlib.sha256(b"emberbench/v1/42/fire").digest()[:8], "big")
    assert stream_seed(42, "fire") == expected
    assert stream_seed(42, "fire") != stream_seed(43, "fire") != stream_seed(42, "other")
    for bad in (-1, 2 ** 64, True, 1.0):
        with pytest.raises(ValueError):
            stream_seed(bad, "fire")


def test_bernoulli_threshold_is_exact():  # SC: exactly ceil(p * 2**53) of 2**53 values succeed
    class Fixed(SplitMix64):
        def __init__(self, u53):
            super().__init__(0)
            self.u53 = u53

        def next_u64(self):
            return self.u53 << 11

    p = 0.3
    k = math.ceil(Fraction(p) * 2 ** 53)  # exact: Fraction(p) is the float's exact value
    assert Fixed(k - 1).bernoulli(p) is True
    assert Fixed(k).bernoulli(p) is False
    assert Fixed(0).bernoulli(0.0) is False and Fixed(2 ** 53 - 1).bernoulli(1.0) is True
    for bad in (True, -0.1, 1.5, "0.5", None):
        with pytest.raises(ValueError):
            Fixed(0).bernoulli(bad)


def test_bernoulli_consumes_exactly_one_draw():
    rng = SplitMix64(7)
    rng.bernoulli(0.0)
    rng.bernoulli(1.0)
    assert draws_between(7, rng.state) == 2
