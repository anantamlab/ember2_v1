"""Seeded property and differential tests for the simulator.

No extra dependency: each property runs over a fixed set of seeded random
scenarios (random.Random), so failures reproduce exactly.

The differential oracle formulates collision rules 2-4 as a least fixpoint
("a winner moves if its target is empty or its occupant moves"), which is a
different algorithm from the chain walk in movement.py.
"""
from __future__ import annotations

import random
import zlib

import pytest

from ember_v1.sim import ACTIONS, Scenario, Simulator, resolve_moves
from ember_v1.sim.fire import Terrain, edge_count
from ember_v1.sim.movement import DELTAS


def random_scenario(rnd: random.Random, *, fire: str | None = None, max_people: int = 12) -> Scenario:
    h, w = rnd.randint(2, 8), rnd.randint(2, 8)
    cells = [(r, c) for r in range(h) for c in range(w)]
    rnd.shuffle(cells)
    n_exits = rnd.randint(1, 3)
    exits, rest = cells[:n_exits], cells[n_exits:]
    n_walls = rnd.randint(0, len(rest) // 4)
    walls, rest = rest[:n_walls], rest[n_walls:]
    n_fire = rnd.randint(0, min(3, len(rest) - 1))
    ignitions, rest = rest[:n_fire], rest[n_fire:]
    n_people = rnd.randint(1, max(1, min(max_people, len(rest))))
    model = fire or rnd.choice(["deterministic", "stochastic"])
    flammable = rnd.random() < 0.5
    fire_cfg = {"model": model, "spread_interval": rnd.randint(1, 3), "exits_flammable": flammable}
    if model == "stochastic":
        fire_cfg["spread_probability"] = rnd.choice([0.0, 0.15, 0.5, 1.0])
    ids = rnd.sample(["P1", "P2", "P3", "P10", "P11", "P20", "A", "B", "b", "Z9", "p_1", "x-2", "P0", "P01"],
                     n_people)
    return Scenario.from_dict({
        "schema_version": "1.0", "scenario_id": "prop",
        "grid": {"width": w, "height": h}, "obstacles": [list(p) for p in walls],
        "exits": [{"id": f"E{i}", "position": list(p)} for i, p in enumerate(exits)],
        "people": [{"id": pid, "position": list(p)} for pid, p in zip(ids, rest)],
        "ignitions": [list(p) for p in ignitions], "fire": fire_cfg,
        "movement": {"conflict_resolution": "lowest_id_wins", "enter_fire": rnd.choice(["blocked", "casualty"])},
        "termination": {"max_steps": rnd.randint(1, 25)}, "seed": rnd.randrange(2 ** 64),
    })


def random_policy(seed: int):
    rnd = random.Random(seed)
    return lambda obs: {p["id"]: rnd.choice(ACTIONS) for p in obs["people"] if p["status"] == "active"}


def oracle(height, width, walls, burning, enter_fire, positions, actions, *, iterative_rule1=False):
    """Independent fixpoint formulation. Returns {id: outcome}."""
    out, target = {}, {}
    for pid, a in actions.items():
        r, c = positions[pid]
        t = (r + DELTAS[a][0], c + DELTAS[a][1])
        if a == "stay":
            out[pid] = "stayed"
        elif not (0 <= t[0] < height and 0 <= t[1] < width):
            out[pid] = "blocked_boundary"
        elif t in walls:
            out[pid] = "blocked_wall"
        elif t in burning and enter_fire == "blocked":
            out[pid] = "blocked_fire"
        else:
            target[pid] = t
    at = {p: pid for pid, p in positions.items()}
    nonmover = set(out)
    changed = True
    while changed:
        changed = False
        for pid, t in target.items():
            if pid not in out and at.get(t) in nonmover:
                out[pid] = "blocked_occupied"
                if iterative_rule1:
                    nonmover.add(pid)
                    changed = True
    live = {pid: t for pid, t in target.items() if pid not in out}
    winners = {}
    for t in set(live.values()):
        claim = sorted(pid for pid, x in live.items() if x == t)
        winners[claim[0]] = t
        out.update({pid: "blocked_conflict" for pid in claim[1:]})
    moved, grew = set(), True
    while grew:
        grew = False
        for pid, t in winners.items():
            if pid not in moved and (at.get(t) is None or at.get(t) in moved):
                moved.add(pid)
                grew = True
    for pid in winners:
        if pid in moved:
            out[pid] = "moved"
            continue
        cur, steps = at.get(winners[pid]), 0
        while cur in winners and cur != pid and steps <= len(winners):
            cur, steps = at.get(winners[cur]), steps + 1
        out[pid] = "blocked_cycle" if cur == pid else "blocked_occupied"
    return out


def random_step_inputs(rnd: random.Random):
    h, w = rnd.randint(1, 6), rnd.randint(1, 6)
    cells = [(r, c) for r in range(h) for c in range(w)]
    rnd.shuffle(cells)
    k = rnd.randint(1, len(cells))
    people_cells, others = cells[:k], cells[k:]
    walls = frozenset(c for c in others if rnd.random() < 0.2)
    burning = frozenset(c for c in others if c not in walls and rnd.random() < 0.2)
    ids = [f"P{rnd.randrange(40)}x{i}" for i in range(k)]
    positions = dict(zip(ids, people_cells))
    # Bias toward dense, conflicting moves.
    actions = {pid: rnd.choice(ACTIONS[1:] if rnd.random() < 0.85 else ACTIONS) for pid in ids}
    return h, w, walls, burning, rnd.choice(["blocked", "casualty"]), positions, actions


N_STEP_CASES = 3000


@pytest.mark.parametrize("chunk", range(6))
def test_resolver_matches_fixpoint_oracle(chunk):
    rnd = random.Random(1000 + chunk)
    for _ in range(N_STEP_CASES // 6):
        h, w, walls, burning, ef, pos, act = random_step_inputs(rnd)
        got = resolve_moves(height=h, width=w, walls=walls, burning=burning, enter_fire=ef,
                            positions=pos, actions=act)
        assert {k: v[0] for k, v in got.items()} == oracle(h, w, walls, burning, ef, pos, act)


def test_rule1_reading_does_not_change_positions():
    """U2: one-pass vs iterative collision rule 1 may relabel, but never moves anyone differently."""
    rnd = random.Random(77)
    relabelled = 0
    for _ in range(N_STEP_CASES):
        h, w, walls, burning, ef, pos, act = random_step_inputs(rnd)
        a = oracle(h, w, walls, burning, ef, pos, act)
        b = oracle(h, w, walls, burning, ef, pos, act, iterative_rule1=True)
        assert {k for k, v in a.items() if v == "moved"} == {k for k, v in b.items() if v == "moved"}
        relabelled += a != b
    assert relabelled > 0  # the labels do differ in some cases, so U2 is a real label ambiguity


def test_resolver_invariants_order_and_rename():
    rnd = random.Random(5)
    for _ in range(N_STEP_CASES):
        h, w, walls, burning, ef, pos, act = random_step_inputs(rnd)
        got = resolve_moves(height=h, width=w, walls=walls, burning=burning, enter_fire=ef,
                            positions=pos, actions=act)
        ends = [end for _, end in got.values()]
        assert len(ends) == len(set(ends))                               # no double occupancy
        for pid, (o, end) in got.items():
            r, c = pos[pid]
            dr, dc = DELTAS[act[pid]]
            assert end == ((r + dr, c + dc) if o == "moved" else (r, c))
            assert end not in walls and 0 <= end[0] < h and 0 <= end[1] < w
            if ef == "blocked" and o == "moved":
                assert end not in burning
            if o == "moved" and end in pos.values():                     # entered an occupied cell:
                occ = next(k for k, v in pos.items() if v == end)        # its occupant moved out
                assert got[occ][0] == "moved"
        keys = list(pos)
        rnd.shuffle(keys)
        shuffled = resolve_moves(height=h, width=w, walls=walls, burning=burning, enter_fire=ef,
                                 positions={k: pos[k] for k in keys}, actions={k: act[k] for k in reversed(keys)})
        assert shuffled == got                                            # submission order
        renamed = resolve_moves(height=h, width=w, walls=walls, burning=burning, enter_fire=ef,
                                positions={"Q" + k: v for k, v in pos.items()},
                                actions={"Q" + k: v for k, v in act.items()})
        assert {k[1:]: v for k, v in renamed.items()} == got             # order-preserving rename


def test_episode_invariants():
    rnd = random.Random(2024)
    for i in range(300):
        sc = random_scenario(rnd)
        sim = Simulator(sc)
        terrain = Terrain(sc.height, sc.width, frozenset(sc.obstacles),
                          frozenset(p for _, p in sc.exits), sc.exits_flammable)
        exits = {p for _, p in sc.exits}
        policy = random_policy(i)
        prev = sim.observe()
        while not sim.terminated:
            before = {tuple(x) for x in prev["burning"]}
            active = [p for p in prev["people"] if p["status"] == "active"]
            for p in active:                                            # start-of-step invariant
                assert tuple(p["position"]) not in exits and tuple(p["position"]) not in before
            rec = sim.step(policy(prev))
            obs = sim.observe()
            after = {tuple(x) for x in obs["burning"]}
            ignited = {tuple(x) for x in rec["ignitions"]}
            assert after == before | ignited and not ignited & before   # monotone, no burn-out
            assert all(terrain.flammable(c) for c in ignited)
            spread_step = rec["step"] % sc.spread_interval == 0
            expected_draws = edge_count(terrain, frozenset(before)) if spread_step and sc.fire_model == "stochastic" else 0
            assert rec["fire_draws"] == expected_draws
            cells = [tuple(p["position"]) for p in obs["people"] if p["status"] == "active"]
            assert len(cells) == len(set(cells))
            old = {p["id"]: p for p in prev["people"]}
            for p in obs["people"]:
                o = old[p["id"]]
                if o["status"] != "active":                              # resolved people are frozen
                    assert p == o
                else:
                    (r0, c0), (r1, c1) = o["position"], p["position"]
                    assert abs(r0 - r1) + abs(c0 - c1) <= 1
                    if p["status"] == "evacuated":
                        assert tuple(p["position"]) in exits and tuple(p["position"]) not in before
                    if p["status"] == "casualty":
                        assert tuple(p["position"]) in after
            prev = obs
        assert sim.termination_reason in ("all_resolved", "max_steps")


def test_fire_is_exogenous_common_random_numbers():
    """Same scenario and seed: every policy sees the same ignitions and RNG positions."""
    rnd = random.Random(31)
    for i in range(60):
        sc = random_scenario(rnd, fire="stochastic")
        runs = []
        for policy in (lambda o: {p["id"]: "stay" for p in o["people"] if p["status"] == "active"},
                       lambda o: {p["id"]: "east" for p in o["people"] if p["status"] == "active"},
                       random_policy(i)):
            sim = Simulator(sc)
            trace = []
            while not sim.terminated:
                rec = sim.step(policy(sim.observe()))
                trace.append((rec["ignitions"], rec["fire_draws"], sim.private_state()["rng"]))
            runs.append(trace)
        n = min(len(t) for t in runs)
        assert runs[0][:n] == runs[1][:n] == runs[2][:n]


def _keyed_policy(obs):
    """Deterministic in (step, id) only, so two runs with equal observations submit equal actions."""
    return {p["id"]: ACTIONS[zlib.crc32(f"{obs['step']}|{p['id']}".encode()) % 5]
            for p in obs["people"] if p["status"] == "active"}


def test_no_future_information_in_observations():
    """The observation before step t is a function of the history only.

    Two runs that differ only in the seed (so only in future fire draws) must give
    identical observations for as long as their burning histories agree. If an
    observation leaked a future draw, the runs would differ earlier.
    """
    rnd = random.Random(77)
    compared = diverged = 0
    for _ in range(150):
        sc = random_scenario(rnd, fire="stochastic")
        a, b = Simulator(sc, rnd.randrange(2 ** 64)), Simulator(sc, rnd.randrange(2 ** 64))
        hist_a = hist_b = sorted(list(p) for p in sc.ignitions)
        while not (a.terminated or b.terminated):
            oa, ob = a.observe(), b.observe()
            # burning shows only past ignitions: initial set plus the records of completed steps
            assert oa["burning"] == hist_a and ob["burning"] == hist_b
            if oa["burning"] != ob["burning"]:
                diverged += 1
                break
            assert oa == ob
            compared += 1
            ra, rb = a.step(_keyed_policy(oa)), b.step(_keyed_policy(ob))
            assert ra["actions"] == rb["actions"]
            hist_a, hist_b = sorted(hist_a + ra["ignitions"]), sorted(hist_b + rb["ignitions"])
    assert compared > 500 and diverged > 10            # both branches are exercised


def test_observe_is_pure():
    rnd = random.Random(5)
    for _ in range(40):
        sim = Simulator(random_scenario(rnd, fire="stochastic"))
        while not sim.terminated:
            state, chain = sim.private_state(), sim.chain_hash
            first = sim.observe()
            first["people"].clear()                     # a caller edit must not reach the simulator
            assert sim.observe() == sim.observe() and sim.private_state() == state and sim.chain_hash == chain
            sim.step(_keyed_policy(sim.observe()))


def test_fire_locality_people_conservation_and_termination():
    rnd = random.Random(909)
    for i in range(300):
        sc = random_scenario(rnd)
        sim = Simulator(sc)
        ids = sorted(pid for pid, _ in sc.people)
        policy = random_policy(i)
        while not sim.terminated:
            before = {tuple(x) for x in sim.observe()["burning"]}
            rec = sim.step(policy(sim.observe()))
            for r, c in rec["ignitions"]:                 # fire spreads only from a burning 4-neighbour
                assert {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)} & before
            obs = sim.observe()
            assert [p["id"] for p in obs["people"]] == ids   # nobody appears or disappears
            active = [p for p in obs["people"] if p["status"] == "active"]
            if not active:
                assert rec["termination_reason"] == "all_resolved"
            elif rec["step"] >= sc.max_steps:
                assert rec["termination_reason"] == "max_steps"
            else:
                assert rec["termination_reason"] is None and not sim.terminated
        assert rec["step"] <= sc.max_steps and set(rec["actions"]) <= set(ids)
