"""Generator analysis against independent brute-force oracles and the simulator."""
from __future__ import annotations

import random
from collections import deque

import pytest

from ember_v1.gen import analysis as A
from ember_v1.sim import Scenario, Simulator
from gen_helpers import random_scenario

DELTAS = {"stay": (0, 0), "north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1)}


def test_fire_arrival_equals_simulator_spread_over_full_horizon():
    rnd = random.Random(11)
    for _ in range(400):
        s = random_scenario(rnd, h=rnd.randint(3, 12), w=rnd.randint(3, 12), max_steps=rnd.randint(1, 40))
        assert A.fire_arrival(s) == A.fire_arrival_reference(s)


def test_fire_arrival_equals_simulator_burning_sets():
    rnd = random.Random(12)
    for _ in range(60):
        s = random_scenario(rnd, max_steps=30)
        arrival = A.fire_arrival(s)
        sim = Simulator(s)
        while not sim.terminated:
            sim.step({pid: "stay" for pid in sim.active_ids()})
            t = sim.observe()["step"]
            assert sorted(map(list, (p for p, a in arrival.items() if a <= t))) == sim.observe()["burning"]


def _lone_escape_bfs(s, arrival, start):
    """Time-expanded BFS over (cell, t); returns an action list that evacuates, or None."""
    walls, exits = set(s.obstacles), {p for _, p in s.exits}
    F = lambda p: arrival.get(p, float("inf"))  # noqa: E731
    if F(start) <= 0:
        return None
    parent = {(start, 0): None}
    queue = deque([(start, 0)])
    while queue:
        a, t0 = queue.popleft()
        t = t0 + 1
        if t > s.max_steps:
            continue
        for act, (dr, dc) in DELTAS.items():
            b = (a[0] + dr, a[1] + dc)
            if not (0 <= b[0] < s.height and 0 <= b[1] < s.width) or b in walls or not t - 1 < F(b):
                continue
            if b in exits:
                plan, node = [act], (a, t0)
                while parent[node] is not None:
                    prev, act2 = parent[node]
                    plan.append(act2)
                    node = prev
                return plan[::-1]
            if t < F(b) and (b, t) not in parent:
                parent[(b, t)] = ((a, t0), act)
                queue.append((b, t))
    return None


def test_latest_departure_matches_time_expanded_bfs_and_simulator():
    rnd = random.Random(13)
    checked = escaped = 0
    for _ in range(250):
        s = random_scenario(rnd)
        arrival = A.fire_arrival(s)
        L = A.latest_departure(s, arrival)
        exits = {p for _, p in s.exits}
        for r in range(s.height):
            for c in range(s.width):
                p = (r, c)
                if p in s.obstacles or p in exits:
                    continue
                plan = _lone_escape_bfs(s, arrival, p)
                assert (plan is not None) == (p in L and L[p] >= 0), (s.to_json(), p)
                checked += 1
        # positive direction in the real simulator: a lone person follows the plan and evacuates
        for pid, p in s.people:
            plan = _lone_escape_bfs(s, arrival, p)
            if plan is None:
                continue
            solo = s.to_dict()
            solo["people"] = [{"id": pid, "position": list(p)}]
            sim = Simulator(Scenario.from_dict(solo))
            for act in plan:
                assert not sim.terminated
                rec = sim.step({pid: act})
            assert pid in rec["evacuations"] and sim.terminated
            escaped += 1
    assert checked > 3000 and escaped > 100


def _reach(s, burning, removed=None):
    exits = {p for _, p in s.exits}
    live = exits - set(burning)
    blocked = set(s.obstacles) | set(burning) | exits | ({removed} if removed else set())
    start = [n for e in live for n in A.neighbours4(e, s.height, s.width) if n not in blocked]
    return set(A.bfs(s.height, s.width, blocked, start))


def test_route_graph_cut_cells_match_brute_force():
    rnd = random.Random(14)
    for _ in range(300):
        s = random_scenario(rnd, h=rnd.randint(3, 8), w=rnd.randint(3, 8))
        burning = s.ignitions
        g = A.RouteGraph(s.height, s.width, s.obstacles, [p for _, p in s.exits], burning)
        base = _reach(s, burning)
        brute = sorted(v for v in base if (base - {v}) - _reach(s, burning, v))
        assert g.cut_cells() == brute
        people = [p for _, p in s.people]
        cuts, dependent = g.bottlenecks_for(people)
        assert cuts == sorted(v for v in base if any(q in base and q != v and q not in _reach(s, burning, v)
                                                    for q in people))
        assert dependent == {q for q in people if q in base
                             and any(q not in _reach(s, burning, v) for v in base if v != q)}


def test_topology_hash_invariant_under_symmetries_and_exit_order():
    rnd = random.Random(15)
    for _ in range(50):
        s = random_scenario(rnd, h=rnd.randint(3, 9), w=rnd.randint(3, 9))
        walls, exits = list(s.obstacles), [p for _, p in s.exits]
        ref = A.topology_hash(s.height, s.width, walls, exits)
        for dims, f in A._transforms(s.height, s.width):
            assert A.topology_hash(*dims, [f(*p) for p in walls], [f(*p) for p in reversed(exits)]) == ref
        if walls:
            assert A.topology_hash(s.height, s.width, walls[1:], exits) != ref


def test_coarse_signature_commutes_with_symmetries():
    rnd = random.Random(16)
    for _ in range(30):
        h, w = rnd.randint(3, 40), rnd.randint(3, 40)
        walls = {(rnd.randrange(h), rnd.randrange(w)) for _ in range(rnd.randint(0, h * w // 2))}
        mats = A._sym(A.coarse(h, w, walls))
        for k, (dims, f) in enumerate(A._transforms(h, w)):
            got = A.coarse(*dims, [f(*p) for p in walls])
            # _transforms and _sym enumerate the group in different orders: compare as sets of matrices
            assert any(all(abs(x - y) < 1e-9 for ra, rb in zip(got, m) for x, y in zip(ra, rb)) for m in mats)
        total = sum(map(sum, A.coarse(h, w, walls))) / (A.SIG_K ** 2)
        assert total == pytest.approx(len(walls) / (h * w))


def test_near_duplicate_detects_symmetries_and_small_edits_but_not_different_maps():
    rnd = random.Random(17)
    h, w = 20, 30
    walls = {(rnd.randrange(h), rnd.randrange(w)) for _ in range(120)}
    exits = [(0, 5), (h - 1, 20)]
    base = A.MapSignature(h, w, walls, exits)
    for dims, f in A._transforms(h, w):
        assert A.near_duplicate(base, A.MapSignature(*dims, [f(*p) for p in walls], [f(*p) for p in exits]))[0]
    edited = set(walls)
    for p in list(walls)[:6]:
        edited.discard(p)
    assert A.near_duplicate(base, A.MapSignature(h, w, edited, [(0, 6), (h - 1, 21)]))[0]
    other = {(rnd.randrange(h), rnd.randrange(w)) for _ in range(120)}
    assert not A.near_duplicate(base, A.MapSignature(h, w, other, exits))[0]
    assert not A.near_duplicate(base, A.MapSignature(h, w, walls, [(10, 0), (10, w - 1)]))[0]
    assert not A.near_duplicate(base, A.MapSignature(2 * h, 2 * w, walls, exits))[0]
