"""T0 fire-aware shortest-path policy (issue #7): routing, tie-breaks, joint movement, leakage, determinism."""
from __future__ import annotations

import ast
import copy
import random
from collections import deque
from pathlib import Path

import pytest

from ember_v1.sim import Scenario, Simulator, resolve_moves
from ember_v1.teacher import (GRID_UTILS_VERSION, POLICY_INTERFACE_VERSION, GreedyNearestExitControl, RandomControl,
                              StayControl, T0ShortestPath)
from gen_helpers import random_scenario

NEVER_FROM_T0 = {"blocked_cycle", "blocked_wall", "blocked_boundary", "blocked_fire"}


def ascii_scenario(rows: list[str], *, ignitions: tuple[tuple[int, int], ...] = (), model: str = "deterministic",
                   interval: int = 100, flammable: bool = True, enter_fire: str = "casualty", max_steps: int = 50,
                   seed: int = 1, scenario_id: str = "t0-test") -> Scenario:
    """'#' wall, digit d -> exit E<d>, lowercase x -> person P<x>, '*' -> ignition, '.' floor."""
    walls, exits, people, ign = [], [], [], list(ignitions)
    for r, line in enumerate(rows):
        for c, ch in enumerate(line):
            if ch == "#":
                walls.append([r, c])
            elif ch.isdigit():
                exits.append({"id": f"E{ch}", "position": [r, c]})
            elif ch.islower():
                people.append({"id": f"P{ch}", "position": [r, c]})
            elif ch == "*":
                ign.append((r, c))
    fire = {"model": model, "spread_interval": interval, "exits_flammable": flammable}
    if model == "stochastic":
        fire["spread_probability"] = 0.5
    return Scenario.from_dict({
        "schema_version": "1.0", "scenario_id": scenario_id, "grid": {"width": len(rows[0]), "height": len(rows)},
        "obstacles": walls, "exits": exits, "people": people, "ignitions": [list(p) for p in ign], "fire": fire,
        "movement": {"conflict_resolution": "lowest_id_wins", "enter_fire": enter_fire},
        "termination": {"max_steps": max_steps}, "seed": seed})


def decide(rows: list[str], **kw):
    sim = Simulator(ascii_scenario(rows, **kw))
    return sim, T0ShortestPath().decide(sim.observe())


def check_joint(rows: list[str], expected: dict[str, str], **kw) -> dict:
    """T0's predicted joint outcome equals the simulator's resolved movement for the full action map."""
    sim, d = decide(rows, **kw)
    rec = sim.step(d.actions)
    actual = {pid: mv["outcome"] for pid, mv in rec["movements"].items()}
    predicted = {pid: p["predicted_outcome"] for pid, p in d.diagnostics["people"].items()}
    assert predicted == actual == expected
    return rec


# ----- interface ------------------------------------------------------------
def test_decision_carries_identity_diagnostics_runtime_and_reproducibility():
    _, d = decide(["0.a"])
    assert d.actions == {"Pa": "west"}
    assert d.teacher["name"] == "T0" and d.teacher["policy"] == "fire_aware_shortest_path"
    assert d.teacher["version"] == "1.0.0" and d.teacher["config_hash"].startswith("sha256:")
    assert d.reproducibility == {"interface": POLICY_INTERFACE_VERSION, "utilities": {"grid": GRID_UTILS_VERSION},
                                 "step": 0, "observation_digest": d.reproducibility["observation_digest"],
                                 "policy_seed": None, "deterministic": True}
    assert set(d.runtime) == {"wall_ms", "cpu_ms"} and d.runtime["wall_ms"] >= 0
    p = d.diagnostics["people"]["Pa"]
    assert p == {"action": "west", "route_found": True, "reason": "shortest_path", "target_exit": "E0",
                 "distance": 2, "equal_first_moves": 1, "predicted_outcome": "moved"}
    assert set(d.to_dict()) == {"actions", "teacher", "diagnostics", "reproducibility", "runtime"}


def test_t0_config_is_fixed_and_identities_are_distinct():
    with pytest.raises(ValueError):
        T0ShortestPath({"no_route_action": "north"})
    hashes = {p.identity.config_hash for p in (T0ShortestPath(), StayControl(), RandomControl(1), RandomControl(2),
                                               GreedyNearestExitControl())}
    assert len(hashes) == 4  # the random seed is policy-instance state, not configuration
    assert RandomControl(1).seed == 1 and T0ShortestPath().seed is None


# ----- routing --------------------------------------------------------------
def test_move_tie_break_is_north_south_east_west():
    _, d = decide(["..0", "...", "a.."])
    assert d.actions == {"Pa": "north"} and d.diagnostics["people"]["Pa"]["equal_first_moves"] == 2
    _, d = decide(["a..", "...", "..0"])
    assert d.actions == {"Pa": "south"}


def test_exit_tie_break_is_lowest_exit_id_not_position():
    _, d = decide(["0.a.1"])
    assert d.actions == {"Pa": "west"} and d.diagnostics["people"]["Pa"]["target_exit"] == "E0"
    _, d = decide(["1.a.0"])
    assert d.actions == {"Pa": "east"} and d.diagnostics["people"]["Pa"]["target_exit"] == "E0"
    _, d = decide(["0..a.1"])   # nearer exit wins over lower ID
    assert d.diagnostics["people"]["Pa"]["target_exit"] == "E1"


def test_routes_around_burning_cells():
    _, d = decide(["0....", ".***.", "..a.."])
    assert d.actions == {"Pa": "west"} and d.diagnostics["people"]["Pa"]["distance"] == 4


def test_burning_exit_is_not_a_target():
    _, d = decide(["0.a...1"], ignitions=((0, 0),))
    assert d.diagnostics["safe_exits"] == ["E1"]
    assert d.actions == {"Pa": "east"} and d.diagnostics["people"]["Pa"]["target_exit"] == "E1"


@pytest.mark.parametrize("rows,ign", [(["0.a"], ((0, 0),)),          # only exit burns
                                      (["0#a"], ()),                  # exit walled off
                                      (["0*a"], ()),                  # only path burns
                                      (["#0#", "###", ".a."], ())])   # exit sealed
def test_no_safe_exit_reachable_stays_and_is_recorded(rows, ign):
    _, d = decide(rows, ignitions=ign)
    assert d.actions == {"Pa": "stay"}
    assert d.diagnostics["no_route"] == ["Pa"]
    assert d.diagnostics["people"]["Pa"] == {"action": "stay", "route_found": False,
                                             "reason": "no_safe_exit_reachable", "target_exit": None,
                                             "distance": None, "equal_first_moves": 0, "predicted_outcome": "stayed"}


def test_exact_ids_inactive_people_are_excluded():
    sim = Simulator(ascii_scenario(["a0b...c"]))
    sim.step(T0ShortestPath().act(sim.observe()))       # Pa evacuates, Pb loses the conflict
    obs = sim.observe()
    assert [(p["id"], p["status"]) for p in obs["people"]] == [("Pa", "evacuated"), ("Pb", "active"),
                                                               ("Pc", "active")]
    assert T0ShortestPath().act(obs) == {"Pb": "west", "Pc": "west"}


# ----- joint movement (exact resolver) --------------------------------------
def test_two_people_same_target_lowest_id_wins():
    rec = check_joint(["a0b"], {"Pa": "moved", "Pb": "blocked_conflict"})
    assert rec["evacuations"] == {"Pa": "E0"}


def test_chain_behind_a_conflict_loser_is_blocked_occupied():
    check_joint(["a0bc"], {"Pa": "moved", "Pb": "blocked_conflict", "Pc": "blocked_occupied"})


@pytest.mark.parametrize("rows", [["0abc"], ["0cba"]])
def test_follow_chain_moves_in_either_id_order(rows):
    check_joint(rows, {"Pa": "moved", "Pb": "moved", "Pc": "moved"})


def test_multiple_exits_split_the_group():
    check_joint(["0ab1"], {"Pa": "moved", "Pb": "moved"})


def test_unreachable_person_stays_while_others_move():
    check_joint(["0a#b"], {"Pa": "moved", "Pb": "stayed"})


def test_bottleneck_queue_records_blocked_moves():
    rec = check_joint([".a.", "#.#", "b0c"], {"Pa": "moved", "Pb": "moved", "Pc": "blocked_conflict"})
    assert rec["evacuations"] == {"Pb": "E0"}


def test_resolver_blocks_swaps_and_rotations_and_t0_cannot_create_them():
    # A swap and a 4-rotation are blocked by the resolver (not T0 output: T0 moves strictly downhill).
    swap = resolve_moves(height=1, width=2, walls=frozenset(), burning=frozenset(), enter_fire="blocked",
                         positions={"A": (0, 0), "B": (0, 1)}, actions={"A": "east", "B": "west"})
    assert {o for o, _ in swap.values()} == {"blocked_cycle"}
    rot = resolve_moves(height=2, width=2, walls=frozenset(), burning=frozenset(), enter_fire="blocked",
                        positions={"A": (0, 0), "B": (0, 1), "C": (1, 1), "D": (1, 0)},
                        actions={"A": "east", "B": "south", "C": "west", "D": "north"})
    assert {o for o, _ in rot.values()} == {"blocked_cycle"}
    # Two people who would swap toward opposite exits do not: each goes to its own exit.
    check_joint(["0ba1"], {"Pa": "moved", "Pb": "moved"})


# ----- properties on random scenarios ---------------------------------------
def _bfs_from(start, h, w, walls, burning):
    dist = {start: 0}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for n in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= n[0] < h and 0 <= n[1] < w and n not in walls and n not in burning and n not in dist:
                dist[n] = dist[(r, c)] + 1
                q.append(n)
    return dist


def test_properties_on_random_closed_loop_episodes():
    rnd = random.Random(20261106)
    t0 = T0ShortestPath()
    deltas = {"north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1)}
    steps = 0
    for _ in range(300):
        s = random_scenario(rnd)
        sim = Simulator(s)
        walls = frozenset(s.obstacles)
        while not sim.terminated:
            obs = sim.observe()
            d = t0.decide(obs)
            active = {p["id"]: tuple(p["position"]) for p in obs["people"] if p["status"] == "active"}
            assert set(d.actions) == set(active) and set(d.actions.values()) <= set(obs["allowed_actions"])
            burning = frozenset(tuple(b) for b in obs["burning"])
            per_exit = {eid: _bfs_from(p, s.height, s.width, walls, burning) for eid, p in s.exits
                        if p not in burning}
            # evaluator-side oracle: T0's no-route set equals the simulator's private trapped set
            assert set(d.diagnostics["no_route"]) == set(sim.private_state()["trapped"])
            for pid, pos in active.items():
                info = d.diagnostics["people"][pid]
                best = min(((dist[pos], eid) for eid, dist in per_exit.items() if pos in dist), default=None)
                if best is None:
                    assert d.actions[pid] == "stay" and not info["route_found"]
                    continue
                assert (info["distance"], info["target_exit"]) == best
                dr, dc = deltas[d.actions[pid]]
                nxt = (pos[0] + dr, pos[1] + dc)
                assert per_exit[best[1]].get(nxt) == best[0] - 1      # one step closer to the target exit
            rec = sim.step(d.actions)
            outcomes = {pid: mv["outcome"] for pid, mv in rec["movements"].items()}
            assert outcomes == {pid: p["predicted_outcome"] for pid, p in d.diagnostics["people"].items()}
            assert not set(outcomes.values()) & NEVER_FROM_T0
            assert "entered_fire" not in rec["casualties"].values()
            steps += 1
    assert steps > 1000


# ----- observation-only and determinism -------------------------------------
PRIVATE_KEYS = ("seed", "rng_state", "scenario_id", "trapped", "difficulty", "label", "lineage_id", "family_id",
                "split", "future_burning", "teacher_outcome")


@pytest.mark.parametrize("policy", [T0ShortestPath(), StayControl(), RandomControl(3), GreedyNearestExitControl()],
                         ids=lambda p: p.identity.name)
@pytest.mark.parametrize("key", PRIVATE_KEYS)
def test_private_fields_are_rejected_not_ignored(policy, key):
    obs = Simulator(ascii_scenario(["0.a"])).observe()
    for where in ("root", "map", "rules", "person"):
        bad = copy.deepcopy(obs)
        {"root": bad, "map": bad["map"], "rules": bad["rules"], "person": bad["people"][0]}[where][key] = 1
        with pytest.raises(ValueError):
            policy.decide(bad)


def _episode_log(s: Scenario, policy) -> list[tuple[dict, dict]]:
    sim = Simulator(s)
    out = []
    while not sim.terminated:
        obs = sim.observe()
        d = policy.decide(obs)
        out.append((obs, d.deterministic_dict()))
        sim.step(d.actions)
    return out


def test_same_public_observation_same_decision_for_any_private_seed_or_lineage():
    rows = ["#########", "0..a..b.#", "#.###.#.#", "#c..*..d1", "#########"]
    logs = []
    for seed, sid in ((1, "lineage-A"), (2 ** 63 + 5, "lineage-B"), (77, "other-family")):
        logs.append(_episode_log(ascii_scenario(rows, interval=2, seed=seed, scenario_id=sid), T0ShortestPath()))
    assert logs[0] == logs[1] == logs[2]          # deterministic fire: same observations -> same decisions
    # stochastic fire: seeds change the trajectory; step 0 still matches and a fresh T0 repeats each decision
    st = [_episode_log(ascii_scenario(rows, model="stochastic", interval=1, seed=seed, scenario_id=f"s{seed}"),
                       T0ShortestPath()) for seed in (11, 12, 13, 14)]
    assert len({str(log[0]) for log in st}) == 1
    assert len({str(log) for log in st}) > 1
    for log in st:
        for obs, det in log:
            assert T0ShortestPath().decide(obs).deterministic_dict() == det


def test_random_control_is_a_function_of_seed_and_observation():
    obs = Simulator(ascii_scenario(["0.a.b.c"])).observe()
    assert RandomControl(5).act(obs) == RandomControl(5).act(copy.deepcopy(obs))
    assert len({str(RandomControl(k).act(obs)) for k in range(20)}) > 1
    with pytest.raises(ValueError):
        RandomControl(None)


def test_greedy_control_moves_toward_manhattan_nearest_safe_exit_and_stalls_at_walls():
    obs = Simulator(ascii_scenario(["0..a", "...."])).observe()
    assert GreedyNearestExitControl().act(obs) == {"Pa": "west"}
    obs = Simulator(ascii_scenario(["0#a", "..."])).observe()
    assert GreedyNearestExitControl().act(obs) == {"Pa": "stay"}   # no path search
    assert T0ShortestPath().act(obs) == {"Pa": "south"}


POLICY_MODULES = ("interface.py", "grid.py", "t0.py", "controls.py", "__init__.py")
FORBIDDEN = ("ember_v1.gen", "ember_v1.sim.simulator", "ember_v1.sim.scenario", "ember_v1.sim.rng", "ember_v1.sim.fire")


def test_policy_modules_do_not_import_simulator_internals_or_generator():
    root = Path(__file__).resolve().parents[1] / "src/ember_v1/teacher"
    for name in POLICY_MODULES:
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                full = {1: f"ember_v1.teacher.{node.module}", 2: f"ember_v1.{node.module}"}.get(node.level, node.module)
                assert not str(full).startswith(FORBIDDEN), (name, full)
            elif isinstance(node, ast.Import):
                assert not any(a.name.startswith(FORBIDDEN) for a in node.names), name
