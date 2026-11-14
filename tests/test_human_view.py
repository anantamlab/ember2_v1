"""Human point of view: what a person sees, remembers and plans; no information beyond that."""
import json
import random

import pytest

from ember_v1.contract import ACTIONS
from ember_v1.human_encoding import QUESTION_ID, encode_view, feature_size, view_features, view_row
from ember_v1.human_view import (HumanObserver, Memory, ViewConfig, admissible, plan, validate_view,
                                 visible_cells)
from ember_v1.rl_online_scenarios import build_scenarios
from ember_v1.shield import admissible_actions
from ember_v1.sim import Scenario, Simulator


def room(walls=(), exits=((2, 6),), people=((2, 1),), fire=((4, 0),), h=5, w=7, max_steps=20):
    return Scenario.from_dict({
        "schema_version": "1.0", "scenario_id": "human-test", "grid": {"height": h, "width": w},
        "obstacles": [list(p) for p in walls], "exits": [{"id": f"E{i}", "position": list(p)} for i, p in enumerate(exits)],
        "people": [{"id": f"P{i}", "position": list(p)} for i, p in enumerate(people)],
        "ignitions": [list(p) for p in fire],
        "fire": {"model": "deterministic", "spread_interval": 50, "exits_flammable": True},
        "movement": {"enter_fire": "blocked", "conflict_resolution": "lowest_id_wins"},
        "termination": {"max_steps": max_steps}, "seed": 7})


DIVIDER = tuple((r, 3) for r in (0, 1, 3, 4))      # a wall with one door at (2, 3)


def at(view, dr, dc):
    return view["window"][view["radius"] + dr][view["radius"] + dc]


def first_view(scenario, cfg=ViewConfig(), pid="P0"):
    return HumanObserver(cfg).observe(Simulator(scenario).observe())[pid]


def test_walls_hide_what_is_behind_them_and_the_edge_looks_like_a_wall():
    view = first_view(room(DIVIDER))
    validate_view(view)
    assert at(view, 0, 0) == "@" and at(view, 0, 2) == "."          # the door
    assert at(view, 0, 4) == "."                                    # seen through the door
    assert at(view, -2, 4) == "?"                                   # behind the divider
    assert at(view, -2, 2) == "#" and at(view, 0, -2) == "#"        # divider wall, outside the grid
    assert view["exits"] == []                                      # the exit at (2,6) is out of range
    assert view["moves"] == {"north": "floor", "south": "floor", "east": "floor", "west": "floor"}
    open_view = first_view(room(DIVIDER), ViewConfig(line_of_sight=False))
    assert at(open_view, -2, 4) == "."


def test_a_view_does_not_change_when_hidden_cells_change():
    base = first_view(room(DIVIDER))
    hidden_person = first_view(room(DIVIDER, people=((2, 1), (0, 5))))
    hidden_fire = first_view(room(DIVIDER, fire=((4, 0), (0, 5))))
    assert base == hidden_person == hidden_fire
    seen_person = first_view(room(DIVIDER, people=((2, 1), (2, 5))))
    assert at(seen_person, 0, 4) == "P" and seen_person != base


def test_a_view_holds_no_absolute_position_or_map_size():
    """The same surroundings at another place in a larger building give the same view."""
    small = first_view(room(DIVIDER), ViewConfig(radius=1))
    shifted = room(tuple((r + 3, c + 4) for r, c in DIVIDER), exits=((5, 13),), people=((5, 5),),
                   fire=((9, 0),), h=10, w=14)
    assert first_view(shifted, ViewConfig(radius=1)) == small
    text = encode_view(small)
    assert "P0" not in text and "SIZE" not in text and "@" in text


def test_the_eight_neighbours_are_always_seen():
    cells = visible_cells((5, 5), lambda p: True, 3)       # every cell is opaque
    assert {(5 + dr, 5 + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)} <= cells
    assert (5, 7) not in cells


def test_memory_keeps_seen_cells_and_no_memory_forgets_them():
    scenario = room(DIVIDER, exits=((2, 0),), people=((2, 4),))
    for cfg, expected in ((ViewConfig(radius=2), "E"), (ViewConfig(radius=2, memory=False), "?")):
        sim, observer = Simulator(scenario), HumanObserver(cfg)
        view = observer.observe(sim.observe())["P0"]
        assert view["exits"] == []
        for _ in range(2):                                  # walk west until the exit is in range
            record = sim.step({"P0": "west"})
            view = observer.observe(sim.observe(), record)["P0"]
        assert view["exits"] == [[0, -2]]
        for _ in range(2):                                  # walk back east
            record = sim.step({"P0": "east"})
            view = observer.observe(sim.observe(), record)["P0"]
        assert view["last_action"] == "east" and view["last_outcome"] == "moved"
        assert (view["exits"] == [[0, -4]]) is (expected == "E")
        assert view["goal"] == ("exit" if expected == "E" else "explore")


def test_exit_knowledge_gives_positions_only_as_prior():
    scenario = room(DIVIDER, exits=((2, 6), (0, 0)))
    assert first_view(scenario, ViewConfig(radius=1))["exits"] == []
    assert first_view(scenario, ViewConfig(radius=1, exit_knowledge="all"))["exits"] == [[-2, -1], [0, 5]]
    one = first_view(scenario, ViewConfig(radius=1, exit_knowledge="one"))
    assert len(one["exits"]) == 1 and one["goal"] == "exit"
    assert first_view(scenario, ViewConfig(radius=1, exit_knowledge="one")) == one      # stable choice


def test_last_blocked_move_reaches_the_view_and_resolved_people_leave():
    scenario = room(people=((2, 5), (0, 0)))
    sim, observer = Simulator(scenario), HumanObserver()
    observer.observe(sim.observe())
    record = sim.step({"P0": "east", "P1": "north"})
    views = observer.observe(sim.observe(), record)
    assert set(views) == {"P1"}                             # P0 evacuated
    assert views["P1"]["last_action"] == "north" and views["P1"]["last_outcome"] == "blocked_boundary"
    with pytest.raises(KeyError):
        observer.memory("P0")


def test_observer_refuses_gaps_and_foreign_records():
    scenario = room()
    sim, observer = Simulator(scenario), HumanObserver()
    observer.observe(sim.observe())
    record = sim.step({"P0": "stay"})
    sim.step({"P0": "stay"})
    with pytest.raises(ValueError):
        observer.observe(sim.observe(), record)             # a step was skipped
    with pytest.raises(ValueError):
        observer.observe(sim.observe())                     # no record
    observer.reset()
    assert set(observer.observe(sim.observe())) == {"P0"}   # a fresh start is allowed anywhere


def test_view_mask_equals_the_public_shield_on_real_episodes():
    rng = random.Random(3)
    for scenario, _ in build_scenarios("human-view-test", "train", 4, 10):
        sim, observer, record = Simulator(scenario), HumanObserver(ViewConfig(radius=2)), None
        while not sim.terminated:
            obs = sim.observe()
            views = observer.observe(obs, record)
            shield = admissible_actions(obs)
            assert {pid: admissible(v) for pid, v in views.items()} == shield
            for v in views.values():
                validate_view(v)
                assert len(view_features(v)) == feature_size(2)
            record = sim.step({pid: rng.choice(ACTIONS) for pid in views})


def test_views_are_deterministic():
    scenario = build_scenarios("human-view-test", "train", 1, 6)[0][0]

    def run():
        sim, observer, record, out = Simulator(scenario), HumanObserver(), None, []
        while not sim.terminated:
            views = observer.observe(sim.observe(), record)
            out.append(json.dumps(views, sort_keys=True))
            record = sim.step({pid: "north" for pid in views})
        return out
    assert run() == run()


def memory_of(rows):
    mem = Memory()
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            mem.see((r, c), ch)
    return mem


def test_plan_prefers_a_route_away_from_fire_within_the_detour_limit():
    mem = memory_of(["#######",
                     "#..F..#",
                     "#....E#",
                     "#.....#",
                     "#######"])
    careful, direct = plan(mem, (2, 1), fire_detour=6), plan(mem, (2, 1), fire_detour=0)
    assert careful["goal"] == direct["goal"] == "exit"
    assert careful["steps"]["east"] == 5 and careful["steps"]["south"] == 5      # round the fire
    assert direct["steps"]["east"] == 3 and direct["steps"]["north"] == 5        # straight past it
    assert direct["steps"]["west"] is None                                       # a wall


def test_plan_explores_then_flees_then_rests():
    closed = ["#####", "#...#", "#...#", "#####"]
    assert plan(memory_of(closed), (1, 1))["goal"] == "none"
    burning = plan(memory_of(["#####", "#..F#", "#...#", "#####"]), (1, 2))
    assert burning["goal"] == "flee"
    assert burning["steps"]["south"] < burning["steps"]["stay"] and burning["steps"]["east"] is None
    assert min(burning["steps"], key=lambda a: (burning["steps"][a] is None, burning["steps"][a] or 0)) != "stay"
    open_side = plan(memory_of(["####", "#...", "####"]), (1, 1))
    assert open_side["goal"] == "explore" and open_side["steps"]["east"] == 1
    told = memory_of(closed)
    told.told = {(1, 9)}                                    # a known exit behind unseen space
    assert plan(told, (1, 1))["goal"] == "none"             # walls are known: no way through
    gap = memory_of(["####", "#..", "####"])
    gap.told = {(1, 6)}
    assert plan(gap, (1, 1))["goal"] == "exit" and plan(gap, (1, 1))["steps"]["east"] == 4


def test_burning_exit_is_no_goal():
    view = first_view(room(exits=((2, 3),), fire=((2, 3),), people=((2, 1),)))
    assert view["exits"] == [] and view["goal"] != "exit" and at(view, 0, 2) == "F"


def test_encodings_and_validation():
    view = first_view(room(DIVIDER))
    row = view_row(view, "east")
    assert set(row) == {"state", "questions", "gold"} and row["gold"] == {QUESTION_ID: "east"}
    assert set(row["questions"][QUESTION_ID]["criteria"]) == set(ACTIONS)
    assert "ROUTE goal=explore" in row["state"] and "NEXT north=floor" in row["state"]
    blind = first_view(room(DIVIDER), ViewConfig(hints=False))
    assert blind["goal"] is None and "ROUTE" not in encode_view(blind)
    assert len(view_features(blind)) == len(view_features(view)) == feature_size(4)
    with pytest.raises(ValueError):
        view_row(view, "jump")
    for bad in ({**view, "seed": 1}, {**view, "window": view["window"][1:]}, {**view, "steps": None},
                {**view, "last_action": "jump"}, {k: v for k, v in view.items() if k != "moves"}):
        with pytest.raises(ValueError):
            validate_view(bad)
    for kwargs in ({"radius": 0}, {"radius": 16}, {"exit_knowledge": "some"}, {"fire_detour": -1}):
        with pytest.raises(ValueError):
            ViewConfig(**kwargs)
