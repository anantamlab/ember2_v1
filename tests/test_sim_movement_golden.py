"""Golden movement parity matrix (phases 3-5).

Every expected outcome is derived by hand from the public text, not from this
simulator. Sources (EMBERBench commit 423143a):
  SC  docs/simulation-contract.md, "Phase order" phase 3 and "Collisions" rules 1-4
  FS  docs/v1-fairness-spec.md, rules R1-R4 and R6
Cases marked U2 depend on the literal one-pass reading of collision rule 1;
their final positions do not (see test_sim_properties).
"""
from __future__ import annotations

import pytest

from ember_v1.sim.movement import DELTAS
from sim_helpers import resolve

# (case id, source, rows, people, actions, enter_fire, expected outcomes)
CASES = [
    ("stay", "SC phase 3", ["..."], {"P1": (0, 1)}, {"P1": "stay"}, "blocked", {"P1": "stayed"}),
    ("boundary", "SC phase 3", ["..."], {"P1": (0, 1)}, {"P1": "north"}, "blocked",
     {"P1": "blocked_boundary"}),
    ("boundary_west_edge", "SC phase 3", ["..."], {"P1": (0, 0)}, {"P1": "west"}, "blocked",
     {"P1": "blocked_boundary"}),
    ("wall", "SC phase 3", [".#."], {"P1": (0, 0)}, {"P1": "east"}, "blocked", {"P1": "blocked_wall"}),
    ("fire_blocked", "SC phase 3; FS R6", [".F."], {"P1": (0, 0)}, {"P1": "east"}, "blocked",
     {"P1": "blocked_fire"}),
    ("fire_casualty_moves", "SC phase 3; FS R6", [".F."], {"P1": (0, 0)}, {"P1": "east"}, "casualty",
     {"P1": "moved"}),
    ("occupied_by_stayer", "SC rule 1", ["..."], {"P1": (0, 0), "P2": (0, 1)},
     {"P1": "east", "P2": "stay"}, "blocked", {"P1": "blocked_occupied", "P2": "stayed"}),
    ("occupied_by_phase3_blocked", "SC rule 1", ["..."], {"P1": (0, 0), "P2": (0, 1)},
     {"P1": "east", "P2": "north"}, "blocked", {"P1": "blocked_occupied", "P2": "blocked_boundary"}),
    ("contest_by_two_on_stayer_is_occupied", "SC rule 1 before rule 2", ["...", "..."],
     {"P1": (0, 0), "P2": (0, 2), "P3": (0, 1)}, {"P1": "east", "P2": "west", "P3": "stay"}, "blocked",
     {"P1": "blocked_occupied", "P2": "blocked_occupied", "P3": "stayed"}),
    ("contest_lowest_id_wins", "SC rule 2; FS R1", ["..."], {"P1": (0, 0), "P2": (0, 2)},
     {"P1": "east", "P2": "west"}, "blocked", {"P1": "moved", "P2": "blocked_conflict"}),
    ("contest_ids_compare_as_strings", "SC coordinates; FS R1", ["..."], {"P10": (0, 0), "P2": (0, 2)},
     {"P10": "east", "P2": "west"}, "blocked", {"P10": "moved", "P2": "blocked_conflict"}),
    ("contest_ids_compare_as_strings_mirrored", "FS R1", ["..."], {"P2": (0, 0), "P10": (0, 2)},
     {"P2": "east", "P10": "west"}, "blocked", {"P10": "moved", "P2": "blocked_conflict"}),
    ("contest_four_way", "SC rule 2", ["...", "...", "..."],
     {"Pd": (0, 1), "Pc": (1, 0), "Pb": (1, 2), "Pa": (2, 1)},
     {"Pd": "south", "Pc": "east", "Pb": "west", "Pa": "north"}, "blocked",
     {"Pa": "moved", "Pb": "blocked_conflict", "Pc": "blocked_conflict", "Pd": "blocked_conflict"}),
    ("follow_chain_a_b_dot_E", "SC rule 3 example `a b . E`", ["...E"], {"Pa": (0, 0), "Pb": (0, 1)},
     {"Pa": "east", "Pb": "east"}, "blocked", {"Pa": "moved", "Pb": "moved"}),
    ("follow_chain_higher_id_leads", "SC rule 3; order independence", ["...."],
     {"P1": (0, 0), "P2": (0, 1), "P3": (0, 2)}, {"P1": "east", "P2": "east", "P3": "east"}, "blocked",
     {"P1": "moved", "P2": "moved", "P3": "moved"}),
    ("follow_chain_turning", "SC rule 3", ["..", ".."], {"P1": (1, 0), "P2": (0, 0)},
     {"P1": "north", "P2": "east"}, "blocked", {"P1": "moved", "P2": "moved"}),
    ("chain_ends_at_stayer", "SC rules 1 and 3", ["...."], {"P1": (0, 0), "P2": (0, 1), "P3": (0, 2)},
     {"P1": "east", "P2": "east", "P3": "stay"}, "blocked",
     {"P1": "blocked_occupied", "P2": "blocked_occupied", "P3": "stayed"}),
    ("chain_ends_at_wall_block", "SC rules 1 and 3", ["..#"], {"P1": (0, 0), "P2": (0, 1)},
     {"P1": "east", "P2": "east"}, "blocked", {"P1": "blocked_occupied", "P2": "blocked_wall"}),
    ("chain_ends_at_fire_block", "SC rules 1 and 3; FS R6", ["..F"], {"P1": (0, 0), "P2": (0, 1)},
     {"P1": "east", "P2": "east"}, "blocked", {"P1": "blocked_occupied", "P2": "blocked_fire"}),
    ("chain_into_fire_under_casualty", "SC rule 3; FS R6", ["..F"], {"P1": (0, 0), "P2": (0, 1)},
     {"P1": "east", "P2": "east"}, "casualty", {"P1": "moved", "P2": "moved"}),
    ("chain_ends_at_conflict_loser", "SC rules 2 and 3", ["...", "...", "..."],
     {"P3": (0, 0), "P2": (0, 1), "P1": (2, 1)}, {"P3": "east", "P2": "south", "P1": "north"}, "blocked",
     {"P1": "moved", "P2": "blocked_conflict", "P3": "blocked_occupied"}),
    ("loser_not_reawarded_when_winner_fails", "SC rule 2; FS R1", ["...", "...", "..."],
     {"P0": (0, 2), "P3": (1, 1), "P1": (1, 0), "P2": (2, 1)},
     {"P0": "south", "P3": "east", "P1": "east", "P2": "north"}, "blocked",
     {"P0": "moved", "P3": "blocked_conflict", "P1": "blocked_occupied", "P2": "blocked_conflict"}),
    ("loser_not_reawarded_U2", "SC rules 1-3 (U2 literal reading)", ["...", "..."],
     {"P1": (0, 0), "P3": (0, 1), "P2": (0, 2), "P4": (1, 1)},
     {"P1": "east", "P3": "south", "P2": "west", "P4": "stay"}, "blocked",
     {"P1": "blocked_occupied", "P2": "blocked_conflict", "P3": "blocked_occupied", "P4": "stayed"}),
    ("swap_blocked", "SC rule 4; FS R2", [".."], {"P1": (0, 0), "P2": (0, 1)},
     {"P1": "east", "P2": "west"}, "blocked", {"P1": "blocked_cycle", "P2": "blocked_cycle"}),
    ("swap_vertical_blocked", "SC rule 4; FS R2", [".", "."], {"P2": (0, 0), "P1": (1, 0)},
     {"P2": "south", "P1": "north"}, "blocked", {"P1": "blocked_cycle", "P2": "blocked_cycle"}),
    ("rotation_4_blocked", "SC rule 4; FS R3", ["..", ".."],
     {"P1": (0, 0), "P2": (0, 1), "P3": (1, 1), "P4": (1, 0)},
     {"P1": "east", "P2": "south", "P3": "west", "P4": "north"}, "blocked",
     {"P1": "blocked_cycle", "P2": "blocked_cycle", "P3": "blocked_cycle", "P4": "blocked_cycle"}),
    ("rotation_6_blocked", "SC rule 4; FS R3", ["...", "..."],
     {"Pa": (0, 0), "Pb": (0, 1), "Pc": (0, 2), "Pd": (1, 2), "Pe": (1, 1), "Pf": (1, 0)},
     {"Pa": "east", "Pb": "east", "Pc": "south", "Pd": "west", "Pe": "west", "Pf": "north"}, "blocked",
     {k: "blocked_cycle" for k in ("Pa", "Pb", "Pc", "Pd", "Pe", "Pf")}),
    ("lower_id_outsider_breaks_rotation", "FS R3", ["..", "..", ".."],
     {"P1": (0, 1), "P2": (1, 0), "P3": (1, 1), "P4": (2, 1), "P5": (2, 0)},
     {"P1": "south", "P2": "east", "P3": "south", "P4": "west", "P5": "north"}, "blocked",
     {"P1": "blocked_occupied", "P2": "blocked_conflict", "P3": "blocked_occupied",
      "P4": "blocked_occupied", "P5": "blocked_occupied"}),
    ("higher_id_outsider_loses_rotation_stays_blocked", "FS R1, R3", ["..", "..", ".."],
     {"P9": (0, 1), "P2": (1, 0), "P3": (1, 1), "P4": (2, 1), "P5": (2, 0)},
     {"P9": "south", "P2": "east", "P3": "south", "P4": "west", "P5": "north"}, "blocked",
     {"P9": "blocked_conflict", "P2": "blocked_cycle", "P3": "blocked_cycle",
      "P4": "blocked_cycle", "P5": "blocked_cycle"}),
    # P1 and P3 both target (0,1): P1 wins, so the P2/P3 swap never forms.
    ("lower_id_outsider_breaks_swap", "SC rules 2-4; FS R3", ["..."],
     {"P1": (0, 0), "P2": (0, 1), "P3": (0, 2)}, {"P1": "east", "P2": "east", "P3": "west"}, "blocked",
     {"P1": "blocked_occupied", "P2": "blocked_occupied", "P3": "blocked_conflict"}),
    ("higher_id_outsider_loses_swap_stays_blocked", "SC rules 2-4; FS R2", ["..."],
     {"P9": (0, 0), "P2": (0, 1), "P3": (0, 2)}, {"P9": "east", "P2": "east", "P3": "west"}, "blocked",
     {"P9": "blocked_conflict", "P2": "blocked_cycle", "P3": "blocked_cycle"}),
    ("exit_is_normal_cell_conflict", "SC exits; FS R4", [".E."], {"P1": (0, 0), "P2": (0, 2)},
     {"P1": "east", "P2": "west"}, "blocked", {"P1": "moved", "P2": "blocked_conflict"}),
    ("exit_queue_flows", "SC exits; FS R4", ["..E"], {"P2": (0, 0), "P1": (0, 1)},
     {"P2": "east", "P1": "east"}, "blocked", {"P1": "moved", "P2": "moved"}),
]


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_golden_movement(case):
    _, _, rows, people, actions, enter_fire, expected = case
    outcomes, ends = resolve(rows, people, actions, enter_fire)
    assert outcomes == expected
    for pid, (r, c) in people.items():
        dr, dc = DELTAS[actions[pid]]
        assert ends[pid] == ((r + dr, c + dc) if expected[pid] == "moved" else (r, c))


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_golden_movement_is_independent_of_submission_order(case):
    _, _, rows, people, actions, enter_fire, expected = case
    reversed_people = dict(reversed(list(people.items())))
    reversed_actions = dict(reversed(list(actions.items())))
    assert resolve(rows, reversed_people, reversed_actions, enter_fire)[0] == expected


def test_golden_matrix_covers_every_movement_outcome():
    from ember_v1.sim import OUTCOMES
    seen = {o for case in CASES for o in case[6].values()}
    assert seen == set(OUTCOMES)


def test_resolver_rejects_bad_inputs():
    with pytest.raises(ValueError):
        resolve(["..."], {"P1": (0, 0)}, {"P1": "up"})
    with pytest.raises(ValueError):
        resolve(["..."], {"P1": (0, 0)}, {})
    with pytest.raises(ValueError):
        resolve(["..."], {"P1": (0, 0)}, {"P1": "stay"}, enter_fire="maybe")
    with pytest.raises(ValueError):
        resolve(["..."], {"P1": (0, 0), "P2": (0, 0)}, {"P1": "stay", "P2": "stay"})
