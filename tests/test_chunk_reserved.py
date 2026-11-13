"""Reserved-chunk coordination study (P1-14 §4): sequential rows with reservations, bounded teacher-free
correction, exit allocation, raw vs corrected records, per-step token counts and paired summaries. CPU only."""
import json
from pathlib import Path

import pytest

from ember_v1 import chunk_joint as cj
from ember_v1 import token_budget as tb
from ember_v1.encoding import encode_observation
from ember_v1.sim import Simulator
from ember_v1.teacher import t1 as t1mod

REPO = Path(__file__).resolve().parents[1]


def tok(text):
    return text.split()


def _two_movers():
    sc = cj.Scenario.from_dict({
        "schema_version": "1.0", "scenario_id": "two-movers", "grid": {"width": 5, "height": 3},
        "obstacles": [], "exits": [{"id": "E0", "position": [1, 4]}],
        "people": [{"id": "P000", "position": [0, 2]}, {"id": "P001", "position": [2, 2]}],
        "ignitions": [[0, 0]], "fire": {"model": "deterministic", "spread_interval": 9, "exits_flammable": False},
        "movement": {"conflict_resolution": "lowest_id_wins", "enter_fire": "blocked"},
        "termination": {"max_steps": 5}, "seed": 1})
    return Simulator(sc).observe()


@pytest.fixture(scope="module")
def scen():
    return cj.synthetic_scenario(23, seed=10, fire_model="stochastic", max_steps=25, tag=cj.RESERVED_STUDY_VERSION)


@pytest.fixture(scope="module")
def obs(scen):
    return Simulator(scen).observe()


def test_reserved_rows_carry_earlier_decisions_only(obs):
    active = sorted(p["id"] for p in obs["people"] if p["status"] == "active")
    joint = {pid: ("east" if i % 2 else "stay") for i, pid in enumerate(active)}
    rows = tb.chunk_rows(obs, joint, k=5, group_id="g", source="synthetic", reservations=True)
    full = encode_observation(obs, mode="ascii")
    assert rows[0]["state"] == full and rows[0]["chunk"]["reserved_ids"] == []
    decided = {}
    for r in rows:
        assert r["state"] == full + tb.reservation_block(obs, decided)
        assert r["chunk"]["reserved_ids"] == sorted(decided)
        decided.update({i: joint[i] for i in r["chunk"]["ids"]})
    last = rows[-1]["state"][len(full):]
    entries = last[len(tb.RESERVATION_HEADER):].split(";")
    assert last.startswith(tb.RESERVATION_HEADER)
    assert len(entries) == sum(len(r["chunk"]["ids"]) for r in rows[:-1]) == len(rows[-1]["chunk"]["reserved_ids"])
    # unreserved rows are unchanged by the new parameters
    plain = tb.chunk_rows(obs, joint, k=5, group_id="g", source="synthetic")
    assert all(r["state"] == full for r in plain)


def test_reserved_target_of_a_blocked_move_is_the_current_cell():
    o = _two_movers()
    assert tb.reserved_target(o, "P000", "north") == (0, 2)          # boundary
    assert tb.reserved_target(o, "P000", "south") == (1, 2)
    assert tb.reserved_target(o, "P001", "stay") == (2, 2)


@pytest.mark.parametrize("chunks", [[["P000"], ["P000", "P001"]], [["P000"]], [["P000", "P001"], []]])
def test_custom_chunks_must_partition(chunks):
    o = _two_movers()
    with pytest.raises(ValueError):
        tb.chunk_rows(o, {"P000": "stay", "P001": "stay"}, k=2, group_id="g", source="s", chunks=chunks)
    with pytest.raises(ValueError):
        tb.chunk_rows(o, {"P000": "stay", "P001": "stay"}, k=1, group_id="g", source="s",
                      chunks=[["P000", "P001"]])


def test_reserved_predictor_respects_the_earlier_chunk():
    o = _two_movers()
    chunks = [["P000"], ["P001"]]
    raw = cj.assemble(o, chunks, cj.predictor("t1_chunk_reserved")(o, chunks))
    assert cj.reservation_conflicts(o, chunks, raw) == 0
    assert cj.cross_chunk_conflicts(o, chunks, raw, cj.resolve(o, raw)) == 0


def test_bounded_correction_is_teacher_free_and_keeps_the_raw_map():
    o = _two_movers()
    chunks = [["P000"], ["P001"]]
    raw = {"P000": "south", "P001": "north"}                            # both target (1, 2); P000 wins
    fixed, log = cj.correct_conflicts(o, chunks, raw, max_changes=1)
    assert raw == {"P000": "south", "P001": "north"}                    # raw is never modified
    assert fixed == {"P000": "south", "P001": "east"} and log[0]["cross_chunk"] is True
    assert not any(x in cj.BLOCKED for x, _ in cj.resolve(o, fixed).values())
    assert cj.correct_conflicts(o, chunks, raw, max_changes=0) == (raw, [])


def test_exit_allocation_respects_capacity_and_groups_chunks(obs):
    alloc = cj.exit_allocation(obs)
    assert alloc == cj.exit_allocation(obs)
    n = len(alloc)
    used = [e for e in alloc.values() if e is not None]
    cap = -(-n // len(set(used)))
    assert max(used.count(e) for e in set(used)) <= cap
    chunks = cj.exit_chunks(obs, 4)
    assert sorted(i for c in chunks for i in c) == sorted(alloc)
    assert all(len(c) <= 4 and len({alloc[i] for i in c}) == 1 for c in chunks)


def test_chunked_methods_never_call_the_full_planner(scen, monkeypatch):
    sizes = []
    real = t1mod.T1CongestionAware.act

    def spy(self, o):
        sizes.append(sum(p["status"] == "active" for p in o["people"]))
        return real(self, o)
    monkeypatch.setattr(t1mod.T1CongestionAware, "act", spy)
    for m in ("t1_chunk_reserved", "t1_chunk_reserved_fix", "t1_chunk_exitalloc"):
        sizes.clear()
        o = Simulator(scen).observe()
        chunks = cj.method_chunks(m, o, 5)
        cj.predictor(m)(o, chunks)
        assert len(chunks) > 1 and sizes and max(sizes) <= 5


@pytest.mark.parametrize("method", ["t1_chunk_reserved", "t1_chunk_reserved_fix", "t1_chunk_exitalloc"])
def test_episode_records_tokens_every_step_and_raw_separately(scen, method):
    r = cj.run_episode(scen, method, 5, tok)
    assert r["complete"] and r["reserved_rows"] and r["reservation_conflicts"] == 0
    assert len(r["step_row_tokens"]) == r["steps"] == len(r["step_decision_ms"]) == r["joint_map_valid_steps"]
    assert all(len(x) >= 2 for x in r["step_row_tokens"][:1])
    assert r["evacuated"] + r["casualties"] + r["unresolved"] == 23
    assert r["corrected"] is (method == "t1_chunk_reserved_fix")
    if not r["corrected"]:
        assert r["corrections_log"] == [] and r["raw_blocked"] == r["blocked"]
    for entry in r["corrections_log"]:
        for c in entry["changes"]:
            assert "reverted" in c or c["raw"] != c["corrected"]


def test_reserved_rows_over_the_limit_fail_closed(scen):
    r = cj.run_episode(scen, "t1_chunk_reserved", 5, tok, limit=150)
    assert not r["complete"] and "tokens" in r["errors"][0]["error"]


def test_paired_summary_reports_each_outcome_and_severe_regressions():
    def run(p, k, h, c, u, e):
        return {"predictor": p, "k": k, "n": 100, "scenario": h, "scenario_hash": h, "complete": True,
                "casualties": c, "unresolved": u, "evacuated": e, "cross_chunk_blocked": 0,
                "raw_cross_chunk_blocked": 0, "reservation_conflicts": 0, "max_row_tokens": 10}
    runs = [run("t1_joint", 100, "a", 10, 0, 90), run("t1_joint", 100, "b", 10, 0, 90),
            run("m", 50, "a", 8, 4, 88), run("m", 50, "b", 16, 0, 84),
            {**run("m", 50, "c", 0, 0, 0), "complete": False}]
    s = cj.paired_summary(runs)[0]
    assert (s["sum_delta_casualties"], s["sum_delta_unresolved"], s["sum_delta_cas_plus_unres"]) == (4, 4, 8)
    assert s["unresolved_increased"] == 1 and s["failed_closed"] == 1
    assert s["severe_regression_threshold"] == 5 and s["severe_regressions"] == ["b"]


def test_v1_study_scenarios_are_unchanged():
    old = json.loads((REPO / "data/training/p1-11-student@1/chunk_study.json").read_text(encoding="utf-8"))
    want = {r["scenario"]: r["scenario_hash"] for r in old["runs"]}
    sc = cj.synthetic_scenario(200, seed=0, fire_model="deterministic", max_steps=160)
    assert want[sc.scenario_id] == sc.content_hash()
    new = cj.synthetic_scenario(200, seed=0, fire_model="deterministic", max_steps=160, tag=cj.RESERVED_STUDY_VERSION)
    assert new.content_hash() not in set(want.values())
