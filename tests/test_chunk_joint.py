"""Chunked inference: rows keep every person once with the full state, joint maps fail closed, and the
resolver sees the complete map at every observation. Fake tokenizer (CPU); the real one is in the experiment."""
import json
from pathlib import Path

import pytest

from ember_v1 import chunk_joint as cj
from ember_v1.sim import Simulator

REPO = Path(__file__).resolve().parents[1]


def tok(text):
    return text.split()


@pytest.fixture(scope="module")
def scen():
    return cj.synthetic_scenario(23, seed=1, fire_model="stochastic", max_steps=30)


@pytest.fixture(scope="module")
def obs(scen):
    return Simulator(scen).observe()


def test_rows_keep_every_person_once_with_full_state(obs):
    rep = cj.check_rows(obs, 5, tok)
    assert rep["active"] == 23 and [r["people"] for r in rep["rows"]] == [5, 5, 5, 5, 3]
    assert sum(r["people"] for r in rep["rows"]) == rep["active"]
    assert len({r["state"] for r in rep["rows"]}) == 1          # same full state count in every row
    assert all(r["total"] < 16384 for r in rep["rows"])
    chunks = cj.chunk_ids(obs, 5)
    assert sorted(i for c in chunks for i in c) == sorted(p["id"] for p in obs["people"])


def test_row_over_the_limit_fails_closed(obs):
    rep = cj.check_rows(obs, 23, tok)
    with pytest.raises(cj.ChunkError, match="tokens"):
        cj.check_rows(obs, 23, tok, limit=rep["rows"][0]["total"])      # equal to the limit: no spare token
    assert cj.check_rows(obs, 23, tok, limit=rep["rows"][0]["total"] + 1)["rows"][0]["total"] < \
        rep["rows"][0]["total"] + 1


@pytest.mark.parametrize("k", [0, -1, 2.0])
def test_bad_k(obs, k):
    with pytest.raises((cj.ChunkError, ValueError)):
        cj.chunk_ids(obs, k)


def _preds(chunks, action="stay"):
    return [{i: action for i in c} for c in chunks]


def test_assemble_fails_closed(obs):
    chunks = cj.chunk_ids(obs, 10)
    ok = cj.assemble(obs, chunks, _preds(chunks))
    assert len(ok) == 23
    missing = _preds(chunks)
    missing[0].pop(chunks[0][0])
    extra = _preds(chunks)
    extra[0]["P999"] = "stay"
    dup = _preds(chunks)
    dup[1][chunks[0][0]] = "stay"
    bad = _preds(chunks)
    bad[2][chunks[2][0]] = "jump"
    for preds, msg in ((missing, "keys differ"), (extra, "keys differ"), (dup, "keys differ"), (bad, "not allowed")):
        with pytest.raises(cj.ChunkError, match=msg):
            cj.assemble(obs, chunks, preds)
    with pytest.raises(cj.ChunkError, match="one prediction per chunk"):
        cj.assemble(obs, chunks, _preds(chunks)[:-1])


def test_cross_chunk_conflict_is_counted():
    sc = cj.Scenario.from_dict({
        "schema_version": "1.0", "scenario_id": "two-movers", "grid": {"width": 5, "height": 3},
        "obstacles": [], "exits": [{"id": "E0", "position": [1, 4]}],
        "people": [{"id": "P000", "position": [0, 2]}, {"id": "P001", "position": [2, 2]}],
        "ignitions": [[0, 0]], "fire": {"model": "deterministic", "spread_interval": 9, "exits_flammable": False},
        "movement": {"conflict_resolution": "lowest_id_wins", "enter_fire": "blocked"},
        "termination": {"max_steps": 5}, "seed": 1})
    o = Simulator(sc).observe()
    chunks = [["P000"], ["P001"]]
    joint = cj.assemble(o, chunks, [{"P000": "south"}, {"P001": "north"}])   # both target (1, 2)
    res = cj.resolve(o, joint)
    assert res["P001"][0] == "blocked_conflict" and res["P000"][0] == "moved"
    assert cj.cross_chunk_conflicts(o, chunks, joint, res) == 1
    assert cj.cross_chunk_conflicts(o, [["P000", "P001"]], joint, res) == 0


def test_every_observation_is_checked(scen):
    r = cj.run_episode(scen, "t1_chunk_blind", 5, tok)
    assert r["complete"] and not r["errors"]
    assert r["decisions"] == r["steps"] and r["steps"] > 1
    assert r["evacuated"] + r["casualties"] + r["unresolved"] == 23


def test_unchunked_reference_has_no_action_difference(scen):
    r = cj.run_episode(scen, "t1_joint", 5, tok)
    assert r["action_differs_from_unchunked"] == 0 and r["blocked"] == r["ref_blocked"]


def test_episode_stops_and_records_an_over_limit_row(scen):
    r = cj.run_episode(scen, "t1_joint", 23, tok, limit=200)
    assert not r["complete"] and "tokens" in r["errors"][0]["error"] and r["steps"] == 0


def test_synthetic_scenarios_are_deterministic_and_not_from_any_split():
    a = cj.synthetic_scenario(200, seed=0, fire_model="deterministic")
    assert a.content_hash() == cj.synthetic_scenario(200, seed=0, fire_model="deterministic").content_hash()
    assert len(a.people) == 200
    hashes = set()
    for s in ("train", "dev", "calibration"):
        m = json.loads((REPO / f"data/splits/p1-02-v1/{s}.json").read_text(encoding="utf-8"))
        hashes |= {e["scenario_hash"] for e in m["scenarios"]}
    assert a.content_hash() not in hashes
