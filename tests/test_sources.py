"""Zip ingestion and source-quality policy (CPU only)."""
import json
import zipfile
from pathlib import Path

import pytest

from ember_v1.sim import Simulator
from ember_v1.sources import ingest_zip, training_records
from sim_helpers import scenario as sim_scenario
from test_sim_properties import random_policy

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _zip(tmp_path, members):
    path = tmp_path / "src.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for name, text in members.items():
            zf.writestr(name, text)
    return path


def test_scaffold_layout_yields_one_fixture_trace(tmp_path):
    path = _zip(tmp_path, {
        "pkg/examples/teacher_trace.json": (EXAMPLES / "teacher_trace.json").read_text(encoding="utf-8"),
        "pkg/examples/observation.json": (EXAMPLES / "observation.json").read_text(encoding="utf-8"),
        "pkg/src/code.py": "raise SystemExit('never executed')",
    })
    records, entry = ingest_zip(path, source_id="p0a-fixtures", quality="synthetic_fixture")
    assert len(records) == 1 and entry["decisions"] == 2
    assert entry["non_trace_json_members"] == 1 and entry["training_eligible"] is False
    assert records[0]["group_id"] == "p0a-fixtures/example-independent-layout-family-A"
    assert training_records(records) == []
    assert training_records(records, allow_fixtures=True) == records


def test_jsonl_and_list_members(tmp_path):
    trace = json.loads((EXAMPLES / "teacher_trace.json").read_text(encoding="utf-8"))
    path = _zip(tmp_path, {"a.jsonl": json.dumps(trace) + "\n" + json.dumps(trace) + "\n",
                           "b.json": json.dumps({"traces": [trace]})})
    records, entry = ingest_zip(path, source_id="s", quality="teacher_unverified")
    assert entry["traces"] == 3 and entry["training_eligible"] is False and training_records(records) == []


# ----- teacher_verified needs full-episode replay -------------------------------------------------

def _verified_traces(step_count: int = 3) -> list[dict]:
    sc = sim_scenario(["F.....", "......", ".#..E.", "......"], {"P2": (3, 0), "P10": (1, 5), "P1": (3, 5)},
                      fire="stochastic", p=0.4, seed=11)
    policy = random_policy(2)
    sim, observations, records = Simulator(sc), [], []
    while not sim.terminated:
        observations.append(sim.observe())
        records.append(sim.step(policy(observations[-1])))
    assert len(records) >= step_count
    episode = {"scenario": sc.to_dict(), "seed": None, "records": records}
    return [{"observation": observations[k], "actions": records[k]["actions"], "group_id": "ep-1",
             "source": "test-teacher", "episode": episode, "step": k + 1} for k in range(len(records))]


def test_teacher_verified_requires_and_records_full_replay(tmp_path):
    traces = _verified_traces()
    records, entry = ingest_zip(_zip(tmp_path, {"t.json": json.dumps({"traces": traces})}),
                                source_id="s", quality="teacher_verified")
    assert entry["training_eligible"] is True and len(records) == len(traces)
    final = traces[0]["episode"]["records"][-1]
    for k, r in enumerate(records):
        assert r["replay"]["status"] == "complete" and r["replay"]["step"] == k + 1
        assert r["replay"]["final_transition_hash"] == final["transition_hash"]
    assert training_records(records) == records


def _bad_partial(t):
    t["episode"]["records"] = t["episode"]["records"][:-1]


def _bad_extra(t):
    recs = t["episode"]["records"]
    recs.append(dict(recs[-1], step=len(recs) + 1))


def _bad_mid(t):
    t["episode"]["records"][1]["ignitions"] = [[3, 3]]


def _bad_reason(t):
    last = t["episode"]["records"][-1]
    last["termination_reason"] = "max_steps" if last["termination_reason"] == "all_resolved" else "all_resolved"


def _bad_hash(t):
    t["episode"]["records"][0]["transition_hash"] = "sha256:" + "0" * 64


def _bad_obs(t):
    t["observation"] = _verified_traces()[1]["observation"]


def _bad_actions(t):
    t["actions"] = _verified_traces()[1]["actions"]


@pytest.mark.parametrize("damage", [
    lambda t: t.pop("episode"), lambda t: t.pop("step"), lambda t: t.update(step=0), lambda t: t.update(step=99),
    lambda t: t.update(step=True), lambda t: t["episode"].pop("seed"), lambda t: t["episode"].update(seed=12),
    _bad_partial, _bad_extra, _bad_mid, _bad_reason, _bad_hash, _bad_obs, _bad_actions,
])
def test_teacher_verified_fails_closed_without_complete_replay(tmp_path, damage):
    trace = _verified_traces()[0]
    damage(trace)
    with pytest.raises(ValueError):
        ingest_zip(_zip(tmp_path, {"t.json": json.dumps(trace)}), source_id="s", quality="teacher_verified")


def test_training_records_rejects_teacher_verified_without_replay_evidence():
    rec = {"quality": "teacher_verified", "observation": {}, "actions": {}}
    with pytest.raises(ValueError):
        training_records([rec])
    with pytest.raises(ValueError):
        training_records([dict(rec, replay={"status": "partial"})])


@pytest.mark.parametrize("bad", ["missing_action", "privileged_field"])
def test_invalid_trace_fails_closed(tmp_path, bad):
    trace = json.loads((EXAMPLES / "teacher_trace.json").read_text(encoding="utf-8"))
    if bad == "missing_action":
        trace["actions"].pop("P2")
    else:
        trace["observation"]["seed"] = 7
    with pytest.raises(ValueError):
        ingest_zip(_zip(tmp_path, {"t.json": json.dumps(trace)}), source_id="s", quality="synthetic_fixture")


def test_bad_quality_or_source_id(tmp_path):
    path = _zip(tmp_path, {})
    with pytest.raises(ValueError):
        ingest_zip(path, source_id="s", quality="gold")
    with pytest.raises(ValueError):
        ingest_zip(path, source_id="a/b", quality="synthetic_fixture")
