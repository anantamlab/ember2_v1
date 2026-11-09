"""scripts/pilot_postrun.py on a supervised attempt over tiny NON-TRAIN fixtures (no TRAIN scenario, no label)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from ember_v1 import train_pilot as tp
from sim_helpers import scenario

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pilot_postrun", REPO / "scripts" / "pilot_postrun.py")
pr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr)
H = "0" * 40


def tiny(i):
    return scenario(["E...#...", "....#...", "........", "..F.....", "......E."],
                    {"P1": (2, 1), "P2": (2, 2), "P3": (4, 5)}, fire="stochastic", p=0.5, interval=1, seed=5 + i,
                    max_steps=6, enter_fire="blocked", sid=f"p1-12-postrun-fixture-{i}")


@pytest.fixture(scope="module")
def attempt(tmp_path_factory):
    out = tmp_path_factory.mktemp("attempt")
    sc = [{"scenario_id": f"fx-{i}", "scenario_hash": tiny(i).content_hash(), "label": "feasible",
           "lineage_id": f"lin-{i}", "family_id": f"fam-{i}"} for i in range(2)]
    plan = {"plan_hash": "sha256:mini", "attempt": 1, "scenarios": sc, "teacher_pins": tp.teacher_pins(),
            "limits": {**tp.LIMITS, "group_wall_s": 120, "pilot_wall_s": 600},
            "regeneration": {"scenario_ids": ["fx-0"]}, "cost_estimate": {"t2_cpu_s_total_limit": 1e9},
            "source": {"manifest_hash": "fixture"}}
    res = tp.execute(plan, [({"scenario_id": f"fx-{i}", "lineage_id": f"lin-{i}"}, tiny(i)) for i in range(2)],
                     out, {"execution_commit": H})
    assert res["accepted"], res["problems"]
    return out, plan, {f"fx-{i}": tiny(i) for i in range(2)}


def _copy(src: Path, dst: Path) -> Path:
    for p in src.rglob("*"):
        if p.is_file():
            q = dst / p.relative_to(src)
            q.parent.mkdir(parents=True, exist_ok=True)
            q.write_bytes(p.read_bytes())
    return dst


def test_clean_attempt_verifies_replays_and_hashes_every_file(attempt):
    out, plan, scs = attempt
    rep = pr.verify_attempt(out, plan, scs)
    assert rep["integrity_verified"] and rep["accepted"], rep["problems"]
    assert rep["complete_replayed_groups"] == 2 and rep["regeneration"]["fx-0"]["identical"]
    assert rep["regeneration"]["fx-0"]["replayed"] and rep["teacher_calls"] == 0
    names = {f["path"] for f in rep["files"]}
    assert {"attempt-1.json", "attempt-1/evidence.json", "attempt-1/log.jsonl", "attempt-1/groups/fx-0.json",
            "attempt-1/groups/fx-1.json", "attempt-1/regen/fx-0.json"} <= names
    assert all(f["sha256"].startswith("sha256:") and f["bytes"] > 0 for f in rep["files"])


def test_wrong_plan_hash_is_reported(attempt):
    out, plan, scs = attempt
    rep = pr.verify_attempt(out, plan, scs, plan_hash=pr.PLAN_HASH)
    assert not rep["integrity_verified"] and any("another plan" in p for p in rep["problems"])


def test_tampered_group_does_not_replay(attempt, tmp_path):
    out, plan, scs = attempt
    d = _copy(out, tmp_path / "t")
    g = d / "attempt-1" / "groups" / "fx-1.json"
    doc = json.loads(g.read_text(encoding="utf-8"))
    doc["outcome_note"] = "edited"
    g.write_text(json.dumps(doc), encoding="utf-8")
    rep = pr.verify_attempt(d, plan, scs)
    assert not rep["integrity_verified"] and not rep["groups"]["fx-1"]["replayed"] and not rep["accepted"]


def test_missing_regeneration_and_evidence_are_reported(attempt, tmp_path):
    out, plan, scs = attempt
    d = _copy(out, tmp_path / "m")
    (d / "attempt-1" / "regen" / "fx-0.json").unlink()
    rep = pr.verify_attempt(d, plan, scs)
    assert not rep["integrity_verified"] and not rep["regeneration"]["fx-0"]["present"]
    (d / "attempt-1" / "evidence.json").unlink()
    rep = pr.verify_attempt(d, plan, scs)
    assert any("evidence.json missing" in p for p in rep["problems"]) and not rep["accepted"]


def test_edited_evidence_fails_its_hash(attempt, tmp_path):
    out, plan, scs = attempt
    d = _copy(out, tmp_path / "e")
    ev = d / "attempt-1" / "evidence.json"
    doc = json.loads(ev.read_text(encoding="utf-8"))
    doc["acceptance"]["pass"] = not doc["acceptance"]["pass"]
    ev.write_text(json.dumps(doc), encoding="utf-8")
    assert any("evidence hash" in p for p in pr.verify_attempt(d, plan, scs)["problems"])


def test_out_inside_checkout_refused(tmp_path):
    with pytest.raises(SystemExit, match="outside the execution checkout"):
        pr.main(["--repo", str(tmp_path), "--out", str(tmp_path / "postrun.json")])
