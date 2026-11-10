"""train_counterfactual: preregistered batch spec and a tiny end-to-end batch on real-scope fixture groups."""
from __future__ import annotations

import json

import pytest

from ember_v1 import train_counterfactual as tc
from ember_v1.sim.canonical import digest
from test_counterfactual_step0 import _group
from test_multiteacher import busy_room


def _spec():
    return json.loads(tc.SPEC_PATH.read_text(encoding="utf-8"))


def _plan():
    return json.loads((tc.REPO / tc.PLAN_FILE).read_text(encoding="utf-8"))


def test_committed_spec_is_frozen_hashed_and_recomputes():
    spec, plan = _spec(), _plan()
    ver = {"plan_hash": plan["plan_hash"], "verification_hash": spec["source"]["verification_hash"],
           "full_train_verified": True}
    assert tc.check_spec(spec, plan, ver) == []
    assert len(spec["selection"]["scenario_ids"]) == 16 and len(tc.jobs_of(spec, plan)) == 32
    assert [c["name"] for c in spec["continuations"]] == ["T1", "T4"]
    assert spec["storage"]["in_git"] is False and spec["budget"]["usd"] == 0
    bad = {**spec, "selection": {**spec["selection"], "per_bin": 3}}
    assert any("hash" in p for p in tc.check_spec(bad, plan, ver))
    assert any("verification hash" in p for p in tc.check_spec(spec, plan, {**ver, "verification_hash": "x"}))
    assert any("full_train_verified" in p for p in tc.check_spec(spec, plan, {**ver, "full_train_verified": False}))


@pytest.fixture()
def tiny(tmp_path):
    s = busy_room()
    sid = "g-00000000000000aa"
    g = _group(s)
    root = tmp_path / "collection"
    (root / "groups").mkdir(parents=True)
    (root / "groups" / f"{sid}.json").write_text(json.dumps(g), encoding="utf-8")
    plan = {"plan_hash": "sha256:" + "2" * 64, "storage": {"root": str(root)},
            "scenarios": [{"scenario_id": sid, "people": len(s.people), "rank": "0", "max_steps": s.max_steps,
                           "fire_model": s.fire_model, "family_id": "fam-t", "lineage_id": "lin-t"}]}
    vb = {"plan_hash": plan["plan_hash"], "full_train_verified": True}
    ver = {**vb, "verification_hash": digest(vb)}
    spec = {**_spec(), "source": {"scope": "p1-17-train-full@1", "plan_hash": plan["plan_hash"],
                                  "verification_hash": ver["verification_hash"]},
            "selection": {"rule": "test", "per_bin": 1, "scenario_ids": [sid]},
            "continuations": [{"name": "T1", "version": "1.0.0"}], "concurrency": 1}
    spec.pop("spec_hash")
    spec["spec_hash"] = digest(spec)
    paths = {}
    for name, obj in (("plan.json", plan), ("ver.json", ver), ("spec.json", spec)):
        paths[name] = tmp_path / name
        paths[name].write_text(json.dumps(obj), encoding="utf-8")
    return {"tmp": tmp_path, "s": s, "sid": sid, "root": root, "paths": paths}


def _run(t, out, **kw):
    return tc.run(root=out, verification_path=t["paths"]["ver.json"], spec_path=t["paths"]["spec.json"],
                  plan_path=t["paths"]["plan.json"], scenarios=[({"scenario_id": t["sid"]}, t["s"])],
                  group_root=t["root"], workers=1, **kw)


def test_tiny_batch_runs_verifies_and_is_write_once(tiny):
    out = tiny["tmp"] / "batch"
    res = _run(tiny, out)
    assert res["studies"] == 1 and res["failed"] == 0 and res["all_verified"], res["results"]
    assert res["records_valid"] and res["approved_preferences"] == 0 and res["training_eligible"] is False
    r = res["results"][0]
    assert r["status"] == "contrasted" and r["alternatives"] >= 2 and r["records"] == res["records"]
    b = json.loads((out / "batch.json").read_text(encoding="utf-8"))
    assert b["batch_hash"] == digest({k: v for k, v in b.items() if k != "batch_hash"})
    with pytest.raises(tc.BatchBlocked, match="write-once"):
        _run(tiny, out)


def test_failed_study_is_recorded_not_retried(tiny, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("injected")
    monkeypatch.setattr(tc.cf, "step0_study", boom)
    out = tiny["tmp"] / "batch"
    res = _run(tiny, out)
    assert res["failed"] == 1 and not res["all_verified"] and "injected" in res["results"][0]["error"]
    assert (out / "failures" / f"{tiny['sid']}-T1.txt").exists()
