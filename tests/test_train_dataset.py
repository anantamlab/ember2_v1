"""train_dataset: gated, write-once build of class A/B records from a verified collection (tiny fixtures)."""
from __future__ import annotations

import hashlib
import json

import pytest

from ember_v1 import multiteacher as mt
from ember_v1 import multiteacher_dataset as ds
from ember_v1 import train_dataset as td
from ember_v1.sim.canonical import digest
from ember_v1.teacher.registry import CANDIDATES
from test_multiteacher import busy_room, real_factories, room


def _group(s, fam):
    pols = {n: mt._new_policy(f, n) for n, f in real_factories().items()}
    res = {n: mt._rollout(s, pols[n], cpu_ms_cap=30000) for n in CANDIDATES}
    return mt.assemble_group(s, res, mt.provenance(s, pols, None), scope=td.SCOPE)


@pytest.fixture()
def world(tmp_path):
    root, specs, scen = tmp_path / "root", [], []
    for i, (s, fam) in enumerate([(busy_room(), "fam-a"), (room(), "fam-a"), (busy_room(), "fam-b")]):
        sid = f"g-{i:016x}"
        g = _group(s, fam)
        gp = root / "groups" / f"{sid}.json"
        gp.parent.mkdir(parents=True, exist_ok=True)
        gp.write_text(json.dumps(g), encoding="utf-8")
        row = {"file_sha256": "sha256:" + hashlib.sha256(gp.read_bytes()).hexdigest()}
        (root / "rows").mkdir(exist_ok=True)
        (root / "rows" / f"{sid}.json").write_text(json.dumps(row), encoding="utf-8")
        specs.append({"scenario_id": sid, "scenario_hash": s.content_hash(), "family_id": fam})
        scen.append(({"scenario_id": sid}, s))
    plan = {"plan_hash": "sha256:" + "1" * 64, "namespace": td.SCOPE, "scenarios": specs,
            "geometry_audit": {"teacher_dev_similar_families": ["fam-b"]}}
    vbody = {"plan_hash": plan["plan_hash"], "verification_version": "p1-17-train-full-verify@1.0.0",
             "full_train_verified": True, "status_counts": {"verified_complete": 3}}
    ver = {**vbody, "verification_hash": digest(vbody)}
    proto = ds.load_protocol()
    proto["collection_gate"] = {**proto["collection_gate"], "plan_hash": plan["plan_hash"], "groups": 3,
                                "verification_hash": ver["verification_hash"]}
    paths = {}
    for name, obj in (("plan.json", plan), ("verification.json", ver), ("protocol.json", proto)):
        paths[name] = tmp_path / name
        paths[name].write_text(json.dumps(obj), encoding="utf-8")
    return {"tmp": tmp_path, "root": root, "scen": scen, "paths": paths, "specs": specs}


def _build(w, out, **kw):
    return td.build(repo=td.REPO, root=w["root"], verification_path=w["paths"]["verification.json"], out=out,
                    authorization="owner run-17 test", workers=kw.pop("workers", 1),
                    plan_path=w["paths"]["plan.json"], protocol_path=w["paths"]["protocol.json"],
                    scenarios=w["scen"], **kw)


def test_build_is_valid_family_weighted_and_flagged(world):
    out = world["tmp"] / "ds"
    res = _build(world, out)
    assert res["status"] == "VALID", res["problems"]
    assert res["n_problems"] == 0 and res["groups"] == 3 and res["training_eligible"] is False
    assert (out / "records").is_dir() and not (out / "staging" / "records").exists()
    recs = [r for p in sorted((out / "records").glob("*.jsonl")) for r in td._load_jsonl(p)]
    assert len(recs) == res["records_total"]
    proto = ds.load_protocol(world["paths"]["protocol.json"])
    assert all(ds.record_problems(r, proto) == [] for r in recs)
    assert not any("T0" in r["teachers"] for r in recs if r["class"] == "A")
    fam = {}
    for r in recs:
        fam[r["family_id"]] = fam.get(r["family_id"], 0) + r["weight"]
    # every family with class A has the same budget before the uninformative factor; room() ties (x0.25)
    a_fams = res["family_scenarios_with_A"]
    assert set(a_fams) <= {"fam-a", "fam-b"} and all(v <= 1.0 + 1e-9 for v in fam.values())
    if a_fams.get("fam-a") == 2:
        assert fam["fam-a"] == pytest.approx(0.5 + 0.5 * 0.25)
    assert all(r["evidence"]["teacher_dev_similar_family"] == (r["family_id"] == "fam-b") for r in recs)
    assert res["audit"]["class_reasons"]["diagnostic_teacher"] >= 1
    d = json.loads((out / "dataset.json").read_text(encoding="utf-8"))
    assert d["dataset_hash"] == digest({k: v for k, v in d.items() if k != "dataset_hash"})
    for sid, h in d["record_files"].items():
        assert h == td._sha256(out / "records" / f"{sid}.jsonl")
    ex = ds.model_example(next(r for r in recs if r["class"] == "A" and r["weight"] > 0))
    assert set(ex) == {"observation", "actions", "weight"}


def test_parallel_build_equals_sequential(world):
    a = _build(world, world["tmp"] / "seq")
    b = _build(world, world["tmp"] / "par", workers=2)
    assert a["record_files"] == b["record_files"] and a["audit"] == b["audit"]


def test_build_is_write_once_and_outside_inputs(world):
    _build(world, world["tmp"] / "ds")
    with pytest.raises(td.BuildBlocked, match="write-once"):
        _build(world, world["tmp"] / "ds")
    with pytest.raises(td.BuildBlocked, match="outside the checkout and the collection root"):
        _build(world, world["root"] / "ds")
    with pytest.raises(td.BuildBlocked, match="outside the checkout"):
        _build(world, td.REPO / ".local-ds-test")


def test_changed_group_file_blocks_before_any_write(world):
    gp = world["root"] / "groups" / f"{world['specs'][0]['scenario_id']}.json"
    gp.write_text(gp.read_text(encoding="utf-8") + " ", encoding="utf-8")
    out = world["tmp"] / "ds"
    with pytest.raises(td.BuildBlocked, match="sha256 differs from its row"):
        _build(world, out)
    assert not out.exists()


def test_unverified_collection_blocks(world):
    p = world["paths"]["verification.json"]
    v = json.loads(p.read_text(encoding="utf-8"))
    p.write_text(json.dumps({**v, "full_train_verified": False}), encoding="utf-8")
    out = world["tmp"] / "ds"
    with pytest.raises(ds.DatasetError, match="DATASET_GENERATION_BLOCKED"):
        _build(world, out)
    assert not out.exists()


def test_a_bad_record_makes_the_dataset_invalid(world, monkeypatch):
    real = ds.record_problems

    def one_bad(r, protocol):
        return real(r, protocol) + (["injected"] if r["class"] == "A" and r["weight"] > 0 else [])
    monkeypatch.setattr(ds, "record_problems", one_bad)
    out = world["tmp"] / "ds"
    res = _build(world, out)
    assert res["status"] == "INVALID" and res["n_problems"] > 0
    assert not (out / "records").exists() and (out / "staging" / "records").is_dir()


def test_light_record_keeps_audit_fields(world):
    out = world["tmp"] / "ds"
    _build(world, out)
    recs = [r for p in sorted((out / "records").glob("*.jsonl")) for r in td._load_jsonl(p)]
    assert ds.audit([td.light(r) for r in recs]) == ds.audit(recs)
    assert ds.selection_bias_audit([td.light(r) for r in recs]) == ds.selection_bias_audit(recs)
