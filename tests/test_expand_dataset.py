"""P1-18 dataset v2: expansion TRAIN groups -> weighted class-A records -> SFT v2 = P1-17 SFT + expansion."""
from __future__ import annotations

import json

import pytest

from ember_v1 import expand_collect as X
from ember_v1 import expand_dataset as XD
from ember_v1 import multiteacher_dataset as ds
from ember_v1.sim.canonical import digest
from test_expand import REPO, _mini, _tiny


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("xd")
    sc = [{"scenario_id": f"fx-{i}", "scenario_hash": _tiny(i).content_hash(), "lineage_id": f"lin-{i}",
           "family_id": f"fam-{i // 2}", "partition": ("train", "train", "train", "train", "val")[i],
           "stratum": "core", "n_people": 3, "diagnostics": {"max_steps": 6}} for i in range(5)]
    m0, _ = _mini(1)
    body = {"plan_hash": m0["plan_hash"], "scenarios": sc}
    m = {**body, "manifest_hash": digest(body)}
    scs = [(e, _tiny(i)) for i, e in enumerate(sc)]
    root = tmp / "root"
    X.run(REPO, root, concurrency=2, manifest=m, scenarios=scs, limit=4, poll_s=0.05)
    return {"tmp": tmp, "root": root, "m": m, "scs": scs}


def _p17(tmp):
    d = tmp / "p17"
    if d.exists():
        return d
    d.mkdir()
    ex = {"example_id": "e0", "observation": {"x": 1}, "actions": {}, "weight": 1.0}
    (d / "train.jsonl").write_text(json.dumps(ex) + "\n", encoding="utf-8", newline="\n")
    (d / "val.jsonl").write_text(json.dumps({**ex, "example_id": "e1"}) + "\n", encoding="utf-8", newline="\n")
    (d / "sft_manifest.json").write_text(json.dumps({"val_families": ["fam-p17-val"], "manifest_hash": "h"}),
                                         encoding="utf-8")
    return d


def test_build_uses_only_complete_train_groups_and_is_valid(world, monkeypatch):
    out = world["tmp"] / "ds"
    r = XD.build(repo=REPO, root=world["root"], out=out, workers=1, manifest=world["m"], scenarios=world["scs"])
    assert r["status"] == "VALID", r["problems"]
    assert r["groups_used"] == 4 and r["groups_skipped"] == {}         # fx-4 is val: never a candidate
    proto = ds.load_protocol(REPO / XD.PROTOCOL)
    recs = [json.loads(x) for p in (out / "records").glob("*.jsonl") for x in p.read_text().splitlines() if x]
    assert recs and all(ds.record_problems(rec, proto) == [] for rec in recs)
    assert {rec["scope"] for rec in recs} == {X.SCOPE}
    fam = {}
    for rec in recs:
        fam[rec["family_id"]] = fam.get(rec["family_id"], 0) + rec["weight"]
    assert all(v <= 1.0 + 1e-9 for v in fam.values())
    d = json.loads((out / "dataset.json").read_text(encoding="utf-8"))
    assert d["dataset_hash"] == digest({k: v for k, v in d.items() if k != "dataset_hash"})
    with pytest.raises(XD.BuildBlocked):                                 # write-once
        XD.build(repo=REPO, root=world["root"], out=out, manifest=world["m"], scenarios=world["scs"])
    with pytest.raises(XD.BuildBlocked):                                 # never inside the checkout
        XD.build(repo=REPO, root=world["root"], out=REPO / "tmp-ds", manifest=world["m"], scenarios=world["scs"])

    # SFT v2: P1-17 bytes first, then expansion class-A examples only
    import ember_v1.student_run as sr
    monkeypatch.setattr(sr, "check_sft_dir", lambda p: {"manifest_hash": "sha256:p17"})
    p17 = _p17(world["tmp"])
    s = XD.sft(dataset_dir=out, out=world["tmp"] / "sft2", p17_sft=p17, manifest=world["m"])
    lines = (world["tmp"] / "sft2" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    assert lines[0] == (p17 / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    n_sel = sum(1 for rec in recs if rec["class"] == "A" and rec["selection"]["selected"] and rec["weight"])
    assert s["splits"]["train"]["examples_expansion"] == n_sel == len(lines) - 1
    assert (world["tmp"] / "sft2" / "val.jsonl").read_bytes() == (p17 / "val.jsonl").read_bytes()
    assert all(set(json.loads(x)) == {"example_id", "observation", "actions", "weight"} for x in lines[1:])


def test_changed_group_file_blocks(world):
    gp = world["root"] / "groups" / "fx-0.json"
    keep = gp.read_bytes()
    try:
        gp.write_bytes(keep + b" ")
        with pytest.raises(XD.BuildBlocked):
            XD.build(repo=REPO, root=world["root"], out=world["tmp"] / "ds-bad", manifest=world["m"],
                     scenarios=world["scs"])
    finally:
        gp.write_bytes(keep)


def test_val_family_leak_into_sft_is_refused(world, monkeypatch):
    out = world["tmp"] / "ds-leak"
    XD.build(repo=REPO, root=world["root"], out=out, workers=1, manifest=world["m"], scenarios=world["scs"])
    import ember_v1.student_run as sr
    monkeypatch.setattr(sr, "check_sft_dir", lambda p: {"manifest_hash": "sha256:p17"})
    p17 = _p17(world["tmp"])
    (p17 / "sft_manifest.json").write_text(json.dumps({"val_families": ["fam-0", "fam-1"]}), encoding="utf-8")
    try:
        with pytest.raises(XD.BuildBlocked):
            XD.sft(dataset_dir=out, out=world["tmp"] / "sft-leak", p17_sft=p17, manifest=world["m"])
    finally:
        (p17 / "sft_manifest.json").write_text(json.dumps({"val_families": ["fam-p17-val"]}), encoding="utf-8")
