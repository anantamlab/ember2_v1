"""sft_prep: SFT files from a VALID dataset; family-isolated validation; held-out isolation; loss masks."""
from __future__ import annotations

import json

import pytest

from ember_v1 import sft_prep as sp
from test_train_dataset import _build, world  # noqa: F401  (fixture)


@pytest.fixture()
def built(world):  # noqa: F811
    out = world["tmp"] / "ds"
    _build(world, out)
    return world, out


def _prep(w, ddir, out, **kw):
    return sp.prepare(dataset_dir=ddir, out=out, plan_path=w["paths"]["plan.json"], **kw)


def test_prepare_writes_model_view_only_and_isolates_families(built):
    w, ddir = built
    out = w["tmp"] / "sft"
    res = _prep(w, ddir, out, n_val_families=1)
    assert res["status"] == sp.STATUS and not (w["tmp"] / "sft.partial").exists()
    idx = [json.loads(x) for x in (out / "examples_index.jsonl").read_text(encoding="utf-8").splitlines()]
    fam = {}
    for e in idx:
        fam.setdefault(e["family_id"], set()).add(e["split"])
    assert all(len(v) == 1 for v in fam.values()) and {s for v in fam.values() for s in v} == {"train", "val"}
    assert res["val_families"] == sp.val_families(sorted(fam), 1)
    for split in ("train", "val"):
        lines = (out / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == res["splits"][split]["examples"]
        for x in lines:
            ex = json.loads(x)
            assert set(ex) == {"example_id", "observation", "actions", "weight"} and ex["weight"] > 0
            assert "T0" not in x and "teacher" not in x
    assert res["splits"]["train"]["rows"] >= res["splits"]["train"]["examples"]
    assert res["files"]["train.jsonl"] == sp._sha256(out / "train.jsonl")
    with pytest.raises(sp.PrepBlocked, match="write-once"):
        _prep(w, ddir, out, n_val_families=1)


def test_heldout_identifier_blocks(built, monkeypatch):
    w, ddir = built
    real = sp.heldout_ids

    def leaky(repo=sp.REPO):
        h = real(repo)
        h["family_id"].add("fam-a")
        return h
    monkeypatch.setattr(sp, "heldout_ids", leaky)
    out = w["tmp"] / "sft"
    with pytest.raises(sp.PrepBlocked, match="held-out identifiers"):
        _prep(w, ddir, out)
    assert not out.exists()


def test_changed_dataset_blocks(built):
    w, ddir = built
    p = next((ddir / "records").glob("*.jsonl"))
    p.write_text(p.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(sp.PrepBlocked, match="record file missing or changed"):
        _prep(w, ddir, w["tmp"] / "sft")


def test_row_masks_cover_every_active_person():
    from test_multiteacher import busy_room
    from ember_v1.sim import Simulator
    s = busy_room()
    obs = Simulator(s).observe()
    acts = {p["id"]: "stay" for p in obs["people"] if p["status"] == "active"}
    ex = {"example_id": "x", "observation": obs, "actions": acts}
    st = sp._row_stats(ex, 2)
    assert st["people"] == len(acts) and st["rows"] == -(-len(acts) // 2)


def test_heldout_manifests_are_disjoint_from_the_collection_plan():
    plan = json.loads((sp.REPO / sp.PLAN_FILE).read_text(encoding="utf-8"))
    h = sp.heldout_ids()
    for s in plan["scenarios"]:
        assert s["family_id"] not in h["family_id"] and s["core_hash"] not in h["core_hash"]
        assert s["scenario_hash"] not in h["scenario_hash"] and s["lineage_id"] not in h["lineage_id"]
