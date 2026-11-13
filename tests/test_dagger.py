"""DAgger labels: replayed student states, group-best teacher maps, inconclusive kept, TRAIN only."""
from __future__ import annotations

import json

import pytest

from ember_v1 import dagger as D
from ember_v1 import expand_collect as X
from ember_v1 import student_run as sr
from ember_v1 import student_train as st
from ember_v1.sim.canonical import digest
from test_expand import REPO, _mini, _tiny


class _Stay:
    def predict(self, row):
        return {p: "stay" for p in row["questions"]}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dg")
    m, scs = _mini(3)                                  # fx-0 train, fx-1 val, fx-2 diagnostic
    root = tmp / "root"
    X.run(REPO, root, concurrency=2, manifest=m, scenarios=scs, poll_s=0.05)
    cfg = st.config_from_protocol(st.load_protocol())
    lst = tmp / "list.jsonl"
    lst.write_text("".join(json.dumps({"set": "x", "scenario_id": e["scenario_id"], "family_id": e["family_id"],
                                       "scenario": s.to_dict(), "scenario_hash": s.content_hash()}) + "\n"
                           for e, s in scs), encoding="utf-8")
    g = st.Guards(1e9, 1e9, 1.0)
    sr.rollouts(_Stay(), cfg, lst, tmp / "roll.jsonl", label="stay", guards=lambda: g, log=lambda s: None)
    rows = [json.loads(x) for x in (tmp / "roll.jsonl").read_text().splitlines()]
    train = tmp / "train_roll.jsonl"
    train.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    return {"tmp": tmp, "root": root, "m": m, "scs": scs, "rows": rows, "train": train, "all": tmp / "roll.jsonl"}


def test_labels_are_group_best_maps_on_replayed_states(world):
    out = world["tmp"] / "dg"
    s = D.run(rollouts=world["train"], root=world["root"], out=out, repo=REPO, workers=1, states=3,
              manifest=world["m"], scenarios=world["scs"])
    assert s["episodes"] == 1 and s["states"] == min(3, len(world["rows"][0]["trace"]))
    recs = [json.loads(x) for x in (out / "labels" / "fx-0.jsonl").read_text().splitlines()]
    g = json.loads((world["root"] / "groups" / "fx-0.json").read_text())
    assert all(r["best_teachers"] == D.best_teachers(g) and "T0" not in r["best_teachers"] for r in recs)
    for r in recs:
        if r["status"] == "label":
            assert digest(r["observation"]) == r["observation_digest"] and r["actions"]
            active = {p["id"] for p in r["observation"]["people"] if p["status"] == "active"}
            assert set(r["actions"]) == active                # a whole joint map
        else:
            assert r["status"] in ("inconclusive", "teacher_cap") and r["actions"] is None
    # resumable: a second run labels nothing new
    assert D.run(rollouts=world["train"], root=world["root"], out=out, repo=REPO, workers=1, states=3,
                 manifest=world["m"], scenarios=world["scs"])["episodes"] == 1
    ex = D.sft_examples(out, world["tmp"] / "dg_sft.jsonl")
    if ex["examples"]:
        assert ex["weight_total"] == pytest.approx(ex["families"])
    ex = D.sft_examples(out, world["tmp"] / "dg_sft_q.jsonl", family_weight=0.25)
    if ex["examples"]:
        assert ex["weight_total"] == pytest.approx(0.25 * ex["families"])


def test_heldout_partitions_are_refused(world):
    with pytest.raises(D.DaggerError):
        D.run(rollouts=world["all"], root=world["root"], out=world["tmp"] / "dg2", repo=REPO, workers=1,
              manifest=world["m"], scenarios=world["scs"])


def test_pick_steps_is_fixed_and_bounded():
    a = D.pick_steps(50, "sha256:x", 8)
    assert a == D.pick_steps(50, "sha256:x", 8) and len(a) == 8 and a == sorted(a)
    assert D.pick_steps(3, "sha256:x", 8) == [0, 1, 2]
