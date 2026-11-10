"""p1-18-expand@1: family-isolated generation, fixed family partitions, and the resumable five-teacher collector."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ember_v1 import expand as E
from ember_v1 import expand_collect as X
from ember_v1 import multiteacher as mt
from ember_v1.sim.canonical import digest
from ember_v1.teacher import confirmation as C
from sim_helpers import scenario

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def built():
    entries, ref, prov = E.historical(REPO)
    pilots = C.pilot_maps()
    m = E.build_manifest(entries, ref, pilots, families={"core": 3, "stress": 3})
    return entries, prov, pilots, m


def test_small_plan_is_isolated_partitioned_and_regenerates(built):
    entries, prov, pilots, m = built
    a = E.isolation_report(m, entries, prov, pilots)
    assert a["pass"] and a["families"] == 6 and a["scenarios"] == 24
    assert all(s["n_people"] <= 120 for s in m["scenarios"] if s["stratum"] == "core")
    assert all(121 <= s["n_people"] <= 200 for s in m["scenarios"] if s["stratum"] == "stress")
    fams = {f["family_id"]: f["partition"] for f in m["families"]}
    assert all(s["partition"] == fams[s["family_id"]] for s in m["scenarios"])   # partition is by family
    assert {f["partition"] for f in m["families"]} == set(E.PARTITIONS)
    hist_fams = {e["family_id"] for es in entries.values() for e in es}
    assert not set(fams) & hist_fams
    assert [e["scenario_hash"] for e, _ in E.materialize(m)] == [s["scenario_hash"] for s in m["scenarios"]]
    assert len(E.materialize(m, partitions=("val",))) == 4 * sum(p == "val" for p in fams.values())


def test_tampered_manifest_is_refused(built):
    m = json.loads(json.dumps(built[3]))
    m["scenarios"][0]["partition"] = "train" if m["scenarios"][0]["partition"] != "train" else "val"
    with pytest.raises(E.ExpandError):
        E.materialize(m)


def test_partition_rule_is_stable_and_exact():
    ids = [f"fam-{i}" for i in range(20)]
    p = E.partition(ids, {"train": 14, "val": 2, "diagnostic": 4})
    assert p == E.partition(list(reversed(ids)), {"train": 14, "val": 2, "diagnostic": 4})
    assert sorted(p.values()).count("val") == 2
    with pytest.raises(E.ExpandError):
        E.partition(ids, {"train": 1, "val": 1, "diagnostic": 1})


def test_full_plan_counts_are_the_preregistered_ones():
    p = E.plan()
    assert sum(s["families"] for s in p["strata"].values()) * E.LINEAGES == 1200
    assert p["strata"]["core"]["spec"]["crowd"]["count"] == [1, 120]
    assert p["strata"]["stress"]["spec"]["crowd"]["count"] == [121, 200]
    assert sum(p["strata"]["core"]["partition"].values()) == 225


def _tiny(i):
    return scenario(["E...#...", "....#...", "........", "..F.....", "......E."],
                    {"P1": (2, 1), "P2": (2, 2), "P3": (4, 5)}, fire="stochastic", p=0.5, interval=1, seed=31 + i,
                    max_steps=6, enter_fire="blocked", sid=f"p1-18-fixture-{i}")


def _mini(n=3):
    sc = [{"scenario_id": f"fx-{i}", "scenario_hash": _tiny(i).content_hash(), "lineage_id": f"lin-{i}",
           "family_id": f"fam-{i}", "partition": ("train", "val", "diagnostic")[i % 3], "stratum": "core",
           "n_people": 3, "diagnostics": {"max_steps": 6}} for i in range(n)]
    body = {"plan_hash": E.plan_hash(), "scenarios": sc}
    m = {**body, "manifest_hash": digest(body)}
    return m, [(e, _tiny(i)) for i, e in enumerate(sc)]


def test_collector_runs_resumes_and_seals_its_own_scope(tmp_path):
    m, scs = _mini()
    root = tmp_path / "ev"
    end = X.run(REPO, root, concurrency=2, manifest=m, scenarios=scs, limit=2, poll_s=0.05)
    assert end["groups"] == {"row": 2}
    # a crashed group (started, no row) is rerun from scratch in the next session
    (root / "started" / "fx-2.json").write_text("{}", encoding="utf-8")
    end2 = X.run(REPO, root, concurrency=2, manifest=m, scenarios=scs, poll_s=0.05)
    assert end2["session"] == 2 and end2["groups"] == {"row": 1}
    assert any(p.name.startswith("s2-started-") for p in (root / "stale").iterdir())
    X.register_scope()
    for i in range(3):
        rep = json.loads((root / "groups" / f"fx-{i}.json").read_text(encoding="utf-8"))
        mt.verify_fixture(rep, _tiny(i))
        assert rep["scope"] == X.SCOPE and rep["group_complete"] and rep["training_eligible"] is False
    s = X.summarize(root)
    assert s["complete_verified"] == 3 and not s["not_complete"]
    # the root refuses a different manifest
    m2 = {**m, "manifest_hash": "sha256:" + "1" * 64}
    with pytest.raises(X.ExpandCollectError):
        X.run(REPO, root, concurrency=1, manifest=m2, scenarios=scs)
