"""Split manifests: isolation, deliberate leakage attempts, immutability and the locked-test seal."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from ember_v1.gen import analysis as A
from ember_v1.gen.splits import (LOCKED, LOCKED_PURPOSE, PUBLIC_SPLITS, SPLITS, AuditItem, LockedSplitError,
                                 ManifestError, SplitPlan, audit, audit_items, build_all, load_index,
                                 load_manifest, materialize, plan_from_index, verify_split_files,
                                 wall_geometry_audit, write_split_files)
from ember_v1.sim import Scenario
from gen_helpers import small_plan

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def built():
    return build_all(small_plan())


def test_small_plan_builds_and_passes_audit(built):
    build, manifests, report, idx = built
    assert report["passed"] and idx["audit"]["passed"]
    for s, n in small_plan().families:
        assert len(manifests[s]["families"]) == n
        assert {e["family_id"] for e in manifests[s]["scenarios"]} <= {f["family_id"] for f in manifests[s]["families"]}
    assert idx["splits"][LOCKED]["file"] is None and idx["splits"][LOCKED]["sealed"]
    assert LOCKED not in idx["coverage"]


def test_splits_are_disjoint_in_families_lineages_and_seeds(built):
    _, manifests, _, _ = built
    seen: dict[str, set] = {"fam": set(), "lin": set(), "seed": set(), "topo": set()}
    for s in SPLITS:
        m = manifests[s]
        fams = {f["family_id"] for f in m["families"]}
        lins = {e["lineage_id"] for e in m["scenarios"]}
        seeds = ({f["family_seed"] for f in m["families"]} | {e["lineage_seed"] for e in m["scenarios"]}
                 | {e["scenario_seed"] for e in m["scenarios"]})
        topo = {f["topology_hash"] for f in m["families"]}
        for key, vals in (("fam", fams), ("lin", lins), ("seed", seeds), ("topo", topo)):
            assert seen[key].isdisjoint(vals), (s, key)
            seen[key] |= vals


def test_manifest_regeneration_is_deterministic(built):
    _, manifests, _, idx = built
    _, again, _, idx2 = build_all(small_plan())
    assert again == manifests and idx2 == idx
    _, other, _, _ = build_all(small_plan(seed=8))
    assert other["train"]["manifest_hash"] != manifests["train"]["manifest_hash"]


def test_assignment_is_archetype_stratified():
    plan = SplitPlan.from_dict({**small_plan().to_dict(),
                                "families": {"train": 8, "dev": 4, "calibration": 4, "locked_test": 4}})
    build, manifests, _, _ = build_all(plan)
    per = {s: {f["archetype"] for f in manifests[s]["families"]} for s in SPLITS}
    assert all(len(v) >= 2 for v in per.values())


# ----- deliberate leakage attempts ------------------------------------------

def _items(build):
    return audit_items(build)


def _variant(s: Scenario, **changes) -> Scenario:
    d = s.to_dict()
    d.update(changes)
    return Scenario.from_dict(d)


def _transform(s: Scenario, k: int) -> Scenario:
    (h, w), f = A._transforms(s.height, s.width)[k]
    d = s.to_dict()
    d["grid"] = {"height": h, "width": w}
    move = lambda p: list(f(*p))  # noqa: E731
    d["obstacles"] = [move(p) for p in s.obstacles]
    d["exits"] = [{"id": i, "position": move(p)} for i, p in s.exits]
    d["people"] = [{"id": i, "position": move(p)} for i, p in s.people]
    d["ignitions"] = [move(p) for p in s.ignitions]
    d["scenario_id"] = f"leak-{k}"
    d["seed"] = 12345 + k
    return Scenario.from_dict(d)


def _leak(build, scenario: Scenario, *, family_id="fam-leak", lineage_id="lin-leak", seeds=(1, 2, 3)):
    items = _items(build) + [AuditItem(LOCKED, family_id, lineage_id, seeds, scenario)]
    return audit(items)


def _failed(report) -> set[str]:
    return {k for k, v in report["checks"].items() if not v["passed"]}


def test_leak_exact_copy_with_new_ids_and_seed_is_caught(built):
    build = built[0]
    src = build.scenarios["train"][0].scenario
    leak = _variant(src, scenario_id="leak", seed=999)
    rep = _leak(build, leak)
    assert not rep["passed"]
    assert {"core_hashes_disjoint", "topology_disjoint"} <= _failed(rep)


def test_leak_same_map_new_crowd_fire_and_rules_is_caught(built):
    build = built[0]
    src = build.scenarios["train"][0].scenario
    free = [(r, c) for r in range(src.height) for c in range(src.width)
            if (r, c) not in src.obstacles and (r, c) not in {p for _, p in src.exits}]
    d = src.to_dict()
    d.update({"scenario_id": "leak", "seed": 4242, "people": [{"id": "Q1", "position": list(free[0])}],
              "ignitions": [list(free[-1])], "fire": {"model": "deterministic", "spread_interval": 3,
                                                      "exits_flammable": False},
              "termination": {"max_steps": 77}})
    rep = _leak(build, Scenario.from_dict(d))
    assert _failed(rep) == {"topology_disjoint"}     # same map: exact topology match


@pytest.mark.parametrize("k", range(1, 8))
def test_leak_rotated_or_mirrored_copy_is_caught(built, k):
    build = built[0]
    src = build.scenarios["dev"][0].scenario
    rep = _leak(build, _transform(src, k))
    assert _failed(rep) == {"topology_disjoint"}     # symmetry-invariant topology hash


def test_leak_small_obstacle_edits_and_exit_shift_are_caught(built):
    build = built[0]
    for g in build.scenarios["train"][:6]:
        src = g.scenario
        walls = list(src.obstacles)
        n_edit = max(1, (src.height * src.width) // 100)          # 1 % of cells
        exits = {p for _, p in src.exits}
        occupied = exits | {p for _, p in src.people} | set(src.ignitions)
        floor = [(r, c) for r in range(src.height) for c in range(src.width)
                 if (r, c) not in src.obstacles and (r, c) not in occupied]
        new_walls = walls[n_edit:] + floor[:n_edit]
        rep = _leak(build, _variant(src, scenario_id="leak", seed=31337, obstacles=[list(p) for p in new_walls]))
        assert "near_duplicate_maps" in _failed(rep), g.family.archetype
        # move one exit to a free 4-neighbour cell
        eid, e = src.exits[0]
        to = next((n for n in A.neighbours4(e, src.height, src.width) if n in floor), None)
        if to is None:
            continue
        moved = [{"id": i, "position": list(to if i == eid else p)} for i, p in src.exits]
        rep = _leak(build, _variant(src, scenario_id="leak", seed=31338, exits=moved))
        assert _failed(rep) == {"near_duplicate_maps"}, g.family.archetype


def _with_far_exit(src: Scenario, k: int = 0) -> Scenario:
    """Same walls (after symmetry k), one exit on the free cell farthest from every original exit."""
    s = _transform(src, k)
    taken = {p for _, p in s.people} | set(s.ignitions)
    old = A._exit_points(s.height, s.width, [p for _, p in s.exits])
    free = [(r, c) for r in range(s.height) for c in range(s.width) if (r, c) not in s.obstacles and (r, c) not in taken]
    far = max(free, key=lambda p: min(max(abs((p[0] + .5) / s.height - a), abs((p[1] + .5) / s.width - b))
                                      for a, b in old))
    return _variant(s, scenario_id="leak-exits", seed=777 + k, exits=[{"id": "E0", "position": list(far)}])


def test_leak_same_walls_with_other_exits_is_caught_by_the_wall_rule(built):
    build = built[0]
    structured = [g.scenario for g in build.scenarios["train"]
                  if len(g.scenario.obstacles) >= A.NEAR_WALL_MIN_FRAC * g.scenario.height * g.scenario.width]
    assert len(structured) >= 3
    for src in structured[:3]:
        for k in (0, 5):
            leak = _with_far_exit(src, k)
            items = _items(build) + [AuditItem(LOCKED, "fam-leak", "lin-leak", (1, 2, 3), leak)]
            assert audit(items)["passed"]                  # the exit-aware rules allow it ...
            rep = wall_geometry_audit(items)
            assert not rep["passed"] and rep["violations"] == 1   # ... the exit-free wall rule does not


def _hall(h: int, w: int, walls, exits, sid: str) -> Scenario:
    taken = set(walls) | set(exits)
    free = [(r, c) for r in range(h) for c in range(w) if (r, c) not in taken]
    return Scenario.from_dict({
        "schema_version": "1.0", "scenario_id": sid, "grid": {"height": h, "width": w},
        "obstacles": [list(p) for p in sorted(walls)],
        "exits": [{"id": f"E{i}", "position": list(p)} for i, p in enumerate(exits)],
        "people": [{"id": "P0", "position": list(free[0])}], "ignitions": [list(free[-1])],
        "fire": {"model": "deterministic", "spread_interval": 2, "exits_flammable": False},
        "movement": {"conflict_resolution": "lowest_id_wins", "enter_fire": "blocked"},
        "termination": {"max_steps": 50}, "seed": 1000 + ord(sid[0])})


def test_sparse_halls_with_far_apart_exits_are_allowed():
    # like the nearest p1-02-v1 cross-split pair: an empty hall and a hall with 2 % clutter
    empty = _hall(26, 50, [], [(0, 3), (25, 40)], "a")
    clutter = [(r, c) for r in range(1, 28, 3) for c in range(2, 50, 5)][:29]
    sparse = _hall(29, 51, clutter, [(10, 0), (28, 20), (0, 50)], "b")
    other_empty = _hall(26, 50, [], [(12, 49)], "c")
    items = [AuditItem("train", "fam-a", "lin-a", (1, 2, 3), empty),
             AuditItem(LOCKED, "fam-b", "lin-b", (4, 5, 6), sparse),
             AuditItem("dev", "fam-c", "lin-c", (7, 8, 9), other_empty)]
    assert audit(items)["passed"]
    assert wall_geometry_audit(items)["passed"]


def test_independent_maps_of_one_archetype_are_allowed(built):
    # similar corridors or room graphs with different geometry are the generalization target
    assert wall_geometry_audit(_items(built[0]))["passed"]


def test_leak_through_reused_ids_or_seeds_is_caught(built):
    build = built[0]
    g = build.scenarios["train"][0]
    # a map from a different plan, so only the reused identifiers can trigger
    fresh = build_all(small_plan(seed=99, plan_id="other"))[0].scenarios["train"][0].scenario
    rep = _leak(build, fresh, family_id=g.family.family_id)
    assert {"family_ids_disjoint", "family_single_split"} <= _failed(rep)
    rep = _leak(build, fresh, lineage_id=g.lineage_id)
    assert {"lineage_ids_disjoint", "lineage_ids_unique"} <= _failed(rep)
    rep = _leak(build, fresh, seeds=(g.family.family_seed, 1, 2))
    assert _failed(rep) == {"seeds_disjoint"}
    rep = _leak(build, replace(fresh, seed=g.scenario.seed))
    assert _failed(rep) == {"seeds_disjoint", "scenario_seeds_unique"}
    clean = _leak(build, fresh)
    assert clean["passed"], _failed(clean)


def test_same_family_in_two_splits_is_caught(built):
    build = built[0]
    items = _items(build)
    moved = [replace(it, split="dev") if it.family_id == items[0].family_id and it is not items[0] else it
             for it in items]
    rep = audit(moved)
    assert {"family_ids_disjoint", "family_single_split", "topology_disjoint"} <= _failed(rep)


def test_pool_rejects_duplicates_and_near_duplicates():
    # a one-archetype, clutter-free plan forces look-alike maps: the pool must reject them
    d = small_plan().to_dict()
    d["spec"]["layout"]["mix"] = {"open_room": 1}
    d["spec"]["obstacles"]["density_pct"] = [0, 0]
    d["spec"]["grid"] = {"short_side": [8, 9], "aspect_pct": [100, 110]}
    d["spec"]["exits"]["count"] = [1, 1]
    d["families"] = {"train": 2, "dev": 1, "calibration": 1, "locked_test": 1}
    _, _, report, idx = build_all(SplitPlan.from_dict(d))
    rej = idx["pool"]["family_rejections"]
    assert rej.get("duplicate_family", 0) + rej.get("near_duplicate_family", 0) > 0
    assert report["passed"]


# ----- files, immutability and the seal -------------------------------------

def test_write_load_materialize_and_seal(tmp_path, built):
    plan = small_plan()
    pub, sealed = tmp_path / "public", tmp_path / "sealed"
    idx = write_split_files(plan, pub, sealed)
    assert sorted(p.name for p in pub.iterdir()) == ["calibration.json", "dev.json", "index.json", "train.json"]
    assert [p.name for p in sealed.iterdir()] == ["locked_test.json"]
    assert load_index(pub / "index.json") == idx
    assert plan_from_index(idx) == plan
    for s in PUBLIC_SPLITS:
        m = load_manifest(pub / f"{s}.json", index_data=idx)
        assert sum(1 for _ in materialize(m, plan.spec)) == len(m["scenarios"])
    with pytest.raises(LockedSplitError):
        load_manifest(sealed / "locked_test.json")
    with pytest.raises(LockedSplitError):
        load_manifest(sealed / "locked_test.json", purpose="teacher_tuning")
    raw = json.loads((sealed / "locked_test.json").read_text(encoding="utf-8"))
    with pytest.raises(LockedSplitError):
        next(materialize(raw, plan.spec))
    locked = load_manifest(sealed / "locked_test.json", index_data=idx, purpose=LOCKED_PURPOSE)
    assert locked["manifest_hash"] == idx["splits"][LOCKED]["manifest_hash"]
    assert verify_split_files(pub, sealed)["locked_test"] == "verified"
    assert verify_split_files(pub)["locked_test"] == "commitment_only"
    # rewriting the same plan is a no-op; a different plan into the same directory is refused
    write_split_files(plan, pub, sealed)
    with pytest.raises(ManifestError):
        write_split_files(small_plan(seed=8), pub, sealed)


def test_sealed_dir_must_be_outside_public_dir(tmp_path):
    with pytest.raises(ManifestError):
        write_split_files(small_plan(), tmp_path / "p", tmp_path / "p" / "sealed")
    with pytest.raises(ManifestError):
        write_split_files(small_plan(), tmp_path / "p", tmp_path)


def test_tampering_is_detected(tmp_path, built):
    plan = small_plan()
    pub = tmp_path / "public"
    idx = write_split_files(plan, pub, None)
    path = pub / "dev.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["scenarios"][0]["label"] = "feasible" if data["scenarios"][0]["label"] != "feasible" else "difficult"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ManifestError):
        load_manifest(path, index_data=idx)
    # a consistent re-hash still fails: the index binds the manifest hash, and regeneration differs
    from ember_v1.sim.canonical import digest
    body = {k: v for k, v in data.items() if k != "manifest_hash"}
    data["manifest_hash"] = digest(body)
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ManifestError):
        load_manifest(path, index_data=idx)
    with pytest.raises(ManifestError):
        verify_split_files(pub)
    # a scenario entry that no longer regenerates
    m = json.loads((pub / "train.json").read_text(encoding="utf-8"))
    m["scenarios"][0]["lineage_seed"] += 1
    m["manifest_hash"] = digest({k: v for k, v in m.items() if k != "manifest_hash"})
    with pytest.raises(ManifestError):
        list(materialize(m, plan.spec))


def test_locked_purpose_is_used_only_by_the_split_module_and_cli():
    users = []
    for path in (ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "LOCKED_PURPOSE" in text or "final_locked_evaluation" in text:
            users.append(path.relative_to(ROOT).as_posix())
    assert sorted(users) == ["src/ember_v1/gen/splits.py"]


def test_no_locked_manifest_is_committed():
    assert not [p for p in ROOT.rglob("locked_test*.json") if ".venv" not in p.parts]
    assert "data/sealed/" in (ROOT / ".gitignore").read_text(encoding="utf-8")


def test_cli_gen_and_verify_splits(tmp_path, monkeypatch, capsys):
    from ember_v1 import cli
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(small_plan().to_dict()), encoding="utf-8")
    pub, sealed = tmp_path / "pub", tmp_path / "sealed"
    monkeypatch.setattr("sys.argv", ["ember-v1", "gen-splits", "--plan", str(plan_path), "--out", str(pub),
                                     "--sealed-dir", str(sealed)])
    cli.main()
    assert json.loads(capsys.readouterr().out)["audit_passed"] is True
    monkeypatch.setattr("sys.argv", ["ember-v1", "verify-splits", "--dir", str(pub), "--sealed-dir", str(sealed)])
    cli.main()
    assert json.loads(capsys.readouterr().out)["locked_test"] == "verified"
