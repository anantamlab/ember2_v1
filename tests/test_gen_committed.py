"""The committed p1-02-v1 split manifests regenerate exactly (about 30 s on one CPU core)."""
from __future__ import annotations

from pathlib import Path

from ember_v1.gen.splits import LOCKED, PUBLIC_SPLITS, SplitPlan, load_index, load_manifest, verify_split_files

DIR = Path(__file__).resolve().parents[1] / "data" / "splits" / "p1-02-v1"


def test_committed_index_uses_the_default_plan_and_passed_its_audit():
    idx = load_index(DIR / "index.json")
    assert SplitPlan.from_dict(idx["plan"]) == SplitPlan.default()
    assert idx["audit"]["passed"] and all(c["passed"] for c in idx["audit"]["checks"].values())
    locked = idx["splits"][LOCKED]
    assert locked["file"] is None and locked["sealed"] and locked["families"] == 24
    assert LOCKED not in idx["coverage"]
    for s in PUBLIC_SPLITS:
        load_manifest(DIR / f"{s}.json", index_data=idx)


def test_committed_manifests_regenerate_exactly():
    report = verify_split_files(DIR)
    assert report["public_splits"] == "verified" and report["locked_test"] == "commitment_only"
    assert report["locked_test_secrecy"] == "procedural_not_blind" and report["wall_geometry_audit"] == "passed"


def test_failed_lineages_are_kept_with_reasons_and_counts():
    """The 5 exhausted lineages stay recorded: unreachable targets on maps with no pocket and no cut cell."""
    idx = load_index(DIR / "index.json")
    assert {s: idx["splits"][s]["failed_lineages"] for s in idx["splits"]} == {
        "train": 3, "dev": 0, "calibration": 0, "locked_test": 2}
    assert {s: idx["splits"][s]["scenarios"] for s in idx["splits"]} == {
        "train": 189, "dev": 48, "calibration": 48, "locked_test": 70}
    train = load_manifest(DIR / "train.json", index_data=idx)
    fams = {f["family_id"]: f for f in train["families"]}
    for fl in train["failed_lineages"]:
        f = fams[fl["family_id"]]
        assert fl["target"] == "unreachable" and fl["attempts"] == 60 and sum(fl["reasons"].values()) == 60
        assert f["pocket_cells"] == 0 and f["cut_cells"] == 0
