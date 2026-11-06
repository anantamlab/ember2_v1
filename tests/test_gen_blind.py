"""Blind test set from a private salt: secret handling, disjointness, commitment only, no training access.

Each test makes a throw-away random salt in pytest's temporary directory. No
salt is stored in the repository, and no test salt is a real blind salt.
"""
from __future__ import annotations

import json
import pickle
import re
import secrets
from pathlib import Path

import pytest

from ember_v1.gen import analysis as A
from ember_v1.gen.blind import (BLIND_SPLIT, BlindPlan, BlindSecret, BlindSecretError, build_blind, load_secret,
                                materialize_blind, verify_blind)
from ember_v1.gen.splits import (SPLITS, LockedSplitError, ManifestError, build_splits, fill_family_pool,
                                 load_manifest, materialize, write_split_files)
from gen_helpers import small_plan

ROOT = Path(__file__).resolve().parents[1]
PLAN = BlindPlan.from_dict({"plan_id": "blind-test", "families": 3, "lineages_per_family": 2})


def _salt_file(tmp_path: Path, name: str = "salt.hex") -> Path:
    path = tmp_path / name
    path.write_text(secrets.token_hex(32), encoding="ascii")
    return path


@pytest.fixture(scope="module")
def public(tmp_path_factory):
    d = tmp_path_factory.mktemp("public")
    write_split_files(small_plan(), d, None)
    return d / "index.json"


@pytest.fixture(scope="module")
def blind(tmp_path_factory, public):
    secret = load_secret(_salt_file(tmp_path_factory.mktemp("salt")), environ={})
    private, commitment = build_blind(secret, PLAN, public)
    return secret, private, commitment


def test_salt_loading_is_guarded(tmp_path):
    path = _salt_file(tmp_path)
    assert repr(load_secret(path, environ={})) == "BlindSecret(<redacted>)"
    for env in ({"CI": "true"}, {"GITHUB_ACTIONS": "true"}):
        with pytest.raises(BlindSecretError):
            load_secret(path, environ=env)
    with pytest.raises(BlindSecretError):            # a salt file inside the repository
        load_secret(path, environ={}, repo_root=tmp_path)
    short = tmp_path / "short.hex"
    short.write_text("ab" * 16, encoding="ascii")
    with pytest.raises(BlindSecretError):
        load_secret(short, environ={})
    with pytest.raises(BlindSecretError):
        load_secret(tmp_path / "missing.hex", environ={})
    with pytest.raises(TypeError):
        pickle.dumps(load_secret(path, environ={}))


def test_blind_families_are_disjoint_from_every_public_split(blind):
    secret, private, commitment = blind
    public = build_splits(small_plan())
    pub_fams = [f for s in SPLITS for f, _ in public.families[s]]
    assert len(private["families"]) == PLAN.families
    pub_topo = {f.topology_hash for f in pub_fams}
    pub_seeds = {f.family_seed for f in pub_fams}
    for f in private["families"]:
        assert f["topology_hash"] not in pub_topo and f["family_seed"] not in pub_seeds
    assert private["audit"]["passed"] and private["wall_geometry_audit"]["passed"]
    assert commitment["disjoint_from_public"] is True


def test_commitment_holds_only_hashes_and_counts(blind):
    secret, private, commitment = blind
    assert set(commitment) == {"blind_version", "plan_id", "generator_revision", "spec_hash", "public_index_hash",
                               "secret_commitment", "manifest_commitment", "families", "scenarios",
                               "failed_lineages", "disjoint_from_public", "distinct_maps_audited",
                               "commitment_hash"}
    text = repr(commitment)
    for f in private["families"]:
        assert f["family_id"] not in text and str(f["family_seed"]) not in text
    for e in private["scenarios"]:
        assert e["lineage_id"] not in text and e["scenario_hash"] not in text
    assert not re.search(r"feasible|difficult|unreachable|label|coverage", text)


def test_rebuild_is_deterministic_and_bound_to_the_salt(tmp_path, blind, public):
    secret, private, commitment = blind
    verify_blind(secret, private, commitment, public)
    other = load_secret(_salt_file(tmp_path, "other.hex"), environ={})
    p2, c2 = build_blind(other, PLAN, public)
    assert c2["manifest_commitment"] != commitment["manifest_commitment"]
    assert c2["secret_commitment"] != commitment["secret_commitment"]
    with pytest.raises(ManifestError):
        verify_blind(other, private, commitment, public)


def test_blind_manifest_is_refused_by_the_public_loaders(tmp_path, blind):
    secret, private, _ = blind
    with pytest.raises(LockedSplitError):
        materialize(private, small_plan().spec)
    path = tmp_path / "private.json"
    path.write_text(json.dumps(private), encoding="utf-8")
    with pytest.raises(ManifestError):
        load_manifest(path)
    got = list(materialize_blind(secret, private, small_plan().spec))
    assert len(got) == len(private["scenarios"]) and all(e["family_id"] for e, _ in got)
    other = BlindSecret(secrets.token_bytes(32))
    with pytest.raises(ManifestError):
        materialize_blind(other, private, small_plan().spec)


def test_pool_rejects_a_candidate_that_copies_a_public_family():
    public = build_splits(small_plan())
    pub_fams = [f for s in SPLITS for f, _ in public.families[s]]
    target = next(f for f in pub_fams if f.attempt == 0)
    pool, _, info = fill_family_pool(small_plan().spec, 1, lambda slot: target.family_seed, reference=pub_fams,
                                     wall_rule=True)
    assert info["family_rejections"].get("duplicate_family", 0) >= 1
    assert pool[0].topology_hash != target.topology_hash
    assert not any(A.near_duplicate(pool[0].signature, f.signature)[0]
                   or A.wall_near_duplicate(pool[0].signature, f.signature)[0] for f in pub_fams)


def test_only_the_blind_module_reaches_blind_scenarios():
    users, regen = [], []
    for path in (ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(ROOT).as_posix()
        if rel != "src/ember_v1/gen/blind.py" and re.search(r"gen\.blind|from \.blind|import blind", text):
            users.append(rel)
        if "_regenerate" in text:
            regen.append(rel)
    assert users == []                     # no teacher, training or tuning module imports it
    assert sorted(regen) == ["src/ember_v1/gen/blind.py", "src/ember_v1/gen/splits.py"]
    assert BLIND_SPLIT not in SPLITS
