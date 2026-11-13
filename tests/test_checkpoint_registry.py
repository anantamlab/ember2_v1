"""The checkpoint registry agrees with tracked evidence, and the local verifier detects bad copies."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("verify_checkpoint", REPO / "scripts" / "verify_checkpoint.py")
vc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vc)

EXPECTED = {
    "r20": "f6b56137acc791bd1ed0862547a1dd3ca2ef81f336af6c969947840b46c66374",
    "r21-u96": "901e22e8be0af3f4ff7002b9041d16e4a0dacd8c602a79b66121429507275120",
    "r22-ppo-u16": "df5afa5a6d09031f5375503a8816eeeec26ee4faeaf4ee0ed040df821524154d",
    "r22-cf-u24": "05d665f4f7acedee57e2fa2acd9faa390df3f65b11de2719058cfb4ab8a00c72",
}


def test_registry_matches_tracked_identity_records():
    reg = vc.load_registry()
    ids = {c["id"]: c for c in reg["checkpoints"]}
    assert set(ids) == {"v1", *EXPECTED}
    for cid, digest in EXPECTED.items():
        item = ids[cid]
        assert item["actor_digest"] == f"sha256:{digest}"
        recorded, files = vc.identity(item)
        assert recorded == item["actor_digest"]
        assert files and all({"path", "sha256", "size"} <= set(f) for f in files)
        assert item["parent"] in ids
    v1 = ids["v1"]
    hf = json.loads((REPO / "data/hf/registry.json").read_text(encoding="utf-8"))
    art = vc.dig(hf, v1["identity_record"]["pointer"])
    assert art["manifest_commit"] == v1["storage"]["revision"] == "1fc405f723718e308c8a22b137fee4a025086fde"
    assert art["manifest_sha256"] == v1["storage"]["manifest_sha256"]


def test_selected_r22_candidate_is_ppo_update_16():
    reg = vc.load_registry()
    frozen = [c["id"] for c in reg["checkpoints"] if c.get("frozen_candidate")]
    assert frozen == ["r22-ppo-u16"]
    best = json.loads((REPO / "data/rl/evidence/r22-counterfactual-ppo@1/best_checkpoint.json").read_text(encoding="utf-8"))
    assert best["frozen_candidate"]["identity"]["actor_digest"] == f"sha256:{EXPECTED['r22-ppo-u16']}"
    assert best["final_evaluation_run"] is False


def _fake(tmp_path):
    content = {"actor/a.bin": b"alpha", "trainer.pt": b"beta"}
    for rel, data in content.items():
        (tmp_path / "ckpt" / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "ckpt" / rel).write_bytes(data)
    files = [{"path": p, "size": len(d), "sha256": hashlib.sha256(d).hexdigest()} for p, d in content.items()]
    (tmp_path / "id.json").write_text(json.dumps({"identity": {"actor_digest": "sha256:x", "files": files}}))
    item = {"id": "t", "actor_digest": "sha256:x", "storage": {"kind": "workstation"},
            "identity_record": {"file": "id.json", "digest_key": ["identity", "actor_digest"],
                                "files_key": ["identity", "files"]}}
    return item, tmp_path / "ckpt"


def test_verify_pass_and_detects_tampering(tmp_path):
    item, root = _fake(tmp_path)
    assert vc.verify(item, root, repo=tmp_path)["verdict"] == "PASS"
    (root / "trainer.pt").write_bytes(b"betA")
    (root / "actor" / "a.bin").unlink()
    out = vc.verify(item, root, repo=tmp_path)
    assert out["verdict"] == "FAIL"
    assert out["problems"] == ["missing actor/a.bin", "sha256 trainer.pt"]


def test_verify_refuses_hf_entry_and_digest_drift(tmp_path):
    with pytest.raises(ValueError, match="hfstore"):
        vc.verify(vc.entry(vc.load_registry(), "v1"), tmp_path)
    item, root = _fake(tmp_path)
    item["actor_digest"] = "sha256:y"
    with pytest.raises(ValueError, match="identity record"):
        vc.verify(item, root, repo=tmp_path)
