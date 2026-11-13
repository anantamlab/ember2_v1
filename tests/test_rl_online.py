import pytest

from ember_v1.rl_online_scenarios import build_scenarios, select_checkpoint, STRATA
from ember_v1.rl_env import EvacuationEnv
from ember_v1.gen.analysis import topology_hash


def test_diverse_generator_reproduces_and_isolates_geometry():
    sets = {s: build_scenarios("test-online", s, 64, 48) for s in ("train", "select", "report")}
    assert sets["train"] == build_scenarios("test-online", "train", 64, 48)
    hashes = []
    for split, rows in sets.items():
        assert {m["stratum"] for _, m in rows} == set(STRATA)
        assert {s.fire_model for s, _ in rows} == {"deterministic", "stochastic"}
        assert min(len(s.people) for s, _ in rows) <= 5
        assert max(len(s.people) for s, _ in rows) >= 18
        for s, meta in rows:
            assert meta["split"] == split
            EvacuationEnv(s).reset()
            hashes.append(topology_hash(s.height, s.width, s.obstacles, [p for _, p in s.exits]))
    assert len(set(hashes)) == len(hashes)


def test_selection_uses_only_selection_dev_and_casualty_constraint():
    baseline = {"people": 100, "evacuated": 40, "casualties": 30, "unresolved": 30}
    bad = {"people": 100, "evacuated": 65, "casualties": 35, "unresolved": 0}
    good = {"people": 100, "evacuated": 50, "casualties": 30, "unresolved": 20}
    rows = [{"update": 16, "selection": bad}, {"update": 32, "selection": good}]
    assert select_checkpoint(rows, baseline)["update"] == 32
    assert select_checkpoint(rows[:1], baseline)["safety_eligible"] is False
    with pytest.raises(ValueError):
        build_scenarios("x", "hidden", 8, 48)


def test_resume_rejects_changed_configuration_and_checkpoint(tmp_path):
    import json
    import hashlib
    from ember_v1.rl_online import verify_snapshot
    actor = tmp_path / "actor.bin"
    actor.write_bytes(b"trained actor")
    (tmp_path / "identity.json").write_text(json.dumps({"config_sha256": "frozen", "files": [
        {"path": "actor.bin", "sha256": hashlib.sha256(actor.read_bytes()).hexdigest()}]}))
    verify_snapshot(tmp_path, "frozen")
    with pytest.raises(RuntimeError, match="config"):
        verify_snapshot(tmp_path, "changed")
    actor.write_bytes(b"corruption")
    with pytest.raises(RuntimeError, match="hash"):
        verify_snapshot(tmp_path, "frozen")
