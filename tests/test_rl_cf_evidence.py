import copy
import hashlib
import json

import pytest

from ember_v1.rl_cf_evidence import lineage_roots, reconcile_history, paired_statistics


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_resume_history_reconciles_prior_raw_evidence_and_rejects_tampering(tmp_path):
    old, current = tmp_path/"attempt-1", tmp_path/"attempt-2"
    common = {"config_sha256": "cfg", "parent_binding": {"actor_digest": "verified-parent"}}
    put(old/"execution.json", {**common, "resume": None})
    resume = old/"resume"/"round-0001"
    resume.mkdir(parents=True)
    (resume/"state.pt").write_bytes(b"fixture-paired-state")
    put(resume/"identity.json", {"config_sha256": "cfg", "round": 1,
        "sha256": hashlib.sha256((resume/"state.pt").read_bytes()).hexdigest()})
    put(current/"execution.json", {**common, "resume": str(resume)})
    history = {}
    for arm in ("ppo", "cf"):
        episodes, updates = [], []
        for i, root in enumerate((old, current)):
            ep = {"label": f"train-{i}", "scenario_id": f"s{i}", "steps": 2}
            update = {"update": i+1, "evidence_root": str(root), "episode_ids": [f"s{i}"],
                      "optimizer_steps": 1, "stage": 0, "rollout": {"episodes": 1}}
            for name, row in (("episodes", ep), ("updates", update)):
                p = root/arm/(name+".jsonl")
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(row)+"\n")
            episodes.append(ep)
            updates.append(update)
        history[arm] = {"episodes": episodes, "updates": updates}
    assert len(lineage_roots(current)) == 2
    proof = reconcile_history(current, history)
    assert len(proof) == 4
    p = old/"cf"/"updates.jsonl"
    changed = json.loads(p.read_text())
    changed["optimizer_steps"] = 99
    p.write_text(json.dumps(changed)+"\n")
    with pytest.raises(ValueError, match="raw update"):
        reconcile_history(current, history)


def test_paired_uncertainty_includes_outcomes_behavior_and_latency():
    a = [{"scenario_id": f"s{i}", "people": 10, "evacuated": 4, "casualties": 3, "unresolved": 3,
          "stay": 20, "person_steps": 100, "blocked": 30, "attempted": 80, "steps": 10,
          "blocked_conflict": 5, "blocked_occupied": 5, "blocked_wall": 10,
          "repeated_ineffective_moves": 8, "inference_seconds": .5}
         for i in range(4)]
    b = [{**r, "evacuated": 6, "casualties": 2, "unresolved": 2, "stay": 10, "blocked": 10,
          "blocked_conflict": 2, "blocked_occupied": 3, "blocked_wall": 2,
          "repeated_ineffective_moves": 2, "inference_seconds": .4}
         for r in a]
    result = paired_statistics(a, b, repeats=200)
    assert result["evacuated"]["lower_95"] > 0
    for key in ("casualties", "unresolved", "stay_rate", "blocked_rate", "wall_boundary_rate", "coordination_conflict_rate",
                "repeated_ineffective_rate", "mean_joint_latency_ms"):
        assert result[key]["upper_95"] < 0
    with pytest.raises(ValueError, match="matched"):
        paired_statistics(a, b[:-1])
