"""Independent CPU verification of online PPO and same-state counterfactual artifacts."""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .rl_cf_training import CounterfactualObjective, verify_parent
from .rl_cf_evidence import lineage_roots, reconcile_history, reporting_statistics, read_rows as evidence_rows
from .rl_counterfactual import fixed_t1, verify_study
from .rl_online import verify_snapshot
from .rl_run import write, summarize
from .rl_verify import verify as verify_online
from .sim import Scenario, Simulator


def rows(path):
    return [json.loads(s) for s in path.open()] if path.exists() else []


def verify_study_path(path):
    study = json.loads(Path(path).read_text())
    return study["study_hash"], verify_study(study, continuation=fixed_t1)


def verify(root):
    root = Path(root)
    summary = json.loads((root/"summary.json").read_text())
    execution = json.loads((root/"execution.json").read_text())
    if summary["status"] != "COMPLETE" or summary["final_evaluation_run"]:
        raise ValueError("experiment incomplete or final evaluation boundary violated")
    if verify_parent(Path(execution["phase2_root"])) != execution["parent_binding"]:
        raise ValueError("Phase 2 parent identity changed")
    online = {"baselines": verify_online(root)}
    for arm in ("ppo", "cf"):
        online[arm] = verify_online(root/arm)
    history = json.loads((root/"training_history.json").read_text())
    if any(len(history[a]["updates"]) != summary["completed_matched_rounds"] or
           len(history[a]["episodes"]) != summary["completed_matched_rounds"]*execution["config"]["episodes_per_update"]
           for a in ("ppo", "cf")):
        raise ValueError("full matched training history is incomplete")
    origins = reconcile_history(root, history)
    ancestry_audits = []
    for (origin, arm), expected_keys in origins.items():
        if origin.resolve() == root.resolve():
            continue  # current complete traces already replayed above
        view = root/"audit_inputs"/origin.name/arm
        view.mkdir(parents=True, exist_ok=True)
        for name in ("execution.json", "scenarios.json"):
            (view/name).write_text((origin/arm/name).read_text())
        for name in ("episodes.jsonl", "transitions.jsonl", "ppo_targets.jsonl"):
            selected = [r for r in evidence_rows(origin/arm/name, allow_trailing_partial=(origin/"failure.json").exists())
                        if (r["label"], r["scenario_id"]) in expected_keys]
            (view/name).write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in selected))
        ancestry_audits.append({"source": str(origin), "arm": arm, "proof": verify_online(view)})
    for arm in ("ppo", "cf"):
        if summarize(history[arm]["episodes"]) != summary["training"][arm]:
            raise ValueError("reported aggregate training outcomes differ from reconciled raw evidence")
    if len(history["ppo"]["updates"]) != len(history["cf"]["updates"]):
        raise ValueError("arms have unequal on-policy update counts")
    if any(a["stage"] != b["stage"] or a["rollout"]["episodes"] != b["rollout"]["episodes"] or a["episode_ids"] != b["episode_ids"]
           for a, b in zip(history["ppo"]["updates"], history["cf"]["updates"])):
        raise ValueError("arms did not share TRAIN curriculum/exposure")
    indices = [entry for u in history["cf"]["updates"] for entry in u["counterfactual_studies"]]
    from .teacher.t1 import T1CongestionAware
    from .provenance import code_fingerprint
    expected_continuation = {"identity": T1CongestionAware().identity.to_dict(),
                             "code_fingerprint": code_fingerprint(T1CongestionAware)}
    visits_cache, totals, teacher = {}, {"studies": 0, "branches": 0, "transitions": 0, "eligible_pairs": 0}, {}
    actual_pairs = {}
    with ProcessPoolExecutor(max_workers=execution["config"]["cpu_workers"],
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        branch_proofs = list(pool.map(verify_study_path, [e["path"] for e in indices]))
    for entry, (verified_hash, branch_proof) in zip(indices, branch_proofs):
        path = Path(entry["path"])
        study = json.loads(path.read_text())
        if study["study_hash"] != entry["study_hash"] or verified_hash != study["study_hash"]:
            raise ValueError("CF index and actual study differ")
        if study["continuation"] != expected_continuation:
            raise ValueError("counterfactual continuation identity is not the frozen T1")
        factual_root = path.parent.parent
        if factual_root not in visits_cache:
            visits_cache[factual_root] = {(r["label"], r["scenario_id"], r["observation"]["step"]): r
                                          for r in rows(factual_root/"cf"/"transitions.jsonl")}
        source = study["source"]
        key = source["on_policy_label"], source["scenario_id"], source["step"]
        if (key[0], key[1]) not in origins.get((factual_root.resolve(), "cf"), set()):
            raise ValueError("counterfactual source is outside the committed model training history")
        if visits_cache[factual_root].get(key) != study["factual_visit"]:
            raise ValueError("CF state is not found in the real on-policy rollout")
        proof = branch_proof
        if not proof["visited_source_verified"] or not proof["continuation_recomputed"]:
            raise ValueError("incomplete causal source/continuation proof")
        totals["studies"] += 1
        totals["branches"] += proof["branches"]
        totals["transitions"] += proof["transitions"]
        pairs = CounterfactualObjective([study]).pairs
        totals["eligible_pairs"] += len(pairs)
        for pair in pairs:
            actual_pairs[study["study_hash"]] = pair
        for name, origin in study["candidate_origins"].items():
            mean = study["estimates"][name]["paired"]["utility"]["mean"]
            for candidate in origin:
                label = candidate["name"]
                s = teacher.setdefault(label, {"states": 0, "helped": 0, "harmed": 0, "tied": 0, "utility_sum": 0.})
                s["states"] += 1
                s["helped" if mean > 1e-12 else "harmed" if mean < -1e-12 else "tied"] += 1
                s["utility_sum"] += mean
    for update in history["cf"]["updates"]:
        expected_pairs = sum(entry["study_hash"] in actual_pairs for entry in update["counterfactual_studies"])
        if expected_pairs != update["counterfactual_pairs"]:
            raise ValueError("reported CF pair count differs from measured outcome records")
        if update["counterfactual_pairs"]:
            applied = [m for m in update["minibatches"] if "gradient_norms" in m and "counterfactual" in m]
            if not applied or not any(m["counterfactual"].get("independent_auxiliary_head_gradient_norm", 0) > 0 for m in applied):
                raise ValueError("eligible CF pairs lack proof of an applied finite auxiliary gradient")
            for minibatch in applied:
                observed = minibatch["counterfactual"]
                pair = actual_pairs.get(observed["study_hash"])
                if pair is None or any(observed[k] != pair[k] for k in ("chosen", "weight", "reference_margin")):
                    raise ValueError("applied CF objective differs from the verified measured pair")
    if any(u["auxiliary_optimizer_steps"] or any("counterfactual" in m for m in u["minibatches"])
           for u in history["ppo"]["updates"]):
        raise ValueError("counterfactual gradients leaked into the PPO-only control")
    if not totals["eligible_pairs"] or not summary["auxiliary_optimizer_steps"]:
        raise ValueError("no genuine counterfactual learning occurred")
    best = json.loads((root/"best_checkpoint.json").read_text())
    for arm, item in best["trained_arms"].items():
        verify_snapshot(Path(item["checkpoint"]), execution["config_sha256"])
        if item["fresh_reload_max_logit_error"] > 1e-5:
            raise ValueError("saved arm failed exact upstream reload")
    scenarios = json.loads((root/"scenarios.json").read_text())
    by_id = {s["scenario_id"]: Scenario.from_dict(s) for s in scenarios["report"]}
    teacher_steps, teacher_failures = 0, 0
    for episode in rows(root/"teacher_episodes.jsonl"):
        sim = Simulator(by_id[episode["scenario_id"]])
        for decision, record in zip(episode["decisions"], episode["records"]):
            if sim.observe() != decision["observation"] or sim.step(decision["actions"]) != record:
                raise ValueError("reference teacher rollout does not replay")
            teacher_steps += 1
        if episode["status"] == "COMPLETE" and not sim.terminated:
            raise ValueError("reference teacher success is incomplete")
        teacher_failures += episode["status"] != "COMPLETE"
    all_collected = {arm: sum(sum(r["label"].startswith("train-") for r in evidence_rows(p/arm/"transitions.jsonl",
                                   allow_trailing_partial=(p/"failure.json").exists())) for p in lineage_roots(root))
                     for arm in ("ppo", "cf")}
    paired = reporting_statistics(root)
    write(root/"verified_counterfactual_dataset.json", {"kind": "fresh-TRAIN-visited-state-counterfactuals",
          "studies": indices, "verified": True, "on_policy_episodes": False})
    result = {"status": "PASS", "online": online, "resume_ancestry_audits": ancestry_audits,
              "all_collected_train_environment_steps": all_collected,
              "committed_train_environment_steps": {a: summary["training"][a]["steps"] for a in ("ppo", "cf")},
              "paired_reporting": paired, "counterfactual": totals,
              "teacher_alternative_effects": teacher, "teacher_transitions_replayed": teacher_steps,
              "teacher_failures_preserved": teacher_failures, "final_evaluation_run": False}
    write(root/"independent_counterfactual_audit.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("evidence", type=Path)
    args = p.parse_args()
    print(json.dumps(verify(args.evidence)))
