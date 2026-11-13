"""Matched-budget PPO-only versus counterfactual-assisted PPO from verified Phase 2."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import multiprocessing
import platform
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .rl_cf_scenarios import scenarios
from .rl_cf_training import (CounterfactualObjective, activate, advance_shared_curriculum, capture, cpu_copy,
                             jobs_from_visits, trained_checkpoint_candidates, verify_parent)
from .rl_counterfactual import curriculum_stage, evaluate_job
from .rl_env import outcomes
from .rl_online import hash_path, identity, verify_snapshot
from .rl_online_scenarios import select_checkpoint
from .rl_run import REPO, append, collect, parameter_delta, summarize, telemetry, verify_inputs, write
from .sim import Simulator
from .sim.canonical import digest


class CounterfactualJobDeadline(BaseException):
    """Bypasses individual teacher exception handling and ends the entire CPU job."""


def rows(path):
    return [json.loads(s) for s in path.open()] if path.exists() else []


def timed_cf_job(job):
    def expired(signum, frame):
        raise CounterfactualJobDeadline("counterfactual CPU job cap")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 120)
    try:
        return evaluate_job(job)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def teacher_episode(payload):
    scenario, name, seconds = payload
    from .teacher.registry import teacher_class
    from .provenance import code_fingerprint
    started = time.monotonic()
    def expired(signum, frame):
        raise TimeoutError("default teacher episode wall cap")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    records, decisions, counts = [], [], Counter()
    try:
        cls = teacher_class(name)
        policy = cls()  # frozen default configuration, distinct from bounded CF candidates
        sim = Simulator(scenario)
        while not sim.terminated:
            observation = sim.observe()
            decision = policy.decide(copy.deepcopy(observation))
            actions = decision.actions
            record = sim.step(actions)
            records.append(record)
            decisions.append({"observation": observation, "actions": actions,
                              "wall_ms": decision.runtime["wall_ms"], "diagnostics": decision.diagnostics})
            counts.update(steps=1, person_steps=len(actions), stay=sum(a == "stay" for a in actions.values()),
                          attempted=sum(a != "stay" for a in actions.values()),
                          blocked=sum(m["outcome"].startswith("blocked_") for m in record["movements"].values()))
            counts.update(m["outcome"] for m in record["movements"].values())
        stats = outcomes(sim.observe())
        utility = (stats["evacuated"]-2*stats["casualties"]-.75*stats["unresolved"])/stats["people"]
        for key in ("stay", "attempted", "blocked"):
            counts.setdefault(key, 0)
        return {"status": "COMPLETE", "label": name, "scenario_id": scenario.scenario_id, **stats, **dict(counts),
                "reward": utility, "terminal_reward": utility, "seconds": time.monotonic()-started,
                "identity": policy.identity.to_dict(), "code_fingerprint": code_fingerprint(cls),
                "records": records, "decisions": decisions}
    except Exception:
        return {"status": "FAILED", "label": name, "scenario_id": scenario.scenario_id,
                "seconds": time.monotonic()-started, "traceback": traceback.format_exc(),
                "records": records, "decisions": decisions}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def save_resume(out, states, round_index, config_hash, binding):
    """Keep two complete paired resume points, plus every DEV-selected actor snapshot."""
    import torch
    folder = out / "resume" / f"round-{round_index:04d}"
    folder.mkdir(parents=True, exist_ok=False)
    torch.save({"arms": states, "completed_rounds": round_index, "config_sha256": config_hash,
                "parent_binding": binding}, folder / "state.pt")
    write(folder / "identity.json", {"sha256": hash_path(folder/"state.pt"), "round": round_index,
                                    "config_sha256": config_hash})
    completed = sorted(p.parent for p in (out/"resume").glob("round-*/identity.json"))
    for old in completed[:-2]:
        if old.resolve().parent != (out/"resume").resolve():
            raise ValueError("resume cleanup escaped evidence root")
        (old/"state.pt").unlink()
        (old/"identity.json").unlink()
        old.rmdir()
    return folder


def save_arm(actor, state, path, cfg_hash, parent, round_index, sha):
    import torch
    path.mkdir(parents=True, exist_ok=False)
    actor.save(path/"actor")
    torch.save(state, path/"trainer.pt")
    identity(path, actor, cfg_hash, parent, round_index, sha)


def run(args):
    cfg = json.loads(args.config.read_text())
    if cfg["status"] != "FROZEN" or sys.version_info[:2] != (3, 11):
        raise ValueError("frozen configuration and Python 3.11 required")
    binding = verify_parent(args.phase2)  # dependency gate BEFORE reserving any attempt
    base = args.output_root/cfg["run_id"]
    base.mkdir(parents=True, exist_ok=True)
    if any(base.glob("attempt-*/summary.json")) or not 1 <= args.attempt <= cfg["max_attempts"]:
        raise ValueError("completed experiment or invalid attempt")
    for old in base.glob("attempt-*/execution.json"):
        if not (old.parent/"failure.json").exists():
            raise ValueError("reconcile previous active/interrupted attempt before retry")
    spent = sum(json.loads(p.read_text())["elapsed_seconds"] for p in base.glob("attempt-*/failure.json"))
    budget = cfg["wall_cap_seconds"]-spent
    prior_episodes = {a: sum(1 for p in base.glob(f"attempt-*/{a}/episodes.jsonl") for r in rows(p)
                             if r["label"].startswith("train-")) for a in ("ppo", "cf")}
    if budget <= cfg["evaluation_reserve_seconds"]:
        raise ValueError("Phase 3 aggregate budget exhausted")
    out = base/f"attempt-{args.attempt}"
    out.mkdir(exist_ok=False)
    started, stop = time.monotonic(), threading.Event()
    monitor = threading.Thread(target=telemetry, args=(out, stop), daemon=True)
    monitor.start()
    def deadline(signum, frame):
        raise TimeoutError("Phase 3 aggregate process deadline")
    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, max(1, budget-15))
    try:
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip():
            raise ValueError("commit experiment source before execution")
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        config_hash = hash_path(args.config)
        write(out/"execution.json", {"config": cfg, "config_sha256": config_hash, "code_commit": sha,
              "started_unix": time.time(), "attempt": args.attempt, "prior_wall_seconds": spent,
              "parent_binding": binding, "phase2_root": str(args.phase2), "python": platform.python_version(),
              "command": sys.argv, "resume": str(args.resume) if args.resume else None})
        write(out/"parent_binding.json", binding)
        verify_inputs(args.model_root, cfg)
        namespace = cfg["scenario_namespace"]
        train_namespace = namespace if args.attempt == 1 else namespace+f"-attempt-{args.attempt}"
        pools = {stage: scenarios(train_namespace, "train", cfg["updates"]*cfg["episodes_per_update"], stage=stage,
                                  max_steps=cfg["max_steps"]) for stage in range(5)}
        selection = scenarios(namespace, "select", cfg["dev_episodes"], max_steps=cfg["max_steps"])
        reporting = scenarios(namespace, "report", cfg["dev_episodes"], max_steps=cfg["max_steps"])
        all_rows = [row for pool in pools.values() for row in pool]+selection+reporting
        from .gen.analysis import topology_hash
        def topo(s):
            return topology_hash(s.height, s.width, s.obstacles, [p for _, p in s.exits])
        hashes = [topo(s) for s, _ in all_rows]
        previous_scenarios = json.loads((args.phase2/"scenarios.json").read_text())
        from .sim import Scenario
        previous_hashes = {topo(Scenario.from_dict(s)) for split in ("train", "select", "report")
                           for s in previous_scenarios[split]}
        if len(set(hashes)) != len(hashes) or set(hashes) & previous_hashes:
            raise ValueError("counterfactual TRAIN/DEV/parent topology overlap")
        scenario_document = {"train": [s.to_dict() for pool in pools.values() for s, _ in pool],
            "select": [s.to_dict() for s, _ in selection], "report": [s.to_dict() for s, _ in reporting],
            "metadata": {s.scenario_id: {**m, "topology_hash": topo(s)} for s, m in all_rows},
            "train_dev_topology_overlap": 0, "phase2_overlap": 0}
        write(out/"scenarios.json", scenario_document)
        dev_select, dev_report = [s for s, _ in selection], [s for s, _ in reporting]
        by_id = {s.scenario_id: s for s, _ in all_rows}
        import unsloth
        import torch
        from .rl_ppo import V1Actor, make_value_head, optimize
        from .student_train import config_from_protocol, load_protocol
        original_identity = json.loads((args.model_root/"identity.json").read_text())
        student_cfg = config_from_protocol(load_protocol(REPO/original_identity["protocol"]["path"]))
        if student_cfg["protocol_hash"] != original_identity["protocol"]["protocol_hash"]:
            raise ValueError("V1 inference protocol differs")
        for key in ("chunk_k", "encoding", "question_style", "max_len", "precision", "reservations"):
            if student_cfg[key] != original_identity["config"][key]:
                raise ValueError(f"original V1 inference configuration differs: {key}")
        write(out/"runtime.json", {"python": platform.python_version(), "torch": torch.__version__,
              "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "student_config": student_cfg})
        torch.set_num_threads(4)
        torch.manual_seed(cfg["seed"])
        actor = V1Actor(student_cfg, Path(binding["checkpoint"])/"actor")
        if actor.ad.weights_digest() != binding["actor_digest"]:
            raise ValueError("loaded actor is not the selected verified Phase 2 checkpoint")
        critic = make_value_head()
        parent_state = torch.load(Path(binding["checkpoint"])/"trainer.pt", map_location="cpu", weights_only=False)
        critic.load_state_dict(parent_state["critic"])
        actor.set_trainable(True)
        optimizer = torch.optim.AdamW(actor.parameter_groups(cfg)+[
            {"params": list(critic.parameters()), "lr": cfg["value_lr"], "name": "critic"}], weight_decay=0)
        optimizer.load_state_dict(parent_state["optimizer"])
        torch.set_rng_state(parent_state["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([v.cpu() for v in parent_state["cuda_rng"]])
        parent_steps = sorted({int(v["step"]) for v in parent_state["optimizer"]["state"].values()})
        del parent_state
        # Capacity test on a fresh largest TRAIN state, with no optimizer step. Exact
        # weights/RNG are preserved; this is not an SFT or RL training update.
        from .rl_env import EvacuationEnv
        largest = max((s for s, _ in pools[4]), key=lambda s: (len(s.people), s.height*s.width))
        probe_rng, probe_cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        probe_batch = actor.prepare([EvacuationEnv(largest).reset()[0]])
        with torch.enable_grad():
            _, _, probe_logp, _, _ = actor.sample(probe_batch, greedy=True)
            (-probe_logp.mean()).backward()
        probe_norms = {}
        for name, predicate in (("lora", lambda n: "lora_" in n), ("head", lambda n: n.startswith("head."))):
            grads = [p.grad for n, p in actor.model.named_parameters() if predicate(n) and p.grad is not None]
            if not grads or any(not torch.isfinite(g).all() for g in grads):
                raise FloatingPointError("largest-state preflight gradient is missing/nonfinite")
            probe_norms[name] = sum(float(g.float().square().sum()) for g in grads)**.5
            if probe_norms[name] <= 0:
                raise FloatingPointError("largest-state preflight has zero gradient")
        optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(probe_rng)
        torch.cuda.set_rng_state_all(probe_cuda_rng)
        if actor.ad.weights_digest() != binding["actor_digest"]:
            raise ValueError("memory preflight changed the frozen parent")
        write(out/"largest_state_preflight.json", {"people": len(largest.people), "grid": [largest.height, largest.width],
              "gradient_norms": probe_norms, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "weights_unchanged": True, "optimizer_steps": 0})
        del probe_batch, probe_logp, grads
        torch.cuda.empty_cache()
        initial = capture(actor, critic, optimizer)
        initial["actor_digest"] = actor.ad.weights_digest()
        states = {arm: {**cpu_copy(initial), "episodes": [], "updates": [], "selections": [],
                        "stage": 0, "episodes_in_stage": 0} for arm in ("ppo", "cf")}
        completed_rounds = 0
        if args.resume:
            resume_identity = json.loads((args.resume/"identity.json").read_text())
            if resume_identity["config_sha256"] != config_hash or hash_path(args.resume/"state.pt") != resume_identity["sha256"]:
                raise ValueError("paired resume hash/config mismatch")
            resumed = torch.load(args.resume/"state.pt", map_location="cpu", weights_only=False)
            if resumed["parent_binding"] != binding:
                raise ValueError("paired resume parent differs")
            states, completed_rounds = resumed["arms"], resumed["completed_rounds"]
        comparisons = {}
        for arm in states:
            folder = out/arm
            folder.mkdir()
            write(folder/"execution.json", {"config": cfg, "parent_binding": binding, "code_commit": sha})
            write(folder/"scenarios.json", scenario_document)
        with ProcessPoolExecutor(max_workers=cfg["cpu_workers"], mp_context=multiprocessing.get_context("spawn")) as pool:
            actor.set_trainable(False)
            ep, _ = collect(actor, None, dev_select, cfg, pool, out, "select-phase2", greedy=True)
            baseline_selection = summarize(ep)
            ep, _ = collect(actor, None, dev_report, cfg, pool, out, "phase2", greedy=True)
            comparisons["phase2"] = summarize(ep)
            original = V1Actor(student_cfg, args.model_root/"final")
            ep, _ = collect(original, None, dev_select, cfg, pool, out, "select-v1", greedy=True)
            original_selection = summarize(ep)
            ep, _ = collect(original, None, dev_report, cfg, pool, out, "v1", greedy=True)
            comparisons["v1"] = summarize(ep)
            del original
            gc.collect()
            torch.cuda.empty_cache()
            paired_times = []
            newly_collected = {a: 0 for a in states}
            # All saved rounds contain BOTH arms at exactly the same update count.
            for update in range(completed_rounds, cfg["updates"]):
                estimate = max(paired_times[-3:] or [420])
                if time.monotonic()-started+estimate+cfg["evaluation_reserve_seconds"] >= budget:
                    break
                if any(prior_episodes[a]+newly_collected[a]+cfg["episodes_per_update"] > cfg["updates"]*cfg["episodes_per_update"]
                       for a in states):
                    break
                round_start = time.monotonic()
                for arm in ("ppo", "cf"):
                    state, folder = states[arm], out/arm
                    activate(state, actor, critic, optimizer)
                    torch.manual_seed(cfg["seed"]+update)
                    stage = state["stage"]
                    chosen = [s for s, _ in pools[stage][update*cfg["episodes_per_update"]:(update+1)*cfg["episodes_per_update"]]]
                    offset = (folder/"transitions.jsonl").stat().st_size if (folder/"transitions.jsonl").exists() else 0
                    t0 = time.monotonic()
                    ep, groups = collect(actor, critic, chosen, cfg, pool, folder, f"train-{update}", train=True)
                    newly_collected[arm] += len(ep)
                    collect_seconds = time.monotonic()-t0
                    objective, study_index, cf_seconds = None, [], 0.0
                    if arm == "cf":
                        cf_started = time.monotonic()
                        with (folder/"transitions.jsonl").open() as f:
                            f.seek(offset)
                            visits = [json.loads(line) for line in f]
                        jobs = jobs_from_visits(visits, by_id, cfg, update, actor.ad.weights_digest())
                        studies = list(pool.map(timed_cf_job, jobs))
                        store = out/"counterfactual"
                        store.mkdir(exist_ok=True)
                        for i, study in enumerate(studies):
                            path = store/f"update-{update+1:04d}-state-{i:02d}.json"
                            write(path, study)
                            entry = {"update": update+1, "path": str(path), "study_hash": study["study_hash"],
                                     "eligible_pairs": sum(r["training_eligible"] for r in study["estimates"].values())}
                            append(out/"counterfactual_index.jsonl", entry)
                            study_index.append(entry)
                        objective = CounterfactualObjective(studies)
                        cf_seconds = time.monotonic()-cf_started
                    t1 = time.monotonic()
                    optimization = optimize(actor, critic, optimizer, groups, cfg, auxiliary=objective)
                    for g in groups:
                        for key, adv, ret in zip(g["transition_keys"], g["advantages"], g["returns"]):
                            append(folder/"ppo_targets.jsonl", {"label": f"train-{update}", "scenario_id": key[0],
                                   "step": key[1], "advantage": adv, "return": ret})
                    del groups
                    old_history = state["episodes"]
                    history = old_history+ep
                    state["episodes_in_stage"] += len(ep)
                    rep = {"update": update+1, "stage": stage, "evidence_root": str(out),
                           "episode_ids": [e["scenario_id"] for e in ep], "rollout": summarize(ep), "collect_seconds": collect_seconds,
                           "optimize_seconds": time.monotonic()-t1, "counterfactual_seconds": cf_seconds,
                           "counterfactual_studies": study_index,
                           "counterfactual_pairs": len(objective.pairs) if objective else 0, **optimization}
                    append(folder/"updates.jsonl", rep)
                    trained = capture(actor, critic, optimizer)
                    trained.update(actor_digest=actor.ad.weights_digest(), episodes=history, updates=state["updates"]+[rep],
                                   selections=state["selections"], stage=stage,
                                   episodes_in_stage=state["episodes_in_stage"])
                    states[arm] = trained
                advance_shared_curriculum(states)
                completed_rounds = update+1
                if completed_rounds % cfg["evaluate_every"] == 0 or completed_rounds == cfg["updates"]:
                    for arm in ("ppo", "cf"):
                        state = states[arm]
                        activate(state, actor, critic, optimizer)
                        actor.set_trainable(False)
                        ep, _ = collect(actor, None, dev_select, cfg, pool, out/arm,
                                        f"select-{completed_rounds}", greedy=True)
                        path = out/arm/"checkpoints"/f"update-{completed_rounds:04d}"
                        state["selections"].append({"update": completed_rounds, "selection": summarize(ep), "checkpoint": str(path)})
                        save_arm(actor, state, path, config_hash, Path(binding["checkpoint"]), completed_rounds, sha)
                latest = save_resume(out, states, completed_rounds, config_hash, binding)
                paired_times.append(time.monotonic()-round_start)
                progress = {"completed_rounds": completed_rounds, "matched_update_counts": True,
                            "resume": str(latest), "elapsed_seconds": time.monotonic()-started,
                            "arms": {a: {"episodes": len(s["episodes"]), "environment_steps": sum(e["steps"] for e in s["episodes"]),
                                         "optimizer_steps": sum(u["optimizer_steps"] for u in s["updates"]),
                                         "auxiliary_optimizer_steps": sum(u["auxiliary_optimizer_steps"] for u in s["updates"]),
                                         "stage": s["stage"]} for a, s in states.items()}}
                write(out/"progress.json", progress)
                print(json.dumps(progress), flush=True)
            if not completed_rounds:
                raise ValueError("budget ended before one matched PPO round")
            best_by_arm = {}
            for arm in ("ppo", "cf"):
                state = states[arm]
                activate(state, actor, critic, optimizer)
                actor.set_trainable(False)
                if not state["selections"] or state["selections"][-1]["update"] != completed_rounds:
                    ep, _ = collect(actor, None, dev_select, cfg, pool, out/arm, f"select-{completed_rounds}", greedy=True)
                    path = out/arm/"checkpoints"/f"update-{completed_rounds:04d}"
                    state["selections"].append({"update": completed_rounds, "selection": summarize(ep), "checkpoint": str(path)})
                    save_arm(actor, state, path, config_hash, Path(binding["checkpoint"]), completed_rounds, sha)
                eligible_selections = trained_checkpoint_candidates(state["selections"], state["updates"],
                                                                     require_counterfactual=arm == "cf")
                if arm == "cf":
                    # An early pure-PPO iterate cannot stand in for the requested
                    # counterfactual-trained checkpoint, even if its DEV score wins.
                    if not eligible_selections:
                        raise ValueError("no DEV candidate contains an actual counterfactual update")
                best = select_checkpoint(eligible_selections, baseline_selection)
                best_path = Path(best["checkpoint"])
                ident = verify_snapshot(best_path, config_hash)
                selected_state = torch.load(best_path/"trainer.pt", map_location="cpu", weights_only=False)
                activate(selected_state, actor, critic, optimizer)
                if actor.ad.weights_digest() != ident["actor_digest"]:
                    raise ValueError("selected arm reload differs")
                actor.set_trainable(False)
                probe_observation = dev_report[0]
                probe_observation = EvacuationEnv(probe_observation).reset()[0]
                with torch.no_grad():
                    probe_before = actor.logits(actor.prepare([probe_observation]))[0].cpu()
                reloaded = V1Actor(student_cfg, best_path/"actor")
                if reloaded.ad.weights_digest() != ident["actor_digest"]:
                    raise ValueError("fresh actor load differs from saved selected weights")
                with torch.no_grad():
                    probe_after = reloaded.logits(reloaded.prepare([probe_observation]))[0].cpu()
                reload_error = float((probe_before-probe_after).abs().max())
                if reload_error > 1e-5:
                    raise ValueError("selected checkpoint changes inference logits after reload")
                del reloaded
                gc.collect()
                torch.cuda.empty_cache()
                ep, _ = collect(actor, None, dev_report, cfg, pool, out/arm, arm, greedy=True)
                comparisons[arm] = summarize(ep)
                delta = parameter_delta(actor, {n: p.float() for n, p in initial["trainable"].items()})
                if min(delta["group_l2"].values()) <= 0:
                    raise ValueError("continued arm did not update real V1 LoRA and head")
                best_by_arm[arm] = {**best, "actor": str(best_path/"actor"), "identity": ident,
                                    "parameter_delta_from_phase2": delta, "fresh_reload_max_logit_error": reload_error,
                                    "auxiliary_optimizer_steps_in_checkpoint": sum(
                                        u["auxiliary_optimizer_steps"] for u in selected_state["updates"])}
            aux_steps = sum(u["auxiliary_optimizer_steps"] for u in states["cf"]["updates"])
            if aux_steps <= 0:
                raise ValueError("no actual counterfactual gradient update occurred")
            # Reporting panel is fixed by scenario index; never choose teacher cases by outcomes.
            teacher_results = []
            for start in range(0, min(cfg["teacher_reporting_episodes"], len(dev_report)), 4):
                if time.monotonic()-started+660 >= budget:
                    break
                payloads = [(s, name, cfg["teacher_episode_wall_cap_seconds"])
                            for s in dev_report[start:start+4] for name in ("T0", "T1", "T2", "T3", "T4")]
                for result in pool.map(teacher_episode, payloads):
                    teacher_results.append(result)
                    append(out/"teacher_episodes.jsonl", result)
            teacher_summaries = {name: summarize([r for r in teacher_results if r["label"] == name and r["status"] == "COMPLETE"])
                                 for name in ("T0", "T1", "T2", "T3", "T4")
                                 if any(r["label"] == name and r["status"] == "COMPLETE" for r in teacher_results)}
        candidates = [{"update": 0, "name": "v1", "selection": original_selection, "actor": str(args.model_root/"final")},
                      {"update": 0, "name": "phase2", "selection": baseline_selection, "actor": str(Path(binding["checkpoint"])/"actor")}]
        candidates += [{**b, "name": arm} for arm, b in best_by_arm.items()]
        frozen = select_checkpoint(candidates, original_selection)
        write(out/"best_checkpoint.json", {"frozen_candidate": frozen, "trained_arms": best_by_arm,
                                          "final_evaluation_run": False})
        write(out/"training_history.json", {a: {"episodes": s["episodes"], "updates": s["updates"],
                                               "selections": s["selections"]} for a, s in states.items()})
        verify_inputs(args.model_root, cfg)
        verify_parent(args.phase2)
        write(out/"summary.json", {"status": "COMPLETE", "run_id": cfg["run_id"], "code_commit": sha,
              "parent_binding": binding, "completed_matched_rounds": completed_rounds,
              "training": {a: summarize(s["episodes"]) for a, s in states.items()},
              "optimizer_steps": {a: sum(u["optimizer_steps"] for u in s["updates"]) for a, s in states.items()},
              "auxiliary_optimizer_steps": aux_steps, "parent_optimizer_state_steps": parent_steps,
              "comparisons": comparisons, "teacher_comparisons": teacher_summaries,
              "teacher_failures": sum(r["status"] != "COMPLETE" for r in teacher_results),
              "best": frozen, "elapsed_seconds": time.monotonic()-started, "prior_wall_seconds": spent,
              "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(), "final_evaluation_run": False})
    except BaseException:
        write(out/"failure.json", {"status": "FAILED", "elapsed_seconds": time.monotonic()-started,
                                   "traceback": traceback.format_exc()})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        stop.set()
        monitor.join(timeout=10)


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ("config", "model-root", "phase2", "output-root"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--attempt", type=int, default=1)
    run(p.parse_args())


if __name__ == "__main__":
    main()
