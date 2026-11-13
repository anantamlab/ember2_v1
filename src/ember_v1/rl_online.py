"""Budgeted, resumable Phase 2 PPO continuation of the verified Phase 1 actor."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import multiprocessing
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .rl_env import EvacuationEnv
from .rl_online_scenarios import build_scenarios, select_checkpoint
from .rl_run import REPO, append, collect, parameter_delta, summarize, telemetry, verify_inputs, write


def hash_path(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_parent(parent, cfg):
    if hash_path(parent / "EVIDENCE_MANIFEST.json") != cfg["phase1_manifest_sha256"]:
        raise RuntimeError("Phase 1 evidence manifest differs")
    identity = json.loads((parent / "checkpoint_identity.json").read_text())
    if identity["rl_digest"] != cfg["phase1_digest"]:
        raise RuntimeError("wrong Phase 1 parent")
    for item in identity["files"]:
        if hash_path(parent / "checkpoint" / item["path"]) != item["sha256"]:
            raise RuntimeError("Phase 1 actor hash mismatch")
    manifest = json.loads((parent / "EVIDENCE_MANIFEST.json").read_text())
    # The manifest also binds the Adam/critic state, not only actor weights.
    entries = manifest["files"]
    entry = next(f for f in entries if f["path"] == "trainer.pt")
    if hash_path(parent / "trainer.pt") != entry["sha256"]:
        raise RuntimeError("Phase 1 trainer hash mismatch")
    return identity


def identity(path, actor, cfg, parent, update, sha):
    write(path / "identity.json", {"config_sha256": cfg, "parent": str(parent), "update": update,
          "code_commit": sha, "actor_digest": actor.ad.weights_digest(),
          "files": [{"path": str(p.relative_to(path)), "sha256": hash_path(p), "size": p.stat().st_size}
                    for p in sorted(path.rglob("*")) if p.is_file()]})


def verify_snapshot(path, config_hash):
    ident = json.loads((path / "identity.json").read_text())
    if ident["config_sha256"] != config_hash:
        raise RuntimeError("resume config differs")
    for f in ident["files"]:
        if hash_path(path / f["path"]) != f["sha256"]:
            raise RuntimeError("resume file hash mismatch")
    return ident


def run(args):
    cfg = json.loads(args.config.read_text())
    if cfg["status"] != "FROZEN" or sys.version_info[:2] != (3, 11):
        raise RuntimeError("frozen plan and Python 3.11 required")
    base = args.output_root / cfg["run_id"]
    base.mkdir(exist_ok=True, parents=True)
    if not 1 <= args.attempt <= cfg["max_attempts"]:
        raise RuntimeError("attempt outside budget")
    previous = [json.loads(p.read_text()) for p in base.glob("attempt-*/failure.json")]
    if any(base.glob("attempt-*/summary.json")):
        raise RuntimeError("experiment already completed")
    spent = sum(p["elapsed_seconds"] for p in previous)
    budget = cfg["wall_cap_seconds"]-spent
    previous_train_episodes = sum(1 for p in base.glob("attempt-*/episodes.jsonl") for line in p.open()
                                  if json.loads(line)["label"].startswith("train-"))
    if budget < cfg["evaluation_reserve_seconds"]:
        raise RuntimeError("experiment budget exhausted")
    out = base / f"attempt-{args.attempt}"
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    stop = threading.Event()
    monitor = threading.Thread(target=telemetry, args=(out, stop), daemon=True)
    monitor.start()
    try:
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip():
            raise RuntimeError("commit source before GPU execution")
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        config_hash = hash_path(args.config)
        write(out / "execution.json", {"config": cfg, "config_sha256": config_hash, "code_commit": sha,
              "started_unix": time.time(), "attempt": args.attempt, "prior_wall_seconds": spent,
              "resume": str(args.resume) if args.resume else None, "command": sys.argv})
        verify_inputs(args.model_root, cfg)
        parent_identity = verify_parent(args.phase1, cfg)
        splits = {s: build_scenarios(cfg["scenario_namespace"], s, n, cfg["max_steps"])
                  for s, n in (("train", cfg["updates"]*cfg["episodes_per_update"]),
                               ("select", cfg["dev_episodes"]), ("report", cfg["dev_episodes"]))}
        # An explicitly authorized retry uses fresh training families, never replays a failed rollout.
        if args.attempt > 1:
            splits["train"] = build_scenarios(cfg["scenario_namespace"]+f"-attempt-{args.attempt}",
                                              "train", cfg["updates"]*cfg["episodes_per_update"], cfg["max_steps"])
        from .gen.analysis import topology_hash
        def topology(s):
            return topology_hash(s.height, s.width, s.obstacles, [p for _, p in s.exits])
        hashes = [topology(s) for rows in splits.values() for s, _ in rows]
        old_scenarios = json.loads((args.phase1 / "scenarios.json").read_text())
        from .sim import Scenario
        old_hashes = {topology(Scenario.from_dict(s)) for split in ("train", "dev") for s in old_scenarios[split]}
        if len(set(hashes)) != len(hashes) or set(hashes) & old_hashes:
            raise RuntimeError("geometry family isolation failed")
        write(out / "scenarios.json", {**{split: [s.to_dict() for s, _ in rows] for split, rows in splits.items()},
              "metadata": {s.scenario_id: {**m, "topology_hash": topology(s)} for rows in splits.values() for s, m in rows},
              "train_dev_topology_overlap": 0, "phase1_overlap": 0})
        import unsloth
        import torch
        from .rl_ppo import V1Actor, make_value_head, optimize
        from .student_train import config_from_protocol, load_protocol
        original = json.loads((args.model_root / "identity.json").read_text())
        student_cfg = config_from_protocol(load_protocol(REPO / original["protocol"]["path"]))
        if student_cfg["protocol_hash"] != original["protocol"]["protocol_hash"]:
            raise RuntimeError("original V1 protocol differs")
        for key in ("chunk_k", "encoding", "question_style", "max_len", "precision", "reservations"):
            if student_cfg[key] != original["config"][key]:
                raise RuntimeError("original inference configuration differs")
        torch.set_num_threads(4)
        torch.manual_seed(cfg["seed"])
        critic = make_value_head()
        dev_select = [s for s, _ in splits["select"]]
        dev_report = [s for s, _ in splits["report"]]
        comparisons = {}
        with ProcessPoolExecutor(max_workers=cfg["cpu_workers"], mp_context=multiprocessing.get_context("spawn")) as pool:
            actor = V1Actor(student_cfg, args.model_root / "final")
            original_digest = actor.ad.weights_digest()
            # Selection baseline, and five-arm reporting baselines, are frozen before TRAIN.
            eps, _ = collect(actor, None, dev_select, cfg, pool, out, "select-v1", greedy=True)
            baseline_selection = summarize(eps)
            for name, processing in (("v1", None), ("v1-mask", "mask"), ("v1-coordinate", "coordinate")):
                eps, _ = collect(actor, None, dev_report, cfg, pool, out, name, greedy=True, postprocess=processing)
                comparisons[name] = summarize(eps)
            del actor
            gc.collect()
            torch.cuda.empty_cache()
            actor = V1Actor(student_cfg, args.phase1 / "checkpoint")
            if actor.ad.weights_digest() != parent_identity["rl_digest"]:
                raise RuntimeError("loaded Phase 1 actor differs")
            eps, _ = collect(actor, None, dev_report, cfg, pool, out, "phase1", greedy=True)
            comparisons["phase1"] = summarize(eps)
            state = torch.load(args.phase1 / "trainer.pt", weights_only=False, map_location="cuda")
            completed, updates, all_train, selections = 0, [], [], []
            if args.resume:
                resume_identity = verify_snapshot(args.resume, config_hash)
                del actor
                gc.collect()
                torch.cuda.empty_cache()
                actor = V1Actor(student_cfg, args.resume / "actor")
                if actor.ad.weights_digest() != resume_identity["actor_digest"]:
                    raise RuntimeError("resumed actor digest differs")
                state = torch.load(args.resume / "trainer.pt", weights_only=False, map_location="cuda")
                completed, updates = state["completed_updates"], state["updates"]
                all_train, selections = state["episodes"], state["selections"]
            critic.load_state_dict(state["critic"])
            actor.set_trainable(True)
            optimizer = torch.optim.AdamW(actor.parameter_groups(cfg)+[
                {"params": list(critic.parameters()), "lr": cfg["value_lr"], "name": "critic"}], weight_decay=0)
            optimizer.load_state_dict(state["optimizer"])
            for pg in optimizer.param_groups:
                pg["lr"] = cfg[{"lora": "lora_lr", "head": "head_lr", "critic": "value_lr"}[pg["name"]]]
            torch.set_rng_state(state["torch_rng"].cpu())
            torch.cuda.set_rng_state_all([r.cpu() for r in state["cuda_rng"]])
            del state
            initial = {n: p.detach().float().cpu().clone() for n, p in actor.model.named_parameters()
                       if "lora_" in n or n.startswith("head.")}
            training_start = time.monotonic()
            newly_collected = 0
            for update in range(completed, cfg["updates"]):
                elapsed = time.monotonic()-start
                estimate = max([u["seconds"] for u in updates[-4:]] or [180])
                if elapsed+estimate+cfg["evaluation_reserve_seconds"] >= budget:
                    break
                if previous_train_episodes+newly_collected+cfg["episodes_per_update"] > cfg["updates"]*cfg["episodes_per_update"]:
                    break
                t0 = time.monotonic()
                torch.manual_seed(cfg["seed"]+10000*args.attempt+update)
                chunk = splits["train"][update*cfg["episodes_per_update"]:(update+1)*cfg["episodes_per_update"]]
                episodes, groups = collect(actor, critic, [s for s, _ in chunk], cfg, pool, out,
                                           f"train-{update}", train=True)
                collected = time.monotonic()-t0
                optimization = optimize(actor, critic, optimizer, groups, cfg)
                # Persist the actual advantages supplied to PPO for independent recomputation.
                for group in groups:
                    for key, adv, ret in zip(group["transition_keys"], group["advantages"], group["returns"]):
                        append(out / "ppo_targets.jsonl", {"label": f"train-{update}", "scenario_id": key[0],
                               "step": key[1], "advantage": adv, "return": ret})
                del groups
                all_train.extend(episodes)
                newly_collected += len(episodes)
                rep = {"update": update+1, "rollout": summarize(episodes), "collect_seconds": collected,
                       "seconds": time.monotonic()-t0, **optimization}
                updates.append(rep)
                append(out / "updates.jsonl", rep)
                if (update+1) % cfg["evaluate_every"] == 0 or update+1 == cfg["updates"]:
                    actor.set_trainable(False)
                    ep, _ = collect(actor, None, dev_select, cfg, pool, out, f"select-{update+1}", greedy=True)
                    selections.append({"update": update+1, "selection": summarize(ep),
                                       "checkpoint": str(out / "checkpoints" / f"update-{update+1:04d}")})
                    actor.set_trainable(True)
                    write(out / "selection.json", select_checkpoint(selections, baseline_selection))
                snapshot = out / "checkpoints" / f"update-{update+1:04d}"
                snapshot.mkdir(parents=True)
                actor.save(snapshot / "actor")
                torch.save({"critic": critic.state_dict(), "optimizer": optimizer.state_dict(),
                            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                            "completed_updates": update+1, "updates": updates, "episodes": all_train,
                            "selections": selections}, snapshot / "trainer.pt")
                identity(snapshot, actor, config_hash, args.phase1, update+1, sha)
                write(out / "progress.json", {"completed_updates": update+1, "episodes": len(all_train),
                      "environment_steps": sum(e["steps"] for e in all_train), "latest": str(snapshot),
                      "optimizer_steps": sum(u["optimizer_steps"] for u in updates), "elapsed_seconds": time.monotonic()-start})
                print(json.dumps({"update": update+1, "rollout": rep["rollout"],
                                  "optimizer_steps": rep["optimizer_steps"], "elapsed": time.monotonic()-start}), flush=True)
            training_seconds = time.monotonic()-training_start
            if not selections:
                raise RuntimeError("budget ended before first DEV selection")
            delta = parameter_delta(actor, initial)
            if min(delta["group_l2"].values()) <= 0:
                raise RuntimeError("actor did not update both LoRA and Clef head")
            selected = select_checkpoint(selections, baseline_selection)
            best = Path(selected["checkpoint"])
            verify_snapshot(best, config_hash)
            del actor, optimizer, critic
            gc.collect()
            torch.cuda.empty_cache()
            actor = V1Actor(student_cfg, best / "actor")
            best_identity = json.loads((best / "identity.json").read_text())
            if actor.ad.weights_digest() != best_identity["actor_digest"]:
                raise RuntimeError("selected exact-save reload differs")
            ep, _ = collect(actor, None, dev_report, cfg, pool, out, "ppo", greedy=True)
            comparisons["ppo"] = summarize(ep)
            write(out / "best_checkpoint.json", {**selected, "actor": str(best / "actor"),
                  "trainer": str(best / "trainer.pt"), "identity": best_identity,
                  "exact_reload_digest_equal": True})
        verify_inputs(args.model_root, cfg)
        verify_parent(args.phase1, cfg)
        summary = {"status": "COMPLETE", "run_id": cfg["run_id"], "training": summarize(all_train),
                   "optimizer_steps": sum(u["optimizer_steps"] for u in updates),
                   "training_seconds": training_seconds, "elapsed_seconds": time.monotonic()-start,
                   "prior_wall_seconds": spent, "comparisons": comparisons, "selection": selected,
                   "parameter_delta": delta, "original_v1_digest": original_digest,
                   "phase1_digest": parent_identity["rl_digest"],
                   "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        write(out / "summary.json", summary)
        print(json.dumps(summary), flush=True)
    except BaseException:
        write(out / "failure.json", {"status": "FAILED", "elapsed_seconds": time.monotonic()-start,
              "traceback": traceback.format_exc(), "attempt": args.attempt})
        raise
    finally:
        stop.set()
        monitor.join(timeout=10)


def main():
    p = argparse.ArgumentParser(__doc__)
    for flag in ("config", "model-root", "phase1", "output-root"):
        p.add_argument("--"+flag, type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--attempt", type=int, default=1)
    run(p.parse_args())


if __name__ == "__main__":
    main()
