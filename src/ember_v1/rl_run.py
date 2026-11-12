"""Bounded Phase-1 PPO experiment, fresh scenarios, persistent evidence and provenance.

Run with the pinned Run-19 Python 3.11 GPU environment; no SFT code is executed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import platform
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .rl_env import EvacuationEnv, RewardConfig, fresh_scenario, gae


REPO = Path(__file__).resolve().parents[2]


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def append(path, value):
    with Path(path).open("a") as f:
        f.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def _step(payload):
    env, actions = payload
    return env, env.step(actions)


def summarize(episodes):
    keys = ("people", "evacuated", "casualties", "unresolved", "steps", "person_steps", "stay", "attempted", "blocked")
    totals = {k: sum(e[k] for e in episodes) for k in keys}
    return {"episodes": len(episodes), **totals,
            **{k+"_rate": totals[k]/totals["people"] for k in ("evacuated", "casualties", "unresolved")},
            "stay_rate": totals["stay"]/max(1, totals["person_steps"]),
            "blocked_per_attempt": totals["blocked"]/max(1, totals["attempted"]),
            "mean_reward": sum(e["reward"] for e in episodes)/len(episodes),
            "mean_terminal_reward": sum(e["terminal_reward"] for e in episodes)/len(episodes)}


def collect(actor, critic, scenarios, cfg, pool, out, label, *, greedy=False, train=False):
    import torch
    reward = RewardConfig(cfg["gamma"], cfg["shaping_eta"], cfg["evacuated_weight"],
                          cfg["casualty_weight"], cfg["unresolved_weight"])
    episodes, groups = [], []
    for start in range(0, len(scenarios), cfg["parallel_envs"]):
        envs = [EvacuationEnv(s, reward) for s in scenarios[start:start+cfg["parallel_envs"]]]
        observations = [e.reset()[0] for e in envs]
        histories = [[] for _ in envs]
        counters = [Counter() for _ in envs]
        live = list(range(len(envs)))
        while live:
            batch = actor.prepare([observations[i] for i in live])
            # bf16 kernels can differ between no_grad and autograd, even in eval mode.
            # TRAIN behavior must use the same forward path as differentiable PPO replay.
            with torch.set_grad_enabled(train):
                maps, indices, logps, entropy, old_lp = actor.sample(batch, greedy)
            logps, entropy = logps.detach(), entropy.detach()
            with torch.no_grad():
                values = critic(batch["features"]).squeeze(-1) if critic is not None else torch.zeros(len(live), device="cuda")
            group = {"batch": batch, "actions": indices, "old_joint": logps.detach(), "old_lp": old_lp}
            group_rows = []
            # Workers receive only simulator objects/actions, never CUDA tensors or model objects.
            transitions = list(pool.map(_step, [(envs[i], a) for i, a in zip(live, maps)])) if pool else [
                _step((envs[i], a)) for i, a in zip(live, maps)]
            next_live = []
            for row, (i, actions, transition) in enumerate(zip(live, maps, transitions)):
                envs[i], (nxt, reward_value, done, truncated, info) = transition
                if truncated or info["record"]["actions"] != actions:
                    raise RuntimeError("sampled/executed action mismatch")
                counters[i].update(steps=1, person_steps=len(actions), stay=sum(a == "stay" for a in actions.values()),
                                   attempted=sum(a != "stay" for a in actions.values()),
                                   blocked=sum(m["outcome"].startswith("blocked_") for m in info["record"]["movements"].values()))
                for movement in info["record"]["movements"].values():
                    counters[i][movement["outcome"]] += 1
                item = {"reward": reward_value, "value": float(values[row]), "done": done,
                        "terminal_reward": info["terminal_reward"], "group_row": row, "group": group}
                histories[i].append(item)
                group_rows.append(item)
                append(out / "transitions.jsonl", {"label": label, "scenario_id": envs[i].scenario.scenario_id,
                       "observation": observations[i], "sampled_actions": actions, "executed_actions": info["record"]["actions"],
                       "old_joint_logprob": float(logps[row]), "person_logprobs": old_lp[row].cpu().tolist(),
                       "value": item["value"], "reward": reward_value, "terminal_reward": info["terminal_reward"],
                       "shaping_reward": info["shaping_reward"], "done": done, "record": info["record"]})
                observations[i] = nxt
                if not done:
                    next_live.append(i)
                else:
                    episode = {"label": label, "scenario_id": envs[i].scenario.scenario_id,
                               **{k: info[k] for k in ("people", "evacuated", "casualties", "unresolved")},
                               **dict(counters[i]), "reward": sum(x["reward"] for x in histories[i]),
                               "terminal_reward": info["terminal_reward"], "termination": info["record"]["termination_reason"]}
                    # Explicit zero counters, for episodes in which a behavior never occurs.
                    for k in ("stay", "attempted", "blocked"):
                        episode.setdefault(k, 0)
                    episodes.append(episode)
                    append(out / "episodes.jsonl", episode)
            if train:
                group["transitions"] = group_rows
                groups.append(group)
            live = next_live
        if train:
            for hist in histories:
                adv, returns = gae([x["reward"] for x in hist], [x["value"] for x in hist],
                                   [x["done"] for x in hist], cfg["gamma"], cfg["gae_lambda"])
                for item, a, ret in zip(hist, adv, returns):
                    item["advantage"], item["return"] = a, ret
        print(json.dumps({"label": label, "episodes": len(episodes), "steps": sum(e["steps"] for e in episodes)}), flush=True)
    for group in groups:
        group["advantages"] = [x["advantage"] for x in group["transitions"]]
        group["returns"] = [x["return"] for x in group["transitions"]]
        del group["transitions"]
    return episodes, groups


def verify_inputs(root, cfg):
    from .hfstore import hash_file
    manifest_path = root / "ARTIFACT.json"
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != cfg["model_manifest_sha256"]:
        raise RuntimeError("V1 ARTIFACT.json differs from pinned registry")
    manifest = json.loads(manifest_path.read_text())
    if manifest["artifact_id"] != cfg["baseline"]:
        raise RuntimeError("wrong V1 artifact")
    for f in manifest["files"]:
        got = hash_file(root / f["path"], scan=False)
        if got["sha256"] != f["sha256"] or got["size"] != f["size"]:
            raise RuntimeError(f"V1 checkpoint hash differs: {f['path']}")
    from .workstation import fetch_base_model
    base = fetch_base_model(root, print)
    return {"checkpoint_files_verified": len(manifest["files"]), "base": base,
            "model_manifest_sha256": cfg["model_manifest_sha256"], "model_revision": cfg["model_revision"]}


def telemetry(out, stop):
    while not stop.is_set():
        result = subprocess.run(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True)
        append(out / "gpu.jsonl", {"time": time.time(), "sample": result.stdout.strip(), "exit": result.returncode})
        stop.wait(2)


def parameter_delta(actor, initial):
    import torch
    diff2, old2, changed = 0.0, 0.0, 0
    per_group = {"lora": 0.0, "head": 0.0}
    for name, p in actor.model.named_parameters():
        if name in initial:
            current = p.detach().float().cpu()
            delta = float((current - initial[name]).square().sum())
            diff2 += delta
            old2 += float(initial[name].square().sum())
            changed += int(delta > 0)
            per_group["head" if name.startswith("head.") else "lora"] += delta
    return {"l2": diff2**0.5, "relative_l2": (diff2/old2)**0.5, "changed_tensors": changed,
            "group_l2": {k: v**0.5 for k, v in per_group.items()}}


def run(config, root, out, attempt, remaining_seconds=None):
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Python 3.11 required")
    cfg = json.loads(config.read_text())
    budget = cfg["wall_cap_seconds"] if remaining_seconds is None else remaining_seconds
    if cfg["status"] != "FROZEN" or not 1 <= attempt <= cfg["max_attempts"]:
        raise RuntimeError("unfrozen plan or attempt outside authorized bounds")
    out.mkdir(parents=True, exist_ok=False)
    started, stop = time.monotonic(), threading.Event()
    monitor = threading.Thread(target=telemetry, args=(out, stop), daemon=True)
    monitor.start()
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip():
            raise RuntimeError("commit code before running pilot")
        write(out / "execution.json", {"config": cfg, "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
              "code_commit": sha, "python": platform.python_version(), "attempt": attempt,
              "model_root": str(root), "command": sys.argv, "started_unix": time.time(),
              "remaining_budget_seconds": budget})
        provenance = verify_inputs(root, cfg)
        write(out / "input_verification.json", provenance)
        train = [fresh_scenario(cfg["scenario_namespace"], "train", i, cfg["max_steps"])
                 for i in range(cfg["updates"]*cfg["episodes_per_update"])]
        dev = [fresh_scenario(cfg["scenario_namespace"], "dev", i, cfg["max_steps"]) for i in range(cfg["dev_episodes"])]
        from .gen.analysis import topology_hash
        families = [topology_hash(s.height, s.width, s.obstacles, [p for _, p in s.exits]) for s in train+dev]
        if len(set(families)) != len(families):
            raise RuntimeError("duplicate/isolation failure in fresh families")
        write(out / "scenarios.json", {"train": [s.to_dict() for s in train], "dev": [s.to_dict() for s in dev],
                                      "topology_hashes": families, "train_dev_topology_overlap": 0})
        # Import Unsloth first so its audited patches precede transformers/PEFT imports.
        import unsloth
        import torch
        from .rl_ppo import V1Actor, make_value_head, optimize
        from .student_train import config_from_protocol, load_protocol
        ident = json.loads((root / "identity.json").read_text())
        student_cfg = config_from_protocol(load_protocol(REPO / ident["protocol"]["path"]))
        if student_cfg["protocol_hash"] != ident["protocol"]["protocol_hash"]:
            raise RuntimeError("V1 inference configuration differs")
        for key in ("chunk_k", "encoding", "question_style", "max_len", "precision", "reservations"):
            if student_cfg[key] != ident["config"][key]:
                raise RuntimeError(f"V1 config differs: {key}")
        torch.set_num_threads(4)
        actor = V1Actor(student_cfg, root / "final")
        torch.manual_seed(cfg["seed"])
        critic = make_value_head()
        initial = {n: p.detach().float().cpu().clone() for n, p in actor.model.named_parameters()
                   if "lora_" in n or n.startswith("head.")}
        frozen_digest = actor.ad.weights_digest()
        # Fixed probe defines measurable action distribution change and save/reload parity.
        probe = actor.prepare([EvacuationEnv(dev[0]).reset()[0]])
        with torch.no_grad():
            before = [z.cpu() for z in actor.logits(probe)]
        actor.save(out / "preflight_roundtrip")
        roundtrip = V1Actor(student_cfg, out / "preflight_roundtrip")
        with torch.no_grad():
            roundtrip_z = roundtrip.logits(roundtrip.prepare([EvacuationEnv(dev[0]).reset()[0]]))
        preflight_error = max(float((a.cpu()-b).abs().max()) for a, b in zip(roundtrip_z, before))
        if preflight_error > 1e-5 or roundtrip.ad.weights_digest() != frozen_digest:
            raise RuntimeError(f"V1 exact-save preflight mismatch: {preflight_error}")
        write(out / "save_preflight.json", {"status": "PASS", "max_logit_error": preflight_error,
              "parameter_digest_equal": True, "head_precision": "float32, exact=True"})
        del roundtrip, roundtrip_z
        torch.cuda.empty_cache()
        with ProcessPoolExecutor(max_workers=cfg["cpu_workers"], mp_context=multiprocessing.get_context("spawn")) as pool:
            torch.manual_seed(cfg["seed"]+1000)
            baseline, _ = collect(actor, None, dev, cfg, pool, out, "v1-greedy", greedy=True)
            torch.manual_seed(cfg["seed"]+2000)
            baseline_sample, _ = collect(actor, None, dev, cfg, pool, out, "v1-stochastic")
            actor.set_trainable(True)
            opt = torch.optim.AdamW(actor.parameter_groups(cfg) + [
                {"params": list(critic.parameters()), "lr": cfg["value_lr"], "name": "critic"}], weight_decay=0.0)
            updates, all_train = [], []
            training_start = time.monotonic()
            for update in range(cfg["updates"]):
                if time.monotonic()-started > budget:
                    raise TimeoutError("bounded pilot wall cap reached")
                t0 = time.monotonic()
                torch.manual_seed(cfg["seed"]+3000+update)
                scenarios = train[update*cfg["episodes_per_update"]:(update+1)*cfg["episodes_per_update"]]
                eps, groups = collect(actor, critic, scenarios, cfg, pool, out, f"train-{update}", train=True)
                collection_s = time.monotonic()-t0
                t1 = time.monotonic()
                optimization = optimize(actor, critic, opt, groups, cfg)
                all_train.extend(eps)
                rep = {"update": update, "rollout": summarize(eps), "collect_seconds": collection_s,
                       "optimize_seconds": time.monotonic()-t1, **optimization}
                updates.append(rep)
                append(out / "updates.jsonl", rep)
                write(out / "progress.json", {"updates": update+1, "episodes": len(all_train),
                      "environment_steps": sum(e["steps"] for e in all_train), "latest": rep})
                print(json.dumps({"update": update, "summary": rep["rollout"], "optimizer_steps": optimization["optimizer_steps"],
                                  "seconds": time.monotonic()-t0}), flush=True)
                del groups
                if (update+1) % 2 == 0:
                    snapshots = out / "intermediate"
                    snapshots.mkdir(exist_ok=True)
                    actor.save(snapshots / f"update-{update+1:04d}")
                    torch.save({"critic": critic.state_dict(), "optimizer": opt.state_dict(),
                                "completed_updates": update+1, "torch_rng": torch.get_rng_state(),
                                "cuda_rng": torch.cuda.get_rng_state_all()}, snapshots / f"trainer-{update+1:04d}.pt")
            training_s = time.monotonic()-training_start
            delta = parameter_delta(actor, initial)
            if delta["group_l2"]["lora"] <= 0 or delta["group_l2"]["head"] <= 0:
                raise RuntimeError("LoRA or head did not update")
            actor.set_trainable(False)
            with torch.no_grad():
                after = [z.cpu() for z in actor.logits(probe)]
            probe_change = max(float((a-b).abs().max()) for a, b in zip(after, before))
            if probe_change <= 0:
                raise RuntimeError("no measurable policy change")
            actor.save(out / "checkpoint")
            torch.save({"critic": critic.state_dict(), "optimizer": opt.state_dict(), "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all(), "updates": updates}, out / "trainer.pt")
            actor.set_trainable(False)
            torch.manual_seed(cfg["seed"]+1000)
            rl, _ = collect(actor, None, dev, cfg, pool, out, "rl-greedy", greedy=True)
            torch.manual_seed(cfg["seed"]+2000)
            rl_sample, _ = collect(actor, None, dev, cfg, pool, out, "rl-stochastic")
        final_digest = actor.ad.weights_digest()
        # A real upstream load of the saved RL adapter/head, not just checking that files exist.
        reloaded = V1Actor(student_cfg, out / "checkpoint")
        with torch.no_grad():
            rz = reloaded.logits(reloaded.prepare([EvacuationEnv(dev[0]).reset()[0]]))
        reload_error = max(float((a.cpu()-b).abs().max()) for a, b in zip(rz, after))
        if reload_error > 1e-5 or reloaded.ad.weights_digest() != final_digest:
            raise RuntimeError(f"saved RL checkpoint reload mismatch: {reload_error}")
        # Rehash immutable inputs after all work, proving that the original baseline was preserved.
        verify_inputs(root, cfg)
        from .hfstore import hash_file
        write(out / "checkpoint_identity.json", {"run_id": cfg["run_id"], "parent": cfg["baseline"],
              "parent_revision": cfg["model_revision"], "code_commit": sha, "frozen_v1_digest": frozen_digest,
              "rl_digest": final_digest, "parameter_delta": delta, "probe_max_logit_change": probe_change,
              "reload_max_logit_error": reload_error, "config": cfg, "student_config": ident["config"],
              "files": [{"path": str(p.relative_to(out / "checkpoint")), **hash_file(p, scan=False)}
                        for p in sorted((out / "checkpoint").rglob("*")) if p.is_file()]})
        summary = {"status": "COMPLETE", "run_id": cfg["run_id"], "attempt": attempt,
                   "code_commit": sha, "training": summarize(all_train), "training_seconds": training_s,
                   "environment_steps_per_second": sum(e["steps"] for e in all_train)/training_s,
                   "collection_steps_per_second": sum(e["steps"] for e in all_train)/sum(u["collect_seconds"] for u in updates),
                   "optimizer_steps": sum(u["optimizer_steps"] for u in updates),
                   "baseline_greedy": summarize(baseline), "rl_greedy": summarize(rl),
                   "baseline_stochastic": summarize(baseline_sample), "rl_stochastic": summarize(rl_sample),
                   "parameter_delta": delta, "probe_max_logit_change": probe_change, "reload_error": reload_error,
                   "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                   "elapsed_seconds": time.monotonic()-started, "checkpoint": str(out / "checkpoint"),
                   "frozen_v1_digest": frozen_digest, "rl_digest": final_digest}
        write(out / "summary.json", summary)
        print(json.dumps(summary), flush=True)
    except BaseException:
        write(out / "failure.json", {"status": "FAILED", "elapsed_seconds": time.monotonic()-started,
                                     "traceback": traceback.format_exc()})
        raise
    finally:
        stop.set()
        monitor.join(timeout=10)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    # Guards live outside checkouts: a new checkout cannot reset an already-started attempt.
    base = args.output_root / cfg["run_id"]
    base.mkdir(parents=True, exist_ok=True)
    spent = sum(json.loads(p.read_text()).get("elapsed_seconds", 0) for p in base.glob("attempt-*/failure.json"))
    if spent >= cfg["wall_cap_seconds"]:
        raise RuntimeError("total authorized retry budget exhausted")
    run(args.config, args.model_root, base / f"attempt-{args.attempt}", args.attempt,
        remaining_seconds=cfg["wall_cap_seconds"]-spent)


if __name__ == "__main__":
    main()
