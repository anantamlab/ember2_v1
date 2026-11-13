"""Counterfactual jobs and the explicitly offline auxiliary objective."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .rl_counterfactual import snapshot
from .sim import Simulator
from .sim.canonical import digest
from .student_train import ACTION_ORDER


def jobs_from_visits(rows, by_id, cfg, update, policy_digest):
    """Select deterministic visited-state indices, independent of their outcomes."""
    groups = {}
    for row in rows:
        groups.setdefault(row["scenario_id"], []).append(row)
    jobs = []
    from .teacher.t1 import T1CongestionAware
    from .provenance import code_fingerprint
    continuation_identity = {"identity": T1CongestionAware().identity.to_dict(),
                             "code_fingerprint": code_fingerprint(T1CongestionAware)}
    for i, (sid, history) in enumerate(list(groups.items())[:cfg["cf_states_per_update"]]):
        history.sort(key=lambda r: r["observation"]["step"])
        step = min((update+i*5) % 24, len(history)-1)
        visit = history[step]
        sim = Simulator(by_id[sid])
        prefix = []
        for row in history[:step]:
            if sim.step(row["executed_actions"]) != row["record"]:
                raise ValueError("on-policy source does not replay")
            prefix.append(row["record"])
        if sim.observe() != visit["observation"] or visit["sampled_actions"] != visit["executed_actions"]:
            raise ValueError("counterfactual visit differs from actual unmodified PPO transition")
        ids = sim.active_ids()
        greedy = {pid: ACTION_ORDER[max(range(5), key=lambda j: (lp[j], -j))]
                  for pid, lp in zip(ids, visit["person_logprobs"])}
        seed = int(digest([cfg["seed"], update, sid, step])[-16:], 16)
        forks = [None]
        if by_id[sid].fire_model == "stochastic":
            forks += [int(digest([seed, k])[-16:], 16) for k in range(1, cfg["cf_fire_replicates"])]
        jobs.append({"snapshot": snapshot(sim), "prefix_records": prefix, "student": visit["sampled_actions"],
                     "greedy": greedy, "seed": seed, "fork_seeds": forks,
                     "continuation_identity": continuation_identity, "teacher_limits": cfg["teacher_candidate_limits"],
                     "factual_visit": visit,
                     "source": {"split": "train", "scenario_id": sid, "step": visit["observation"]["step"],
                                "on_policy_label": visit["label"], "policy_digest": policy_digest,
                                "kind": "fresh_on_policy_visited_state", "update": update+1}})
    return jobs


class CounterfactualObjective:
    def __init__(self, studies):
        self.pairs, self.cursor, self.gradient_checked = [], 0, False
        for study in studies:
            eligible = [(name, data) for name, data in study["estimates"].items() if data["training_eligible"]]
            if not eligible:
                continue
            chosen, measured = max(eligible, key=lambda item: (item[1]["paired"]["utility"]["mean"], item[0]))
            public = study["public_observation"]
            ids = sorted(study["candidates"]["student"])
            positive = [ACTION_ORDER.index(study["candidates"][chosen][p]) for p in ids]
            negative = [ACTION_ORDER.index(study["candidates"]["student"][p]) for p in ids]
            old_lp = study["factual_visit"]["person_logprobs"]
            reference = sum(lp[a]-lp[b] for lp, a, b in zip(old_lp, positive, negative))
            weight = min(1., max(.05, measured["paired"]["utility"]["mean"]))
            self.pairs.append({"public": public, "positive": positive, "negative": negative,
                               "reference_margin": reference, "weight": weight,
                               "study_hash": study["study_hash"], "chosen": chosen})

    def __call__(self, actor):
        if not self.pairs:
            return None, {"eligible_pairs": 0}
        import torch
        from .rl_ppo import counterfactual_pair_loss
        pair = self.pairs[self.cursor % len(self.pairs)]
        self.cursor += 1
        z = actor.logits(actor.prepare([pair["public"]]))[0]
        loss = counterfactual_pair_loss(z, torch.tensor(pair["positive"], device="cuda"),
            torch.tensor(pair["negative"], device="cuda"), pair["reference_margin"], pair["weight"])
        evidence = {k: pair[k] for k in ("study_hash", "chosen", "weight", "reference_margin")}
        evidence["kind"] = "offline_measured_joint_counterfactual"
        if not self.gradient_checked:
            params = [p for n, p in actor.model.named_parameters() if n.startswith("head.") and p.requires_grad]
            grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            norm2 = sum(float(g.detach().float().square().sum()) for g in grads if g is not None)
            if norm2 <= 0 or any(not torch.isfinite(g).all() for g in grads if g is not None):
                raise FloatingPointError("counterfactual objective has no finite nonzero head gradient")
            evidence["independent_auxiliary_head_gradient_norm"] = norm2**.5
            self.gradient_checked = True
        return loss, evidence


def cpu_copy(value):
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def capture(actor, critic, optimizer):
    import torch
    return {"trainable": {n: p.detach().cpu().clone() for n, p in actor.model.named_parameters()
                           if "lora_" in n or n.startswith("head.")},
            "critic": cpu_copy(critic.state_dict()), "optimizer": cpu_copy(optimizer.state_dict()),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}


def activate(state, actor, critic, optimizer):
    import torch
    with torch.no_grad():
        for name, param in actor.model.named_parameters():
            if name in state["trainable"]:
                param.copy_(state["trainable"][name])
    critic.load_state_dict(state["critic"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
    actor.set_trainable(True)


def verify_parent(root):
    from .rl_online import hash_path
    required = {"summary.json": "COMPLETE", "independent_replay.json": "PASS", "optimizer_resume_audit.json": "PASS"}
    for name, expected in required.items():
        if not (root/name).exists() or json.loads((root/name).read_text()).get("status") != expected:
            raise ValueError(f"Phase 2 is not complete and verified: {name}")
    manifest = json.loads((root/"EVIDENCE_MANIFEST.json").read_text())
    pinned = {f["path"]: f for f in manifest["files"]}
    best = json.loads((root/"best_checkpoint.json").read_text())
    if not best["exact_reload_digest_equal"]:
        raise ValueError("Phase 2 selected actor did not exactly reload")
    selected = Path(best["actor"]).parent
    for p in [root/name for name in required] + [root/"best_checkpoint.json", selected/"identity.json", selected/"trainer.pt"] + list((selected/"actor").iterdir()):
        if p.is_file() and (str(p.relative_to(root)) not in pinned or hash_path(p) != pinned[str(p.relative_to(root))]["sha256"]):
            raise ValueError(f"Phase 2 input differs from its verified manifest: {p.name}")
    selection = json.loads((root/"selection.json").read_text())
    if best["update"] != selection["update"] or best["selection"] != selection["selection"]:
        raise ValueError("Phase 2 parent is not the DEV-selected candidate")
    return {"phase2_manifest_sha256": hash_path(root/"EVIDENCE_MANIFEST.json"),
            "checkpoint": str(selected), "actor_digest": best["identity"]["actor_digest"],
            "selected_update": best["update"], "selection": selection}
