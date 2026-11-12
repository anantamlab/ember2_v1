"""Differentiable V1 Clef actor and joint-action PPO. Torch is needed only for RL."""
from __future__ import annotations

import math
from pathlib import Path

import torch

from .dataset import decision_questions
from .encoding import encode_observation
from .rl_env import state_features
from .student_ingest import id_chunks
from .student_train import ACTION_ORDER, ClefLoraAdapter


def joint_statistics(logits, actions):
    """Five categorical options per person; product policy, NOT token-level PPO.

    Float64 accumulation avoids loss of precision in large multi-person joint ratios.
    Autograd remains connected through the sum. Entropy is the exact joint entropy.
    """
    lp = logits.float().log_softmax(-1)
    selected = lp.gather(-1, actions[:, None]).squeeze(-1)
    return selected.double().sum(), -(lp.exp() * lp).double().sum(), lp


def ppo_actor_loss(new_joint, old_joint, advantages, clip):
    log_ratio = new_joint - old_joint
    if not torch.isfinite(log_ratio).all() or log_ratio.abs().max() > 50:
        raise FloatingPointError("nonfinite or explosively large joint probability ratio")
    ratio = log_ratio.exp()
    loss = -torch.minimum(ratio * advantages, ratio.clamp(1-clip, 1+clip) * advantages).mean()
    return loss, ratio


class V1Actor:
    def __init__(self, cfg, checkpoint):
        self.ad = ClefLoraAdapter(cfg, checkpoint=checkpoint)
        self.model, self.cfg = self.ad.model, cfg
        self.model.eval()  # same deterministic dropout-free forward for sampling and PPO replay
        self.set_trainable(False)

    def set_trainable(self, enabled):
        for name, p in self.model.named_parameters():
            p.requires_grad_(enabled and ("lora_" in name or name.startswith("head.")))
        self.model.eval()

    def parameter_groups(self, cfg):
        lora, head = [], []
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                (head if name.startswith("head.") else lora).append(p)
        if not lora or not head:
            raise RuntimeError("PPO requires the real V1 LoRA and decision head")
        return [{"params": lora, "lr": cfg["lora_lr"], "name": "lora"},
                {"params": head, "lr": cfg["head_lr"], "name": "head"}]

    def prepare(self, observations):
        items, lengths, ids = [], [], []
        for obs in observations:
            qs = decision_questions(obs, style=self.cfg["question_style"])
            people = sorted(qs)
            state = encode_observation(obs, mode=self.cfg["encoding"])
            for chunk in id_chunks(obs, self.cfg["chunk_k"]):
                row = {"state": state, "questions": {pid: qs[pid] for pid in chunk},
                       "gold": {pid: ACTION_ORDER[0] for pid in chunk}}
                item = self.ad.item(row)  # validates complete encoding, no truncation or skips
                actual = list(item["source"]["questions"])
                if actual != list(chunk):
                    raise RuntimeError("upstream changed question ordering")
                items.append(item)
            lengths.append(len(people))
            ids.append(people)
        return {"items": items, "lengths": lengths, "ids": ids,
                "features": torch.tensor([state_features(o) for o in observations], device="cuda")}

    def logits(self, batch):
        from unsloth.models.decision import _no_cudnn_attention
        b = self.ad.collate(batch["items"])
        mask = b.pop("marker_mask").cuda()
        b.pop("target")
        b.pop("ordinal", None)
        b = {k: v.cuda() if torch.is_tensor(v) else v for k, v in b.items()}
        with _no_cudnn_attention(), torch.autocast("cuda", dtype=self.ad.amp, enabled=self.ad.amp is not None):
            z, _ = self.model(**b)
        if z.shape != mask.shape or z.shape[-1] != len(ACTION_ORDER) or not mask.all():
            raise RuntimeError("unexpected action option layout")
        if not torch.isfinite(z).all():
            raise FloatingPointError("nonfinite policy logits")
        return z.float().split(batch["lengths"])

    def sample(self, batch, greedy=False):
        logits = self.logits(batch)
        maps, indices, logps, entropies, distributions = [], [], [], [], []
        for ids, z in zip(batch["ids"], logits):
            a = z.argmax(-1) if greedy else torch.distributions.Categorical(logits=z).sample()
            logp, entropy, lp = joint_statistics(z, a)
            maps.append({pid: ACTION_ORDER[int(x)] for pid, x in zip(ids, a)})
            indices.append(a.detach())
            logps.append(logp)
            entropies.append(entropy)
            distributions.append(lp.detach())
        return maps, indices, torch.stack(logps), torch.stack(entropies), distributions

    def save(self, path):
        path = Path(path)
        path.mkdir(exist_ok=False)
        self.model.save_pretrained(str(path), self.ad.tok)


def make_value_head():
    # The head is centralized across the full joint state, not one critic per person.
    return torch.nn.Sequential(torch.nn.Linear(6*16*16+12, 128), torch.nn.Tanh(),
                               torch.nn.Linear(128, 64), torch.nn.Tanh(),
                               torch.nn.Linear(64, 1)).cuda()


def optimize(actor, critic, optimizer, groups, cfg):
    """Replay original inference batches, keeping padding/precision identical at ratio=1.

    A group contains several JOINT transitions. Accumulate groups to limit GPU memory.
    Shared advantages are normalized once across states, never across people/tokens.
    """
    advantages = torch.tensor([a for g in groups for a in g["advantages"]], device="cuda")
    mean, std = advantages.mean(), advantages.std(unbiased=False).clamp_min(1e-8)
    results, max_replay_error, steps = [], 0.0, 0
    # Fundamental on-policy audit before the first update, not confounded by changed weights.
    with torch.enable_grad():
        for g in groups:
            zs = actor.logits(g["batch"])
            lp = torch.stack([joint_statistics(z, a)[0] for z, a in zip(zs, g["actions"])])
            error = float((lp - g["old_joint"]).detach().abs().max())
            max_replay_error = max(error, max_replay_error)
            del zs, lp  # never retain multiple full backbone graphs during the audit
    if max_replay_error > 1e-5:
        raise RuntimeError(f"on-policy probability replay differs: {max_replay_error}")
    all_params = [p for group in optimizer.param_groups for p in group["params"]]
    stop = False
    for epoch in range(cfg["ppo_epochs"]):
        order = torch.randperm(len(groups)).tolist()
        for start in range(0, len(order), cfg["accumulation_groups"]):
            block = [groups[i] for i in order[start:start+cfg["accumulation_groups"]]]
            count = sum(len(g["advantages"]) for g in block)
            optimizer.zero_grad(set_to_none=True)
            block_kls = []
            for g in block:
                zs = actor.logits(g["batch"])
                stats = [joint_statistics(z, a) for z, a in zip(zs, g["actions"])]
                joint = torch.stack([s[0] for s in stats])
                entropy = torch.stack([s[1] for s in stats]).mean()
                adv = (torch.tensor(g["advantages"], device="cuda")-mean)/std
                actor_loss, ratio = ppo_actor_loss(joint, g["old_joint"], adv, cfg["clip"])
                value = critic(g["batch"]["features"]).squeeze(-1)
                target = torch.tensor(g["returns"], device="cuda")
                critic_loss = (value - target).square().mean()
                kl = torch.stack([(old.exp() * (old-new[2])).double().sum()
                                  for old, new in zip(g["old_lp"], stats)]).mean()
                if not torch.isfinite(kl) or not torch.isfinite(critic_loss):
                    raise FloatingPointError("nonfinite KL/value loss")
                k = float(kl.detach())
                block_kls.append(k)
                loss = actor_loss + cfg["value_coef"]*critic_loss - cfg["entropy_coef"]*entropy
                (loss * (len(g["advantages"])/count)).backward()
                results.append({"actor_loss": float(actor_loss.detach()), "critic_loss": float(critic_loss.detach()),
                                "joint_kl": k, "joint_entropy": float(entropy.detach()),
                                "person_entropy": float(sum(s[1].detach() for s in stats)/sum(g["batch"]["lengths"])),
                                "clip_fraction": float(((ratio-1).abs() > cfg["clip"]).float().mean()),
                                "epoch": epoch})
            norms = {}
            for pg in optimizer.param_groups:
                grads = [p.grad for p in pg["params"] if p.grad is not None]
                if not grads or any(not torch.isfinite(g).all() for g in grads):
                    raise FloatingPointError(f"missing/nonfinite {pg['name']} gradients")
                norms[pg["name"]] = math.sqrt(sum(float(g.float().square().sum()) for g in grads))
                if norms[pg["name"]] <= 0:
                    raise FloatingPointError(f"zero {pg['name']} gradient")
            norm = torch.nn.utils.clip_grad_norm_(all_params, cfg["max_grad_norm"], error_if_nonfinite=True)
            if not torch.isfinite(norm):
                raise FloatingPointError("nonfinite gradient norm")
            # Stop BEFORE applying a further update when the current joint policy has left the trust region.
            if sum(block_kls)/len(block_kls) > cfg["target_joint_kl"]:
                optimizer.zero_grad(set_to_none=True)
                stop = True
                break
            optimizer.step()
            steps += 1
            results[-1].update(gradient_norms=norms, total_grad_norm=float(norm))
        if stop:
            break
    return {"optimizer_steps": steps, "max_probability_replay_error": max_replay_error,
            "kl_early_stop": stop, "minibatches": results}
