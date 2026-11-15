"""Per-person PPO for human-view policies. Torch is needed only here.

Differences from the joint PPO of R20-R22 (`rl_ppo.py`), each aimed at a measured failure:
  * one probability ratio per PERSON, not one product over the crowd (clip fraction rose to 26%);
  * each person is credited with their own result (`human_env.HumanRewardConfig`);
  * moves the person can see are wasted are masked in sampling AND in the loss;
  * an imitation term toward the belief teacher, decayed per update, keeps the start policy;
  * actors are decentralized (own view); the critic alone sees the full public state.

Actors share one interface: `logits(views) -> [n, 5]` in `ACTION_ORDER`, `parameter_groups(cfg)`,
`save(path)`, `device`. `SmallActor` is a small network (CPU or GPU). `HumanLLMActor` wraps the
Clef/LoRA student (GPU only).

  python -m ember_v1.human_ppo --config data/rl/human-pov-ppo.draft.json --out DIR [--actor small|llm
                               --checkpoint CKPT --device cpu]
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .human_encoding import feature_size, view_features, view_row
from .human_env import EpisodeStats, HumanEvacuationEnv, HumanRewardConfig, summarize
from .human_dagger import topology as _topology
from .human_teacher import belief_policy
from .human_view import ViewConfig, admissible
from .rl_env import gae, safe_distances, state_features
from .rl_online_scenarios import build_scenarios, select_checkpoint
from .student_train import ACTION_ORDER

PPO_VERSION = "1.0"
CRITIC_INPUT = 6 * 16 * 16 + 12 + 3
_INDEX = {a: i for i, a in enumerate(ACTION_ORDER)}


def mask_tensor(views: Sequence[Mapping[str, Any]], device, enabled: bool = True) -> torch.Tensor:
    rows = [[(not enabled) or a in admissible(v) for a in ACTION_ORDER] for v in views]
    return torch.tensor(rows, dtype=torch.bool, device=device)


def masked_log_probs(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if logits.shape != mask.shape or logits.shape[-1] != len(ACTION_ORDER) or not torch.isfinite(logits).all():
        raise FloatingPointError("bad or non-finite policy logits")
    if not mask.any(-1).all():
        raise ValueError("every person needs one allowed action")
    return logits.float().masked_fill(~mask, float("-inf")).log_softmax(-1)


def entropy(log_probs: torch.Tensor) -> torch.Tensor:
    p = log_probs.exp()
    return -(p * torch.where(p > 0, log_probs, torch.zeros_like(log_probs))).sum(-1)


def ppo_loss(new: torch.Tensor, old: torch.Tensor, advantages: torch.Tensor, clip: float):
    """Clipped surrogate with one ratio per person-decision."""
    log_ratio = new - old
    if not torch.isfinite(log_ratio).all() or log_ratio.abs().max() > 20:
        raise FloatingPointError("non-finite or exploding probability ratio")
    ratio = log_ratio.exp()
    loss = -torch.minimum(ratio * advantages, ratio.clamp(1 - clip, 1 + clip) * advantages).mean()
    return loss, ratio


class SmallActor(torch.nn.Module):
    """Small network on `human_encoding.view_features`. It answers one question cheaply: what can
    RL reach from a human view in this simulator? It can also act as a teacher for the LLM."""

    def __init__(self, radius: int, hidden: int = 128) -> None:
        super().__init__()
        self.radius, self.hidden = radius, hidden
        self.net = torch.nn.Sequential(torch.nn.Linear(feature_size(radius), hidden), torch.nn.Tanh(),
                                       torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
                                       torch.nn.Linear(hidden, len(ACTION_ORDER)))

    @property
    def device(self):
        return next(self.parameters()).device

    def logits(self, views: Sequence[Mapping[str, Any]]) -> torch.Tensor:
        if any(v["radius"] != self.radius for v in views):
            raise ValueError("view radius differs from the actor's")
        return self.net(torch.tensor([view_features(v) for v in views], dtype=torch.float32, device=self.device))

    def parameter_groups(self, cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
        return [{"params": list(self.parameters()), "lr": cfg["actor_lr"], "name": "actor"}]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        torch.save(self.state_dict(), path / "actor.pt")
        (path / "actor.json").write_text(json.dumps({"kind": "small", "radius": self.radius, "hidden": self.hidden}),
                                         encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, path: str | Path, device="cpu") -> "SmallActor":
        meta = json.loads((Path(path) / "actor.json").read_text(encoding="utf-8"))
        actor = cls(meta["radius"], meta["hidden"])
        actor.load_state_dict(torch.load(Path(path) / "actor.pt", map_location=device))
        return actor.to(device)


class HumanLLMActor:  # pragma: no cover - GPU only; check it with experiments/human_gpu_checks.py first
    """The Clef/LoRA student on human views: one short row per person (own view, one question)."""

    preserve_forward_batches = True  # bf16 padded forwards must replay the sampling context

    def __init__(self, cfg: Mapping[str, Any], checkpoint: str | Path) -> None:
        from .rl_ppo import V1Actor
        self.inner = V1Actor(cfg, checkpoint)
        self.inner.set_trainable(True)
        self.device = torch.device("cuda")

    def logits(self, views: Sequence[Mapping[str, Any]]) -> torch.Tensor:
        items = [self.inner.ad.item(view_row(v, ACTION_ORDER[0])) for v in views]
        return torch.cat(self.inner.logits({"items": items, "lengths": [1] * len(items)}))

    def parameter_groups(self, cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
        return self.inner.parameter_groups(cfg)

    def save(self, path: str | Path) -> None:
        self.inner.save(path)


def make_critic(device="cpu") -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(CRITIC_INPUT, 128), torch.nn.Tanh(),
                               torch.nn.Linear(128, 64), torch.nn.Tanh(), torch.nn.Linear(64, 1)).to(device)


def critic_features(observation: Mapping[str, Any], pids: Sequence[str]) -> list[list[float]]:
    """Centralized critic input per person: the full public state plus that person's position and
    true safe distance. Training only; an actor never receives it."""
    base, m = state_features(observation), observation["map"]
    dist, cap = safe_distances(observation), m["height"] * m["width"]
    pos = {p["id"]: p["position"] for p in observation["people"]}
    return [base + [pos[pid][0] / m["height"], pos[pid][1] / m["width"], dist.get(tuple(pos[pid]), cap) / cap]
            for pid in pids]


def collect(scenarios: Sequence[Any], actor, critic, cfg: Mapping[str, Any], view: ViewConfig,
            reward: HumanRewardConfig, generator: torch.Generator, *,
            env_factory=None, teacher_fn=belief_policy) -> dict[str, Any]:
    """Sample one episode per scenario. Returns flat per-person-decision lists with GAE advantages."""
    batch: dict[str, list] = {k: [] for k in ("views", "masks", "actions", "old", "values", "features",
                                               "teacher", "advantages", "returns")}
    episodes = []
    preserve = getattr(actor, "preserve_forward_batches", False)
    forward_groups: dict[int, list[int]] = {}
    next_group = 0
    for scenario in scenarios:
        env = (HumanEvacuationEnv(scenario, view, reward) if env_factory is None
               else env_factory(scenario, reward))
        views, info = env.reset()
        stats, done = EpisodeStats(scenario.scenario_id), False
        tracks: dict[str, dict[str, list]] = {}
        while not done:
            pids = sorted(views)
            vs = [views[p] for p in pids]
            gold = teacher_fn(views, env.observer.plans())
            feats = critic_features(env.observation, pids)
            mask = mask_tensor(vs, actor.device, cfg["mask"])
            # Optional grad-enabled forward: R20 found that a no-grad bf16 forward differs from the
            # autograd forward that PPO replays. Stored values are detached either way.
            with torch.set_grad_enabled(bool(cfg.get("collect_with_grad", False))):
                if preserve:
                    size = cfg["minibatch"]
                    lp = torch.cat([masked_log_probs(actor.logits(vs[j:j + size]), mask[j:j + size]).detach()
                                    for j in range(0, len(vs), size)])
                else:
                    lp = masked_log_probs(actor.logits(vs), mask).detach()
                value = critic(torch.tensor(feats, dtype=torch.float32, device=actor.device)).squeeze(-1).detach()
            choice = torch.multinomial(lp.exp().cpu(), 1, generator=generator).squeeze(-1)
            actions = {p: ACTION_ORDER[int(a)] for p, a in zip(pids, choice)}
            views, rewards, done, info = env.step(actions)
            stats.add(info["record"])
            for i, p in enumerate(pids):
                t = tracks.setdefault(p, {k: [] for k in ("views", "masks", "actions", "old", "values",
                                                           "features", "teacher", "rewards", "dones")})
                t["views"].append(vs[i])
                t["masks"].append(mask[i].cpu())
                t["actions"].append(int(choice[i]))
                t["old"].append(float(lp[i, int(choice[i])]))
                t["values"].append(float(value[i]))
                t["features"].append(feats[i])
                t["teacher"].append(_INDEX[gold[p]])
                t["rewards"].append(rewards[p])
                t["dones"].append(p not in views)       # resolved, or the episode ended
                if preserve:
                    t.setdefault("forward_group", []).append(next_group + i // size)
            if preserve:
                next_group += (len(vs) + size - 1) // size
        for p in sorted(tracks):
            t = tracks[p]
            adv, ret = gae(t["rewards"], t["values"], t["dones"], cfg["gamma"], cfg["gae_lambda"])
            if preserve:
                offset = len(batch["views"])
                for i, group in enumerate(t["forward_group"]):
                    forward_groups.setdefault(group, []).append(offset + i)
            for k in ("views", "masks", "actions", "old", "values", "features", "teacher"):
                batch[k] += t[k]
            batch["advantages"] += adv
            batch["returns"] += ret
        episodes.append(stats.result(info))
    batch["episodes"] = summarize(episodes)
    if preserve:
        batch["forward_groups"] = [forward_groups[g] for g in sorted(forward_groups)]
    return batch


def optimize(actor, critic, optimizer, batch: Mapping[str, Any], cfg: Mapping[str, Any],
             imitation_coef: float, generator: torch.Generator) -> dict[str, Any]:
    """PPO epochs over person-decisions. Gradients accumulate over `accumulation` minibatches
    (memory bound for the LLM). Stops BEFORE a step when the mean KL leaves the trust region."""
    dev, n = actor.device, len(batch["views"])
    adv = torch.tensor(batch["advantages"], dtype=torch.float32, device=dev)
    adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-8)
    old = torch.tensor(batch["old"], dtype=torch.float32, device=dev)
    actions = torch.tensor(batch["actions"], device=dev)
    teacher = torch.tensor(batch["teacher"], device=dev)
    returns = torch.tensor(batch["returns"], dtype=torch.float32, device=dev)
    features = torch.tensor(batch["features"], dtype=torch.float32, device=dev)
    masks = torch.stack(batch["masks"]).to(dev)
    params = [p for g in optimizer.param_groups for p in g["params"]]
    size, accumulation = cfg["minibatch"], cfg["accumulation"]

    def forward(idx):
        lp = masked_log_probs(actor.logits([batch["views"][i] for i in idx.tolist()]), masks[idx])
        return lp, lp.gather(-1, actions[idx, None]).squeeze(-1)

    preserved = batch.get("forward_groups")
    if getattr(actor, "preserve_forward_batches", False) and preserved is None:
        raise ValueError("batch-sensitive actors require their sampling forward groups")
    if preserved is not None:
        if (not preserved or any(not g or len(g) > size for g in preserved)
                or sorted(i for g in preserved for i in g) != list(range(n))):
            raise ValueError("sampling forward groups must partition all person-decisions within minibatch size")
        groups = [torch.tensor(g, dtype=torch.long) for g in preserved]
    else:
        groups = [torch.arange(i, min(i + size, n)) for i in range(0, n, size)]
    replay_error = 0.0
    with torch.enable_grad():   # audit EVERY sampling group before any optimizer step
        for group in groups:
            probe = group.to(dev)
            replay_error = max(replay_error, float((forward(probe)[1] - old[probe]).detach().abs().max()))
    if replay_error > cfg["replay_tolerance"]:
        raise RuntimeError(f"on-policy probability replay differs: {replay_error}")
    logs, steps, stop = [], 0, False
    for epoch in range(cfg["ppo_epochs"]):
        if preserved is None:
            order = torch.randperm(n, generator=generator)
            chunks = [[order[i:i + size]] for i in range(0, n, size)]
        else:
            # Shuffle whole sampling groups; pack them into minibatches without changing padding.
            # Separate forwards/backwards bound graph memory while keeping per-person loss weights.
            order = torch.randperm(len(groups), generator=generator).tolist()
            chunks, pack, packed = [], [], 0
            for i in order:
                group = groups[i]
                if pack and packed + len(group) > size:
                    chunks.append(pack)
                    pack, packed = [], 0
                pack.append(group)
                packed += len(group)
            if pack:
                chunks.append(pack)
        for start in range(0, len(chunks), accumulation):
            block = [idx for pack in chunks[start:start + accumulation] for idx in pack]
            count = sum(len(c) for c in block)
            optimizer.zero_grad(set_to_none=True)
            kls = []
            for idx in block:
                idx = idx.to(dev)
                lp, new = forward(idx)
                actor_loss, ratio = ppo_loss(new, old[idx], adv[idx], cfg["clip"])
                critic_loss = (critic(features[idx]).squeeze(-1) - returns[idx]).square().mean()
                ent = entropy(lp).mean()
                imitation = -lp.gather(-1, teacher[idx, None]).squeeze(-1).clamp_min(-50).mean()
                loss = (actor_loss + cfg["value_coef"] * critic_loss - cfg["entropy_coef"] * ent
                        + imitation_coef * imitation)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite PPO loss")
                (loss * (len(idx) / count)).backward()
                kls.append(float((old[idx] - new).mean().detach()) * len(idx) / count)
                logs.append({"epoch": epoch, "actor_loss": float(actor_loss.detach()),
                             "critic_loss": float(critic_loss.detach()), "entropy": float(ent.detach()),
                             "imitation": float(imitation.detach()),
                             "clip_fraction": float(((ratio - 1).abs() > cfg["clip"]).float().mean())})
            if sum(kls) > cfg["target_kl"]:
                optimizer.zero_grad(set_to_none=True)
                stop = True
                break
            norm = torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            logs[-1].update(kl=sum(kls), grad_norm=float(norm))
        if stop:
            break

    def mean(key):
        return sum(x[key] for x in logs) / len(logs)
    return {"decisions": n, "optimizer_steps": steps, "kl_early_stop": stop, "replay_error": replay_error,
            **{k: mean(k) for k in ("actor_loss", "critic_loss", "entropy", "imitation", "clip_fraction")}}


def actor_policy(actor, greedy: bool = True, masked: bool = True, generator: torch.Generator | None = None):
    """A `(views, routes) -> actions` policy from an actor (evaluation and DAgger rollouts)."""
    def act(views, routes=None):
        pids = sorted(views)
        vs = [views[p] for p in pids]
        with torch.no_grad():
            lp = masked_log_probs(actor.logits(vs), mask_tensor(vs, actor.device, masked)).cpu()
        choice = lp.argmax(-1) if greedy else torch.multinomial(lp.exp(), 1, generator=generator).squeeze(-1)
        return {p: ACTION_ORDER[int(a)] for p, a in zip(pids, choice)}
    return act


def check_config(cfg: Mapping[str, Any]) -> None:
    need = {"updates", "episodes_per_update", "max_steps", "gamma", "gae_lambda", "ppo_epochs", "clip", "target_kl",
            "entropy_coef", "value_coef", "max_grad_norm", "minibatch", "accumulation", "critic_lr", "mask",
            "imitation_coef", "imitation_decay", "replay_tolerance", "eval_every", "select_count", "seed",
            "namespace", "view", "reward"}
    missing = need - set(cfg)
    if missing:
        raise ValueError(f"PPO config lacks {sorted(missing)}")
    if cfg.get("status") != "authorized" and not cfg.get("smoke_test"):
        raise PermissionError("the PPO config is a draft: a training run needs a frozen, owner-authorized config")


def train(actor, critic, cfg: Mapping[str, Any], out: str | Path) -> dict[str, Any]:
    """Online loop: fresh TRAIN families per update, greedy checks on a fixed selection-DEV set,
    one checkpoint per check, selection with the project's casualty guard. Reporting DEV and the
    sealed final set are never read here."""
    check_config(cfg)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)     # write-once: no silent rerun of a started attempt
    view, reward = ViewConfig(**cfg["view"]), HumanRewardConfig(**cfg["reward"])
    generator = torch.Generator().manual_seed(cfg["seed"])
    optimizer = torch.optim.Adam(actor.parameter_groups(cfg) + [{"params": list(critic.parameters()),
                                                                 "lr": cfg["critic_lr"], "name": "critic"}])
    select = [s for s, _ in build_scenarios(cfg["namespace"], "select", cfg["select_count"], cfg["max_steps"])]
    held = {_topology(s) for s in select}

    def selection(update):
        from .human_env import run_episode
        s = summarize([run_episode(x, actor_policy(actor, masked=cfg["mask"]), view) for x in select])
        return {"update": update, "selection": {k: s[k] for k in ("people", "evacuated", "casualties", "unresolved")},
                "utility": s["utility"], "blocked_per_attempt": s["blocked_per_attempt"]}

    baseline = selection(0)
    rows, log = [], []
    for update in range(1, cfg["updates"] + 1):
        scenarios = [s for s, _ in build_scenarios(f"{cfg['namespace']}/u{update}", "train",
                                                   cfg["episodes_per_update"], cfg["max_steps"])]
        if held & {_topology(s) for s in scenarios}:
            raise RuntimeError("a TRAIN family shares its topology with selection DEV")
        batch = collect(scenarios, actor, critic, cfg, view, reward, generator)
        coef = cfg["imitation_coef"] * cfg["imitation_decay"] ** (update - 1)
        stats = optimize(actor, critic, optimizer, batch, cfg, coef, generator)
        entry = {"update": update, "imitation_coef": coef, "train": batch["episodes"], **stats}
        if update % cfg["eval_every"] == 0 or update == cfg["updates"]:
            entry["selection"] = selection(update)
            rows.append(entry["selection"])
            actor.save(out / f"update-{update:04d}")
        log.append(entry)
        with open(out / "log.jsonl", "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
    best = select_checkpoint(rows, baseline["selection"])
    summary = {"ppo_version": PPO_VERSION, "config": dict(cfg), "baseline": baseline, "checks": rows,
               "selected_update": best["update"], "selected_utility": best["selection_utility"],
               "safety_eligible": best["safety_eligible"],
               "improved_over_baseline": best["selection_utility"] > baseline["utility"]}
    with open(out / "summary.json", "x", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(summary, sort_keys=True, indent=1) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - a training entry point
    ap = argparse.ArgumentParser(prog="python -m ember_v1.human_ppo", description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--actor", choices=("small", "llm"), default="small")
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    check_config(cfg)
    torch.manual_seed(cfg["seed"])
    if a.actor == "llm":
        if a.checkpoint is None:
            ap.error("--actor llm needs --checkpoint (a human-view DAgger checkpoint)")
        from .student_train import config_from_protocol, load_protocol
        actor = HumanLLMActor({**config_from_protocol(load_protocol()), **cfg.get("llm", {})}, a.checkpoint)
    else:
        actor = (SmallActor.load(a.checkpoint, a.device) if a.checkpoint
                 else SmallActor(cfg["view"]["radius"], cfg.get("hidden", 128)).to(a.device))
    summary = train(actor, make_critic(actor.device), cfg, a.out)
    print(json.dumps({k: summary[k] for k in ("selected_update", "selected_utility", "improved_over_baseline")}))
    return 0 if math.isfinite(summary["selected_utility"]) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
