"""GPU checks for the human-view LLM path. Run on the workstation BEFORE any human-view training.

The CPU suite cannot load the Clef/LoRA student, so these facts are unverified until this passes:
  1. a human-view row builds one Clef item with no truncation and scores five finite options;
  2. the option order of the scores is the action order the trainer assumes;
  3. one cross-entropy backward gives finite, nonzero LoRA and head gradients;
  4. (with --checkpoint) `HumanLLMActor.logits` agrees with `ClefLoraAdapter.scores`, the
     autograd replay reproduces the sampling forward, and PPO gradients are finite.

This is a check, not a training run: it takes at most one optimizer-free backward pass, saves no
checkpoint and reads only fresh TRAIN-split geometry.

  python experiments/human_gpu_checks.py --out checks.json [--checkpoint HUMAN_SFT_CKPT]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from ember_v1.contract import ACTIONS
from ember_v1.human_dagger import collect
from ember_v1.human_encoding import QUESTION_ID, view_row
from ember_v1.rl_online_scenarios import build_scenarios
from ember_v1.student_train import ACTION_ORDER, ClefLoraAdapter, config_from_protocol, load_protocol


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--rows", type=int, default=32)
    a = ap.parse_args()
    if a.out.exists():
        ap.error(f"{a.out} exists")
    import torch

    cfg = config_from_protocol(load_protocol())
    scenarios = [s for s, _ in build_scenarios("human-gpu-check", "train", 2, 12)]
    rows = collect(scenarios)[0][:a.rows]
    checks: dict[str, object] = {"rows": len(rows), "torch": torch.__version__, "checkpoint": str(a.checkpoint or "")}

    adapter = ClefLoraAdapter(cfg)                      # frozen base + fresh LoRA + fresh head
    tokens, worst = [], 0.0
    for r in rows:
        row = view_row(r["view"], r["label"])
        item = adapter.item(row)                        # raises on truncation or a skipped row
        tokens.append(int(len(item["input_ids"])) if "input_ids" in item else -1)
        scores = adapter.scores(row)[QUESTION_ID]
        if list(scores) != list(ACTION_ORDER) or set(scores) != set(ACTIONS):
            raise SystemExit(f"FAIL option order {list(scores)} != {list(ACTION_ORDER)}")
        if not all(math.isfinite(v) for v in scores.values()):
            raise SystemExit("FAIL non-finite scores")
        worst = max(worst, abs(math.log(sum(math.exp(v) for v in scores.values()))))
    checks.update(max_tokens=max(tokens), option_order=list(ACTION_ORDER), max_log_normalizer_error=worst)
    if worst > 1e-3:
        raise SystemExit(f"FAIL scores are not log-probabilities over the five actions ({worst})")

    loss = adapter.loss_and_backward(view_row(rows[0]["view"], rows[0]["label"]), 1.0)
    norms = adapter.grad_norms()
    adapter.opt.zero_grad(set_to_none=True)             # no optimizer step: the weights stay unchanged
    checks.update(first_loss=loss, grad_norms=norms)
    if not math.isfinite(loss) or not all(math.isfinite(v) and v > 0 for v in norms.values()):
        raise SystemExit(f"FAIL gradients {norms}")

    if a.checkpoint is not None:
        del adapter
        torch.cuda.empty_cache()
        from ember_v1.human_ppo import HumanLLMActor, mask_tensor, masked_log_probs, ppo_loss
        actor = HumanLLMActor(cfg, a.checkpoint)
        views = [r["view"] for r in rows[:8]]
        mask = mask_tensor(views, actor.device)
        with torch.enable_grad():
            first = masked_log_probs(actor.logits(views), mask)
            second = masked_log_probs(actor.logits(views), mask)
        replay = float((first - second).detach().masked_fill(~mask, 0).abs().max())
        with torch.no_grad():
            nograd = masked_log_probs(actor.logits(views), mask)
        drift = float((first.detach() - nograd).masked_fill(~mask, 0).abs().max())
        reference = actor.inner.ad.scores(view_row(views[0], ACTION_ORDER[0]))[QUESTION_ID]
        unmasked = actor.logits(views[:1]).float().log_softmax(-1)[0]
        agreement = max(abs(float(unmasked[i]) - reference[k]) for i, k in enumerate(ACTION_ORDER))
        chosen = first.argmax(-1)
        loss, _ = ppo_loss(first.gather(-1, chosen[:, None]).squeeze(-1),
                           second.detach().gather(-1, chosen[:, None]).squeeze(-1),
                           torch.ones(len(views), device=actor.device), 0.2)
        loss.backward()
        grads = {g["name"]: math.sqrt(sum(float(p.grad.float().square().sum()) for p in g["params"]
                                          if p.grad is not None))
                 for g in actor.parameter_groups({"lora_lr": 0.0, "head_lr": 0.0})}
        checks.update(autograd_replay_error=replay, nograd_vs_autograd=drift, actor_vs_adapter=agreement,
                      ppo_grad_norms=grads,
                      advice="set collect_with_grad=true if nograd_vs_autograd exceeds replay_tolerance")
        if replay > 1e-5 or agreement > 5e-3 or not all(math.isfinite(v) and v > 0 for v in grads.values()):
            raise SystemExit(f"FAIL actor checks {checks}")

    checks["status"] = "PASS"
    a.out.write_text(json.dumps(checks, sort_keys=True, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(checks, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
