"""Run explicitly with pytest in the GPU environment; no torch dependency in the CPU suite."""
import itertools


def test_joint_probability_normalizes_and_gradient_is_sum():
    import torch
    from ember_v1.rl_ppo import joint_statistics
    z = torch.tensor([[0.1, -0.2, 0.3, 0.4, -0.1], [-0.3, 0.4, 0.6, 0.2, -0.2]], requires_grad=True)
    total = sum(joint_statistics(z, torch.tensor(a))[0].exp() for a in itertools.product(range(5), repeat=2))
    assert torch.allclose(total, torch.tensor(1.0, dtype=torch.float64), atol=1e-6)
    a = torch.tensor([1, 3])
    lp, entropy, _ = joint_statistics(z, a)
    lp.backward()
    expected = torch.nn.functional.one_hot(a, 5) - z.detach().softmax(-1)
    assert torch.allclose(z.grad, expected, atol=1e-6)
    assert torch.isfinite(entropy)


def test_joint_ppo_ratio_clips_once_for_whole_map():
    import torch
    from ember_v1.rl_ppo import ppo_actor_loss
    # Two persons each change likelihood by 1.15: true joint ratio 1.3225 clips at 1.2.
    new = torch.tensor([2 * __import__("math").log(1.15)], requires_grad=True)
    loss, ratio = ppo_actor_loss(new, torch.zeros(1), torch.ones(1), 0.2)
    assert abs(float(ratio.detach()) - 1.3225) < 1e-6
    assert abs(float(loss.detach()) + 1.2) < 1e-6
    loss.backward()
    assert float(new.grad) == 0


def test_ppo_negative_advantage_and_nonfinite_rejected():
    import torch
    import pytest
    from ember_v1.rl_ppo import ppo_actor_loss
    loss, _ = ppo_actor_loss(torch.tensor([0.5]).log(), torch.zeros(1), -torch.ones(1), 0.2)
    assert float(loss) == pytest.approx(0.8)
    with pytest.raises(FloatingPointError):
        ppo_actor_loss(torch.tensor([float("nan")]), torch.zeros(1), torch.ones(1), 0.2)


def audit_checkpoint(model_root, evidence):
    """Independently replay every first-rollout behavior batch WITH autograd enabled.

    Tests identical gradient/inference forward probabilities, person ordering and sum,
    and executes a real backward on a real simulator observation, without an optimizer.
    """
    import json
    from pathlib import Path
    import unsloth
    import torch
    from ember_v1.rl_ppo import V1Actor, joint_statistics
    from ember_v1.student_train import ACTION_ORDER, config_from_protocol, load_protocol
    from ember_v1.rl_run import write, REPO
    model_root, evidence = Path(model_root), Path(evidence)
    identity = json.loads((model_root / "identity.json").read_text())
    actor = V1Actor(config_from_protocol(load_protocol(REPO / identity["protocol"]["path"])), model_root / "final")
    actor.set_trainable(True)
    rows = [json.loads(line) for line in (evidence / "transitions.jsonl").open()]
    first = [r for r in rows if r["label"] == "train-0"]
    scenario_ids = list(dict.fromkeys(r["scenario_id"] for r in first))
    parallel = json.loads((evidence / "execution.json").read_text())["config"]["parallel_envs"]
    by_id = {sid: [r for r in first if r["scenario_id"] == sid] for sid in scenario_ids}
    max_error, groups, backward_done = 0.0, 0, False
    for start in range(0, len(scenario_ids), parallel):
        block = scenario_ids[start:start+parallel]
        for step in range(max(len(by_id[sid]) for sid in block)):
            transitions = [by_id[sid][step] for sid in block if len(by_id[sid]) > step]
            batch = actor.prepare([r["observation"] for r in transitions])
            zs = actor.logits(batch)
            loss = 0
            for r, ids, z in zip(transitions, batch["ids"], zs):
                a = torch.tensor([ACTION_ORDER.index(r["sampled_actions"][pid]) for pid in ids], device="cuda")
                lp, _, full_lp = joint_statistics(z, a)
                max_error = max(max_error, abs(float(lp.detach()) - r["old_joint_logprob"]),
                                float((full_lp.detach().cpu()-torch.tensor(r["person_logprobs"])).abs().max()))
                loss = loss - lp / len(transitions)
            if not backward_done:
                loss.backward()
                norms = {}
                for label in ("lora", "head"):
                    grads = [p.grad for n, p in actor.model.named_parameters() if p.grad is not None
                             and ((label == "lora" and "lora_" in n) or (label == "head" and n.startswith("head.")))]
                    assert grads and all(torch.isfinite(g).all() for g in grads)
                    norms[label] = sum(float(g.float().square().sum()) for g in grads)**0.5
                    assert norms[label] > 0
                backward_done = True
                actor.model.zero_grad(set_to_none=True)
            groups += 1
            del zs, loss, batch, lp, full_lp, z
    result = {"autograd_behavior_replay_max_error": max_error, "groups": groups,
              "transitions": len(first), "finite_nonzero_gradient_norms": norms}
    write(evidence / "autograd_probability_audit.json", result)
    print(result)
    assert max_error <= 1e-5


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()
    audit_checkpoint(args.model_root, args.evidence)
