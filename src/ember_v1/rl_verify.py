"""Independent, CPU-only replay of RL evidence, action likelihoods, rewards and GAE."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from .rl_env import EvacuationEnv, RewardConfig, gae
from .sim import Scenario
from .student_train import ACTION_ORDER


def verify(root: Path):
    cfg = json.loads((root / "execution.json").read_text())["config"]
    scenarios = json.loads((root / "scenarios.json").read_text())
    scenarios = {s["scenario_id"]: Scenario.from_dict(s) for split in ("train", "dev") for s in scenarios[split]}
    envs, histories = {}, defaultdict(list)
    errors, probability_error, normalization_error, transitions = [], 0.0, 0.0, 0
    rc = RewardConfig(cfg["gamma"], cfg["shaping_eta"], cfg["evacuated_weight"],
                      cfg["casualty_weight"], cfg["unresolved_weight"])
    for line in (root / "transitions.jsonl").open():
        row = json.loads(line)
        key = row["label"], row["scenario_id"]
        if key not in envs:
            envs[key] = EvacuationEnv(scenarios[key[1]], rc)
            envs[key].reset()
        env = envs[key]
        if row["observation"] != env.sim.observe() or row["sampled_actions"] != row["executed_actions"]:
            errors.append(f"{key}: observation/action mismatch")
        ids = env.sim.active_ids()
        distributions = row["person_logprobs"]
        if len(distributions) != len(ids) or any(len(lp) != 5 for lp in distributions):
            raise ValueError("action distribution does not match active people")
        selected = sum(lp[ACTION_ORDER.index(row["sampled_actions"][pid])] for pid, lp in zip(ids, distributions))
        probability_error = max(probability_error, abs(selected-row["old_joint_logprob"]))
        normalization_error = max(normalization_error, max(abs(sum(math.exp(x) for x in lp)-1) for lp in distributions))
        _, reward, done, _, info = env.step(row["executed_actions"])
        if info["record"] != row["record"] or done != row["done"] or abs(reward-row["reward"]) > 1e-10:
            errors.append(f"{key}: transition/reward mismatch")
        if row["label"].startswith("train-"):
            histories[key].append(row)
        transitions += 1
    unfinished = [key for key, env in envs.items() if not env.sim.terminated]
    if unfinished:
        errors.append(f"unfinished episodes: {unfinished}")
    advantages = []
    for key, history in histories.items():
        adv, returns = gae([r["reward"] for r in history], [r["value"] for r in history],
                           [r["done"] for r in history], cfg["gamma"], cfg["gae_lambda"])
        for row, a, ret in zip(history, adv, returns):
            advantages.append({"label": key[0], "scenario_id": key[1], "step": row["observation"]["step"],
                               "advantage": a, "return": ret})
    if probability_error > 1e-8 or normalization_error > 1e-6:
        errors.append("action probability audit failed")
    if any(not math.isfinite(r[k]) for r in advantages for k in ("advantage", "return")):
        errors.append("nonfinite advantages/returns")
    result = {"status": "PASS" if not errors else "FAIL", "episodes_replayed": len(envs),
              "transitions_replayed": transitions, "training_advantages_recomputed": len(advantages),
              "max_joint_sum_error": probability_error, "max_distribution_normalization_error": normalization_error,
              "errors": errors}
    (root / "advantages.jsonl").write_text("".join(json.dumps(a, sort_keys=True)+"\n" for a in advantages))
    (root / "independent_replay.json").write_text(json.dumps(result, indent=2)+"\n")
    if errors:
        raise RuntimeError(result)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("evidence", type=Path)
    print(json.dumps(verify(parser.parse_args().evidence)))


if __name__ == "__main__":
    main()
