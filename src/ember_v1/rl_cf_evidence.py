"""CPU-only resume ancestry reconciliation and paired reporting statistics."""
from __future__ import annotations

import json
import random
from pathlib import Path


def read_rows(path, *, allow_trailing_partial=False):
    if not path.exists():
        return []
    lines = path.read_text().splitlines(keepends=True)
    result = []
    for i, line in enumerate(lines):
        if not line.endswith("\n") and allow_trailing_partial and i == len(lines)-1:
            break  # preserved failed-attempt suffix; never part of a committed update
        result.append(json.loads(line))
    return result


def lineage_roots(root):
    from .rl_online import hash_path
    root = Path(root).resolve()
    expected = json.loads((root/"execution.json").read_text())
    roots, current = [], root
    while current is not None:
        if current in roots or len(roots) >= 8 or current.parent != root.parent:
            raise ValueError("invalid or cyclic resume evidence ancestry")
        execution = json.loads((current/"execution.json").read_text())
        if any(execution[k] != expected[k] for k in ("config_sha256", "parent_binding")):
            raise ValueError("resume ancestry config/parent differs")
        roots.append(current)
        if not execution.get("resume"):
            current = None
            continue
        resume = Path(execution["resume"]).resolve()
        if resume.parent.name != "resume":
            raise ValueError("paired resume is not under its source evidence root")
        identity = json.loads((resume/"identity.json").read_text())
        if identity["config_sha256"] != expected["config_sha256"] or identity["sha256"] != hash_path(resume/"state.pt"):
            raise ValueError("paired resume bytes do not match their declared identity")
        current = resume.parent.parent
    return roots


def reconcile_history(root, history):
    """Every carried update and episode must match its original raw attempt logs."""
    lineage = set(lineage_roots(root))
    proof = {}
    cache = {}
    for arm in ("ppo", "cf"):
        if [u["update"] for u in history[arm]["updates"]] != list(range(1, len(history[arm]["updates"])+1)):
            raise ValueError("carried training history omits or duplicates an earlier update")
        reconstructed = []
        for update in history[arm]["updates"]:
            origin = Path(update["evidence_root"]).resolve()
            if origin not in lineage:
                raise ValueError("training history points outside the verified resume ancestry")
            key = origin, arm
            if key not in cache:
                interrupted = (origin/"failure.json").exists()
                raw_updates = read_rows(origin/arm/"updates.jsonl", allow_trailing_partial=interrupted)
                raw_episodes = read_rows(origin/arm/"episodes.jsonl", allow_trailing_partial=interrupted)
                cache[key] = ({r["update"]: r for r in raw_updates},
                              {(r["label"], r["scenario_id"]): r for r in raw_episodes})
            raw_update, raw_episode = cache[key]
            if raw_update.get(update["update"]) != update:
                raise ValueError("carried training history differs from the original raw update")
            selected = proof.setdefault(key, set())
            for sid in update["episode_ids"]:
                episode_key = f'train-{update["update"]-1}', sid
                if episode_key in selected or episode_key not in raw_episode:
                    raise ValueError("missing or duplicated carried raw training episode")
                reconstructed.append(raw_episode[episode_key])
                selected.add(episode_key)
        if reconstructed != history[arm]["episodes"]:
            raise ValueError("aggregate training episodes do not reconcile to prior/current raw evidence")
    return proof


METRICS = {
    "evacuated": (lambda r: r["evacuated"], lambda r: r["people"], "proportion"),
    "casualties": (lambda r: r["casualties"], lambda r: r["people"], "proportion"),
    "unresolved": (lambda r: r["unresolved"], lambda r: r["people"], "proportion"),
    "stay_rate": (lambda r: r["stay"], lambda r: r["person_steps"], "proportion"),
    "blocked_rate": (lambda r: r["blocked"], lambda r: r["attempted"], "proportion"),
    "coordination_conflict_rate": (lambda r: sum(r.get(k, 0) for k in ("blocked_conflict", "blocked_occupied", "blocked_cycle")),
                                   lambda r: r["attempted"], "proportion"),
    "repeated_ineffective_rate": (lambda r: r.get("repeated_ineffective_moves", 0), lambda r: r["attempted"], "proportion"),
    "mean_joint_latency_ms": (lambda r: 1000*r.get("inference_seconds", 0), lambda r: r["steps"], "milliseconds"),
    "restricted_episode_steps": (lambda r: r["steps"], lambda r: 1, "steps"),
}


def paired_statistics(a, b, repeats=5000):
    a, b = sorted(a, key=lambda r: r["scenario_id"]), sorted(b, key=lambda r: r["scenario_id"])
    if not a or [r["scenario_id"] for r in a] != [r["scenario_id"] for r in b]:
        raise ValueError("paired reporting needs identical matched scenario families")
    result = {}
    for metric, (numerator, denominator, unit) in METRICS.items():
        aa = [(numerator(r), denominator(r)) for r in a]
        bb = [(numerator(r), denominator(r)) for r in b]
        def difference(indices):
            return (sum(bb[i][0] for i in indices)/max(1, sum(bb[i][1] for i in indices))
                    -sum(aa[i][0] for i in indices)/max(1, sum(aa[i][1] for i in indices)))
        rng = random.Random(922)
        values = sorted(difference(rng.choices(range(len(a)), k=len(a))) for _ in range(repeats))
        result[metric] = {"difference": difference(range(len(a))), "lower_95": values[int(.025*repeats)],
                          "upper_95": values[int(.975*repeats)], "unit": unit,
                          "families": len(a), "bootstrap_replicates": repeats}
    return result


def reporting_statistics(root):
    root = Path(root)
    ppo = [r for r in read_rows(root/"ppo"/"episodes.jsonl") if r["label"] == "ppo"]
    cf = [r for r in read_rows(root/"cf"/"episodes.jsonl") if r["label"] == "cf"]
    document = json.loads((root/"scenarios.json").read_text())
    expected = {s["scenario_id"] for s in document["report"]}
    if {r["scenario_id"] for r in ppo} != expected or {r["scenario_id"] for r in cf} != expected:
        raise ValueError("reporting episodes do not match the fixed independent DEV panel")
    families = sorted({document["metadata"][sid]["family"] for sid in expected})
    return {"overall": paired_statistics(ppo, cf),
            "by_family": {name: paired_statistics([r for r in ppo if document["metadata"][r["scenario_id"]]["family"] == name],
                                                  [r for r in cf if document["metadata"][r["scenario_id"]]["family"] == name], repeats=2000)
                          for name in families},
            "interpretation": "Paired bootstrap over independently generated topology families; conditional on one training seed per arm."}
