"""Reproducible CPU analysis of the completed online PPO experiment."""
from __future__ import annotations

import argparse
import collections
import html
import json
import random
import statistics
from pathlib import Path

from ember_v1.rl_run import summarize, write


def read_rows(path):
    return [json.loads(line) for line in path.open()] if path.exists() else []


def ci_paired(a, b, repeats=5000):
    a, b = sorted(a, key=lambda r: r["scenario_id"]), sorted(b, key=lambda r: r["scenario_id"])
    assert [r["scenario_id"] for r in a] == [r["scenario_id"] for r in b]
    rng = random.Random(921)
    result = {}
    for metric in ("evacuated", "casualties", "unresolved"):
        values = []
        for _ in range(repeats):
            indices = rng.choices(range(len(a)), k=len(a))
            values.append(sum(b[i][metric]-a[i][metric] for i in indices)/sum(a[i]["people"] for i in indices))
        values.sort()
        result[metric] = {"difference": sum(y[metric]-x[metric] for x, y in zip(a, b))/sum(x["people"] for x in a),
                          "lower_95": values[int(.025*repeats)], "upper_95": values[int(.975*repeats)]}
    return result


def line_plot(series, title, width=740, height=210):
    colors = ["#276b50", "#b04939", "#5568b0", "#ac762d"]
    all_points = [p for _, points in series for p in points]
    if not all_points:
        return ""
    lo, hi = min(p[1] for p in all_points), max(p[1] for p in all_points)
    if hi == lo:
        hi += 1
    mx = max(p[0] for p in all_points) or 1
    def point(x, y):
        return f"{55+x/mx*(width-75):.1f},{height-35-(y-lo)/(hi-lo)*(height-65):.1f}"
    svg = f'<h3>{html.escape(title)}</h3><svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">'
    for val, y in ((hi, 30), (lo, height-35)):
        svg += f'<text x="2" y="{y}" font-size="12">{val:.3g}</text><path d="M50 {y}H{width-20}" stroke="var(--border,#d4d9d2)"/>'
    for i, (label, points) in enumerate(series):
        svg += f'<polyline fill="none" stroke="{colors[i]}" stroke-width="2" points="'+" ".join(point(x,y) for x,y in points)+'"/>'
        svg += f'<text x="{65+i*170}" y="{height-5}" fill="{colors[i]}" font-size="12">{html.escape(label)}</text>'
    return svg+'</svg>'


def report(root):
    summary = json.loads((root / "summary.json").read_text())
    cfg = json.loads((root / "execution.json").read_text())["config"]
    scenarios = json.loads((root / "scenarios.json").read_text())
    best = json.loads((root / "best_checkpoint.json").read_text())
    episodes = read_rows(root / "episodes.jsonl")
    updates = read_rows(root / "updates.jsonl")
    arms = ("v1", "v1-mask", "v1-coordinate", "phase1", "ppo")
    by_arm = {a: [r for r in episodes if r["label"] == a] for a in arms}
    uncertainty = {a: ci_paired(by_arm["v1"], by_arm[a]) for a in arms[1:]}
    per_family = {}
    for a in arms:
        family = collections.defaultdict(list)
        for r in by_arm[a]:
            family[scenarios["metadata"][r["scenario_id"]]["stratum"]].append(r)
        per_family[a] = {f: summarize(rows) for f, rows in family.items()}
    latency = collections.defaultdict(list)
    for r in read_rows(root / "transitions.jsonl"):
        if r["label"] in arms:
            latency[r["label"]].append(r["inference_seconds"]*1000)
    latency_stats = {}
    for arm, values in latency.items():
        values.sort()
        latency_stats[arm] = {"mean_ms": statistics.mean(values), "p50_ms": statistics.median(values),
                              "p95_ms": values[min(len(values)-1, int(.95*len(values)))], "joint_states": len(values)}
    telemetry = read_rows(root / "gpu.jsonl")
    gpu = []
    for r in telemetry:
        try:
            _, util, mem, power = r["sample"].split(",")
            gpu.append((float(util), float(mem), float(power)))
        except (ValueError, KeyError):
            continue
    gpu_stats = {"mean_utilization_percent": statistics.mean(v[0] for v in gpu),
                 "max_utilization_percent": max(v[0] for v in gpu), "peak_memory_mib": max(v[1] for v in gpu),
                 "mean_power_watts": statistics.mean(v[2] for v in gpu), "samples": len(gpu)}
    means = {key: statistics.mean(m[key] for u in updates for m in u["minibatches"])
             for key in ("actor_loss", "critic_loss", "joint_kl", "joint_entropy", "person_entropy", "clip_fraction")}
    diagnostics = {arm: {key: sum(r.get(key, 0) for r in by_arm[arm]) for key in (
        "ineffective_moves", "repeated_ineffective_moves", "blocked_conflict", "blocked_occupied", "blocked_cycle",
        "action_east", "action_north", "action_south", "action_west", "action_stay", "postprocess_changes")} for arm in arms}
    derived = {"paired_family_bootstrap_95": uncertainty, "bootstrap_replicates": 5000,
               "per_family": per_family, "latency": latency_stats, "gpu": gpu_stats, "optimization_means": means,
               "diagnostics": diagnostics, "probability_replay_max_error": max(u["max_probability_replay_error"] for u in updates),
               "kl_early_stop_updates": sum(u["kl_early_stop"] for u in updates),
               "collection_steps_per_second": sum(u["rollout"]["steps"] for u in updates)/sum(u["collect_seconds"] for u in updates)}
    write(root / "analysis.json", derived)
    table = "| Reporting DEV arm | E | C | U | Stay | Blocked / attempted |\n|---|---:|---:|---:|---:|---:|\n"
    for a in arms:
        s = summary["comparisons"][a]
        table += f'| {a} | {s["evacuated_rate"]:.2%} | {s["casualties_rate"]:.2%} | {s["unresolved_rate"]:.2%} | {s["stay_rate"]:.2%} | {s["blocked_per_attempt"]:.2%} |\n'
    diff = uncertainty["ppo"]
    success = diff["evacuated"]["lower_95"] > 0 and diff["casualties"]["upper_95"] <= .02
    verdict = ("Improved evacuation with the prespecified casualty tolerance supported on reporting DEV."
               if success else "The prespecified improvement and casualty criterion was not established on reporting DEV.")
    text = f'''# EMBER V1 Phase 2 — online PPO report

{verdict} Selection and reporting use newly generated independent maps, not EMBERBench or Run 19.

{table}
Reporting DEV: {summary["comparisons"]["v1"]["episodes"]} independent geometry families,
{summary["comparisons"]["v1"]["people"]} initial people. All arms use greedy decoding. Masking and
coordination are existing external heuristics applied only to original V1; PPO is unmasked.

## Experiment and compute

Phase 2 collected {summary["training"]["episodes"]} on-policy episodes,
{summary["training"]["steps"]} simultaneous environment transitions and
{summary["training"]["person_steps"]} active-person decisions; {summary["optimizer_steps"]} actual Adam steps.
Parent Phase 1 adds 64 episodes / 1,946 transitions / 141 steps to the model's RL history.
Process wall time {summary["elapsed_seconds"]:.2f} seconds, prior failed attempts
{summary["prior_wall_seconds"]:.2f} seconds; training loop including periodic DEV/saving
{summary["training_seconds"]:.2f} seconds. Six-hour aggregate cap, no paid remote GPU.
Collection throughput {derived["collection_steps_per_second"]:.3f} transitions/s.
Whole-process GPU telemetry: mean {gpu_stats["mean_utilization_percent"]:.2f}% utilization,
peak {gpu_stats["max_utilization_percent"]:.0f}%, peak {gpu_stats["peak_memory_mib"]:.0f} MiB;
Torch peak allocated/reserved {summary["torch_peak_allocated_bytes"]/2**30:.3f}/
{summary["torch_peak_reserved_bytes"]/2**30:.3f} GiB. One public state per GPU forward avoids
the Phase 1 WDDM memory problem; the CPU worker interface supports parallel environments,
but this run dispatched one simulator transition at a time. Encoding, simulation, backward,
checkpoint writes and evaluation prevent continuous GPU saturation.

## Objective, optimization and verification

Exact original public observations and simultaneous actions; unmodified categorical
samples; one product policy with summed person log probabilities and a single PPO ratio.
Restored Phase 1 LoRA, float32 Clef head, centralized critic and Adam moments. Base weights
frozen. No SFT, expansion data, teacher outcomes or hidden simulator access. Terminal
`(E-2C-.75U)/N`; gamma=1, GAE lambda=.97, public-distance shaping eta=.1 with terminal zero.
Increasing entropy from .001 to .005 tests exploration without a weak-SFT reference anchor.
No movement or stay reward. Full frozen configuration and scenario JSON accompany the run.

Training mean shaped return {summary["training"]["mean_reward"]:.6f}, terminal outcome return
{summary["training"]["mean_terminal_reward"]:.6f}. Logged minibatch means: {json.dumps(means)}.
KL stopping occurred in {derived["kl_early_stop_updates"]} updates; each applied step checked
finite nonzero LoRA/head/critic gradients. Pre-update behavior replay maximum error
{derived["probability_replay_max_error"]:.8g}. Actual PPO advantages/returns are saved for
independent CPU recomputation. Exact selected save/reload parameter digest matched.
Final-versus-initial trainable parameter delta: {json.dumps(summary["parameter_delta"])}.

## Selection and uncertainty

Fixed 32 topology-disjoint selection families every 128 episodes; maximize outcome utility
subject to at most +2 percentage points casualties versus raw V1. Selected update
{best["update"]}; safety eligible: {best["safety_eligible"]}. Reporting families never select
weights or configurations. No hyperparameter sweep was performed.

Paired family bootstrap (5,000 replicates), PPO minus V1, percentage points:
'''
    for metric, values in diff.items():
        text += f'- {metric}: {100*values["difference"]:+.2f} [{100*values["lower_95"]:+.2f}, {100*values["upper_95"]:+.2f}].\n'
    text += '\nThe intervals reflect 32 sampled geometry families and one fixed simulator seed per family, not uncertainty over training seeds or unseen deployment distributions. Selection DEV is adaptive; the separate reporting split limits selection optimism. No official benchmark claim follows.\n\n## Per-family outcomes\n\n| Stratum | V1 E/C/U | PPO E/C/U |\n|---|---|---|\n'
    for family in sorted(per_family["v1"]):
        def fmt(a):
            s = per_family[a][family]
            return '/'.join(f'{s[k+"_rate"]:.1%}' for k in ("evacuated", "casualties", "unresolved"))
        text += f'| {family} | {fmt("v1")} | {fmt("ppo")} |\n'
    text += '\n## Latency and behavior\n\n| Arm | Joint forward p50 / p95 ms | Repeated ineffective moves |\n|---|---:|---:|\n'
    for a in arms:
        text += f'| {a} | {latency_stats[a]["p50_ms"]:.2f} / {latency_stats[a]["p95_ms"]:.2f} | {diagnostics[a]["repeated_ineffective_moves"]} |\n'
    text += '\nCUDA-synchronized forward and action-selection latency, batch size one; encoding and CPU simulation excluded. Full action-direction, conflict and repeat counts are in analysis.json. No behavior diagnostic is a training reward.\n\n## Checkpoints and reproduction\n\n'
    text += f'- Selected actor: `{best["actor"]}`.\n- Resumable trainer: `{best["trainer"]}`.\n- Actor digest: `{best["identity"]["actor_digest"]}`.\n- Training source: `{best["identity"]["code_commit"]}`.\n'
    text += '\nRun `python -m ember_v1.rl_online --config data/rl/r21-online-ppo@1.json --model-root <verified-SFT-root> --phase1 <verified-Phase1-evidence> --output-root <external-root> --attempt <unused-authorized-attempt>` in the pinned Python 3.11 GPU environment. `--resume <complete-checkpoint-dir>` restores actor, critic, optimizer, RNG and counters; interrupted partial rollouts are preserved and a retry uses fresh TRAIN families. Re-running an occupied attempt or completed experiment is refused.\n\nRun `python -m ember_v1.rl_verify <attempt>` and `python experiments/rl_online_report.py <attempt>` to recompute the audits and this report. All baseline/parent hashes are verified before and after training. Exact manifests bind raw evidence outside Git. Attempts 1 and 2 stopped before GPU/rollouts on geometry capacity and D4 isolation guards; fixes and negative evidence are preserved. No Phase 3 work is authorized by this run.\n'
    (root / "REPORT.md").write_text(text)
    charts = ''
    for title, keys in (("TRAIN outcome rates", ("evacuated_rate", "casualties_rate", "unresolved_rate")),):
        charts += line_plot([(k.replace('_rate',''), [(u["update"], u["rollout"][k]) for u in updates]) for k in keys], title)
    charts += line_plot([("shaped return", [(u["update"], u["rollout"]["mean_reward"]) for u in updates])], "TRAIN return by update")
    for key in ("actor_loss", "critic_loss", "joint_kl", "person_entropy"):
        charts += line_plot([(key, [(u["update"], statistics.mean(m[key] for m in u["minibatches"])) for u in updates])], key)
    bars = '<h2>Independent reporting DEV</h2><p>Evacuated · Casualties · Unresolved</p>'
    for a in arms:
        s = summary["comparisons"][a]
        bars += f'<p>{a} — {s["evacuated_rate"]:.1%} / {s["casualties_rate"]:.1%} / {s["unresolved_rate"]:.1%}</p><div class="bar">'
        for key, color in (("evacuated", "#276b50"), ("casualties", "#b04939"), ("unresolved", "#acaa9a")):
            bars += f'<span style="width:{s[key+"_rate"]*100}%;background:{color}"></span>'
        bars += '</div>'
    page = '<!doctype html><html><head><meta charset="utf-8"><title>EMBER Phase 2</title><style>body{font:16px var(--font-sans,system-ui);color:var(--foreground);margin:0}h1{font-size:28px}.bar{display:flex;height:24px;border-radius:3px;overflow:hidden}svg{width:100%;height:210px}h3{margin-top:30px}</style></head><body><h1>EMBER V1 · online PPO</h1><p>'+html.escape(verdict)+'</p>'+bars+charts+'</body></html>'
    (root / "curves.html").write_text(page)
    return derived


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    report(args.evidence)
