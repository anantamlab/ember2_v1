"""Recompute the PPO-only versus counterfactual comparison from raw evidence."""
from __future__ import annotations

import argparse
import html
import json
import statistics
from pathlib import Path

from ember_v1.rl_run import summarize, write
from ember_v1.rl_cf_evidence import reporting_statistics
from rl_online_report import ci_paired, line_plot, read_rows


def report(root):
    summary = json.loads((root/"summary.json").read_text())
    audit = json.loads((root/"independent_counterfactual_audit.json").read_text())
    history = json.loads((root/"training_history.json").read_text())
    best = json.loads((root/"best_checkpoint.json").read_text())
    documents = json.loads((root/"scenarios.json").read_text())
    episodes = read_rows(root/"episodes.jsonl")
    by_arm = {a: [r for r in episodes if r["label"] == a] for a in ("v1", "phase2")}
    by_arm.update({a: [r for r in read_rows(root/a/"episodes.jsonl") if r["label"] == a] for a in ("ppo", "cf")})
    paired = reporting_statistics(root)
    if paired != audit["paired_reporting"]:
        raise ValueError("paired report differs from independently audited raw evidence")
    delta = {k: paired["overall"][k] for k in ("evacuated", "casualties", "unresolved")}
    credible = delta["evacuated"]["lower_95"] > 0 and delta["casualties"]["upper_95"] <= .02
    verdict = ("Counterfactual-assisted PPO shows a statistically supported evacuation benefit beyond PPO-only on reporting DEV within the casualty tolerance."
               if credible else "A statistically credible evacuation benefit beyond PPO-only without casualty regression was not established on reporting DEV.")
    teachers = read_rows(root/"teacher_episodes.jsonl")
    common = set(r["scenario_id"] for r in by_arm["cf"])
    for name in ("T0", "T1", "T2", "T3", "T4"):
        common &= {r["scenario_id"] for r in teachers if r["label"] == name and r["status"] == "COMPLETE"}
    teacher_panel = {a: summarize([r for r in rows if r["scenario_id"] in common]) for a, rows in by_arm.items()} if common else {}
    if common:
        teacher_panel.update({name: summarize([r for r in teachers if r["label"] == name and r["scenario_id"] in common and r["status"] == "COMPLETE"])
                              for name in ("T0", "T1", "T2", "T3", "T4")})
    families = {}
    for arm, values in by_arm.items():
        families[arm] = {family: summarize([r for r in values if documents["metadata"][r["scenario_id"]]["family"] == family])
                         for family in sorted({m["family"] for m in documents["metadata"].values()})}
    latency = {}
    for arm in by_arm:
        folder = root/arm if arm in ("ppo", "cf") else root
        times = sorted(r["inference_seconds"]*1000 for r in read_rows(folder/"transitions.jsonl") if r["label"] == arm)
        latency[arm] = {"p50_joint_ms": statistics.median(times), "p95_joint_ms": times[int(.95*len(times))]}
    for name in ("T0", "T1", "T2", "T3", "T4"):
        times = sorted(d["wall_ms"] for r in teachers if r["label"] == name and r["status"] == "COMPLETE" for d in r["decisions"])
        if times:
            latency[name] = {"p50_joint_ms": statistics.median(times), "p95_joint_ms": times[int(.95*len(times))]}
    effects = audit["teacher_alternative_effects"]
    for value in effects.values():
        value["mean_utility_delta"] = value["utility_sum"]/value["states"]
    replay = {a: max(u["max_probability_replay_error"] for u in h["updates"]) for a, h in history.items()}
    optimization = {a: {key: statistics.mean(m[key] for u in h["updates"] for m in u["minibatches"])
                        for key in ("actor_loss", "critic_loss", "joint_kl", "person_entropy", "clip_fraction")}
                    for a, h in history.items()}
    data = {"verdict": verdict, "credible_cf_benefit": credible, "cf_minus_ppo_paired_family_95": delta,
            "matched_teacher_panel_families": len(common), "matched_teacher_panel": teacher_panel,
            "per_family": families, "latency": latency, "teacher_alternative_effects": effects,
            "preupdate_probability_replay_max": replay, "optimization_means": optimization,
            "paired_behavior_and_family_uncertainty": paired}
    write(root/"analysis.json", data)
    table = '| Raw learned policy | Evacuated | Casualties | Unresolved | Stay | Blocked / attempted |\n|---|---:|---:|---:|---:|---:|\n'
    for arm in by_arm:
        s = summary["comparisons"][arm]
        table += f'| {arm} | {s["evacuated_rate"]:.2%} | {s["casualties_rate"]:.2%} | {s["unresolved_rate"]:.2%} | {s["stay_rate"]:.2%} | {s["blocked_per_attempt"]:.2%} |\n'
    text = f'''# EMBER V1 Phase 3 — counterfactual-assisted online PPO

{verdict}

{table}
Reporting DEV has 32 independent fresh topology families, separate from selection DEV and
all TRAIN experience. Original V1, frozen selected Phase 2, equal-budget PPO continuation
and equal-budget PPO+CF use raw argmax, without external movement masking/coordination.
All remain in the original Qwen3.5-2B V1 / LoRA / Clef lineage. No SFT, V1.1 artifacts,
expansion data or hidden benchmark simulator was used. Final evaluation remains sealed.

## Actual training and compute

Both arms completed {summary["completed_matched_rounds"]} matched rounds. On-policy metrics:
'''
    for arm in ("ppo", "cf"):
        s = summary["training"][arm]
        text += f'- {arm}: {s["episodes"]} episodes / {s["steps"]} joint transitions / {s["person_steps"]} person decisions; {summary["optimizer_steps"][arm]} optimizer steps. Mean shaped/terminal returns {s["mean_reward"]:.6f}/{s["mean_terminal_reward"]:.6f}.\n'
    text += f'\nCF auxiliary gradients were applied in {summary["auxiliary_optimizer_steps"]} actual optimizer steps. Process wall {summary["elapsed_seconds"]:.2f}s plus {summary["prior_wall_seconds"]:.2f}s prior failed attempts. GPU peak allocated/reserved {summary["torch_peak_allocated_bytes"]/2**30:.3f}/{summary["torch_peak_reserved_bytes"]/2**30:.3f} GiB. The model base stayed frozen. Both arms restored the same Phase 2 actor/critic/Adam and used identical TRAIN-controlled curriculum stages, family pools and episode budget. Their on-policy visited states naturally diverged after learning. Full GPU utilization/power samples are preserved in gpu.jsonl.\n'
    text += f'\nPPO settings and primary outcome reward are fixed in the experiment config. The CF-only auxiliary coefficient is .05; no offline record is counted as an on-policy episode. Optimization means: `{json.dumps(optimization)}`. Maximum pre-update replay discrepancy: `{json.dumps(replay)}`.\n'
    totals = audit["counterfactual"]
    text += f'''\n## Verified counterfactual evidence

{totals["studies"]} actual TRAIN visited-state studies, {totals["branches"]} branch episodes,
{totals["transitions"]} counterfactual transitions, {totals["eligible_pairs"]} selected offline
pairs. These counts are separate from PPO interactions. Each study stores its full private
snapshot/RNG/chain, independently replayed factual prefix and matching public observation.
Policies saw public observations only. Interventions include the factual student sample,
student argmax, all available T0–T4 proposals and seeded two-person alternatives, deduplicated.

Every candidate starts from the identical saved state within each fire realization.
Deterministic fire has one exact branch; stochastic studies use the factual future stream
plus seven matched controlled forks. Every common-step fire ignition/draw count and every
branch outcome was independently replayed. All continuations are the same frozen default
T1 policy, whose actions were independently regenerated. Reported effects are conditional
on that continuation; they are not individual causal blame or globally optimal Q estimates.

Value comes from simulator `(E-2C-.75U)/N`, not teacher scores. Pairwise utility advantages,
evacuation/casualty/unresolved deltas, restricted completion time, censoring and empirical
future-fire uncertainty remain in every study. Training admits a positive-versus-factual
pair only when its empirical paired lower interval is positive and measured casualties
never increase. Candidate-selection and small-fire-sample uncertainty are exploratory;
only the independent reporting comparison can support a generalization claim.

| Proposal source | States | Helped | Harmed | Tied | Mean utility change vs factual |
|---|---:|---:|---:|---:|---:|
'''
    for name in ("T0", "T1", "T2", "T3", "T4"):
        e = effects.get(name)
        if e:
            text += f'| {name} | {e["states"]} | {e["helped"]} | {e["harmed"]} | {e["tied"]} | {e["mean_utility_delta"]:+.5f} |\n'
    text += '\nCF proposal T2/T3 use the explicitly recorded bounded configurations (500ms/24 rollouts and 200ms/20,000 ant steps); their effects must not be confused with the frozen default teachers evaluated below. Other proposal configurations and every failure/limit flag are preserved. Local variations can beat, tie or harm the factual action independently of any teacher.\n\n## Independent uncertainty\n\nPaired 5,000-replicate bootstrap over fixed reporting families, CF minus PPO, percentage points:\n'
    for metric, d in delta.items():
        text += f'- {metric}: {100*d["difference"]:+.2f} [{100*d["lower_95"]:+.2f}, {100*d["upper_95"]:+.2f}].\n'
    text += '\nBehavior and latency deltas with paired family intervals:\n'
    for metric, d in paired["overall"].items():
        if metric in delta:
            continue
        factor, unit = (100, 'pp') if d["unit"] == 'proportion' else (1, d["unit"])
        text += f'- {metric}: {factor*d["difference"]:+.3f} [{factor*d["lower_95"]:+.3f}, {factor*d["upper_95"]:+.3f}] {unit}.\n'
    text += '\nIntervals describe these sampled geometry families and simulator seeds. One training seed per arm does not quantify training-seed variability. Selection DEV chose checkpoints; reporting DEV did not affect gradients, difficulty or selection. No claim of superiority over all teachers follows from isolated counterfactual wins.\n\n## Frozen default teacher comparison\n\n'
    text += f'Matched completed panel: {len(common)} reporting families; {summary["teacher_failures"]} teacher episodes failed or timed out and remain explicit incomplete records. The table uses the same scenario IDs for every arm. A reduced panel is conditional on completion and cannot establish population superiority.\n\n| Arm | E | C | U | Stay | Blocked / attempted |\n|---|---:|---:|---:|---:|---:|\n'
    for arm, s in teacher_panel.items():
        text += f'| {arm} | {s["evacuated_rate"]:.2%} | {s["casualties_rate"]:.2%} | {s["unresolved_rate"]:.2%} | {s["stay_rate"]:.2%} | {s["blocked_per_attempt"]:.2%} |\n'
    text += '\n## Generalization and latency\n\n| Family | PPO E/C/U | CF E/C/U |\n|---|---|---|\n'
    for family in families["ppo"]:
        def fmt(a):
            return '/'.join(f'{families[a][family][k+"_rate"]:.1%}' for k in ("evacuated", "casualties", "unresolved"))
        text += f'| {family} | {fmt("ppo")} | {fmt("cf")} |\n'
    text += '\n| Arm | Joint decision p50 / p95 ms |\n|---|---:|\n'
    for arm, d in latency.items():
        text += f'| {arm} | {d["p50_joint_ms"]:.2f} / {d["p95_joint_ms"]:.2f} |\n'
    text += '\nNeural latency is CUDA-synchronized forward plus action selection, excluding encoding and simulator time. Teacher latency is its public-observation decision wall time. Workload sizes and hardware differ; raw latency distributions remain available.\n\n## Checkpoints and reproduction\n\n'
    text += f'Best DEV-frozen candidate: **{best["frozen_candidate"]["name"]}**, `{best["frozen_candidate"]["actor"]}`. Final evaluation has not run.\n\n'
    for arm, d in best["trained_arms"].items():
        text += f'- {arm}: `{d["actor"]}`, digest `{d["identity"]["actor_digest"]}`, update {d["update"]}; fresh saved-adapter reload max logit error {d["fresh_reload_max_logit_error"]:.8g}.\n'
    text += '\nRun `python -m ember_v1.rl_cf_run --config data/rl/r22-counterfactual-ppo@1.json --model-root <verified-original-V1> --phase2 <verified-complete-Phase2-evidence> --output-root <external-evidence> --attempt <unused-authorized-attempt>`. `--resume <paired-round-directory>` restores both actors/critics/Adam/RNG/curriculum counters at equal update counts. Frozen input identities and topology isolation fail closed. Run `python -m ember_v1.rl_cf_verify <attempt>` and `python experiments/rl_cf_report.py <attempt>` to regenerate the independent evidence and this report. Exact runtime/source identities, command, manifests and raw branches are preserved outside Git.\n'
    (root/"REPORT.md").write_text(text)
    bars = f'<p>{html.escape(verdict)}</p><p>Evacuated · Casualties · Unresolved</p>'
    for arm, s in summary["comparisons"].items():
        bars += f'<p>{arm}: {s["evacuated_rate"]:.1%} / {s["casualties_rate"]:.1%} / {s["unresolved_rate"]:.1%}</p><div class="bar">'
        for key, color in (("evacuated", "#276b50"), ("casualties", "#b04939"), ("unresolved", "#aaa89b")):
            bars += f'<span style="width:{100*s[key+"_rate"]}%;background:{color}"></span>'
        bars += '</div>'
    curves = ''
    for key in ("mean_reward", "evacuated_rate", "casualties_rate"):
        curves += line_plot([(a, [(u["update"], u["rollout"][key]) for u in h["updates"]]) for a, h in history.items()], key)
    for key in ("actor_loss", "critic_loss", "joint_kl", "person_entropy"):
        curves += line_plot([(a, [(u["update"], statistics.mean(m[key] for m in u["minibatches"])) for u in h["updates"]]) for a, h in history.items()], key)
    def page(content):
        return '<!doctype html><html><head><meta charset="utf-8"><style>body{margin:0;font:16px var(--font-sans,system-ui);color:var(--foreground)}.bar{height:24px;display:flex}svg{width:100%;height:210px}</style></head><body>'+content+'</body></html>'
    (root/"comparison.html").write_text(page(bars))
    (root/"curves.html").write_text(page(curves))
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("evidence", type=Path)
    report(parser.parse_args().evidence)
