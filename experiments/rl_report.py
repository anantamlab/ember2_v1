"""Regenerate the Phase-1 tables, matched bootstrap and report from audited evidence."""
import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path

parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--evidence', type=Path, required=True)
parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
args = parser.parse_args()
root, repo = args.evidence.resolve(), args.repo.resolve()
s = json.loads((root/'summary.json').read_text())
identity = json.loads((root/'checkpoint_identity.json').read_text())
updates = [json.loads(x) for x in (root/'updates.jsonl').read_text().splitlines()]
episodes = [json.loads(x) for x in (root/'episodes.jsonl').read_text().splitlines()]
replay = json.loads((root/'independent_replay.json').read_text())
autograd = json.loads((root/'autograd_probability_audit.json').read_text())
cfg = json.loads((root/'execution.json').read_text())['config']
gpu = [json.loads(x) for x in (root/'gpu.jsonl').read_text().splitlines()]
gpu_values = [[float(v.strip()) for v in r['sample'].split(',')[1:]] for r in gpu if r['exit']==0 and len(r['sample'].split(','))==4]
metrics = [m for u in updates for m in u['minibatches']]
means = {k: statistics.mean(m[k] for m in metrics) for k in ('actor_loss','critic_loss','joint_kl','joint_entropy','person_entropy','clip_fraction')}
gradients = [m['gradient_norms'] for m in metrics if 'gradient_norms' in m]
means['gradient_min'] = {k: min(g[k] for g in gradients) for k in ('lora','head','critic')}
means['gradient_max'] = {k: max(g[k] for g in gradients) for k in ('lora','head','critic')}
gpu_summary = {'samples': len(gpu_values), 'mean_utilization_pct': statistics.mean(v[0] for v in gpu_values),
               'peak_utilization_pct': max(v[0] for v in gpu_values), 'peak_memory_MiB': max(v[1] for v in gpu_values),
               'mean_power_W': statistics.mean(v[2] for v in gpu_values), 'peak_power_W': max(v[2] for v in gpu_values)}
blocked = {}
for label in ('v1-greedy','rl-greedy','v1-stochastic','rl-stochastic'):
    blocked[label] = {k: sum(e.get(k,0) for e in episodes if e['label']==label) for k in (
        'blocked_boundary','blocked_wall','blocked_fire','blocked_occupied','blocked_conflict','blocked_cycle')}
baseline = {e['scenario_id']: e for e in episodes if e['label']=='v1-greedy'}
rl = {e['scenario_id']: e for e in episodes if e['label']=='rl-greedy'}
assert set(baseline)==set(rl) and len(baseline)==16
rng = random.Random(20261112)
deltas = {k: [] for k in ('evacuated','casualties','unresolved')}
keys = list(baseline)
for _ in range(2000):
    sample = rng.choices(keys,k=len(keys))
    population = sum(baseline[k]['people'] for k in sample)
    for metric in deltas:
        deltas[metric].append(sum(rl[k][metric]-baseline[k][metric] for k in sample)/population)
ci = {k: [sorted(v)[49],sorted(v)[1949]] for k,v in deltas.items()}
report_values = {'summary': s, 'optimizer_minibatch_means': means, 'gpu': gpu_summary,
                 'blocked_types': blocked, 'greedy_rate_difference_95pct_bootstrap': ci,
                 'independent_replay': replay, 'autograd_audit': autograd}
(root/'report_values.json').write_text(json.dumps(report_values,indent=2)+'\n')
out = repo/'data/rl/evidence/r20-ppo-pilot@1'
out.mkdir(parents=True,exist_ok=True)
for name in ('summary.json','checkpoint_identity.json','save_preflight.json','independent_replay.json','autograd_probability_audit.json','report_values.json'):
    (out/name).write_bytes((root/name).read_bytes())
def pct(x): return f'{100*x:.2f}%'
lines = []
for label,key in [('Frozen V1, raw argmax','baseline_greedy'),('RL pilot, raw argmax','rl_greedy'),
                  ('Frozen V1, stochastic','baseline_stochastic'),('RL pilot, stochastic','rl_stochastic')]:
    d=s[key]
    lines.append(f"| {label} | {d['evacuated']}/{d['people']} ({pct(d['evacuated_rate'])}) | {d['casualties']}/{d['people']} ({pct(d['casualties_rate'])}) | {d['unresolved']}/{d['people']} ({pct(d['unresolved_rate'])}) | {pct(d['stay_rate'])} | {d['blocked']}/{d['attempted']} ({pct(d['blocked_per_attempt'])}) |")
update_lines=[]
for u in updates:
    ms=u['minibatches']
    avg=lambda k: statistics.mean(m[k] for m in ms)
    update_lines.append(f"| {u['update']+1} | {u['rollout']['steps']} | {u['rollout']['mean_reward']:.5f} | {avg('actor_loss'):.5f} | {avg('critic_loss'):.5f} | {avg('joint_kl'):.6f} | {avg('joint_entropy'):.5f} | {u['optimizer_steps']} |")
weights={f['path']: f['sha256'] for f in identity['files']}
delta=s['parameter_delta']
result='improved' if s['rl_greedy']['evacuated_rate']>s['baseline_greedy']['evacuated_rate'] else ('matched' if s['rl_greedy']['evacuated_rate']==s['baseline_greedy']['evacuated_rate'] else 'regressed')
text=f'''# EMBER V1 Phase 1: first PPO pilot

The existing V1 model completed genuine PPO training: {s['training']['episodes']} on-policy episodes, {s['training']['steps']} environment steps and {s['optimizer_steps']} optimizer steps. Raw argmax evacuation {result} on the fixed fresh DEV set: {pct(s['baseline_greedy']['evacuated_rate'])} for frozen V1 and {pct(s['rl_greedy']['evacuated_rate'])} for the final RL actor. Casualties rose from {pct(s['baseline_greedy']['casualties_rate'])} to {pct(s['rl_greedy']['casualties_rate'])}, while unresolved cases fell from {pct(s['baseline_greedy']['unresolved_rate'])} to {pct(s['rl_greedy']['unresolved_rate'])}. The pilot demonstrates a working RL update and exact checkpoint persistence, with no demonstrated improvement in evacuation quality.

## Checkpoints and provenance

- Frozen parent: `sankalphs/ember-v1-models`, artifact `v1/p1-18-v1-train@1`, manifest revision `1fc405f723718e308c8a22b137fee4a025086fde`.
- Parent artifact manifest SHA256: `fb7573a93f9630b8486eeaa6d3e2bfce58ad1e2e8e7db74ff3f39651949d0952`. All 14 checkpoint payloads were content-hashed before training and again after evaluation.
- Pinned base: `unsloth/Qwen3.5-2B@fffbc0d8711cf87df6d1f97938c5e50ef94aadc7`; all 16 base files verified against V1 identity. The LoRA adapter, float32 Clef head, tokenizer and inference configuration were loaded from the real trained V1 artifact. The base was never used as a replacement actor.
- Experiment: `r20-ppo-pilot@1/attempt-6`, execution commit `{s['code_commit']}`. The original checkpoint remains unchanged and supplies the frozen baseline. No SFT optimizer or teacher label collection was executed.
- RL actor: `{root / 'checkpoint'}`.
- Windows evidence copy: `C:/Users/Admin/Documents/ember-evidence/r20-ppo-pilot@1/attempt-6/`.
- The RL checkpoint is local. No new Hub revision was created.
- RL adapter SHA256: `{weights['adapter_model.safetensors']}`.
- RL Clef head SHA256: `{weights['joint_head.safetensors']}`.
- Reloaded V1 parameter digest: `{s['frozen_v1_digest']}`. RL digest: `{s['rl_digest']}`. These digests use sorted LoRA/head parameter names and float32 parameter bytes.

Run 19 did not reproduce V1 bit-for-bit across A10 and Blackwell: 1 of 1,538 decisions exceeded its frozen near-tie threshold. This limitation is retained. All comparisons here use this workstation and the same inference batching. The report does not equate these fresh small-map DEV rates with Run 19's historical rates.

## Environment, data and policy

`rl_env.EvacuationEnv` wraps the repository's existing independent CPU simulator. `reset()` returns its exact public observation; `step()` submits one complete simultaneous map, unchanged. Movement blocking is an environment outcome. `rollout()` yields complete episodes. A step-cap ending is an MDP terminal with unresolved people, not a rollout truncation requiring critic bootstrap.

The new generator reads no historical scenarios, labels or outcomes. It creates 64 distinct TRAIN families and 16 separately seeded DEV families under `ember-r20-independent-rl-20261112-v1`. Rooms are 8-12 cells per side, with 4-12 people, 1-3 boundary exits, random obstacles and optional dividing wall/door, and stochastic fire with intervals 2-4 and probabilities 0.25/0.45/0.65. Each horizon is 40 steps. Initial placement uses public static reachability, without outcome-based selection. Every map and split was serialized before the first baseline evaluation. A symmetry-invariant topology audit found zero duplicate families or TRAIN/DEV overlap.

No `p1-18-expand@1`, `p1-18-dataset-v2`, 1,200-scenario expansion artifact, EMBERBench hidden scenario or internal benchmark simulator was an experiment input. The fresh DEV set is used only for the initial/final comparison. The final iterate is selected by the fixed update count; DEV outcomes do not choose checkpoints or hyperparameters. This pilot does not access a locked test set.

The actor retains V1's ASCII observation encoding, compact questions, sorted person IDs, chunk size 100, no reservations, bf16 encoder and original float32 Clef head. Stochastic actions use the five categorical Clef options at temperature 1. There is no action mask, coordinator or action repair. All original base weights stay frozen; the original trained LoRA and decision head receive PPO updates. Simulator state is isolated from the model except for the public observation. Dummy gold fields are used only to construct the existing Clef input schema; their targets are discarded, and no supervised cross-entropy loss is computed.

For a simultaneous joint action A, `log pi(A|s)` is the sum of selected person log-probabilities across all chunks. PPO exponentiates the difference of these joint sums and clips once per environment transition. It does not average person log-probabilities before the ratio or apply a token PPO loss. The exact categorical joint entropy and old-to-current joint KL sum across people. Rollout inference batches are replayed with the same membership and padding.

The centralized critic is a fresh 1548→128→64→1 MLP with tanh activations. Its six 16×16 spatial channels contain public obstacles, exits, fire and person statuses, with 12 public scalar features. The actor receives the full exact observation. The critic receives no seed, simulator-private trapped set, future fire or sampled actions.

## Reward and optimization

Terminal reward is `(E - 2C - 0.75U)/N`. Optional shaping is `eta*(gamma*Phi(next)-Phi(current))`, with `eta=0.1`, `gamma=0.99`, and zero terminal potential. Phi is negative public safe-exit BFS distance, capped at map area for unreachable active people and normalized by map area and initial population. Discounted shaping telescopes; there is no direct movement bonus or universal stay penalty.

GAE uses lambda 0.95, complete episodes and terminal bootstrap zero. Advantages are normalized once over joint transitions in each rollout batch. Each of eight updates collects eight new episodes; up to two environments run concurrently in CPU worker processes, with batched GPU inference. PPO uses two epochs, joint ratio clipping 0.2, target joint KL 0.04, joint entropy coefficient 0.001, value coefficient 0.5, and gradient norm clipping 1. AdamW learning rates are 3e-6 for LoRA, 1e-5 for the Clef head and 3e-4 for the critic, with zero weight decay. Sixteen inference groups are accumulated per optimizer step. A KL breach stops before a further optimizer update. The plan caps total compute including retries at three hours.

Training mean undiscounted shaped episode return: {s['training']['mean_reward']:.6f}; mean terminal reward: {s['training']['mean_terminal_reward']:.6f}. Optimizer minibatch means (including any final KL-stop diagnostic block): actor loss {means['actor_loss']:.6f}, critic MSE {means['critic_loss']:.6f}, joint KL {means['joint_kl']:.6f}, joint entropy {means['joint_entropy']:.6f} nats, per-person entropy {means['person_entropy']:.6f} nats. These means are unweighted across logged minibatches; the complete logs preserve each loss. Fresh scenarios differ across updates, so the reward rows alone are not a controlled learning curve.

| Rollout update | Environment steps | Episode return | Actor loss | Critic MSE | Joint KL | Joint entropy | Optimizer steps |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(update_lines)}

## Fixed DEV comparison

Each arm uses the same 16 DEV scenarios and fire seeds. Rates pool people across scenarios; stay rates use active person decisions, and blocked rates use attempted non-stay moves. Greedy comparison is primary. Stochastic comparison uses one fixed sampling seed per policy, with the same initial RNG seed; divergent episode lengths change later random draw consumption.

| Actor | Evacuated | Casualties | Unresolved | Stay share | Blocked / attempted |
|---|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

For RL-minus-V1 greedy rate differences, 2,000 matched scenario/family bootstrap resamples (seed 20261112) give 95% intervals: evacuation [{pct(ci['evacuated'][0])}, {pct(ci['evacuated'][1])}], casualty [{pct(ci['casualties'][0])}, {pct(ci['casualties'][1])}], unresolved [{pct(ci['unresolved'][0])}, {pct(ci['unresolved'][1])}]. Sixteen families and one training seed limit the strength of any performance conclusion.

Blocked-movement counts are retained by cause in `report_values.json` and every episode: boundary, wall, fire, occupied cell, conflict and cycle. No controller rewrites actions to eliminate these counts. TRAIN stay share is {pct(s['training']['stay_rate'])}; blocked movement is {s['training']['blocked']}/{s['training']['attempted']} attempts ({pct(s['training']['blocked_per_attempt'])}).

## Verification and cost

- Focused correctness checks: nine CPU environment/evidence tests and three explicit Torch joint-probability/PPO tests pass. The complete CPU-suite receipt for the final review head is recorded in [PR #53](https://github.com/anantamlab/ember_v1/pull/53), with its full local output retained. Existing expected failures and skips are reported there without weakening tests.
- Independent CPU replay: {replay['episodes_replayed']} complete episodes and {replay['transitions_replayed']} transitions reproduce every simulator record, public observation and reward. All sampled and executed action maps match. It recomputes {replay['training_advantages_recomputed']} training advantages/returns from stored rewards and state values.
- Joint log-probability sum error: {replay['max_joint_sum_error']:.3g}; maximum categorical normalization error: {replay['max_distribution_normalization_error']:.3g}.
- Independent autograd-enabled replay of {autograd['groups']} original first-rollout batches / {autograd['transitions']} transitions: maximum behavior probability discrepancy {autograd['autograd_behavior_replay_max_error']:.3g}. A separate real backward on an actual simulator observation confirms finite nonzero LoRA/head gradients. The final source also performs its pre-update likelihood audit with autograd enabled.
- Applied optimizer steps have finite nonzero LoRA, head and critic gradients. Minimum group norms: {means['gradient_min']}; maximum group norms: {means['gradient_max']}.
- Parameter change: L2 {delta['l2']:.7f}, relative L2 {delta['relative_l2']:.7g}; {delta['changed_tensors']} tensors changed. LoRA L2 {delta['group_l2']['lora']:.7f}, Clef-head L2 {delta['group_l2']['head']:.7f}. Fixed-observation maximum logit change: {s['probe_max_logit_change']:.7f}.
- Exact-save preflight passes before training: frozen V1 is saved and reloaded with equal parameter digest and zero logit discrepancy. The final adapter/head are loaded through the real V1 loader. Reloaded parameter digest matches exactly; maximum final logit reload discrepancy is {s['reload_error']:.3g}. The critic, optimizer and Torch CPU/CUDA RNG states are saved in `trainer.pt`.
- Training wall time: {s['training_seconds']:.2f} s; training throughput including optimization {s['environment_steps_per_second']:.3f} environment steps/s; collection-only throughput {s['collection_steps_per_second']:.3f} steps/s. End-to-end attempt time including loading, evaluation and reload: {s['elapsed_seconds']:.2f} s.
- RTX 5090 telemetry: mean utilization {gpu_summary['mean_utilization_pct']:.2f}%, peak {gpu_summary['peak_utilization_pct']:.0f}%, peak whole-GPU memory {gpu_summary['peak_memory_MiB']:.0f} MiB; mean power {gpu_summary['mean_power_W']:.2f} W, peak {gpu_summary['peak_power_W']:.2f} W. Sampling covers the whole attempt, including loading and verification. Torch peak allocated memory {s['torch_peak_allocated_bytes']/2**30:.3f} GiB; peak reserved {s['torch_peak_reserved_bytes']/2**30:.3f} GiB. The desktop also uses GPU memory.

Two attempts stopped before GPU model loading: attempt 1 caught a Windows/WSL line-ending interpretation mismatch, fixed with checkout `core.autocrlf=true`; attempt 2 caught a wrong cached-base verifier helper name. Attempt 3 completed three rollout updates and 48 optimizer steps, then saturated WDDM GPU memory near 32,105 MiB and slowed sharply during the fourth update. It was interrupted, its partial traces/metrics retained, and the bounded plan amended to use two environments per GPU batch with sixteen accumulated groups and intermediate checkpoints every two updates.

Attempt 4's stronger autograd-enabled probability audit caught a fundamental numerical mismatch: bf16 no-grad behavior forwards and autograd-enabled PPO forwards differed by up to 0.002176 in the joint log-probability, despite unchanged weights and identical batches. It stopped before any optimizer update. The fix collects TRAIN actions using the same autograd-enabled forward as PPO, then detaches stored rollout quantities. Attempt 3 used the earlier no-grad collection path, so its updates are excluded from valid PPO evidence.

Attempt 5 completed all 64 episodes, 1,877 steps and 140 optimizer steps with exact on-policy probabilities, then failed the final reload check by 0.015649 in logits. Inspection of the pinned upstream saver found that public `save_pretrained` converts non-scalar Clef head weights to bf16. It silently ignores an `exact=True` keyword. Its separate resumable-checkpoint helper `_save_clef_adapter(..., exact=True)` preserves float32 head weights and does not fold temperature. Two independent loads of the rounded artifact agreed with each other, but disagreed with the evaluated in-memory actor, confirming a save-path error. The attempt's rounded artifacts and evaluation traces are retained and excluded from the final checkpoint claim.

Attempt 6 restarts from frozen V1, uses exact float32 saves with a pre-training roundtrip check, and collects new on-policy trajectories. No previous attempt's trajectories enter its PPO optimizer. Reward, learning rates, fresh TRAIN scenario set and final update count were unchanged across numerical/serialization fixes. Updated actors are evaluated on DEV only after finishing the fixed training schedule; DEV scores do not tune those fixes or select weights. All failures remain in the evidence root.

An initial CPU-suite invocation in the GPU environment was interrupted after identifying the existing test assumption that Unsloth is absent and the new standalone config's incorrect placement under Run 19's config directory. The PPO config now lives under `data/rl`, Torch-specific checks run explicitly in the GPU environment, and the complete CPU suite runs in a separate Python 3.11 environment. No existing test was weakened or skipped to resolve these issues.

The pinned Unsloth stack disables its unsupported xformers FA3 path on sm_120 and uses its available attention implementation. The Qwen recurrent-attention fallback emits the same optional-kernel warning as Run 19. No GPU dependency was silently upgraded. The CPU verification environment adds only the repository's test dependencies.

## Reproduce and resume

Use the committed plan `data/rl/r20-ppo-pilot@1.json`, Python 3.11, the Run 19 hash-locked GPU environment, the verified local V1 model root and pinned base cache. Keep all evidence outside the checkout. On this workstation:

```bash
export HF_HOME=/home/darshith/ember-ws/hf-home
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$PWD/src
python -m ember_v1.rl_run \\
  --config data/rl/r20-ppo-pilot@1.json \\
  --model-root /home/darshith/ember-ws/artifacts/models/1fc405f723718e308c8a22b137fee4a025086fde/v1/p1-18-v1-train@1 \\
  --output-root /path/to/authorized/fresh-evidence-root --attempt 1
python -m ember_v1.rl_verify /path/to/evidence/run/attempt
python -m pytest -q experiments/rl_gpu_checks.py
python experiments/rl_gpu_checks.py --model-root /path/to/original-v1-artifact --evidence /path/to/evidence/run/attempt
python experiments/rl_report.py --evidence /path/to/evidence/run/attempt
```

The first GPU attempt used execution commit `0563a87284ec` and the initial plan under `data/runs`; the completed attempt uses the amended memory-bounded plan under `data/rl`, which respects Run 19's distinct config schema. The supplemental autograd audit verifies the completed attempt's original behavior batches. Output directories are write-once and attempt guards live outside checkouts. Failed and completed attempts are retained. `trainer.pt` contains restart state, and every second update has an intermediate adapter/head plus optimizer/critic/RNG snapshot; automatic mid-run resume is not implemented. The runner refuses to overwrite an attempt. Any new training experiment needs its own authorized plan/output identity. The checkpoint is a LoRA-plus-Clef actor on the pinned base, with the centralized critic stored separately.

No Phase 2 work was started.
'''
(repo/'docs/R20_PPO_PHASE1_REPORT.md').write_text(text)
print(json.dumps({'result': result,'summary':s,'means':means,'gpu':gpu_summary,'ci':ci},indent=2))
