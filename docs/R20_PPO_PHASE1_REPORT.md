# EMBER V1 Phase 1: first PPO pilot

The existing V1 model completed genuine PPO training: 64 on-policy episodes, 1946 environment steps and 141 optimizer steps. Raw argmax evacuation matched on the fixed fresh DEV set: 55.81% for frozen V1 and 55.81% for the final RL actor. Casualties rose from 37.98% to 40.31%, while unresolved cases fell from 6.20% to 3.88%. The pilot demonstrates a working RL update and exact checkpoint persistence, with no demonstrated improvement in evacuation quality.

## Checkpoints and provenance

- Frozen parent: `sankalphs/ember-v1-models`, artifact `v1/p1-18-v1-train@1`, manifest revision `1fc405f723718e308c8a22b137fee4a025086fde`.
- Parent artifact manifest SHA256: `fb7573a93f9630b8486eeaa6d3e2bfce58ad1e2e8e7db74ff3f39651949d0952`. All 14 checkpoint payloads were content-hashed before training and again after evaluation.
- Pinned base: `unsloth/Qwen3.5-2B@fffbc0d8711cf87df6d1f97938c5e50ef94aadc7`; all 16 base files verified against V1 identity. The LoRA adapter, float32 Clef head, tokenizer and inference configuration were loaded from the real trained V1 artifact. The base was never used as a replacement actor.
- Experiment: `r20-ppo-pilot@1/attempt-6`, execution commit `bb8c74034dc755ab8962605f2606fe0b3ef27d7f`. The original checkpoint remains unchanged and supplies the frozen baseline. No SFT optimizer or teacher label collection was executed.
- RL actor: `/home/darshith/ember-ws/rl-evidence/r20-ppo-pilot@1/attempt-6/checkpoint`.
- Windows evidence copy: `C:/Users/Admin/Documents/ember-evidence/r20-ppo-pilot@1/attempt-6/`.
- The RL checkpoint is local. No new Hub revision was created.
- RL adapter SHA256: `172f09e115853b972bc77cbd016b71b06b488e39ea7bb20e891680488f11c62c`.
- RL Clef head SHA256: `2fa988fad107c4386f459d469f94e622be1062557113204288f0bcc894606d74`.
- Reloaded V1 parameter digest: `sha256:c31cae7011dbee64304d091bd744f39ab5a90be35df72a7206477748b98822d3`. RL digest: `sha256:f6b56137acc791bd1ed0862547a1dd3ca2ef81f336af6c969947840b46c66374`. These digests use sorted LoRA/head parameter names and float32 parameter bytes.

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

Training mean undiscounted shaped episode return: -0.024053; mean terminal reward: -0.033476. Optimizer minibatch means (including any final KL-stop diagnostic block): actor loss -0.020483, critic MSE 0.095956, joint KL 0.008051, joint entropy 1.340279 nats, per-person entropy 0.349433 nats. These means are unweighted across logged minibatches; the complete logs preserve each loss. Fresh scenarios differ across updates, so the reward rows alone are not a controlled learning curve.

| Rollout update | Environment steps | Episode return | Actor loss | Critic MSE | Joint KL | Joint entropy | Optimizer steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 156 | -0.17677 | -0.08267 | 0.17018 | 0.010209 | 1.70860 | 14 |
| 2 | 251 | -0.08147 | 0.04233 | 0.12258 | 0.014870 | 1.57018 | 20 |
| 3 | 232 | -0.16479 | -0.01987 | 0.14980 | 0.010798 | 1.03240 | 20 |
| 4 | 273 | -0.06343 | 0.02211 | 0.06057 | 0.003585 | 1.45699 | 20 |
| 5 | 259 | -0.03105 | 0.05804 | 0.17218 | 0.008434 | 1.57631 | 11 |
| 6 | 232 | -0.07329 | -0.12135 | 0.04521 | 0.004251 | 1.14697 | 16 |
| 7 | 279 | 0.39338 | -0.01244 | 0.04324 | 0.006763 | 1.30045 | 20 |
| 8 | 264 | 0.00500 | -0.05837 | 0.05462 | 0.005726 | 1.09445 | 20 |

## Fixed DEV comparison

Each arm uses the same 16 DEV scenarios and fire seeds. Rates pool people across scenarios; stay rates use active person decisions, and blocked rates use attempted non-stay moves. Greedy comparison is primary. Stochastic comparison uses one fixed sampling seed per policy, with the same initial RNG seed; divergent episode lengths change later random draw consumption.

| Actor | Evacuated | Casualties | Unresolved | Stay share | Blocked / attempted |
|---|---:|---:|---:|---:|---:|
| Frozen V1, raw argmax | 72/129 (55.81%) | 49/129 (37.98%) | 8/129 (6.20%) | 7.38% | 925/1519 (60.90%) |
| RL pilot, raw argmax | 72/129 (55.81%) | 52/129 (40.31%) | 5/129 (3.88%) | 3.30% | 975/1583 (61.59%) |
| Frozen V1, stochastic | 85/129 (65.89%) | 36/129 (27.91%) | 8/129 (6.20%) | 8.27% | 696/1475 (47.19%) |
| RL pilot, stochastic | 82/129 (63.57%) | 42/129 (32.56%) | 5/129 (3.88%) | 6.65% | 702/1389 (50.54%) |

For RL-minus-V1 greedy rate differences, 2,000 matched scenario/family bootstrap resamples (seed 20261112) give 95% intervals: evacuation [-2.13%, 2.36%], casualty [0.00%, 5.11%], unresolved [-5.38%, 0.00%]. Sixteen families and one training seed limit the strength of any performance conclusion.

Blocked-movement counts are retained by cause in `report_values.json` and every episode: boundary, wall, fire, occupied cell, conflict and cycle. No controller rewrites actions to eliminate these counts. TRAIN stay share is 9.31%; blocked movement is 3386/6208 attempts (54.54%).

## Verification and cost

- Focused correctness checks: nine CPU environment/evidence tests and three explicit Torch joint-probability/PPO tests pass. The complete CPU-suite receipt for the final review head is recorded in [PR #53](https://github.com/anantamlab/ember_v1/pull/53), with its full local output retained. Existing expected failures and skips are reported there without weakening tests.
- Independent CPU replay: 128 complete episodes and 3791 transitions reproduce every simulator record, public observation and reward. All sampled and executed action maps match. It recomputes 1946 training advantages/returns from stored rewards and state values.
- Joint log-probability sum error: 0; maximum categorical normalization error: 1.66e-07.
- Independent autograd-enabled replay of 103 original first-rollout batches / 156 transitions: maximum behavior probability discrepancy 0. A separate real backward on an actual simulator observation confirms finite nonzero LoRA/head gradients. The final source also performs its pre-update likelihood audit with autograd enabled.
- Applied optimizer steps have finite nonzero LoRA, head and critic gradients. Minimum group norms: {'lora': 0.034018784156425924, 'head': 0.01314983458368845, 'critic': 0.06365497354414024}; maximum group norms: {'lora': 36.32176014659993, 'head': 5.369324896849876, 'critic': 1.0512911333237982}.
- Parameter change: L2 0.5924163, relative L2 0.003283785; 494 tensors changed. LoRA L2 0.1583149, Clef-head L2 0.5708708. Fixed-observation maximum logit change: 0.5195236.
- Exact-save preflight passes before training: frozen V1 is saved and reloaded with equal parameter digest and zero logit discrepancy. The final adapter/head are loaded through the real V1 loader. Reloaded parameter digest matches exactly; maximum final logit reload discrepancy is 0. The critic, optimizer and Torch CPU/CUDA RNG states are saved in `trainer.pt`.
- Training wall time: 521.94 s; training throughput including optimization 3.728 environment steps/s; collection-only throughput 20.952 steps/s. End-to-end attempt time including loading, evaluation and reload: 631.21 s.
- RTX 5090 telemetry: mean utilization 45.22%, peak 90%, peak whole-GPU memory 17919 MiB; mean power 246.77 W, peak 380.52 W. Sampling covers the whole attempt, including loading and verification. Torch peak allocated memory 15.199 GiB; peak reserved 16.717 GiB. The desktop also uses GPU memory.

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
python -m ember_v1.rl_run \
  --config data/rl/r20-ppo-pilot@1.json \
  --model-root /home/darshith/ember-ws/artifacts/models/1fc405f723718e308c8a22b137fee4a025086fde/v1/p1-18-v1-train@1 \
  --output-root /path/to/authorized/fresh-evidence-root --attempt 1
python -m ember_v1.rl_verify /path/to/evidence/run/attempt
python -m pytest -q experiments/rl_gpu_checks.py
python experiments/rl_gpu_checks.py --model-root /path/to/original-v1-artifact --evidence /path/to/evidence/run/attempt
python experiments/rl_report.py --evidence /path/to/evidence/run/attempt
```

The first GPU attempt used execution commit `0563a87284ec` and the initial plan under `data/runs`; the completed attempt uses the amended memory-bounded plan under `data/rl`, which respects Run 19's distinct config schema. The supplemental autograd audit verifies the completed attempt's original behavior batches. Output directories are write-once and attempt guards live outside checkouts. Failed and completed attempts are retained. `trainer.pt` contains restart state, and every second update has an intermediate adapter/head plus optimizer/critic/RNG snapshot; automatic mid-run resume is not implemented. The runner refuses to overwrite an attempt. Any new training experiment needs its own authorized plan/output identity. The checkpoint is a LoRA-plus-Clef actor on the pinned base, with the centralized critic stored separately.

No Phase 2 work was started.
