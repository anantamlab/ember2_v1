# EMBER V1 Phase 2: online PPO results

Phase 2 produced a verified PPO checkpoint, but it did not improve evacuation over raw V1 on the independent reporting split. Both policies evacuated 174 of 358 people. PPO had three fewer casualties and three more unresolved people. The casualty interval still includes a regression just above the prespecified two percentage point tolerance.

| Policy | Evacuated | Casualties | Unresolved | Stay | Blocked / attempted |
|---|---:|---:|---:|---:|---:|
| Original V1 | 174 (48.60%) | 130 (36.31%) | 54 (15.08%) | 10.68% | 68.33% |
| Original V1 + movement mask | 202 (56.42%) | 99 (27.65%) | 57 (15.92%) | 24.58% | 51.97% |
| Original V1 + coordinator | 218 (60.89%) | 90 (25.14%) | 50 (13.97%) | 49.55% | 0.00% |
| Phase 1 PPO pilot | 168 (46.93%) | 129 (36.03%) | 61 (17.04%) | 12.01% | 68.55% |
| Phase 2 PPO, update 96 | 174 (48.60%) | 127 (35.47%) | 57 (15.92%) | 10.66% | 71.11% |

All arms used greedy decoding on the same 32 reporting geometry families and matched simulator seeds. Outcome percentages pool people; stay is per active-person decision and blocked is per attempted move. The mask and coordinator are external changes to V1 execution. The coordinator changed 2,640 actions; its gain does not belong to the learned policy. These fresh DEV maps are separate from Run 19 and EMBERBench.

## What the policy learned

The strongest supported explanation for the flat evacuation result is that the probability updates did not resolve the greedy policy's most common movement errors. On the same 32 initial public states, PPO changed 37 of 358 V1 argmax actions. Per-person entropy rose from 0.648 to 0.721 nats, and KL to V1 was 0.0666 nats per person. Probability mass on statically admissible moves was essentially flat (89.39% to 89.26%). Mass on admissible moves that shortened the public safe-exit distance fell from 64.33% to 62.94%. These are diagnostics on identical inputs; they do not establish the cause of downstream outcomes.

Over full reporting episodes, wall blocks rose from 1,932 to 2,149 and boundary blocks from 102 to 161. That increase outweighed declines in destination conflicts (199 to 177), occupied-cell blocks (1,572 to 1,483), and fire blocks (221 to 171); cycle blocks rose from 130 to 160. Repeated ineffective moves increased from 3,597 to 3,774 despite slightly fewer attempts. The repeated-move count requires the same ineffective non-stay action from the same position on consecutive steps. Stay rates were nearly identical, so stay collapse does not explain the result. Directions changed (east rose from 747 to 1,028; north fell from 1,795 to 1,647), but south remained the most frequent action. Aggregate direction counts do not distinguish preference from map or state exposure.

The gradients, likelihoods, and checkpoint audits show that the actor did update. This run cannot distinguish the causal effects of shared terminal credit, the sampled-to-greedy change in execution, and conservative update sizes. The higher pooled TRAIN evacuation in updates 113–128 (63.01% versus 55.34% in updates 1–16) used different maps and stochastic actions; meanwhile selection DEV utility declined after update 96. TRAIN outcomes alone do not establish generalization. One configuration and one training seed were run; no instability justified spending the remaining budget on a sweep.

## Selection and uncertainty

Every 16 updates, greedy evaluation used the fixed 32-family selection split (355 people). Eligible checkpoints had at most +2 percentage points casualties versus raw V1. Among eligible trained checkpoints, update 96 maximized pooled utility `(E - 2C - 0.75U)/N`: −0.417606 (161/131/63), versus −0.457746 (155/134/66) for V1. The final iterate, update 128, scored −0.519718 and had 146 casualties. Selection occurred before evaluating the selected actor on reporting DEV.

| Selection update | Evacuated | Casualties | Unresolved | Pooled utility |
|---|---:|---:|---:|---:|
| V1 | 155 | 134 | 66 | −0.457746 |
| 16 | 153 | 136 | 66 | −0.474648 |
| 32 | 152 | 137 | 66 | −0.483099 |
| 48 | 158 | 137 | 60 | −0.453521 |
| 64 | 159 | 136 | 60 | −0.445070 |
| 80 | 154 | 140 | 61 | −0.483803 |
| **96** | **161** | **131** | **63** | **−0.417606** |
| 112 | 154 | 135 | 66 | −0.466197 |
| 128 | 151 | 146 | 58 | −0.519718 |

On reporting DEV, paired family bootstrap intervals (5,000 replicates) for PPO minus V1 were:

| Outcome | Difference (pp) | 95% interval (pp) |
|---|---:|---:|
| Evacuated | 0.00 | [−2.89, +3.36] |
| Casualties | −0.84 | [−3.76, +2.06] |
| Unresolved | +0.84 | [−0.95, +2.56] |

The lower evacuation bound was below zero and the upper casualty bound exceeded the +2 pp tolerance. The sample supports neither an evacuation gain nor a confident casualty reduction. By comparison, the V1 coordinator's evacuation gain versus raw V1 was +12.29 pp [7.58, 17.63]; the mask's was +7.82 pp [3.49, 12.46]. These intervals reflect the 32 generated families at one simulator seed per family. They do not capture variation across training seeds, new fire realizations, or deployment distributions.

| Geometry stratum | People | V1 E/C/U counts | PPO E/C/U counts |
|---|---:|---:|---:|
| Corridor | 54 | 24/22/8 | 27/17/10 |
| Door | 43 | 30/11/2 | 28/12/3 |
| Inaccessible | 46 | 12/14/20 | 12/13/21 |
| Irregular | 39 | 15/22/2 | 16/22/1 |
| Open | 45 | 25/19/1 | 26/18/1 |
| Pillars | 43 | 22/11/10 | 21/11/11 |
| Rooms | 43 | 18/17/8 | 17/18/8 |
| Two doors | 45 | 28/14/3 | 27/16/2 |

Each stratum has four independent reporting families. Corridor gains were offset elsewhere; this small stratum breakdown is descriptive. [Analysis JSON](../data/rl/evidence/r21-online-ppo@1/analysis.json) includes all five arms by stratum, uncertainty, full direction and conflict counts, and latency. The [curves](../data/rl/evidence/r21-online-ppo@1/curves.html) show TRAIN outcomes, shaped return, losses, KL, and entropy for all 128 updates.

## Training and compute

Attempt 3 ran 1,024 fresh on-policy episodes: 38,950 simultaneous environment transitions, 198,735 person decisions, and 2,489 applied Adam steps. TRAIN outcome rates were 59.31% evacuated, 25.90% casualties, and 14.79% unresolved; stay was 8.06% and blocked attempts were 57.00%. The selected actor at update 96 had seen 768 episodes and 29,108 transitions, with 1,855 Phase 2 Adam steps. Phase 1 supplied 64 episodes, 1,946 transitions and 141 prior steps.

One RTX 5090 ran the experiment in 18,803.47 s; the training loop with periodic DEV and checkpoint writes accounted for 18,317.46 s. Attempts 1 and 2 added 15.81 s combined, leaving 2,780.73 s under the six-hour aggregate cap. Both failed before GPU load or rollouts: first a geometry capacity error, then a D4 topology collision. Their records were retained. The next attempt used fresh TRAIN geometry and no failed scenario was replayed. Attempt 4 was unused.

Collection ran at 13.20 transitions/s; the whole attempt delivered 2.07 TRAIN transitions/s including setup, training, saves and DEV. Across 9,258 telemetry samples GPU utilization averaged 37.91% (peak 88%); peak device memory was 16,564 MiB. Torch peak allocated/reserved memory was 11.307/14.951 GiB. A single untruncated public state per forward kept memory within budget. Simulation workers received no CUDA tensors; the run dispatched one joint state at a time. Synchronized batch-one forward and action selection p50/p95 were 54.75/65.97 ms for PPO and 56.08/68.55 ms for raw V1. Encoding, simulator time and external postprocessing are excluded.

The original trained Qwen3.5-2B V1 remained frozen as a baseline. Training resumed the Phase 1 adapter, float32 Clef head, critic and Adam moments. The public observation and simultaneous joint-action contract stayed intact. The unmasked actor sampled a categorical action for each active person, summed their log probabilities and used one clipped ratio per joint transition. The centralized public-state critic estimated the shared return. The terminal reward was `(E - 2C - 0.75U)/N`, with gamma 1, GAE lambda 0.97 and public distance potential shaping at eta 0.1 with terminal potential zero. The objective does not reward moving or penalize staying by itself.

The [frozen configuration](../data/rl/r21-online-ppo@1.json) used two PPO epochs, clip 0.2, joint KL early stopping at 0.04, entropy coefficient 0.005, and learning rates of 3e-6 (LoRA), 1e-5 (head) and 3e-4 (critic). There was no reference KL penalty. The mean episode return was −0.037441 shaped and −0.045739 terminal. Logged minibatch means were actor loss −0.022509, critic MSE 0.076444, joint KL 0.007215, joint entropy 3.514280 nats and per-person entropy 0.609239 nats. The ratio was outside the clip range in 10.17% of logged state evaluations; 12 updates stopped early on KL.

The independent generator varied eight structural strata, 4–20 people, exits and bottlenecks, and deterministic and stochastic fire. All 1,088 Phase 2 geometry hashes were distinct under the D4 topology check and disjoint from Phase 1. No SFT ran; the expansion, V1.1 artifacts, teachers, and the benchmark were excluded from this experiment. Reporting DEV did not select weights or configurations.

## Verification and checkpoint identity

- All 2,489 applied steps passed finite, nonzero LoRA, Clef head and critic gradient checks. The maximum pre-update likelihood replay discrepancy was zero across all 128 updates.
- Independent CPU replay reconstructed 1,472 complete episodes and 56,775 transitions. It recomputed all 38,950 TRAIN advantages and returns with zero target discrepancy. Joint log-probability sum error was zero; the largest categorical normalization error was 1.935e-7.
- The fresh-load GPU audit restored the actual Phase 1 parent, replayed all 369 first-rollout transitions through the differentiable path with zero likelihood discrepancy, and obtained finite nonzero LoRA/head gradients. The audit also ran three explicit Torch checks, all passing.
- Update 96 reloaded to the same actor digest. Its Adam state advanced from step 141 to 1,996, with critic and RNG states present. Relative to the Phase 1 parent, the selected adapter/head tensor L2 changes were 0.609299 and 2.330547. The last iterate changed 494 trainable tensors and had a relative L2 change of 0.015476.
- CPU replay, GPU audit and archive supervisors each exited 0. `FINAL_AUDIT_EXIT_CODES.json` binds the first two actual exit markers into the manifest; the archive's separate exit marker was read after completion. The Windows copy was independently rehashed: 42 files, 821,844,776 bytes, including the selected actor, trainer and identity.

| Identity | Value |
|---|---|
| Run / selected checkpoint | `r21-online-ppo@1/attempt-3/checkpoints/update-0096` |
| Selected actor digest | `sha256:901e22e8be0af3f4ff7002b9041d16e4a0dacd8c602a79b66121429507275120` |
| Selected LoRA SHA-256 | `7f5fcba8a78a3ad1ad4617e988bbb0775c6db0c25fd618e8ff0cb94b028413c2` |
| Selected Clef head SHA-256 | `4bf0cd37d1dbc3696b785b004183bca057f6be308a975ea1797a4e451601442e` |
| Selected trainer SHA-256 | `874c4741d10ec17c9610f3677cf5947f6c3415bca952051f7e243ce322ef83e5` |
| Training source | `90d8e672198733768d38f2960e9d7cb0c4e88af2` |
| Configuration SHA-256 | `f670ecf6aeab7a2ad1d10ae5e3e73ca7ab057b090001eb31c6a2d14788c632c6` |
| Original V1 digest | `sha256:c31cae7011dbee64304d091bd744f39ab5a90be35df72a7206477748b98822d3` |
| Phase 1 parent digest | `sha256:f6b56137acc791bd1ed0862547a1dd3ca2ef81f336af6c969947840b46c66374` |
| Canonical evidence manifest SHA-256 | `04c3b6f35f54f968da142221947882fbe9722974c79e987f9c4dfd8de322d452` |
| Windows copy verification SHA-256 | `9d8c06b8bad5daad48cec5d7bc0516a0e654f45e198a50e81bf35d04b868ca87` |

Canonical evidence root: `/home/darshith/ember-ws/rl-evidence/r21-online-ppo@1/attempt-3`. Within it, `checkpoints/update-0096/actor` contains the adapter, head and tokenizer; `checkpoints/update-0096/trainer.pt` contains the resumable training state. The manifest covers 1,565 files and 76,342,664,199 bytes. The verified Windows mirror is `C:/Users/Admin/Documents/ember-evidence/r21-online-ppo@1/attempt-3`, with the same selected files under `selected_checkpoint/`. The [checkpoint record](../data/rl/evidence/r21-online-ppo@1/best_checkpoint.json) lists every selected file hash.

## Reproduction

The [Phase 2 plan](R21_PHASE2_PLAN.md), [execution record](../data/rl/evidence/r21-online-ppo@1/execution.json), and archived `reproduction_ops/` specify the command, source and inputs. The run used the pinned WSL Python 3.11 environment `/home/darshith/ember-ws/venvs/9613f929b53c0fed/bin/python`, offline cached `unsloth/Qwen3.5-2B@fffbc0d8711cf87df6d1f97938c5e50ef94aadc7`, Torch 2.8.0+cu128, Transformers 5.2.0, Unsloth 2026.10.11, and Triton 3.4.0. It verified all 14 original V1 artifact files and 16 base files before and after training.

For a separately authorized, unused attempt only, the runner interface is `python -m ember_v1.rl_online --config data/rl/r21-online-ppo@1.json --model-root <verified-original-V1> --phase1 <verified-Phase1-run> --output-root <bound-evidence-root> --attempt <unused-attempt>`. The runner refuses an occupied attempt or a completed experiment. `--resume <verified-checkpoint-dir>` restores actor, critic, Adam state, RNG and counters; retries use new TRAIN families. Do not use this command to repeat attempt 3.

To reproduce the analysis without training, use `python -m ember_v1.rl_verify <attempt-copy>` and `python experiments/rl_online_report.py <attempt-copy>` from the repository with `src` on `PYTHONPATH`. Use a scratch copy for generated outputs so the canonical manifest stays intact. The raw transition, episode, PPO target, scenario and update logs are in the external archive. The separately preserved attempt 1 and 2 failures are included in the compact evidence.

Phase 3 remains on hold. No final evaluation was run.
