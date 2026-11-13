# EMBER V1 Phase 3: counterfactual-assisted online PPO

The counterfactual arm reduced casualties relative to PPO-only on the 32 reporting DEV families, but it did not establish an evacuation gain. PPO+CF evacuated 197 of 473 people, compared with 193 for PPO-only and 197 for original V1. It recorded 104 casualties, versus 114 in each of those two arms. Relative to PPO-only, the ten fewer casualties coincided with four more evacuations and six more unresolved people. The predeclared joint criterion required an evacuation interval above zero and a casualty upper bound within +2 percentage points. Only the casualty criterion was met.

## What changed on reporting DEV

- **Repeated ineffective moves:** 8,596 for PPO+CF versus 8,759 for PPO-only, with almost equal numbers of attempted moves. The paired difference was −1.38 percentage points per attempt (95% interval [−4.19, +1.39]). The result does not establish a reduction. Compared with frozen Phase 2, the rate was 2.09 points lower [−4.06, −0.11], a secondary comparison.
- **Wall or boundary blocks:** 5,857 for PPO+CF versus 5,853 for PPO-only. The paired difference was +0.06 points per attempt [−2.90, +3.12]. Wall blocks fell by 40, while boundary blocks rose by 44. The rate was 3.83 points lower than frozen Phase 2 [−6.67, −1.00].
- **Outcome and action trade-offs:** PPO+CF had 94 fewer total blocked moves, including 56 fewer destination conflicts and 34 fewer occupancy blocks. The aggregate block rate difference was −0.78 points [−3.32, +1.70]. Stay increased only 0.25 points [−0.80, +1.63]; attempted moves fell by five. There is no evidence that widespread action suppression drove the casualty difference.
- **Generalization:** The point estimate for evacuation was flat or positive in every reporting stratum, but no stratum's interval excluded zero. The largest unseen geometry group gained 0.57 points [−2.78, +3.98]. The overall casualty difference was −2.11 points [−3.96, −0.39]; subgroup results and other secondary comparisons are descriptive and were not adjusted for multiplicity.

The strongest supported limitation is that useful counterfactual feedback was sparse and conditional. Only 26 of 108 targeted states yielded a usable pair, and 418 auxiliary steps reused those 26 states. Most teacher proposals tied the factual action in terminal utility under the fixed T1 continuation. One changed joint action can be valuable under T1 and still fail to alter the student's own downstream sequence. Whole-map preferences also do not identify the individual move responsible for the gain. On reporting DEV, more than 81% of attempts remained blocked in both learned arms and roughly half hit a wall or boundary. The high TRAIN entropy at larger populations and lower mean joint KL in the CF arm provide no indication of exploration collapse or excessive update size specific to that arm. The experiment cannot separate sparse feedback, joint credit assignment, or the factorized action representation as causes.

The comparison uses separately DEV-selected checkpoints from equal-budget training runs. The control selected update 16 and PPO+CF selected update 24. Thus the reporting difference estimates the two training-and-selection procedures on this panel; it is not a fixed-update contrast. The prespecified selector froze PPO-only update 16 because its selection utility was higher (−0.087691 versus −0.115468 for CF), subject to the casualty guard. Reporting DEV did not change that choice.

| Policy | Evacuated | Casualties | Unresolved | Stay | Blocked / attempted | Wall/boundary blocks | Repeated ineffective |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1 | 41.65% | 24.10% | 34.25% | 6.66% | 81.20% | 5860 | 8325 |
| v1-mask | 54.12% | 16.91% | 28.96% | 19.98% | 56.18% | 0 | 4204 |
| v1-coordinate | 60.68% | 15.22% | 24.10% | 46.82% | 0.00% | 0 | 0 |
| phase2 | 39.53% | 23.47% | 37.00% | 5.24% | 83.06% | 6323 | 8870 |
| ppo | 40.80% | 24.10% | 35.10% | 4.63% | 82.61% | 5853 | 8759 |
| cf | 41.65% | 21.99% | 36.36% | 4.88% | 81.83% | 5857 | 8596 |

Reporting DEV has 32 independent fresh topology families (473 people per arm), separate from selection DEV and
all TRAIN experience. Original V1, frozen selected Phase 2, equal-budget PPO continuation
and equal-budget PPO+CF use raw argmax, without external movement masking/coordination.
V1-mask and V1-coordinate are separate reporting-only ablations with external postprocessing.
All remain in the original Qwen3.5-2B V1 / LoRA / Clef lineage. No SFT, V1.1 artifacts,
expansion data or hidden benchmark simulator was used. Final evaluation remains sealed.

## Actual training and compute

Both arms completed 27 matched rounds. On-policy metrics:
- ppo: 216 episodes / 9675 joint transitions / 125235 person decisions; 609 optimizer steps. Mean shaped/terminal returns -0.058079/-0.063114.
- cf: 216 episodes / 9668 joint transitions / 124768 person decisions; 611 optimizer steps. Mean shaped/terminal returns -0.040680/-0.045715.

CF auxiliary gradients were applied in 418 actual optimizer steps. Process wall 10202.68s plus 1751.49s prior failed attempts. GPU peak allocated/reserved 14.967/24.010 GiB. The model base stayed frozen. Both arms restored the same Phase 2 actor/critic/Adam and used identical TRAIN-controlled curriculum stages, family pools and episode budget. Their on-policy visited states naturally diverged after learning. Full GPU utilization/power samples are preserved in gpu.jsonl.

PPO settings and primary outcome reward are fixed in the experiment config. The CF-only auxiliary coefficient is .05; no offline record is counted as an on-policy episode. Optimization means: `{"cf": {"actor_loss": -0.03296192351257421, "critic_loss": 0.03863934243944584, "joint_kl": 0.012981804950103551, "person_entropy": 0.8022181765046886, "clip_fraction": 0.205471659081506}, "ppo": {"actor_loss": -0.03148617018993038, "critic_loss": 0.04222719907067963, "joint_kl": 0.013745606100059244, "person_entropy": 0.7583737740744604, "clip_fraction": 0.21755525295904474}}`. Maximum pre-update replay discrepancy: `{"cf": 0.0, "ppo": 0.0}`.

## Attempt history, isolation, and numerical checks

Attempt 1 ran for 1,751.49 s and failed when Torch's compiled backward rejected a retained graph on the first eligible auxiliary pair. Its first four matched rounds were saved and independently replayed. The abandoned fifth round consumed eight episodes per arm and 305 PPO / 324 CF transitions, which remain in the raw logs and count against the budget. The repair uses independent single-use graphs for the auxiliary gradient probe and applied update. It passed on the archived failing pair with the original behavior actor; that diagnostic took no optimizer step.

Attempt 2 resumed the verified round-0004 state (`bf36802dbfc8ae2bf0011440a7ac13949420c536578c7d6eb0d4539ca5935a2d`) and collected 184 fresh episodes per arm. The 216 committed episodes per arm include the 32 from attempt 1. Across both attempts, the simulator ran 224 TRAIN episodes per arm (9,980 PPO and 9,992 CF transitions). Four hours were available for experiment processes, including DEV; attempts used 11,954.17 s in total, leaving 2,445.83 s. The one-hour independent audit allowance is separate. Attempt 3 was unused.

The fresh retry TRAIN pool had zero D4 topology overlap with attempt 1, Phase 1, Phase 2, and either fixed DEV split. Both arms saw identical ordered TRAIN families and curriculum stages; the PPO-only arm determined stage advances from TRAIN outcomes. Phase 2 update 96 remained immutable and supplied the shared critic, actor and Adam state (step 1,996). No new SFT, V1.1, excluded expansion or hidden benchmark inputs were used.

| Curriculum stage | Matched rounds | PPO joint KL | CF joint KL | PPO clip fraction | CF clip fraction |
|---|---:|---:|---:|---:|---:|
| Navigation | 2 | .00456 | .00445 | 5.53% | 5.07% |
| Congestion | 2 | .00749 | .00676 | 12.18% | 10.47% |
| Dynamic fire | 2 | .00633 | .00700 | 9.70% | 8.85% |
| Competing exits | 4 | .01068 | .01053 | 17.33% | 16.86% |
| Large unseen geometry | 17 | .01660 | .01541 | 26.33% | 24.75% |

These are means over logged state evaluations. The product-policy ratio compounds changes as more active people act together. Clipping and KL rose in the largest stage, where the KL guard stopped one PPO update and three CF updates early. Map type, fire, policy maturity and population changed together; the table is not an isolated population experiment. The 32-person forward/backward preflight preserved the weights and performed zero optimizer steps.

All 1,220 applied steps had finite, nonzero LoRA, head and critic gradients. The online probability replay discrepancy was zero. Independent CPU replay verified 944 complete online episodes (40,825 transitions) across both attempts, recomputed all 19,343 committed TRAIN targets exactly, and replayed all 3,453 teacher DEV transitions. The 43,201 counterfactual branch transitions are separate from these counts. Attempt 2 averaged 63.75% GPU utilization (peak 95%; peak device memory 25,813 MiB); collection throughput was 12.38 joint transitions/s for PPO and 12.36 for CF. Counterfactual CPU work in committed updates took 79.29 s of process wall time.

## Verified counterfactual evidence

108 actual TRAIN visited-state studies, 4038 branch episodes,
43201 counterfactual transitions, 26 selected offline
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

The public priority signal counts are {"bottleneck": 14, "coordination_conflict": 75, "delayed_escape": 2, "repeated_ineffective": 108, "wall_or_boundary": 107}.
Helped and harmed below refer to the sign of the measured mean. Inconclusive counts
nonzero means whose paired interval includes zero. Eligible applies the casualty rule as well.

| Proposal source | States | Helped | Harmed | Tied | Inconclusive | Eligible | Mean utility change vs factual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T0 | 108 | 24 | 1 | 83 | 1 | 23 | +0.01483 |
| T1 | 108 | 27 | 0 | 81 | 1 | 25 | +0.01719 |
| T2 | 108 | 27 | 0 | 81 | 1 | 25 | +0.01719 |
| T3 | 108 | 26 | 0 | 82 | 0 | 24 | +0.01671 |
| T4 | 108 | 28 | 0 | 80 | 1 | 26 | +0.01767 |

CF proposal T2/T3 use the explicitly recorded bounded configurations (500ms/24 rollouts and 200ms/20,000 ant steps); their effects must not be confused with the frozen default teachers evaluated below. Other proposal configurations and every failure/limit flag are preserved. Local variations can beat, tie or harm the factual action independently of any teacher.

## Independent uncertainty

Paired 5,000-replicate bootstrap over fixed reporting families, CF minus PPO, percentage points:
- evacuated: +0.85 [-1.86, +3.48].
- casualties: -2.11 [-3.96, -0.39].
- unresolved: +1.27 [-1.27, +3.98].

Behavior and latency deltas with paired family intervals:
- stay_rate: +0.253 [-0.798, +1.626] pp.
- blocked_rate: -0.778 [-3.318, +1.705] pp.
- wall_boundary_rate: +0.057 [-2.905, +3.118] pp.
- coordination_conflict_rate: -1.009 [-4.811, +2.563] pp.
- repeated_ineffective_rate: -1.379 [-4.193, +1.393] pp.
- mean_joint_latency_ms: +0.204 [-1.140, +1.506] milliseconds.
- restricted_episode_steps: -0.219 [-0.875, +0.312] steps.

Intervals describe these sampled geometry families and simulator seeds. One training seed per arm does not quantify training-seed variability. Selection DEV chose checkpoints; reporting DEV did not affect gradients, difficulty or selection. No claim of superiority over all teachers follows from isolated counterfactual wins.

## Frozen default teacher comparison

Matched completed panel: 32 reporting families; 0 teacher episodes failed or timed out and remain explicit incomplete records. The table uses the same scenario IDs for every arm. A reduced panel is conditional on completion and cannot establish population superiority.

| Arm | E | C | U | Stay | Blocked / attempted |
|---|---:|---:|---:|---:|---:|
| v1 | 41.65% | 24.10% | 34.25% | 6.66% | 81.20% |
| v1-mask | 54.12% | 16.91% | 28.96% | 19.98% | 56.18% |
| v1-coordinate | 60.68% | 15.22% | 24.10% | 46.82% | 0.00% |
| phase2 | 39.53% | 23.47% | 37.00% | 5.24% | 83.06% |
| ppo | 40.80% | 24.10% | 35.10% | 4.63% | 82.61% |
| cf | 41.65% | 21.99% | 36.36% | 4.88% | 81.83% |
| T0 | 87.74% | 8.03% | 4.23% | 25.29% | 7.07% |
| T1 | 89.01% | 1.48% | 9.51% | 26.64% | 0.00% |
| T2 | 89.01% | 1.48% | 9.51% | 26.20% | 0.00% |
| T3 | 89.22% | 1.90% | 8.88% | 34.33% | 0.00% |
| T4 | 89.22% | 1.48% | 9.30% | 37.75% | 0.00% |

## Generalization and latency

| Family | PPO E/C/U | CF E/C/U |
|---|---|---|
| competing-exits | 41.8%/22.1%/36.1% | 41.8%/18.9%/39.3% |
| congestion | 56.9%/27.8%/15.3% | 59.7%/23.6%/16.7% |
| dynamic-fire | 35.8%/52.2%/11.9% | 35.8%/49.3%/14.9% |
| large-unseen | 37.5%/15.3%/47.2% | 38.1%/14.8%/47.2% |
| navigation | 30.6%/13.9%/55.6% | 33.3%/13.9%/52.8% |

| Arm | Joint decision p50 / p95 ms |
|---|---:|
| v1 | 57.00 / 79.36 |
| v1-mask | 57.62 / 85.90 |
| v1-coordinate | 57.42 / 85.55 |
| phase2 | 56.51 / 78.34 |
| ppo | 57.22 / 78.72 |
| cf | 57.65 / 79.10 |
| T0 | 0.14 / 0.25 |
| T1 | 0.52 / 1.58 |
| T2 | 32.01 / 208.87 |
| T3 | 1.38 / 10.06 |
| T4 | 0.93 / 3.85 |

Neural latency is CUDA-synchronized forward plus action selection, excluding encoding and simulator time. Teacher latency is its public-observation decision wall time. Workload sizes and hardware differ; raw latency distributions remain available.

## Audit and exact checkpoint files

The original completion audit passed CPU replay but failed the fresh-load greedy check. Repeating the GPU audit with the trainer's Torch thread count of four reproduced all 1,383 PPO and 1,376 CF reporting decisions, with zero probability discrepancy. Both first-rollout autograd replays also passed with zero discrepancy. No tolerances were relaxed and no training was repeated. The initial failure, diagnostic and retry logs are preserved. Trainer, CPU replay, GPU retry and archive completion each exited 0. The first three actual exits are bound into the archive by `FINAL_AUDIT_EXIT_CODES.json`.

| Checkpoint | PPO-only update 16, frozen candidate | PPO+CF update 24 |
|---|---|---|
| Actor digest | `sha256:df5afa5a6d09031f5375503a8816eeeec26ee4faeaf4ee0ed040df821524154d` | `sha256:05d665f4f7acedee57e2fa2acd9faa390df3f65b11de2719058cfb4ab8a00c72` |
| LoRA SHA-256 | `67564ef918231be2c96792b86c39abfb62494cf0a11eca3146ae629dc5511da4` | `edb54c9ce303951a4943b0159c48f865f634797e6abd719e86726421a99a8949` |
| Clef head SHA-256 | `5dd47b3debf21b971dce172a8a2b33c8d473a1378c0d9e105b5d56bf644b6b79` | `1d13d1d4bf020b686ff078e4d678b6aef043ec31bf95511dacca6f03a2af05ce` |
| Trainer SHA-256 | `8f26e0d57e60562dd463667bff0f1e7a1404b114726fc89559b7f1b970295724` | `034ddce751faa8a2d81d97bfa6226f6b9fae49d8847c19ec0d547d1313bb30e5` |
| Adam state step | 2,352 | 2,537 |
| Applied auxiliary steps in checkpoint | 0 | 372 |
| LoRA / Clef L2 change from Phase 2 | 0.256187 / 0.760025 | 0.312121 / 0.846853 |

Canonical evidence: `/home/darshith/ember-ws/rl-evidence/r22-counterfactual-ppo@1/attempt-2`. Selected actors and resumable trainers are under `ppo/checkpoints/update-0016` and `cf/checkpoints/update-0024`. The verified Windows mirror is `C:/Users/Admin/Documents/ember-evidence/r22-counterfactual-ppo@1/attempt-2`; each arm's `selected_checkpoint` contains `actor/`, `trainer.pt` and `identity.json`.

- [Canonical evidence manifest](../data/rl/evidence/r22-counterfactual-ppo@1/EVIDENCE_MANIFEST.json): SHA-256 `c88ff645362a96366ad5f386d7e48211b5f32cc8df37c476424c73b0b07d7e17` (286 files, 8,980,296,320 bytes).
- [Attempt 1 ancestry manifest](../data/rl/evidence/r22-counterfactual-ppo@1/ANCESTRY_MANIFEST.json): SHA-256 `c047db6f56331616f2efa66a81c882c58bc17762530e9b13f3170cd0463d04c2` (47 files rehashed).
- [Publication verification](../data/rl/evidence/r22-counterfactual-ppo@1/publication_verification.json): independently rehashed 211 Windows mirror files (2,002,637,005 bytes), matched both selected identities, and checked exit markers. The Windows copy receipt SHA-256 is `05ab21a4223e6e91078694f616d9b85242e6a2b885864da8c6f4c76ff45e981f`.
- [Verified CF dataset index](../data/rl/evidence/r22-counterfactual-ppo@1/verified_counterfactual_dataset.json): the study paths resolve in the canonical archive across both attempts. Replay uses the stored private snapshots in the evaluator; the actor receives only public observations.
- [All arms by family](../data/rl/evidence/r22-counterfactual-ppo@1/per_family.json), [movement and stage diagnostics](../data/rl/evidence/r22-counterfactual-ppo@1/detailed_analysis.json), [paired intervals](../data/rl/evidence/r22-counterfactual-ppo@1/independent_counterfactual_audit.json), and [training curves](../data/rl/evidence/r22-counterfactual-ppo@1/curves.html).

Training source for attempt 1 was `7ab95ff1fd12238331b3e9e90430db405190cf94`; attempt 2 used `db331d5fe4735d7f654c1485070d6ce3882ac4e0`. Both used configuration SHA-256 `7f82b442f4a600da9e8db9ee941c108b2a0d94b30d59525285112618fd6abe7d`. The execution and runtime records preserve the command, source, resume ancestry and package versions. Reproduce verification on a scratch copy with the same ancestry available so generated outputs do not change the manifested archive. The archived completion scripts specify the pinned Python 3.11 environments and the fresh actor replay settings. No final evaluation was run.

## Runner and replay commands

Best DEV-frozen candidate: **ppo**, `/home/darshith/ember-ws/rl-evidence/r22-counterfactual-ppo@1/attempt-2/ppo/checkpoints/update-0016/actor`. Final evaluation has not run.

- cf: `/home/darshith/ember-ws/rl-evidence/r22-counterfactual-ppo@1/attempt-2/cf/checkpoints/update-0024/actor`, digest `sha256:05d665f4f7acedee57e2fa2acd9faa390df3f65b11de2719058cfb4ab8a00c72`, update 24; fresh saved-adapter reload max logit error 0.
- ppo: `/home/darshith/ember-ws/rl-evidence/r22-counterfactual-ppo@1/attempt-2/ppo/checkpoints/update-0016/actor`, digest `sha256:df5afa5a6d09031f5375503a8816eeeec26ee4faeaf4ee0ed040df821524154d`, update 16; fresh saved-adapter reload max logit error 0.

Run `python -m ember_v1.rl_cf_run --config data/rl/r22-counterfactual-ppo@1.json --model-root <verified-original-V1> --phase2 <verified-complete-Phase2-evidence> --output-root <external-evidence> --attempt <unused-authorized-attempt>`. `--resume <paired-round-directory>` restores both actors/critics/Adam/RNG/curriculum counters at equal update counts. Frozen input identities and topology isolation fail closed. Run `python -m ember_v1.rl_cf_verify <attempt>` and `python experiments/rl_cf_report.py <attempt>` to regenerate the independent evidence and this report. Exact runtime/source identities, command, manifests and raw branches are preserved outside Git.
