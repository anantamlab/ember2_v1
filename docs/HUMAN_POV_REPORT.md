# Human point-of-view training and report

The human-view LLM completed supervised training and 64 PPO updates. The frozen selection rule chose PPO update 20 before the single 32-family report evaluation. Reporting results below describe that checkpoint; they did not change the choice. Private storage is pending upload.

## Reporting panel

Namespace `human-pov-ws-full-20261113t214243z`, report split, 32 families, 48 steps, radius 4, line of sight and memory enabled, exits learned by sight, route hints enabled. Every arm uses the same generated scenarios.

| Arm | Evacuated | Casualties | Unresolved | Utility | Blocked / attempt | Wall / boundary |
|---|---:|---:|---:|---:|---:|---:|
| Belief teacher (rule-based) | 303/352 (86.08%) | 28 (7.95%) | 21 (5.97%) | 0.656960 | 20.79% | 0 / 0 |
| Random legal moves (rule-assisted) | 99/352 (28.12%) | 132 (37.50%) | 121 (34.38%) | -0.726562 | 10.24% | 0 / 0 |
| Selected LLM, raw | 303/352 (86.08%) | 27 (7.67%) | 22 (6.25%) | 0.660511 | 21.20% | 0 / 0 |
| Selected LLM + legal mask (rule-assisted) | 303/352 (86.08%) | 27 (7.67%) | 22 (6.25%) | 0.660511 | 21.20% | 0 / 0 |

`student_raw` is the learned model alone. The masked model and random legal moves use an external legal-action rule. The belief teacher plans routes with rules using the same partial view. Utility is `(evacuated - 2*casualties - 0.75*unresolved) / people`.

## Training and selection

Round 0 collected 43,019 TRAIN rows across 256 families. SFT ran 5,378 optimizer steps, lowering loss from 1.600112617 to 0.000002702; the 2,000-row TRAIN probe had accuracy/legal rate 1.0. Round 1 collected 40,742 rows with beta 0.5. Student-only agreement was 20,358/20,386 (99.862651%), above the frozen 0.99 stop threshold. DAgger stopped and retained s0; rounds 2/3 and an s1 fit were unnecessary under that rule. Both d0 and d1 were checked during reporting.

The raw s0 selection utility was 0.696175637, within 0.008498584 of the belief teacher, with zero wall/boundary blocks. This passed the pre-PPO gate. Small CPU PPO ran 200 updates / 3,200 episodes; update 160 had selection utility 0.686260623, below the teacher's 0.704674221 on that panel.

The first LLM attempt stopped on replay error 0.001501784 above the unchanged 0.001 limit. Diagnosis showed that regrouping bf16 padded forwards changed probabilities. Commit `df4b431892a7b1190b44df0353651daeeffbb001` preserves the sampling forward groups through PPO, packs whole groups within minibatch 16, and audits every group before any optimizer step. Both CPU regressions failed before repair. The optimizer-free eight-episode GPU regression replayed exactly.

The repaired LLM attempt completed 64 updates, 512 TRAIN episodes and 750 optimizer steps in 9,696.596 seconds. Every update has finite metrics; maximum replay error is 0. All seven checkpoint file sets and tensors passed independent checks. Native casualty-guarded selection chose update 20: utility 0.711756374, 312 evacuated / 24 casualties / 17 unresolved of 353. The immutable s0 baseline was 311/27/15, utility 0.696175637. Gain 0.015580737 and casualty-rate change −0.008498584 satisfied the frozen parent comparison. Earlier checkpoints win ties. The checkpoint bytes changed from s0 and are recorded by file hashes.

## Isolation and reproducibility

Candidate freeze `561f28f47d1b79a34aac3174cf7cf25209bcd322b13444d71ce7c861d2fbe711` precedes the report start. The single report invocation includes d0 and d1. Symmetry-invariant topology checks found zero overlap between report and selection, both DAgger datasets and all 3,200 small / 512 LLM PPO TRAIN families. The final candidate has the same hashes after reporting. Sealed FINAL and reserved V1.1 were never used.

Compact evidence: [`summary.json`](../data/rl/evidence/human-pov/human-pov-ws-full-20261113t214243z/summary.json), [`candidate-freeze.json`](../data/rl/evidence/human-pov/human-pov-ws-full-20261113t214243z/candidate-freeze.json), [`report.json`](../data/rl/evidence/human-pov/human-pov-ws-full-20261113t214243z/report.json). Checkpoint file identities and storage are in [`human-pov-checkpoint-registry.json`](../data/rl/human-pov-checkpoint-registry.json). Source for training/reporting: `df4b431892a7b1190b44df0353651daeeffbb001`. The public base and pinned runtime match `CURRENT_STATE.md`.

```bash
# The report command already ran once; this is its recorded invocation, not a request to rerun.
python -m ember_v1.human_dagger eval --namespace human-pov-ws-full-20261113t214243z --split report --count 32 \
  --student <run>/ppo-llm-repair-01/update-0020 --kind clef --max-steps 48 --seed 0 \
  --train-data <run>/d0 <run>/d1 --out <run>/final-report.json
```

## Storage and budgets

The local package is hash-verified: 269 files, 2,092,776,002 bytes. Private upload and remote hash verification remain pending because no HF write credential is available. The intended private target is `sankalphs/ember-v1-models/human-pov/human-pov-ws-full-20261113t214243z`. Completion is not claimed.

Weights, tokenizer files, s0 optimizer state, all produced PPO actor checkpoints, dataset rows and raw logs remain outside Git. The package also retains the failed attempt, diagnostics, immutable phase records, source archive and test receipt. PPO critic/optimizer state was not saved by this runner, so its actor checkpoints support inference rather than exact training continuation.

Completed GPU phase durations total20669.581seconds (5.742hours) of 96 hours, including the failed attempt and diagnostics. CPU used through packaging was4395.115seconds of 43,200; further publication checks are charged separately. Accounting sums each immutable phase once and excludes idle gaps. Cloud spend was $0; the package is below 100 GiB.

This is one training seed and 32 report families. There is no training-seed variance estimate. Partial views change the task, so these rates cannot be compared directly with R19–R22. The selection gain against s0 does not establish a reporting gain against s0, which was not evaluated again on report.
