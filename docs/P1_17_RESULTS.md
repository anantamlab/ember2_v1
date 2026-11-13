# P1-17 results — full TRAIN collection, dataset, counterfactual batch, SFT preparation

Machine-readable summary: `data/collection/p1-17-train-full@1/result/summary.json`. Raw evidence is outside git
in `D:/Coding/ember-evidence/` (each directory has `SHA256SUMS` and is read-only). Collection method:
`docs/P1_17_TRAIN_COLLECTION.md`. Dataset protocol: `docs/P1_14_MULTITEACHER_DATASET.md` (1.0.0 FROZEN).

## 1. Collection `p1-17-train-full@1` (attempt 1, session 1)

| Item | Value |
|---|---|
| Authorization | PR #59 (merge `a915a76`), single-owner, frozen plan `sha256:bff49b47…c341`, execution commit `6df51e8` |
| Coverage | **189 / 189 groups verified complete** (945 teacher episodes, 8,571 people), 0 interrupted, 0 retries |
| Wall time | 6,502 s (1 h 48 min), 8 groups at once, 0 RAM waits, min free RAM 5.3 GiB |
| CPU | T0 172 s, T1 347 s, T2 43,568 s, T3 808 s, T4 451 s; 0 decision-cap overruns |
| Verifier | `verification_hash sha256:4aeb9293…76f1`; full replay of every group; 792 files in `SHA256SUMS` |

Casualties + unresolved (lower is better), all 189 groups:

| T0 | T1 | T2 | T3 | T4 |
|---:|---:|---:|---:|---:|
| 3,508 | 2,919 | 2,923 | 3,055 | 2,933 |

Families (64) where the teacher is tied best: T0 22, T1 43, T2 41, T3 35, T4 51. All five teachers chose the
same step-0 map in 12.2 % of groups.

**Replay versus regeneration.** Replay of stored actions is exact for all 189 groups. Regeneration (running the
teachers again) is exact except where T2 hit its 20 s CPU budget: 7 / 8 regeneration groups identical
(`g-9655b53de725243e` differs in T2 only, 10 budget-limited decisions); 7 / 8 pilot-overlap groups identical
with the P1-12 pilot (`g-a2993d188ad45093` differs in T2 only, 67 budget-limited decisions). T2 had 325
budget-limited decisions in 20 groups and 17 proposed moves into burning cells.

## 2. Dataset `p1-17-dataset@1` (protocol 1.0.0, frozen at `b2437d4`)

The protocol was frozen before the builder read any per-group outcome. Only the verifier's aggregate totals had
been seen. Rules: class A = group-best outcome (casualties+unresolved and casualties, over all five teachers),
not T0 (diagnostic), and no move into a burning cell; one selected whole map per state; family weighting unit.

| Item | Value |
|---|---|
| Status | **VALID**, 0 problems, `dataset_hash sha256:d7e374d2…6951`, 333 s with 3 workers |
| Records | 46,852: A 26,411 (26,043 selected SFT examples), B 20,441 |
| Class reasons (per teacher decision) | best_outcome 32,409; outcome_not_best 15,908; diagnostic_teacher (T0) 4,827; unsafe_fire_entry 8 |
| Class-A weight share | T1 23.6 %, T2 22.7 %, T3 19.3 %, T4 34.4 % (review threshold 50 %: none) |
| Families | 64 with class A; max family share 2.7 % (cap 5 %) |
| No class A | 2 scenarios (`g-44f1e96b951b895a`, `g-673c66e0077d2604`): no teacher has both the lowest C+U and the lowest casualties |
| Competing maps | 240 states had two or more class-A maps; decided by mean trajectory steps (64) or the name-free tie-break (176) |
| Selection-bias flags | `n>110 / deterministic` stratum (under-selected) and T0 (excluded by rule) |
| Teacher-dev-similar families | 3 families, class-A weight 2.75 of 37.0; flagged in every record |

## 3. Counterfactual batch `p1-17-cf-batch@1` (preregistration `f73ca69`, FROZEN)

16 scenarios (4 lowest-rank per people bin) × continuations T1 and T4. State: step 0 (matched initial state).
Interventions: the distinct **stored** step-0 maps of the five teachers (no live teacher call). Replicates 3
(stochastic) or 1 (deterministic); horizon = max_steps.

| Item | T1 continuation | T4 continuation |
|---|---:|---:|
| Studies (verified, continuation re-run) | 16 (16) | 16 (16) |
| No step-0 disagreement | 2 | 2 |
| Contrasts: a/b better | 21 | 8 |
| Contrasts: no difference | 66 | 73 |
| Contrasts: inconclusive | 3 | 9 |
| Censored | 0 | 0 |

Records: 12 class C (preference candidates) and 168 class D. Every record has weight 0; preference approval is
disabled. CPU 1,539 s, wall 290 s. The class C winners differ by continuation and scenario (T0, T1, T1+T2, T2,
T3 and T4 maps each win at least once; the losing map is T3 or T0). The batch is small (16 states); it gives no
general ranking.

`batch.json` reports `records_valid: false`. Cause: the validator treated the same map pair under T1 and under
T4 (two study reports) as a duplicate. Fix `9d76916` (the duplicate key includes the report); the unchanged
records re-validate OK (`D:/Coding/ember-evidence/p1-17-ops/cf_batch_revalidation.json`).

## 4. SFT preparation `p1-17-sft@1` (`0485e2f`)

| Split | Examples | Rows (k = 100) | 2-row examples | Families | Weight |
|---|---:|---:|---:|---:|---:|
| train | 23,913 | 26,000 | 2,087 | 58 | 33.75 |
| val (family-isolated) | 2,130 | 2,285 | 155 | 6 | 3.25 |

Model view only (observation, whole map, weight). The DEV and calibration manifests share no family, lineage,
scenario or geometry core with the dataset. Each row's loss mask covers every active person once. Status
`PREPARED_NOT_AUTHORIZED_FOR_TRAINING`. **Token counts are not measured**: the pinned Qwen3.5-2B tokenizer is not
on this machine. They must be measured on the training host and any row over 16,384 tokens must fail closed.

## 5. Not done

No student training, no GPU or Modal spend (USD 0), no EMBERBench or DEV evaluation, no preference approval. No
claim is made that a student learned anything.
