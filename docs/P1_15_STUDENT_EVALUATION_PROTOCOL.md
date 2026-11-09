# P1-15 independent student evaluation — protocol draft

**Status:** `DRAFT_NOT_FROZEN` (`data/protocols/p1-15-student-evaluation.json`, version `0.2.0-draft`).
Written before any student outcome exists. **Authorizes nothing:** no test generation, no student run, no GPU.
Issue #48. Decision code: `src/ember_v1/student_eval_protocol.py` (tested on synthetic numbers only).

## 1. Scientific claim

The raw student (no teacher at inference, no deterministic shield) has fewer casualties+unresolved than the
**strongest single deployable teacher**, with casualties AND unresolved not worse (bound and point estimate),
on new family-disjoint scenarios, within simulator-safety, privacy and latency limits. "Fewer casualties and
fewer unresolved" is a separate, stronger claim (`both_improved`, §4). Matching T1, beating only weaker
teachers, or a shield-only improvement is not success. Beating the hindsight oracle is not required.

## 2. Sets

| Set | Use | Rule |
|---|---|---|
| Comparator selection | name the strongest deployable teacher; measure each teacher's latency | new families; all five teachers; student never runs here |
| Test | the claim | 32 families × 3 scenarios, new; sealed until student, comparator and protocol are frozen |
| Forbidden | — | p1-02-v1 train/dev/calibration, P1-09, sealed index, EMBERBench, any P1-12/13/14 TRAIN or counterfactual scenario, any model-selection family |

`check_test_isolation` refuses any family or geometry-core overlap between the test set and every other set.

## 3. Comparator and oracle

- Comparator: among teachers flagged **deployable** (running alone, it meets the frozen latency limits on the
  deployment hardware, measured on the comparator-selection set), lowest casualties+unresolved; ties by
  casualties, then teacher order T0..T4 (`select_comparator(..., deployable=...)`). A non-deployable teacher
  is ranked and reported, never the comparator. `set_kind="test"` is refused. No outcome of the test set is used.
- Oracle: per-scenario best teacher in hindsight (`oracle_reference`, `deployable: false`), reported apart. It is
  never a baseline and never a target of the claim.
- P1-09 totals (T4 419, T2 428, T1 435, T3 460, T0 608 casualties) are history, not the comparator rule.

## 4. Decision rule (`decide`)

| Item | Rule |
|---|---|
| Unit | paired per scenario (same scenario, seed, simulator); averaged per family (family-clustered) |
| Primary | one-sided 97.5 % family-bootstrap upper bound of mean(student − comparator) casualties+unresolved < 0 (10 000 resamples, seed 20261109) |
| Guard: casualties | family-bootstrap upper bound ≤ 0 **and** family-mean point estimate ≤ 0 |
| Guard: unresolved | upper bound ≤ 0 **and** point estimate ≤ 0 (v0.1 allowed +0.5 per scenario; withdrawn) |
| Severe regression | a scenario where ANY of casualties+unresolved, casualties, unresolved is worse by ≥ max(3, 0.05 N); count ≤ 5 % of scenarios **and** one-sided Clopper-Pearson upper bound of the rate ≤ 0.10 |
| Simulator safety | 0 failures: invalid/incomplete joint map, simulator error, fail-closed step, unfinished episode |
| Privacy | `model_input_privacy_violations` must be audited and 0, else `INVALID_RUN` |
| Independence | 0 teacher calls in the student path, else `INVALID_RUN` |
| Latency | p50/p95/p99 per step (all rows of the joint map), **not yet set**; null or missing evidence blocks freeze (§5) |
| Freeze check | `freeze_problems` refuses null latency, missing latency evidence and any positive guard margin |

Claims (`decide(...)["claims"]`), all require the primary test and every guard:

| Claim | Extra condition | Multiplicity |
|---|---|---|
| `casualties_plus_unresolved_improved` | — | the single primary test |
| `casualties_improved` / `unresolved_improved` | that outcome's upper bound < 0 and Holm-significant | Holm over the two outcomes at one-sided 0.025 |
| `both_improved` | casualties AND unresolved upper bounds < 0 | intersection-union test: each at 0.025, no adjustment |
| secondary: beats teacher t | same primary test against each teacher | Holm over 5 at one-sided 0.025 |

Arms: raw student (claim) and student + deterministic shield (reported separately, never the claim).

Metrics reported: casualties, unresolved, evacuated, casualties+unresolved, severe regressions (with the
outcome per scenario), simulator-safety failures, joint-map-invalid and fail-closed steps, privacy violations,
movement conflicts, latency p50/p95/p99, CPU and GPU seconds, teacher calls in the student path.

## 5. Latency budget (recommendation; no number is frozen)

Evidence now:

- A10 inspect (attempt 7): one call, N = 3, 617 input tokens, 113.4 s client elapsed **including container
  start**. There is no forward timing. It supports no limit.
- CPU token study (#44 `p1-chunk-reserved@1.0.0`): rows reach ≈ 15–16 k tokens at N = 110–200. Sequential
  reservations make the rows of one step sequential, so step latency ≈ Σ row forwards + tokenization.
- The deployment step deadline (HLD A7) is not documented.

Rule (`guards.latency_ms.budget_rule`): p99_max ≤ the documented step deadline; p50/p95 from the measured
warm distribution plus a preregistered margin. Teachers get the deployable flag from the same limits on the
deployment CPU. Required GPU measurements (`gpu_measurements_required`): warm A10 forward per row at
1 k/4 k/8 k/16 k tokens, batch 1, CUDA-synchronized; rows per step for N = 50/100/110/200 with the chosen k;
end-to-end step latency with n ≥ 100 steps for p99; cold start reported apart.

## 6. Open items before freeze

1. Deployment step deadline (A7) and the measured GPU preflight (§5).
2. Power: 32 families is a planning number; check the family-level variance of teacher differences on the
   comparator-selection set before freezing the test size. With 96 scenarios, 0 severe regressions gives a
   Clopper-Pearson upper bound of 0.038; 4 gives > 0.10 (the rate guard then fails).
3. Generator plan for the two new sets (new lineages; firewall against all used families and cores).
4. Privacy audit tool: field-level check of every model input against the public observation schema.
