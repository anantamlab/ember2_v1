# P1-15 independent student evaluation — protocol draft

**Status:** `DRAFT_NOT_FROZEN` (`data/protocols/p1-15-student-evaluation.json`). Written before any student
outcome exists. **Authorizes nothing:** no test generation, no student run, no GPU. Issue #48.
Decision code: `src/ember_v1/student_eval_protocol.py` (tested on synthetic numbers only).

## 1. Scientific claim

The raw student (no teacher at inference, no deterministic shield) has fewer casualties+unresolved than the
**strongest single deployable teacher**, with no worse casualties, on new family-disjoint scenarios, within
safety and latency limits. Matching T1, beating only weaker teachers, or a shield-only improvement is not
success. Beating the hindsight oracle is not required.

## 2. Sets

| Set | Use | Rule |
|---|---|---|
| Comparator selection | name the strongest teacher | new families; all five teachers; student never runs here |
| Test | the claim | 32 families × 3 scenarios, new; sealed until student, comparator and protocol are frozen |
| Forbidden | — | p1-02-v1 train/dev/calibration, P1-09, sealed index, EMBERBench, any P1-12/13/14 TRAIN or counterfactual scenario, any model-selection family |

`check_test_isolation` refuses any family or geometry-core overlap between the test set and every other set.

## 3. Comparator and oracle

- Comparator: lowest casualties+unresolved on the comparator-selection set; ties by casualties, then by
  teacher order T0..T4 (`select_comparator`; refuses `set_kind="test"`).
- Oracle: per-scenario best teacher in hindsight (`oracle_reference`, `deployable: false`), reported apart.
- P1-09 totals (T4 419, T2 428, T1 435, T3 460, T0 608 casualties) are history, not the comparator rule.

## 4. Decision rule (`decide`)

| Item | Rule |
|---|---|
| Unit | paired per scenario (same scenario, seed, simulator); averaged per family |
| Primary | one-sided 97.5 % family-bootstrap upper bound of mean(student − comparator) casualties+unresolved < 0 (10 000 resamples, seed 20261109) |
| Guard: casualties | upper bound ≤ 0 |
| Guard: unresolved | upper bound ≤ 0.5 |
| Severe regression | scenario with student − comparator ≥ max(3, 0.10 N); at most 5 % of scenarios |
| Safety | 0 safety failures (invalid maps, crashes) |
| Independence | 0 teacher calls in the student path, else `INVALID_RUN` |
| Latency | p50/p95/p99 limits, **not yet set** (null blocks freeze; set from GPU preflight and HLD A7) |
| Secondary | same test against each teacher, Holm over 5 at one-sided 0.025 |
| Arms | raw student (claim) and student + deterministic shield (reported separately) |

Metrics reported: casualties, unresolved, evacuated, casualties+unresolved, severe regressions, safety
failures, movement conflicts, latency p50/p95/p99, CPU and GPU seconds, teacher calls in the student path.

## 5. Open items before freeze

1. Latency limits (measured GPU preflight).
2. Power: 32 families is a planning number; check the family-level variance of teacher differences on the
   comparator-selection set before freezing the test size.
3. Generator plan for the two new sets (new lineages; firewall against all used families and cores).
