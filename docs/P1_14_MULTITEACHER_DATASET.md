# P1-14 multi-teacher outcome-aware dataset — protocol draft

**Status:** `DRAFT_NOT_FROZEN` (`data/protocols/p1-14-multiteacher-dataset.json`, version `0.1.0-draft`).
**Authorizes nothing.** `multiteacher_dataset.export` always raises `DATASET_GENERATION_BLOCKED`. Records can be
built only from `synthetic_fixture` reports. Issue #47. Depends on #45 (P1-12) and #46 (P1-13).

## 1. Evidence classes

| Class | Name | Built from | Use | Weight |
|---|---|---|---|---|
| A | successful demonstration | trajectory of a complete matched group whose casualties+unresolved AND casualties equal the group minimum (ties allowed) | SFT positive | scenario budget, equal split |
| B | unsuccessful demonstration | every other complete trajectory | contrastive only: analysis, and the states where P1-13 probes look for better actions | reported, not a positive target |
| C | counterfactual preference | verified P1-13 contrast: `a_better`/`b_better`, uncensored, ≥3 replicates and ≥2 strict improvements (stochastic fire), or 1/1 under deterministic fire | preference pair (chosen, rejected) at one state | scenario budget, equal split |
| D | inconclusive intervention | every other verified contrast (censored, mixed sign, no difference, below threshold) | excluded from every loss; kept and counted | 0 |
| E | student-visited state | reserved for P1-16 (#49) | refused by the validator | — |

Rules that follow from the classes:

- A class comes from **matched outcome evidence**, never from the teacher's name. T0 can be class A in a
  scenario where shortest-path is optimal; T1 can be class B where it loses people.
- A trajectory-level label is weak per-decision evidence: a best trajectory can still hold poor moves, and a
  worse trajectory can hold good moves. Class C is the only per-state preference evidence.
- A failed teacher makes the group incomplete. Then no A/B record is built from that group.
- No action probability, confidence, value estimate or causal label is ever created
  (`forbidden_fields`). `teacher_values` holds only values T2 really computed, copied from its proposal.
- B records are **not** unlikelihood targets: a B action at a B state can be correct. B states feed the
  P1-13 probe queue.

## 2. Record schema `mt-dataset-record@0.1.0`

Common: `record_version, class, use, scope, scenario_hash, family_id, source, observation,
observation_digest, weight, record_hash`.

| Class | Extra fields |
|---|---|
| A/B | `teachers` (all teachers that proposed this exact joint action at this exact state), `actions` (complete joint map), `proposal_hashes`, `teacher_values` (T2 only or null), `step`, `evidence` (group best, per-teacher trajectory outcomes, `uninformative_scenario`) |
| C | `chosen`, `rejected` (each: name, complete joint action, proposer set or composite sources, replicate-0 outcome), `evidence` (report hash, verdict, replicates, strict improvements, horizon, censoring, continuation identity, per-replicate deltas, causality limit) |
| D | `alternatives` (both sides), `evidence` (as C, plus `below_threshold`) |

`validate_dataset` checks every field set, hash, public-observation leakage, completeness of every joint
map, the outcome rule for A/B, the thresholds for C, zero weight for D, refusal of E, duplicates (same class,
state and joint action), and — once ≥20 scenarios exist — the share caps below.

## 3. Sampling and weighting (auditable)

| Rule | Value | Reason |
|---|---|---|
| Unit | one budget of 1.0 per (scenario, class) | long episodes and large N do not dominate |
| Inside a scenario | equal split over distinct (state, joint action) records | duplicated states count once |
| Uninformative scenario (all five teachers equal on primary and casualties) | budget × 0.25 | easy cases do not dominate |
| Teacher share of class A weight | ≤ 0.5 | no single policy dominates the imitation target |
| Family share of A+C weight | ≤ 0.05 | no family dominates; needs ≥ 20 families |
| Epoch sampling | rows drawn ∝ weight to a fixed token budget, seed published | cost is bounded before training |

`audit` prints class counts, weight per class, teacher shares and family shares. The fixture test shows, for
one hand-built scenario: A 9, B 17, C 4, D 2 records; A teachers are exactly the matched-best teachers (T2, T3).

## 4. Open design decision (needs review before freeze)

When two class-A teachers propose different joint maps at the same state, a per-person cross-entropy over both
targets trains a mixture of marginals. The argmax of a mixture can be a joint map no teacher proposed, with
blocked moves. Proposed default: keep both with split weight, and (a) prefer the side verified by a class C
pair when one exists, (b) report on DEV the blocked-move rate of the student's assembled argmax map. The
alternative is to keep one target per state by a preregistered rule.

## 5. Gates before real generation

1. P1-12 TRAIN collection protocol frozen and authorized; pilot passed (`docs/P1_12_TRAIN_COLLECTION_PROTOCOL.md`).
2. P1-13 TRAIN counterfactual budget frozen: states per scenario, replicates, horizon (to termination or ≥ the
   scenario's remaining steps), continuation(s).
3. This protocol frozen in its own commit with its hash; owner authorization in writing.
4. Leakage audit: no family, lineage or geometry core of DEV, calibration, selection, P1-15 test or the sealed
   index in any record.
