# T4 v1.0.0 failure analysis (overnight P1)

**Status:** exploratory diagnosis. It explains the failed G5 of the frozen v1.0.0 pilot. It is not confirmation evidence. The v1.0.0 code, hashes, pilot report (`data/baselines/p1-08-t4-pilot.json`) and the `BLOCKED_BY_MECHANISM_CHECK` decision are unchanged.

## 1. Method

- Tool: `python -m ember_v1.teacher.t4_v1_diag` (`src/ember_v1/teacher/t4_v1_diag.py`). Report: `data/baselines/p1-08-t4-v1-diagnosis.json`.
- `TracedPlanner` subclasses the frozen v1.0.0 planner. It only records events. The run compares every decision with the frozen pilot row (yields over the ID rule, conversions, repair rounds, vacate refusals, step) and stops on a difference. Result: `reproduces_frozen_pilot_rows: true` for all 3 755 pilot decisions of A0, A1 and A2.
- Per plan call it records a **counterfactual**: the τ = 1 cell that the same call chooses when every reservation of a later-ranked person is removed. A difference means a later-ranked reservation changed the decision.
- Risk strata for every variant come from the frozen A2 risk module on the same observation.

## 2. Defects and the invariants they break

| ID | Defect (v1.0.0 code) | Broken invariant | Fixture (`tests/test_teacher_t4_v11.py`) |
|---|---|---|---|
| D1 | A person that fails to vacate stays. Its stay is written by `reserve()` over the τ = 1 claim of the earlier planner (`_own_stay` skips the vertex check). The earlier planner's move is then blocked at the resolver. | I2 ownership, I4 consistency | `D1_stay_overwrites_claim` |
| D1b | When no admissible successor exists, `least_hazard()` ignores reservations and `forbid`. It takes a cell an earlier planner holds. Repair re-plans the same person to the same cell in every round (7 identical resolver results), then converts the move to stay. | I2, I4, I7 (wasted repair work) | `D1b_fallback_overwrite_repeats` |
| D2 | Repair un-reserves and re-plans only the root. Reservations of later-ranked people stay in the table. A later-ranked follower that had claimed the root's cell now makes the root "vacate" for it. In A0 a lower ID steps aside for a higher ID. | I1 precedence, I5 A0 integrity | `D2_genuine_a0_inversion` |
| D3 | People whose move depends on a repaired person keep their plans. The resolver blocks them one link per round. With a ring of vacates, one refusal unravels the chain over all 6 rounds: 7 overwrites, 8 of 9 people stay, and `Pd` is converted to stay without ever being repaired. | I3 dependency invalidation, I4 | `D3_dependency_chain` |

The fixtures were found by a seeded search over random small ASCII scenarios with the traced A0 planner, then minimized (cells removed while the defect stayed). Each fixture has two kinds of tests:

- `test_v1_defect_fixtures_reproduce_and_tracing_is_passive` passes. It shows the frozen v1.0.0 policy produces the documented trace and that tracing does not change decisions.
- Invariant tests (I1, I2, I4, I5) run on both planner versions. The v1.0.0 cases are `xfail(strict=True)`: they must fail. All 10 do. The v1.1.0 cases pass.

## 3. The 238 A0 "yields over the ID rule"

All 238 are vacate events from repair re-plans (A0's "yielded_wait" accounting only credits earlier-ranked takers, which in A0 always have the lower ID).

| Class | Definition | Count |
|---|---|---|
| `label_only` | The yielder's τ = 1 cell equals the counterfactual. The later-ranked reservation did not change the action; the label only records that it existed. | 118 |
| `priority_cascade` | The beneficiary's claim on the yielder's cell came from a plan call in which the beneficiary itself vacated for a person ranked before the yielder. The chain starts at an earlier planner. | 18 |
| `genuine_inversion` | Otherwise. A later-ranked reservation changed the action of an earlier-ranked re-planner. | 102 |

The counts equal the earlier exploratory notes (118 / 18 / 102). The definitions above are now fixed in code.

The same classes for the other variants: A1 3 313 / 25 / 74 (3 412 total); A2 1 969 / 19 / 56 (2 044 total). In A1 and A2 most over-ID yields are label-only, because their order is not the ID order.

## 4. Stale reservations and precedence

| | A0 | A1 | A2 |
|---|---|---|---|
| Decisions | 1 252 | 1 258 | 1 245 |
| Decisions with ≥ 1 repair round | 388 | 491 | 528 |
| Decisions that used all 6 rounds | 194 | 255 | 244 |
| Silent overwrites (`reserve()`) | 5 038 | 8 682 | 10 137 |
| … by a stay | 4 498 | 8 020 | 9 506 |
| … during repair | 3 096 | 5 732 | 5 741 |
| Repair re-plans | 5 703 | 9 116 | 10 566 |
| Re-plans changed by later-ranked reservations (I1 violations) | 1 509 | 2 233 | 1 945 |

Initial-pass plan calls never violate I1 (no later-ranked reservation exists yet). Every I1 violation is in repair.

## 5. Moves converted to stay (attribution)

Cause of the block in the last resolver call before conversion. "Chain end" follows the follow chain to the person who did not move.

| Cause | A0 | A1 | A2 |
|---|---|---|---|
| Chain ends at a person whose vacate was refused | 568 | 1 580 | 1 060 |
| Chain ends at a sheltering person | 244 | 437 | 376 |
| Chain ends at a waiting / yielding person | 49 | 1 | 0 |
| Lowest-ID conflict / cycle / other | 26 | 19 | 25 |
| **Total** | **887** | **2 037** | **1 461** |
| … of which the converted person was never repaired | 573 | 1 575 | 803 |

Most conversions are not the person's own failure. They are D3: the person followed someone whose vacate was refused, and repair never reached it within 6 rounds.

## 6. Crowded-room failures

Closed-loop synthetic episodes (16×20, frozen pilot):

| Cell | Variant | Casualties | Evacuated | Overwrites | Genuine inversions | Conversions | Vacate refused | Decisions at round limit |
|---|---|---|---|---|---|---|---|---|
| multi room, N = 200, stochastic | A0 | 31 | 166 | 1 325 | 42 | 128 | 1 186 | 42 / 72 |
| | A2 | 64 | 136 | 2 369 | 17 | 341 | 2 063 | 40 / 59 |
| open room, N = 100, deterministic | A0 | 19 | 81 | 137 | 1 | 18 | 100 | 8 / 62 |
| | A2 | 38 | 62 | 1 396 | 0 | 136 | 978 | 42 / 62 |

In both cells A2 has about 10× (open room) or 1.8× (multi room) more overwrites and more conversions than A0. The repair loop is saturated in most A2 decisions.

**Queue order.** Leader–follower pairs (the follower's static next cell holds the leader):

| | A0 | A1 | A2 |
|---|---|---|---|
| Pairs | 22 931 | 24 051 | 24 114 |
| Follower planned before its leader | 34.8 % | 61.0 % | 89.5 % |
| Spearman (planning rank, target distance), mean per decision | +0.17 | −0.12 | −0.65 |

Risk priority plans the back of a queue first. The congestion and evacuation components grow with queue position, and the margin falls with queueing delay. A follower planned first claims its leader's cell. The leader must vacate. In a dense queue the vacate fails (the cell ahead is held), the D1 stay overwrites the claim, and D3 spreads the block backward.

**Who dies.** Casualties by stratum at step 0 (frozen A2 risk module):

| Stratum at step 0 | A0 | A1 | A2 |
|---|---|---|---|
| critical | 152 | 198 | 192 |
| high | 13 | 13 | 13 |
| elevated | 12 | 17 | 19 |
| low | 10 | 10 | 9 |
| no viable route | 40 | 40 | 40 |
| **Total** | **227** | **278** | **273** |

The A2 excess (+46) is almost all critical people (+40). Per decision, critical people under A2 had 3 472 refused vacates (A0: 1 317) and 864 conversions (A0: 497). Fairness rule F2 lets a critical person vacate only by a progress move. In dense critical groups this makes refusals, and refusals feed D1/D3. Priority made the people it meant to protect stall.

## 7. A priority yield is not a rescue

An "effective yield over the ID rule" means: in one decision, the beneficiary is predicted to move into the contested cell. It says nothing about the end of the episode. For A2 effective over-ID yields in full episodes (1 499 events): beneficiary finally evacuated 1 352, casualty 124, unresolved 23; the yielder finally evacuated 1 324, casualty 157, unresolved 18. For A0 (228 events): beneficiary evacuated 183, casualty 41. A yield moves risk between people for one step. A rescue claim needs an episode-level counterfactual, which these traces do not give.

## 8. Limitations

- **Observational, policy-generated traces.** Every state in an episode was created by the variant being traced. A0 and A2 see different states after step 0. Differences between variants mix the mechanism with state divergence. Section 6 is descriptive, not causal.
- The counterfactual removes reservations for one plan call only. It does not re-run the rest of the decision, so it measures a local constraint, not the final joint outcome.
- The class definitions in section 3 are one reasonable choice. They are now fixed in code; they were not preregistered.
- The pilot cells are synthetic. Two crowded cells give most of the A0–A2 casualty difference. The pilot was not powered for outcome claims.
- The diagnosis used the v1 pilot cells after their results were known. Those cells and their seeds are therefore **excluded** from any v2 confirmation (namespace `t4-pilot@2/…`).
