# T4 v2 verdict (overnight P4–P6)

**Verdict: `T4_V2_BLOCKED`.** The preregistered v2 confirmation pilot (`docs/P1_T4_RISK_PRIORITY_V2.md`) failed gate check **G4 planner health**. Under protocol section 6 there is **no DEV comparison** (P5 skipped), no change to v1.1.0, and no rerun of this sample.

## 1. Run record

| Item | Value |
|---|---|
| Report | `data/baselines/p1-08-t4-v2-pilot.json` (one run) |
| Code | commit `07bc90dcccb38657d73c556a18d8e57df42f9375` (the preregistration commit), `src_dirty: false` |
| Identities | A0 `sha256:97e1…75db`, A1 `sha256:d1c5…b326`, A2 `sha256:67a3…37ad` (v1.1.0, as preregistered) |
| Cells | 74 defined, 74 run, 0 rejected, 0 errors |
| Resources | wall 461 s, process CPU 453 s, peak RSS 47 MB, 1 process, no GPU, no network, $0 |
| DEV use | frozen T1 replay for workload and timing only; no T4 decision on DEV |

## 2. Gate

| Check | Result | Key numbers |
|---|---|---|
| G1 bounded | pass | 0 predicted blocked, 0 blocked moves, 0 prediction and 0 verification mismatches; max decision CPU 203 ms (limit 2 500); RSS 47 MB |
| G2 DEV CPU | pass | upper estimate 472 s (limit 7 200 s) |
| G3 cap hits | pass | predicted cap fraction 0.0 for all variants on the DEV profile |
| **G4 planner health** | **fail** | fallback fraction A0 0.046, **A1 0.191, A2 0.185** (limit 0.05); budget-limited decisions A0 0.017, **A1 0.074, A2 0.071** (limit 0.05) |
| G5a precedence | pass | 0 audit violations, 0 precedence inversions (closed loop and 1 792 neutral decisions) |
| G5b A0 integrity | pass | 0 A0 ID inversions; all 2 941 A0 over-ID yields are chains from a lower ID |
| G5c priority effect | pass | 378 verified yields to an at-risk person; at-risk immediate progress A2 9 891 ≥ A0 8 344 |

The mechanism checks that failed in v1.0.0 (G5) pass in v1.1.0. The new planner fails a health check that v1.0.0 passed (v1 pilot: fallback ≤ 0.009, budget-limited 0.0).

## 3. Where G4 fails (descriptive)

Fallback fraction of routable person decisions:

| Grid, N | A0 | A1 | A2 |
|---|---|---|---|
| 16×20, N < 100 | 0.017 | 0.019 | 0.024 |
| 16×20, N = 100 | 0.094 | 0.234 | 0.264 |
| 16×20, N = 200 | 0.087 | 0.522 | 0.485 |
| 48×64, all N | ≤ 0.028 | ≤ 0.026 | ≤ 0.026 |

- All budget limits are `node_budget_total` (A0 32, A1 139, A2 130 decisions). When the shared node budget (20 000, the v1.0.0 value) is used up, the remaining people get the fallback action.
- Push rollbacks equal refused vacates: A0 40 719, A1 116 226, A2 131 407. Each rollback forbids a cell and re-plans, so each one spends more A* nodes. A non-ID planning order (A1, A2) causes about 3× more rollbacks than ID order (A0).
- In the densest A2 cells (16×20, N = 200): multi room 0.63, open room 0.49, irregular 0.35, corridor 0.00.

Interpretation (exploratory, not confirmed): the transactional retry loop of v1.1.0 is correct (G1, G5) but not budget-efficient in dense rooms. In v1.0.0 the repair loop converted moves to stay; in v1.1.0 the retries consume the frozen total node budget. The smoke checks before the freeze counted blocked moves, mismatches and conversions, but did not count fallbacks. That was a gap in pre-freeze testing.

## 4. Reported outcomes (not gated, not evidence)

Full-episode cells (42 per variant, 2 185 people):

| | A0 | A1 | A2 |
|---|---|---|---|
| Evacuated | 1 550 | 1 522 | 1 544 |
| Casualties | 553 | 570 | 564 |
| Unresolved | 82 | 93 | 77 |
| Critical-at-step-0 casualties | 261 | 278 | 283 |

These are policy-generated closed-loop traces on synthetic cells. The pilot is not powered for outcome claims, and G4 says the A1/A2 runs used the fallback action often. No conclusion about risk priority follows from these numbers.

## 5. What this does not permit

- No DEV run of v1.1.0. No claim that v1.1.0 is better or worse than T1, A0 or A1.
- No change of `node_budget_total`, retry rules or any other v1.1.0 value after this result, and no rerun of `t4-pilot@2`.
- Any next attempt needs a new planner identity (for example a budget-aware retry rule), a new pilot namespace (`t4-pilot@3`), new cells, and a new preregistration committed before its run. A fix must be motivated by the mechanism (retry cost), not by tuning the budget until G4 passes on these cells.

## 6. Status for issue #24 and the five-teacher gate (#25)

T4 has no valid DEV evidence: v1.0.0 is `BLOCKED_BY_MECHANISM_CHECK`, v1.1.0 is `T4_V2_BLOCKED`. For #25, T4 stays a candidate without DEV evidence. The selection gate cannot pass until T4 has evidence or a frozen protocol states how a blocked candidate is handled.
