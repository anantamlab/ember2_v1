# Overnight final report (2026-11-08)

## 1. Outcome

| Phase | Result | Where |
|---|---|---|
| P0 ledger | done | `docs/overnight/` |
| P1 T4 v1 failure analysis | done. The trace reproduces all 3 755 frozen v1 pilot decisions. A0 238 over-ID yields = 118 label-only / 18 cascade / 102 genuine inversions. Four defects D1, D1b, D2, D3 | `T4_V1_FAILURE_ANALYSIS.md`, PR #33 |
| P2 T4 v1.1.0 planner | done. New identities. The v1 defect fixtures fail on v1.0.0 (strict xfail) and pass on v1.1.0 | `t4_v11.py`, PR #33 |
| P3 v2 preregistration | done. Committed at `07bc90d`, before any v1.1.0 run on a v2 cell or DEV | `docs/P1_T4_RISK_PRIORITY_V2.md` |
| P4 v2 pilot | ran **once**: **`T4_V2_BLOCKED`**. G4 planner health failed; G1–G3 and G5a/b/c passed | `data/baselines/p1-08-t4-v2-pilot.json` |
| P5 DEV comparison | **skipped** by protocol (gate failed) | – |
| P6 verdict | `T4_V2_BLOCKED` | `T4_V2_VERDICT.md` |
| P7 #25 selection prep | done. Proposal record, evidence inventory, protocol freeze and gate. **No selection** (`NO_SELECTION`) | PR #35 |
| P8 #10 provenance | done. Hash-bound provenance, regeneration spot check, leakage audit, bulk-S1 gate with negative tests. **Bulk S1 BLOCKED** | PR #34 |
| P9 cross-review | done (section 4) | this file |

All PRs are drafts and unmerged: #33 (stacked on #32), #34 and #35 (from main `b68535f`).

## 2. T4 in one paragraph

v1.0.0 failed its mechanism check because its repair loop overwrote reservations and re-planned against stale later-ranked claims. v1.1.0 replaces the repair loop with a transactional push/rollback planner. In the independent v2 pilot, v1.1.0 shows **0** precedence violations, **0** A0 ID inversions and **378** verified yields to at-risk people. At-risk immediate progress is higher under A2 than A0 on neutral T1 states. But in dense 16×20 rooms its rollback retries use up the frozen total node budget. Fallback fractions are A1 0.191 and A2 0.185, against a limit of 0.05; v1.0.0 had ≤ 0.009. The gate failed on planner health. There is no DEV result and no claim about casualties.

## 3. Verification

| Branch | Head | Local full suite | CI |
|---|---|---|---|
| `p1/t4-v11-repair` (#33) | see PR | 841 passed, 10 xfailed (strict) | pass at `7cd38dd` |
| `p1/teacher-provenance` (#34) | `7d42a47` | 733 passed | pass |
| `p1/teacher-selection-prep` (#35) | `e55a218` | 724 passed | pending when this file was written |

Resources: CPU only, one process. The v2 pilot used 461 s wall and 47 MB RSS. No Modal, no GPU, no training, no teacher-label generation, no EMBERBench, no hidden or blind data, no secrets. Spend: $0.

## 4. Cross-review findings (P9)

| PR | Finding | Severity | Action |
|---|---|---|---|
| #33 | The pre-freeze smoke checks counted blocked moves, mismatches and conversions, but **not fallbacks**. So the G4 regression was first seen in the confirmation pilot. | process | Before any future freeze, smoke-test every gate metric on non-pilot scenes. |
| #33 | v1.1.0 retries spend the shared `node_budget_total`; push rollbacks = refused vacates (A2 131 407). | design | A v1.2 needs a mechanism-level fix (for example, a bounded retry budget separate from the search budget, or reuse of A* results across retries), a new identity, `t4-pilot@3` and a new preregistration. Do not raise the budget until G4 passes. |
| #33 | `t4_v2_eval` was never run on DEV (by design). Only its guards are tested. | info | If a later pilot passes, review the harness before its single run. |
| #33 | The pilot report is 8.3 MB (the v1 report, 4.2 MB, set the precedent). | info | Reviewer decides whether to keep it in git. |
| #34 | The record digest is a binding, not a signature. A party with repository control can forge consistent records; regeneration limits that. | limit | Signing needs a key decision by the owner. |
| #34 | `REGISTRY` has T0–T3. T4 is not on main. | info | Add T4 when #24 merges. |
| #34/#35 | `VALUE_TEACHERS` is defined in both PRs. | cleanup | Merge the two copies after both PRs land. |
| #35 | The evidence table joins separate preregistered comparisons. It is not a five-way paired comparison. T0 CPU is not recorded in P1-03. | limit | The selection protocol must run one paired comparison. |
| #34 | Checked: the committed DEV manifest passes `_check_manifest` (48 scenarios). | ok | – |

## 5. Decisions for the owner

1. Review #32 and then #33. Decide whether a T4 v1.2 attempt is worth its CPU cost, or whether T4 stays a blocked candidate for #25.
2. Write the #25 selection protocol (fields: `docs/P1_TEACHER_SELECTION_PREP.md` §4). It must say how a blocked T4 is handled before any result.
3. Decide on signing for #10, or accept hash binding plus regeneration.

## 6. Local state

- Worktrees `D:/Coding/ember_v1-p7` and `D:/Coding/ember_v1-p8` hold the P7 and P8 branches. After review they can be removed (`git worktree remove`).
- `t4_v2_pilot` refuses to overwrite its report, and `t4_v2_eval` refuses a blocked pilot. Neither may run again for v1.1.0.
