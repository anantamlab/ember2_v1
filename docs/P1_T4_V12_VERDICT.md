# T4 v1.2.0 verdict: confirmation pilot `t4-pilot@3`

**Decision: `T4_READY_FOR_INDEPENDENT_SELECTION_EVALUATION`.** T4 is **not selected**. This result does not authorize a DEV comparison, S1 data or training. It only makes T4 v1.2.0 a candidate that the independent selection protocol (issue #25) may admit through its own frozen T4 entry rule and evaluate on its own prospective, family-disjoint data.

## 1. Run record

| Item | Value |
|---|---|
| Protocol | `docs/P1_T4_V12_NODE_BUDGET.md` (preregistered) |
| Freeze commit | `b7ff040f8abe065ced5ed1331c0cd6854bd878a1` (report `environment.git_sha`, `src_dirty: false`) |
| Command | `python -m ember_v1.teacher.t4_v12_pilot` (run once; no rerun) |
| Report | `data/baselines/p1-08-t4-v12-pilot.json` (`evidence_class: prospective_confirmation`), LF-normalized `sha256:b64b5589a5bec04fadf3dc8d8f0008f75fc3d9d7d0ab09e53d8f8f5075ed2655` |
| Reproducibility | `tests/test_teacher_t4_v12.py` recomputes the gate from the committed rows and checks the identities, cells and freeze commit |
| Cells | 74 defined, 74 run, 0 rejected, 0 errors |
| Resources | wall 483 s, process CPU 435 s, peak RSS 49.5 MB; one CPU process, no GPU, no network |
| DEV use | frozen-T1 replay for public workload and reference timing only; no T4 decision on DEV |

## 2. Gate (`gate_v2`, unchanged)

| Check | Value | Limit | Pass |
|---|---|---|---|
| G1 bounded | complete; 0 predicted blocked, 0 blocked moves, 0 prediction mismatches, 0 verification mismatches; max decision CPU 234.4 ms; RSS 49.5 MB | ≤ 2 500 ms; ≤ 1 024 MB | yes |
| G2 DEV CPU | upper estimate 437 s | ≤ 7 200 s | yes |
| G3 cap hits | 0.0 for A0, A1, A2 | ≤ 0.05 | yes |
| G4 A0 | fallback 1 268 / 53 836 = 0.024; budget-limited 5 / 1 765 = 0.003 | ≤ 0.05 each | yes |
| G4 A1 | fallback 1 532 / 55 036 = 0.028; budget-limited 7 / 1 797 = 0.004 | ≤ 0.05 each | yes |
| G4 A2 | fallback 1 615 / 55 026 = 0.029; budget-limited 10 / 1 792 = 0.006 | ≤ 0.05 each | yes |
| G5a precedence | 0 audit violations, 0 inversions (closed loop and neutral) | 0 | yes |
| G5b A0 integrity | 0 ID inversions, 0 unexplained over-ID yields (2 611 over-ID yields, all chains from a lower ID) | 0 | yes |
| G5c priority effect | 304 verified yields to at-risk people; at-risk progress A2 9 772 ≥ A0 8 267 (18 604 at-risk states, 1 776 neutral decisions) | ≥ 1; A2 ≥ A0 | yes |

Failed list: none. v1.1.0 on `t4-pilot@2` had A2 fallback 0.185 and budget-limited 0.071 (G4 failed). The pilot values are higher than the exploratory smoke (A2 0.015 and 0.000) but stay below the limits.

## 3. Reported outcomes (not gated, not evidence)

42 full episodes per variant (2 313 people): A0 1 590 evacuated / 673 casualties / 50 unresolved; A1 1 602 / 661 / 50; A2 1 601 / 662 / 50. The pilot has too few independent cells for an outcome claim. These totals do not compare T4 with T1 and do not support selection.

## 4. What this permits and what it does not

- Permitted: issue #25 may define, before its own protocol freeze, a T4 entry rule that names T4 v1.2.0 (`sha256:79bd3322b9cf7ccd25928f3a70502089732496af5427f88f00ecd2ff6ba12504`) and this report hash, and may then evaluate it with T0–T3 on the prospective confirmation set.
- Not permitted: calling T4 selected; any DEV comparison under this protocol; using this pilot as selection evidence; S1 data; changing v1.2.0 or rerunning `t4-pilot@3`.
- T4 v1.0.0 (`BLOCKED_BY_MECHANISM_CHECK`) and v1.1.0 (`T4_V2_BLOCKED`) stay blocked records.
