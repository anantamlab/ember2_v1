# T4 risk priority v2: v1.1.0 planner, confirmation pilot and DEV comparison (preregistration)

**Status:** PREREGISTERED. This document, `src/ember_v1/teacher/t4_v2_pilot.py`, `src/ember_v1/teacher/t4_v2_eval.py` and `tests/test_teacher_t4_v2.py` are committed **before** any v1.1.0 policy runs on a v2 pilot cell or on DEV. After that commit, the planner, the identities, the cells, the gate and the statistics do not change. The v1 protocol (`docs/P1_T4_RISK_PRIORITY.md`), the v1.0.0 policies and hashes, the v1 pilot report and its `BLOCKED_BY_MECHANISM_CHECK` decision, and the v1 diagnosis stay as they are.

## 1. Why v2

The v1.0.0 pilot failed G5. The diagnosis (`docs/overnight/T4_V1_FAILURE_ANALYSIS.md`) found four planner defects (D1, D1b, D2, D3). They come from the repair loop, not from the risk model. v1.1.0 replaces the repair loop with a transactional reservation planner. The risk model, its weights, strata, urgency thresholds, fairness rules and the priority orders are **the v1.0.0 values, unchanged**. No value was chosen from v1 pilot or DEV outcomes.

## 2. Policies (frozen identities)

| Name | Class | Priority | Config hash |
|---|---|---|---|
| T4 (A2) | `T4RiskPriorityV11` | risk | `sha256:67a3517de6fd109d128c0cf31836867fa0a1ec30cb09844e7a1541acae237ad4` |
| T4-A0 | `T4NoPriorityV11` | none (ID order) | `sha256:97e134923cd63d55b79246c12e6e10b3800c4706bb2354e40fa8cc1588f675db` |
| T4-A1 | `T4DistancePriorityV11` | fire distance | `sha256:d1c5acb82cfb693a0d1e6898c3af5293b7f405ada7c94cfe794b405cb2b7f326` |
| T1 | `T1CongestionAware` | – | as in `data/baselines/p1-04-t1-dev-v1.json` |

Planner `t4-priority-planner@1.1.0`. Config difference from v1.0.0: `repair_rounds` removed, `push_depth_limit: 256` added (a resource bound), descriptive keys for the new planner. `tests/test_teacher_t4_v11.py` pins the hashes and the exact config diff.

## 3. v1.1.0 invariants

| | Invariant | Check |
|---|---|---|
| I1 | Precedence: a plan never depends on a reservation of a later-ranked root. | `_audit` per root; `precedence_audit_violations` |
| I2 | Ownership: no reservation overwrites another owner. | `reserve()` raises `ReservationConflict` |
| I3 | Dependency: a failed vacate rolls back the whole push chain. | rollback tests (D2, D3) |
| I4 | Consistency: the planner's predicted outcomes equal the public resolver. | verification resolve; `verification_mismatches` |
| I5 | A0 integrity: under ID order, nobody yields to a higher ID except through a chain that starts at a lower ID. | `a0_id_inversions`, `a0_unexplained_over_id_yields` |
| I6 | Fairness: F1/F2 of v1.0.0 (protected people vacate only by a progress move). | unit tests |
| I7 | Bounded work: push depth ≤ `push_depth_limit`; budget limits as v1. | budget tests |
| I8 | Determinism and order invariance of the observation encoding. | unit tests |

## 4. Pilot cells (synthetic, independent)

- Namespace `t4-pilot@2/...`, scenario seeds `derive_seed("t4-pilot@2/sim", ...)`. These are not generator plans, families or splits.
- 74 cells (`pilot_cells_v2`):
  - base: 2 grids (16×20, 48×64) × 4 topologies × N ∈ {1, 10, 50, 100, 200}, stochastic p = 0.5, 3 exits (40);
  - base deterministic: 2 grids × {open room, corridor, multi room} × N ∈ {10, 100} (12);
  - exits: 48×64, {open room, multi room} × {1, 6} exits, N = 50 (4);
  - bottleneck: 16×20 corridor, 1 exit, N ∈ {50, 100} × 2 fire models (4);
  - high ID risk: 16×20, 3 topologies × 2 fire models, N = 50, nearest-to-fire gets the highest ID (6);
  - near entrapment: 16×20 corridor, N ∈ {10, 50} × 2 fire models (4);
  - unreachable: multi room, 2 grids × 2 fire models, N = 50, half in sealed rooms (4).
- 16×20 cells run full episodes. 48×64 cells run the first 12 decisions (v1 `PREFIX_DECISIONS`).
- **Exclusion (fixed before the run).** A cell is rejected, recorded and not replaced if its content hash (scenario without ID and seed) or its seed equals any v1 pilot scenario, if its layout equals one of the two crowded v1 cells inspected in the diagnosis, or if it duplicates another v2 cell. The pre-run check (scenario construction only, no policy) rejected 0 of 74.

## 5. Pilot measurements and gate

Per cell: A0, A1 and A2 closed loop. Then **neutral paired states**: frozen T1 runs the cell closed loop; on every T1 observation A0 and A2 decide but do not act. These states are not created by A0 or A2.

Immediate progress of a person = static exit distance at the start minus at the public-resolver end cell of the decided joint action (0 at an exit; end cell without exit path counts as no progress). At-risk = stratum `critical` or `high` from the frozen risk module on the same observation.

Gate. **All** must pass, else `T4_V2_BLOCKED` with the failed list:

| Check | Pass condition |
|---|---|
| G1 bounded | pilot complete in the wall limit (3 600 s); 0 predicted blocked, 0 resolver-blocked moves, 0 prediction mismatches; max decision CPU ≤ budget + 500 ms; peak RSS ≤ 1 024 MB (v1 values) |
| G2 DEV CPU | upper DEV comparison CPU estimate ≤ 7 200 s (v1 cost model, v1 limit) |
| G3 cap hits | predicted budget-cap fraction ≤ 0.05 per variant (v1) |
| G4 planner health | fallback ≤ 0.05 of routable person decisions and budget-limited ≤ 0.05 of decisions, per variant (v1) |
| G5a precedence | 0 precedence audit violations and 0 precedence inversions in all closed-loop and neutral decisions |
| G5b A0 integrity | A0 rows exist; 0 A0 ID inversions; 0 A0 over-ID yields that are not chains with origin ID < yielder ID |
| G5c priority effect | on neutral states: (i) ≥ 1 verified case where an at-risk person H has more immediate progress under A2 than under A0 and a lower-stratum person L yields to H the cell where H ends; and (ii) total at-risk immediate progress under A2 ≥ under A0 |

Exit-reassignment links (L changes exit so that H takes L's A0 exit) are reported but do not count for G5c(i). Any exception in a cell is recorded, marks the pilot incomplete and fails G1. Outcomes (casualties, evacuations) of pilot episodes are **reported, not gated**: the pilot has too few independent cells for outcome claims.

## 6. Pilot run rules

- `python -m ember_v1.teacher.t4_v2_pilot` runs **once**, one CPU process, no GPU, no network. Report: `data/baselines/p1-08-t4-v2-pilot.json`. The entry point refuses to overwrite it.
- A failed gate is a result. The v1.1.0 planner, the identities and this sample are not changed, tuned or re-run after the result. A new attempt needs a new planner identity, a new namespace and a new preregistration.
- An interrupted run (crash or reconnect) is not restarted silently. Its state is recorded in the overnight ledger, and the decision is made there with the reason.

## 7. DEV comparison (only if the pilot gate passes)

- `python -m ember_v1.teacher.t4_v2_eval` refuses unless the pilot report has `T4_V2_FULL_DEV_FEASIBLE` for the identities of section 2. It runs once and refuses to overwrite `data/baselines/p1-08-t4-v2-dev.json`.
- Data: the 48 DEV scenarios of plan `p1-02-v1` (`t1_eval.load_dev`). No train, calibration, locked test, blind or EMBERBench data. T1, A0, A1, A2 run every scenario with its own simulator seed.
- T1 reproduction: every T1 final transition hash must equal the committed T1 DEV baseline. A mismatch makes the comparison invalid.
- **Primary endpoint:** A2 − T1 casualties per scenario.
- Secondary: A2 − A0, A2 − A1, A1 − A0 and the other paired metrics of v1 (`t4_eval.PAIRED_METRICS`). Secondary results are descriptive.
- Statistics (v1 rule, `t4_eval.verdict`): family-cluster bootstrap (10 000 resamples, 95 %, seeds `t4v2-...@2`) and the family sign test. A result is `better` or `worse` only if the CI excludes 0 **and** the sign test p < 0.05 in the same direction. Otherwise `no_evidence_of_difference`.
- Wall limit: `min(6 h, max(900 s, 1.5 × upper pilot estimate))`. An incomplete run gives no verdict.

## 8. Verdicts

| Condition | Verdict |
|---|---|
| Pilot gate fails | `T4_V2_BLOCKED` (no DEV run) |
| DEV incomplete or T1 reproduction fails | `T4_V2_DEV_INVALID` |
| Primary `better` | `T4_V2_DEV_FEWER_CASUALTIES_THAN_T1` |
| Primary `worse` | `T4_V2_DEV_MORE_CASUALTIES_THAN_T1` |
| Otherwise | `T4_V2_DEV_NO_EVIDENCE_OF_DIFFERENCE` |

## 9. Limitations (stated in advance)

- DEV was used by P1-03 … P1-08. It is a development comparison, not a fresh confirmation set. No verdict here supports a claim of generalization; that needs a locked test or a new split under its own protocol.
- v1.1.0 was designed after the v1 pilot failure. The v1 cells informed the design. They are excluded from v2 (section 4).
- 48 scenarios in few families give low power. `no_evidence_of_difference` is not evidence of equivalence.
- G5c is a mechanism check on single decisions. It does not show episode-level rescue.
