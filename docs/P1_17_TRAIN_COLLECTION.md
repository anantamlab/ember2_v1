# P1-17 — Full five-teacher TRAIN collection (`p1-17-train-full@1`)

Code: `src/ember_v1/train_collect.py`. Tests: `tests/test_train_collect.py`.
Plan: `data/collection/p1-17-train-full@1/plan_PROPOSED.json` → `plan.json` (FROZEN, same hash).
Calibration: `data/collection/p1-17-train-full@1/calibration.json`.

**Frozen plan hash: `sha256:bff49b473c82d89c041a1aa5e279bff612aec80148f088228cad644bfe07c341`** (189 scenarios, concurrency 8, T2 CPU central 26,102 s, total cap 160,672 s, expected wall ~55 min, upper ~4.5 h).
Geometry: 0 hits against held-out evaluation maps; 3 TRAIN families (`fam-3de0109c7324e29e`, `fam-872f364d45255ac3`, `fam-cb18a7ae2d90994a`) are scale-free near-duplicates of open-room maps of T4 development pilots. They are collected and flagged: a teacher-comparison caveat, not student leakage.
Pilot result summary: `data/collection/p1-12-train-pilot@1/result/attempt-1-summary.json`.

## 1. Scope

- All 189 independent TRAIN scenarios of `p1-02-v1` (64 families, N up to 120, deterministic and stochastic fire).
- 189 matched groups × 5 retained teachers (T0, T1, T2, T3, T4 v1.2.0) = 945 teacher episodes.
- Every teacher of a group gets the same initial scenario and the same exogenous fire randomness (simulator
  seed is part of the scenario; `multiteacher.check_matched_fire` verifies it).
- CPU only. No Modal, GPU, training, DEV, EMBERBench or locked-test data.
- Teachers are unchanged: same pins, same internal budgets (T2 `cpu_budget_ms` 20 s), same caps as the pilot
  (`decision_cpu_s` T2 30 s, others 5 s; `group_wall_s` 7200 s). Nothing was changed to make collection faster.

## 2. Pilot evidence (Phase 1)

| Teacher | Casualties | Evacuated | Unresolved | Casualties + unresolved |
|---|---:|---:|---:|---:|
| T0 | 171 | 293 | 2 | 173 |
| T1 | 124 | 340 | 2 | 126 |
| T2 | 122 | 342 | 2 | 124 |
| T3 | 109 | 355 | 2 | 111 |
| T4 | 121 | 343 | 2 | 123 |

Recomputed from the eight stored group files (`train_collect pilot-summary`); 466 people; T2 CPU 3351.6 s.

| Execution | Result | Archive |
|---|---|---|
| 1 (`ember_v1-pilot3`) | interrupted in group 1 (console closed), 0/8 groups | `D:/Coding/ember-evidence/p1-12-train-pilot@1/execution-1-interrupted/` |
| 2 (`ember_v1-pilot4`, fresh clone) | 8/8 verified, 2/2 regenerations identical | `D:/Coding/ember-evidence/p1-12-train-pilot@1/execution-2-complete/` |

- **Technical integrity: PASS** (execution 2).
- **Protocol compliance: DEVIATION.** Execution 2 ran attempt 1 again in a fresh clone. The guard did not fire
  because the clone had no marker: the guard was bypassed, not satisfied. The owner chose this after a warning.
- Pilot records are immutable and diagnostic only. They are never promoted to training data.
- `MANIFEST.sha256` in the archive covers every file; the original directories are unchanged.

## 3. Pilot overlap

Eight of the 189 scenarios are pilot scenarios. Decision: **fresh collection**. All 189 run in the new
namespace under the new plan and authorization. Pilot groups are not counted, merged or reused. The eight new
groups are compared with the pilot (per-teacher final transition hashes) as an extra cross-run regeneration check.
This is attempt 1 of `p1-17-train-full@1`, not attempt 2 of the pilot.

## 4. Evidence root, checkpoints and resume

The evidence root is `D:/Coding/ember-evidence/p1-17-train-full@1` (plan `storage.root`), outside every checkout.
The plan and the authorization bind it; the runner refuses any other root. A fresh clone therefore sees the same
markers.

| Path | Written | Meaning |
|---|---|---|
| `session-<n>.json` | once, before the first teacher call of session n | plan hash, authorization, commit, limits, pid |
| `started/<sid>.json` | once, before the group runner starts | the group started in this attempt |
| `groups/<sid>.json` | once, by the group runner | sealed matched group (`p1-17-train-full@1` scope) |
| `rows/<sid>.json` | once, after the group file | summary, sha256 of the group file, CPU, peak RSS |
| `interrupted/<kind>-<sid>.json` | once, by a later session | started but no row: terminal, never rerun |
| `regen*/<sid>.json` | once | preregistered regeneration sample |
| `logs/<kind>-<sid>.jsonl`, `log.jsonl` | append, fsync per line | full event logs |
| `session-<n>-end.json` | once | halt reason, peak RSS, RAM waits |
| `evidence.json` | once, when every group is terminal | coverage, regeneration, CPU |
| `progress.json` | replaced every 30 s | status only, not evidence |
| `STOP` | by the operator | graceful stop: no new group starts |

Resume (`run --resume`): every gate runs again; session n ≤ `max_sessions` (4). Finished groups are re-checked
(group file sha256 = row). A started group without a row is recorded `interrupted` and is never rerun. Only groups
that never started run. A retry of any started group needs a new authorization that names it.

## 5. Parallel execution and resources

- One group-runner process per group; it spawns the five teacher processes. At most `max_concurrent_groups`
  runners are live. A new group starts only if host free RAM ≥ `min_free_ram_gib_to_start`.
- Groups start longest-expected first (people, then max_steps), which keeps the makespan short.
- If the supervisor dies, each group runner sees its parent gone and ends its live teachers as `interrupted`.
- Per group the row records worker peak RSS, process CPU per teacher, host minimum free RAM and wall time.
- The concurrency choice and its reasons are in `calibration.json` and `plan.limits`.

## 6. Verification

`python -m ember_v1.train_collect verify --plan <plan.json> --root <root> --pilot-dir <pilot attempt dir> --out <file>`
replays every stored group (`multiteacher.verify_fixture`: observations, transitions, diagnostics, outcomes, chain
hash), checks leakage, scope, plan hash, scenario provenance, teacher fingerprints, caps and file hashes, counts
T2 budget-limited decisions and T2 proposals into burning cells, computes step-0 disagreement, per-teacher totals
over verified complete groups, the pilot overlap comparison, and writes `SHA256SUMS`.

Replay of stored actions is deterministic. Regeneration of teacher decisions is deterministic only without
budget-limited decisions: T2's budget is thread CPU time, so a budget-limited decision depends on host speed and
load. Such groups are reported, never called deterministic.

## 7. Acceptance

Full TRAIN collection succeeds only if all 189 groups are verified complete. Coverage is reported exactly.
`training_eligible` stays false in every collection file; the dataset gate (PR #53) decides eligibility.
