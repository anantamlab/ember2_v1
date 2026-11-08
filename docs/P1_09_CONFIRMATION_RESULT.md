# P1-09 prospective confirmation: result (p1-09-selection@1)

Status: **COMPLETE. Gate decision `RETAIN_T1_NO_IMPROVEMENT`. Receipt is a DRAFT, PENDING REVIEW.** No human review approved anything. This branch does not change the frozen protocol, manifest, audit, teachers, generator, seeds or statistical rules.

| Identity | Value |
|---|---|
| Protocol | `p1-09-teacher-selection`, `sha256:ae72c2cf1a0dae4f46b3a6d158f426af0853bc4100cd15706c74f203a6765c21`, freeze commit `64b2d9873953ba750fd6c08fb84761f1bcd8e0bb` |
| Manifest | `sha256:4e9283b206296d7251514140613f0f8cc234fe9e2dc68877f7a71e906dc5cf9e` (24 families x 3 lineages = 72 scenarios) |
| Isolation audit | `sha256:802b8fa509767c3b3cc881289a81c9285ae254498aa0a1de255fb0e035ea2ef2`, `pass: true` |
| Code commit | `46c445e0a0ffec8e3433696cab3c768da0e5335e` (`main`, descends from the freeze commit) |
| Attempt marker | `sha256:f83212669e18f11c4687d3f94be43dca4c2d33faec23e10caf884cca24756ea7`, started 2026-11-08T10:39:31Z |
| Evidence | `sha256:524e17c2d3b7d560c9556166f06068160f87486fbc6b5e77e213dd681d9478a7`, `complete: true`, 0 errors, stage `completed` |
| Receipt draft | `sha256:d6e94bd4a43631a5138cc110b3b04d75f94a150ce15fe07b89094443918d4022`, `review.status: pending`, `reviewers: []` |

## 1. Files

| Path | Content |
|---|---|
| `data/selection/p1-09-selection@1/attempt.json` | atomic attempt marker (runner output, unchanged) |
| `data/selection/p1-09-selection@1/attempt_log.jsonl` | append-only supervisor log, 374 lines (runner output, unchanged) |
| `data/selection/p1-09-selection@1/evidence.json` | write-once evidence, 72 rows x 5 teachers (runner output, unchanged) |
| `.../outcome/selection_gate.json` | verbatim output of `selection.selection_gate` (frozen code) |
| `.../outcome/statistics.json` | descriptive tables from `experiments/p1_09_confirmation_eval.py` |
| `.../outcome/receipt_DRAFT_PENDING_REVIEW.json` | the gate's `receipt_draft`, verbatim |
| `.../outcome/preflight_post_run.json` | read-only preflight re-run after the run (hashes and regeneration still equal; the runner now refuses a second attempt, as designed) |
| `.../outcome/launcher/` | launcher script, keep-awake helper, stdout, stderr (empty), start and exit stamps |
| `experiments/p1_09_preflight.py`, `experiments/p1_09_confirmation_eval.py` | the preflight and evaluation scripts |

## 2. Preflight (before the marker)

All gates passed before launch. No teacher ran before the marker.

| Gate | Result |
|---|---|
| `main` CI on `46c445e` | `contract (3.11)`, `contract (3.12)`, `update-pip-graph`: success |
| Trusted objects | protocol and manifest load from `64b2d98` on `origin/main`; `verify_frozen_protocol` OK; working-tree files equal the trusted objects |
| Hashes | protocol, manifest, audit equal the committed values; the protocol preregisters the manifest |
| Regeneration | manifest and isolation audit regenerated in memory: equal to the committed objects (24 families, 72 scenarios, audit pass); all 72 scenarios materialize with their recorded hashes |
| Teachers | `check_pins` and `check_registry` empty; identities and seeds equal the protocol (table in section 3) |
| Frozen code | `git diff 64b2d98 46c445e` is empty for `src/ember_v1/{teacher,sim,gen}`, `trust.py`, `provenance.py`, `data/selection`, `data/baselines` |
| Prior attempt | no `evidence.json`, `attempt.json` or `attempt_log.jsonl` in any worktree or any ref (`git log --all`) |
| `run_preconditions` | pass (clean checkout; only `.venv/`, `src/*.egg-info/`, `__pycache__/` ignored) |
| Compute preflight | estimate 8805.7 s of 14400 s (T2 7615 s upper estimate is over the 7200 s replacement cap; recorded, not a stop) |
| Local tests | `tests/test_selection_confirmation.py`, `tests/test_teacher_selection_gate.py`: 128 passed |
| Host | Windows 11, Python 3.14.7, AMD Ryzen AI 9 HX 370 (12 cores, 24 threads), 23 GB RAM (3.1 GB available), 55 GB disk free, AC power, CPU load about 5 %, no competing CPU-heavy process |

Launch: one detached `cmd.exe` (WMI `Win32_Process.Create`, so no tool timeout could end it) ran exactly `python -m ember_v1.teacher.confirmation run --protocol-commit 64b2d9873953ba750fd6c08fb84761f1bcd8e0bb` from `.venv`. A separate helper held `SetThreadExecutionState` (no system sleep) until the run exited. Limits, concurrency (one process per teacher) and code were the frozen ones.

## 3. Teachers

| | Identity | Config hash | Code fingerprint | Policy seed |
|---|---|---|---|---|
| T0 | `fire_aware_shortest_path` 1.0.0 | `sha256:bdc4d358…` | `sha256:8ddf9370…` | none |
| T1 | `congestion_aware_cooperative_planner` 1.0.0 | `sha256:5047e942…` | `sha256:52dc61f4…` | none |
| T2 | `stochastic_fire_forecast_rollout` 1.0.0 | `sha256:302139e1…` | `sha256:afe987d8…` | 20261107 |
| T3 | `dynamic_ant_colony_optimization` 1.0.0 | `sha256:5a8d4985…` | `sha256:82294a3f…` | 20261123 |
| T4 | `risk_based_priority` 1.2.0 | `sha256:79bd3322…` | `sha256:fa66a0f3…` | none |

Full hashes: `outcome/preflight_post_run.json` (`teachers`).

## 4. Supervision and compute

| | T0 | T1 | T2 | T3 | T4 | Total |
|---|---|---|---|---|---|---|
| Scenarios | 72 | 72 | 72 | 72 | 72 | 360 |
| Worker done (UTC) | 10:39:49 | 10:40:27 | 12:58:37 | 10:41:59 | 10:40:53 | |
| Process CPU (s, OS) | 17.2 | 54.7 | 7573.7 | 144.7 | 79.4 | **7869.8** of 14400 |
| Policy CPU (s) | 7.1 | 43.3 | **7555.2** | 134.0 | 68.8 | 7808.3 |
| Decisions | 2906 | 3302 | 3277 | 3307 | 3369 | |
| Mean decision CPU (ms) | 2.4 | 13.1 | 2305.5 | 40.5 | 20.4 | |
| Max decision CPU (ms, record) | 15.6 | 78.1 | **20062.5** | 312.5 | 187.5 | cap 30000 |
| Peak live decision CPU (ms, supervisor) | 15.6 | 62.5 | 20093.8 | 281.2 | 93.8 | cap 30000 |
| Budget-limited decisions | 0 | 0 | 24 | 0 | 5 | |
| Episode failures | 0 | 0 | 0 | 0 | 0 | |

Wall time 8346.5 s (2 h 19 min) of 14400 s. No breach, no abort, all exit codes 0. Spend: local CPU only, $0 of the $15 cap. T2 took 96 % of the CPU. Its largest decision used 67 % of the 30 s cap.

## 5. Primary analysis (paired against T1; unit = family)

Casualty difference = candidate − T1 (negative favours the candidate). Family-cluster bootstrap: 10000 resamples, seed 20261109, 95 % CI. Exact two-sided family sign test. Holm over {T0, T2, T3, T4}, familywise alpha 0.05.

| | T0 | T2 | T3 | T4 |
|---|---|---|---|---|
| Total casualties (T1 435) | 608 | 428 | 460 | 419 |
| Mean paired casualty difference per scenario | +2.403 | −0.097 | +0.347 | −0.222 |
| 95 % family-cluster CI | [+1.083, +4.194] | [−0.208, −0.014] | [−0.222, +1.056] | [−0.722, +0.097] |
| Families fewer / equal / more casualties | 2 / 4 / 18 | 4 / 20 / 0 | 6 / 12 / 6 | 5 / 16 / 3 |
| Scenarios fewer / equal / more casualties | 2 / 44 / 26 | 4 / 68 / 0 | 8 / 55 / 9 | 6 / 62 / 4 |
| Raw sign-test p | 0.000402 | 0.125 | 1.0 | 0.727 |
| Holm-adjusted p | 0.00161 | 0.375 | 1.0 | 1.0 |
| Direction favours candidate | no (harmful) | yes | no (tie) | yes |
| CI upper bound < 0 | no | **yes** | no | no |
| Holm p < 0.05 | yes, harmful direction | no | no | no |
| **Superior** | **no** | **no** | **no** | **no** |

Holm order: T0 0.000402 x 4 = 0.00161; T2 0.125 x 3 = 0.375; T4 0.727 x 2 → 1.0; T3 1.0.

## 6. Safety and replacement compute

| | T1 | T0 | T2 | T3 | T4 |
|---|---|---|---|---|---|
| People | 2835 | 2835 | 2835 | 2835 | 2835 |
| Evacuated | 2329 | 2211 | 2333 | 2346 | 2340 |
| Unresolved | 71 | 16 | 74 | 29 | 76 |
| Casualties + unresolved | 506 | 624 | 502 | 489 | 495 |
| Safety (c+u ≤ 506 and evacuated ≥ 2329) | baseline | **FAIL** | pass | pass | pass |
| Policy CPU ≤ 7200 s (replacement cap) | | pass | **FAIL (7555.2 s)** | pass | pass |
| Qualifies (superior and safe and feasible) | | no | no | no | no |

## 7. Every scenario with more casualties than T1

The protocol defines no severity threshold. This list has every scenario where the candidate has more casualties than T1. Δunres = candidate − T1 unresolved. Full records: `outcome/statistics.json` (`vs_T1.*.worse_scenarios`).

**T2: none.**

**T4 (4):** `g-0ef69adf39311113` +3 (120 people); `g-42a7dc914766e212` +1 (Δunres −3); `g-b17bf2e574772429` +1 (Δunres −1); `g-c1c21ac1a2f1e04f` +1.

**T3 (9):** `g-ab44068a942771ae` +35 (120 people, Δunres −36); `g-b17bf2e574772429` +7 (Δunres −4); `g-0ef69adf39311113` +6; `g-e28acaaf4ed6dddd` +5; `g-5baee3fbb0800d25` +3; `g-42a7dc914766e212` +2 (Δunres −2); `g-2948258c63a8bd50` +1; `g-c69617769bc287b7` +1; `g-e1ba051b6e8fcb95` +1.

**T0 (26):** largest `g-ab44068a942771ae` +43 (Δunres −41), `g-8f028c021d785b77` +15, `g-42a7dc914766e212` +14, `g-58fe9d5c43438e25` +12, `g-2948258c63a8bd50` +9, `g-b17bf2e574772429` +9, `g-e28acaaf4ed6dddd` +9, `g-0ef69adf39311113` +8; 18 more with +1 to +7.

Observation: `g-ab44068a942771ae` (family `fam-50833fbbc2256e41`) is the worst case for T0 and T3. There, both convert T1's unresolved people into casualties (T1 ends with about 40 more unresolved). `g-0ef69adf39311113` is worse than T1 for T0, T3 and T4.

## 8. Decision

Per the frozen rules: no candidate is superior. T0 is significantly worse and unsafe. T2 is safe, has zero worse families and a CI that excludes zero, but fails Holm (0.375) and the replacement-compute cap (7555.2 s > 7200 s). T3 and T4 are safe but not superior. **Decision: `RETAIN_T1_NO_IMPROVEMENT` (frozen no-improvement rule).**

This is a proposal from the gate. The receipt `outcome/receipt_DRAFT_PENDING_REVIEW.json` has `review.status: pending` and no reviewers. S1 (`s1.verify_receipt`) refuses it until named reviewers approve it and commit it on `main`. Bulk S1 is still not authorized (issue #10 dependencies).

## 9. Limits of this result (descriptive, no rule change)

- Power: an exact two-sided sign test with k informative families has minimum p = 2 / 2^k. T2 had 4 informative families (all favourable): p = 0.125 is the smallest possible value. At Holm position 2 it needed raw p < 0.0167, so at least 7 informative families, all favourable. The design could not confirm a small but consistent T2 gain on this set. Any new test needs a new, separately frozen protocol.
- T2 uses 175 times the policy CPU of T1 for 7 fewer casualties in 2835 people.
- One run per identity on one host (Windows, Python 3.14.7). CPU times are this host's.
