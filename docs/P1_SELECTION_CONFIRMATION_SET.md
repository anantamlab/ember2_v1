# P1-09: prospective confirmation set `p1-09-selection@1`

Status: **prepared, not run.** No teacher has run on any scenario of this set. The protocol in this PR is a freeze *candidate*: it becomes the freeze only when a reviewed commit with it reaches `origin/main`. Teacher selection, bulk S1 labels and Qwen training are not started.

Code: `src/ember_v1/teacher/confirmation.py`, `src/ember_v1/teacher/selection.py`. Tests: `tests/test_selection_confirmation.py`, `tests/test_teacher_selection_gate.py`.

Revision 2 (review of 52df90e): Holm multiplicity correction, live run supervision with an attempt marker, a clean-checkout check that includes untracked and ignored files, and a pilot-map near-duplicate geometry screen. The 24 families and 72 scenarios did not change; the plan rules, the plan hash and therefore the manifest, audit and protocol hashes changed. No outcome was observed before or during this revision.

## 1. Files

| File | Content | Hash |
|---|---|---|
| `data/selection/p1-09-selection@1/manifest.json` | 24 families x 3 lineages = 72 scenarios (IDs, seeds, hashes; no scenario bodies) | `manifest_hash` `sha256:4e9283b206296d7251514140613f0f8cc234fe9e2dc68877f7a71e906dc5cf9e` |
| `data/selection/p1-09-selection@1/isolation_audit.json` | overlap of every key with every earlier set; pilot-map geometry screen | `audit_hash` `sha256:802b8fa509767c3b3cc881289a81c9285ae254498aa0a1de255fb0e035ea2ef2` |
| `data/selection/p1-09-selection@1/protocol.json` | frozen protocol candidate `teacher-selection-protocol@1.2.0` | `protocol_hash` `sha256:ae72c2cf1a0dae4f46b3a6d158f426af0853bc4100cd15706c74f203a6765c21` |
| `data/selection/p1-09-selection@1/attempt.json` | **absent.** Created exclusively (`open(..., "x")`, fsync) by the one confirmatory run before its first scenario. | — |
| `data/selection/p1-09-selection@1/attempt_log.jsonl` | **absent.** Append-only, fsynced event log of that attempt. | — |
| `data/selection/p1-09-selection@1/evidence.json` | **absent.** Written once (`open(..., "x")`) at the end of the attempt, also when it is aborted. | — |

Rebuild (deterministic; refuses to overwrite a file with other content): `python -m ember_v1.teacher.confirmation build`. A test rebuilds the manifest and the audit and compares them to the committed files as JSON.

## 2. Generator specification (frozen)

| Item | Value |
|---|---|
| Plan / namespace | `p1-09-selection@1`, split `selection_confirmation`, `conf_version` `p1-09-confirmation-set@1.1.0` |
| Seed namespace | `p1-09-selection@1/seeds`, master seed `20261108`; family seed = `derive_seed(master, namespace, "family", slot)`, lineage seed = `derive_seed(family_seed, "lineage", j)` |
| Generator | `gen-1.0.0`, `DEFAULT_SPEC` (hash in `spec_hash`) |
| Pool | slots 0 to 71 in order; stop at 24 accepted families |
| Family rejection | exact topology duplicate, or near-duplicate / wall near-duplicate of any of the 120 reference families from `historical_sets()` or an earlier accepted family |
| Pilot-map exclusion | the family map duplicates a T2/T3/T4 pilot map (section 3.2); the whole family is excluded and recorded; the next slot is tried |
| Lineage exclusion | a lineage fails to generate, or any of its 3 lineages collides on an isolation key; the whole family is excluded and recorded |
| Result | 25 slots used; 1 family excluded (`fam-689c192ea688783c`, slot 5, `lineage_generation_failed`); 0 pilot-map exclusions |
| Mix | archetypes corridor 7, irregular 6, multi_room 6, open_room 5; targets feasible 46, difficult 18, unreachable 8 |

## 3. Isolation audit

### 3.1 Identity keys

Keys: `family_id`, `lineage_id`, `scenario_hash`, `core_hash`, `scenario_seed`. Sets (equal to `selection.REQUIRED_ISOLATION`): `p1-02-v1/train`, `dev`, `calibration`, `locked_test` (sealed: checked by its committed index commitment, not opened), `t2-pilot@1`, `t3-pilot@1`, `t4-pilot@1`, `t4-pilot@2`, `t4-explore@v12`, `t4-pilot@3`. Every overlap is empty; all keys are unique inside the set.

### 3.2 Pilot-map geometry (new)

All 340 pilot maps are regenerated from the committed cell lists (t2-pilot@1 24, t3-pilot@1 42, t4-pilot@1 52, t4-pilot@2 74, t4-pilot@3 74, t4-explore@v12 74; 340 distinct topologies). Only walls and exits are compared; no teacher outcome is read. `pilot_match` excludes a family if any of these holds against any pilot map:

| Rule | Test |
|---|---|
| `topology_duplicate` | equal topology hash (invariant under the 8 grid symmetries and exit renaming) |
| `near_duplicate`, `wall_near_duplicate` | the frozen generator rules (scale 1.3, aspect 1.15, wall MAD 0.03 on the 16x16 coarse grid, exit chamfer 0.1, wall rule only at >= 5 % walls) |
| `scale_free_near_duplicate`, `scale_free_wall_near_duplicate` | the same thresholds over all 8 rotations/reflections, **without** the scale and aspect gates (catches a rescaled copy) |

Result: 8160 family x pilot pairs, 173 within the scale/aspect gates, **0 hits**. Nearest pilot map of any family (scale-free): wall MAD 0.043 (threshold 0.03), `t4-pilot-base-corridor-48x64-10-stochastic-3`. The audit lists the nearest map per family. `pass: true` requires no hit.

Not checked (recorded in the audit): `blind_test` (secret seeds; another root; private contents stay inaccessible); dynamic pilot content (people, ignitions, fire model) beyond the exact core hash; local edits below the wall threshold, crops, translations and partial overlaps.

## 4. Preregistered rules (in `protocol.json`)

| Rule | Value |
|---|---|
| Eligible identities | T0, T1, T2, T3, T4 v1.2.0 (exact name, version, config hash from the registry). T4 v1.0.0 (G5) and v1.1.0 (G4) stay blocked records. |
| Baseline | T1 |
| Primary | casualties, paired against T1, unit = family; family-cluster bootstrap, 10000 resamples, seed 20261109, 95 % CI; exact two-sided family sign test |
| Multiplicity | Holm step-down over the 4 hypotheses T0, T2, T3, T4 vs T1, familywise alpha 0.05 (`primary.multiplicity`). The family is every eligible non-baseline candidate, fixed by the protocol validator. |
| Superiority | mean paired casualty difference favours the candidate (more families with fewer casualties) AND the 95 % CI upper bound < 0 AND Holm-adjusted p < 0.05 |
| Safety | casualties + unresolved: no net deterioration; evacuated: no net deterioration |
| Compute | total CPU cap 14400 s; per-decision cap 30000 ms; replacement CPU cap 7200 s; wall-time cap 14400 s; USD cap $15; enforcement `live_supervisor_fail_closed` |
| Tie | lower CPU, then simplicity T0, T1, T3, T4, T2 |
| Runs | one confirmatory attempt; any error, missing row, drift or supervisor abort stops selection |
| Seeds | simulator seed = scenario seed (matched across teachers); policy seeds T2 `20261107`, T3 `20261123` (the class defaults; checked against the classes); a fresh teacher instance per episode |
| Excluded data | locked_test, blind_test, calibration, emberbench |

The gate reports raw and Holm-adjusted p-values per candidate (`analyses[c].sign_test_p`, `analyses[c].sign_test_p_holm`, and `multiplicity`). It also checks that the evidence manifest is the preregistered one, that the supervisor completed under the protocol limits, and that the attempt marker names this protocol and code commit.

Holm rule: sort the 4 raw p-values ascending; adjusted p_(i) = max over j <= i of min(1, (4 - j + 1) p_(j)). Example: one candidate with 18 of 24 families better has raw p = 0.023 and adjusted p = 0.091: not superior.

## 5. Runner, preconditions and supervision

`python -m ember_v1.teacher.confirmation run --protocol-commit <sha>` runs only if all of these hold, else it stops before any marker or teacher run:

1. The protocol and the manifest load from that commit, and the commit is on the trusted ref (`origin/main`).
2. The protocol is frozen and its hash verifies. The manifest hash is the preregistered one.
3. The compute preflight passes (estimated total <= `cpu_s_cap`).
4. The checkout is clean (`tree_problems`, section 5.1); `HEAD` descends from the protocol commit.
5. No registry pin or structure drift (`check_pins`, `check_registry`).
6. `evidence.json`, `attempt.json` and `attempt_log.jsonl` do not exist.

Then the runner creates `attempt.json` atomically and starts one spawned worker per teacher. Each worker regenerates the scenarios, checks their hashes and streams every result to the supervisor. The supervisor polls every 0.25 s and stops **all** workers (terminate, then kill after 5 s, then reap) at the first of:

| Stage | Trigger |
|---|---|
| `per_decision_cpu_cap` | CPU of the current decision of a worker > 30000 ms (OS process CPU now minus the CPU at decision start) |
| `total_cpu_cap` | sum of worker process CPU > 14400 s (process CPU includes the simulator, so it is >= policy CPU) |
| `wall_time_cap` | elapsed wall time > 14400 s |
| `worker_error`, `worker_crashed` | a worker raises, or exits without a done message or with a non-zero code |
| `worker_unresponsive` | only if the OS CPU reading is unavailable: worker heartbeat older than 60 s |
| `interrupted`, `supervisor_error` | Ctrl-C or a supervisor fault |

Each result records `decisions` and `budget_limited_decisions` (decisions where the teacher's frozen internal search budget was hit: T1 `budget_exhausted`, T2/T3/T4 `budget.limit_hit`, T3 `status`). This count is descriptive; the gate reports it per candidate and does not stop on it. A policy exception (`failure`), an invalid joint action map (the simulator requires exactly the active IDs with supported actions) or a per-decision CPU breach stops selection.

CPU is read from the OS (`GetProcessTimes` on Windows, `/proc/<pid>/stat` on Linux); a worker heartbeat thread also reports its own CPU and exits the worker if the supervisor process is gone. After an abort the evidence is still written, with `complete: false`, the stage, the partial rows and the supervision report. The gate refuses it (`NO_SELECTION`). The CLI exits non-zero.

**No second attempt.** `attempt.json` stays after a crash, an abort or a power loss, and the preconditions refuse any later run. Re-execution needs a separately documented scientific authorization: a new protocol version with its own namespace, reviewed and frozen on `main`. The code has no override flag.

### 5.1 Clean checkout

`git status --porcelain=v1 -z --untracked-files=all --ignored=matching`. Every modified, staged or untracked path stops the run. An ignored path stops the run unless it matches this allowlist (caches and build output only):

| Allowed ignored path | Reason |
|---|---|
| `.venv/` | local virtual environment |
| `.pytest_cache/` | pytest cache |
| `**/__pycache__/` | bytecode caches inside `__pycache__` directories |
| `src/*.egg-info/` | editable-install metadata |
| `build/`, `dist/` | build output (not on the import path) |

Any other ignored path (for example `.env`, `results/`, `data/raw/`, a stray `.pyc` outside `__pycache__`) stops the run. Run in a fresh checkout. Note: `uv run` creates an untracked `uv.lock` in this repository; use `uv run --frozen` with a committed lock, or the virtual-environment Python directly.

## 6. CPU estimate (upper)

Basis: committed DEV policy CPU x 72/48 x 1.5. T4: the whole upper DEV comparison estimate of the v1.2 pilot (three variants + T1 + evaluator).

| T0 | T1 | T2 | T3 | T4 | Total |
|---|---|---|---|---|---|
| 6.2 s | 46.4 s | 7615 s | 155.1 s | 983 s | 8806 s (cap 14400 s) |

Wall time is set by T2 (one process): about 1.4 h (central) to 2.1 h (upper), inside the 4 h wall cap.

Reviewer decision (review of 52df90e): keep `replacement_cpu_s_cap` = 7200 s. The upper T2 estimate (7615 s) exceeds it; the central estimate (about 5077 s) does not. The gate judges compute feasibility on *measured* policy CPU, so T2 may finish the run and still be not compute-feasible as a replacement. The run is not stopped at 7200 s for T2; only the total, per-decision and wall limits stop the run.

## 7. Before the run

1. Review this PR; merge it to `main` (the freeze). Record the merge commit.
2. On a fresh clean checkout of that commit: `python -m ember_v1.teacher.confirmation run --protocol-commit <merge sha>`.
3. Commit `attempt.json`, `attempt_log.jsonl` and `evidence.json` (also after an abort), then run `selection_gate` with the trusted protocol ref. A selection result is a proposal: the receipt still needs review before S1.
