# P1-09: prospective confirmation set `p1-09-selection@1`

Status: **prepared, not run.** No teacher has run on any scenario of this set. The protocol in this PR is a freeze *candidate*: it becomes the freeze only when a reviewed commit with it reaches `origin/main`. Teacher selection, bulk S1 labels and Qwen training are not started.

Code: `src/ember_v1/teacher/confirmation.py`. Tests: `tests/test_selection_confirmation.py`, `tests/test_teacher_selection_gate.py`.

## 1. Files

| File | Content | Hash |
|---|---|---|
| `data/selection/p1-09-selection@1/manifest.json` | 24 families x 3 lineages = 72 scenarios (IDs, seeds, hashes; no scenario bodies) | `manifest_hash` `sha256:14bc36318ddbc3e7798860c7dae28904be3c56740ca9e7ab4da352152bc150f2` |
| `data/selection/p1-09-selection@1/isolation_audit.json` | overlap of every key with every earlier set | `audit_hash` `sha256:4a08f3f0897e0fb94b5baf3189fff2a5524440f7c87ae99365560a3a8b15ef18` |
| `data/selection/p1-09-selection@1/protocol.json` | frozen protocol candidate `teacher-selection-protocol@1.1.0` | `protocol_hash` `sha256:dbdae86b9ac874eba56b50ef701fc90b993f0231dd362573beb72c67ae1c5165` |
| `data/selection/p1-09-selection@1/evidence.json` | **absent.** Written once (`open(..., "x")`) by the one confirmatory run. | — |

Rebuild (deterministic; refuses to overwrite a file with other content): `python -m ember_v1.teacher.confirmation build`. A test rebuilds the manifest and the audit and compares them to the committed files byte for byte as JSON.

## 2. Generator specification (frozen)

| Item | Value |
|---|---|
| Plan / namespace | `p1-09-selection@1`, split `selection_confirmation` |
| Seed namespace | `p1-09-selection@1/seeds`, master seed `20261108`; family seed = `derive_seed(master, namespace, "family", slot)`, lineage seed = `derive_seed(family_seed, "lineage", j)` |
| Generator | `gen-1.0.0`, `DEFAULT_SPEC` (hash in `spec_hash`) |
| Pool | slots 0 to 71 in order; stop at 24 accepted families |
| Family rejection | exact topology duplicate, or near-duplicate / wall near-duplicate of any of the 120 reference families from `historical_sets()` or an earlier accepted family |
| Family exclusion | a lineage fails to generate, or any of its 3 lineages collides on an isolation key; the whole family is excluded and recorded |
| Result | 25 slots used; 1 family excluded (`fam-689c192ea688783c`, slot 5, `lineage_generation_failed`) |
| Mix | archetypes corridor 7, irregular 6, multi_room 6, open_room 5; targets feasible 46, difficult 18, unreachable 8 |

## 3. Isolation audit

Keys: `family_id`, `lineage_id`, `scenario_hash`, `core_hash`, `scenario_seed`. Sets (equal to `selection.REQUIRED_ISOLATION`): `p1-02-v1/train`, `dev`, `calibration`, `locked_test` (sealed: checked by its committed index commitment, not opened), `t2-pilot@1`, `t3-pilot@1`, `t4-pilot@1`, `t4-pilot@2`, `t4-explore@v12`, `t4-pilot@3`. Every overlap is empty; all keys are unique inside the set; `pass: true`.

Not checked (recorded in the audit): `blind_test` (secret seeds; it derives from another root), and near-duplicate geometry against the pilot maps (exact core hash only; the generator rejects near-duplicates of the 120 reference families).

## 4. Preregistered rules (in `protocol.json`)

| Rule | Value |
|---|---|
| Eligible identities | T0, T1, T2, T3, T4 v1.2.0 (exact name, version, config hash from the registry). T4 v1.0.0 (G5) and v1.1.0 (G4) stay blocked records. |
| Baseline | T1 |
| Primary | casualties, paired against T1, unit = family; family-cluster bootstrap, 10000 resamples, seed 20261109, 95 % CI; sign test alpha 0.05 |
| Safety | casualties + unresolved: no net deterioration; evacuated: no net deterioration |
| Compute | total CPU cap 14400 s; per-decision cap 30000 ms; replacement CPU cap 7200 s; USD cap $15 |
| Tie | lower CPU, then simplicity T0, T1, T3, T4, T2 |
| Runs | one confirmatory run per identity; any error, missing row or drift stops selection |
| Seeds | simulator seed = scenario seed (matched across teachers); policy seeds T2 `20261107`, T3 `20261123` (the class defaults; checked against the classes); a fresh teacher instance per episode |
| Excluded data | locked_test, blind_test, calibration, emberbench |

The gate also checks that the evidence manifest is the one whose hash the protocol preregisters.

## 5. Runner and preconditions

`python -m ember_v1.teacher.confirmation run --protocol-commit <sha>` runs only if all of these hold, else it stops before any teacher runs:

1. The protocol and the manifest load from that commit, and the commit is on the trusted ref (`origin/main`).
2. The protocol is frozen and its hash verifies. The manifest hash is the preregistered one.
3. The compute preflight passes (estimated total <= `cpu_s_cap`).
4. The working tree is clean; `HEAD` descends from the protocol commit.
5. No registry pin or structure drift (`check_pins`, `check_registry`).
6. `evidence.json` does not exist.

Then one spawned process per teacher runs all 72 scenarios (each process regenerates the scenarios and checks their hashes). Errors are recorded, never repaired.

## 6. CPU estimate (upper)

Basis: committed DEV policy CPU x 72/48 x 1.5. T4: the whole upper DEV comparison estimate of the v1.2 pilot (three variants + T1 + evaluator).

| T0 | T1 | T2 | T3 | T4 | Total |
|---|---|---|---|---|---|
| 6.2 s | 46.4 s | 7615 s | 155.1 s | 983 s | 8806 s (cap 14400 s) |

Wall time is set by T2 (one process): about 1.4 h (central) to 2.1 h (upper).

**Reviewer decision needed (not changed here):** the upper T2 estimate (7615 s) exceeds `replacement_cpu_s_cap` = 7200 s; the central estimate (about 5077 s) does not. The gate judges compute feasibility on *measured* CPU, so T2 may finish the run and still be marked not compute-feasible as a replacement. Keep the cap (T2 must earn its cost), or raise it **before** the freeze. The cap must not change after any outcome.

## 7. Before the run

1. Review this PR; merge it to `main` (the freeze). Record the merge commit.
2. On a clean checkout of that commit: `python -m ember_v1.teacher.confirmation run --protocol-commit <merge sha>`.
3. Commit `evidence.json`, then run `selection_gate` with the trusted protocol ref. A selection result is a proposal: the receipt still needs review before S1.
