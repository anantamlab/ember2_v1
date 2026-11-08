# P1-10 S1 distillation dataset pipeline (issue #10)

Status: **IMPLEMENTED AND TESTED ON SYNTHETIC FIXTURES ONLY. `BULK_S1_BLOCKED`.** No p1-02-v1 scenario is labelled. No bulk label exists.

Real-repository gate today (`python -m ember_v1.s1_pipeline status --repo .`, `origin/main` = `a39c3f3`):

| Receipt path | Result |
|---|---|
| `outcome/receipt_DRAFT_PENDING_REVIEW.json` | refused: "receipt is not reviewed" |
| `receipt.json` (PR #39, not merged) | refused: not on `origin/main` |
| Decision | **`BULK_S1_BLOCKED`** |

## 1. Pipeline (`src/ember_v1/s1_pipeline.py`)

`build_shard` does these steps in this order. Each step fails closed.

| # | Step | Check | Code |
|---|---|---|---|
| 1 | Receipt gate, before any teacher call | protocol and receipt from trusted git history; `s1.verify_receipt` (approved, named reviewers, T1 identity, code fingerprint) | `trust.load_trusted_json`, `s1.verify_receipt` |
| 2 | Teacher pin | registry pins and registry drift empty; teacher built by `provenance.frozen_teacher`; code fingerprint and config hash equal the receipt | `registry.check_pins`, `check_registry`, `provenance.code_fingerprint` |
| 3 | Scenario plan | TRAIN manifest only; the whole TRAIN manifest disjoint from DEV, calibration, the receipt's confirmation set and extra sets (family, lineage, scenario hash, core geometry); scenario objects hash to their manifest entries | `scenario_plan`, `heldout_from_*` |
| 4 | Bulk guard | plan size ≤ required `max_episodes`; otherwise stop (no truncation) | `build_shard` |
| 5 | Labelling | one episode per scenario to termination; one `provenance_record` per decision; full replay `verify_replay(partial=False)` | `label_episode` |
| 6 | Separation | a teacher exception or replay failure gives only a diagnostic record, never a trace. Every episode gets a diagnostic record (outcome, casualty events with the last action, movement outcomes, replay verdict). Traces are not filtered by outcome | `label_episode`, `_outcome` |
| 7 | Shard files | deterministic zip of shared-episode members, diagnostics JSONL, hash-bound shard manifest; write-once | `write_zip`, `build_shard` |
| 8 | End-to-end S1 gate | `s1.build_s1_training_set` on the written shard: ingest with full replay and provenance, regeneration spot check (max(50, 2 %) decisions), leakage audit, `bulk_s1_gate` | `s1.build_s1_training_set` |

`verify_shard` re-hashes the zip, the diagnostics and the manifest, and checks members against labelled episodes.

### Training trace (one per decision)

`{observation, actions, group_id = family/lineage, source = "T1@1.0.0", step, provenance}`. The episode (`scenario`, `seed`, `records`) is stored once per member.

- `actions` are T1's real actions. They equal the replayed step actions.
- No `values`, probabilities, logits or scores. `provenance.values_digest` is `null`. A value-carrying decision stops the run (`T1` is not in `registry.VALUE_TEACHERS`).
- `quality.values_written` and `quality.probabilities_written` are 0 by construction.

### Shard manifest (`s1-shard-manifest@1.0.0`)

Binds: receipt hash and ref; protocol hash and ref; teacher identity, config hash, code fingerprint, seed `null`; fingerprints of `s1_pipeline`, `sources` and `s1` (each with its transitive `ember_v1` imports); provenance version; member format; TRAIN split plan, manifest, generator revision and spec hashes; held-out audit; per-episode member, scenario hash, status and diagnostic hash; zip and diagnostics sha256; quality statistics; S1 gate decision. `runtime` (host CPU) is outside `manifest_hash`, so two runs give the same hash.

## 2. Ingestion changes (`src/ember_v1/sources.py`)

| Change | Reason |
|---|---|
| Shared-episode member `s1-shard-member@1.0.0` = `{member_format, episode, traces}`. A trace inside must not carry its own episode. Unknown formats are refused | The per-trace episode copy is quadratic. Upper estimate for the largest TRAIN episode: 2.43 GB per member, far above the 50 MiB member bound. Shared: 8.1 MB |
| Episode digest computed once per episode object (id memo that keeps the object alive) | Avoid quadratic hashing in replay verification |
| Bounds: at most 100 000 members, 2 GiB declared total, member ratio ≤ 200 above 1 MiB, unsafe member paths (`..`, absolute, `\`, `:`) refused | Issue #10 zip bounds (P12) |

Legacy per-trace members still ingest unchanged.

## 3. Tests (`tests/test_s1_pipeline.py`, 17 tests)

Fixtures: a synthetic split plan (`gen_helpers.small_plan(seed=31)`) and one hand-built scenario with a trapped person (a casualty) and a free person. Each test uses a temporary git repository as the trusted ref.

| Test | Proves |
|---|---|
| honest shard | every gate passes; two runs give the same zip sha256 and manifest hash; `verify_shard` passes |
| no values or scores | trace keys are exact; `values_digest` is null; actions equal replayed actions and a fresh T1 decision |
| pending receipt | blocked before any teacher call (patched `decide` raises if called); no output directory |
| other teacher in receipt | blocked |
| family in DEV, core geometry in an extra set, family in the confirmation set | blocked |
| dev manifest as TRAIN | refused |
| `max_episodes`, missing scenario | blocked |
| teacher failure | diagnostic `teacher_error`, no member for that episode, shard still verifies |
| hand-built negative episode | labelled, `negative: true`, casualty event for the trapped person, people = evacuated + casualties + unresolved |
| write-once, tamper | second write refused; flipped zip byte and edited manifest detected |
| forged actions in a rewritten shard | S1 gate refuses (replay or provenance) |
| shared-episode rules, zip bounds | malformed format, own-episode trace, unsafe path, member count, ratio refused |
| storage model | exact on noise-free samples; cover ratio > 1 on noisy samples |
| real repository status | `BULK_S1_BLOCKED` |

## 4. Storage and CPU estimate (synthetic probe, no p1-02-v1 label)

Script: `experiments/s1_storage_estimate.py`. Output: `data/s1/p1-10-storage-estimate.json`. Probe plan `s1-storage-probe-1` (p1-02-v1 generator spec, master seed 991, 24 TRAIN families x 2 lineages). Its scenarios are disjoint from all four p1-02-v1 splits, all pilot namespaces and the p1-09 confirmation set (overlap 0 on every key). T1 ran on 46 probe episodes, 3426 decisions; the actions were measured and discarded.

Model: bytes per step = a + b·people + c·walls (least squares), scaled up to cover every probe sample (factor 1.29 uncompressed, 1.19 zip). Upper bound: every person stays active until `max_steps`.

| Quantity (p1-02-v1 TRAIN, 189 scenarios, 64 families) | Central | Upper |
|---|---|---|
| Decisions (training rows) | 11 894 | 28 421 |
| Shard bytes, uncompressed | 165 MB | 397 MB |
| Shard bytes, zip | 16 MB | 39 MB |
| Largest member | | 8.1 MB (bound 50 MiB: fits) |
| T1 CPU | about 221 s (mean 18.6 ms per decision) | 4 000 s (every decision at the probe max 140.6 ms) |

Data-quality consequence: the whole TRAIN split gives about 12 000 decision rows. That is small for a 2B-parameter student. A larger TRAIN split needs a new generator plan with its own isolation audit (outside this PR).

## 5. Issue #10 checklist (status after this PR)

| # | Check | State |
|---|---|---|
| P1 | Receipt approved by named reviewers on `main` | **OPEN** (PR #39 pending review) |
| P2 | Receipt bound to the frozen protocol and evidence | verified by `receipt_review` (PR #39) |
| P3 | Selected identity = registry pin; fingerprint = current code | enforced in `build_shard` step 2 and `s1` |
| P4 | Per-decision provenance | enforced, tested |
| P5 | Full-episode replay before `teacher_verified` | enforced twice (pipeline and ingest), tested |
| P6 | Regeneration spot check | enforced by `s1` on every shard, tested |
| P7 | No invented values for action-only teachers | enforced, tested |
| P8 | Scenario bound to generator revision and split manifest | enforced, tested |
| P9 | Held-out isolation (family, lineage, scenario, core) | enforced before labelling and again by `s1`, tested |
| P10 | Duplicate and fingerprint audit | `provenance.leakage_audit` in `s1`, tested |
| P11 | Fixtures and unverified tiers never in training | unchanged `s1` rule |
| P12 | Zip bounds | added, tested |
| P13 | Bulk-S1 dependencies | `s1` dependency map; blocked by P1 |
| P14 | Confirmation set never a training source | enforced (receipt lists + core hashes from the confirmation manifest), tested |

## 6. Next authorized action (not done here)

After a named reviewer approves and merges the receipt (PR #39), and this PR is reviewed and merged:

1. One pilot shard: `build_shard` on 5 TRAIN scenarios with `extra_heldout` = locked_test and pilot namespaces (`confirmation.historical_sets`), `confirmation_manifest` = the p1-09 manifest, `max_episodes=5`. Review the shard manifest and diagnostics.
2. Then the full TRAIN shard (`max_episodes=189`), only with a separate written authorization.
