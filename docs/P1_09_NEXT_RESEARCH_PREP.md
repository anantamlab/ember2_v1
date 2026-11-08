# P1-09 follow-up: next research preparation (no execution)

Status: PREPARATION ONLY. This document does not change the confirmation result (`docs/P1_09_CONFIRMATION_RESULT.md`). No bulk S1 labels, RL, distillation, Qwen training, EMBERBench evaluation or GPU job is started or authorized here.

## 1. Teacher viability for distillation

Basis: confirmation `p1-09-selection@1`, evidence `sha256:524e17c2…`, gate decision `RETAIN_T1_NO_IMPROVEMENT` (draft, review pending).

| Teacher | Confirmation result | Viability as the distillation teacher |
|---|---|---|
| **T1** | baseline; retained by the frozen no-improvement rule; 43.3 s policy CPU for 72 scenarios | **Viable, after the receipt review.** The only teacher a reviewed receipt can name now. Cheap enough for bulk labels |
| T2 | safe, 0 worse families, CI excludes 0, Holm p 0.375 (not superior); 7555.2 s policy CPU > 7200 s cap | **Not viable as replacement** (compute-infeasible, not superior). Possible future candidate only under a new frozen protocol with a larger set and a compute plan |
| T4 v1.2.0 | safe, not superior (CI [−0.72, +0.10], Holm p 1.0); 68.8 s CPU | **Not selected.** Viable later only through a new protocol. Action-only teacher: no per-action values (issue #10) |
| T3 | safe, not superior (6/12/6 families, Holm p 1.0); 134.0 s CPU | **Not selected.** Action-only teacher |
| T0 | significantly worse (Holm p 0.0016, harmful direction), safety FAIL | **Not viable** as a teacher. Keep as a lower-bound reference only |

Constraints that stay in force:

- The receipt must be approved by named reviewers and committed on `main` before S1 accepts it (`s1.verify_receipt`).
- The confirmation set `p1-09-selection@1` is held out from every training source.
- A second selection attempt on this namespace is impossible by design. A new comparison needs a new protocol version and namespace.

## 2. Issue #10 provenance verification checklist (prepared, not executed)

Bulk S1 stays NOT AUTHORIZED. Each item names the existing code that checks it. Tick an item only with committed evidence.

| # | Check | Existing code | Evidence to commit |
|---|---|---|---|
| P1 | Selection receipt reviewed and approved by named humans, committed on `main` (review status not `pending`) | `s1.verify_receipt` | receipt JSON with reviewers, merge commit |
| P2 | Receipt names the frozen protocol (`ae72c2cf…`, commit `64b2d98`) and the confirmation evidence hash | `s1.verify_receipt`, `trust.load_trusted_json` | receipt `protocol_ref`, `confirmation.evidence_hash` |
| P3 | Selected identity (name, policy, version, config hash) equals the registry pin; code fingerprint equals current code | `registry.check_pins`, `provenance.code_fingerprint` | fingerprint table (section 1 of the result doc) |
| P4 | Each trace record binds teacher code revision, config hash, shared-utility versions and policy seed | `provenance.provenance_record`, `verify_provenance` | per-shard provenance records |
| P5 | Full-episode deterministic replay PASS before `teacher_verified` (`verify_replay(partial=False)`) | #5 replay gate, `s1.build_s1_training_set` | replay PASS/FAIL counts and reasons |
| P6 | Regeneration spot check: re-run the frozen teacher on sampled observations; identical actions (and values, where computed) | `provenance.regeneration_spot_check` | sample seed, k, mismatches = 0 |
| P7 | No invented per-action values for action-only teachers (T3 ACO, T4 priority) | `provenance.decision_values` | value-field audit |
| P8 | Scenario bound to generator revision `gen-1.0.0` and an immutable split manifest (#6) | `provenance._check_manifest` | manifest hashes per shard |
| P9 | Held-out isolation: no family, lineage, scenario or core-geometry overlap with locked_test, blind_test, calibration, EMBERBench, or `p1-09-selection@1` | `s1.heldout_overlap`, `provenance.leakage_audit` | overlap report, all empty |
| P10 | Cross-source duplicate and fingerprint audit | `provenance.leakage_audit` | duplicate report |
| P11 | `synthetic_fixture` (S0) and `teacher_unverified` never enter any split or metric | `s1.build_s1_training_set` | tier counts per split |
| P12 | Zip ingestion bounds (size, ratio, member count, path normalisation, no execution) | `s1` ingestion (`ingest_zip`) | ingestion log with input SHA-256 |
| P13 | Bulk-S1 gate dependencies (#6, #7-#9, #23, #24, #25) all closed | `provenance.bulk_s1_gate` | gate output |
| P14 | The confirmation set `p1-09-selection@1` is never a training source | `s1.heldout_overlap` with the confirmation manifest | overlap report |

## 3. GPU inspect evidence review (existing records only; no new GPU job)

Source: `docs/P0B_GPU_INSPECT.md`, `docs/p0b_attempts/attempt7-A10-inspect-pass.json` (A10 PASS 11/11, code `705dc3a`, `EMBER_FAST_KERNELS` unset).

### 3.1 G6: FLA kernels without an FLA distribution

Facts in the record:

| Field | Value |
|---|---|
| `metadata.kernel_backend.detected_packages` | `["fla"]` (`importlib.util.find_spec`) |
| `metadata.kernel_backend.evidence` | `import_check_only`; `executed` is `null` |
| `metadata.runtime_versions` `flash-linear-attention`, `causal-conv1d` | `null` (no distribution metadata) |
| `pip_freeze` (125 lines) | no `flash-linear-attention`, `fla-core`, `causal-conv1d` or direct-URL entry; has `triton==3.4.0`, `einops==0.8.2`, `unsloth==2026.10.11`, `unsloth_zoo==2026.10.11`, `transformers==5.2.0` |
| `infra/p0b-requirements.lock.txt` | no FLA provider |
| profiler markers (`kernels`, `kernels_native_predict`) | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`, `chunk_gated_delta_rule_fwd_kkt_solve_kernel`, `chunk_fwd_kernel_o`, `_causal_conv1d_fwd_kernel`, Triton fused ops |
| backbone | `unsloth_compiled_module_qwen3_5.Qwen3_5ForConditionalGeneration` |

Conclusions from these records:

1. `causal_conv1d` does not import (`detected_packages` has only `fla`), but `_causal_conv1d_fwd_kernel` ran. FLA ships a Triton causal-conv1d kernel. So one `fla` module most probably supplies all three kernel families. This is an inference, not a measurement.
2. An importable `fla` without distribution metadata means one of: a module vendored inside a locked wheel, a directory on `sys.path` without `.dist-info`, or a runtime-downloaded kernel tree. The record cannot tell which: it has no `find_spec("fla").origin`.
3. The inspect gates do not depend on the kernel source. The smoke backend comparison does (HLD A5). G6 stays OPEN.

Cheapest resolution (CPU only, no Modal; needs authorization because it downloads wheels): download the locked `unsloth`, `unsloth_zoo` and `transformers` wheels with `pip download --no-deps --require-hashes` into an empty directory and list members under `fla/`. If none has it, the smoke run must record the origin (section 4, item S5).

### 3.2 G10: N=200 preservation probe skipped

Facts: probe N=1 `OK`, 350 tokens; inspect encoding N=3, 617 tokens, not truncated; probe N=200 `FAIL_ENCODE_SKIPPED`, all token fields `null`. `p0b.task_inspect` keeps only `status`, `truncated`, `input_tokens`, `full_tokens` and `token_limit` of the probe; it drops the `build_dataset` `report`, so the skip reason is not in the record.

Inference: (617 − 350) / 2 ≈ 134 tokens per person in the compact style. N=200 then needs about 350 + 199 × 134 ≈ 26 900 tokens, above the 16 384 limit (`AUDITED_SERVE_MAX_LEN`, `max_seq`). The largest N that fits is about 120. A length overflow is the probable cause (questions for people past the cut are skipped). This is unconfirmed: the per-person token cost may be non-linear.

Consequence: a smoke at `ns=1,10,50,100,200` is expected to fail at N=200 with the current compact encoding. That is a design finding about the state format, not a GPU fault. `smoke` keeps the full `encode_one` evidence (including `report`), so it records the reason.

## 4. Bounded Modal A10 smoke protocol (PREPARED, NOT EXECUTED, NOT AUTHORIZED)

No GPU job runs without a new written authorization that names this protocol.

| Item | Value |
|---|---|
| S1 Command | `modal run infra/modal_app.py --task smoke --gpu A10 --question-style compact --repeats 25 --ns 1,10,50,100,120,200 --authorize smoke` |
| S2 Code | clean checkout of current `main`; `EMBER_FAST_KERNELS` unset; image from `infra/p0b-requirements.lock.txt` |
| S3 Limits | `retries=0`, `max_containers=1`, `scaledown_window=2 s`, startup timeout 900 s, call timeout 1800 s (`p0b.TASK_TIMEOUT_S`), client queue limit 300 s |
| S4 Cost | `p0b.cost_estimate("smoke", "A10")`: configured 2700 s, envelope $1.16 (estimate, not a billing cap). Proposed authorization cap: $2.00, one attempt, no automatic retry, no GPU fallback to L4/L40S |
| S5 Extra record (needs a reviewed code change first) | `find_spec("fla").origin` and `submodule_search_locations`; `importlib.metadata.packages_distributions().get("fla")`; same for `causal_conv1d` |
| S6 Pass criteria | N ≤ 120: `status PASS` with positive preservation evidence per row, `predict` OK, latency recorded. N = 200: expected `FAIL_ENCODE_SKIPPED` or `FAIL_NO_SPARE_TOKEN`; record `report`, token counts and reason |
| S7 Stop rules | any startup crash, queue timeout or remote exception ends the attempt; record it under `docs/p0b_attempts/`; no rerun without new authorization |
| S8 Out of scope | training, throughput sweep, padding benchmark, EMBERBench, private data, `EMBER_FAST_KERNELS` layer |
