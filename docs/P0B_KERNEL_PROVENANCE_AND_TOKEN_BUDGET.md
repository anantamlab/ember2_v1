# P0B CPU-only follow-up: kernel provenance and the N=200 token budget (G10)

Status: **CPU evidence only.** No GPU, no Modal job, no model weights, no training, no EMBERBench. The frozen decision model and the frozen P0B code are not changed. No padding-parity claim.

Inputs:

| Input | Pin |
|---|---|
| Lock | `infra/p0b-requirements.lock.txt`, sha256 `ec7234eb34ed79461bd2d0801f4aa97878fbe95136fb5a54c57aa87d9805b629` |
| A10 record | `docs/p0b_attempts/attempt7-A10-inspect-pass.json` |
| Clef source | `unsloth/_vendor/clef/joint_schema_model.py` in `unsloth-2026.10.11-py3-none-any.whl` (sha256 `baff520e…0992e`), file sha256 `0e304cf7…dc3a3` = `p0b.AUDITED_CLEF_SHA256` |
| Tokenizer | `unsloth/Qwen3.5-2B@fffbc0d8711cf87df6d1f97938c5e50ef94aadc7` `tokenizer.json`, sha256 `87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4`, read with `tokenizers==0.22.2` |

Outputs:

| File | sha256 |
|---|---|
| `data/p0b/p0b-wheel-provenance.json` (`experiments/p0b_wheel_inspect.py`) | `fff746dc1f69ba3d03ea495f258bede34c268f82be952afd97966f1f7a86623d` |
| `data/p0b/p0b-g10-token-budget.json` (`experiments/p0b_token_budget.py`) | `42ed36b5fe87038576ba8f20b70f95d437b6920ac097dc1f32eabfc891296d43` |

Tokenizer identity: the sha256 of the serialized tokenizer state (`to_str()`, the `p0b._tokenizer_sha256` definition) is `a57b5c62cd5f242a06c632b82cc0b4f8239d08cc80fd9347fc1f629be977c6ca`. It equals `metadata.tokenizer_sha256` of the A10 record. The experiment refuses to run if it differs.

## 1. Kernel provenance

### 1.1 Method

`experiments/p0b_wheel_inspect.py` reads each locked wheel from PyPI. It never installs, imports or runs package code.

- 106 of 107 locked packages have a linux/py3.11 wheel (`multiprocess` has none; it is not a kernel package).
- 94 wheels ≤ 60 MiB: full download, sha256 equal to the lock, every `.py` file scanned for kernel names and imports.
- 12 wheels > 60 MiB (`torch`, `triton`, `xformers`, 9 `nvidia-*`): central directory only. No top-level `fla` or `causal_conv1d` in any of them.
- No hash mismatch.

### 1.2 Findings

The A10 report shows `fla` importable but no `flash-linear-attention` or `causal-conv1d` distribution (`runtime_versions` null, `kernel_backend.detected_packages = ["fla"]`). The wheel contents explain this.

| A10 profiler name | Defined in (locked wheel) | Evidence |
|---|---|---|
| `ChunkGatedDeltaRuleFunction` | `unsloth_zoo/_vendored/fla/ops/gated_delta_rule/chunk.py` | source scan |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | `unsloth_zoo/_vendored/fla/ops/common/chunk_delta_h.py:54` | source scan, extracted file |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | `unsloth_zoo/_vendored/fla/ops/gated_delta_rule/chunk_fwd.py` | source scan |
| `chunk_fwd_kernel_o` | `unsloth_zoo/_vendored/fla/ops/common/chunk_o.py` | source scan |
| `_causal_conv1d_fwd_kernel` | `unsloth_zoo/temporary_patches/gdn_causal_conv1d.py:110` | source scan, extracted file; the only definition with this name in the lock |

- **Vendored FLA.** `unsloth_zoo-2026.10.11` (sha256 `8e638f8e…`) ships `unsloth_zoo/_vendored/fla/`: 73 `.py` files, fla-core **0.5.1** (tag v0.5.1, MIT), with narrowed package exports. `_vendored/_licenses.py` holds the license and provenance text. The `MANIFEST` and `LICENSE` files that `_licenses.py` names are not in the wheel.
- **Injection.** `unsloth_zoo/temporary_patches/fla_vendor.py` registers the vendored tree as top-level `sys.modules["fla"]` (`spec_from_file_location`) and sets the marker `_UNSLOTH_VENDORED_FLA`. Conditions: torch ≥ 2.7, triton ≥ 3.3, CUDA. An installed fla wins only if it is strictly newer than 0.5.1 or has no version, unless `UNSLOTH_FORCE_VENDORED_FLA=1`. `UNSLOTH_DISABLE_VENDORED_FLA=1` stops the injection.
- **Consumers.** `transformers-5.2.0` `models/qwen3_5/modeling_qwen3_5.py` imports `fla`. `unsloth/models/_gdn_op.py` and `_utils.py` also import it.
- **Causal conv1d.** No locked wheel contains the `causal_conv1d` package. `gdn_causal_conv1d.py` patches a Triton `causal_conv1d_fn` (marker attribute `_unsloth_triton_causal_conv1d`) when no usable `causal_conv1d` exists. `UNSLOTH_DISABLE_TRITON_CAUSAL_CONV1D=1` keeps the torch fallback. Its docstring says `seq_idx` keeps taps inside one packed sequence.

### 1.3 What CPU evidence proves, and what needs a GPU run

| Claim | CPU proof | Needs GPU |
|---|---|---|
| The lock contains no `flash-linear-attention`, `fla-core` or `causal-conv1d` distribution | **Proven** (lock + wheel scan) | — |
| The only `fla` source in the lock is `unsloth_zoo/_vendored/fla` 0.5.1 | **Proven** for 94 scanned wheels; top-level check for 12 large wheels | — |
| Every profiled GDN kernel name has exactly one definition in the lock | **Proven** by name (source scan) | — |
| At runtime `fla.__file__` is under `unsloth_zoo/_vendored/fla` and `fla._UNSLOTH_VENDORED_FLA` is set | Probable (injection conditions hold on A10: torch, triton, CUDA) | **Yes**: record `fla.__file__`, `fla.__version__`, the marker |
| The bound conv function is the Unsloth Triton one | Probable (no `causal_conv1d` package) | **Yes**: record `getattr(fn, "_unsloth_triton_causal_conv1d")` for the function the GDN layer calls |
| Kernel numerics, varlen (`cu_seqlens`, `seq_idx`) correctness | Not provable on CPU | **Yes** |

Proposed GPU addition (next P0B run, not done): add these three runtime attributes to `kernel_backend` evidence. This is an evidence change only. It does not change the model.

## 2. N=200 preservation skip (G10)

### 2.1 Method

`ember_v1.token_budget.clef_counts` re-states the audited `encode_record` layout piece by piece (`prefix + state + schema + suffix`; `fixed = prefix + schema + suffix`). Upstream raises `ValueError` if `fixed > max_length`; `build_dataset` then skips the row. Otherwise upstream cuts the state silently. The module only counts and classifies; it never cuts people or questions.

**Validation against the A10 record (exact; same tokenizer state hash as the A10 host):**

| Row | A10 GPU-side count | CPU count |
|---|---|---|
| compact, N=1 | 350 | 350 |
| compact, N=3 | 617 | 617 |

### 2.2 Cause of the skip

Synthetic rows (`p0b.synthetic_row`, 24×24 grid), limit 16384 (`decision_config.max_len` = `CLEF_SERVE_MAX_LEN`):

| N | compact fixed / total | compact status | verbose fixed / total | verbose status |
|---|---|---|---|---|
| 50 | 6 409 / 7 104 | OK | 7 409 / 8 104 | OK |
| 100 | 12 788 / 14 035 | OK | 14 788 / 16 035 | OK |
| 120 | 15 402 / 16 892 | **STATE_CUT** | 17 802 / 19 292 | SCHEMA_TOO_LONG |
| 150 | 19 316 / 21 163 | SCHEMA_TOO_LONG | 22 316 / 24 163 | SCHEMA_TOO_LONG |
| 200 | **25 845** / 28 294 | **SCHEMA_TOO_LONG** | 29 845 / 32 294 | SCHEMA_TOO_LONG |

- **Cause:** at N=200 the compact schema alone needs 25 845 tokens before the state. That is more than 16384. `encode_record` raises, and `build_dataset` skips the row. P0B records this as `FAIL_ENCODE_SKIPPED`. It is a context limit, not a kernel or model fault.
- The schema costs about 129 tokens per person (compact) and 149 (verbose). The state is small (2 449 tokens at N=200).
- Largest single-row N with a spare token: **116** (compact), **102** (verbose).
- Lower bound of the upstream schema format (ID-only instruction, options without description): 93.4 tokens per person, 18 733 fixed at N=200, still SCHEMA_TOO_LONG. **No question wording can put N=200 into one 16384-token row.**
- At N=120 compact, upstream cuts the state silently. P0B's reference check catches it (`FAIL_TRUNCATED`). Any other caller of `encode_record` does not.

### 2.3 Real p1-02-v1 rows (initial observations; counting only; `locked_test` not read)

| Split | Scenarios | Max people | ascii/compact max total | ascii/compact not OK | coords/compact not OK |
|---|---|---|---|---|---|
| TRAIN | 189 | 120 | 17 589 | **22** | 25 |
| DEV | 48 | 120 | 17 378 | 2 | 3 |
| calibration | 48 | 119 | 16 977 | 1 | 1 |

A single-row encoding cannot hold 22 of 189 TRAIN scenarios at step 0. The coords state grows with the fire, so step 0 is not its worst case. The ascii state does not grow with the fire.

## 3. Encoding strategies (prepared, not trained, not parity-checked)

| # | Strategy | Lossless? | Bound | Cost | Gate before use |
|---|---|---|---|---|---|
| A | **Compact global, one row** (`question_style="compact"`, ascii) | Only if `status == OK` | N ≤ ~100–116 (grid-dependent) | 1 forward per decision | Per-row `clef_counts` status must be `OK`; a non-OK row is refused, never cut |
| B | **Chunked rows: shared full global state + questions of ≤ k people** (`token_budget.chunk_rows`, optional `local_neighbors=True` for per-person local context) | Yes: every person in exactly one chunk, full state in every chunk | k = 100 fits every TRAIN row (`max_chunk`: 110 for the worst TRAIN row, 107 for synthetic N=200 → 2 rows) | ⌈N/k⌉ forwards; joint decision only inside a chunk; cross-chunk conflicts go to the simulator rule (`lowest_id_wins`) | Paired student evaluation of chunked versus single-row inference on rows that fit both ways |
| C | **Batching: length-sorted buckets, or packed varlen** | Yes, if masks or `cu_seqlens` are correct | — | Padding fraction below | GPU padding-parity test (batched vs batch-1 logits) before any batched training or evaluation |

Padding on the 218 chunked TRAIN rows (k=100, ascii/compact, 1.32 M real tokens):

| Batch | Random order | Length-sorted | Packed varlen (first-fit decreasing, 16384 per pack) |
|---|---|---|---|
| 4 | 46.4 % | 1.7 % | 1.4 % |
| 8 | 54.8 % | 3.7 % | 1.4 % |
| 16 | 59.0 % | 7.4 % | 1.4 % |

Packed varlen needs one sequence per record in the Clef head and correct `cu_seqlens`/`seq_idx` through the vendored GDN kernels. Neither is proven. Length-sorted buckets reach most of the gain with ordinary padding masks.

## 4. Recommendation for N=200

1. **Close G10 as explained, not as passed.** One-row N=200 at 16384 is infeasible by exact count (28 294 tokens; 18 733 even with the minimal schema). Keep the probe's `FAIL_ENCODE_SKIPPED` record. Add the exact counts and the reason (`SCHEMA_TOO_LONG`) to the probe record in a later P0B change.
2. **Bound single-row use at N ≤ 100 (compact, ascii)**, with a fail-closed per-row status check. `smoke` must not claim preservation above that bound for one row.
3. **For N > 100 use strategy B** (chunks of k ≤ 100 with the full shared state). It drops no person and no action. It changes the inference protocol, so it needs its own paired evaluation. It is not a parity claim.
4. **Alternative (GPU-gated):** `max_len = 32768`. `predict` re-encodes at `max(max_len, 16384)`, so both audited paths would accept it. Its A10 memory and latency are unknown. Measure them before use.
5. **Batching:** batch size 1 until a GPU padding-parity check passes; then length-sorted buckets.

## 5. Tests

`tests/test_token_budget.py` (23 tests, fake tokenizers): pins equal `p0b`; the per-piece count equals the length of an independent re-statement of the upstream text; outcome classes at every boundary; list criteria and empty instructions as upstream; `chunk_rows` keeps every person once, in order, with the full state; bad `k` and bad actions refused; `max_chunk` returns a verified size.
