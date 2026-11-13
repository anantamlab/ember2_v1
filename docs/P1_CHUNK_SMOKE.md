# P1-11 `chunk_smoke` — bounded A10 smoke test (prepared, NOT RUN)

Status: **NOT AUTHORIZED. NOT RUN.** Running it needs new written authorization from the owner for this
exact task. No GPU training, throughput sweep or padding experiment is part of it. Batch size stays 1.

## 1. Command

From a clean checkout of the commit to test (the plan refuses a dirty tree):

```
modal run infra/modal_app.py --task chunk_smoke --gpu A10 --repeats 5 --authorize chunk_smoke
```

| Limit | Value | Source |
|---|---|---|
| GPU | A10 (one attempt; no automatic fallback to another GPU) | `p0b.ALLOWED_GPUS`, operator choice |
| containers / retries | 1 / 0 | `p0b.MAX_CONTAINERS`, `p0b.RETRIES` |
| startup + call timeout | 900 s + 1 800 s | `p0b.STARTUP_TIMEOUT_S`, `TASK_TIMEOUT_S["chunk_smoke"]` |
| cost estimate | $1.16 (GPU $0.83, assumed CPU $0.14, assumed memory $0.19) | `p0b.cost_estimate("chunk_smoke", "A10")` |
| estimate kind | `ESTIMATE_NOT_A_BILLING_CAP`; excludes image build, volume storage, egress, queueing | `p0b.ESTIMATE_EXCLUDES` |
| model | pinned Qwen3.5-2B revision, bf16, `max_seq` 16 384 | `p0b.gpu_run_plan` |
| output | `results/chunk_smoke-A10-bf16-compact.json` (the `compact` in the name is the CLI default; the task runs both styles) | `infra/modal_app.py` |

## 2. Checks and expected outputs

Code: `src/ember_v1/p0b_chunk_smoke.py`; CPU tests with fakes: `tests/test_p0b_chunk_smoke.py`.
Token counts below are CPU counts with the pinned tokenizer (`token_budget.clef_counts`). The GPU result
must agree; a disagreement is a finding, not a pass.

| # | Check | Expected | FAIL / UNVERIFIED if |
|---|---|---|---|
| K1 | loaded FLA package file | under `unsloth_zoo/_vendored/fla`; `__version__` and `_UNSLOTH*` markers recorded | the file is elsewhere (FAIL); no FLA module loaded (UNVERIFIED) |
| K2 | bound causal-conv function | every module attribute named `*causal_conv1d*` is defined in `unsloth_zoo.temporary_patches.gdn_causal_conv1d` or `unsloth_zoo._vendored.fla` (the two conv paths of the pinned wheel, `data/p0b/p0b-wheel-provenance.json`) | any binding from another module (FAIL); no binding found (UNVERIFIED: the attribute names then need a source read; never PASS) |
| S1 | compact single row, N = 1, 10, 50, 100 | `encode_one` status `OK`, `truncated: false`; validated `predict()`; latency over 5 repeats after 2 warmups | any other status |
| S2 | verbose single row, N = 1, 10, 50, 100 | as S1 | as S1 |
| R1 | single row, N = 200, compact and verbose | refused: status is not `OK` (CPU: 28 294 and 32 294 tokens, `SCHEMA_TOO_LONG`) | status `OK` (FAIL) |
| C1 | N = 200, compact, k = 100 | 2 rows, each `OK` and `truncated: false` (CPU: 15 441 and 15 255 tokens); 2 validated predictions; joint map of 200 people; resolver outcome counts sum to 200 | any row not `OK`, a missing or extra person (`ChunkError`), or a resolver error |
| C2 | N = 200, verbose, k = 100 | refused: both rows not `OK` (CPU: 17 441 and 17 255 tokens, over the bound) | any row `OK` (FAIL) |
| C3 | N = 200, verbose, k = 50 | 4 rows `OK`; joint map of 200 people; resolver outcome counts sum to 200 | as C1 |
| M1 | GPU memory | `peak_memory_gb` = `torch.cuda.max_memory_allocated()` over the task | recorded only; no threshold |

Overall `status` is `PASS` only if K1, K2, S1, S2, R1 and C1–C3 all pass. Any `FAIL` gives `FAIL`;
otherwise any `UNVERIFIED` gives `UNVERIFIED`. The client writes the report first and exits 0 only for an
explicit `PASS` (`p0b.result_exit_code`). A failed remote call writes a `NOT_PASS` record and exits 2.

"No silent truncation" is shown positively: `encode_one` compares the bounded encoding with an unbounded
reference encoding of the same row and requires one spare token (`p0b.encoding_evidence`).

## 3. What a PASS supports and does not support

- Supports: R3 (A10 smoke at the declared bound, compact), R4 (kernel provenance), and the GPU part of R7
  for one synthetic N=200 state.
- Does not support: decision quality, chunked joint decision quality (see `docs/P1_CHUNK_JOINT_MAP_STUDY.md`),
  batched inference, padding parity, throughput, or training cost. Batch size stays 1 until a separately
  authorized padding parity task passes.
- The predictions come from the base model with the default head. They test the pipeline, not a policy.

## 4. Known risks before running

| Risk | Effect | Handling |
|---|---|---|
| The GDN conv attribute is not a `*causal_conv1d*` instance attribute | K2 `UNVERIFIED`, overall not PASS | read the loaded source; fix the probe; ask for a new authorization |
| Verbose N=100 single row is 16 035 tokens (349 spare) | a small encoder difference can refuse it | S2 FAIL is reported, not repaired |
| Synthetic state only | no TRAIN, DEV or benchmark state is used | by design |
