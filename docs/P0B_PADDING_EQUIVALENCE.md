# ember_v1 P0B: Qwen3.5 Gated DeltaNet padding-equivalence gate

**Status: SPECIFIED, NOT EXECUTED ON A GPU.** This document is for the engineer building the P0B Modal harness. The CPU-only `ember_v1.padding_gate` checker tests the *verdict logic*; it does not prove padding safety.

## Why this is essential

Qwen3.5-2B contains recurrent DeltaNet and causal convolution layers alongside attention. The attention mask of a conventional attention block does not by itself demonstrate that the DeltaNet recurrence, convolution, or Clef decision head is insensitive to padding. In particular, a head that aggregates over padded hidden states can change scores even when causal backbone outputs at valid tokens are unchanged. We must measure the whole **decision score**, not only model hidden states.

The current upstream `unsloth/_vendor/clef/joint_schema_model.py` `collate_records` uses **right padding** and returns an `attention_mask`. The inspected function has **no `cu_seqlens` path**. Do not mark packed varlen supported unless an explicitly verified model+head execution path exists. `FastDecisionModel.predict` handles one state; calling it in a Python loop is NOT cross-environment GPU batching.

## Measurements required

1. Pin exact model revision, Unsloth, Transformers, PyTorch, Triton / FLA kernel revisions, CUDA driver, GPU type, autocast dtype, and exact head path. Profile the real optimized-kernel path (not only imports).
2. Build **at least two independent, model-visible synthetic observations of unequal encoded lengths**, with stable person IDs and 1/10/50/100/200 active people as supported. Create batches where shorter sequences occur first, last, and between longer sequences. Include a long-short length ratio stress case.
3. For each *identical observation*, measure a **singleton unpadded** result, right-padded mixed-length batch, and left-padded mixed-length batch, from the **same weights, eval mode, seed/config and decision head**. No prompt rewriting, no truncation. Compare all five action probabilities for every active person, not just the chosen move.
4. For **left padding**, shift encoded Clef question spans, option spans, any media spans and position IDs consistently; verify that the constructed left-padding experiment is semantically identical. Padding left without updating spans produces a false failure. If the current head cannot represent left padding correctly, mark `LEFT_UNSUPPORTED` and fail the equivalence gate until a supported implementation exists; do not fake it with duplicated right padding.
5. **Optional packed varlen:** only if the full encoder+head supports real packed boundaries and resets the recurrent/conv state at each sequence. A flat concatenation or `attention_mask` alone is **not** `cu_seqlens` support. Verify the actual kernel backend and compare packed logits/probabilities to singleton. Otherwise record `PACKED_UNSUPPORTED` and keep the padded path if it passes.
6. Sweep batches `1,4,8,16,32,64` and GPU L4, L40S. Report per-env tokens, padded token ratio, warmed-up p50/p95 wall-clock, peak allocated/reserved VRAM, environment steps/sec, and per-person scores. Include results from both ordering permutations.
7. For equal weights and same input, require initial proposed thresholds: per-action **max absolute probability change ≤0.01**, per-person **L1 change ≤0.02**, and **zero top-1 flips** when the singleton top-two probability margin ≥0.02. Report all numeric deltas and even low-margin flips. These are initial limits; if legitimate floating-point variation requires adjustment, preregister the amended values *before evaluating task success*. Hard errors, missing persons, unexpected action sets and silent truncation always fail.

## Handoff to Modal harness

Generate a JSON result file with metadata that passes `padding_gate.validate_provenance` (`model_revision` as `<model>@<40-hex commit>`, `tokenizer_revision`, `runtime_versions`, `gpu`, `precision`, `max_seq`, `token_limit`, `head_path`, `evidence_source`, `code_commit`, and `kernel_backend` with `detected_packages` plus profiler-observed `executed` kernels; optional `packed_varlen_verified`), plus one case per `(env_id, variant)` with `variant ∈ {single,right,left,packed}`, `input_tokens`, `truncated: false`, and per-person normalized 5-action probabilities. Each env needs the singleton, left and right variant. Include two or more envs of differing length, recorded separately in the metadata. The checker enforces structural fidelity and score parity but cannot independently prove that upstream GPU processing, correct left-span shifts or real variable-length kernels were used. Save profiler traces/logs as separate artifacts.

Run:

```bash
python -m ember_v1.padding_gate /path/to/modal-padding-results.json --output /path/to/padding-verdict.json
```

## v3 amendments (HLD v3, D3)

- **Harness:** `modal run infra/modal_app.py --task padding --ns 1,10,50,200` produces one gate input per batch order (`short_first`, `short_last`, `short_middle`) in `results/` and runs the gate locally on each. The run is `PASS` only if every order passes **and** the batched adapter agrees 100% with `predict()` argmax.
- **`input_tokens` = unpadded valid tokens.** The gate fails closed if a variant's count differs from the singleton's (prompt rewrite or truncation).
- **Left padding scope.** The Clef head takes `attention_mask.sum()` as the length, so left padding is unsupported. The project excludes it explicitly: the harness sets `metadata.left_unsupported_reason`. The gate then needs only `single` + `right`, reports `left_tested: false`, and fails if left results are also present.
- **Tie-break.** Flip detection now uses the same alphabetical tie-break for the singleton and the batched top-1.

## P0B-01 amendments (CPU preflight, see [P0B_PREFLIGHT.md](P0B_PREFLIGHT.md))

- `truncated` must be exactly `false`, and the harness sets it only after positive preservation evidence. `null` (UNVERIFIED) fails.
- Placeholder provenance fails: an `unknown-commit` revision, a kernel backend from import checks only, or an absent `evidence_source`.
- `DecisionPolicy(batched_forward=True)` does not trust a verdict file. It re-runs this gate on the raw inputs of all three orders and requires the same stack.
- Still NOT EXECUTED on a GPU.

Exit code 0 means `PASS`; 2 means numeric parity `FAIL`; invalid/insufficient evidence raises a `ValueError` and is not a PASS. `PASS` here is only the **padding-equivalence subgate**: full P0B also requires model loading, no silent truncation, kernel verification, LoRA stability, throughput and resource limits.

**A critical interpretation:** if right padding passes but left padding fails because the left-pad path is not supported, choose supported right padding only *if* the project explicitly updates its P0 requirements to exclude left padding, records why, and repeats mixed-length and ordering tests. Never describe unsupported left padding as passing. Prefer a verified packed varlen execution if it is faster and equivalent, but do not introduce a custom packed head without first demonstrating need.
