# ember_v1 P0B preflight (P0B-01): fail-closed GPU harness

**Status: CPU-only hardening done. P0B GPU feasibility: NOT RUN.** No Modal job, GPU, model download or training was used. Every statement about GPU behavior below is either read from upstream source or listed as unverified.

## 1. Upstream contract audit

Source read: `unslothai/unsloth@fa2a5e714955ecc5ffcae555432036deeb52c37c` (`main`, 2026-11-03). Vendored Clef revision `6a3ac5fc030a3f33f7fa860a1271886b64a6606d`. Files: `unsloth/models/decision.py`, `unsloth/models/_decision_common.py`, `unsloth/_vendor/clef/joint_schema_model.py`.

| Fact (from source) | Effect on the old harness |
|---|---|
| `build_dataset` Clef items are **dicts**: `{"input_ids", "record": EncodedRecord, "source": {"state", "questions"}, ...}` | `_records()` required attributes, so every real item raised `TypeError`. |
| `EncodedQuestion` has `question_id` and `option_ids` (sorted for `choice`), not `name` | `_question_names()` fell back to positional order. |
| `encode_record` cuts **only** the end of the state: `state_ids[: max_length - fixed_length]` | `input_tokens >= state_tokens` is only a lower bound: a cut state plus long questions passes. |
| `report["truncated"]` counts items with `len(input_ids) >= max_len`; `report["skipped"]` counts questions | Report flags are conservative and per batch, not proof. |
| `FastDecisionModel.predict()` re-encodes at `max(max_len, CLEF_SERVE_MAX_LEN=16384)` and returns `_clef_decide(...)["answers"]` | Its `truncated` flag never reaches the caller. The old `predict_answers` returned `truncated=None`, and `decide()` treated it as safe. |
| `predict()` rounds probabilities to 4 decimals | Normalization tolerance must allow 2.5e-4. |
| `ClefDecisionModel.forward(input_ids, attention_mask, records, **media)` returns `(flat_padded_logits, None)` | The old `model(batch)` call did not match the signature. |
| `_clef_decide` runs under `torch.autocast(dtype=_clef_amp_dtype(...))` | The batched adapter must use the same autocast to compare with `predict()`. |

## 2. What changed (root causes fixed)

1. **Positive truncation evidence** (`p0b.encode_one`). The harness re-encodes the same `item["source"]` with the vendored `encode_record` and no length bound. A row is `OK` only if all of these are true:
   - the vendored Clef file sha256 equals the audited hash (else `UNVERIFIED`),
   - the item has the audited shape and `build_dataset` did not change the state text (else `UNVERIFIED`),
   - encoded question IDs equal the request IDs and each has exactly the five actions (else `FAIL_QUESTION_MISMATCH` / `FAIL_OPTION_MISMATCH`),
   - the bounded record equals the unbounded reference (else `FAIL_TRUNCATED`),
   - the full length is below `min(decision_config["max_len"], 16384)`, so `predict()` can not cut either (else `FAIL_NO_SPARE_TOKEN`),
   - `report["truncated"]` agrees (else `UNVERIFIED_REPORT_CONFLICT`).
   Only `OK` sets `truncated: false`. Every other status sets `truncated: null` or `true`, and the padding gate accepts only `false`.
2. **Fail-closed predictions** (`predict_answers`, `validate_answers`, `validate_questions`). `predict()` runs only after `OK` evidence for the same state and questions (sha256 bound). Output must have the exact ID set, the five actions, finite probabilities in [0, 1] that sum to 1 within 1e-3, and an answer equal to the top action. A wrapped result with `truncated` not exactly `false` fails. There is no repair and no `stay` fallback.
3. **Serving** (`decide_batch`, used by `DecisionPolicy.decide`). Any unverified request raises for the batch. `batched_forward=False` stays the default. `batched_forward=True` needs `padding_evidence_path`; `verify_batched_forward_evidence` ignores stored verdicts, re-runs the gate on the raw inputs of all three batch orders, and requires 100% adapter agreement, `OK` encodings and the same stack (model and tokenizer revision, library versions, GPU, precision, `max_seq`, token limit, head path, upstream hashes).
4. **Provenance** (`runtime_metadata`, `padding_gate.validate_provenance`). The model revision is `name@<40-hex>` or `null`, never a placeholder. `kernel_backend` separates `detected_packages` (import check) from `executed` (torch.profiler). The gate requires profiled kernels, pinned revision, library versions, GPU, precision, `max_seq`, token limit, head path and `evidence_source`. A `null` tokenizer revision or code commit is reported in `provenance_gaps`. A missing key fails.
5. **Timing.** Every timed repetition synchronizes CUDA before and after. Percentiles use nearest rank. `p95_s` needs n >= 20 and `p99_s` needs n >= 100; otherwise they are `null` with a `tail_note`. Server responses report `timing.batch_exec_s` (whole Modal batch, server side, synchronized) and `amortized_per_request_s`. They exclude Modal batching wait, queueing, network and cold start. `modal_predictor().timings` records client end-to-end time next to the server timing.

## 3. Unverified assumptions (need the GPU run)

| # | Assumption | How the next phase checks it |
|---|---|---|
| U1 | The installed `unsloth` vendors the audited Clef file (sha256 `0e304cf7…dc3a3`). If not, every row is `UNVERIFIED` (safe, but P0B can not pass). | `inspect`: `clef_encoder == "MATCHES_AUDITED"`, `upstream_contract.clef_matches_audited` |
| U2 | The installed `build_dataset` returns the audited dict items. The Modal image installs `unsloth` unpinned from PyPI, which can differ from `main`. | `inspect`: `item_keys` holds `record`, `source`; `encoding.status == "OK"` |
| U3 | `predict()` encodes with a context of at least `min(max_len, 16384)`. | `upstream_contract.decision_matches_audited`; if false, read the installed `decision.py` before trusting `smoke` |
| U4 | The encoder config exposes `_commit_hash`. If not, `model_revision` is `null` and the gate fails closed. | `inspect`/`smoke` metadata |
| U5 | The tokenizer exposes `init_kwargs["_commit_hash"]`. If not, it is reported as a provenance gap. | metadata `tokenizer_revision` |
| U6 | `unsloth.models.decision._clef_amp_dtype` exists (private helper) and the flat logits rows follow record-then-question order (also for a head that returns `flat_padded`). | `inspect`: `batched_adapter == "OK"`, `batched_vs_predict_argmax_equal`; `padding`: agreement 100% |
| U7 | (HLD A5) fla / causal-conv1d kernels run for DeltaNet layers. | `kernel_backend.executed` in `inspect` and `padding` |
| U8 | Right-padded batches match singletons within the D3 thresholds. | `padding`, all three orders |

## 4. Next-phase command checklist (authorized GPU run only)

Run in order. Stop at the first non-zero exit (2 = `FAIL` or `UNVERIFIED`).

```bash
python -m pip install -e '.[test,modal]'
python -m pytest -q
modal setup
modal run infra/modal_app.py --task inspect --gpu L4                      # U1-U7; read the JSON
modal run infra/modal_app.py --task smoke --gpu L4 --question-style compact --repeats 100
modal run infra/modal_app.py --task smoke --gpu L4 --question-style verbose --repeats 100
modal run infra/modal_app.py --task padding --gpu L4 --ns 1,10,50,200 --question-style compact
modal run infra/modal_app.py --task throughput --gpu L4 --repeats 100
modal run infra/modal_app.py --task throughput --gpu L40S --repeats 100
EMBER_FAST_KERNELS=1 modal run infra/modal_app.py --task inspect --gpu L4
EMBER_FAST_KERNELS=1 modal run infra/modal_app.py --task throughput --gpu L4 --repeats 100
```

After a PASSing `inspect` + `smoke`, pin `unsloth` / `unsloth_zoo` to the exact versions in `runtime_versions` (`infra/modal_app.py`) and re-run `inspect`. Then, to enable path B on that exact stack only:

```bash
modal volume put ember-v1-checkpoints results/padding-L4-bf16-compact.json /p0b/padding-evidence.json
# client: modal_predictor(batched_forward=True, padding_evidence_path="/ckpt/p0b/padding-evidence.json")
```

## 5. Spend

- This PR: **$0 GPU**, no Modal calls.
- Next phase estimate: about 2–3 GPU-hours (mostly L4, about 1 hour L40S, including model download, image build with the optional kernel layer, and the runs above). Check current Modal prices before the run. Hard cap: **$10** (HLD §5.3). Stop and report if a run reaches the cap.
