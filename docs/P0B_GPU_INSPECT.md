# P0B-02 (issue #2): GPU inspect preparation

**Status: prepared, NOT RUN.** No Modal job (CPU or GPU), no model download and no training were used. This document fixes the first GPU command, the dependency lock, the cost envelope and the stop conditions. GPU results go in a later PR, after the first `inspect` is explicitly authorized.

## 1. First GPU command (one bounded `inspect`)

Run from a clean checkout of this branch (the entrypoint refuses uncommitted changes):

```bash
python -m pip install -e '.[test,modal]'
python -m pytest -q
modal setup                                   # once
modal run infra/modal_app.py --task inspect --gpu L4 --authorize inspect
```

- Do not set `EMBER_FAST_KERNELS`. That layer is not locked and builds on a GPU builder (billed).
- The command makes one `P0BHarness.inspect` call: `Qwen3.5-2B`, bf16, compact questions, N=3, plus an encoding-only probe at N=1 and N=200. No training, no throughput sweep, no padding benchmark, no EMBERBench, no private data.
- Output: `results/inspect-L4-bf16-compact.json` (git-ignored). Exit 0 only for an explicit `PASS`; else 2.
- After the run, stop. The next task (`smoke`) needs its own authorization.

## 2. Dependency audit and pinned runtime strategy

### 2.1 Audit of the old image pins

| Old image pin | Upstream at `unsloth@fa2a5e7` (2026-11-03) | Finding | New pin |
|---|---|---|---|
| `unsloth` (unpinned) | `_version.py` = 2026.10.11 | PyPI wheel 2026.10.11 (2026-10-30) has the audited bytes of `_vendor/clef/joint_schema_model.py`, `models/decision.py`, `models/decision_from_lm.py`, `models/_decision_common.py` | `unsloth[huggingface]==2026.10.11` |
| (missing) | Core deps (`peft`, `accelerate`, `datasets`, `sentencepiece`, ...) are only in the `huggingface` extra | A plain `pip install unsloth` does not install them | `[huggingface]` extra, all locked |
| `unsloth_zoo` (unpinned) | `>=2026.10.11` | Requires torch `>=2.4,<2.16`, transformers range below | `2026.10.11` |
| `transformers==5.2.0` | `>=4.52.4,<=5.17.0`, excludes 5.0.0/5.1.0 | Inside range. The 5.2.0 wheel has `models/qwen3_5` (fla/causal-conv1d optional, torch fallback otherwise) | kept |
| `trl==0.22.2` | `>=0.18.2,<=1.15.0` | Inside range | kept |
| `torch==2.8.0`, `torchvision==0.23.0`, `xformers==0.0.32.post2` | torch `<2.16` | Inside range. PyPI torch 2.8.0 Linux wheel is CUDA 12.8 | kept (lock: `triton==3.4.0`) |
| `tokenizers>=0.22.0,<=0.23.0` | notebook | Inside range | lock: `0.22.2` |
| `torchao>=0.16.0` | zoo: `>=0.13.0` | **Risk:** torchao 0.16.0 is built for torch 2.10; on torch 2.8 it skips its C++ extensions (`torchao/__init__.py` compatibility table) | `0.14.0` (last build for torch 2.8.0) |
| `bitsandbytes` (unpinned) | — | Not used by bf16 | lock: `0.50.2` |
| base `nvidia/cuda:12.8.1-devel-ubuntu22.04` (tag) | — | Tag can move | pinned by digest `sha256:a99a1860…027e97` |
| model `unsloth/Qwen3.5-2B` (no revision) | `from_pretrained(revision=)` reaches the backbone load (`decision_from_lm.py`) | Unpinned weights | commit `fffbc0d8711cf87df6d1f97938c5e50ef94aadc7` (HF API, not gated) |

The Unsloth notebook installs `unsloth`/`unsloth_zoo` from git `main` and then upgrades them from PyPI. Its exact stack is therefore not reproducible. This image uses the PyPI wheel whose contract files match the audit.

Load path facts (source, wheel 2026.10.11): `Qwen3.5-2B` is a plain LM, so `FastDecisionModel.from_pretrained` calls `load_lm_as_decision_model`. It loads the backbone with `FastModel.from_pretrained(..., revision=)` and creates a **fresh** `JointSchemaHead` under `torch.manual_seed(3407)`. The head is untrained: inspect checks compatibility, not quality.

### 2.2 How the image is built

1. Base image by digest, Python 3.11, `git` + `build-essential` (triton needs gcc).
2. `pip install -r infra/p0b-requirements.lock.txt`: 107 packages, exact versions, sha256 hashes (pip hash-checking mode). The lock comes from `uv pip compile infra/p0b-requirements.in --python-version 3.11 --python-platform x86_64-manylinux_2_28 --generate-hashes`.
3. `python -m pip check`.
4. `python -m ember_v1.p0b verify-install` (CPU, no unsloth import): the four upstream files must have the audited sha256, and `unsloth`, `unsloth_zoo`, `torch`, `transformers`, `trl`, `xformers`, `triton`, `torchao` must equal `PINNED_RUNTIME`. Else the build fails.

Steps 2–4 run on Modal's CPU image builder, before any GPU container. The lock sha256 is in the container env (`EMBER_IMAGE_LOCK_SHA256`) and in the result metadata (`host.image_lock_sha256`).

Freeze rule (issue #2): after a PASSing `inspect`, the `pip_freeze` list in the result is the environment lock to keep. Do not change the lock inside P0B.

## 3. What `inspect` records

| Requirement | Result field | Gating check |
|---|---|---|
| Runtime provenance | `metadata`: `model_revision`, `requested_model_revision`, `tokenizer_revision`, `tokenizer_sha256`, `encoder_config_sha256`, `head_sha256`, `runtime_versions` (14 packages), `cuda`, `gpu`, `gpu_capability`, `host` (python, platform, NVIDIA driver, cuDNN, image lock sha256), `backbone` (class, parameter dtypes, forced-float32 flag), `pip_freeze_sha256`, `code_commit`; `pip_freeze` list | `provenance`, `model_pin` |
| Upstream identity | `upstream_contract`: sha256 and match flag of the four audited files | `upstream_contract`, `clef_encoder` |
| Full-state preservation | `encoding` (N=3, unbounded re-encode proof); `preservation_probe` (N=1, N=200, encoding only) | `encoding` (probe: recorded, not gating; `smoke` owns N<=200) |
| Native prediction | `predict_status`; ID set, five actions, probabilities | `native_predict` |
| Decision head | `decision_head`: head/encoder classes, `forward` signature (`input_ids`, `attention_mask`, `records`), head parameter count, dtypes, state shapes; `record_type`, `question_type`, `question_options` (question order and option order) | `decision_head`, `item_shape` |
| Batched adapter | `batched_vs_predict_argmax_equal` | `batched_adapter`, `adapter_agreement` |
| Executed GPU kernels | `kernels_native_predict` and `kernels` (top 25 by device time, device event count, `markers` for fla gated-delta, causal-conv1d and triton names) | `kernel_profile` (both profiles must exist) |

`status` is `PASS` only if all 11 checks pass. A missing check is `UNVERIFIED`. All PR #21 gates are unchanged: preservation proof, sealed evidence, `predict_answers` validation, path B evidence and scope.

## 4. Cost envelope

Prices: modal.com/pricing, read 2026-11-03: L4 $0.000222/s, L40S $0.000542/s, CPU $0.0000131/core/s, memory $0.00000222/GiB/s. Starter plan: $30/month free credit.

| Item | Estimate | Bound |
|---|---|---|
| Image build (CPU builder, about 6 GB of wheels) | a few cents of CPU time if billed; no GPU | build fails closed (section 2.2) |
| `inspect` on L4: cold start, model download (about 4–5 GB, to the `ember-v1-hf-cache` volume), load, one N=3 check, two profiles | 4–8 min: **about $0.10–$0.20** (GPU about $0.05–$0.11, CPU/memory the rest) | **$0.52**: 900 s start + 600 s call at L4 + 4 cores + 32 GiB (`gpu_run_plan.worst_case_usd`) |
| Volume storage of the weights | under $0.01/month | — |

P0B phase cap: $10 (HLD 5.3). This command uses at most about 5% of it. The $40 reserve is never used automatically. Report actual dollars from the Modal usage page after the run.

## 5. Stop conditions

Automatic, in order:

1. **Local refusal, no Modal call:** no `--authorize inspect`, uncommitted changes, model not `unsloth/Qwen3.5-2B`, GPU not L4/L40S, precision not bf16 (`p0b.gpu_run_plan`).
2. **Image build fails, no GPU container:** a wheel hash differs, `pip check` fails, or `verify-install` is `FAIL` (upstream bytes or versions drifted).
3. **Container start > 900 s** (`startup_timeout`): Modal stops it. `retries=0`, `max_containers=1`.
4. **`inspect` call > 600 s** (`timeout`): Modal stops it.
5. **Any check not `PASS`:** exit code 2. The report is written first. Do not run `smoke`.
6. Container idle: `scaledown_window=2` s. `modal run` stops the ephemeral app when the entrypoint returns.

Manual: `modal app list` and `modal app stop <app-id>` stop a stuck run.

## 6. Remaining GPU-only uncertainties

| # | Question | Evidence in the inspect result |
|---|---|---|
| G1 | Does `FastModel.from_pretrained` load Qwen3.5-2B (a VLM checkpoint) with this stack? The notebook used git `main`, not this wheel combination. | load succeeds; `backbone` |
| G2 | Does the loaded config expose `_commit_hash` = the pinned commit (U4), and the tokenizer its commit (U5)? | `model_revision`, `tokenizer_revision`, `model_pin` |
| G3 | Is the backbone bf16 on L4, or on the forced float32 path? Memory? | `backbone.param_dtypes`, `unsloth_forced_float32`, `cuda_mem_reserved_gb` |
| G4 | Does the `predict()` output pass the strict checks (ID set, five actions, sum within 1e-3)? | `native_predict` |
| G5 | Does `_clef_amp_dtype` exist and do flat logits follow record-then-question order (U6, A1)? | `batched_adapter`, `adapter_agreement` |
| G6 | Which DeltaNet kernels run (A5)? fla and causal-conv1d are not in the base image, so the transformers torch fallback is expected. | `kernels*.markers`, `executed` |
| G7 | Does the profiler see CUDA kernels in a Modal container (CUPTI)? | `kernel_profile` |
| G8 | Are the identity hashes computable (U9), and is `head_sha256` reproducible across runs (seed 3407)? | `metadata` hashes; compare two runs later |
| G9 | Does the hash-checked install of 107 wheels succeed on the Modal builder? (CPU, before GPU) | image build log |
| G10 | Real compact token count at N=200 versus 16384. | `preservation_probe` |
