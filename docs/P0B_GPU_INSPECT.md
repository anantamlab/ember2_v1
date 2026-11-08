# P0B-02 (issue #2): GPU inspect preparation

**Status: prepared, NOT RUN.** No Modal job (CPU or GPU), no model download and no training were used. This document fixes the first GPU command, the dependency lock, the cost estimate, the stop conditions and the failure report. GPU results go in a later PR, after the first `inspect` is explicitly authorized.

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
- Output: `results/inspect-L4-bf16-compact.json` (git-ignored), the inspect report or a failure report (section 6). Exit 0 only for an explicit `PASS`; else 2.
- After the run, stop. The next task (`smoke`) needs its own authorization.

### 1.1 GPU preference and capacity fallback (2026-11-08 authorization)

Order: **L4 → A10 → L40S**, as separate bounded attempts, never concurrent. Modal's ordered `gpu=[...]` fallback list is **not** used: the watchdog, the cost estimate and the queue rule assume one GPU type per call, and that interaction is not verified.

1. Run the L4 command above.
2. Go to the next GPU **only** if the attempt ended with stage `queue_timeout` (no container ever seen). Before the next attempt, check that the call is no longer queued or running (`modal app list`, `modal container list`, the report `cancellation` field) and record any uncertainty.
3. A started container that fails (model load, any check) is a result. Do not retry it on another GPU.
4. `run.allocated_gpu` records the device name and capability from CUDA in the container; `matches_request` is `null` when no container started. No GPU compatibility claim without it.

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
4. `PYTHONPATH=/root python -m ember_v1.p0b verify-install` (CPU, no unsloth import). `add_local_python_source` copies the package to `/root/ember_v1`, and Modal puts `/root` on `sys.path` only at function run time, not in builder `RUN` steps. The first attempt (2026-11-03, app `ap-3xQNSAAuF38oW9Y9O3CQHJ`) failed here with `ModuleNotFoundError: No module named 'ember_v1'`, before any GPU container. The CPU test `test_verify_install_build_step_imports_from_pythonpath` checks only the import path; only a real Modal image build verifies this step. In this step: the four upstream files must have the audited sha256, and `unsloth`, `unsloth_zoo`, `torch`, `transformers`, `trl`, `xformers`, `triton`, `torchao` must equal `PINNED_RUNTIME`. Else the build fails.

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

## 4. Cost estimate (not a billing cap)

All numbers here are **estimates**. Nothing in this repository stops Modal charges at a dollar value. The real charges are on the Modal usage page; report them after the run. Use Modal's workspace budget settings for an enforced financial limit.

Prices (inputs only, they can change): `modal billing rates` for this workspace, read 2026-11-08: L4 $0.80/h ($0.000222/s), A10G $1.10/h ($0.000306/s), L40S $1.95/h ($0.000542/s), CPU $0.0473/core/h, memory $0.008/GiB/h. Starter plan: $30/month free credit.

Estimated envelope of one `inspect` attempt at the configured limits: L4 $0.52, A10 $0.65, L40S $1.00. With the sequential rule only one attempt can start a container, so the planned maximum for the 2026-11-08 preflight is one envelope (≤ $1.00) plus the image build, within the $2 authorization.

### 4.1 GPU container runtime (estimated)

`p0b.cost_estimate` (in `gpu_run_plan()["cost_estimate"]`, `kind: ESTIMATE_NOT_A_BILLING_CAP`) computes the cost of one container that runs for the full configured time:

| Part | Value at the configured limits (L4, `inspect`) |
|---|---|
| Configured container time | 900 s startup timeout + 600 s call timeout = 1500 s |
| GPU (L4) | $0.333 |
| CPU, **assumed** 4 cores | $0.079 |
| Memory, **assumed** 32 GiB | $0.107 |
| **Estimated envelope** | **$0.52** |

The $0.52 holds only if all of these are true:
- the container ends within the two timeouts, with one attempt (`retries=0`, `max_containers=1`);
- CPU and memory stay at or below 4 cores and 32 GiB (these are assumptions, not Modal limits; more usage costs more);
- the listed prices apply to this workspace and region;
- no platform restart (for example a host preemption) adds a second container.

Expected value for a normal run: 4–8 min of container time, about **$0.10–$0.20** (estimate; GPU about $0.05–$0.11).

### 4.2 Charges outside the estimate

| Charge | Status |
|---|---|
| Image build on the Modal builder (about 6 GB of wheels, `pip check`, `verify-install`) | Not estimated. No billing evidence yet; do not assume it is free. Read it on the usage page. |
| Optional `EMBER_FAST_KERNELS` layer | Builds on a GPU builder. Not part of this run. |
| Volume storage (`ember-v1-hf-cache` holds about 4–5 GB of weights; `ember-v1-checkpoints`) | Not estimated; billed per Modal storage pricing, if any. |
| Network or egress, local client time, queueing before the container starts | Not estimated. |

P0B phase budget: $10 (HLD 5.3), a planning number that Modal does not enforce. The $40 reserve is never used automatically.

## 5. Stop conditions

Automatic, in the order they occur:

1. **Image build fails, no GPU container:** a wheel hash differs, `pip check` fails, or `verify-install` is `FAIL` (upstream bytes or versions drifted). `modal run` builds the image when it creates the app, before the entrypoint runs.
2. **App creation refused, no GPU container:** for example `InvalidError: Please add a payment method to use L4 GPU functions` (attempt 2). The entrypoint does not run.
3. **Entrypoint refusal, no GPU container:** no `--authorize inspect`, uncommitted changes, model not `unsloth/Qwen3.5-2B`, GPU not L4/A10/L40S, precision not bf16 (`p0b.gpu_run_plan`). The ephemeral app exists and the image build may already have run at this point.
4. **Confirmed queue wait > 300 s** (client watchdog, section 6): the call is cancelled. Modal's own timeouts do not cover queue time.
5. **Container start > 900 s** (`startup_timeout`): Modal stops it. `retries=0`, `max_containers=1`.
6. **`inspect` call > 600 s** (`timeout`): Modal stops it.
7. **Client deadline 1860 s** (300 + 900 + 600 + 60 grace, section 6): the call is cancelled when the queue state was not confirmed.
8. **Any check not `PASS`, or a failed call:** exit code 2. The report is written first. Do not run `smoke`.
9. Container idle: `scaledown_window=2` s. `modal run` stops the ephemeral app when the entrypoint returns.

## 6. Bounded wait and failure reports

Attempt 3 (app `ap-WB7GlV2CTIyIJ2WQRX7xYF`, 2026-11-03) waited 2069 s with no container: the old entrypoint used a blocking `.remote()` call, and Modal's startup and execution timeouts start only when a container exists. The entrypoint now uses `ember_v1.run_report.watch_remote_call`.

### 6.1 Watchdog

| Step | Behavior (Modal 1.6.3 public APIs only) |
|---|---|
| Submit | `harness.<task>.spawn(...)` exactly once. No resubmission on any path. |
| Record | Right after submission, a provisional `UNVERIFIED`/`in_progress` report with the FunctionCall ID, app ID and manual cleanup steps replaces any older result. |
| Post-submit guard | Supervision starts when `spawn()` returns. If the provisional write, the log callback or Ctrl-C fails before the first poll, or any other BaseException (for example `SystemExit`, a failed report write) leaves the wait loop: cancel the call, write the report if possible (else delete an older result, so no stale `PASS` stays), print one JSON line with the FunctionCall ID, app ID, cancel outcome and cleanup steps to stderr, and re-raise the original error. A failed stderr write is ignored. Stage `post_submit_error` (or `interrupted`). |
| Wait | `FunctionCall.get(timeout=30)` in a loop. An empty poll raises the builtin `TimeoutError` (Modal job-queue guide); it does **not** cancel the call. |
| Queue state | After each empty poll: `get_current_stats()` of this instance's variant (`backlog`, `num_total_runners`, `num_running_inputs`). `queued` = backlog >= 1, 0 runners, 0 running inputs. |
| Queue timeout | Cancel at 300 s only if every reading since submission was `queued` (at least 3 readings, no stats error, no container seen). Stage `queue_timeout`. |
| Client deadline | Otherwise cancel at `client_deadline_s` = 300 + startup 900 + task timeout 600 + 60 = 1860 s for inspect. Stage `client_deadline`, with `execution_state` `unknown` or `started_observed`; the report never claims the call stayed queued. |
| Poll errors | A client `ConnectionError` while polling is counted. 3 in a row: cancel, stage `poll_failed`. |
| Ctrl-C | Cancel, stage `interrupted`. |
| Cancel | `FunctionCall.cancel(terminate_containers=True)`, then one `get(timeout=15)`. Outcomes: `failed` (`cancel()` raised); `terminal_error_observed` (`get()` then raised a non-timeout error); `unverified` (no terminal state, or the check was interrupted); `completed_before_cancel` (the result arrived; it is kept). There is no `confirmed` outcome: in Modal 1.6.3 a terminated input reaches the client as a generic `RemoteError` (`_process_result`), the same class as other failures, and `InputCancellation` exists only inside the container. |

`FunctionCall.get_call_graph()` is not used: Modal documents its data as best-effort and not real-time. `FunctionCurrentStats` has no per-call state, so the queue rule assumes this ephemeral app's variant has only this one input (true for `modal run` with `max_containers=1`).

### 6.2 Report

Every non-success writes `results/<tag>.json` atomically (temp file, fsync, rename) and exits 2. The report replaces any older result for that tag, so no stale `PASS` remains.

| Field | Content |
|---|---|
| `schema` | `ember_v1.p0b.run_failure/2` |
| `task`, `gpu`, `code_commit`, `model_revision` | from the plan (`unsloth/Qwen3.5-2B@fffbc0d8…`) |
| `status`, `stage` | `FAIL` + `model_load` or `remote_task` (an exception carrying remote load/task frames). `UNVERIFIED` + `in_progress`, `submit_failed`, `post_submit_error`, `queue_timeout`, `client_deadline`, `interrupted`, `poll_failed`, `remote_timeout` or `unknown`. A call that never reached the model is never `FAIL`. |
| `error` | exception type and a sanitized message (max 500 characters), or `null` for a deadline |
| `remote_frames` | function names only, no file paths |
| `local_elapsed_s` | client wall clock from submission |
| `function_call_id`, `app_id` | Modal IDs (not secrets) |
| `watchdog` | limits, poll and error counts, last 20 stats readings, `execution_state`, `first_start_evidence_s`, `cancellation` |
| `outstanding_function_call_id`, `manual_cleanup` | set after every cancel except `completed_before_cancel` (outcomes `failed`, `unverified`, `terminal_error_observed`), and in the provisional record |
| `cost_estimate` | the plan estimate, `kind: ESTIMATE_NOT_A_BILLING_CAP` |
| `compatibility_pass_established` | always `false` |

The message is sanitized: Hugging Face and Modal tokens, `Bearer` values, `token=`/`secret=`/`password=`/`api_key=` values, URL credentials and home or `/root` paths are replaced. No environment variables are written.

**No local report is possible** if `modal run` fails while it creates the app (image build, GPU access refused): the entrypoint never runs. If the local process is killed abruptly (SIGKILL, closed terminal, power loss), only the provisional record remains, and this code does not cancel the call. Then Modal logs (`modal app logs <app-id>`, `modal function calls <function-id>`, the dashboard) are the evidence. Record that case by hand as `UNVERIFIED`.

Manual cleanup: `python -c "import modal; modal.FunctionCall.from_id('<fc-id>').cancel(terminate_containers=True)"`, or `modal app stop --yes <app-id>`; then `modal app list` and `modal container list`.

### 6.3 Attempt log

| # | App | Result | Stage |
|---|---|---|---|
| 1 | `ap-3xQNSAAuF38oW9Y9O3CQHJ` | image build failed: `ModuleNotFoundError: ember_v1` in `verify-install` | image build (fixed in `b3e10f3`) |
| 2 | `ap-Ct45UWWa05s5Bdj2daPyQK` | `verify-install` PASS; `InvalidError: Please add a payment method to use L4 GPU functions` | app creation |
| 3 | `ap-WB7GlV2CTIyIJ2WQRX7xYF` | call `fc-01M6GFMPN4FEN4YYWVV5P9M24W` enqueued 10:35:24 UTC, never started, no container; stopped by hand after 2069 s; `UNVERIFIED` | queue (reason unknown) |

GPU spend for all three: $0. All 11 checks: NOT RUN.

## 7. Remaining GPU-only uncertainties

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
| G9 | Does the hash-checked install of 107 wheels succeed on the Modal builder? (CPU, before GPU) | **Resolved 2026-11-03: yes.** Attempt 1: 107 wheels, `pip check` clean. Attempt 2: `verify-install` PASS, all audited hashes. |
| G10 | Real compact token count at N=200 versus 16384. | `preservation_probe` |
