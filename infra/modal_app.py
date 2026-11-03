"""Modal GPU harness: P0B feasibility tasks + batched decision-policy serving.

Prerequisites (local): `pip install -e '.[modal]'` then `modal setup`.

P0B tasks (results land in ./results, padding verdicts are computed locally). Every GPU
task needs `--authorize <task>` (the same task name), a clean git checkout, the pinned
model, and runs under the timeouts of `p0b.gpu_run_plan`. A failed remote call writes a
FAIL/UNVERIFIED report (`ember_v1.run_report`); an abrupt kill or a failed image build
writes none (use Modal logs). First command (issue #2):
  modal run infra/modal_app.py --task inspect --gpu L4 --authorize inspect
Later tasks need their own written authorization (docs/P0B_GPU_INSPECT.md):
  modal run infra/modal_app.py --task smoke --gpu L4 --authorize smoke
  modal run infra/modal_app.py --task padding --gpu L4 --ns 1,10,50,200 --authorize padding
  modal run infra/modal_app.py --task throughput --gpu L40S --batch-sizes 1,4,8,16,32,64 --authorize throughput

Serving (CPU simulator/agent locally or in CPU containers, GPU inference here):
  modal deploy infra/modal_app.py
  # then ember_v1.agent.modal_predictor(...) -> DecisionAgent(predictor)

Fast DeltaNet/causal-conv kernels are an opt-in image layer because
causal-conv1d compiles from source:  EMBER_FAST_KERNELS=1 modal run ...
Run P0B both with and without it; record which backend actually executed.

Checklist, exit codes and budget: docs/P0B_PREFLIGHT.md.
"""
# No `from __future__ import annotations`: Modal reads real class-parameter types.
import hashlib
import json
import os
import subprocess
from pathlib import Path

import modal

from ember_v1.p0b import (MAX_CONTAINERS, PINNED_MODEL, PINNED_MODEL_COMMIT, RETRIES, SCALEDOWN_WINDOW_S,
                          STARTUP_TIMEOUT_S, TASK_TIMEOUT_S)

APP_NAME = "ember-v1"
DEFAULT_MODEL = PINNED_MODEL
# nvidia/cuda:12.8.1-devel-ubuntu22.04, pinned by manifest-list digest (Docker Hub, 2026-11-03).
# devel (nvcc) only for the optional causal-conv1d build; triton needs the gcc from build-essential.
BASE_IMAGE = "nvidia/cuda@sha256:a99a1860ba8e2916e5c3e73b72ec4c4301653a84586e05bfc9a2aa2d58027e97"
# Every package, exact version and sha256 (uv pip compile of infra/p0b-requirements.in).
# pip enters hash-checking mode, so an unlisted or changed wheel fails the CPU image build.
LOCK_FILE = Path(__file__).resolve().with_name("p0b-requirements.lock.txt")

image = (
    modal.Image.from_registry(BASE_IMAGE, add_python="3.11")
    .apt_install("git", "build-essential")
    .pip_install_from_requirements(str(LOCK_FILE))
    .run_commands("python -m pip check")
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false",
          "EMBER_IMAGE_LOCK_SHA256": hashlib.sha256(LOCK_FILE.read_bytes().replace(b"\r\n", b"\n")).hexdigest()})
)
if os.environ.get("EMBER_FAST_KERNELS") == "1":
    # NOT locked and built on a GPU builder (billed). Not part of the first inspect.
    image = image.pip_install("flash-linear-attention", "causal-conv1d", gpu="L4")
# Last build step, CPU only: audited upstream bytes + exact pinned versions, else the build fails.
image = (image.add_local_python_source("ember_v1", copy=True)
         .run_commands("python -m ember_v1.p0b verify-install"))

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("ember-v1-hf-cache", create_if_missing=True)
checkpoints = modal.Volume.from_name("ember-v1-checkpoints", create_if_missing=True)
VOLUMES = {"/cache": hf_cache, "/ckpt": checkpoints}


# Bounds for direct callers too; the entrypoint sets `timeout` per task (p0b.gpu_run_plan).
@app.cls(image=image, gpu="L4", volumes=VOLUMES, timeout=TASK_TIMEOUT_S["inspect"], startup_timeout=STARTUP_TIMEOUT_S,
         scaledown_window=SCALEDOWN_WINDOW_S, retries=RETRIES, max_containers=MAX_CONTAINERS)
class P0BHarness:
    model_name: str = modal.parameter(default=DEFAULT_MODEL)
    revision: str = modal.parameter(default=PINNED_MODEL_COMMIT)
    precision: str = modal.parameter(default="bf16")
    max_seq: int = modal.parameter(default=16384)
    code_commit: str = modal.parameter(default="")

    @modal.enter()
    def load(self):
        from ember_v1 import p0b

        self.model, self.tokenizer = p0b.load_model(self.model_name, self.max_seq, self.precision,
                                                    revision=self.revision)
        self.metadata = p0b.runtime_metadata(
            self.model, self.tokenizer, self.model_name, self.precision, self.max_seq,
            evidence_source="modal:P0BHarness", code_commit=self.code_commit or None,
            requested_revision=self.revision)
        hf_cache.commit()

    @modal.method()
    def inspect(self, question_style: str) -> dict:
        from ember_v1 import p0b

        return {"metadata": self.metadata,
                **p0b.task_inspect(self.model, self.tokenizer, question_style, metadata=self.metadata)}

    @modal.method()
    def smoke(self, ns: list[int], repeats: int, question_style: str) -> dict:
        from ember_v1 import p0b

        return {"metadata": self.metadata, **p0b.task_smoke(self.model, self.tokenizer, ns, repeats, question_style)}

    @modal.method()
    def padding(self, ns: list[int], question_style: str) -> dict:
        from ember_v1 import p0b

        return p0b.task_padding(self.model, self.tokenizer, ns, question_style, self.metadata)

    @modal.method()
    def throughput(self, batch_sizes: list[int], ns: list[int], repeats: int, question_style: str) -> dict:
        from ember_v1 import p0b

        return {"metadata": self.metadata,
                **p0b.task_throughput(self.model, self.tokenizer, batch_sizes, ns, repeats, question_style)}


@app.cls(image=image, gpu="L4", volumes=VOLUMES, timeout=3600, scaledown_window=300)
class DecisionPolicy:
    """Rollout inference. Modal groups concurrent `.decide.remote()` calls from
    many simulator workers into one GPU batch (real cross-environment batching).

    `batched_forward=False` (default) loops the verified `predict()` path inside
    the Modal batch. `batched_forward=True` also needs `padding_evidence_path`: the
    full `padding` task result (on the /ckpt volume). `load()` re-runs the gate on
    its raw measurements and checks they match this exact stack; otherwise the
    container refuses to start. A verdict file alone is not accepted.
    `model_name` may be a LoRA checkpoint dir under /ckpt (P3+, verify loading).
    """
    model_name: str = modal.parameter(default=DEFAULT_MODEL)
    precision: str = modal.parameter(default="bf16")
    max_seq: int = modal.parameter(default=16384)
    batched_forward: bool = modal.parameter(default=False)
    padding_evidence_path: str = modal.parameter(default="")

    @modal.enter()
    def load(self):
        from ember_v1 import p0b

        self.model, self.tokenizer = p0b.load_model(self.model_name, self.max_seq, self.precision)
        self.metadata = p0b.runtime_metadata(self.model, self.tokenizer, self.model_name, self.precision,
                                             self.max_seq, evidence_source="modal:DecisionPolicy")
        self.batched_evidence = None
        if self.batched_forward:
            if not self.padding_evidence_path:
                raise RuntimeError("batched_forward=True needs padding_evidence_path; refusing to start")
            evidence = json.loads(Path(self.padding_evidence_path).read_text(encoding="utf-8"))
            self.batched_evidence = p0b.verify_batched_forward_evidence(evidence, self.metadata)

    @modal.batched(max_batch_size=32, wait_ms=10)
    def decide(self, requests: list[dict]) -> list[dict]:
        from ember_v1 import p0b

        return p0b.decide_batch(self.model, self.tokenizer, requests, metadata=self.metadata,
                                batched_forward=self.batched_forward, batched_evidence=self.batched_evidence)


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _code_commit() -> str:
    """Local git HEAD (+ "-dirty" for uncommitted changes); "" when git is unavailable."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return ""
    return head + ("-dirty" if dirty.strip() else "")


@app.local_entrypoint()
def main(task: str = "inspect", gpu: str = "L4", model: str = DEFAULT_MODEL, precision: str = "bf16",
         max_seq: int = 16384, question_style: str = "compact", repeats: int = 25,
         ns: str = "1,10,50,100,200", batch_sizes: str = "1,4,8,16,32,64", out_dir: str = "results",
         authorize: str = ""):
    from ember_v1.p0b import gpu_run_plan

    # Runs after `modal run` has created the ephemeral app (the CPU image build may already have
    # happened) but before any GPU container starts: authorization, clean commit, pinned model, GPU.
    try:
        plan = gpu_run_plan(task, gpu, model, precision, authorize, _code_commit())
    except (ValueError, PermissionError) as exc:
        raise SystemExit(f"refused: {exc}")
    from ember_v1.run_report import call_remote_reported, write_json_atomic

    print(json.dumps({"plan": plan}))
    harness = P0BHarness.with_options(
        gpu=gpu, timeout=plan["timeout_s"], retries=plan["retries"], max_containers=plan["max_containers"],
        scaledown_window=plan["scaledown_window_s"])(model_name=model, revision=plan["revision"],
                                                     precision=precision, max_seq=max_seq,
                                                     code_commit=plan["code_commit"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{task}-{gpu}-{precision}-{question_style}"
    path = out / f"{tag}.json"
    calls = {"inspect": lambda: harness.inspect.remote(question_style),
             "smoke": lambda: harness.smoke.remote(_ints(ns), repeats, question_style),
             "throughput": lambda: harness.throughput.remote(_ints(batch_sizes), _ints(ns), repeats, question_style),
             "padding": lambda: harness.padding.remote(_ints(ns), question_style)}
    # The container starts on this call. Load errors, remote exceptions, timeouts and Ctrl-C
    # give a FAIL/UNVERIFIED report at `path` (replacing any older result for this tag).
    result, elapsed = call_remote_reported(calls[task], plan=plan, path=path)
    if result is None:
        print(json.dumps({"task": task, "gpu": gpu, "status": "NOT_PASS", "file": str(path)}))
        raise SystemExit(2)
    if task == "padding":
        from ember_v1.padding_gate import assess_padding_parity

        verdicts = {}
        for order, doc in result["gate_inputs"].items():
            (out / f"{tag}-{order}-input.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
            try:
                verdicts[order] = assess_padding_parity(doc)
            except ValueError as exc:  # invalid evidence: keep the report, never PASS
                verdicts[order] = {"status": "INVALID", "error": str(exc)}
        result["verdicts"] = verdicts
        result["status"] = "PASS" if all(v["status"] == "PASS" for v in verdicts.values()) and all(
            a == 1.0 for a in result["adapter_agreement_with_predict"].values()) else "FAIL"
    # Client wall clock of the remote call (excludes image build). Compare it with the Modal
    # usage page; `plan.cost_estimate` is an estimate, not a bill.
    result["run"] = {"plan": plan, "client_elapsed_s": round(elapsed, 1)}
    write_json_atomic(path, result)
    from ember_v1.p0b import result_exit_code

    # The report is written above first, so it survives a FAIL or UNVERIFIED exit.
    print(json.dumps({"task": task, "gpu": gpu, "status": result.get("status"), "file": str(path)}))
    code = result_exit_code(result)  # 0 only for an explicit PASS
    if code:
        raise SystemExit(code)
