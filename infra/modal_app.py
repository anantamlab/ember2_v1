"""Modal GPU harness: P0B feasibility tasks + batched decision-policy serving.

Prerequisites (local): `pip install -e '.[modal]'` then `modal setup`.

P0B tasks (results land in ./results, padding verdicts are computed locally). Every GPU
task needs `--authorize <task>` (the same task name), a clean git checkout, the pinned
model, and runs under the timeouts of `p0b.gpu_run_plan`. A failed remote call writes a
FAIL/UNVERIFIED report (`ember_v1.run_report`); an abrupt kill or a failed image build
writes none (use Modal logs). First command (issue #2):
  modal run infra/modal_app.py --task inspect --gpu L4 --authorize inspect
If L4 has no capacity (queue_timeout, cancel checked), one separate attempt each, never concurrent:
  modal run infra/modal_app.py --task inspect --gpu A10 --authorize inspect
  modal run infra/modal_app.py --task inspect --gpu L40S --authorize inspect
Later tasks need their own written authorization (docs/P0B_GPU_INSPECT.md):
  modal run infra/modal_app.py --task smoke --gpu L4 --authorize smoke
  modal run infra/modal_app.py --task padding --gpu L4 --ns 1,10,50,200 --authorize padding
  modal run infra/modal_app.py --task throughput --gpu L40S --batch-sizes 1,4,8,16,32,64 --authorize throughput
P1-11 chunk smoke (NOT RUN; needs its own written authorization, docs/P1_CHUNK_SMOKE.md):
  modal run infra/modal_app.py --task chunk_smoke --gpu A10 --repeats 5 --authorize chunk_smoke

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


def _image_env() -> dict:
    """Image env. The lock hash is read on the local client only: Modal imports this module again in the
    container (/root/modal_app.py), where the lock file does not exist and the image is already built
    (A10 attempt, 2026-11-08: FileNotFoundError at import, 13 failed startups)."""
    env = {"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"}
    if modal.is_local():
        env["EMBER_IMAGE_LOCK_SHA256"] = hashlib.sha256(LOCK_FILE.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return env


image = (
    modal.Image.from_registry(BASE_IMAGE, add_python="3.11")
    .apt_install("git", "build-essential")
    .pip_install_from_requirements(str(LOCK_FILE))
    .run_commands("python -m pip check")
    .env(_image_env())
)
if os.environ.get("EMBER_FAST_KERNELS") == "1":
    # NOT locked and built on a GPU builder (billed). Not part of the first inspect.
    image = image.pip_install("flash-linear-attention", "causal-conv1d", gpu="L4")
# Last build step, CPU only: audited upstream bytes + exact pinned versions, else the build fails.
# The source is copied to /root/ember_v1. Modal adds /root to sys.path only at function run time,
# not in builder RUN steps (first inspect, 2026-11-03: ModuleNotFoundError), so set PYTHONPATH here.
image = (image.add_local_python_source("ember_v1", copy=True)
         .run_commands("PYTHONPATH=/root python -m ember_v1.p0b verify-install"))

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

    @modal.method()
    def chunk_smoke(self, repeats: int) -> dict:
        """P1-11 R3/R4/R7: kernel provenance, compact+verbose N=1/10/50/100, N=200 refusal, chunked N=200.
        Inference only, batch 1 (docs/P1_CHUNK_SMOKE.md)."""
        import torch

        from ember_v1 import p0b
        from ember_v1.p0b_chunk_smoke import chunk_smoke

        torch.cuda.reset_peak_memory_stats()
        return {"metadata": self.metadata,
                **chunk_smoke(self.model, self.tokenizer, encode_one=p0b.encode_one,
                              predict_answers=p0b.predict_answers, synthetic_row=p0b.synthetic_row,
                              timed=p0b._timed, repeats=repeats,
                              peak_memory_gb=lambda: round(torch.cuda.max_memory_allocated() / 2**30, 3))}


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


@app.cls(image=image, gpu="A10", volumes=VOLUMES, timeout=3600, startup_timeout=STARTUP_TIMEOUT_S,
         scaledown_window=SCALEDOWN_WINDOW_S, retries=RETRIES, max_containers=MAX_CONTAINERS, cpu=4.0, memory=32768)
class StudentTrainer:
    """P1-18 student phases (pilot / train / eval) of the FROZEN p1-14-student-train protocol.
    Data and outputs live on the `ember-v1-checkpoints` volume (/ckpt); the volume is committed at every
    checkpoint, so a stopped train run resumes from its last checkpoint (student_run.gpu_phase)."""
    code_commit: str = modal.parameter(default="")

    @modal.method()
    def run(self, phase: str, protocol_json: str, spec_json: str) -> dict:
        from ember_v1 import student_run as sr
        from ember_v1 import student_train as st

        checkpoints.reload()
        spec = {**json.loads(spec_json), "code_commit": self.code_commit}
        return sr.gpu_phase(phase, json.loads(protocol_json), spec,
                            adapter_factory=lambda cfg, checkpoint=None: st.ClefLoraAdapter(cfg, checkpoint=checkpoint),
                            persist=checkpoints.commit, image_lock_sha256=os.environ.get("EMBER_IMAGE_LOCK_SHA256"))


@app.local_entrypoint()
def student(phase: str, spec: str, authorize: str = "", out: str = "", protocol: str = "", spawn: bool = False):
    """modal run [--detach] infra/modal_app.py::student --phase pilot --spec <spec.json> --authorize pilot
    [--protocol data/protocols/p1-18-student-v2.json] (default: the V1 protocol p1-14-student-train.json)
    [--spawn]: with --detach, start the call and return at once (results on the volume), so a later local
    network error cannot tear the app down."""
    from ember_v1 import student_train as st
    from ember_v1.run_report import write_json_atomic

    protocol = st.load_protocol(protocol) if protocol else st.load_protocol()
    try:
        plan = st.gpu_train_plan(phase, protocol, authorize=authorize, code_commit=_code_commit())
    except (ValueError, PermissionError) as exc:
        raise SystemExit(f"refused: {exc}")
    print(json.dumps({"plan": plan}))
    trainer = StudentTrainer.with_options(gpu=plan["gpu"], timeout=plan["timeout_s"], retries=plan["retries"],
                                          max_containers=plan["max_containers"])(code_commit=plan["code_commit"])
    if spawn:
        call = trainer.run.spawn(phase, json.dumps(protocol), Path(spec).read_text(encoding="utf-8"))
        print(json.dumps({"spawned": call.object_id}))
        return
    result = trainer.run.remote(phase, json.dumps(protocol), Path(spec).read_text(encoding="utf-8"))
    if out:
        write_json_atomic(Path(out), result)
    print(json.dumps({k: result.get(k) for k in ("phase", "elapsed_s", "go", "reload")}, default=str))


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
    from ember_v1.p0b import allocated_gpu, gpu_run_plan

    # Runs after `modal run` has created the ephemeral app (the CPU image build may already have
    # happened) but before any GPU container starts: authorization, clean commit, pinned model, GPU.
    try:
        plan = gpu_run_plan(task, gpu, model, precision, authorize, _code_commit())
    except (ValueError, PermissionError) as exc:
        raise SystemExit(f"refused: {exc}")
    from ember_v1.run_report import watch_remote_call, write_json_atomic

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
    submits = {"inspect": lambda: harness.inspect.spawn(question_style),
               "smoke": lambda: harness.smoke.spawn(_ints(ns), repeats, question_style),
               "throughput": lambda: harness.throughput.spawn(_ints(batch_sizes), _ints(ns), repeats, question_style),
               "padding": lambda: harness.padding.spawn(_ints(ns), question_style),
               "chunk_smoke": lambda: harness.chunk_smoke.spawn(repeats)}
    # One spawn(), then bounded get(timeout=) polls with get_current_stats() snapshots of this
    # variant (public Modal 1.6.3 APIs). A confirmed queue wait over 300 s, or the overall client
    # deadline, cancels the call explicitly (terminate_containers=True). Load errors, remote
    # exceptions, timeouts and Ctrl-C give a FAIL/UNVERIFIED report at `path`; a provisional
    # record with the FunctionCall ID replaces any older result right after submission.
    result, watch = watch_remote_call(submits[task], plan=plan, path=path,
                                      current_stats=getattr(harness, task).get_current_stats,
                                      app_id=app.app_id, log=print)
    elapsed = watch["elapsed_s"]
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
    result["run"] = {"plan": plan, "client_elapsed_s": round(elapsed, 1), "watchdog": watch,
                     "allocated_gpu": allocated_gpu(gpu, result.get("metadata"))}
    write_json_atomic(path, result)
    from ember_v1.p0b import result_exit_code

    # The report is written above first, so it survives a FAIL or UNVERIFIED exit.
    print(json.dumps({"task": task, "gpu": gpu, "status": result.get("status"), "file": str(path)}))
    code = result_exit_code(result)  # 0 only for an explicit PASS
    if code:
        raise SystemExit(code)
