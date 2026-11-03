"""Modal GPU harness: P0B feasibility tasks + batched decision-policy serving.

Prerequisites (local): `pip install -e '.[modal]'` then `modal setup`.

P0B tasks (results land in ./results, padding verdicts are computed locally):
  modal run infra/modal_app.py --task inspect                  # characterize Unsloth internals FIRST
  modal run infra/modal_app.py --task smoke --gpu L4
  modal run infra/modal_app.py --task padding --gpu L4 --ns 1,10,50,200
  modal run infra/modal_app.py --task throughput --gpu L40S --batch-sizes 1,4,8,16,32,64

Serving (CPU simulator/agent locally or in CPU containers, GPU inference here):
  modal deploy infra/modal_app.py
  # then ember_v1.agent.modal_predictor(...) -> DecisionAgent(predictor)

Fast DeltaNet/causal-conv kernels are an opt-in image layer because
causal-conv1d compiles from source:  EMBER_FAST_KERNELS=1 modal run ...
Run P0B both with and without it; record which backend actually executed.
"""
# No `from __future__ import annotations`: Modal reads real class-parameter types.
import json
import os
import time
from pathlib import Path

import modal

APP_NAME = "ember-v1"
DEFAULT_MODEL = "unsloth/Qwen3.5-2B"
# Pins follow the Unsloth Qwen3.5 Decision notebook. Replace "unsloth"/"unsloth_zoo"
# with exact `==` versions after the first PASSing `inspect`+`smoke` run (see
# runtime_versions in results) and never change them inside a phase.
UNSLOTH_SPECS = ("unsloth", "unsloth_zoo")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential")
    .pip_install("torch==2.8.0", "torchvision==0.23.0", "triton>=3.3.0", "xformers==0.0.32.post2", "bitsandbytes")
    .pip_install(*UNSLOTH_SPECS, "transformers==5.2.0", "trl==0.22.2",
                 "tokenizers>=0.22.0,<=0.23.0", "torchao>=0.16.0")
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
)
if os.environ.get("EMBER_FAST_KERNELS") == "1":
    image = image.pip_install("flash-linear-attention", "causal-conv1d", gpu="L4")
image = image.add_local_python_source("ember_v1")

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("ember-v1-hf-cache", create_if_missing=True)
checkpoints = modal.Volume.from_name("ember-v1-checkpoints", create_if_missing=True)
VOLUMES = {"/cache": hf_cache, "/ckpt": checkpoints}


@app.cls(image=image, gpu="L4", volumes=VOLUMES, timeout=3600, scaledown_window=120)
class P0BHarness:
    model_name: str = modal.parameter(default=DEFAULT_MODEL)
    precision: str = modal.parameter(default="bf16")
    max_seq: int = modal.parameter(default=16384)

    @modal.enter()
    def load(self):
        from ember_v1 import p0b

        self.model, self.tokenizer = p0b.load_model(self.model_name, self.max_seq, self.precision)
        self.metadata = p0b.runtime_metadata(self.model, self.model_name, self.precision, self.max_seq)
        hf_cache.commit()

    @modal.method()
    def inspect(self, question_style: str) -> dict:
        from ember_v1 import p0b

        return {"metadata": self.metadata, **p0b.task_inspect(self.model, self.tokenizer, question_style)}

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
    the Modal batch. Set True ONLY after the P0B padding gate PASSes for the
    pinned stack; the padded path then runs one forward per batch.
    `model_name` may be a LoRA checkpoint dir under /ckpt (P3+, verify loading).
    """
    model_name: str = modal.parameter(default=DEFAULT_MODEL)
    precision: str = modal.parameter(default="bf16")
    max_seq: int = modal.parameter(default=16384)
    batched_forward: bool = modal.parameter(default=False)

    @modal.enter()
    def load(self):
        from ember_v1 import p0b

        self.model, self.tokenizer = p0b.load_model(self.model_name, self.max_seq, self.precision)
        self.metadata = p0b.runtime_metadata(self.model, self.model_name, self.precision, self.max_seq)

    @modal.batched(max_batch_size=32, wait_ms=10)
    def decide(self, requests: list[dict]) -> list[dict]:
        from ember_v1 import p0b

        t0 = time.perf_counter()
        if not self.batched_forward:
            out = []
            for req in requests:
                res = p0b.predict_answers(self.model, self.tokenizer, req)
                if res["truncated"]:
                    raise RuntimeError("input truncated: fail closed")
                out.append({"answers": res["answers"], "path": "predict"})
        else:
            records = []
            for req in requests:
                # gold is a dummy needed only by build_dataset; it does not affect scores.
                row = {**req, "gold": {pid: "stay" for pid in req["questions"]}}
                record, entry = p0b.encode_one(row, self.tokenizer, self.model)
                if record is None or entry["truncated"]:
                    raise RuntimeError(f"encoding failed or truncated: {entry['status']}")
                records.append(record)
            probs = p0b.batched_probabilities(self.model, self.tokenizer, records, [r["questions"] for r in requests])
            out = [{"answers": {pid: {"answer": min(p, key=lambda a: (-p[a], a)), "probabilities": p}
                                for pid, p in per.items()}, "path": "batched_right_pad"} for per in probs]
        gpu_s = time.perf_counter() - t0
        for item in out:
            item.update(batch_size=len(requests), server_s=gpu_s, model_revision=self.metadata["model_revision"])
        return out


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


@app.local_entrypoint()
def main(task: str = "smoke", gpu: str = "L4", model: str = DEFAULT_MODEL, precision: str = "bf16",
         max_seq: int = 16384, question_style: str = "compact", repeats: int = 25,
         ns: str = "1,10,50,100,200", batch_sizes: str = "1,4,8,16,32,64", out_dir: str = "results"):
    harness = P0BHarness.with_options(gpu=gpu)(model_name=model, precision=precision, max_seq=max_seq)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{task}-{gpu}-{precision}-{question_style}"
    if task == "inspect":
        result = harness.inspect.remote(question_style)
    elif task == "smoke":
        result = harness.smoke.remote(_ints(ns), repeats, question_style)
    elif task == "throughput":
        result = harness.throughput.remote(_ints(batch_sizes), _ints(ns), repeats, question_style)
    elif task == "padding":
        from ember_v1.padding_gate import assess_padding_parity

        result = harness.padding.remote(_ints(ns), question_style)
        verdicts = {}
        for order, doc in result["gate_inputs"].items():
            (out / f"{tag}-{order}-input.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
            verdicts[order] = assess_padding_parity(doc)
        result["verdicts"] = verdicts
        result["status"] = "PASS" if all(v["status"] == "PASS" for v in verdicts.values()) and all(
            a == 1.0 for a in result["adapter_agreement_with_predict"].values()) else "FAIL"
    else:
        raise SystemExit(f"unknown task {task!r}: use inspect|smoke|padding|throughput")
    path = out / f"{tag}.json"
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"task": task, "gpu": gpu, "status": result.get("status", "SEE_FILE"), "file": str(path)}))
    if result.get("status") == "FAIL":
        raise SystemExit(2)
