"""OPTIONAL GPU Phase 0B on a local CUDA host (Modal users: `infra/modal_app.py`).

Example:
 python experiments/phase0b_unsloth_smoke.py \
   --model unsloth/Qwen3.5-2B --max-seq 16384 --precision bf16 --question-style compact

Requires Unsloth, transformers, a compatible NVIDIA GPU and model download.
Records build_dataset/truncation stats and predict latency; no benchmark episodes.
All logic lives in `ember_v1.p0b` (shared with the Modal harness).
"""
from __future__ import annotations

import argparse
import json

from ember_v1 import p0b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unsloth/Qwen3.5-2B")
    parser.add_argument("--max-seq", type=int, default=16384)
    parser.add_argument("--precision", choices=("4bit", "bf16"), default="bf16")
    parser.add_argument("--question-style", choices=("verbose", "compact"), default="compact")
    parser.add_argument("--repeats", type=int, default=25)
    args = parser.parse_args()
    model, tokenizer = p0b.load_model(args.model, args.max_seq, args.precision)
    report = p0b.task_smoke(model, tokenizer, (1, 10, 50, 100, 200), args.repeats, args.question_style)
    report["metadata"] = p0b.runtime_metadata(model, args.model, args.precision, args.max_seq)
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
