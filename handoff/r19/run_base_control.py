"""Untrained (plain Qwen3.5-2B) control through the V1 decision interface, on the frozen 148 scenarios.

No training, no fine-tuning: base encoder weights as pinned, LoRA freshly created, head freshly created with
the protocol seed. The arm code (`policy_arms`) and the simulator/evaluation code are the same objects used by
`r19-v1-arms@1`, so A/B/C here mean exactly what they mean there.

Usage: python run_base_control.py [--smoke]
"""
import argparse, json, os, pathlib, sys, time

import torch

from ember_v1 import student_train as st
from ember_v1.policy_arms import ARMS, run_arms

REPO = pathlib.Path(os.environ["EMBER_CHECKOUT"])
ROOT = pathlib.Path.home()/"ember-ws"
V1 = ROOT/"runs/r19-v1-arms@1/attempt-1/final"
OUT = ROOT/"runs/r19-base-control@1"
ARMSET = list(ARMS)


class Guards:
    def __init__(self, cap_s):
        self.t0, self.cap_s = time.monotonic(), cap_s

    def __call__(self):
        from types import SimpleNamespace
        return SimpleNamespace(wall_cap_s=self.cap_s - (time.monotonic() - self.t0), usd_cap=float("inf"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    ident = json.loads((ROOT/"artifacts/models/1fc405f723718e308c8a22b137fee4a025086fde/v1/p1-18-v1-train@1/identity.json").read_text())
    proto = st.load_protocol(REPO/ident["protocol"]["path"])
    if proto["protocol_hash"] != ident["protocol"]["protocol_hash"]:
        raise SystemExit("protocol hash differs from the model identity")
    cfg = st.config_from_protocol(proto)

    scn = V1/"scenarios.jsonl"
    rows = [json.loads(l) for l in scn.read_text().splitlines() if l.strip()]
    out = OUT/("smoke" if a.smoke else "run")
    out.mkdir(parents=True, exist_ok=True)
    lst = out/"scenarios.jsonl"
    if a.smoke:
        lst.write_text(json.dumps(rows[0], sort_keys=True)+"\n", encoding="utf-8", newline="\n")
        rows = rows[:1]
    else:
        lst.write_text(scn.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")

    torch.manual_seed(cfg["seed"])
    t = time.monotonic()
    ad = st.ClefLoraAdapter(cfg, checkpoint=None)
    from unsloth import FastDecisionModel
    FastDecisionModel.for_inference(ad.model)
    load_s = round(time.monotonic()-t, 1)

    head = getattr(ad.model, "head", None)
    from ember_v1 import p0b
    head_sha = p0b._head_sha256(head) if head is not None else None
    lora_zero = {}
    for n, p in ad.model.named_parameters():
        if "lora_" in n:
            lora_zero[n] = bool(torch.count_nonzero(p) == 0)
    zero_n = sum(lora_zero.values())
    untrained = {
        "definition": "base encoder = pinned public weights (unchanged); LoRA r16 created fresh; head created fresh "
                      "with protocol seed; zero training steps",
        "seed": cfg["seed"], "load_s": load_s,
        "head_sha256_now": head_sha,
        "head_init_sha256_a10_in_identity": ident["runtime_pins"]["head_init_sha256_a10"],
        "head_matches_a10_init": head_sha == ident["runtime_pins"]["head_init_sha256_a10"],
        "lora_tensors": len(lora_zero), "lora_tensors_all_zero": zero_n,
        "trainable": ad.trainable,
        "precision": cfg["precision"], "max_len": cfg["max_len"], "chunk_k": cfg["chunk_k"],
        "question_style": cfg["question_style"], "encoding": cfg["encoding"],
        "base_model": ident["base_model"]["repo"]+"@"+ident["base_model"]["revision"],
        "peak_memory_gb": round(ad.memory_bytes()/2**30, 3) if hasattr(ad, "memory_bytes") else None,
    }
    print(json.dumps(untrained, indent=1))
    (out/"untrained_identity.json").write_text(json.dumps(untrained, indent=1, sort_keys=True)+"\n", encoding="utf-8", newline="\n")

    guards = Guards(6*3600 if not a.smoke else 1800)
    t = time.monotonic()
    res = run_arms(ad, cfg, lst, out, arms=ARMSET, label="BASE/untrained", guards=guards,
                   block=int(os.environ.get("EMBER_BLOCK", "4")), log=lambda m: print(m, flush=True))
    elapsed = round(time.monotonic()-t, 1)
    metrics = {"kind": "base-control", "arms": res["arms"], "scenarios": res["scenarios"],
               "stopped": res["stopped"], "elapsed_s": elapsed,
               "verdict": "COMPLETE" if all(r["finished"] == res["scenarios"] for r in res["arms"].values()) else "INCOMPLETE",
               "untrained": untrained}
    (out/"metrics.json").write_text(json.dumps(metrics, indent=1, sort_keys=True)+"\n", encoding="utf-8", newline="\n")
    print("VERDICT", metrics["verdict"], "in", elapsed, "s")


if __name__ == "__main__":
    main()
