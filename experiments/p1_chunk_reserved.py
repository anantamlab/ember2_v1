"""CPU-only reserved-chunk coordination study (`t1_chunk_reserved`) with the pinned Qwen3.5-2B tokenizer.

Coordination research for the multi-teacher student (P1-14 §4), not a return to T1-only learning. T1 on one
chunk's sub-observation stands in for one student row. No model, no GPU, no label, no TRAIN/DEV/P1-09/test or
EMBERBench scenario: rooms come from `chunk_joint.synthetic_scenario` with the study's own tag and new seeds.
The 1.0.0 study (`data/training/p1-11-student@1/chunk_study.json`) is kept unchanged.

Usage:
    PYTHONPATH=src python experiments/p1_chunk_reserved.py <tokenizer.json> <out.json> [workers]
"""
import hashlib
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ember_v1 import chunk_joint as cj
from ember_v1 import token_budget as tb

TOKENIZER_SHA256 = "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4"
A10_TOKENIZER_STATE_SHA256 = "a57b5c62cd5f242a06c632b82cc0b4f8239d08cc80fd9347fc1f629be977c6ca"
NS = {50: 100, 100: 120, 110: 120, 200: 160}          # N -> max_steps (as the 1.0.0 study)
KS = (50, 80, 100)
FIRES = ("deterministic", "stochastic")
SEEDS = (10, 11, 12, 13, 14)                          # new seeds; the 1.0.0 study used 0-2 with another tag
CHUNKED = ("t1_chunk_static", "t1_chunk_reserved", "t1_chunk_reserved_fix")
ALWAYS = ("t1_chunk_exitalloc",)                      # exit groups make several rows even when k >= N
_TOK = None


def _init(tok_path: str) -> None:
    global _TOK
    from tokenizers import Tokenizer
    hf = Tokenizer.from_str(Path(tok_path).read_text(encoding="utf-8"))
    cache: dict[str, int] = {}

    def tok(text):            # pure function of the text; the cache only saves time
        n = cache.get(text)
        if n is None:
            n = cache[text] = len(hf.encode(text, add_special_tokens=False).ids)
        return range(n)
    _TOK = tok


def _scenario_runs(job):
    n, fire, seed = job
    sc = cj.synthetic_scenario(n, seed=seed, fire_model=fire, max_steps=NS[n], tag=cj.RESERVED_STUDY_VERSION)
    # t1_joint decides on the whole state whatever k is; k only sets the rows whose tokens are checked. One row
    # of 200 people exceeds 16384 tokens, so the N=200 reference is checked with k=100 rows (as the 1.0.0 study).
    plan = [("t1_joint", n if n <= 110 else 100)] + [(p, k) for k in KS for p in CHUNKED if k < n] + \
        [(p, k) for k in KS for p in ALWAYS]
    out = []
    for p, k in plan:
        ts = time.time()
        r = cj.run_episode(sc, p, k, _TOK)
        r["scenario_hash"] = sc.content_hash()
        r["wall_s"] = round(time.time() - ts, 2)
        out.append(r)
    return out


def main(tok_path: str, out_path: str, workers: int) -> None:
    data = Path(tok_path).read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if sha != TOKENIZER_SHA256:
        raise SystemExit(f"tokenizer sha256 {sha} != pinned {TOKENIZER_SHA256}")
    from tokenizers import Tokenizer
    state_sha = hashlib.sha256(Tokenizer.from_str(data.decode("utf-8")).to_str().encode("utf-8")).hexdigest()
    if state_sha != A10_TOKENIZER_STATE_SHA256:
        raise SystemExit(f"tokenizer state {state_sha} != A10 metadata.tokenizer_sha256")
    jobs = [(n, f, s) for n in NS for f in FIRES for s in SEEDS]
    t0 = time.time()
    partial = Path(out_path + ".partial.jsonl")         # every finished scenario is saved before anything else
    partial.parent.mkdir(parents=True, exist_ok=True)
    runs = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(tok_path,)) as ex, \
            partial.open("w", encoding="utf-8", newline="\n") as fh:
        for rs in ex.map(_scenario_runs, jobs):
            for r in rs:
                fh.write(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n")
            fh.flush()
            runs.extend(rs)
            for r in rs:
                print(json.dumps({x: r.get(x) for x in ("scenario", "k", "predictor", "complete", "evacuated",
                                                        "casualties", "unresolved", "cross_chunk_blocked",
                                                        "max_row_tokens", "errors")}), flush=True)
    head = {"study_version": cj.RESERVED_STUDY_VERSION, "tokenizer": {"sha256": sha, "state_sha256": state_sha},
            "limit": tb.SERVE_MAX_LEN, "ns_max_steps": NS, "ks": KS, "fires": FIRES, "seeds": SEEDS,
            "methods": ["t1_joint", *CHUNKED, *ALWAYS], "workers": workers,
            "latency_note": "proxy compute on a shared CPU with parallel workers; not student or GPU latency",
            "host": {"python": platform.python_version(), "platform": platform.platform(), "cpus": os.cpu_count()},
            "wall_s": round(time.time() - t0, 1), "summary_vs_t1_joint": cj.paired_summary(runs),
            "summary_vs_t1_chunk_static": cj.paired_summary(runs, baseline="t1_chunk_static")}
    body = json.dumps(head, indent=1, sort_keys=True)[:-2]       # strip the closing brace, append runs one per line
    lines = ",\n".join(json.dumps(r, sort_keys=True, separators=(",", ":")) for r in runs)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(body + ',\n "runs": [\n' + lines + "\n ]\n}\n", encoding="utf-8", newline="\n")
    partial.unlink()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4)
