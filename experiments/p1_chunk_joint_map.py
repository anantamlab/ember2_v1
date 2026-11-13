"""CPU-only chunked-inference study (P1-11 R7) with the pinned Qwen3.5-2B tokenizer. No GPU, no model, no label.

Independent synthetic rooms only (`chunk_joint.synthetic_scenario`); no TRAIN, DEV, held-out or benchmark
scenario is read. At EVERY observation of every episode: chunk rows are built by `token_budget.chunk_rows`,
each active person must appear once, every row must carry the full public state and encode below 16384
tokens, the joint map is assembled from per-chunk predictions and resolved by the public movement resolver,
and the decision is compared with the unchunked T1 decision on the same observation.

Usage:
    PYTHONPATH=src python experiments/p1_chunk_joint_map.py <tokenizer.json> <out.json>
"""
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

from tokenizers import Tokenizer

from ember_v1 import chunk_joint as cj
from ember_v1 import token_budget as tb

TOKENIZER_SHA256 = "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4"
A10_TOKENIZER_STATE_SHA256 = "a57b5c62cd5f242a06c632b82cc0b4f8239d08cc80fd9347fc1f629be977c6ca"
# (N, k values, max_steps). k = N is the unchunked single row.
GRID = [(50, (25, 50), 100), (100, (50, 100), 120), (110, (100, 110), 120), (200, (100,), 160)]
FIRES = ("deterministic", "stochastic")
SEEDS = (0, 1, 2)
PREDICTORS = ("t1_joint", "t1_chunk_static", "t1_chunk_blind")


def main(tok_path: str, out_path: str) -> None:
    data = Path(tok_path).read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if sha != TOKENIZER_SHA256:
        raise SystemExit(f"tokenizer sha256 {sha} != pinned {TOKENIZER_SHA256}")
    hf = Tokenizer.from_str(data.decode("utf-8"))
    state_sha = hashlib.sha256(hf.to_str().encode("utf-8")).hexdigest()
    if state_sha != A10_TOKENIZER_STATE_SHA256:
        raise SystemExit(f"tokenizer state {state_sha} != A10 metadata.tokenizer_sha256")
    cache: dict[str, int] = {}

    def tok(text):            # pure function of the text; the cache only saves time
        n = cache.get(text)
        if n is None:
            n = cache[text] = len(hf.encode(text, add_special_tokens=False).ids)
        return range(n)

    runs, t0 = [], time.time()
    for n, ks, max_steps in GRID:
        for fire in FIRES:
            for seed in SEEDS:
                sc = cj.synthetic_scenario(n, seed=seed, fire_model=fire, max_steps=max_steps)
                for k in ks:
                    for p in PREDICTORS:
                        if k >= n and p != "t1_joint":
                            continue          # one chunk: every predictor equals t1_joint
                        ts = time.time()
                        r = cj.run_episode(sc, p, k, tok)
                        r["scenario_hash"] = sc.content_hash()
                        r["wall_s"] = round(time.time() - ts, 2)
                        runs.append(r)
                        print(json.dumps({x: r[x] for x in ("scenario", "k", "predictor", "complete", "evacuated",
                                                             "casualties", "unresolved", "blocked",
                                                             "max_row_tokens", "wall_s")}), flush=True)
    out = {"study_version": cj.STUDY_VERSION, "tokenizer": {"sha256": sha, "state_sha256": state_sha},
           "limit": tb.SERVE_MAX_LEN, "grid": GRID, "fires": FIRES, "seeds": SEEDS, "predictors": PREDICTORS,
           "host": {"python": platform.python_version(), "platform": platform.platform()},
           "wall_s": round(time.time() - t0, 1), "runs": runs}
    Path(out_path).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
