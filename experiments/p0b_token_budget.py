"""CPU-only P0B G10 token budget: exact Clef token counts with the pinned Qwen3.5-2B tokenizer.

No GPU, no Modal, no model weights, no teacher call, no label. Only the tokenizer file of the pinned model
revision is read (`tokenizer.json`, sha256 checked). It counts the audited Clef `encode_record` layout
(`ember_v1.token_budget`) for:

1. the p0b synthetic rows (`p0b.synthetic_row`), validated against the A10 inspect record (N=1: 350, N=3: 617);
2. the largest synthetic N that fits 16384 tokens per question style;
3. the initial observations of p1-02-v1 TRAIN, DEV and calibration scenarios (locked_test is not read);
4. question chunking (`token_budget.max_chunk`) for the worst cases;
5. padding waste of three batching strategies on the real row-length distribution.

Usage (needs only the `tokenizers` package, not torch):
    PYTHONPATH=src <python with tokenizers==0.22.2> experiments/p0b_token_budget.py <tokenizer.json> <out.json>
"""
import hashlib
import json
import random
import sys
from pathlib import Path

from tokenizers import Tokenizer

from ember_v1 import p0b
from ember_v1 import token_budget as tb
from ember_v1.dataset import build_decision_row
from ember_v1.gen.splits import DEFAULT_PLAN, SplitPlan, build_splits
from ember_v1.sim import Simulator

TOKENIZER_SHA256 = "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4"
TOKENIZER_SOURCE = (f"https://huggingface.co/{p0b.PINNED_MODEL}/resolve/{p0b.PINNED_MODEL_COMMIT}/tokenizer.json")
# A10 record metadata.tokenizer_sha256: sha256 of backend_tokenizer.to_str() on the GPU host (tokenizers 0.22.2)
A10_TOKENIZER_STATE_SHA256 = "a57b5c62cd5f242a06c632b82cc0b4f8239d08cc80fd9347fc1f629be977c6ca"
A10_RECORD = {("compact", 1): 350, ("compact", 3): 617}   # docs/p0b_attempts/attempt7-A10-inspect-pass.json
SYNTH_NS = (1, 3, 10, 25, 50, 75, 100, 120, 150, 200)
STYLES = ("compact", "verbose")
ENCODINGS = ("ascii", "coords")
SPLITS = ("train", "dev", "calibration")    # locked_test is never read
LIMIT = tb.SERVE_MAX_LEN
PAD_K = 100                                 # chunk size for the padding study (fits every TRAIN row)


def _row(obs, enc, style, gid):
    acts = {p["id"]: "stay" for p in obs["people"] if p["status"] == "active"}
    return build_decision_row(obs, acts, group_id=gid, source="token-count-only", encoding=enc,
                              question_style=style)


def _max_synth_n(tok, style):
    lo, hi, best = 1, 572, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        r = p0b.synthetic_row(mid, style)
        if tb.clef_counts(tok, r["state"], r["questions"], LIMIT)["status"] == "OK":
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


def _padding(lengths, batch, seed=0):
    """Padded-token fraction per strategy. Real tokens are the same in all; only padding differs."""
    real = sum(lengths)
    rnd = list(lengths)
    random.Random(seed).shuffle(rnd)

    def padded(seq):
        return sum(max(seq[i:i + batch]) * len(seq[i:i + batch]) for i in range(0, len(seq), batch))

    def packed(seq, cap):
        free = []                                    # first-fit decreasing; each bin is one packed sequence
        for n in sorted(seq, reverse=True):
            for i, f in enumerate(free):
                if f >= n:
                    free[i] -= n
                    break
            else:
                free.append(cap - n)
        return len(free) * cap

    cap = LIMIT
    if max(lengths) > cap:
        raise ValueError("a row exceeds the pack capacity")
    return {"batch": batch, "rows": len(lengths), "real_tokens": real,
            "random_padded_fraction": round(1 - real / padded(rnd), 4),
            "length_sorted_padded_fraction": round(1 - real / padded(sorted(lengths)), 4),
            "packed_varlen_padded_fraction_upper": round(1 - real / packed(lengths, cap), 4),
            "pack_capacity": cap}


def main(tok_path, out_path):
    data = Path(tok_path).read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if sha != TOKENIZER_SHA256:
        raise SystemExit(f"tokenizer sha256 {sha} != pinned {TOKENIZER_SHA256}")
    hf = Tokenizer.from_str(data.decode("utf-8"))
    state_sha = hashlib.sha256(hf.to_str().encode("utf-8")).hexdigest()   # = p0b._tokenizer_sha256 definition
    if state_sha != A10_TOKENIZER_STATE_SHA256:
        raise SystemExit(f"tokenizer state {state_sha} != A10 metadata.tokenizer_sha256")

    def tok(text):
        return hf.encode(text, add_special_tokens=False).ids

    out = {"tokenizer": {"source": TOKENIZER_SOURCE, "sha256": sha, "state_sha256": state_sha,
                         "state_sha256_equals_a10": True},"clef_sha256": tb.CLEF_SHA256,
           "limit": LIMIT, "synthetic": [], "a10_validation": {}, "max_synthetic_n": {}, "real": {},
           "chunking": {}, "padding": {}}

    for style in STYLES:
        for n in SYNTH_NS:
            r = p0b.synthetic_row(n, style)
            c = tb.clef_counts(tok, r["state"], r["questions"], LIMIT)
            out["synthetic"].append({"n": n, "style": style, **c})
            if (style, n) in A10_RECORD:
                out["a10_validation"][f"{style}/N={n}"] = {"a10": A10_RECORD[(style, n)], "cpu": c["total"],
                                                           "equal": c["total"] == A10_RECORD[(style, n)]}
        out["max_synthetic_n"][style] = _max_synth_n(tok, style)
        r = p0b.synthetic_row(200, style)
        k = tb.max_chunk(tok, r["state"], r["questions"], LIMIT)
        out["chunking"][f"synthetic-200/{style}"] = {"max_k": k, "rows": -(-200 // k) if k else None}

    # Lower bound of the upstream schema format: instruction = ID only, options without description.
    # Counting only; this is not a dataset style.
    r = p0b.synthetic_row(200, "compact")
    bare = {i: {"type": "choice", "instructions": "", "criteria": sorted(q["criteria"])}
            for i, q in r["questions"].items()}
    c = tb.clef_counts(tok, r["state"], bare, LIMIT)
    out["minimal_schema_bound"] = {"n": 200, **c, "per_question_mean": round((c["schema"]) / 200, 2)}

    build = build_splits(SplitPlan.from_dict(dict(DEFAULT_PLAN)))
    lengths = {}
    for split in SPLITS:
        rows = []
        for g in build.scenarios[split]:
            obs = Simulator(g.scenario).observe()
            n = sum(p["status"] == "active" for p in obs["people"])
            rec = {"scenario": g.scenario.scenario_id, "people": n}
            for enc in ENCODINGS:
                for style in STYLES:
                    r = _row(obs, enc, style, g.scenario.scenario_id)
                    c = tb.clef_counts(tok, r["state"], r["questions"], LIMIT)
                    rec[f"{enc}/{style}"] = {k: c[k] for k in ("state", "fixed", "total", "status")}
            rows.append(rec)
            if split == "train":                     # chunked rows (k=PAD_K), each must fit: padding input
                r = _row(obs, "ascii", "compact", "pad")
                for ch in tb.chunk_rows(obs, r["gold"], k=PAD_K, group_id="pad", source="token-count-only"):
                    c = tb.clef_counts(tok, ch["state"], ch["questions"], LIMIT)
                    if c["status"] != "OK":
                        raise SystemExit(f"chunk k={PAD_K} does not fit: {g.scenario.scenario_id}")
                    lengths.setdefault("ascii/compact", []).append(c["total"])
        summ = {"scenarios": len(rows), "max_people": max(r["people"] for r in rows)}
        for enc in ENCODINGS:
            for style in STYLES:
                key = f"{enc}/{style}"
                tot = sorted(r[key]["total"] for r in rows)
                summ[key] = {"max_total": tot[-1], "median_total": tot[len(tot) // 2],
                             "max_state": max(r[key]["state"] for r in rows),
                             "max_fixed": max(r[key]["fixed"] for r in rows),
                             "not_ok": sum(r[key]["status"] != "OK" for r in rows)}
        worst = max(rows, key=lambda r: r["ascii/verbose"]["total"])
        summ["worst_scenario"] = {"scenario": worst["scenario"], "people": worst["people"]}
        out["real"][split] = summ
        if split == "train":
            g = next(x for x in build.scenarios[split] if x.scenario.scenario_id == worst["scenario"])
            obs = Simulator(g.scenario).observe()
            for style in STYLES:
                r = _row(obs, "ascii", style, "w")
                k = tb.max_chunk(tok, r["state"], r["questions"], LIMIT)
                out["chunking"][f"train-worst/ascii/{style}"] = {"people": worst["people"], "max_k": k}
    out["padding"][f"train_initial_ascii_compact_chunk{PAD_K}"] = [_padding(lengths["ascii/compact"], b) for b in (4, 8, 16)]
    Path(out_path).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: out[k] for k in ("a10_validation", "max_synthetic_n", "chunking")}, indent=1))
    print(json.dumps({s: {k: v for k, v in d.items()} for s, d in out["real"].items()}, indent=1))
    print(json.dumps(out["padding"], indent=1))
    print(json.dumps([{k: s[k] for k in ("n", "style", "fixed", "state", "total", "status")}
                      for s in out["synthetic"]]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
