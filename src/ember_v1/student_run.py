"""P1-18 student runs: measured token counts, the published epoch, and the GPU training/eval glue.

Inputs are the write-once SFT directories of `sft_prep` (train.jsonl / val.jsonl / sft_manifest.json /
SHA256SUMS). Every file hash is checked before use.

* `measure`: per-row Clef token counts with the pinned Qwen3.5-2B tokenizer (`tokenizer.json`, sha256 pinned in
  the student-train protocol). Any row whose status is not `OK` at the protocol `max_len` (16,384) fails the
  whole measurement: nothing is truncated or dropped. Output: `token_counts.jsonl` (example_id -> cost = sum of
  its row totals, max row) and `token_summary.json`.
* `build_epoch`: `student_ingest.sample_epoch` with the measured cost; the manifest is the published epoch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from . import student_ingest as si
from . import student_objective as so
from . import token_budget as tb
from .dataset import decision_questions
from .sim.canonical import digest

VERSION = "p1-18-student-run@1.0.0"
TOKENIZER_JSON_SHA256 = "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4"


class RunError(RuntimeError):
    pass


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_sft_dir(d: Path) -> dict[str, Any]:
    """sft_manifest.json file hashes must match the files on disk."""
    man = json.loads((d / "sft_manifest.json").read_text(encoding="utf-8"))
    for name, want in man["files"].items():
        got = "sha256:" + _sha256(d / name)
        if got != want:
            raise RunError(f"{d / name}: sha256 {got} != manifest {want}")
    return man


def iter_examples(d: Path, split: str) -> Iterator[dict[str, Any]]:
    with open(d / f"{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def hf_tokenizer(tokenizer_json: Path) -> Callable[[str], Sequence[int]]:
    """`tokenizer(text, add_special_tokens=False)` of the pinned fast tokenizer, cached (pieces repeat)."""
    got = _sha256(tokenizer_json)
    if got != TOKENIZER_JSON_SHA256:
        raise RunError(f"tokenizer.json sha256 {got} is not the pinned {TOKENIZER_JSON_SHA256}")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tokenizer_json))

    @lru_cache(maxsize=1 << 18)
    def _ids(text: str) -> tuple[int, ...]:
        return tuple(tok.encode(text, add_special_tokens=False).ids)
    return _ids


def example_rows(ex: Mapping[str, Any], k: int, tokenize: Callable[[str], Sequence[int]], max_len: int,
                 question_style: str = "compact", encoding: str = "ascii") -> list[dict[str, Any]]:
    """Clef counts of every row of one example (the rows `student_train.rows_from_epoch` builds)."""
    obs = ex["observation"]
    qs = decision_questions(obs, style=question_style)
    out = []
    for r in so.rows_for(obs, ex["actions"], si.id_chunks(obs, k), None, encoding=encoding):
        cnt = tb.clef_counts(tokenize, r["state"], {i: qs[i] for i in r["ids"]}, max_len)
        out.append({"index": r["index"], "people": len(r["ids"]), "total": cnt["total"], "state": cnt["state"],
                    "status": cnt["status"]})
    return out


def measure(sft_dir: Path, tokenizer_json: Path, out: Path, *, k: int = 100, max_len: int = 16384,
            log: Callable[[str], None] = print) -> dict[str, Any]:
    if out.exists():
        raise RunError(f"{out} exists (write-once)")
    man = check_sft_dir(sft_dir)
    if man["k"] != k:
        raise RunError(f"SFT k {man['k']} != {k}")
    tokenize = hf_tokenizer(tokenizer_json)
    tmp = out.with_name(out.name + ".partial")
    tmp.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {}
    bad: list[str] = []
    t0 = time.monotonic()
    with open(tmp / "token_counts.jsonl", "x", encoding="utf-8", newline="\n") as fh:
        for split in ("train", "val"):
            rows_total, ex_total, tokens, mx, hist, status = 0, 0, 0, 0, Counter(), Counter()
            for ex in iter_examples(sft_dir, split):
                rows = example_rows(ex, k, tokenize, max_len)
                for r in rows:
                    status[r["status"]] += 1
                    if r["status"] != "OK":
                        bad.append(f"{split}:{ex['example_id'][:19]}:{r['index']}:{r['status']}:{r['total']}")
                cost = sum(r["total"] for r in rows)
                rmax = max(r["total"] for r in rows)
                fh.write(json.dumps({"example_id": ex["example_id"], "split": split, "cost": cost, "max_row": rmax,
                                     "rows": [r["total"] for r in rows]}, separators=(",", ":")) + "\n")
                ex_total += 1
                rows_total += len(rows)
                tokens += cost
                mx = max(mx, rmax)
                hist[min(rmax // 1024, 16) * 1024] += 1
                if ex_total % 2000 == 0:
                    log(f"{split} {ex_total} examples {time.monotonic() - t0:.0f} s")
            summary[split] = {"examples": ex_total, "rows": rows_total, "tokens": tokens, "max_row_tokens": mx,
                              "mean_tokens_per_example": round(tokens / max(ex_total, 1), 1),
                              "row_status": dict(status), "max_row_hist_kib": dict(sorted(hist.items()))}
    if bad:
        raise RunError(f"{len(bad)} rows not OK at max_len {max_len} (no truncation allowed): {bad[:5]} "
                       f"(partial output kept at {tmp})")
    body = {"version": VERSION, "sft_manifest_hash": man["manifest_hash"], "dataset_hash": man["dataset_hash"],
            "tokenizer_json_sha256": TOKENIZER_JSON_SHA256, "k": k, "max_len": max_len, "question_style": "compact",
            "encoding": "ascii", "splits": summary, "all_rows_ok": True,
            "token_counts_sha256": "sha256:" + _sha256(tmp / "token_counts.jsonl"),
            "elapsed_s": round(time.monotonic() - t0, 1)}
    (tmp / "token_summary.json").write_text(json.dumps({**body, "summary_hash": digest(body)}, indent=1,
                                                       sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, out)
    return body


def load_costs(token_dir: Path, split: str) -> dict[str, int]:
    summ = json.loads((token_dir / "token_summary.json").read_text(encoding="utf-8"))
    if "sha256:" + _sha256(token_dir / "token_counts.jsonl") != summ["token_counts_sha256"]:
        raise RunError("token_counts.jsonl does not match token_summary.json")
    out = {}
    with open(token_dir / "token_counts.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["split"] == split:
                out[d["example_id"]] = d["cost"]
    return out


def build_epoch(sft_dir: Path, token_dir: Path, *, seed: str, token_budget: int, out: Path) -> dict[str, Any]:
    """The published epoch: weight-proportional draws (with replacement) to the token budget."""
    if out.exists():
        raise RunError(f"{out} exists (write-once)")
    check_sft_dir(sft_dir)
    costs = load_costs(token_dir, "train")
    exs = [{"example_id": e["example_id"], "weight": e["weight"]} for e in iter_examples(sft_dir, "train")]
    ep = si.sample_epoch(exs, seed=seed, token_budget=token_budget, cost=lambda e: costs[e["example_id"]])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ep, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return {k: v for k, v in ep.items() if k != "draws"}


# ----- GPU phases (run inside the Modal container; the adapter factory is injectable for CPU tests) ----------
def _read_json(p: Path) -> Any:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _write_json(p: Path, doc: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, p)


def host_tokenize(tok: Any) -> Callable[[str], Sequence[int]]:
    tk = getattr(tok, "tokenizer", tok)

    @lru_cache(maxsize=1 << 18)
    def _ids(text: str) -> tuple[int, ...]:
        return tuple(tk(text, add_special_tokens=False)["input_ids"])
    return _ids


def val_rows(sft_dir: Path, n_examples: int | None, cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Evaluation rows of the family-isolated validation split (first `n_examples` in file order, or all)."""
    out = []
    for i, ex in enumerate(iter_examples(sft_dir, "val")):
        if n_examples is not None and i >= n_examples:
            break
        qs = decision_questions(ex["observation"], style=cfg["question_style"])
        for r in so.rows_for(ex["observation"], ex["actions"], si.id_chunks(ex["observation"], cfg["chunk_k"]),
                             None, encoding=cfg["encoding"]):
            out.append({"state": r["state"], "questions": {p: qs[p] for p in r["ids"]}, "gold": dict(r["targets"]),
                        "example_id": ex["example_id"], "index": r["index"], "weight": ex["weight"]})
    return out


def imitation_metrics(adapter: Any, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-person accuracy / NLL and whole-example joint-map exact match on validation rows."""
    people = correct = 0
    nll = 0.0
    by_ex: dict[str, bool] = {}
    conf = Counter()
    t0 = time.perf_counter()
    lat = []
    for r in rows:
        t = time.perf_counter()
        sc = adapter.scores(r)
        lat.append((time.perf_counter() - t) * 1000)
        for pid, gold in r["gold"].items():
            pred = min(sc[pid], key=lambda a: (-sc[pid][a], a))
            people += 1
            correct += pred == gold
            nll -= sc[pid][gold]
            conf[f"{gold}->{pred}"] += 1
            by_ex[r["example_id"]] = by_ex.get(r["example_id"], True) and pred == gold
    lat.sort()
    return {"rows": len(rows), "people": people, "accuracy": round(correct / max(people, 1), 6),
            "nll_per_person": round(nll / max(people, 1), 6), "examples": len(by_ex),
            "joint_exact": round(sum(by_ex.values()) / max(len(by_ex), 1), 6),
            "confusion": dict(sorted(conf.items())), "row_latency_ms": {
                "p50": round(lat[len(lat) // 2], 1) if lat else None,
                "p95": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 1) if lat else None},
            "wall_s": round(time.perf_counter() - t0, 1)}


def _train_rows(spec: Mapping[str, Any], cfg: Mapping[str, Any], tokenize: Callable[[str], Sequence[int]]
                ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from . import student_train as st
    sft = Path(spec["sft_dir"])
    man = check_sft_dir(sft)
    epoch = _read_json(Path(spec["epoch"]))
    need = set(epoch["draws"])
    exs = [e for e in iter_examples(sft, "train") if e["example_id"] in need]
    rows = st.rows_from_epoch({"version": si.INGEST_VERSION, "examples": exs}, epoch, tokenize, cfg)
    local = {}
    with open(spec["token_counts"], encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["example_id"] in need:
                local[d["example_id"]] = d["rows"]
    mism = sum(r["tokens"] != local[r["example_id"]][r["index"]] for r in rows)
    return rows, {"sft_manifest_hash": man["manifest_hash"], "epoch_manifest_hash": epoch["manifest_hash"],
                  "epoch_draws": epoch["n_draws"], "epoch_tokens": epoch["tokens_used"], "rows": len(rows),
                  "host_vs_local_token_mismatch_rows": mism, "max_row_tokens": max(r["tokens"] for r in rows)}


def gpu_phase(phase: str, protocol: Mapping[str, Any], spec: Mapping[str, Any], *,
              adapter_factory: Callable[..., Any], persist: Callable[[], None] = lambda: None,
              image_lock_sha256: str | None = None, log: Callable[[str], None] = print) -> dict[str, Any]:
    """`pilot`: bounded real-data training + gradient / masking / save-reload / latency checks.
    `train`: the protocol run (resumable from its own periodic checkpoints).
    `eval`: imitation metrics of a saved checkpoint on the validation split."""
    from . import student_train as st
    st.check_protocol(protocol, require_frozen=True)
    cfg = st.config_from_protocol(protocol)
    ph = cfg["phases"]["pilot" if phase == "pilot" else phase]
    out = Path(spec["out"])
    out.mkdir(parents=True, exist_ok=True)
    prog_path = out / "progress.json"
    prior = _read_json(prog_path) if prog_path.exists() else None
    prior_s = float(prior["elapsed_total_s"]) if prior else 0.0
    t_start = time.monotonic()

    def guards_now() -> Any:           # caps minus everything already spent (earlier containers + this setup)
        spent = prior_s + time.monotonic() - t_start
        return st.Guards(ph["wall_cap_s"] - spent, ph["usd_cap"] - spent / 3600 * cfg["hourly_usd"], cfg["hourly_usd"])
    rep: dict[str, Any] = {"version": VERSION, "phase": phase, "spec": dict(spec), "protocol_hash": cfg["protocol_hash"],
                           "config_hash": digest(dict(cfg)), "prior_elapsed_s": prior_s}
    if phase == "eval":
        t = time.monotonic()
        ad = adapter_factory(cfg, checkpoint=spec["checkpoint"])
        rep["load_s"] = round(time.monotonic() - t, 1)
        rows = val_rows(Path(spec["sft_dir"]), spec.get("val_examples"), cfg)
        rep["val"] = imitation_metrics(ad, rows)
        rep["peak_memory_gb"] = round(ad.memory_bytes() / 2 ** 30, 3)
        rep["elapsed_s"] = round(time.monotonic() - t_start, 1)
        _write_json(out / "eval_report.json", rep)
        persist()
        return rep
    t = time.monotonic()
    ad = adapter_factory(cfg)
    rep["load_s"] = round(time.monotonic() - t, 1)
    if getattr(ad, "metadata", None) is not None:
        st.verify_runtime_pins(ad.metadata, cfg, image_lock_sha256=image_lock_sha256)
        rep["runtime_pins"] = "verified"
    rep["trainable"] = getattr(ad, "trainable", None)
    t = time.monotonic()
    rows, data = _train_rows(spec, cfg, host_tokenize(ad.tok) if hasattr(ad, "tok") else spec["tokenize"])
    rep["data"] = {**data, "build_s": round(time.monotonic() - t, 1)}
    if data["host_vs_local_token_mismatch_rows"]:
        raise RunError(f"{data['host_vs_local_token_mismatch_rows']} rows: host token count != local measurement")
    if hasattr(ad, "item"):            # encoder length = measured length, every target inside its option mask
        chk = []
        for r in rows[:spec.get("encode_check_rows", 8)]:
            it = ad.item(r)
            chk.append({"tokens": r["tokens"], "input_ids": len(it["input_ids"]), "questions": len(r["questions"]),
                        "targets": len(it["targets"]),
                        "one_hot": all(abs(sum(tg) - 1.0) < 1e-6 and max(tg) == 1.0 for tg in it["targets"])})
        rep["encode_check"] = chk
        if not all(c["questions"] == c["targets"] and c["one_hot"] and c["input_ids"] <= cfg["max_len"]
                   for c in chk):
            raise RunError(f"loss-mask / encoding check failed: {chk}")
    resume = None
    if prior and prior.get("last_checkpoint") and phase == "train":
        resume = {"path": prior["last_checkpoint"]["path"], "meter": prior["last_checkpoint"]["meter"],
                  "checkpoints": prior["last_checkpoint"]["checkpoints"]}
        rep["resumed_from"] = prior["last_checkpoint"]["path"]

    def on_ckpt(c: dict[str, Any]) -> None:
        _write_json(prog_path, {"elapsed_total_s": prior_s + time.monotonic() - t_start, "last_checkpoint": c})
        persist()
        log(f"checkpoint step {c['meter']['optimizer_steps']} loss {c['meter']['losses'][-1]:.4f}")

    run_dir = out / "run"
    try:
        tr = st.train(ad, rows, cfg, guards=guards_now(), out_dir=run_dir, mode="gpu_pilot" if phase == "pilot" else "protocol",
                      protocol=protocol, max_rows=len(rows) if phase == "pilot" else None, resume=resume,
                      on_checkpoint=on_ckpt)
    finally:
        persist()
    rep["train"] = tr
    if phase == "pilot":
        vr = val_rows(Path(spec["sft_dir"]), spec.get("probe_val_examples", 16), cfg)
        before = [ad.scores(r) for r in vr]
        rep["probe_trained_model"] = imitation_metrics(ad, vr)
        rep["train_peak_memory_gb"] = round(ad.memory_bytes() / 2 ** 30, 3)
        del ad
        try:
            import gc

            import torch
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass
        fresh = adapter_factory(cfg, checkpoint=str(run_dir / "final"))
        after = [fresh.scores(r) for r in vr]
        diff = max(abs(b[p][a] - c[p][a]) for b, c in zip(before, after) for p in b for a in b[p])
        same_arg = sum(min(b[p], key=lambda a: (-b[p][a], a)) == min(c[p], key=lambda a: (-c[p][a], a))
                       for b, c in zip(before, after) for p in b)
        rep["reload"] = {"probe_rows": len(vr), "max_abs_logprob_diff": diff, "argmax_agree_people": same_arg,
                         "people": sum(len(b) for b in before), "pass": diff < 5e-2 and same_arg == sum(len(b) for b in before)}
        rep["probe_reloaded"] = imitation_metrics(fresh, vr)
        rep["inference_peak_memory_gb"] = round(fresh.memory_bytes() / 2 ** 30, 3)
        m = tr["meter"]
        rep["go"] = st.pilot_go_rule(protocol, tokens_per_epoch_actual=int(spec.get("full_pass_tokens", 0)) or 1,
                                     measured_tokens_per_s=m["tokens_per_s"] or 1e-9,
                                     peak_memory_gb=rep["train_peak_memory_gb"],
                                     loss_finite=all(math.isfinite(x) for x in m["losses"]))
    rep["elapsed_s"] = round(time.monotonic() - t_start, 1)
    _write_json(out / f"{phase}_report.json", rep)
    _write_json(prog_path, {"elapsed_total_s": prior_s + time.monotonic() - t_start, "done": True,
                            "last_checkpoint": None})
    persist()
    return rep


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ember_v1.student_run")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--sft", required=True, type=Path)
    m.add_argument("--tokenizer-json", required=True, type=Path)
    m.add_argument("--out", required=True, type=Path)
    e = sub.add_parser("epoch")
    e.add_argument("--sft", required=True, type=Path)
    e.add_argument("--tokens", required=True, type=Path)
    e.add_argument("--seed", required=True)
    e.add_argument("--token-budget", required=True, type=int)
    e.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "measure":
            r = measure(a.sft, a.tokenizer_json, a.out)
        else:
            r = build_epoch(a.sft, a.tokens, seed=a.seed, token_budget=a.token_budget, out=a.out)
    except RunError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(r, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
