"""P0B GPU feasibility measurements for Unsloth FastDecisionModel (no training).

Runs on any CUDA host; `infra/modal_app.py` runs it on Modal L4/L40S.
Imports torch/unsloth lazily so the CPU package and tests stay dependency-free.

Public API facts used (read from unsloth `main`, 2026-11; pin and re-check):
- `FastDecisionModel.from_pretrained(model_name=, max_seq_length=, load_in_4bit=)`
  returns `(model, tokenizer)`; `for_inference(model)`.
- `build_dataset(rows, tokenizer, model) -> (items, report)`; report has
  `total`/`skipped`. Over-length rows: the STATE is cut, questions are kept.
- `predict(model, tokenizer, state, questions_dict)` gives per-question
  `answer` and per-option `probabilities` (temperature-scaled).
- Vendored Clef `collate_records(records, pad_token_id, device)` RIGHT-pads;
  the head uses `attention_mask.sum()` as the length (left padding unsupported).

Unverified internals are isolated in `_records`, `_question_names`,
`batched_probabilities`. Run task `inspect` first; `padding` cross-checks the
batched adapter against `predict()` and reports any disagreement.
"""
from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from .contract import ACTIONS
from .dataset import build_decision_row
from .synthetic import make_synthetic_observation

LEFT_UNSUPPORTED_REASON = (
    "HLD v3 decision D3: the Clef head takes attention_mask.sum() as the sequence "
    "length and uses absolute question/option spans, so only right padding is "
    "representable without a custom head."
)


# ---------------------------------------------------------------- loading
def load_model(model_name: str, max_seq: int, precision: str):
    from unsloth import FastDecisionModel

    if precision not in ("bf16", "4bit"):
        raise ValueError("precision must be bf16 or 4bit")
    model, tokenizer = FastDecisionModel.from_pretrained(
        model_name=model_name, max_seq_length=max_seq, load_in_4bit=precision == "4bit")
    FastDecisionModel.for_inference(model)
    return model, tokenizer


def _version(pkg: str) -> str | None:
    import importlib.metadata as md

    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def runtime_metadata(model: Any, model_name: str, precision: str, max_seq: int) -> dict[str, Any]:
    import importlib.util

    import torch

    config = getattr(model, "config", None) or getattr(getattr(model, "encoder", None), "config", None)
    kernels = [k for k in ("fla", "causal_conv1d") if importlib.util.find_spec(k)]
    return {
        "model_revision": f"{model_name}@{getattr(config, '_commit_hash', None) or 'unknown-commit'}",
        "runtime_versions": {p: _version(p) for p in (
            "unsloth", "unsloth_zoo", "transformers", "torch", "triton", "flash-linear-attention",
            "causal-conv1d", "bitsandbytes")},
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "head_path": f"{type(model).__module__}.{type(model).__name__} + unsloth._vendor.clef.collate_records",
        # Import availability is NOT proof the fast kernel ran; see `inspect` profiler output.
        "kernel_backend": "+".join(kernels) if kernels else "torch_fallback(no fla/causal_conv1d)",
        "precision": precision,
        "max_seq": max_seq,
    }


# ---------------------------------------------------------------- encoding
def synthetic_row(n: int, question_style: str) -> dict[str, Any]:
    obs = make_synthetic_observation(n)
    return build_decision_row(obs, {f"P{i}": "stay" for i in range(n)}, group_id=f"synthetic-{n}",
                              source="synthetic-contract-only", question_style=question_style)


def _tok(tokenizer: Any) -> Any:
    return getattr(tokenizer, "tokenizer", tokenizer)


def _pad_id(tokenizer: Any) -> int:
    tok = _tok(tokenizer)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if pad is None:
        raise RuntimeError("tokenizer has no pad or eos token id")
    return pad


def _records(items: Sequence[Any]) -> list[Any]:
    for item in items:
        if not (hasattr(item, "input_ids") and hasattr(item, "questions")):
            raise TypeError(f"build_dataset item {type(item).__name__} is not an EncodedRecord; run task=inspect")
    return list(items)


def encode_one(row: Mapping[str, Any], tokenizer: Any, model: Any) -> tuple[Any, dict[str, Any]]:
    """Encode ONE row so the report and truncation signal belong to that row."""
    from unsloth import FastDecisionModel

    items, report = FastDecisionModel.build_dataset([dict(row)], tokenizer, model)
    report = dict(report)
    if report.get("skipped") or not items:
        return None, {"status": "FAIL_ENCODE_SKIPPED", "report": report}
    record = _records(items)[0]
    state_tokens = len(_tok(tokenizer)(row["state"], add_special_tokens=False)["input_ids"])
    input_tokens = len(record.input_ids)
    # Lower-bound check: the full record must hold at least the whole state.
    truncated = bool(getattr(record, "truncated", False) or report.get("truncated")
                     or input_tokens < state_tokens)
    return record, {"status": "FAIL_TRUNCATED" if truncated else "OK", "truncated": truncated,
                    "input_tokens": input_tokens, "state_tokens": state_tokens, "report": report}


def _question_names(record: Any, questions: Mapping[str, Any]) -> list[str]:
    names = [getattr(q, "name", None) for q in record.questions]
    if len(names) != len(questions):
        raise RuntimeError("encoded record question count differs from the row")
    if all(isinstance(n, str) for n in names):
        if set(names) != set(questions):
            raise RuntimeError("encoded question names differ from person IDs")
        return names
    return list(questions)  # positional: dict insertion order; verified by predict() cross-check


# ---------------------------------------------------------------- inference
def _answers(result: Mapping[str, Any], questions: Mapping[str, Any]) -> Mapping[str, Any]:
    """Normalize predict() output: {pid: {...}} or {"answers": {pid: {...}}, ...}."""
    if set(result) != set(questions) and isinstance(result.get("answers"), Mapping):
        return result["answers"]
    return result


def predict_answers(model: Any, tokenizer: Any, row: Mapping[str, Any]) -> dict[str, Any]:
    from unsloth import FastDecisionModel

    raw = FastDecisionModel.predict(model, tokenizer, row["state"], row["questions"])
    answers = _answers(raw, row["questions"])
    if set(answers) != set(row["questions"]):
        raise RuntimeError("predict() ID set differs from the active person IDs")
    return {"answers": dict(answers), "truncated": bool(raw.get("truncated", False)) if "answers" in raw else None}


def batched_probabilities(model: Any, tokenizer: Any, records: Sequence[Any],
                          questions: Sequence[Mapping[str, Any]]) -> list[dict[str, dict[str, float]]]:
    """One right-padded GPU forward for many environments (softmax at T=1)."""
    import torch
    from unsloth._vendor.clef.joint_schema_model import collate_records

    device = next(model.parameters()).device
    batch = collate_records(list(records), _pad_id(tokenizer), device)
    with torch.inference_mode():
        try:
            out = model(batch)
        except TypeError as exc:
            raise RuntimeError("Clef batch forward signature differs; run task=inspect") from exc
    if len(out) != len(records):
        raise RuntimeError("batched forward returned a different number of records")
    results = []
    for record, logits_list, qs in zip(records, out, questions):
        names = _question_names(record, qs)
        if len(logits_list) != len(names):
            raise RuntimeError("batched forward returned a different number of questions")
        per = {}
        for q, name, logits in zip(record.questions, names, logits_list):
            options = list(getattr(q, "option_ids", None) or qs[name]["criteria"])
            probs = torch.softmax(logits.float()[: len(options)], -1).tolist()
            per[name] = dict(zip(options, probs))
            if set(per[name]) != set(ACTIONS):
                raise RuntimeError(f"unexpected option set for {name}: {sorted(per[name])}")
        results.append(per)
    return results


def _sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _pct(sorted_values: Sequence[float], q: float) -> float:
    return sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))]


def _timed(fn, repeats: int, warmup: int = 2) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        times.append(time.perf_counter() - t0)
    times.sort()
    return {"p50_s": _pct(times, .5), "p95_s": _pct(times, .95), "p99_s": _pct(times, .99), "n": repeats}


# ---------------------------------------------------------------- tasks
def task_inspect(model: Any, tokenizer: Any, question_style: str = "compact") -> dict[str, Any]:
    """Characterize the unverified internals the batched adapter relies on."""
    import inspect

    import torch

    def attrs(obj: Any) -> list[str]:
        return sorted(a for a in dir(obj) if not a.startswith("_"))[:60]

    row = synthetic_row(3, question_style)
    from unsloth import FastDecisionModel

    items, report = FastDecisionModel.build_dataset([row], tokenizer, model)
    item = items[0] if items else None
    raw = FastDecisionModel.predict(model, tokenizer, row["state"], row["questions"])
    info: dict[str, Any] = {
        "model_type": f"{type(model).__module__}.{type(model).__name__}",
        "has_head": hasattr(model, "head"), "has_encoder": hasattr(model, "encoder"),
        "forward_signature": str(inspect.signature(type(model).forward)) if hasattr(type(model), "forward") else None,
        "report": {k: str(v)[:200] for k, v in dict(report).items()},
        "item_type": type(item).__name__ if item is not None else None,
        "item_attrs": attrs(item) if item is not None else None,
        "question_attrs": attrs(item.questions[0]) if item is not None and getattr(item, "questions", None) else None,
        "predict_top_keys": sorted(map(str, raw))[:10],
    }
    try:
        rec = _records(items)
        probs = batched_probabilities(model, tokenizer, rec, [row["questions"]])
        info["batched_adapter"] = "OK"
        info["batched_vs_predict_argmax_equal"] = all(
            max(p, key=p.get) == _answers(raw, row["questions"])[pid]["answer"] for pid, p in probs[0].items())
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            batched_probabilities(model, tokenizer, rec, [row["questions"]])
            _sync()
        events = prof.key_averages()
        key = "device_time_total" if hasattr(events[0], "device_time_total") else "cuda_time_total"
        info["top_cuda_kernels"] = [e.key[:90] for e in sorted(events, key=lambda e: -getattr(e, key, 0))[:15]]
    except Exception as exc:  # report, do not hide: inspect is a characterization task
        info["batched_adapter"] = f"FAILED: {type(exc).__name__}: {exc}"[:500]
    info["cuda_mem_reserved_gb"] = round(torch.cuda.memory_reserved() / 2**30, 3) if torch.cuda.is_available() else None
    return info


def task_smoke(model: Any, tokenizer: Any, ns: Sequence[int], repeats: int, question_style: str) -> dict[str, Any]:
    """Native single-state path: encode, truncation, ID binding, predict() latency."""
    if repeats < 2:
        raise ValueError("at least two repetitions")
    results = []
    for n in ns:
        row = synthetic_row(n, question_style)
        record, entry = encode_one(row, tokenizer, model)
        entry = {"n": n, **entry}
        if record is not None and not entry["truncated"]:
            first = predict_answers(model, tokenizer, row)
            if first["truncated"]:
                entry["status"] = "FAIL_TRUNCATED_PREDICT"
            else:
                entry["latency"] = _timed(lambda: predict_answers(model, tokenizer, row), repeats)
                entry["status"] = "PASS"
        results.append(entry)
    return {"task": "smoke", "question_style": question_style, "results": results,
            "status": "PASS" if all(r["status"] == "PASS" for r in results) else "FAIL",
            "note": "Synthetic shape + latency only. p99 needs >=100 repeats to mean anything."}


def task_padding(model: Any, tokenizer: Any, ns: Sequence[int], question_style: str,
                 metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Singleton vs right-padded mixed-length batches -> padding_gate inputs."""
    if len(set(ns)) < 2:
        raise ValueError("need at least two crowd sizes of different length")
    rows, records, enc = [], [], []
    for n in sorted(set(ns)):
        row = synthetic_row(n, question_style)
        record, entry = encode_one(row, tokenizer, model)
        if record is None:
            raise RuntimeError(f"encoding failed for n={n}: {entry}")
        rows.append(row)
        records.append(record)
        enc.append(entry)
    qs = [r["questions"] for r in rows]
    single = [batched_probabilities(model, tokenizer, [rec], [q])[0] for rec, q in zip(records, qs)]
    # Adapter check: batched T=1 argmax must equal predict() answers (temperature keeps order).
    agreement = {}
    for row, probs in zip(rows, single):
        answers = predict_answers(model, tokenizer, row)["answers"]
        agreement[row["group_id"]] = sum(
            min(p, key=lambda a: (-p[a], a)) == answers[pid]["answer"] for pid, p in probs.items()) / len(probs)
    k = len(rows)
    orders = {"short_first": list(range(k)), "short_last": list(range(k))[::-1],
              "short_middle": [k - 1] + list(range(k - 1)) if k > 2 else [1, 0]}
    docs = {}
    for name, order in orders.items():
        batched = batched_probabilities(model, tokenizer, [records[i] for i in order], [qs[i] for i in order])
        by_env = {order[j]: batched[j] for j in range(k)}
        cases = []
        for i in range(k):
            env = f"n{len(qs[i])}"
            for variant, probs in (("single", single[i]), ("right", by_env[i])):
                cases.append({"env_id": env, "variant": variant, "input_tokens": enc[i]["input_tokens"],
                              "truncated": enc[i]["truncated"], "probabilities": probs})
        docs[name] = {"metadata": {**metadata, "order": [f"n{len(qs[i])}" for i in order],
                                   "left_unsupported_reason": LEFT_UNSUPPORTED_REASON,
                                   "packed_varlen_verified": False},
                      "cases": cases}
    return {"task": "padding", "question_style": question_style, "encodings": enc,
            "adapter_agreement_with_predict": agreement, "gate_inputs": docs}


def task_throughput(model: Any, tokenizer: Any, batch_sizes: Sequence[int], ns: Sequence[int],
                    repeats: int, question_style: str) -> dict[str, Any]:
    """Real cross-environment GPU batching: one forward per batch of states."""
    import torch

    pool = []
    for n in ns:
        row = synthetic_row(n, question_style)
        record, entry = encode_one(row, tokenizer, model)
        if record is None or entry["truncated"]:
            raise RuntimeError(f"encoding failed or truncated for n={n}: {entry}")
        pool.append((record, row["questions"], entry["input_tokens"]))
    results = []
    for b in batch_sizes:
        chosen = [pool[i % len(pool)] for i in range(b)]
        recs, qs, lens = zip(*chosen)
        torch.cuda.reset_peak_memory_stats()
        try:
            lat = _timed(lambda: batched_probabilities(model, tokenizer, recs, qs), repeats)
        except torch.cuda.OutOfMemoryError:
            results.append({"batch": b, "status": "OOM"})
            torch.cuda.empty_cache()
            break
        results.append({
            "batch": b, "status": "OK", "latency": lat,
            "env_steps_per_s": round(b / lat["p50_s"], 2),
            "real_tokens": sum(lens), "padded_token_ratio": round(1 - sum(lens) / (b * max(lens)), 4),
            "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 3),
        })
    return {"task": "throughput", "question_style": question_style, "crowd_sizes": list(ns), "results": results}
