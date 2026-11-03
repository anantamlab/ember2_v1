"""P0B GPU feasibility measurements for Unsloth FastDecisionModel (no training).

Runs on any CUDA host; `infra/modal_app.py` runs it on Modal L4/L40S.
Imports torch/unsloth lazily so the CPU package and tests stay dependency-free.

Upstream contract, read from source at `unslothai/unsloth@AUDITED_UNSLOTH_COMMIT`
(vendored Clef `AUDITED_CLEF_REVISION`). Details: `docs/P0B_PREFLIGHT.md`.
- `FastDecisionModel.build_dataset(rows, tokenizer, model) -> (items, report)`.
  Clef items are dicts: `item["record"]` is a Clef `EncodedRecord` (`input_ids`,
  `questions[*].question_id/option_ids`), `item["source"]` is the encoded
  `{"state", "questions"}`. `report` has `total`, `skipped` (per question) and
  `truncated` (count of items with `len(input_ids) >= max_len`).
- Clef `encode_record` cuts ONLY the end of the state: `state_ids[:max_length - fixed]`.
- `FastDecisionModel.predict` re-encodes at `max(max_len, CLEF_SERVE_MAX_LEN)` and
  returns only the answers: its own `truncated` flag never reaches the caller.
- `ClefDecisionModel.forward(input_ids=, attention_mask=, records=)` returns
  `(flat_padded_logits, None)`: one row per question, in record order. The vendored
  `collate_records` RIGHT-pads; the head uses `attention_mask.sum()` as the length.

Truncation proof (`encode_one`): re-encode the same source with the hash-pinned
vendored `encode_record` and no length bound. The row is preserved only if the
bounded record is identical and the full length is below
`min(max_len, CLEF_SERVE_MAX_LEN)`. Anything not proven is UNVERIFIED and never
passes a gate.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .contract import ACTIONS
from .dataset import build_decision_row
from .padding_gate import assess_padding_parity, validate_provenance
from .synthetic import make_synthetic_observation

AUDITED_UNSLOTH_COMMIT = "fa2a5e714955ecc5ffcae555432036deeb52c37c"
AUDITED_CLEF_REVISION = "6a3ac5fc030a3f33f7fa860a1271886b64a6606d"
# sha256 of unsloth/_vendor/clef/joint_schema_model.py; equals the entry in its clef_manifest.json.
AUDITED_CLEF_SHA256 = "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3"
AUDITED_DECISION_SHA256 = "784a0a36f50f82c1608b6ef53392249a585e8d16859c2ec3cfe7f4d64fbf56b0"
AUDITED_SERVE_MAX_LEN = 16384  # unsloth/models/_decision_common.py CLEF_SERVE_MAX_LEN

PADDING_ORDERS = ("short_first", "short_last", "short_middle")
# Stack identity that batched-forward evidence must share with the serving runtime.
EVIDENCE_MATCH_KEYS = ("model_revision", "tokenizer_revision", "runtime_versions", "gpu", "precision",
                       "max_seq", "token_limit", "head_path", "upstream_contract")
P95_MIN_SAMPLES = 20
P99_MIN_SAMPLES = 100
# predict() rounds each probability to 4 decimals: worst-case sum error is 5 x 5e-5.
PROB_SUM_TOL = 1e-3
_COMMIT = re.compile(r"^[0-9a-f]{40}$")

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


def _sha256_file(path: Any) -> str | None:
    if not path:
        return None
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return None


def model_revision(model_name: str, commit: Any) -> str | None:
    """`name@<40-hex commit>` or None. Never substitute a guessed or placeholder commit."""
    return f"{model_name}@{commit}" if isinstance(commit, str) and _COMMIT.match(commit) else None


def _upstream_contract() -> dict[str, Any]:
    import importlib.util

    def origin(name: str) -> str | None:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            return None
        return getattr(spec, "origin", None)

    clef = _sha256_file(origin("unsloth._vendor.clef.joint_schema_model"))
    decision = _sha256_file(origin("unsloth.models.decision"))
    return {"audited_unsloth_commit": AUDITED_UNSLOTH_COMMIT, "audited_clef_revision": AUDITED_CLEF_REVISION,
            "clef_sha256": clef, "clef_matches_audited": clef == AUDITED_CLEF_SHA256,
            "decision_sha256": decision, "decision_matches_audited": decision == AUDITED_DECISION_SHA256}


def runtime_metadata(model: Any, tokenizer: Any, model_name: str, precision: str, max_seq: int, *,
                     evidence_source: str, code_commit: str | None = None) -> dict[str, Any]:
    """Provenance for result files. Unknown values stay None; the gate rejects them."""
    import importlib.util

    import torch

    config = getattr(model, "config", None) or getattr(getattr(model, "encoder", None), "config", None)
    init_kwargs = getattr(_tok(tokenizer), "init_kwargs", None) or {}
    gpu, capability = None, None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu, capability = props.name, f"{props.major}.{props.minor}"
    head = getattr(model, "head", None)
    return {
        "schema": "ember_v1.p0b.runtime/2",
        "model_name": model_name,
        "model_revision": model_revision(model_name, getattr(config, "_commit_hash", None)),
        "tokenizer_revision": model_revision(model_name, init_kwargs.get("_commit_hash")),
        "runtime_versions": {p: _version(p) for p in (
            "unsloth", "unsloth_zoo", "transformers", "tokenizers", "torch", "triton", "xformers", "peft",
            "flash-linear-attention", "causal-conv1d", "bitsandbytes")},
        "cuda": torch.version.cuda,
        "gpu": gpu,
        "gpu_capability": capability,
        "head_path": f"{type(model).__module__}.{type(model).__name__}"
                     f"+{type(head).__module__}.{type(head).__name__}+clef.collate_records(right_pad)",
        # Detected = importable. Executed = seen by torch.profiler (filled by the padding/inspect tasks).
        "kernel_backend": {"detected_packages": [k for k in ("fla", "causal_conv1d") if importlib.util.find_spec(k)],
                           "executed": None, "evidence": "import_check_only"},
        "precision": precision,
        "max_seq": max_seq,
        "token_limit": _token_limit(model),
        "upstream_contract": _upstream_contract(),
        "code_commit": code_commit,
        "evidence_source": evidence_source,
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


def validate_questions(questions: Any) -> dict[str, Any]:
    """Exactly one five-action choice question per person ID. No repair."""
    if not isinstance(questions, Mapping) or not questions:
        raise ValueError("questions must be a nonempty mapping of person ID to question")
    for pid, question in questions.items():
        if not isinstance(pid, str) or not pid:
            raise ValueError(f"invalid person ID {pid!r}")
        if not isinstance(question, Mapping) or question.get("type") != "choice":
            raise ValueError(f"question for {pid} is not a choice question")
        criteria = question.get("criteria")
        if not isinstance(criteria, Mapping) or set(criteria) != set(ACTIONS):
            raise ValueError(f"question for {pid} must offer exactly the options {list(ACTIONS)}")
    return dict(questions)


def inference_row(request: Mapping[str, Any]) -> dict[str, Any]:
    """Row for `build_dataset`, which skips questions without gold. The placeholder
    label is never an action: Clef `encode_record` reads only state and questions."""
    if not isinstance(request, Mapping):
        raise ValueError("request must be a mapping")
    state = request.get("state")
    if not isinstance(state, str) or not state:
        raise ValueError("state must be a nonempty string")
    questions = validate_questions(request.get("questions"))
    return {"state": state, "questions": questions, "gold": {pid: "stay" for pid in questions}}


def _row_digest(row: Mapping[str, Any]) -> str:
    # Question order changes the encoding, so keep it (sort_keys sorts only inner dicts).
    payload = json.dumps([row["state"], list(row["questions"].items())], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _clef_encoder() -> tuple[Callable[..., Any] | None, str]:
    """Vendored Clef `encode_record`, only if its source is byte-identical to the audited file."""
    try:
        from unsloth._vendor.clef import joint_schema_model as jsm
    except Exception as exc:  # noqa: BLE001 - any import failure means no evidence
        return None, f"UNVERIFIED: vendored Clef import failed ({type(exc).__name__})"
    digest = _sha256_file(getattr(jsm, "__file__", None))
    if digest != AUDITED_CLEF_SHA256:
        return None, f"UNVERIFIED: vendored Clef sha256 {digest} is not the audited {AUDITED_CLEF_SHA256}"
    return jsm.encode_record, "MATCHES_AUDITED"


def _token_limit(model: Any) -> int | None:
    """Lowest context any audited encode path uses: build_dataset (max_len) or predict (serve)."""
    config = getattr(model, "decision_config", None)
    max_len = config.get("max_len") if isinstance(config, Mapping) else None
    if type(max_len) is not int or max_len <= 0:
        return None
    return min(max_len, AUDITED_SERVE_MAX_LEN)


def _record(item: Any) -> Any | None:
    record = item.get("record") if isinstance(item, Mapping) else None
    if record is None or not hasattr(record, "input_ids") or not hasattr(record, "questions"):
        return None
    return record


def encoding_evidence(record: Any, reference: Any, questions: Mapping[str, Any], *, limit: int,
                      report: Mapping[str, Any]) -> dict[str, Any]:
    """Positive evidence that the entire state and every question survived encoding.

    `reference` is the same source encoded by the same audited `encode_record` with
    no length bound. Equal records prove no cut at `max_len`; a full length below
    `limit` proves `predict()` (which re-encodes) does not cut either.
    """
    base = {"input_tokens": len(record.input_ids), "full_tokens": len(reference.input_ids),
            "token_limit": limit, "report": dict(report)}
    ids = [getattr(q, "question_id", None) for q in record.questions]
    if len(ids) != len(set(ids)) or set(ids) != set(questions):
        return {**base, "status": "FAIL_QUESTION_MISMATCH", "truncated": None,
                "reason": f"encoded IDs {sorted(map(str, ids))} differ from the request"}
    bad = [q.question_id for q in record.questions if sorted(getattr(q, "option_ids", ())) != sorted(ACTIONS)]
    if bad:
        return {**base, "status": "FAIL_OPTION_MISMATCH", "truncated": None, "reason": f"bad options for {bad}"}
    if tuple(record.input_ids) != tuple(reference.input_ids) or tuple(record.questions) != tuple(reference.questions):
        return {**base, "status": "FAIL_TRUNCATED", "truncated": True,
                "reason": "encoded record differs from the unbounded encoding: the state was cut"}
    if len(reference.input_ids) >= limit:
        # Same rule as upstream _clef_truncated: one spare token tells a cut from an exact fit.
        return {**base, "status": "FAIL_NO_SPARE_TOKEN", "truncated": None,
                "reason": f"{len(reference.input_ids)} tokens leave no spare token under {limit}; raise max_seq"}
    if report.get("truncated"):
        return {**base, "status": "UNVERIFIED_REPORT_CONFLICT", "truncated": None,
                "reason": "build_dataset reports truncation that the reference check does not show"}
    return {**base, "status": "OK", "truncated": False,
            "evidence": f"unbounded encode_record reference, clef sha256 {AUDITED_CLEF_SHA256[:12]}"}


def encode_one(row: Mapping[str, Any], tokenizer: Any, model: Any) -> tuple[Any, dict[str, Any]]:
    """Encode ONE row. Returns (record, evidence); record is None unless status == "OK"."""
    from unsloth import FastDecisionModel

    questions = validate_questions(row.get("questions"))
    digest = _row_digest(row)
    encode_record, contract = _clef_encoder()
    items, report = FastDecisionModel.build_dataset([dict(row)], tokenizer, model)
    report = dict(report)
    if report.get("skipped") or len(items) != 1:
        return None, {"status": "FAIL_ENCODE_SKIPPED", "truncated": None, "report": report, "row_sha256": digest}
    item, record, limit = items[0], _record(items[0]), _token_limit(model)
    source = item.get("source") if isinstance(item, Mapping) else None
    reasons = [r for r, bad in (
        (contract, encode_record is None),
        ("build_dataset item is not {'record': EncodedRecord, 'source': ...}",
         record is None or not isinstance(source, Mapping)),
        ("model.decision_config['max_len'] is unreadable", limit is None),
        ("build_dataset changed the state text",
         isinstance(source, Mapping) and source.get("state") != row["state"]),
    ) if bad]
    if not reasons:
        try:
            reference = encode_record(_tok(tokenizer), dict(source), max_length=sys.maxsize)
        except Exception as exc:  # noqa: BLE001 - no reference means no proof
            reasons.append(f"reference encoding failed: {type(exc).__name__}: {exc}"[:300])
    if reasons:
        return None, {"status": "UNVERIFIED", "truncated": None, "reason": "; ".join(reasons),
                      "report": report, "row_sha256": digest}
    evidence = {**encoding_evidence(record, reference, questions, limit=limit, report=report), "row_sha256": digest}
    return (record if evidence["status"] == "OK" else None), evidence


def _question_names(record: Any, questions: Mapping[str, Any]) -> list[str]:
    names = [getattr(q, "question_id", None) for q in record.questions]
    if not all(isinstance(n, str) for n in names) or len(set(names)) != len(names) or set(names) != set(questions):
        raise RuntimeError("encoded question IDs differ from the active person IDs")
    return names


# ---------------------------------------------------------------- inference
def _answers(result: Any, questions: Mapping[str, Any]) -> Mapping[str, Any]:
    """Normalize predict() output: {pid: {...}} or {"answers": {pid: {...}}, ...}."""
    if not isinstance(result, Mapping):
        raise ValueError("predict() did not return a mapping")
    if "answers" not in questions and isinstance(result.get("answers"), Mapping):
        return result["answers"]
    return result


def validate_answers(answers: Any, questions: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Exact ID set, five actions, finite normalized probabilities, answer = top action.

    Raises ValueError on any defect; never repairs or substitutes an action.
    """
    if not isinstance(answers, Mapping):
        raise ValueError("prediction is not a mapping")
    missing, extra = sorted(set(questions) - set(answers)), sorted(map(str, set(answers) - set(questions)))
    if missing or extra:
        raise ValueError(f"prediction ID set differs from the active IDs: missing={missing} extra={extra}")
    out = {}
    for pid in questions:
        result = answers[pid]
        if not isinstance(result, Mapping):
            raise ValueError(f"prediction for {pid} is not a mapping")
        answer, probs = result.get("answer"), result.get("probabilities")
        if not isinstance(answer, str) or answer not in ACTIONS:
            raise ValueError(f"prediction for {pid} has invalid answer {answer!r}")
        if not isinstance(probs, Mapping) or set(probs) != set(ACTIONS):
            raise ValueError(f"prediction for {pid} must score exactly {list(ACTIONS)}")
        values = {}
        for action in ACTIONS:
            p = probs[action]
            if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1:
                raise ValueError(f"invalid probability {p!r} for {pid}/{action}")
            values[action] = float(p)
        if abs(sum(values.values()) - 1.0) > PROB_SUM_TOL:
            raise ValueError(f"probabilities for {pid} are not normalized (sum={sum(values.values())})")
        if values[answer] < max(values.values()):
            raise ValueError(f"answer {answer} for {pid} is not the top-probability action")
        out[pid] = {"answer": answer, "probabilities": values}
    return out


def predict_answers(model: Any, tokenizer: Any, request: Mapping[str, Any], *,
                    evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Native `predict()` behind positive preservation evidence.

    Returns `answers` only when `status == "OK"`; otherwise `answers` is None and
    `predict()` is not called. Malformed predictions raise.
    """
    from unsloth import FastDecisionModel

    row = inference_row(request)
    if evidence is None:
        _, evidence = encode_one(row, tokenizer, model)
    elif evidence.get("row_sha256") != _row_digest(row):
        raise ValueError("encoding evidence belongs to a different state/questions")
    evidence = dict(evidence)
    if evidence.get("status") != "OK" or evidence.get("truncated") is not False:
        return {"status": evidence.get("status") or "UNVERIFIED", "truncated": evidence.get("truncated"),
                "answers": None, "encoding": evidence}
    raw = FastDecisionModel.predict(model, tokenizer, row["state"], row["questions"])
    answers = _answers(raw, row["questions"])
    if answers is not raw:  # a wrapped result carries its own evidence; it must agree
        if "truncated" in raw and raw["truncated"] is not False:
            return {"status": "FAIL_TRUNCATED_PREDICT", "truncated": raw["truncated"], "answers": None,
                    "encoding": evidence}
        if "input_tokens" in raw and raw["input_tokens"] != evidence.get("full_tokens"):
            raise RuntimeError("predict() input_tokens differs from the verified full encoding")
    return {"status": "OK", "truncated": False, "answers": validate_answers(answers, row["questions"]),
            "encoding": evidence}


def _amp_dtype(model: Any, device: Any) -> Any:
    # Same autocast dtype as predict(); private upstream helper, valid only on the audited stack.
    from unsloth.models.decision import _clef_amp_dtype

    return _clef_amp_dtype(model, device)


def batched_probabilities(model: Any, tokenizer: Any, records: Sequence[Any],
                          questions: Sequence[Mapping[str, Any]]) -> list[dict[str, dict[str, float]]]:
    """One right-padded GPU forward for many environments (softmax at T=1, uncalibrated)."""
    import torch
    from unsloth._vendor.clef.joint_schema_model import collate_records

    if len(records) != len(questions) or not records:
        raise ValueError("records and questions must be nonempty and aligned")
    names = [_question_names(record, qs) for record, qs in zip(records, questions)]
    for record in records:
        for q in record.questions:
            if sorted(q.option_ids) != sorted(ACTIONS):
                raise RuntimeError(f"unexpected option set for {q.question_id}: {sorted(q.option_ids)}")
    device = next(model.parameters()).device
    batch = collate_records(list(records), _pad_id(tokenizer), device)
    if batch.get("media"):
        raise RuntimeError("media inputs are out of scope for the text-only P0B adapter")
    amp = _amp_dtype(model, device)
    with torch.inference_mode(), torch.autocast(device.type, dtype=amp, enabled=amp is not None):
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], records=batch["records"])
    logits = out[0] if isinstance(out, tuple) else out
    total = sum(len(n) for n in names)
    if not torch.is_tensor(logits) or logits.dim() != 2 or logits.shape[0] != total or logits.shape[1] < len(ACTIONS):
        raise RuntimeError(f"forward returned {getattr(logits, 'shape', type(logits))}; expected ({total}, >=5)")
    rows = iter(logits.float().cpu())
    results = []
    for record, record_names in zip(records, names):
        per = {}
        for q, name in zip(record.questions, record_names):
            probs = torch.softmax(next(rows)[: len(q.option_ids)], -1).tolist()
            per[name] = dict(zip(q.option_ids, probs))
        results.append(per)
    return results


# ---------------------------------------------------------------- serving
def _argmax(probs: Mapping[str, float]) -> str:
    # Upstream picks the first maximum over sorted option IDs: alphabetical tie-break.
    return min(probs, key=lambda a: (-probs[a], a))


def verify_batched_forward_evidence(evidence: Any, runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Allow the right-padded fast path only on recomputed, complete, matching evidence.

    `evidence` is the full `padding` task result. Stored `status`/`verdicts` are
    ignored: every gate input is re-assessed here. Raises ValueError otherwise.
    """
    if not isinstance(evidence, Mapping) or evidence.get("task") != "padding":
        raise ValueError("batched_forward evidence must be the full `padding` task result, not a verdict")
    validate_provenance(dict(runtime), require_kernel_profile=False)
    docs = evidence.get("gate_inputs")
    if not isinstance(docs, Mapping) or set(docs) != set(PADDING_ORDERS):
        raise ValueError(f"padding evidence needs raw gate inputs for exactly {list(PADDING_ORDERS)}")
    agreement = evidence.get("adapter_agreement_with_predict")
    if not isinstance(agreement, Mapping) or not agreement or any(
            type(v) not in (int, float) or v != 1 for v in agreement.values()):
        raise ValueError("batched adapter does not agree 100% with predict()")
    encodings = evidence.get("encodings")
    if not isinstance(encodings, list) or not encodings or any(
            not isinstance(e, Mapping) or e.get("status") != "OK" or e.get("truncated") is not False
            for e in encodings):
        raise ValueError("padding evidence contains unverified or truncated encodings")
    verdicts = {}
    for order in PADDING_ORDERS:
        verdict = assess_padding_parity(docs[order])  # raises on invalid evidence
        if verdict["status"] != "PASS":
            raise ValueError(f"padding parity FAILS for order {order}")
        meta = docs[order]["metadata"]
        diff = [k for k in EVIDENCE_MATCH_KEYS if meta.get(k) != runtime.get(k)]
        if diff:
            raise ValueError(f"padding evidence was measured on a different stack: {diff}")
        verdicts[order] = verdict["status"]
    canonical = json.dumps(evidence, sort_keys=True, default=str).encode("utf-8")
    return {"status": "VERIFIED", "orders": verdicts, "question_style": evidence.get("question_style"),
            "evidence_sha256": hashlib.sha256(canonical).hexdigest()}


def decide_batch(model: Any, tokenizer: Any, requests: Sequence[Mapping[str, Any]], *,
                 metadata: Mapping[str, Any], batched_forward: bool = False,
                 batched_evidence: Mapping[str, Any] | None = None,
                 sync: Callable[[], bool] | None = None) -> list[dict[str, Any]]:
    """Serve one Modal batch. Fails closed: any unverified request raises for the batch."""
    if batched_forward and (not isinstance(batched_evidence, Mapping) or batched_evidence.get("status") != "VERIFIED"):
        raise RuntimeError("batched_forward needs evidence from verify_batched_forward_evidence; refusing")
    sync = sync or _sync
    rows = [inference_row(req) for req in requests]
    synced = sync()
    t0 = time.perf_counter()
    if not batched_forward:
        out = []
        for i, row in enumerate(rows):
            res = predict_answers(model, tokenizer, row)
            if res["status"] != "OK":
                raise RuntimeError(f"request {i}: {res['status']}: fail closed, no fallback action")
            out.append({"answers": res["answers"], "path": "predict", "probability_scale": "served_temperature",
                        "encoding": {k: res["encoding"].get(k) for k in ("status", "input_tokens", "full_tokens")}})
    else:
        records, encodings = [], []
        for i, row in enumerate(rows):
            record, entry = encode_one(row, tokenizer, model)
            if entry["status"] != "OK":
                raise RuntimeError(f"request {i}: {entry['status']}: fail closed, no fallback action")
            records.append(record)
            encodings.append(entry)
        probs = batched_probabilities(model, tokenizer, records, [r["questions"] for r in rows])
        out = []
        for row, per, entry in zip(rows, probs, encodings):
            answers = validate_answers({pid: {"answer": _argmax(p), "probabilities": p} for pid, p in per.items()},
                                       row["questions"])
            out.append({"answers": answers, "path": "batched_right_pad", "probability_scale": "softmax_T1_uncalibrated",
                        "encoding": {k: entry.get(k) for k in ("status", "input_tokens", "full_tokens")}})
    sync()
    exec_s = time.perf_counter() - t0
    timing = {"scope": "server_batch_execution", "batch_exec_s": exec_s, "batch_size": len(rows),
              "amortized_per_request_s": exec_s / max(1, len(rows)), "cuda_synchronized": bool(synced),
              "includes": "preservation check, encoding, inference, output validation",
              "excludes": "Modal batching wait (wait_ms), queueing, network, cold start; measure client-side"}
    for item in out:
        item.update(status="OK", truncated=False, model_revision=metadata.get("model_revision"), timing=timing)
    return out


# ---------------------------------------------------------------- timing
def _sync() -> bool:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return True
    return False


def _pct(sorted_values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile."""
    return sorted_values[max(0, math.ceil(q * len(sorted_values)) - 1)]


def latency_summary(times: Sequence[float], **extra: Any) -> dict[str, Any]:
    """Tail percentiles are withheld (None) when the sample is too small to estimate them."""
    s = sorted(times)
    n = len(s)
    if n < 2:
        raise ValueError("at least two timed repetitions")
    out = {"n": n, "method": "nearest_rank", "min_s": s[0], "p50_s": _pct(s, .5), "max_s": s[-1],
           "mean_s": sum(s) / n,
           "p95_s": _pct(s, .95) if n >= P95_MIN_SAMPLES else None,
           "p99_s": _pct(s, .99) if n >= P99_MIN_SAMPLES else None, **extra}
    if n < P99_MIN_SAMPLES:
        out["tail_note"] = (f"n={n}: p95 needs n>={P95_MIN_SAMPLES}, p99 needs n>={P99_MIN_SAMPLES}; "
                            "withheld values are not tail-latency evidence, max_s is descriptive only")
    return out


def _timed(fn: Callable[[], Any], repeats: int, warmup: int = 2,
           sync: Callable[[], bool] | None = None) -> dict[str, Any]:
    sync = sync or _sync
    if repeats < 2:
        raise ValueError("at least two repetitions")
    for _ in range(warmup):
        fn()
    synced = sync()
    times = []
    for _ in range(repeats):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append(time.perf_counter() - t0)
    return latency_summary(times, warmup=warmup, cuda_synchronized=bool(synced), clock="time.perf_counter")


def _profile_kernels(fn: Callable[[], Any], top: int = 25) -> dict[str, Any]:
    """Device ops/kernels that actually ran, ranked by device time (torch.profiler)."""
    try:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            fn()
            _sync()
        events = prof.key_averages()
        key = "device_time_total" if events and hasattr(events[0], "device_time_total") else "cuda_time_total"
        ran = sorted((e for e in events if getattr(e, key, 0) > 0), key=lambda e: -getattr(e, key, 0))
        names = [e.key[:120] for e in ran[:top]]
        return {"executed": names, "evidence": "torch_profiler" if names else "torch_profiler_no_device_events"}
    except Exception as exc:  # noqa: BLE001 - report, never invent kernel evidence
        return {"executed": None, "evidence": f"profiler_failed: {type(exc).__name__}: {exc}"[:300]}


def _overall(statuses: Sequence[str]) -> str:
    if statuses and all(s == "PASS" for s in statuses):
        return "PASS"
    return "FAIL" if any(s.startswith("FAIL") for s in statuses) else "UNVERIFIED"


# ---------------------------------------------------------------- tasks
def task_inspect(model: Any, tokenizer: Any, question_style: str = "compact") -> dict[str, Any]:
    """Characterize the upstream internals the harness relies on."""
    import inspect

    import torch
    from unsloth import FastDecisionModel

    def attrs(obj: Any) -> list[str]:
        return sorted(a for a in dir(obj) if not a.startswith("_"))[:60]

    row = synthetic_row(3, question_style)
    items, report = FastDecisionModel.build_dataset([row], tokenizer, model)
    item = items[0] if items else None
    record = _record(item)
    info: dict[str, Any] = {
        "model_type": f"{type(model).__module__}.{type(model).__name__}",
        "has_head": hasattr(model, "head"), "has_encoder": hasattr(model, "encoder"),
        "forward_signature": str(inspect.signature(type(model).forward)) if hasattr(type(model), "forward") else None,
        "decision_config": {k: str(v)[:100] for k, v in dict(getattr(model, "decision_config", None) or {}).items()},
        "report": {k: str(v)[:200] for k, v in dict(report).items()},
        "item_type": type(item).__name__ if item is not None else None,
        "item_keys": sorted(map(str, item)) if isinstance(item, Mapping) else None,
        "record_attrs": attrs(record) if record is not None else None,
        "question_attrs": attrs(record.questions[0]) if record is not None and record.questions else None,
        "upstream_contract": _upstream_contract(),
        "clef_encoder": _clef_encoder()[1],
    }
    rec, evidence = encode_one(row, tokenizer, model)
    info["encoding"] = evidence
    try:
        raw = FastDecisionModel.predict(model, tokenizer, row["state"], row["questions"])
        info["predict_top_keys"] = sorted(map(str, raw))[:10]
        checked = predict_answers(model, tokenizer, row, evidence=evidence)
        info["predict_status"] = checked["status"]
        if rec is None or checked["status"] != "OK":
            raise RuntimeError(f"encoding {evidence['status']}, predict {checked['status']}")
        probs = batched_probabilities(model, tokenizer, [rec], [row["questions"]])
        info["batched_adapter"] = "OK"
        info["batched_vs_predict_argmax_equal"] = all(
            _argmax(p) == checked["answers"][pid]["answer"] for pid, p in probs[0].items())
        info["kernels"] = _profile_kernels(lambda: batched_probabilities(model, tokenizer, [rec], [row["questions"]]))
    except Exception as exc:  # report, do not hide: inspect is a characterization task
        info["batched_adapter"] = f"FAILED: {type(exc).__name__}: {exc}"[:500]
    info["cuda_mem_reserved_gb"] = round(torch.cuda.memory_reserved() / 2**30, 3) if torch.cuda.is_available() else None
    return info


def task_smoke(model: Any, tokenizer: Any, ns: Sequence[int], repeats: int, question_style: str) -> dict[str, Any]:
    """Native single-state path: preservation proof, ID binding, predict() latency."""
    if repeats < 2:
        raise ValueError("at least two repetitions")
    results = []
    for n in ns:
        row = synthetic_row(n, question_style)
        t0 = time.perf_counter()
        _, evidence = encode_one(row, tokenizer, model)
        entry = {"n": n, **evidence, "preservation_check_s": time.perf_counter() - t0}
        if evidence["status"] == "OK":
            first = predict_answers(model, tokenizer, row, evidence=evidence)
            entry["status"] = "PASS" if first["status"] == "OK" else first["status"]
            if first["status"] == "OK":
                entry["predict_latency"] = {
                    "scope": "FastDecisionModel.predict + output validation, one state, evidence precomputed",
                    **_timed(lambda: predict_answers(model, tokenizer, row, evidence=evidence), repeats)}
        results.append(entry)
    return {"task": "smoke", "question_style": question_style, "results": results,
            "status": _overall([r["status"] for r in results]),
            "note": "Synthetic shape + single-request latency. PASS needs positive preservation evidence per row."}


def task_padding(model: Any, tokenizer: Any, ns: Sequence[int], question_style: str,
                 metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Singleton vs right-padded mixed-length batches -> padding_gate inputs."""
    if len(set(ns)) < 2:
        raise ValueError("need at least two crowd sizes of different length")
    rows, records, enc = [], [], []
    for n in sorted(set(ns)):
        row = synthetic_row(n, question_style)
        record, entry = encode_one(row, tokenizer, model)
        if entry["status"] != "OK" or record is None:
            raise RuntimeError(f"encoding not verified for n={n}: {entry['status']}: {entry.get('reason')}")
        rows.append(row)
        records.append(record)
        enc.append(entry)
    qs = [r["questions"] for r in rows]
    single = [batched_probabilities(model, tokenizer, [rec], [q])[0] for rec, q in zip(records, qs)]
    # Adapter check: batched T=1 argmax must equal predict() answers (temperature keeps order).
    agreement = {}
    for row, probs, entry in zip(rows, single, enc):
        res = predict_answers(model, tokenizer, row, evidence=entry)
        if res["status"] != "OK":
            raise RuntimeError(f"predict() not verified for {row['group_id']}: {res['status']}")
        agreement[row["group_id"]] = sum(
            _argmax(p) == res["answers"][pid]["answer"] for pid, p in probs.items()) / len(probs)
    kernels = _profile_kernels(lambda: batched_probabilities(model, tokenizer, records, qs))
    measured = {**metadata, "kernel_backend": {**dict(metadata.get("kernel_backend") or {}), **kernels}}
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
        docs[name] = {"metadata": {**measured, "order": [f"n{len(qs[i])}" for i in order],
                                   "left_unsupported_reason": LEFT_UNSUPPORTED_REASON,
                                   "packed_varlen_verified": False},
                      "cases": cases}
    return {"task": "padding", "question_style": question_style, "crowd_sizes": sorted(set(ns)),
            "encodings": enc, "adapter_agreement_with_predict": agreement, "gate_inputs": docs}


def task_throughput(model: Any, tokenizer: Any, batch_sizes: Sequence[int], ns: Sequence[int],
                    repeats: int, question_style: str) -> dict[str, Any]:
    """Real cross-environment GPU batching: one forward per batch of pre-encoded states."""
    import torch

    pool = []
    for n in ns:
        row = synthetic_row(n, question_style)
        record, entry = encode_one(row, tokenizer, model)
        if entry["status"] != "OK" or record is None:
            raise RuntimeError(f"encoding not verified for n={n}: {entry['status']}: {entry.get('reason')}")
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
            "env_steps_per_s_at_p50": round(b / lat["p50_s"], 2),
            "real_tokens": sum(lens), "padded_token_ratio": round(1 - sum(lens) / (b * max(lens)), 4),
            "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 3),
        })
    return {"task": "throughput", "question_style": question_style, "crowd_sizes": list(ns), "results": results,
            "scope": "one right-padded forward per batch on pre-encoded records; excludes tokenization, "
                     "preservation checks, Modal queueing and network. Path B is not parity-cleared by this task."}
