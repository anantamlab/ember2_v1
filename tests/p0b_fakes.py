"""CPU fakes that mirror the audited upstream shapes (unsloth@fa2a5e7, Clef 6a3ac5f).

They reproduce only the contract `ember_v1.p0b` relies on: build_dataset items are
dicts with an `EncodedRecord`, `encode_record` cuts only the end of the state, and
`predict()` returns bare answers. No model, tokenizer or GPU is involved.
"""
from __future__ import annotations

import copy
import sys
import types
from dataclasses import dataclass

from ember_v1.contract import ACTIONS

COMMIT = "0123456789abcdef0123456789abcdef01234567"


def provenance(**overrides):
    meta = {
        "schema": "ember_v1.p0b.runtime/2",
        "model_name": "unsloth/Qwen3.5-2B",
        "model_revision": f"unsloth/Qwen3.5-2B@{COMMIT}",
        "tokenizer_revision": f"unsloth/Qwen3.5-2B@{COMMIT}",
        "runtime_versions": {"torch": "2.8.0", "transformers": "5.2.0", "unsloth": "fake"},
        "gpu": "NVIDIA L4 (fake)",
        "head_path": "fake.ClefDecisionModel+fake.JointSchemaHead",
        "kernel_backend": {"detected_packages": [], "executed": ["fake_kernel"], "evidence": "torch_profiler"},
        "precision": "bf16",
        "max_seq": 16384,
        "token_limit": 16384,
        "upstream_contract": {"clef_matches_audited": True},
        "code_commit": "fake-commit",
        "evidence_source": "test",
    }
    meta.update(overrides)
    return copy.deepcopy(meta)


@dataclass(frozen=True)
class EncodedQuestion:
    question_id: str
    question_span: tuple
    option_ids: tuple


@dataclass(frozen=True)
class EncodedRecord:
    input_ids: tuple
    questions: tuple


class CharTokenizer:
    """One token per character."""
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return types.SimpleNamespace(input_ids=[ord(c) % 997 + 1 for c in text])


def encode_record(tokenizer, record, max_length=16384):
    """Same cut semantics as Clef encode_record: only the end of the state is dropped."""
    schema, questions = [], []
    for i, (qid, q) in enumerate(record["questions"].items()):
        schema += tokenizer(f"FIELD {i + 1} ID: {qid} INSTRUCTION: ").input_ids
        start = len(schema)
        schema += tokenizer(str(q.get("instructions") or qid)).input_ids
        options = tuple(sorted(q["criteria"]))
        for option in options:
            schema += tokenizer(f"OPTION {option}\n").input_ids
        questions.append((qid, start, options))
    prefix = tokenizer("STATE:\n").input_ids
    suffix = tokenizer("\nDECISIONS:").input_ids
    state = tokenizer(record["state"]).input_ids
    fixed = len(prefix) + len(schema) + len(suffix)
    if fixed > max_length:
        raise ValueError("schema does not fit")
    state = state[: max_length - fixed]
    offset = len(prefix) + len(state)
    return EncodedRecord(
        input_ids=tuple(prefix + state + schema + suffix),
        questions=tuple(EncodedQuestion(qid, (start + offset, start + offset + 1), options)
                        for qid, start, options in questions))


def answers_for(questions, top="south"):
    probs = {a: (0.6 if a == top else 0.1) for a in ACTIONS}
    return {pid: {"answer": top, "probabilities": dict(probs), "choice": top} for pid in questions}


class FakeModel:
    """`decision_config` as upstream. Other attributes inject upstream misbehaviour."""

    def __init__(self, max_len=16384, predict_output=None, *, drop_question=False, alter_state=False,
                 report_truncated="computed"):
        self.decision_config = {"max_len": max_len} if max_len is not None else {}
        self.predict_output = predict_output  # callable(questions) -> raw predict() result
        self.predict_calls = 0
        self.drop_question = drop_question  # silently encode one question fewer
        self.alter_state = alter_state  # build_dataset rewrites the state text
        self.report_truncated = report_truncated  # "computed" | "absent" | forced int


class FastDecisionModel:
    @staticmethod
    def build_dataset(rows, tokenizer, model):
        row = rows[0]
        questions = dict(row["questions"])
        if model.drop_question:
            questions = dict(list(questions.items())[1:])
        source = {"state": row["state"] + (" " if model.alter_state else ""), "questions": questions}
        max_len = model.decision_config.get("max_len", 512)  # upstream default
        record = encode_record(tokenizer, source, max_length=max_len)
        report = {"total": len(row["questions"]), "skipped": 0, "reason": None}
        if model.report_truncated == "computed":
            report["truncated"] = int(len(record.input_ids) >= max_len)
        elif model.report_truncated != "absent":
            report["truncated"] = model.report_truncated
        return [{"input_ids": list(record.input_ids), "record": record, "source": source}], report

    @staticmethod
    def predict(model, tokenizer, state, questions):
        model.predict_calls += 1
        if model.predict_output is not None:
            return model.predict_output(questions)
        return answers_for(questions)

    @staticmethod
    def for_inference(model):
        return model


def padding_evidence(monkeypatch, meta=None):
    """Full `padding` task result built from fake GPU measurements (wiring only)."""
    from ember_v1 import p0b

    meta = meta or provenance(kernel_backend={"detected_packages": [], "executed": None,
                                              "evidence": "import_check_only"})

    def probs(questions):
        return {pid: dict(stay=.1, north=.1, south=.6, east=.1, west=.1) for pid in questions}

    def fake_encode(row, tokenizer, model):
        rec = types.SimpleNamespace(input_ids=[0] * (100 + 7 * len(row["questions"])), questions=list(row["questions"]))
        return rec, {"status": "OK", "truncated": False, "input_tokens": len(rec.input_ids),
                     "row_sha256": p0b._row_digest(row)}

    def fake_predict(model, tokenizer, row, evidence=None):
        return {"status": "OK", "truncated": False,
                "answers": {pid: {"answer": "south", "probabilities": p} for pid, p in probs(row["questions"]).items()}}

    monkeypatch.setattr(p0b, "encode_one", fake_encode)
    monkeypatch.setattr(p0b, "batched_probabilities", lambda m, t, records, qs: [probs(q) for q in qs])
    monkeypatch.setattr(p0b, "predict_answers", fake_predict)
    monkeypatch.setattr(p0b, "_profile_kernels", lambda fn: {"executed": ["fake_kernel"], "evidence": "torch_profiler"})
    return p0b.task_padding(None, None, [1, 10, 50], "compact", meta)


def install(monkeypatch, *, contract_ok=True):
    """Install a fake `unsloth` package and the audited-encoder hook."""
    from ember_v1 import p0b

    module = types.ModuleType("unsloth")
    module.FastDecisionModel = FastDecisionModel
    monkeypatch.setitem(sys.modules, "unsloth", module)
    if contract_ok:
        monkeypatch.setattr(p0b, "_clef_encoder", lambda: (encode_record, "MATCHES_AUDITED"))
    else:
        monkeypatch.setattr(p0b, "_clef_encoder", lambda: (None, "UNVERIFIED: vendored Clef sha256 differs"))
