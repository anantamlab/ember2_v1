"""CPU check that the P0B harness emits documents the padding gate accepts.

GPU calls are replaced by fakes; this proves schema wiring only, not parity.
"""
from types import SimpleNamespace

from ember_v1 import p0b
from ember_v1.padding_gate import assess_padding_parity

META = {"model_revision": "fake@0", "runtime_versions": {"torch": "fake"}, "gpu": "fake",
        "head_path": "fake", "kernel_backend": "fake"}


def _uniform(questions):
    return {pid: dict(stay=.1, north=.1, south=.6, east=.1, west=.1) for pid in questions}


def test_padding_task_output_passes_gate(monkeypatch):
    def fake_encode(row, tokenizer, model):
        rec = SimpleNamespace(input_ids=[0] * (100 + 7 * len(row["questions"])), questions=list(row["questions"]))
        return rec, {"status": "OK", "truncated": False, "input_tokens": len(rec.input_ids)}

    def fake_batched(model, tokenizer, records, questions):
        return [_uniform(q) for q in questions]

    def fake_predict(model, tokenizer, row):
        return {"answers": {pid: {"answer": "south"} for pid in row["questions"]}, "truncated": False}

    monkeypatch.setattr(p0b, "encode_one", fake_encode)
    monkeypatch.setattr(p0b, "batched_probabilities", fake_batched)
    monkeypatch.setattr(p0b, "predict_answers", fake_predict)
    result = p0b.task_padding(None, None, [1, 10, 50], "compact", META)
    assert set(result["gate_inputs"]) == {"short_first", "short_last", "short_middle"}
    assert all(v == 1.0 for v in result["adapter_agreement_with_predict"].values())
    for doc in result["gate_inputs"].values():
        verdict = assess_padding_parity(doc)
        assert verdict["status"] == "PASS" and verdict["left_tested"] is False


def test_predict_output_normalization():
    qs = {"P1": {}, "P2": {}}
    direct = {"P1": {"answer": "stay"}, "P2": {"answer": "east"}}
    wrapped = {"answers": direct, "input_tokens": 9, "truncated": False}
    assert p0b._answers(direct, qs) is direct
    assert p0b._answers(wrapped, qs) is direct
