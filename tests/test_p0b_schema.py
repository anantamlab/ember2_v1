"""CPU check that the P0B harness emits documents the padding gate accepts.

GPU calls are replaced by fakes; this proves schema wiring only, not parity.
"""
import pytest

from ember_v1 import p0b
from ember_v1.padding_gate import assess_padding_parity
from p0b_fakes import padding_evidence


def test_padding_task_output_passes_gate(monkeypatch):
    result = padding_evidence(monkeypatch)
    assert set(result["gate_inputs"]) == {"short_first", "short_last", "short_middle"}
    assert all(v == 1.0 for v in result["adapter_agreement_with_predict"].values())
    for doc in result["gate_inputs"].values():
        assert doc["metadata"]["kernel_backend"]["evidence"] == "torch_profiler"
        verdict = assess_padding_parity(doc)
        assert verdict["status"] == "PASS" and verdict["left_tested"] is False


def test_padding_task_refuses_unverified_encoding(monkeypatch):
    padding_evidence(monkeypatch)  # installs the fakes
    monkeypatch.setattr(p0b, "encode_one", lambda row, t, m: (None, {"status": "UNVERIFIED", "truncated": None}))
    with pytest.raises(RuntimeError, match="UNVERIFIED"):
        p0b.task_padding(None, None, [1, 10], "compact", {})


def test_predict_output_normalization():
    qs = {"P1": {}, "P2": {}}
    direct = {"P1": {"answer": "stay"}, "P2": {"answer": "east"}}
    wrapped = {"answers": direct, "input_tokens": 9, "truncated": False}
    assert p0b._answers(direct, qs) is direct
    assert p0b._answers(wrapped, qs) is direct
