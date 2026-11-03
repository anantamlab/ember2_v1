"""P0B preflight regressions (CPU only, mocks + synthetic observations).

Every test here shows that an unsafe or unverified result can not be read as a
successful gate. The fakes mirror the audited upstream shapes (`p0b_fakes`).
"""
import copy
import sys
import types
from types import SimpleNamespace

import pytest

import p0b_fakes as fakes
from ember_v1 import agent, p0b
from ember_v1.contract import ACTIONS
from ember_v1.synthetic import make_synthetic_observation
from p0b_fakes import CharTokenizer, FakeModel, answers_for, provenance

TOK = CharTokenizer()


@pytest.fixture(autouse=True)
def _no_cuda(monkeypatch):
    monkeypatch.setattr(p0b, "_sync", lambda: False)  # CI has no torch


def _row(n=3, style="compact"):
    return p0b.synthetic_row(n, style)


def _full(row):
    return len(fakes.encode_record(TOK, row, max_length=sys.maxsize).input_ids)


# ------------------------------------------------------------ truncation proof
def test_full_state_is_positively_verified(monkeypatch):
    fakes.install(monkeypatch)
    row = _row()
    record, ev = p0b.encode_one(row, TOK, FakeModel(max_len=_full(row) + 10))
    assert ev["status"] == "OK" and ev["truncated"] is False and record is not None
    assert ev["input_tokens"] == ev["full_tokens"] == _full(row)


@pytest.mark.parametrize("report_truncated", ["computed", "absent", 0])
def test_cut_state_fails_even_when_old_lower_bound_passes(monkeypatch, report_truncated):
    fakes.install(monkeypatch)
    row = _row()
    model = FakeModel(max_len=_full(row) - 5, report_truncated=report_truncated)  # cut 5 state tokens
    record, ev = p0b.encode_one(row, TOK, model)
    state_tokens = len(TOK(row["state"]).input_ids)
    assert ev["input_tokens"] >= state_tokens  # the removed `input_tokens >= state_tokens` check accepted this
    assert ev["status"] == "FAIL_TRUNCATED" and ev["truncated"] is True and record is None


def test_exact_fit_has_no_spare_token_and_fails_closed(monkeypatch):
    fakes.install(monkeypatch)
    row = _row()
    record, ev = p0b.encode_one(row, TOK, FakeModel(max_len=_full(row)))
    assert ev["status"] == "FAIL_NO_SPARE_TOKEN" and ev["truncated"] is None and record is None


def test_predict_serve_limit_bounds_the_proof(monkeypatch):
    fakes.install(monkeypatch)
    row = {**_row(), "state": "x" * (p0b.AUDITED_SERVE_MAX_LEN + 10)}
    record, ev = p0b.encode_one(row, TOK, FakeModel(max_len=_full(row) + 10))
    assert ev["token_limit"] == p0b.AUDITED_SERVE_MAX_LEN
    assert ev["status"] == "FAIL_NO_SPARE_TOKEN" and record is None


@pytest.mark.parametrize("case", ["contract_drift", "no_max_len", "old_item_shape", "state_rewritten"])
def test_unprovable_encoding_is_unverified(monkeypatch, case):
    fakes.install(monkeypatch, contract_ok=case != "contract_drift")
    row = _row()
    model = FakeModel(max_len=None if case == "no_max_len" else 16384, alter_state=case == "state_rewritten")
    if case == "old_item_shape":
        rec = fakes.encode_record(TOK, row)
        monkeypatch.setattr(fakes.FastDecisionModel, "build_dataset",
                            staticmethod(lambda rows, t, m: ([rec], {"total": 3, "skipped": 0})))
    record, ev = p0b.encode_one(row, TOK, model)
    assert ev["status"] == "UNVERIFIED" and ev["truncated"] is None and record is None


def test_skipped_question_fails(monkeypatch):
    fakes.install(monkeypatch)
    monkeypatch.setattr(fakes.FastDecisionModel, "build_dataset",
                        staticmethod(lambda rows, t, m: ([], {"total": 3, "skipped": 1})))
    record, ev = p0b.encode_one(_row(), TOK, FakeModel())
    assert ev["status"] == "FAIL_ENCODE_SKIPPED" and record is None


def test_silently_dropped_question_fails(monkeypatch):
    fakes.install(monkeypatch)
    record, ev = p0b.encode_one(_row(), TOK, FakeModel(drop_question=True))
    assert ev["status"] == "FAIL_QUESTION_MISMATCH" and record is None


def test_report_conflict_is_not_resolved_in_our_favour(monkeypatch):
    fakes.install(monkeypatch)
    record, ev = p0b.encode_one(_row(), TOK, FakeModel(report_truncated=1))
    assert ev["status"] == "UNVERIFIED_REPORT_CONFLICT" and ev["truncated"] is None and record is None


def test_clef_encoder_requires_the_audited_source(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "unsloth", None)
    assert p0b._clef_encoder()[0] is None
    drifted = tmp_path / "joint_schema_model.py"
    drifted.write_text("# not the audited file\n", encoding="utf-8")
    mods = {name: types.ModuleType(name) for name in
            ("unsloth", "unsloth._vendor", "unsloth._vendor.clef", "unsloth._vendor.clef.joint_schema_model")}
    mods["unsloth._vendor.clef.joint_schema_model"].__file__ = str(drifted)
    mods["unsloth._vendor.clef.joint_schema_model"].encode_record = fakes.encode_record
    mods["unsloth._vendor.clef"].joint_schema_model = mods["unsloth._vendor.clef.joint_schema_model"]
    for name, module in mods.items():
        monkeypatch.setitem(sys.modules, name, module)
    encode, status = p0b._clef_encoder()
    assert encode is None and status.startswith("UNVERIFIED") and "sha256" in status


# ------------------------------------------------------------ predictions
def test_unverified_encoding_never_reaches_predict(monkeypatch):
    fakes.install(monkeypatch, contract_ok=False)
    model = FakeModel()
    res = p0b.predict_answers(model, TOK, _row())
    assert res["status"] == "UNVERIFIED" and res["answers"] is None and res["truncated"] is None
    assert model.predict_calls == 0


def test_bare_upstream_answers_pass_only_with_encoding_evidence(monkeypatch):
    fakes.install(monkeypatch)
    res = p0b.predict_answers(FakeModel(), TOK, _row())
    assert res["status"] == "OK" and res["truncated"] is False and res["encoding"]["status"] == "OK"
    assert {r["answer"] for r in res["answers"].values()} == {"south"}


@pytest.mark.parametrize("flag", [True, None, "unknown"])
def test_wrapped_truncation_flag_must_be_exactly_false(monkeypatch, flag):
    fakes.install(monkeypatch)
    model = FakeModel(predict_output=lambda qs: {"answers": answers_for(qs), "truncated": flag})
    res = p0b.predict_answers(model, TOK, _row())
    assert res["status"] == "FAIL_TRUNCATED_PREDICT" and res["answers"] is None


def test_wrapped_input_tokens_must_match_the_proof(monkeypatch):
    fakes.install(monkeypatch)
    model = FakeModel(predict_output=lambda qs: {"answers": answers_for(qs), "truncated": False, "input_tokens": 7})
    with pytest.raises(RuntimeError, match="input_tokens"):
        p0b.predict_answers(model, TOK, _row())


def test_evidence_from_another_row_is_rejected(monkeypatch):
    fakes.install(monkeypatch)
    _, ev = p0b.encode_one(_row(3), TOK, FakeModel())
    with pytest.raises(ValueError, match="different"):
        p0b.predict_answers(FakeModel(), TOK, _row(4), evidence=ev)


def _mutate(answers, kind):
    a = copy.deepcopy(answers)
    p = a["P0"]["probabilities"]
    if kind == "missing_id":
        del a["P1"]
    elif kind == "extra_id":
        a["P99"] = copy.deepcopy(a["P0"])
    elif kind == "missing_action":
        del p["west"]
    elif kind == "extra_action":
        p["jump"] = 0.0
    elif kind in ("nan", "inf", "negative", "above_one", "bool", "string"):
        p["north"] = {"nan": float("nan"), "inf": float("inf"), "negative": -0.1, "above_one": 1.5,
                      "bool": True, "string": "0.1"}[kind]
    elif kind == "not_normalized":
        p["stay"] = 0.5
    elif kind == "answer_not_top":
        a["P0"]["answer"] = "north"
    elif kind == "answer_unknown":
        a["P0"]["answer"] = "jump"
    elif kind == "answer_missing":
        del a["P0"]["answer"]
    elif kind == "not_mapping":
        a["P0"] = "south"
    return a


MALFORMED = ["missing_id", "extra_id", "missing_action", "extra_action", "nan", "inf", "negative", "above_one",
             "bool", "string", "not_normalized", "answer_not_top", "answer_unknown", "answer_missing", "not_mapping"]


@pytest.mark.parametrize("kind", MALFORMED)
def test_malformed_predictions_fail_closed(monkeypatch, kind):
    row = _row()
    with pytest.raises(ValueError):
        p0b.validate_answers(_mutate(answers_for(row["questions"]), kind), row["questions"])
    fakes.install(monkeypatch)
    model = FakeModel(predict_output=lambda qs: _mutate(answers_for(qs), kind))
    with pytest.raises(ValueError):
        p0b.predict_answers(model, TOK, row)


def test_rounded_upstream_probabilities_are_accepted():
    qs = _row(1)["questions"]
    rounded = {"P0": {"answer": "east", "probabilities": dict(stay=.1111, north=.1111, south=.1111, east=.5556,
                                                               west=.1111)}}
    assert p0b.validate_answers(rounded, qs)["P0"]["answer"] == "east"


@pytest.mark.parametrize("change", ["four_options", "extra_option", "not_choice", "empty"])
def test_mismatched_questions_are_rejected(monkeypatch, change):
    fakes.install(monkeypatch)
    row = _row()
    q = row["questions"]["P0"]
    if change == "four_options":
        q["criteria"].pop("west")
    elif change == "extra_option":
        q["criteria"]["jump"] = "jump"
    elif change == "not_choice":
        q["type"] = "score"
    else:
        row["questions"] = {}
    with pytest.raises(ValueError):
        p0b.predict_answers(FakeModel(), TOK, row)


# ------------------------------------------------------------ serving and batching
def test_default_path_serves_verified_predict_with_labelled_timing(monkeypatch):
    fakes.install(monkeypatch)
    row = _row()
    out = p0b.decide_batch(FakeModel(), TOK, [row, row], metadata=provenance(), sync=lambda: False)
    assert [o["path"] for o in out] == ["predict", "predict"]
    assert all(o["status"] == "OK" and o["truncated"] is False for o in out)
    timing = out[0]["timing"]
    assert timing["scope"] == "server_batch_execution" and timing["batch_size"] == 2
    assert timing["cuda_synchronized"] is False and "network" in timing["excludes"]


def test_default_path_fails_whole_batch_on_unverified_request(monkeypatch):
    fakes.install(monkeypatch, contract_ok=False)
    with pytest.raises(RuntimeError, match="UNVERIFIED"):
        p0b.decide_batch(FakeModel(), TOK, [_row()], metadata=provenance())


@pytest.mark.parametrize("evidence", [None, {"status": "PASS"}, {"status": "PASS", "verdicts": {"short_first": "PASS"}}])
def test_batched_forward_refused_without_verified_evidence(monkeypatch, evidence):
    fakes.install(monkeypatch)
    called = []
    monkeypatch.setattr(p0b, "batched_probabilities", lambda *a: called.append(1))
    with pytest.raises(RuntimeError, match="refusing"):
        p0b.decide_batch(FakeModel(), TOK, [_row()], metadata=provenance(), batched_forward=True,
                         batched_evidence=evidence)
    assert not called


RUNTIME = provenance(kernel_backend={"detected_packages": ["fla"], "executed": None, "evidence": "import_check_only"})


def test_complete_matching_padding_evidence_is_accepted(monkeypatch):
    evidence = fakes.padding_evidence(monkeypatch)
    result = p0b.verify_batched_forward_evidence(evidence, RUNTIME)
    assert result["status"] == "VERIFIED" and set(result["orders"]) == set(p0b.PADDING_ORDERS)


def _tamper(evidence, kind):
    e = copy.deepcopy(evidence)
    first = e["gate_inputs"]["short_first"]
    if kind == "verdict_only":
        return {"status": "PASS", "verdicts": {o: {"status": "PASS"} for o in p0b.PADDING_ORDERS}}
    if kind == "stored_pass_but_parity_fails":
        e["status"] = "PASS"
        right = next(c for c in first["cases"] if c["variant"] == "right")
        right["probabilities"]["P0"] = dict(stay=.6, north=.1, south=.1, east=.1, west=.1)
    elif kind == "missing_order":
        del e["gate_inputs"]["short_middle"]
    elif kind == "agreement_below_100":
        key = next(iter(e["adapter_agreement_with_predict"]))
        e["adapter_agreement_with_predict"][key] = 0.9
    elif kind == "unverified_encoding":
        e["encodings"][0].update(status="UNVERIFIED", truncated=None)
    elif kind == "unknown_revision":
        for doc in e["gate_inputs"].values():
            doc["metadata"]["model_revision"] = "unsloth/Qwen3.5-2B@unknown-commit"
    elif kind == "import_only_kernels":
        for doc in e["gate_inputs"].values():
            doc["metadata"]["kernel_backend"] = RUNTIME["kernel_backend"]
    elif kind == "wrong_task":
        e["task"] = "smoke"
    return e


@pytest.mark.parametrize("kind", ["verdict_only", "stored_pass_but_parity_fails", "missing_order",
                                  "agreement_below_100", "unverified_encoding", "unknown_revision",
                                  "import_only_kernels", "wrong_task"])
def test_incomplete_or_tampered_padding_evidence_is_refused(monkeypatch, kind):
    evidence = fakes.padding_evidence(monkeypatch)
    with pytest.raises(ValueError):
        p0b.verify_batched_forward_evidence(_tamper(evidence, kind), RUNTIME)


@pytest.mark.parametrize("field, value", [("gpu", "NVIDIA L40S"), ("precision", "4bit"), ("max_seq", 8192),
                                          ("runtime_versions", {"torch": "2.9.0", "transformers": "5.2.0",
                                                                "unsloth": "fake"})])
def test_evidence_from_a_different_stack_is_refused(monkeypatch, field, value):
    evidence = fakes.padding_evidence(monkeypatch)
    with pytest.raises(ValueError, match="different stack"):
        p0b.verify_batched_forward_evidence(evidence, {**RUNTIME, field: value})


def test_serving_runtime_without_pinned_revision_cannot_enable_batching(monkeypatch):
    evidence = fakes.padding_evidence(monkeypatch)
    with pytest.raises(ValueError, match="model_revision"):
        p0b.verify_batched_forward_evidence(evidence, {**RUNTIME, "model_revision": None})


# ------------------------------------------------------------ provenance and timing
def test_model_revision_never_guesses():
    assert p0b.model_revision("m", None) is None
    assert p0b.model_revision("m", "unknown-commit") is None
    assert p0b.model_revision("m", "abc123") is None
    assert p0b.model_revision("m", fakes.COMMIT) == f"m@{fakes.COMMIT}"


def test_tail_percentiles_are_withheld_for_small_samples():
    small = p0b.latency_summary([i / 100 for i in range(1, 26)])
    assert small["n"] == 25 and small["p95_s"] is not None and small["p99_s"] is None
    assert "not tail-latency evidence" in small["tail_note"]
    tiny = p0b.latency_summary([0.2, 0.1, 0.3])
    assert tiny["p95_s"] is None and tiny["p99_s"] is None and tiny["max_s"] == 0.3
    big = p0b.latency_summary([float(i) for i in range(1, 101)])
    assert (big["p50_s"], big["p95_s"], big["p99_s"]) == (50.0, 95.0, 99.0) and "tail_note" not in big
    with pytest.raises(ValueError):
        p0b.latency_summary([1.0])


def test_timed_synchronizes_around_every_repetition():
    calls = []
    result = p0b._timed(lambda: calls.append("fn"), repeats=3, warmup=1, sync=lambda: calls.append("sync") or True)
    assert calls == ["fn", "sync"] + ["sync", "fn", "sync"] * 3
    assert result["n"] == 3 and result["cuda_synchronized"] is True and result["warmup"] == 1


def test_smoke_is_not_pass_without_preservation_evidence(monkeypatch):
    fakes.install(monkeypatch, contract_ok=False)
    report = p0b.task_smoke(FakeModel(), TOK, [1, 3], repeats=2, question_style="compact")
    assert report["status"] == "UNVERIFIED"
    assert all(r["status"] == "UNVERIFIED" and "predict_latency" not in r for r in report["results"])


def test_smoke_fails_on_cut_state_and_passes_on_full_state(monkeypatch):
    fakes.install(monkeypatch)
    cut = FakeModel(max_len=_full(_row(3)) - 5)
    assert p0b.task_smoke(cut, TOK, [3], 2, "compact")["status"] == "FAIL"
    ok = p0b.task_smoke(FakeModel(), TOK, [1, 3], 3, "compact")
    assert ok["status"] == "PASS"
    assert ok["results"][0]["predict_latency"]["p99_s"] is None


# ------------------------------------------------------------ agent adapters
def test_local_predictor_fails_closed_on_unverified_input(monkeypatch):
    fakes.install(monkeypatch, contract_ok=False)
    decision_agent = agent.DecisionAgent(agent.unsloth_predictor(FakeModel(), TOK), question_style="compact")
    with pytest.raises(RuntimeError, match="UNVERIFIED"):
        decision_agent.act(make_synthetic_observation(3))


def test_local_predictor_returns_actions_on_verified_input(monkeypatch):
    fakes.install(monkeypatch)
    decision_agent = agent.DecisionAgent(agent.unsloth_predictor(FakeModel(), TOK), question_style="compact")
    actions = decision_agent.act(make_synthetic_observation(3))
    assert set(actions.values()) == {"south"} and set(actions.values()) <= set(ACTIONS)


@pytest.mark.parametrize("response", [
    None,
    {"answers": {}},
    {"status": "OK", "truncated": None, "answers": {}},
    {"status": "UNVERIFIED", "truncated": None, "answers": None},
])
def test_remote_response_without_ok_evidence_is_rejected(response):
    with pytest.raises(ValueError, match="not verified"):
        agent.checked_response(response, _row()["questions"])


def test_remote_response_ids_are_checked():
    qs = _row()["questions"]
    good = {"status": "OK", "truncated": False, "answers": answers_for(qs)}
    assert set(agent.checked_response(good, qs)) == set(qs)
    bad = {**good, "answers": {**answers_for(qs), "P99": answers_for(qs)["P0"]}}
    with pytest.raises(ValueError, match="extra"):
        agent.checked_response(bad, qs)


def test_old_object_items_do_not_bypass_proof(monkeypatch):
    """The pre-audit harness assumed attribute records; they now give UNVERIFIED, not a crash or PASS."""
    fakes.install(monkeypatch)
    rec = SimpleNamespace(input_ids=list(range(5000)), questions=["P0", "P1", "P2"], truncated=False)
    monkeypatch.setattr(fakes.FastDecisionModel, "build_dataset",
                        staticmethod(lambda rows, t, m: ([rec], {"total": 3, "skipped": 0})))
    assert p0b.encode_one(_row(), TOK, FakeModel())[1]["status"] == "UNVERIFIED"
