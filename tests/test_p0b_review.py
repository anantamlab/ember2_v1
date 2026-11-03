"""Regressions for the PR #21 review findings (CPU only, mocks + synthetic observations).

1. Question preservation  2. Inspect fails closed  3. Batched-path evidence identity
and scope  4. Forged encoding evidence.
"""
import copy
import re

import pytest

import p0b_fakes as fakes
from ember_v1 import p0b
from ember_v1.dataset import question_style_of
from p0b_fakes import CharTokenizer, FakeModel, answers_for, provenance

TOK = CharTokenizer()


@pytest.fixture(autouse=True)
def _no_cuda(monkeypatch):
    monkeypatch.setattr(p0b, "_sync", lambda: False)


def _row(n=3, style="compact"):
    return p0b.synthetic_row(n, style)


# ------------------------------------------------------------ 1. question preservation
def _edit(kind):
    def mutate(source):
        qs = source["questions"]
        first = next(iter(qs))
        if kind == "instruction":
            qs[first]["instructions"] = qs[first]["instructions"] + " now"
        elif kind == "option_description":
            qs[first]["criteria"] = {**qs[first]["criteria"], "north": "move one row toward the bottom"}
        elif kind == "option_order":
            qs[first]["criteria"] = dict(reversed(list(qs[first]["criteria"].items())))
        elif kind == "question_order":
            source["questions"] = dict(reversed(list(qs.items())))
        elif kind == "renamed_id":
            source["questions"] = {("X" + k if k == first else k): v for k, v in qs.items()}
        elif kind == "type":
            qs[first]["type"] = "score"
        elif kind == "extra_key":
            qs[first]["hint"] = "go south"
        elif kind == "non_string_instruction":
            qs[first]["instructions"] = ["Move for P0 at 0,1"]
    return mutate


@pytest.mark.parametrize("kind", ["instruction", "option_description", "option_order", "question_order",
                                  "renamed_id", "type", "extra_key", "non_string_instruction"])
def test_changed_question_fails_preservation_with_all_five_actions_present(monkeypatch, kind):
    fakes.install(monkeypatch)
    model = FakeModel(mutate_source=_edit(kind))
    record, ev = p0b.encode_one(_row(), TOK, model)
    assert ev["status"] == "FAIL_QUESTION_CHANGED" and ev["truncated"] is None and record is None
    res = p0b.predict_answers(model, TOK, _row())
    assert res["status"] == "FAIL_QUESTION_CHANGED" and res["answers"] is None and model.predict_calls == 0


def test_option_text_change_keeps_action_ids_but_changes_the_encoding():
    """Why IDs alone are not enough: the encoder renders every option description."""
    row = _row()
    edited = copy.deepcopy(row)
    edited["questions"]["P0"]["criteria"]["north"] = "row+1"
    assert set(edited["questions"]["P0"]["criteria"]) == set(row["questions"]["P0"]["criteria"])
    assert fakes.encode_record(TOK, edited).input_ids != fakes.encode_record(TOK, row).input_ids


@pytest.mark.parametrize("change", ["extra_key", "empty_instructions", "non_string_description"])
def test_questions_outside_the_encoded_contract_are_rejected(change):
    qs = _row()["questions"]
    if change == "extra_key":
        qs["P0"]["hint"] = "x"
    elif change == "empty_instructions":
        qs["P0"]["instructions"] = ""
    else:
        qs["P0"]["criteria"]["stay"] = None
    with pytest.raises(ValueError):
        p0b.validate_questions(qs)


def test_unchanged_questions_still_pass(monkeypatch):
    fakes.install(monkeypatch)
    record, ev = p0b.encode_one(_row(), TOK, FakeModel())
    assert ev["status"] == "OK" and record is not None


# ------------------------------------------------------------ 2. inspect fails closed
def _inspect(monkeypatch, *, contract_ok=True, model=None, batched=None, kernels=None, metadata="default",
             upstream=None):
    fakes.install(monkeypatch, contract_ok=contract_ok)
    if upstream is not None:
        monkeypatch.setattr(p0b, "_upstream_contract", lambda: upstream)
    monkeypatch.setattr(p0b, "batched_probabilities", batched or (
        lambda m, t, records, qs: [{pid: a["probabilities"] for pid, a in answers_for(q).items()} for q in qs]))
    monkeypatch.setattr(p0b, "_profile_kernels",
                        lambda fn: kernels or {"executed": ["fake_kernel"], "evidence": "torch_profiler"})
    meta = provenance() if metadata == "default" else metadata
    return p0b.task_inspect(model or FakeModel(), TOK, "compact", metadata=meta)


def test_successful_inspect_is_an_explicit_pass(monkeypatch):
    report = _inspect(monkeypatch)
    assert report["status"] == "PASS" and p0b.result_exit_code(report) == 0
    assert set(report["checks"]) == set(p0b.INSPECT_CHECKS)
    assert all(v == "PASS" for v in report["checks"].values()) and report["errors"] == {}
    assert report["encoding"]["status"] == "OK" and report["batched_vs_predict_argmax_equal"] is True


def _raise(*args):
    raise RuntimeError("forward signature differs")


@pytest.mark.parametrize("case, status, failed_check", [
    ("encoder_drift", "FAIL", "clef_encoder"),
    ("adapter_raises", "FAIL", "batched_adapter"),
    ("adapter_disagrees", "FAIL", "adapter_agreement"),
    ("malformed_predict", "FAIL", "native_predict"),
    ("cut_state", "FAIL", "encoding"),
    ("bad_provenance", "FAIL", "provenance"),
    ("no_provenance", "UNVERIFIED", "provenance"),
    ("profiler_failed", "UNVERIFIED", "kernel_profile"),
    ("upstream_drift", "UNVERIFIED", "upstream_contract"),
    ("forward_signature_drift", "FAIL", "decision_head"),
    ("no_head", "FAIL", "decision_head"),
    ("other_model_commit", "FAIL", "model_pin"),
])
def test_failed_inspect_is_never_pass_and_keeps_its_report(monkeypatch, case, status, failed_check):
    kwargs = {}
    if case == "encoder_drift":
        kwargs["contract_ok"] = False
    elif case == "adapter_raises":
        kwargs["batched"] = _raise
    elif case == "adapter_disagrees":
        kwargs["batched"] = lambda m, t, recs, qs: [{pid: dict(stay=.6, north=.1, south=.1, east=.1, west=.1)
                                                     for pid in q} for q in qs]
    elif case == "malformed_predict":
        kwargs["model"] = FakeModel(predict_output=lambda qs: {pid: {"answer": "fly"} for pid in qs})
    elif case == "cut_state":
        full = len(fakes.encode_record(TOK, _row(), max_length=10**9).input_ids)
        kwargs["model"] = FakeModel(max_len=full - 5)
    elif case == "bad_provenance":
        kwargs["metadata"] = provenance(model_revision="unsloth/Qwen3.5-2B@unknown-commit")
    elif case == "no_provenance":
        kwargs["metadata"] = None
    elif case == "profiler_failed":
        kwargs["kernels"] = {"executed": None, "evidence": "profiler_failed: RuntimeError"}
    elif case == "upstream_drift":
        kwargs["upstream"] = {**fakes.audited_contract(), "decision_from_lm_matches_audited": False,
                              "all_match_audited": False}
    elif case == "forward_signature_drift":
        class OldForward(FakeModel):
            def forward(self, batch):
                return batch
        kwargs["model"] = OldForward()
    elif case == "no_head":
        model = FakeModel()
        model.head = None
        kwargs["model"] = model
    elif case == "other_model_commit":
        kwargs["metadata"] = provenance(model_revision="unsloth/Qwen3.5-2B@" + "a" * 40)
    report = _inspect(monkeypatch, **kwargs)
    assert report["status"] == status and p0b.result_exit_code(report) == 2
    assert report["checks"][failed_check] == status
    assert report["required_checks"] == list(p0b.INSPECT_CHECKS) and "upstream_contract" in report


def test_missing_checks_never_pass():
    assert p0b.inspection_status({}) == "UNVERIFIED"
    almost = {name: "PASS" for name in p0b.INSPECT_CHECKS[:-1]}
    assert p0b.inspection_status(almost) == "UNVERIFIED"
    assert p0b.inspection_status({**almost, p0b.INSPECT_CHECKS[-1]: "FAIL"}) == "FAIL"


@pytest.mark.parametrize("result", [None, {}, {"status": "SEE_FILE"}, {"status": "UNVERIFIED"}, {"status": "FAIL"},
                                    {"status": "pass"}, "PASS"])
def test_entrypoint_exit_code_is_zero_only_for_explicit_pass(result):
    assert p0b.result_exit_code(result) == 2
    assert p0b.result_exit_code({"status": "PASS"}) == 0


def test_throughput_status():
    assert p0b.throughput_status([{"status": "OK"}, {"status": "OOM"}]) == "PASS"
    assert p0b.throughput_status([{"status": "OOM"}]) == "FAIL"
    assert p0b.throughput_status([]) == "FAIL"


# ------------------------------------------------------------ 3. batched evidence identity and scope
RUNTIME = provenance(kernel_backend={"detected_packages": [], "executed": None, "evidence": "import_check_only"})


def test_verified_evidence_records_its_measured_scope(monkeypatch):
    result = p0b.verify_batched_forward_evidence(fakes.padding_evidence(monkeypatch), RUNTIME)
    scope = result["scope"]
    assert scope["question_style"] == "compact" and scope["max_batch_size"] == 3
    assert scope["min_input_tokens"] < scope["max_input_tokens"]


@pytest.mark.parametrize("key", p0b.REQUIRED_IDENTITY_KEYS)
def test_unknown_implementation_identity_keeps_fast_path_off(monkeypatch, key):
    evidence = fakes.padding_evidence(monkeypatch, meta={**RUNTIME, key: None})
    with pytest.raises(ValueError):
        p0b.verify_batched_forward_evidence(evidence, {**RUNTIME, key: None})


@pytest.mark.parametrize("key, value", [
    ("ember_v1_source_sha256", "5" * 64), ("head_sha256", "6" * 64), ("encoder_config_sha256", "7" * 64),
    ("tokenizer_sha256", "8" * 64), ("decision_config", {"layout": "clef", "max_len": 4096}),
    ("gpu_capability", "8.6"), ("cuda", "12.4"), ("model_name", "unsloth/Qwen3.5-4B"),
    ("kernel_backend", {"detected_packages": ["fla", "causal_conv1d"], "executed": None,
                        "evidence": "import_check_only"}),
])
def test_evidence_is_not_reusable_across_implementations(monkeypatch, key, value):
    evidence = fakes.padding_evidence(monkeypatch)
    with pytest.raises(ValueError, match="different stack"):
        p0b.verify_batched_forward_evidence(evidence, {**RUNTIME, key: value})


def test_unaudited_encoder_keeps_fast_path_off(monkeypatch):
    runtime = {**RUNTIME, "upstream_contract": {"clef_matches_audited": False}}
    evidence = fakes.padding_evidence(monkeypatch, meta=runtime)
    with pytest.raises(ValueError, match="audited"):
        p0b.verify_batched_forward_evidence(evidence, runtime)


def test_evidence_without_question_style_is_refused(monkeypatch):
    evidence = fakes.padding_evidence(monkeypatch)
    evidence.pop("question_style")
    with pytest.raises(ValueError, match="question_style"):
        p0b.verify_batched_forward_evidence(evidence, RUNTIME)


def _scope_fixture(monkeypatch, row, **scope):
    fakes.install(monkeypatch)
    _, ev = p0b.encode_one(row, TOK, FakeModel())
    calls = []
    monkeypatch.setattr(p0b, "batched_probabilities", lambda m, t, recs, qs: calls.append(len(recs)) or [
        {pid: a["probabilities"] for pid, a in answers_for(q).items()} for q in qs])
    base = {"question_style": "compact", "min_input_tokens": ev["input_tokens"],
            "max_input_tokens": ev["input_tokens"], "max_batch_size": 2}
    return {"status": "VERIFIED", "scope": {**base, **scope}}, calls


def test_batched_path_serves_inside_the_validated_scope(monkeypatch):
    row = _row()
    evidence, calls = _scope_fixture(monkeypatch, row)
    out = p0b.decide_batch(FakeModel(), TOK, [row, row], metadata=provenance(), batched_forward=True,
                           batched_evidence=evidence, sync=lambda: False)
    assert calls == [2] and [o["path"] for o in out] == ["batched_right_pad"] * 2


@pytest.mark.parametrize("case", ["style", "too_many", "too_long", "too_short", "unknown_style"])
def test_batched_path_refuses_requests_outside_the_validated_scope(monkeypatch, case):
    row = _row(3, "verbose" if case == "style" else "compact")
    if case == "unknown_style":
        row["questions"]["P0"]["instructions"] = "Pick a move for P0"
    evidence, calls = _scope_fixture(monkeypatch, row)
    if case == "style":
        evidence["scope"]["question_style"] = "compact"
    elif case == "too_long":
        evidence["scope"]["max_input_tokens"] -= 1
    elif case == "too_short":
        evidence["scope"]["min_input_tokens"] += 1
    requests = [row] * (3 if case == "too_many" else 1)
    with pytest.raises(RuntimeError, match="validated|exceeds"):
        p0b.decide_batch(FakeModel(), TOK, requests, metadata=provenance(), batched_forward=True,
                         batched_evidence=evidence, sync=lambda: False)
    assert calls == []


def test_question_style_detection_is_exact():
    assert question_style_of(_row(3, "compact")["questions"]) == "compact"
    assert question_style_of(_row(3, "verbose")["questions"]) == "verbose"
    mixed = {**_row(2, "compact")["questions"], **{"P9": _row(1, "verbose")["questions"]["P0"]}}
    assert question_style_of(mixed) is None
    wrong_id = {"P5": _row(1, "compact")["questions"]["P0"]}  # instruction names P0, key says P5
    assert question_style_of(wrong_id) is None


def test_package_source_hash_is_a_stable_digest():
    first = p0b.package_source_sha256()
    assert re.fullmatch(r"[0-9a-f]{64}", first) and first == p0b.package_source_sha256()


# ------------------------------------------------------------ 4. forged encoding evidence
def test_fabricated_ok_evidence_cannot_bypass_the_proof(monkeypatch):
    fakes.install(monkeypatch, contract_ok=False)  # the real proof would be UNVERIFIED
    model, row = FakeModel(), _row()
    forged = {"status": "OK", "truncated": False, "row_sha256": p0b._row_digest(row), "input_tokens": 10,
              "full_tokens": 10, "token_limit": 16384, "seal": "0" * 64}
    for evidence in (forged, {k: v for k, v in forged.items() if k != "seal"}):
        with pytest.raises(ValueError, match="not produced by encode_one"):
            p0b.predict_answers(model, TOK, row, evidence=evidence)
    assert model.predict_calls == 0


def test_modified_genuine_evidence_is_rejected(monkeypatch):
    fakes.install(monkeypatch, contract_ok=False)
    model, row = FakeModel(), _row()
    _, genuine = p0b.encode_one(row, TOK, model)
    assert genuine["status"] == "UNVERIFIED"
    upgraded = {**genuine, "status": "OK", "truncated": False}
    with pytest.raises(ValueError, match="not produced by encode_one"):
        p0b.predict_answers(model, TOK, row, evidence=upgraded)
    assert model.predict_calls == 0


def test_genuine_evidence_reuse_skips_re_encoding_and_stays_correct(monkeypatch):
    fakes.install(monkeypatch)
    model, row = FakeModel(), _row()
    _, evidence = p0b.encode_one(row, TOK, model)
    assert model.build_calls == 1
    res = p0b.predict_answers(model, TOK, row, evidence=evidence)
    assert res["status"] == "OK" and res["encoding"] == evidence and model.build_calls == 1
    with pytest.raises(ValueError, match="different"):
        p0b.predict_answers(model, TOK, _row(4), evidence=evidence)  # genuine, but another row
    unverified_model = FakeModel(max_len=10**9, report_truncated=1)
    _, conflict = p0b.encode_one(row, TOK, unverified_model)
    assert p0b.predict_answers(unverified_model, TOK, row, evidence=conflict)["status"] == \
        "UNVERIFIED_REPORT_CONFLICT"


# ------------------------------------------------------------ review round 2: ordered digest
def _reorder_criteria(row, pid="P1"):
    out = copy.deepcopy(row)
    criteria = out["questions"][pid]["criteria"]
    out["questions"][pid]["criteria"] = dict(reversed(list(criteria.items())))
    return out


def test_criteria_reorder_cannot_reuse_old_evidence(monkeypatch):
    fakes.install(monkeypatch)
    model, row = FakeModel(), _row()
    _, evidence = p0b.encode_one(row, TOK, model)
    assert evidence["status"] == "OK"
    reordered = _reorder_criteria(row)
    assert list(reordered["questions"]["P1"]["criteria"]) != list(row["questions"]["P1"]["criteria"])
    with pytest.raises(ValueError, match="different state/questions"):
        p0b.predict_answers(model, TOK, reordered, evidence=evidence)
    assert model.predict_calls == 0


def _changed(row, kind):
    out = copy.deepcopy(row)
    q = out["questions"]
    if kind == "criteria_order":
        return _reorder_criteria(row)
    if kind == "question_order":
        out["questions"] = dict(reversed(list(q.items())))
    elif kind == "description":
        q["P0"]["criteria"]["stay"] += " "
    elif kind == "instructions":
        q["P0"]["instructions"] += " "
    elif kind == "type":
        q["P0"]["type"] = "multi_choice"
    elif kind == "state":
        out["state"] += " "
    elif kind == "question_id":
        out["questions"] = {("Q0" if k == "P0" else k): v for k, v in q.items()}
    return out


@pytest.mark.parametrize("kind", ["criteria_order", "question_order", "description", "instructions", "type",
                                  "state", "question_id"])
def test_row_digest_changes_with_every_model_relevant_part(kind):
    row = _row()
    assert p0b._row_digest(_changed(row, kind)) != p0b._row_digest(row)


def test_row_digest_is_stable_for_an_unchanged_row():
    row = _row()
    assert p0b._row_digest(row) == p0b._row_digest(copy.deepcopy(row)) == p0b._row_digest(_row())
    assert re.fullmatch(r"[0-9a-f]{64}", p0b._row_digest(row))


# ------------------------------------------------------------ review round 2: environment binding
def _bound_evidence(monkeypatch):
    fakes.install(monkeypatch)
    model, tokenizer, row = FakeModel(), CharTokenizer(), _row()
    _, evidence = p0b.encode_one(row, tokenizer, model)
    assert evidence["status"] == "OK" and evidence["binding"] is not None
    return model, tokenizer, row, evidence


def _rejects(model, tokenizer, row, evidence):
    with pytest.raises(ValueError, match="different model/tokenizer/encoding configuration"):
        p0b.predict_answers(model, tokenizer, row, evidence=evidence)
    assert model.predict_calls == 0


def test_evidence_is_not_reusable_with_another_model_instance(monkeypatch):
    model, tokenizer, row, evidence = _bound_evidence(monkeypatch)
    _rejects(FakeModel(), tokenizer, row, evidence)
    assert model.predict_calls == 0


def test_evidence_is_not_reusable_with_another_tokenizer_instance(monkeypatch):
    model, tokenizer, row, evidence = _bound_evidence(monkeypatch)
    _rejects(model, CharTokenizer(), row, evidence)


@pytest.mark.parametrize("change", ["max_len_down", "max_len_up", "extra_key"])
def test_evidence_is_not_reusable_after_an_encoding_config_change(monkeypatch, change):
    model, tokenizer, row, evidence = _bound_evidence(monkeypatch)
    if change == "max_len_down":
        model.decision_config["max_len"] = evidence["full_tokens"]  # the row no longer fits
    elif change == "max_len_up":
        model.decision_config["max_len"] = 32768
    else:
        model.decision_config["chat_template"] = "other"
    _rejects(model, tokenizer, row, evidence)


def test_evidence_is_reusable_with_the_same_model_tokenizer_and_row(monkeypatch):
    model, tokenizer, row, evidence = _bound_evidence(monkeypatch)
    for _ in range(2):
        res = p0b.predict_answers(model, tokenizer, copy.deepcopy(row), evidence=evidence)
        assert res["status"] == "OK" and res["encoding"] == evidence
    assert model.build_calls == 1 and model.predict_calls == 2


def test_untrackable_identity_recomputes_the_proof(monkeypatch):
    assert p0b._object_token(7) is None  # an int can not be weak-referenced
    a, b = CharTokenizer(), CharTokenizer()
    assert p0b._object_token(a) == p0b._object_token(a) != p0b._object_token(b)
    model, tokenizer, row, evidence = _bound_evidence(monkeypatch)
    monkeypatch.setattr(p0b, "_encoding_binding", lambda m, t: None)
    res = p0b.predict_answers(model, tokenizer, row, evidence=evidence)
    assert res["status"] == "OK" and model.build_calls == 2 and res["encoding"]["binding"] is None
