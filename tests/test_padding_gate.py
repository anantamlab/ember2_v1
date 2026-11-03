"""CPU-only verification of the verdict logic, NOT an actual GPU parity test."""
import copy
import pytest
from ember_v1.padding_gate import assess_padding_parity
from p0b_fakes import provenance


def measured_fixture():
    p1 = dict(stay=.05, north=.1, south=.1, east=.70, west=.05)
    p2 = dict(stay=.05, north=.05, south=.75, east=.10, west=.05)
    return {
        "metadata": provenance(),
        "cases": [{"env_id": env, "variant": variant, "input_tokens": length,
                   "truncated": False, "probabilities": {pid: copy.deepcopy(values)}}
                  for env, pid, length, values in (("e_short", "P0", 20, p1), ("e_long", "P10", 120, p2))
                  for variant in ("single", "right", "left")],
    }


def test_identical_predictions_pass():
    report = assess_padding_parity(measured_fixture())
    assert report["status"] == "PASS"
    assert report["environments"] == 2
    assert not report["packed_tested"]


def test_padding_leakage_detected_even_when_argmax_stable():
    data = measured_fixture()
    data["cases"][1]["probabilities"]["P0"]["east"] -= .03
    data["cases"][1]["probabilities"]["P0"]["west"] += .03
    assert assess_padding_parity(data)["status"] == "FAIL"


def test_confident_action_flip_fails():
    data = measured_fixture()
    p = data["cases"][2]["probabilities"]["P0"]
    p["east"], p["north"] = p["north"], p["east"]
    verdict = assess_padding_parity(data)
    assert verdict["status"] == "FAIL"
    assert any(r["confident_flips"] == ["P0"] for r in verdict["comparisons"])


@pytest.mark.parametrize("mutation", ["missing_right", "missing_person", "truncated", "unverified_truncation",
                                      "missing_backend"])
def test_missing_or_invalid_evidence_fails_closed(mutation):
    data = measured_fixture()
    if mutation == "missing_right":
        data["cases"] = [x for x in data["cases"] if not (x["env_id"] == "e_short" and x["variant"] == "right")]
    elif mutation == "missing_person":
        data["cases"][1]["probabilities"] = {"WRONG": data["cases"][1]["probabilities"]["P0"]}
    elif mutation == "truncated":
        data["cases"][1]["truncated"] = True
    elif mutation == "unverified_truncation":
        data["cases"][1]["truncated"] = None  # UNVERIFIED is never "not truncated"
    elif mutation == "missing_backend":
        data["metadata"].pop("kernel_backend")
    with pytest.raises(ValueError):
        assess_padding_parity(data)


@pytest.mark.parametrize("field, value, match", [
    ("model_revision", "unsloth/Qwen3.5-2B@unknown-commit", "model_revision"),
    ("model_revision", None, "model_revision"),
    ("runtime_versions", "test-only", "runtime_versions"),
    ("runtime_versions", {"torch": "2.8.0", "transformers": "5.2.0", "unsloth": None}, "runtime_versions"),
    ("gpu", "", "gpu"),
    ("evidence_source", None, "evidence_source"),
    ("precision", "fp8", "precision"),
    ("max_seq", None, "max_seq"),
    ("token_limit", 0, "token_limit"),
    ("kernel_backend", "fla+causal_conv1d", "kernel_backend"),
    ("kernel_backend", {"detected_packages": ["fla"], "executed": None, "evidence": "import_check_only"},
     "profiled kernel"),
])
def test_placeholder_or_absent_provenance_fails_closed(field, value, match):
    data = measured_fixture()
    data["metadata"][field] = value
    with pytest.raises(ValueError, match=match):
        assess_padding_parity(data)


def test_unknown_tokenizer_revision_is_reported_not_hidden():
    data = measured_fixture()
    data["metadata"]["tokenizer_revision"] = None
    assert assess_padding_parity(data)["provenance_gaps"] == ["tokenizer_revision"]
    data["metadata"].pop("tokenizer_revision")
    with pytest.raises(ValueError, match="tokenizer_revision"):
        assess_padding_parity(data)


def test_packed_requires_provenance():
    data = measured_fixture()
    for case in copy.deepcopy(data["cases"]):
        if case["variant"] == "single":
            case["variant"] = "packed"
            data["cases"].append(case)
    with pytest.raises(ValueError, match="packed path not verified"):
        assess_padding_parity(data)
    data["metadata"]["packed_varlen_verified"] = True
    assert assess_padding_parity(data)["status"] == "PASS"


def test_left_scope_exclusion_must_be_explicit():
    data = measured_fixture()
    data["cases"] = [x for x in data["cases"] if x["variant"] != "left"]
    with pytest.raises(ValueError, match="padding variant"):
        assess_padding_parity(data)
    data["metadata"]["left_unsupported_reason"] = "Clef head assumes right padding"
    verdict = assess_padding_parity(data)
    assert verdict["status"] == "PASS" and verdict["left_tested"] is False


def test_left_results_conflict_with_unsupported_flag():
    data = measured_fixture()
    data["metadata"]["left_unsupported_reason"] = "Clef head assumes right padding"
    with pytest.raises(ValueError, match="marked unsupported"):
        assess_padding_parity(data)


def test_rewritten_prompt_token_count_fails_closed():
    data = measured_fixture()
    data["cases"][2]["input_tokens"] = 19  # e_short/left differs from singleton
    with pytest.raises(ValueError, match="unpadded token count"):
        assess_padding_parity(data)


def test_identical_lengths_are_not_mixed_length_evidence():
    data = measured_fixture()
    for case in data["cases"]:
        case["input_tokens"] = 20
    with pytest.raises(ValueError, match="same encoded length"):
        assess_padding_parity(data)
