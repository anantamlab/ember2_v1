"""CPU tests of the A10 `chunk_smoke` task logic with fake encode/predict (no torch, no unsloth, no GPU)."""
import types

import pytest

from ember_v1 import p0b_chunk_smoke as cs
from ember_v1.contract import ACTIONS
from ember_v1.dataset import question_style_of
from ember_v1.p0b import synthetic_row

N200_STATE = None


def _n200_state():
    global N200_STATE
    if N200_STATE is None:
        N200_STATE = synthetic_row(200, "compact")["state"]
    return N200_STATE


def fake_encode(limit_compact=100, limit_verbose_n200=50, single_n200_ok=False):
    """Mirror of the CPU token counts: <=100 people fit, except verbose rows that carry the N=200 state."""
    def encode_one(row, tokenizer, model):
        q = row["questions"]
        style = question_style_of(q)
        big_state = row["state"] == _n200_state()
        cap = limit_verbose_n200 if (style == "verbose" and big_state) else limit_compact
        ok = len(q) <= cap or (single_n200_ok and len(q) == 200)
        status = "OK" if ok else ("SCHEMA_TOO_LONG" if len(q) > 100 else "FAIL_NO_SPARE_TOKEN")
        return None, {"status": status, "truncated": False if ok else None, "input_tokens": 10 * len(q),
                      "full_tokens": 10 * len(q)}
    return encode_one


def fake_predict(drop=None):
    def predict_answers(model, tokenizer, row, *, evidence):
        probs = {a: (1.0 if a == "stay" else 0.0) for a in ACTIONS}
        ans = {pid: {"answer": "stay", "probabilities": probs} for pid in row["questions"] if pid != drop}
        return {"status": "OK", "truncated": False, "answers": ans}
    return predict_answers


PROV_OK = {"status": "PASS"}


def run(**kw):
    args = dict(encode_one=fake_encode(), predict_answers=fake_predict(), synthetic_row=synthetic_row,
                timed=lambda fn, r: {"repeats": r, "mean_s": 0.0}, peak_memory_gb=lambda: 1.5, provenance=PROV_OK)
    args.update(kw)
    return cs.chunk_smoke(object(), object(), **args)


def test_expected_outcomes_pass():
    r = run()
    assert r["status"] == "PASS"
    assert [(e["style"], e["n"]) for e in r["single_rows"]] == [(s, n) for s in cs.STYLES for n in cs.SINGLE_NS]
    assert all(e["status"] == "PASS" and e["latency"]["repeats"] == 5 for e in r["single_rows"])
    assert all(e["status"] == "PASS" and e["encode_status"] != "OK" for e in r["n200_single_row"])
    by = {(e["style"], e["k"]): e for e in r["n200_chunked"]}
    assert by[("compact", 100)]["joint_people"] == 200 and len(by[("compact", 100)]["rows"]) == 2
    assert by[("verbose", 50)]["joint_people"] == 200 and len(by[("verbose", 50)]["rows"]) == 4
    assert by[("verbose", 100)]["status"] == "PASS" and "joint_people" not in by[("verbose", 100)]
    assert sum(by[("compact", 100)]["resolver_outcomes"].values()) == 200
    assert r["peak_memory_gb"] == 1.5 and "no training" in r["scope"]


def test_n200_single_row_accepted_is_a_fail():
    r = run(encode_one=fake_encode(single_n200_ok=True))
    assert r["status"] == "FAIL" and all(e["status"] == "FAIL" for e in r["n200_single_row"])


def test_verbose_k100_accepted_is_a_fail():
    r = run(encode_one=fake_encode(limit_verbose_n200=100))
    by = {(e["style"], e["k"]): e["status"] for e in r["n200_chunked"]}
    assert by[("verbose", 100)] == "FAIL" and r["status"] == "FAIL"


def test_missing_person_in_a_chunk_prediction_fails_closed():
    r = run(predict_answers=fake_predict(drop="P150"))
    by = {(e["style"], e["k"]): e for e in r["n200_chunked"]}
    assert by[("compact", 100)]["status"] == "FAIL" and "keys differ" in by[("compact", 100)]["joint_error"]
    assert r["status"] == "FAIL"


def test_unverified_provenance_blocks_pass():
    assert run(provenance={"status": "UNVERIFIED"})["status"] == "UNVERIFIED"


def test_gpu_plan_needs_its_own_authorization():
    from ember_v1 import p0b
    with pytest.raises(PermissionError, match="--authorize chunk_smoke"):
        p0b.gpu_run_plan("chunk_smoke", "A10", p0b.PINNED_MODEL, "bf16", "smoke", "a" * 40)
    plan = p0b.gpu_run_plan("chunk_smoke", "A10", p0b.PINNED_MODEL, "bf16", "chunk_smoke", "a" * 40)
    assert plan["retries"] == 0 and plan["max_containers"] == 1 and plan["timeout_s"] == cs.TIMEOUT_S
    assert plan["cost_estimate"]["kind"] == "ESTIMATE_NOT_A_BILLING_CAP"


def test_repeats_bound():
    with pytest.raises(ValueError):
        run(repeats=1)


class _Mod:
    def __init__(self, **fns):
        self.__dict__.update(fns)


def _fn(module):
    def f(*a):
        return a
    f.__module__ = module
    return f


def _model(*fns):
    return types.SimpleNamespace(named_modules=lambda: [(f"layers.{i}.linear_attn", _Mod(causal_conv1d_fn=fn))
                                                        for i, fn in enumerate(fns)])


VEND = types.SimpleNamespace(__file__="/usr/lib/python3/site-packages/unsloth_zoo/_vendored/fla/__init__.py",
                             __version__="0.4.0", _UNSLOTH_VENDORED_FLA=True)


def test_provenance_pass_records_facts():
    fn = _fn("unsloth_zoo.temporary_patches.gdn_causal_conv1d")
    p = cs.kernel_provenance(_model(fn, fn), modules={"unsloth_zoo._vendored.fla": VEND})
    assert p["status"] == "PASS" and p["vendored_fla"]
    assert len(p["conv_bindings"]) == 1 and p["conv_bindings"][0]["attribute"] == "causal_conv1d_fn"
    assert p["fla_modules"][0]["version"] == "0.4.0" and "_UNSLOTH_VENDORED_FLA" in p["fla_modules"][0]["markers"]


def test_provenance_fails_on_upstream_conv_or_fla():
    up = _fn("causal_conv1d.causal_conv1d_interface")
    ok = _fn("unsloth_zoo._vendored.fla.modules.conv.causal_conv1d")
    assert cs.kernel_provenance(_model(ok, up), modules={"fla": VEND})["status"] == "FAIL"
    site = types.SimpleNamespace(__file__="/site-packages/fla/__init__.py", __version__="0.4.0")
    assert cs.kernel_provenance(_model(ok), modules={"fla": site})["status"] == "FAIL"


def test_provenance_without_observed_binding_is_unverified():
    assert cs.kernel_provenance(_model(), modules={"fla": VEND})["status"] == "UNVERIFIED"
    assert cs.kernel_provenance(_model(_fn("unsloth_zoo._vendored.fla.x")), modules={})["status"] == "UNVERIFIED"
