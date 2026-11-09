"""P1-15: preregistered evaluation decision code on synthetic numbers only (no student, no test set)."""
from __future__ import annotations

import copy

import pytest

from ember_v1 import student_eval_protocol as ep
from ember_v1.teacher.registry import CANDIDATES


def frozen():
    p = copy.deepcopy(ep.load_protocol())
    p["status"] = "FROZEN"                              # in memory only; the committed file stays a draft
    p["guards"]["latency_ms"].update({"p50_max": 1000, "p95_max": 3000, "p99_max": 6000,
                                      "evidence": "synthetic-test-only"})
    p["primary"]["bootstrap_resamples"] = 2000
    return p


def outcome(c, u):
    return {"casualties": c, "unresolved": u, "casualties_plus_unresolved": c + u}


def rows(student_delta=(-2, 0), comparator="T4", families=12, per=3, people=50, others=(5, 0)):
    out = []
    for f in range(families):
        for s in range(per):
            base = (6 + (f + s) % 3, 1)
            r = {"family_id": f"fam{f}", "scenario_id": f"fam{f}-s{s}", "people": people}
            for t in CANDIDATES:
                r[t] = outcome(base[0] + (0 if t == comparator else others[0]), base[1] + others[1])
            r["raw_student"] = outcome(max(0, base[0] + student_delta[0]), max(0, base[1] + student_delta[1]))
            out.append(r)
    return out


RUN = {"set_kind": "test", "safety_failures": 0, "teacher_calls_in_student_path": 0,
       "model_input_privacy_violations": 0,
       "latency_p50_ms": 200, "latency_p95_ms": 900, "latency_p99_ms": 2000}
DEPLOY = {t: True for t in CANDIDATES}
COMP = {"comparator": "T4", "set_kind": "comparator_selection"}


def test_committed_protocol_is_a_draft_that_cannot_freeze_without_latency_limits():
    p = ep.load_protocol()
    assert p["status"] == "DRAFT_NOT_FROZEN" and p["written_before_any_student_outcome"] is True
    assert sorted(ep.freeze_problems(p)) == ["latency limits have no measured GPU evidence (preflight report hash)",
                                             "latency p50_max is not set", "latency p95_max is not set",
                                             "latency p99_max is not set"]
    assert ep.freeze_problems(frozen()) == []
    loose = frozen()
    loose["guards"]["unresolved_upper_bound_max"] = 0.5            # the v0.1 margin
    assert ep.freeze_problems(loose) == ["a positive guard margin lets casualties or unresolved increase"]
    assert ep.decide(rows(), COMP, RUN, p)["verdict"] == ep.INVALID


def test_comparator_comes_from_selection_set_with_tie_breaks():
    tot = {t: {"casualties_plus_unresolved": 100, "casualties": 90} for t in CANDIDATES}
    tot["T2"] = {"casualties_plus_unresolved": 95, "casualties": 95}
    tot["T4"] = {"casualties_plus_unresolved": 95, "casualties": 80}
    sel = ep.select_comparator(tot, set_kind="comparator_selection", deployable=DEPLOY)
    assert sel["comparator"] == "T4" and sel["ranking"][:2] == ["T4", "T2"]
    tot["T4"] = dict(tot["T2"])
    assert ep.select_comparator(tot, set_kind="comparator_selection",
                                deployable=DEPLOY)["comparator"] == "T2"   # teacher order
    slow = {**DEPLOY, "T2": False}                       # best teacher misses the latency budget
    sel = ep.select_comparator(tot, set_kind="comparator_selection", deployable=slow)
    assert sel["comparator"] == "T4" and sel["ranking"][0] == "T2"      # ranked and reported, not the comparator
    with pytest.raises(ep.EvalProtocolError, match="no teacher is deployable"):
        ep.select_comparator(tot, set_kind="comparator_selection", deployable={t: False for t in CANDIDATES})
    with pytest.raises(ep.EvalProtocolError, match="never from test"):
        ep.select_comparator(tot, set_kind="test", deployable=DEPLOY)
    with pytest.raises(ep.EvalProtocolError, match="all five"):
        ep.select_comparator({"T1": tot["T1"]}, set_kind="comparator_selection", deployable=DEPLOY)
    with pytest.raises(ep.EvalProtocolError, match="comparator-selection"):
        ep.decide(rows(), {"comparator": "T4", "set_kind": "test"}, RUN, frozen())


def test_clear_improvement_is_superior_and_reports_oracle_separately():
    d = ep.decide(rows(), COMP, RUN, frozen())
    assert d["verdict"] == ep.SUPERIOR, d["reasons"]
    assert d["upper_bounds"]["casualties_plus_unresolved"] < 0
    assert d["oracle"]["deployable"] is False
    assert all(d["secondary_beats_teacher_holm"].values())


def test_matching_the_comparator_or_t1_is_not_success():
    assert ep.decide(rows(student_delta=(0, 0)), COMP, RUN, frozen())["verdict"] == ep.NOT_SUPERIOR
    # student equals T1, which is weaker than the comparator T4: still not success
    r = rows(student_delta=(0, 0), others=(5, 0))
    for x in r:
        x["raw_student"] = dict(x["T1"])
    d = ep.decide(r, COMP, RUN, frozen())
    assert d["verdict"] == ep.NOT_SUPERIOR and any("primary" in m for m in d["reasons"])


def test_casualty_guard_blocks_trading_deaths_for_fewer_unresolved():
    r = rows()
    for x in r:
        x["raw_student"] = outcome(x["T4"]["casualties"] + 1, 0)
        x["T4"] = outcome(x["T4"]["casualties"], 3)
    d = ep.decide(r, COMP, RUN, frozen())
    assert d["upper_bounds"]["casualties_plus_unresolved"] < 0
    assert d["verdict"] == ep.NOT_SUPERIOR and any("casualties may be worse" in m for m in d["reasons"])


def test_severe_regressions_safety_latency_and_independence():
    r = rows()
    for x in r[:3]:
        x["raw_student"] = outcome(x["T4"]["casualties"] + 10, 0)
    d = ep.decide(r, COMP, RUN, frozen())
    assert any("severe regressions 3" in m for m in d["reasons"])
    assert ep.decide(rows(), COMP, {**RUN, "safety_failures": 1}, frozen())["verdict"] == ep.NOT_SUPERIOR
    late = ep.decide(rows(), COMP, {**RUN, "latency_p99_ms": None}, frozen())
    assert any("p99" in m for m in late["reasons"])
    cheat = ep.decide(rows(), COMP, {**RUN, "teacher_calls_in_student_path": 1}, frozen())
    assert cheat["verdict"] == ep.INVALID
    with pytest.raises(ep.EvalProtocolError, match="sealed test"):
        ep.decide(rows(), COMP, {**RUN, "set_kind": "dev"}, frozen())


def test_bootstrap_is_deterministic_and_family_clustered():
    diffs = {"a": [-1, -1, -1], "b": [-2], "c": [0, 0]}
    kw = {"alpha": 0.025, "resamples": 500, "seed": 7}
    assert ep.bootstrap_upper(diffs, **kw) == ep.bootstrap_upper(diffs, **kw)
    assert ep.family_means(diffs) == [-1, -2, 0]           # one value per family, not per scenario
    with pytest.raises(ep.EvalProtocolError):
        ep.family_means({"a": []})
    assert ep.holm({"T0": 0.001, "T1": 0.02, "T2": 0.5}, 0.025) == {"T0": True, "T1": False, "T2": False}


def test_test_families_must_be_disjoint_from_every_other_set():
    ep.check_test_isolation({"f1", "f2"}, {"train": {"f3"}, "dev": {"f4"}, "counterfactual": {"f5"}})
    with pytest.raises(ep.EvalProtocolError, match="overlap"):
        ep.check_test_isolation({"f1", "f2"}, {"train": {"f2"}, "dev": {"f4"}})
    assert ep.family_overlaps({"x": {"1"}, "y": {"1", "2"}}) == {"x&y": ["1"]}


def test_unresolved_rise_blocks_success_even_when_the_primary_improves():
    # v0.1 allowed an unresolved upper bound of +0.5 per scenario: this student would have passed.
    r = rows()
    for x in r:
        x["raw_student"] = outcome(max(0, x["T4"]["casualties"] - 3), x["T4"]["unresolved"] + 1)
    d = ep.decide(r, COMP, RUN, frozen())
    assert d["upper_bounds"]["casualties_plus_unresolved"] < 0 and d["estimates"]["unresolved"] == 1
    assert d["verdict"] == ep.NOT_SUPERIOR and any("unresolved may be worse" in m for m in d["reasons"])
    assert not any(d["claims"].values())


def test_both_claim_needs_each_outcome_superior():
    only_cas = ep.decide(rows(student_delta=(-2, 0)), COMP, RUN, frozen())
    assert only_cas["verdict"] == ep.SUPERIOR
    assert only_cas["claims"]["casualties_improved"] and not only_cas["claims"]["unresolved_improved"]
    assert only_cas["claims"]["both_improved"] is False
    both = ep.decide(rows(student_delta=(-2, -1)), COMP, RUN, frozen())
    assert both["claims"]["both_improved"] and both["claims"]["unresolved_improved"]


def test_severe_regression_is_checked_per_outcome_and_by_rate_bound():
    r = rows()
    for x in r[:1]:                                  # C+U equal, unresolved +3: invisible to a combined check
        x["raw_student"] = outcome(max(0, x["T4"]["casualties"] - 3), x["T4"]["unresolved"] + 3)
    assert ep.severe_regression(r[0], "raw_student", "T4", frozen()["guards"]["severe_regression"]) == ["unresolved"]
    d = ep.decide(r, COMP, RUN, frozen())
    assert d["severe_regressions"] == 1 and d["severe_detail"] == {"fam0-s0": ["unresolved"]}
    # 1 of 36 is under 5 %, but its exact upper bound exceeds 0.10: a small test set cannot show a low rate
    assert d["severe_rate_upper"] > 0.10 and any("severe regressions 1 of 36" in m for m in d["reasons"])


def test_clopper_pearson_upper_bound():
    assert ep.clopper_pearson_upper(0, 96, 0.025) == pytest.approx(1 - 0.025 ** (1 / 96), abs=1e-9)
    assert ep.clopper_pearson_upper(5, 5, 0.025) == 1.0
    assert ep.clopper_pearson_upper(4, 96, 0.025) > 4 / 96
    with pytest.raises(ep.EvalProtocolError):
        ep.clopper_pearson_upper(3, 2, 0.025)


def test_privacy_audit_is_required_and_must_be_clean():
    assert ep.decide(rows(), COMP, {**RUN, "model_input_privacy_violations": 1}, frozen())["verdict"] == ep.INVALID
    assert ep.decide(rows(), COMP, {**RUN, "model_input_privacy_violations": None}, frozen())["verdict"] == ep.INVALID
