"""P1-12 TRAIN pilot preparation: plan, authorization and acceptance (no teacher runs on TRAIN)."""
from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from ember_v1 import train_pilot as tp
from ember_v1.s1_pipeline import HELD_OUT_KEYS

REPO = Path(__file__).resolve().parents[1]
TRAIN = json.loads((REPO / "data/splits/p1-02-v1/train.json").read_text(encoding="utf-8"))
EMPTY = {"other": {k: set() for k in HELD_OUT_KEYS}}
PINS = {"teachers": {t: {"version": "x", "config_hash": "c", "code_fingerprint": f"fp-{t}"}
                     for t in ("T0", "T1", "T2", "T3", "T4")}, "pin_problems": []}


@pytest.fixture(scope="module")
def plan():
    return tp.build_plan(TRAIN, EMPTY, pins=PINS)


def test_plan_is_deterministic_stratified_and_one_per_family(plan):
    assert plan == tp.build_plan(TRAIN, EMPTY, pins=PINS)
    sc = plan["scenarios"]
    assert len(sc) == 8 and Counter(s["label"] for s in sc) == Counter(tp.STRATA)
    assert len({s["family_id"] for s in sc}) == 8
    assert max(s["people"] for s in sc) == max(e["diagnostics"]["people"] for e in TRAIN["scenarios"])
    fire = Counter(s["fire_model"] for s in sc)
    assert min(fire.values()) >= 3 and set(fire) == {"deterministic", "stochastic"}
    assert plan["status"] == tp.PLAN_STATUS and plan["source"]["manifest_hash"] == TRAIN["manifest_hash"]
    other = tp.build_plan(TRAIN, EMPTY, pins=PINS, salt="another-salt")
    assert other["plan_hash"] != plan["plan_hash"]


def test_plan_fails_closed_on_heldout_overlap_and_bad_pins():
    fam = TRAIN["scenarios"][0]["family_id"]
    leak = {"dev": {**{k: set() for k in HELD_OUT_KEYS}, "family": {fam}}}
    with pytest.raises(ValueError, match="overlaps held-out"):
        tp.build_plan(TRAIN, leak, pins=PINS)
    with pytest.raises(ValueError, match="pins"):
        tp.build_plan(TRAIN, EMPTY, pins={**PINS, "pin_problems": ["T4 drift"]})


def test_cost_estimate_and_limits(plan):
    c = plan["cost_estimate"]
    assert c["t2_cpu_s"] == pytest.approx(7555.15625 / 72 * 8 * 2, abs=0.1) and c["modal_usd"] == 0
    assert plan["limits"]["retries"] == 0 and plan["limits"]["decision_cpu_s"]["T2"] == 30
    assert "never retried" in plan["timeout_rule"]


def auth(plan, **kw):
    a = {"kind": tp.AUTH_KIND, "plan_hash": plan["plan_hash"], "scenarios": 8, "attempt": 1,
         "teacher_fingerprints": {t: v["code_fingerprint"] for t, v in plan["teacher_pins"]["teachers"].items()},
         "status": "approved", "reviewers": ["Sankalp H S"]}
    a.update(kw)
    return a


def test_authorization_is_separate_and_refuses_the_p1_09_receipt(plan):
    frozen = {**plan, "status": "FROZEN"}
    assert tp.check_authorization(auth(frozen), frozen) == []
    assert tp.check_authorization(None, frozen) == ["no multi-teacher TRAIN pilot authorization"]
    assert "the pilot plan is not FROZEN" in tp.check_authorization(auth(plan), plan)
    old = {"kind": "s1-selection-receipt", "namespace": "p1-09-selection@1", "status": "approved",
           "reviewers": ["Sankalp H S"]}
    assert any("P1-09" in e for e in tp.check_authorization(old, frozen))
    assert any("plan hash" in e for e in tp.check_authorization(auth(frozen, plan_hash="sha256:0"), frozen))
    assert any("exactly 8" in e for e in tp.check_authorization(auth(frozen, scenarios=189), frozen))
    assert any("fingerprints" in e for e in tp.check_authorization(auth(frozen, teacher_fingerprints={}), frozen))
    for r in ([], ["TBD"], ["claude"]):
        assert any("named reviewers" in e for e in tp.check_authorization(auth(frozen, reviewers=r), frozen))


def test_execution_and_second_attempt_are_refused(tmp_path):
    with pytest.raises(tp.PilotBlocked, match="not authorized"):
        tp.run_pilot()
    with pytest.raises(tp.PilotBlocked, match="not authorized"):
        tp.main(["run"])
    tp.check_attempt_marker(tmp_path, 1)
    (tmp_path / "attempt-1.json").write_text("{}", encoding="utf-8")
    with pytest.raises(tp.PilotBlocked, match="marker exists"):
        tp.check_attempt_marker(tmp_path, 2)
    with pytest.raises(tp.PilotBlocked, match="attempt 1"):
        tp.check_attempt_marker(tmp_path / "new", 2)


def groups(plan):
    return [{"scenario_id": s["scenario_id"], "verified": True, "group_complete": True, "failed_teachers": [],
             "leakage_ok": True, "wall_s": 600, "t2_cpu_s": 100.0, "regenerated_identical": i < 2}
            for i, s in enumerate(plan["scenarios"])]


def test_acceptance_gate(plan):
    assert tp.pilot_acceptance(plan, groups(plan))["pass"]
    g = groups(plan)
    g[0]["failed_teachers"], g[0]["group_complete"] = ["T2"], False        # T2 timeout -> group invalid
    r = tp.pilot_acceptance(plan, g)
    assert not r["pass"] and r["incomplete_rate_per_stratum"][plan["scenarios"][0]["label"]] > 0
    g = groups(plan)
    g[7] = {**g[7], "scenario_id": "g-substitute"}
    assert any("substitution" in p for p in tp.pilot_acceptance(plan, g)["problems"])
    g = copy.deepcopy(groups(plan))
    for x in g:
        x["t2_cpu_s"] = 500.0
    assert any("re-plan" in p for p in tp.pilot_acceptance(plan, g)["problems"])
    g = groups(plan)
    g[0]["regenerated_identical"] = False
    assert any("regenerated" in p for p in tp.pilot_acceptance(plan, g)["problems"])
    g = groups(plan)
    g[3]["wall_s"] = 8000
    assert any("wall cap" in p for p in tp.pilot_acceptance(plan, g)["problems"])
