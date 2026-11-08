"""Shared teacher registry (#10/#25): pinned identities, admitted T4 v1.2.0, blocked T4 records, eligibility."""
from __future__ import annotations

import copy
import json
import subprocess
import sys

import pytest

from ember_v1 import provenance as pv
from ember_v1.teacher import registry as reg

V12 = "sha256:79bd3322b9cf7ccd25928f3a70502089732496af5427f88f00ecd2ff6ba12504"


def test_pins_match_code_and_blocked_and_admission_reports_match_committed_files():
    assert reg.check_pins() == [] and reg.check_registry() == []
    for i in reg.IDENTITIES:
        for rec in (i.block, i.admission):
            if rec:
                assert reg.report_sha256(rec.report) == rec.report_sha256
                assert (reg.REPO / rec.record).is_file()


def test_inventory_keeps_all_five_candidates_admitted_t4_v12_and_both_blocked_t4_identities():
    inv = reg.inventory()
    assert tuple(inv["candidates"]) == reg.CANDIDATES == ("T0", "T1", "T2", "T3", "T4")
    t4 = inv["candidates"]["T4"]
    assert [(x["version"], x["status"], (x.get("block") or {}).get("failed_gate")) for x in t4] == [
        ("1.0.0", "blocked", "G5"), ("1.1.0", "blocked", "G4"), ("1.2.0", "retained", None)]
    assert t4[2]["config_hash"] == V12 and t4[2]["admission"]["namespace"] == "t4-pilot@3"
    assert all(len(inv["candidates"][c]) == 1 and inv["candidates"][c][0]["status"] == "retained"
               for c in ("T0", "T1", "T2", "T3"))


def test_t4_v12_admission_record_matches_the_committed_pilot_report():
    a = reg.admission_record("T4")
    assert a["version"] == "1.2.0" and a["config_hash"] == V12 and a["gate"] == "gate_v2"
    assert a["verdict"] == "T4_READY_FOR_INDEPENDENT_SELECTION_EVALUATION"
    rep = json.loads((reg.REPO / a["report"]).read_text(encoding="utf-8"))
    assert rep["namespace"] == a["namespace"] and rep["gate"]["decision"] == a["verdict"]
    assert rep["environment"]["git_sha"] == a["freeze_commit"] and rep["environment"]["src_dirty"] is False
    assert all(reg.admission_record(c) is None for c in ("T0", "T1", "T2", "T3"))


def test_registry_structure_rules_fail_closed(monkeypatch):
    v12 = reg.identity("T4", "1.2.0")
    no_adm = reg.Identity(**{**v12.__dict__, "admission": None})
    monkeypatch.setattr(reg, "IDENTITIES", tuple(no_adm if i is v12 else i for i in reg.IDENTITIES))
    assert any("without an admission" in e for e in reg.check_registry())
    bad = reg.Identity(**{**v12.__dict__, "admission": reg.Admission(
        **{**v12.admission.__dict__, "report_sha256": "sha256:" + "0" * 64})})
    monkeypatch.setattr(reg, "IDENTITIES", tuple(bad if i is no_adm else i for i in reg.IDENTITIES))
    assert any("hash differs" in e for e in reg.check_registry())
    two = reg.Identity(**{**v12.__dict__, "version": "9.9.9"})
    monkeypatch.setattr(reg, "IDENTITIES", (*reg.IDENTITIES, two))
    assert any("more than one retained" in e for e in reg.check_registry())


def test_retained_classes_include_admitted_t4_v12_and_never_blocked_t4():
    from ember_v1.teacher.t4_v12 import T4RiskPriorityV12
    assert sorted(reg.retained_classes()) == ["T0", "T1", "T2", "T3", "T4"]
    assert reg.retained_classes()["T4"] is T4RiskPriorityV12
    assert reg.teacher_class("T4") is reg.teacher_class("T4", "1.2.0") is T4RiskPriorityV12
    for v in ("1.0.0", "1.1.0"):
        with pytest.raises(reg.RegistryError, match="blocked"):
            reg.teacher_class("T4", v)
    with pytest.raises(reg.RegistryError, match="unknown"):
        reg.teacher_class("T9")


def test_value_teachers_are_one_shared_registry():
    assert pv.VALUE_TEACHERS is reg.VALUE_TEACHERS


def test_registry_import_has_no_teacher_or_provenance_side_effects():
    code = ("import sys; import ember_v1.teacher.registry as r; "
            "print(any(m in sys.modules for m in ('ember_v1.provenance', 'ember_v1.s1', 'ember_v1.teacher.selection')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
    for first in ("ember_v1.provenance", "ember_v1.teacher.registry", "ember_v1.s1"):
        subprocess.run([sys.executable, "-c", f"import {first}; import ember_v1.s1, ember_v1.provenance"], check=True)


def test_default_eligibility_lists_all_five_with_t4_v12():
    e = reg.default_eligibility()
    assert reg.validate_eligibility(e) == e
    assert all(e[c]["status"] == "eligible" for c in reg.CANDIDATES)
    assert e["T4"]["identity"] == reg.identity("T4", "1.2.0").key()
    assert [b["verdict"] for b in reg.blocked_records("T4")] == ["BLOCKED_BY_MECHANISM_CHECK", "T4_V2_BLOCKED"]


T4_OUT = {"status": "excluded", "reason": "excluded_by_protocol_rule", "rule": "test exclusion",
          "blocked_identities": reg.blocked_records("T4")}


def _mut(fn, t4_excluded=False):
    e = copy.deepcopy(reg.default_eligibility())
    if t4_excluded:
        e["T4"] = copy.deepcopy(T4_OUT)
    fn(e)
    return e


def _x(fn):
    """Mutation applied to a T4 that the protocol excludes by rule."""
    def go(e):
        e["T4"] = copy.deepcopy(T4_OUT)
        fn(e)
    return go


BLOCKED_V11 = reg.identity("T4", "1.1.0")
BAD_ELIGIBILITY = {
    "T4 missing": lambda e: e.pop("T4"),
    "extra candidate": lambda e: e.update(T5=e["T0"]),
    "T4 eligible as blocked v1.1.0": lambda e: e.update(T4={"status": "eligible", "identity": BLOCKED_V11.key()}),
    "T4 eligible as blocked v1.0.0": lambda e: e.update(
        T4={"status": "eligible", "identity": reg.identity("T4", "1.0.0").key()}),
    "T4 eligible with wrong v1.2 hash": lambda e: e["T4"]["identity"].update(config_hash="sha256:" + "2" * 64),
    "T4 excluded as blocked although v1.2.0 is retained": lambda e: e.update(T4={
        "status": "excluded", "reason": "blocked_by_preregistered_gate",
        "blocked_identities": reg.blocked_records("T4")}),
    "T4 excluded without reason": _x(lambda e: e["T4"].pop("reason")),
    "T4 excluded with empty reason": _x(lambda e: e["T4"].update(reason="")),
    "T4 excluded, unknown reason": _x(lambda e: e["T4"].update(reason="slow")),
    "T4 missing one blocked identity": _x(lambda e: e["T4"]["blocked_identities"].pop()),
    "T4 wrong verdict": _x(lambda e: e["T4"]["blocked_identities"][1].update(verdict="PASSED")),
    "T4 wrong gate": _x(lambda e: e["T4"]["blocked_identities"][0].update(failed_gate="G4")),
    "T4 wrong report hash": _x(lambda e: e["T4"]["blocked_identities"][1].update(report_sha256="sha256:" + "0" * 64)),
    "T4 status ambiguous": _x(lambda e: e["T4"].update(status="blocked")),
    "T4 rule-exclusion without rule": _x(lambda e: e["T4"].pop("rule")),
    "T4 rule-exclusion with blank rule": _x(lambda e: e["T4"].update(rule=" ")),
    "T0 blocked without blocked record": lambda e: e.update(
        T0={"status": "excluded", "reason": "blocked_by_preregistered_gate", "blocked_identities": []}),
    "T1 eligible with wrong hash": lambda e: e["T1"]["identity"].update(config_hash="sha256:" + "1" * 64),
    "T2 eligible with extra field": lambda e: e["T2"].update(note="x"),
    "nothing eligible": _x(lambda e: [e.update({c: {"status": "excluded", "reason": "excluded_by_protocol_rule",
                                                    "rule": "r", "blocked_identities": []}})
                                      for c in ("T0", "T1", "T2", "T3")]),
}


@pytest.mark.parametrize("case", sorted(BAD_ELIGIBILITY))
def test_wrong_or_ambiguous_eligibility_fails_closed(case):
    with pytest.raises(reg.RegistryError):
        reg.validate_eligibility(_mut(BAD_ELIGIBILITY[case]))


def test_rule_exclusion_of_a_retained_candidate_needs_a_written_rule():
    ok = _mut(lambda e: e.update(T3={"status": "excluded", "reason": "excluded_by_protocol_rule",
                                     "rule": "resource gate failed before confirmation", "blocked_identities": []}))
    assert reg.validate_eligibility(ok)["T3"]["status"] == "excluded"
    assert reg.validate_eligibility(_mut(lambda e: None, t4_excluded=True))["T4"]["status"] == "excluded"
