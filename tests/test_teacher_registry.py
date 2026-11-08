"""Shared teacher registry (#10/#25): pinned identities, blocked T4 records and fail-closed eligibility."""
from __future__ import annotations

import copy
import subprocess
import sys

import pytest

from ember_v1 import provenance as pv
from ember_v1.teacher import registry as reg


def test_pins_match_code_and_blocked_reports_match_committed_files():
    assert reg.check_pins() == []
    for i in reg.IDENTITIES:
        if i.block:
            assert reg.report_sha256(i.block.report) == i.block.report_sha256
            assert (reg.REPO / i.block.record).is_file()


def test_inventory_keeps_all_five_candidates_and_both_blocked_t4_identities():
    inv = reg.inventory()
    assert tuple(inv["candidates"]) == reg.CANDIDATES == ("T0", "T1", "T2", "T3", "T4")
    t4 = inv["candidates"]["T4"]
    assert [(x["version"], x["status"], x["block"]["failed_gate"]) for x in t4] == [
        ("1.0.0", "blocked", "G5"), ("1.1.0", "blocked", "G4")]
    assert all(len(inv["candidates"][c]) == 1 and inv["candidates"][c][0]["status"] == "retained"
               for c in ("T0", "T1", "T2", "T3"))


def test_retained_classes_never_include_blocked_t4():
    assert sorted(reg.retained_classes()) == ["T0", "T1", "T2", "T3"]
    for v in (None, "1.0.0", "1.1.0"):
        with pytest.raises(reg.RegistryError, match="blocked|no single retained"):
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


def test_default_eligibility_excludes_t4_with_its_exact_blocked_records():
    e = reg.default_eligibility()
    assert reg.validate_eligibility(e) == e
    assert e["T4"]["status"] == "excluded" and e["T4"]["reason"] == "blocked_by_preregistered_gate"
    assert [b["verdict"] for b in e["T4"]["blocked_identities"]] == ["BLOCKED_BY_MECHANISM_CHECK", "T4_V2_BLOCKED"]
    assert all(e[c]["status"] == "eligible" for c in ("T0", "T1", "T2", "T3"))


def _mut(fn):
    e = copy.deepcopy(reg.default_eligibility())
    fn(e)
    return e


BLOCKED_V11 = reg.identity("T4", "1.1.0")
BAD_ELIGIBILITY = {
    "T4 missing": lambda e: e.pop("T4"),
    "extra candidate": lambda e: e.update(T5=e["T0"]),
    "T4 eligible as blocked v1.1.0": lambda e: e.update(T4={"status": "eligible", "identity": BLOCKED_V11.key()}),
    "T4 eligible as blocked v1.0.0": lambda e: e.update(
        T4={"status": "eligible", "identity": reg.identity("T4", "1.0.0").key()}),
    "T4 excluded without reason": lambda e: e["T4"].pop("reason"),
    "T4 excluded with empty reason": lambda e: e["T4"].update(reason=""),
    "T4 excluded, unknown reason": lambda e: e["T4"].update(reason="slow"),
    "T4 missing one blocked identity": lambda e: e["T4"]["blocked_identities"].pop(),
    "T4 wrong verdict": lambda e: e["T4"]["blocked_identities"][1].update(verdict="PASSED"),
    "T4 wrong gate": lambda e: e["T4"]["blocked_identities"][0].update(failed_gate="G4"),
    "T4 wrong report hash": lambda e: e["T4"]["blocked_identities"][1].update(report_sha256="sha256:" + "0" * 64),
    "T4 status ambiguous": lambda e: e["T4"].update(status="blocked"),
    "T4 rule-exclusion without rule": lambda e: e["T4"].update(reason="excluded_by_protocol_rule"),
    "T0 blocked without blocked record": lambda e: e.update(
        T0={"status": "excluded", "reason": "blocked_by_preregistered_gate", "blocked_identities": []}),
    "T1 eligible with wrong hash": lambda e: e["T1"]["identity"].update(config_hash="sha256:" + "1" * 64),
    "T2 eligible with extra field": lambda e: e["T2"].update(note="x"),
    "nothing eligible": lambda e: [e.update({c: {"status": "excluded", "reason": "excluded_by_protocol_rule",
                                               "rule": "r", "blocked_identities": []}})
                                   for c in ("T0", "T1", "T2", "T3")],
}


@pytest.mark.parametrize("case", sorted(BAD_ELIGIBILITY))
def test_wrong_or_ambiguous_eligibility_fails_closed(case):
    with pytest.raises(reg.RegistryError):
        reg.validate_eligibility(_mut(BAD_ELIGIBILITY[case]))


def test_rule_exclusion_of_a_retained_candidate_needs_a_written_rule():
    ok = _mut(lambda e: e.update(T3={"status": "excluded", "reason": "excluded_by_protocol_rule",
                                     "rule": "resource gate failed before confirmation", "blocked_identities": []}))
    assert reg.validate_eligibility(ok)["T3"]["status"] == "excluded"
