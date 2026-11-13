"""Unit tests for scripts/verify_local.py (the local replacement for GitHub Actions; see AGENTS.md).

These tests never start pytest recursively; they check the verdict, summary parsing and pre-push gate logic.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("verify_local", REPO / "scripts" / "verify_local.py")
vl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vl)

SHA = "a" * 40
OTHER = "b" * 40


def test_parse_summary_counts_quiet_form():
    c = vl.parse_summary(["....", "1376 passed, 10 xfailed in 123.45s (0:02:03)"])
    assert c["passed"] == 1376 and c["xfailed"] == 10 and c["failed"] == 0


def test_parse_summary_counts_failures_errors_and_deselection():
    c = vl.parse_summary(["FAILED tests/x.py::t - assert 0",
                          "2 failed, 5 passed, 1 skipped, 3 deselected, 1 xpassed, 1 error, 4 warnings in 1.0s"])
    assert (c["failed"], c["passed"], c["skipped"], c["deselected"], c["xpassed"], c["errors"], c["warnings"]) \
        == (2, 5, 1, 3, 1, 1, 4)
    c = vl.parse_summary(["===== 7 passed in 0.50s ====="])
    assert c["passed"] == 7


def test_parse_summary_none_without_summary():
    assert vl.parse_summary(["no tests ran in 0.01s"]) is None
    assert vl.parse_summary(["collecting ...", "INTERNALERROR> boom"]) is None


def ok_counts(**kw):
    c = {"passed": 10, "failed": 0, "errors": 0, "skipped": 0, "xfailed": 2, "xpassed": 0, "warnings": 0,
         "deselected": 0}
    c.update(kw)
    return c


def test_verdict_pass_only_for_clean_complete_run():
    assert vl.verdict(0, ok_counts(), True, True) == ("PASS", [])
    assert vl.verdict(1, ok_counts(failed=1), True, True)[0] == "FAIL"
    assert vl.verdict(0, None, True, True)[0] == "FAIL"
    assert vl.verdict(0, ok_counts(deselected=1), True, True)[0] == "FAIL"
    assert vl.verdict(0, ok_counts(passed=0), True, True)[0] == "FAIL"
    assert vl.verdict(0, ok_counts(), True, False)[0] == "FAIL"
    assert vl.verdict(0, ok_counts(), False, True)[0] == "SOURCE_CHANGED"
    assert vl.verdict(2, ok_counts(), True, True)[0] == "FAIL"


def receipt(head=SHA, minor="3.11", **kw):
    r = {"kind": vl.RECEIPT_KIND, "status": "PASS", "head": head, "tree_clean": True, "python_minor": minor,
         "log_verified": True}
    r.update(kw)
    return r


def push(sha=SHA, ref="refs/heads/x"):
    return f"{ref} {sha} {ref} {OTHER}"


def test_push_needs_pass_receipt_for_every_required_python():
    both = [receipt(minor="3.11"), receipt(minor="3.12")]
    assert vl.missing_for_push([push()], both, ["3.11", "3.12"]) == []
    assert vl.missing_for_push([push()], both[:1], ["3.11", "3.12"])
    assert vl.missing_for_push([push(OTHER)], both, ["3.11"])


def test_push_refuses_failed_dirty_or_unverified_receipts():
    for bad in ({"status": "FAIL"}, {"status": "SOURCE_CHANGED"}, {"tree_clean": False},
                {"log_verified": False}, {"kind": "forged"}):
        assert vl.missing_for_push([push()], [receipt(**bad)], ["3.11"]), bad


def test_push_ignores_deletions_and_checks_each_ref():
    assert vl.missing_for_push([push(vl.ZERO_SHA)], [], ["3.11"]) == []
    probs = vl.missing_for_push([push(SHA, "refs/heads/a"), push(OTHER, "refs/heads/b")], [receipt()], ["3.11"])
    assert len(probs) == 1 and "refs/heads/b" in probs[0]


def test_committed_hook_equals_installed_hook():
    hook = (REPO / "scripts" / "git-hooks" / "pre-push").read_bytes().replace(b"\r\n", b"\n").decode("utf-8")
    assert hook == vl.HOOK


def test_run_strips_selection_environment(monkeypatch):
    import os
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k nothing")
    monkeypatch.setenv("PYTEST_PLUGINS", "evil")
    env = vl._env_for(REPO)
    assert "PYTEST_ADDOPTS" not in env and "PYTEST_PLUGINS" not in env
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(REPO / "src")

def test_slow_tier_flag_is_explicit_and_recorded_in_command():
    assert vl.pytest_args() == ["-m", "pytest", "-q"]
    assert vl.pytest_args(with_slow=True) == ["-m", "pytest", "-q", "--run-slow"]
