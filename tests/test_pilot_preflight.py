"""scripts/pilot_preflight.py: read-only, fail-closed status for attempt 1 of p1-12-train-pilot@1."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pilot_preflight", REPO / "scripts" / "pilot_preflight.py")
pf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pf)


def _has(commit: str) -> bool:
    return subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{commit}^{{commit}}"],
                          capture_output=True).returncode == 0


needs_history = pytest.mark.skipif(not (_has(pf.FREEZE_COMMIT) and _has(pf.EXECUTION_COMMIT)),
                                   reason="needs git history with the freeze and execution commits")
ABSENT = ["authorization: data/collection/p1-12-train-pilot@1/authorization.json does not exist at commit 1a0128ca5711",
          "authorization: no multi-teacher TRAIN pilot authorization"]


def test_absent_authorization_is_pending():
    assert pf.classify(ABSENT, []) == pf.PENDING


def test_unapproved_review_reason_is_pending():
    reasons = ["approval: no current APPROVED review by 'Sathvikar01' (allowlisted, not the PR author 'sankalphs')",
               "verified approvers: []"]
    assert pf.classify(reasons, []) == pf.PENDING


@pytest.mark.parametrize("extra", ["working tree not clean: ['ignored .local-test-logs/']",
                                   "HEAD is not the authorized execution commit",
                                   "live rebuild differs from the frozen plan (code, pins, data or audit drift): []",
                                   "attempt evidence exists (attempt-1.json); a new attempt needs a new authorization",
                                   "teacher identity drift (registry pins)", "something new"])
def test_any_non_authorization_reason_blocks(extra):
    assert pf.classify([*ABSENT, extra], []) == pf.BLOCKED


def test_own_failed_check_blocks_even_without_runner_reasons():
    assert pf.classify([], ["machine"]) == pf.BLOCKED
    assert pf.classify(ABSENT, ["attempt_1_unused"]) == pf.BLOCKED


def test_no_reason_and_no_failure_is_ready():
    assert pf.classify([], []) == pf.READY


def test_machine_thresholds(tmp_path):
    ok = pf.machine_check(tmp_path, mem={"total": 32.0, "available": 16.0}, cpus=8, python=(3, 12))
    assert ok["pass"] and ok["requirements"]["source"] == "this preflight, not the frozen plan"
    low = pf.machine_check(tmp_path, mem={"total": 8.0, "available": 1.0}, cpus=4, python=(3, 13))
    assert not low["pass"] and len(low["problems"]) == 3
    unknown = pf.machine_check(tmp_path, mem={"total": None, "available": None}, cpus=8, python=(3, 11))
    assert not unknown["pass"]


def test_memory_probe_reports_numbers_or_none():
    m = pf.memory_gib()
    assert set(m) == {"total", "available"}
    assert all(v is None or v > 0 for v in m.values())


def test_attempt_artifacts_found(tmp_path):
    assert pf.attempt_artifacts(tmp_path) == []
    d = tmp_path / pf.OUT_DIR
    d.mkdir(parents=True)
    (d / "plan.json").write_text("{}", encoding="utf-8")
    assert pf.attempt_artifacts(tmp_path) == []
    (d / "attempt-1").mkdir()
    (d / "attempt-2.json").write_text("{}", encoding="utf-8")
    assert [Path(p).name for p in pf.attempt_artifacts(tmp_path)] == ["attempt-1", "attempt-2.json"]


def test_out_inside_checkout_refused(tmp_path):
    with pytest.raises(SystemExit, match="outside the execution checkout"):
        pf.main(["--repo", str(tmp_path), "--out", str(tmp_path / "x" / "preflight.json")])


def test_pinned_constants_match_the_frozen_plan():
    plan = json.loads((REPO / pf.PLAN_PATH).read_text(encoding="utf-8"))
    assert plan["status"] == "FROZEN" and plan["plan_hash"] == pf.PLAN_HASH
    assert plan["attempt"] == 1 and len(plan["scenarios"]) == 8


@needs_history
def test_commit_constants_are_related_as_the_runbook_says():
    anc = subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", pf.FREEZE_COMMIT,
                          pf.EXECUTION_COMMIT], capture_output=True).returncode
    assert anc == 0
    trees = {subprocess.run(["git", "-C", str(REPO), "rev-parse", f"{c}^{{tree}}"], capture_output=True,
                            text=True).stdout.strip() for c in (pf.FREEZE_COMMIT, pf.EXECUTION_COMMIT)}
    assert len(trees) == 1          # the execution commit holds exactly the reviewed freeze tree


@needs_history
def test_end_to_end_on_this_checkout_is_read_only_and_blocked(tmp_path):
    """A test checkout is never a ready execution checkout (it adds this file, or is dirty while editing),
    so the status is BLOCKED; nothing is written to it."""
    before = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=all"],
                            capture_output=True, text=True).stdout
    out = tmp_path / "preflight.json"
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "pilot_preflight.py"), "--repo", str(REPO),
                        "--trusted-ref", "HEAD", "--out", str(out)], capture_output=True, text=True, timeout=900)
    after = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=all"],
                           capture_output=True, text=True).stdout
    assert before == after
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert r.returncode == 1 and rep["status"] == pf.BLOCKED and rep["teacher_calls"] == 0
    # The pilot ran (attempt 1 consumed). P1-17 then changed only collector code (pilot_supervisor 1.1.0, the
    # multiteacher scope list), so the historical pilot plan's live rebuild differs in `identities` alone; the
    # scenarios and every teacher identity must still match.
    assert rep["checks"]["plan_hash"]["pass"]
    lr = rep["checks"]["live_rebuild"]
    assert lr["pass"] or lr["differs"] == ["identities"], lr["differs"]
    assert rep["checks"]["scenario_fingerprints"]["pass"] and rep["checks"]["teacher_fingerprints"]["pass"]
    assert rep["checks"]["scenarios_regenerate"]["pass"] and rep["checks"]["teacher_fingerprints"]["pass"]
    gi = rep["checks"]["git_identity"]
    assert gi["pass"] == (gi["head"] == pf.EXECUTION_COMMIT) and rep["failed_checks"]
