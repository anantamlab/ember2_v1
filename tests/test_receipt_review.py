"""p1-09 selection receipt review support: read-only verification and the human-only approve step.

Every test that approves a receipt works on a temporary copy. The committed receipt stays pending.
"""
from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest

from ember_v1 import receipt_review as rr
from ember_v1 import s1

REPO = Path(__file__).resolve().parents[1]
NS = REPO / rr.NAMESPACE_DIR


def _have_history() -> bool:
    r = subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", rr.PROTOCOL_COMMIT, "origin/main"],
                       capture_output=True)
    return r.returncode == 0


needs_history = pytest.mark.skipif(not _have_history(), reason="needs git history with origin/main (fetch-depth 0)")


def _load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def test_committed_receipt_is_the_pending_draft():
    receipt, draft = _load(NS / "receipt.json"), _load(NS / "outcome/receipt_DRAFT_PENDING_REVIEW.json")
    assert receipt == draft
    assert receipt["review"] == {"status": "pending", "reviewers": []}
    assert receipt["receipt_hash"] == rr.EXPECTED["draft_receipt_hash"] == s1.receipt_hash(receipt)
    assert receipt["decision"] == "RETAIN_T1_NO_IMPROVEMENT" and receipt["selected"] == rr.EXPECTED["selected"]


@needs_history
def test_full_verification_passes_and_stays_pending():
    rep = rr.verify_pending(REPO, regate=True)
    assert rep["status"] == "VERIFIED_PENDING_REVIEW" and all(rep["checks"].values()), rep["checks"]
    assert rep["bulk_s1"] == "BULK_S1_BLOCKED"
    assert rep["notes"]["s1_verify_receipt"] == rr.REVIEW_MESSAGE
    committed = _load(NS / "review/receipt_verification.json")
    assert committed["status"] == rep["status"] and committed["checks"] == rep["checks"]
    assert committed["receipt_hash"] == rep["receipt_hash"] == rr.EXPECTED["draft_receipt_hash"]


@pytest.mark.parametrize("names", [[], [""], ["  "], ["reviewer-a"], ["TBD"], ["Claude"], ["<name>"],
                                   ["A Person", "A Person"]])
def test_approve_refuses_placeholder_or_missing_names(names):
    with pytest.raises(ValueError, match="real reviewer names"):
        rr.approve(_load(NS / "receipt.json"), names)


def test_approve_refuses_non_pending_or_tampered_receipts():
    r = _load(NS / "receipt.json")
    bad = copy.deepcopy(r)
    bad["decision"] = "SELECT_STANDALONE"
    with pytest.raises(ValueError, match="hash"):
        rr.approve(bad, ["A Person"])
    ok = rr.approve(r, ["A Person"])
    with pytest.raises(ValueError, match="pending"):
        rr.approve(ok, ["B Person"])


@needs_history
def test_approved_copy_verifies_only_in_a_temporary_file(tmp_path):
    out = tmp_path / "receipt.json"
    approved = rr.approve(_load(NS / "receipt.json"), ["A Person"])
    out.write_text(json.dumps(approved), encoding="utf-8")
    rep = rr.verify_pending(REPO, receipt_path=out, regate=False)
    assert rep["status"] == "VERIFIED_APPROVED" and rep["notes"]["s1_verify_receipt"] == "verified"
    assert approved["receipt_hash"] != rr.EXPECTED["draft_receipt_hash"]
    assert _load(NS / "receipt.json")["review"]["status"] == "pending"       # the committed file is unchanged


def _rehashed(mut):
    r = copy.deepcopy(_load(NS / "receipt.json"))
    mut(r)
    r["receipt_hash"] = s1.receipt_hash(r)
    return r


@needs_history
@pytest.mark.parametrize("mut, failed", [
    (lambda r: r["selected"].update(name="T2"), "selects_exactly_T1_v1_0_0"),
    (lambda r: r["selected"].update(code_fingerprint="sha256:" + "0" * 64), "fingerprint_equals_preflight"),
    (lambda r: r["selected"].update(version="1.0.1"), "selects_exactly_T1_v1_0_0"),
    (lambda r: r["confirmation"]["family_ids"].pop(), "confirmation_lists_equal_manifest"),
    (lambda r: r["confirmation"].update(evidence_hash="sha256:" + "1" * 64), "receipt_equals_draft_except_review"),
    (lambda r: r.update(decision="SELECT_STANDALONE"), "selects_exactly_T1_v1_0_0"),
])
def test_tampered_receipts_fail(tmp_path, mut, failed):
    out = tmp_path / "receipt.json"
    out.write_text(json.dumps(_rehashed(mut)), encoding="utf-8")
    rep = rr.verify_pending(REPO, receipt_path=out, regate=False)
    assert rep["status"] == "FAILED" and rep["checks"][failed] is False
    assert rep["bulk_s1"] == "BULK_S1_BLOCKED"


@needs_history
def test_unrehashed_edit_fails(tmp_path):
    r = _load(NS / "receipt.json")
    r["review"] = {"status": "approved", "reviewers": ["A Person"]}
    out = tmp_path / "receipt.json"
    out.write_text(json.dumps(r), encoding="utf-8")
    rep = rr.verify_pending(REPO, receipt_path=out, regate=False)
    assert rep["status"] == "FAILED" and rep["checks"]["receipt_hash"] is False
