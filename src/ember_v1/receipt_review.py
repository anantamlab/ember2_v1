"""Review support for the p1-09 teacher-selection receipt (issue #25 -> #10). Read-only by default.

`verify_pending` checks every receipt binding that a machine can check, from the committed evidence:

1 the frozen protocol, loaded from the trusted git history (commit `64b2d98`);
2 manifest, isolation audit, attempt marker and evidence hashes;
3 the frozen `selection.selection_gate`, re-evaluated on the committed evidence (no teacher runs), equals the
  committed `outcome/selection_gate.json`, and its `receipt_draft` equals the committed draft;
4 the receipt selects exactly T1 v1.0.0 with the frozen config hash and the code fingerprint of the current code,
  and that fingerprint equals the one the preflight recorded before the run;
5 the confirmation lists equal the confirmation manifest;
6 `s1.verify_receipt` fails ONLY on the review check (it is the last check, so every earlier check passed).

`approve` writes an approved copy. Only a human reviewer runs it, with their own names. It never runs in CI, and
no code in this repository calls it with real data. A reviewer name is a recorded claim, not a verified
identity (docs/P1_TEACHER_PROVENANCE.md, "Trust boundary").
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import provenance as pv
from . import s1, trust
from .sim.canonical import canonical_json, digest
from .teacher import registry as reg

REVIEW_VERSION = "receipt-review@1.0.0"
NAMESPACE_DIR = "data/selection/p1-09-selection@1"
PROTOCOL_COMMIT = "64b2d9873953ba750fd6c08fb84761f1bcd8e0bb"
PROTOCOL_REF = {"kind": "git", "commit": PROTOCOL_COMMIT, "path": f"{NAMESPACE_DIR}/protocol.json"}
EXPECTED = {
    "protocol_hash": "sha256:ae72c2cf1a0dae4f46b3a6d158f426af0853bc4100cd15706c74f203a6765c21",
    "manifest_hash": "sha256:4e9283b206296d7251514140613f0f8cc234fe9e2dc68877f7a71e906dc5cf9e",
    "audit_hash": "sha256:802b8fa509767c3b3cc881289a81c9285ae254498aa0a1de255fb0e035ea2ef2",
    "attempt_hash": "sha256:f83212669e18f11c4687d3f94be43dca4c2d33faec23e10caf884cca24756ea7",
    "evidence_hash": "sha256:524e17c2d3b7d560c9556166f06068160f87486fbc6b5e77e213dd681d9478a7",
    "draft_receipt_hash": "sha256:d6e94bd4a43631a5138cc110b3b04d75f94a150ce15fe07b89094443918d4022",
    "decision": "RETAIN_T1_NO_IMPROVEMENT",
    "selected": {"name": "T1", "policy": "congestion_aware_cooperative_planner", "version": "1.0.0",
                 "config_hash": "sha256:5047e942db3326706b6184c9f29e0b992078920f03c8a53adc8cc24258345361",
                 "code_fingerprint": "sha256:52dc61f4ff9599e55bfd29f47068bf301895321ce85f6c9aaa9feec1dc568eac"},
}
REVIEW_MESSAGE = "receipt is not reviewed (review.status approved with named reviewers)"
# Names that are obviously not a person: refuse them so a placeholder cannot pass as a review.
_PLACEHOLDER = re.compile(r"(?i)^(|reviewer[-_ ]?\w*|tbd|todo|n/?a|none|name|claude|ai|bot|test|x+|\?+|<.*>)$")


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _self_hash(doc: Mapping[str, Any], key: str) -> bool:
    return digest({k: v for k, v in doc.items() if k != key}) == doc.get(key)


def verify_pending(repo: str | Path, *, receipt_path: str | Path | None = None,
                   trusted_ref: str = "origin/main", regate: bool = True) -> dict[str, Any]:
    """Every machine-checkable receipt binding. `status` is `VERIFIED_PENDING_REVIEW` only if all pass."""
    repo = Path(repo)
    d = repo / NAMESPACE_DIR
    rpath = Path(receipt_path) if receipt_path else d / "receipt.json"
    checks: dict[str, bool] = {}
    notes: dict[str, Any] = {}

    protocol, p_trust = trust.load_trusted_json(repo, PROTOCOL_REF, trusted_ref=trusted_ref)
    checks["protocol_trusted_and_hash"] = (s1.protocol_hash(protocol) == protocol["protocol_hash"]
                                           == EXPECTED["protocol_hash"])
    checks["protocol_worktree_equals_trusted"] = trust.sha256_lf((d / "protocol.json").read_bytes()) == p_trust["sha256"]
    manifest = _load(d / "manifest.json")
    checks["manifest_hash"] = (_self_hash(manifest, "manifest_hash") and manifest["manifest_hash"]
                               == protocol["confirmation"]["manifest_hash"] == EXPECTED["manifest_hash"])
    audit = _load(d / "isolation_audit.json")
    checks["isolation_audit"] = (_self_hash(audit, "audit_hash") and audit["audit_hash"] == EXPECTED["audit_hash"]
                                 and audit["pass"] is True and audit["manifest_hash"] == manifest["manifest_hash"])
    attempt = _load(d / "attempt.json")
    checks["attempt_marker"] = (_self_hash(attempt, "attempt_hash") and attempt["attempt_hash"] == EXPECTED["attempt_hash"]
                                and attempt["protocol_hash"] == protocol["protocol_hash"]
                                and attempt["manifest_hash"] == manifest["manifest_hash"])
    evidence = _load(d / "evidence.json")
    checks["evidence_hash_and_complete"] = (_self_hash(evidence, "evidence_hash")
                                            and evidence["evidence_hash"] == EXPECTED["evidence_hash"]
                                            and evidence["complete"] is True and not evidence["errors"])
    gate_committed = _load(d / "outcome/selection_gate.json")
    draft = _load(d / "outcome/receipt_DRAFT_PENDING_REVIEW.json")
    checks["committed_gate_decision"] = (gate_committed["decision"] == EXPECTED["decision"]
                                         and not gate_committed["reasons"] and gate_committed["selected"] == "T1")
    checks["draft_equals_committed_gate_draft"] = canonical_json(draft) == canonical_json(gate_committed["receipt_draft"])
    if regate:
        from .teacher import confirmation as C
        from .teacher import selection as sel
        historical, _, _ = C.historical_sets()
        gate = sel.selection_gate(protocol, evidence, protocol_ref=PROTOCOL_REF, repo=repo, historical=historical,
                                  trusted_ref=trusted_ref)
        checks["gate_reproduces_identically"] = canonical_json(gate) == canonical_json(gate_committed)
    else:
        notes["gate_reproduces_identically"] = "skipped (regate=False)"

    receipt = _load(rpath)
    checks["receipt_hash"] = s1.receipt_hash(receipt) == receipt.get("receipt_hash")
    body = {k: v for k, v in receipt.items() if k not in ("review", "receipt_hash")}
    dbody = {k: v for k, v in draft.items() if k not in ("review", "receipt_hash")}
    checks["receipt_equals_draft_except_review"] = canonical_json(body) == canonical_json(dbody)
    checks["receipt_draft_hash"] = draft["receipt_hash"] == EXPECTED["draft_receipt_hash"] and _self_hash(draft,
                                                                                                         "receipt_hash")
    sel_ = receipt.get("selected", {})
    cls = reg.teacher_class("T1", "1.0.0")
    pre = _load(d / "outcome/preflight_post_run.json")["teachers"]["T1"]
    checks["selects_exactly_T1_v1_0_0"] = (receipt.get("decision") == EXPECTED["decision"]
                                           and dict(sel_) == EXPECTED["selected"])
    checks["fingerprint_equals_current_code"] = pv.code_fingerprint(cls) == EXPECTED["selected"]["code_fingerprint"]
    checks["fingerprint_equals_preflight"] = (pre["code_fingerprint"], pre["config_hash"], pre["version"]) == (
        sel_.get("code_fingerprint"), sel_.get("config_hash"), sel_.get("version"))
    checks["config_equals_registry_pin"] = cls().identity.config_hash == EXPECTED["selected"]["config_hash"]
    conf = receipt.get("confirmation", {})
    sc = manifest["scenarios"]
    checks["confirmation_lists_equal_manifest"] = (
        conf.get("namespace") == manifest["plan_id"] and conf.get("manifest_hash") == manifest["manifest_hash"]
        and conf.get("evidence_hash") == evidence["evidence_hash"]
        and conf.get("family_ids") == sorted({s["family_id"] for s in sc})
        and conf.get("lineage_ids") == sorted({s["lineage_id"] for s in sc})
        and conf.get("scenario_hashes") == sorted({s["scenario_hash"] for s in sc}))
    notes["confirmation_counts"] = {k: len(conf.get(k, [])) for k in ("family_ids", "lineage_ids", "scenario_hashes")}
    try:
        s1.verify_receipt(receipt, protocol, PROTOCOL_REF)
        verify_error = None
    except ValueError as e:
        verify_error = str(e)
    review = receipt.get("review", {})
    pending = review == {"status": "pending", "reviewers": []}
    checks["s1_refuses_only_for_review"] = verify_error == REVIEW_MESSAGE if pending else verify_error is None
    notes["s1_verify_receipt"] = verify_error or "verified"
    ok = all(checks.values())
    status = ("VERIFIED_PENDING_REVIEW" if pending else "VERIFIED_APPROVED") if ok else "FAILED"
    return {"review_version": REVIEW_VERSION, "status": status, "review": review, "checks": checks, "notes": notes,
            "receipt_path": rpath.relative_to(repo).as_posix() if rpath.is_relative_to(repo) else str(rpath),
            "receipt_hash": receipt.get("receipt_hash"), "protocol_trust": p_trust,
            "bulk_s1": "BULK_S1_BLOCKED" if pending or not ok else "RECEIPT_GATE_ONLY_PASSED"}


def approve(receipt: Mapping[str, Any], reviewers: Sequence[str]) -> dict[str, Any]:
    """Approved copy of a pending receipt, rehashed. For a human reviewer only, with their own names."""
    names = [r.strip() for r in reviewers]
    if not names or any(_PLACEHOLDER.match(n) for n in names) or len(set(names)) != len(names):
        raise ValueError("approve needs distinct real reviewer names (placeholders are refused)")
    if receipt.get("review") != {"status": "pending", "reviewers": []}:
        raise ValueError("only a pending receipt can be approved")
    if s1.receipt_hash(receipt) != receipt.get("receipt_hash"):
        raise ValueError("receipt hash does not match its content")
    body = {k: v for k, v in receipt.items() if k != "receipt_hash"}
    body["review"] = {"status": "approved", "reviewers": names}
    return {**body, "receipt_hash": s1.receipt_hash(body)}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ember_v1.receipt_review")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="read-only verification of the p1-09 receipt")
    v.add_argument("--repo", default=".")
    v.add_argument("--receipt", default=None)
    v.add_argument("--trusted-ref", default="origin/main")
    v.add_argument("--out", default=None, help="write the report here (JSON)")
    a = sub.add_parser("approve", help="HUMAN REVIEWER ONLY: write an approved copy with your own name(s)")
    a.add_argument("--receipt", required=True)
    a.add_argument("--reviewer", action="append", required=True)
    a.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "verify":
        rep = verify_pending(args.repo, receipt_path=args.receipt, trusted_ref=args.trusted_ref)
        text = json.dumps(rep, indent=1, sort_keys=True) + "\n"
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8", newline="\n")
        sys.stdout.write(text)
        return 0 if rep["status"] != "FAILED" else 1
    out = approve(_load(Path(args.receipt)), args.reviewer)
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(out["receipt_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
