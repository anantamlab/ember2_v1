"""Mandatory bulk-S1 training entrypoint (issue #10). Fail closed; no opt-out.

`build_s1_training_set` is the only function that returns S1 training
records. It returns an `S1TrainingSet`, which code outside this module cannot
construct, so a future S1 trainer that accepts only `S1TrainingSet` cannot be
fed legacy replay-only records (`sources.training_records`).

It does every check itself, from its own inputs; it never trusts a
caller-supplied "verified" flag:

1 Trust root. The frozen #25 protocol and the reviewed selection receipt are
  read from the git history of the repository (`git show <commit>:<path>`),
  and the commit must be an ancestor of the trusted ref (default
  `origin/main`, whose changes pass pull-request review). Any other trust
  root kind is unknown and fails.
2 Protocol: frozen, hash-consistent, and its eligibility record valid against
  `teacher.registry` (all five candidates; T4 excluded only with its exact
  blocked records).
3 Receipt: hash-consistent, bound to that protocol and its commit, reviewed,
  selecting one eligible retained identity whose config hash and code
  fingerprint equal the current code. An arbiter receipt is refused: there is
  no arbiter provenance yet.
4 Manifests: train, dev and calibration manifests of one split plan.
5 Every source is ingested with full-episode replay AND provenance
  (`sources.ingest_zip(..., manifests=...)`); every record is from the
  selected identity and the train split.
6 Regeneration spot check: max(`MIN_SPOT_CHECKS`, 2 %) decisions, sample seed
  fixed by the receipt and source hashes.
7 Leakage: duplicates, cross-split leaks, and family/lineage/scenario overlap
  with dev, calibration and the receipt's selection-confirmation set.
8 `provenance.bulk_s1_gate` on the computed inputs.

Trust boundary: hashes and git ancestry are not signatures. They protect
against accidental and inconsistent edits by trusted internal contributors.
They do not authenticate a party that can write to the trusted ref or the
local repository (docs/P1_TEACHER_PROVENANCE.md, "Trust boundary").
"""
from __future__ import annotations

import math
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import provenance as pv
from .sim.canonical import digest
from .sources import ingest_zip
from .teacher import registry as reg
from .trust import TRUST_ROOT_KINDS, check_manifest, load_trusted_json

S1_GATE_VERSION = "s1-gate@1.0.0"
RECEIPT_VERSION = "teacher-selection-receipt@1.0.0"
MIN_SPOT_CHECKS = 50
SPOT_FRACTION = 0.02
REQUIRED_MANIFESTS = ("train", "dev", "calibration")
RECEIPT_DECISIONS = ("SELECT_STANDALONE", "RETAIN_T1_NO_IMPROVEMENT", "SELECT_ARBITER")
RECEIPT_KEYS = {"receipt_version", "protocol_id", "protocol_hash", "protocol_ref", "decision", "selected",
                "eligibility", "confirmation", "review", "receipt_hash"}
CONFIRMATION_KEYS = {"namespace", "manifest_hash", "evidence_hash", "family_ids", "lineage_ids", "scenario_hashes"}
_AUTH = object()


class S1GateError(RuntimeError):
    """Bulk S1 is blocked. `report` holds every reason."""

    def __init__(self, report: Mapping[str, Any]):
        super().__init__("BULK_S1_BLOCKED: " + "; ".join(report.get("reasons", [])))
        self.report = dict(report)


@dataclass(frozen=True)
class S1TrainingSet:
    """Records authorized by `build_s1_training_set`. Not constructible elsewhere."""
    records: tuple[Mapping[str, Any], ...]
    report: Mapping[str, Any]
    _auth: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._auth is not _AUTH:
            raise S1GateError({"reasons": ["S1TrainingSet can only come from build_s1_training_set"]})


# ----- protocol and receipt -------------------------------------------------------------------------------
def protocol_hash(protocol: Mapping[str, Any]) -> str:
    return digest({k: v for k, v in protocol.items() if k != "protocol_hash"})


def verify_protocol(protocol: Any) -> dict[str, Any]:
    """Minimum #10 checks of a frozen #25 protocol. Raise ValueError (fail closed)."""
    if not isinstance(protocol, Mapping):
        raise ValueError("protocol must be a mapping")
    for k in ("protocol_id", "eligibility", "frozen", "protocol_hash"):
        if k not in protocol:
            raise ValueError(f"protocol misses {k}")
    if protocol["frozen"] is not True:
        raise ValueError("protocol is not frozen")
    if protocol_hash(protocol) != protocol["protocol_hash"]:
        raise ValueError("protocol hash does not match its content")
    try:
        reg.validate_eligibility(protocol["eligibility"])
    except reg.RegistryError as e:
        raise ValueError(f"protocol eligibility invalid: {e}") from None
    from .teacher import selection          # lazy: selection builds receipts with this module
    try:
        selection.verify_frozen_protocol(protocol)      # the full #25 rules: one validator for both gates
    except selection.ProtocolError as e:
        raise ValueError(f"selection protocol invalid: {e}") from None
    return {"protocol_id": protocol["protocol_id"], "protocol_hash": protocol["protocol_hash"]}


def receipt_hash(receipt: Mapping[str, Any]) -> str:
    return digest({k: v for k, v in receipt.items() if k != "receipt_hash"})


def verify_receipt(receipt: Any, protocol: Mapping[str, Any], protocol_ref: Mapping[str, Any]) -> dict[str, Any]:
    """Verify a reviewed #25 selection receipt against its frozen protocol. Raise ValueError (fail closed)."""
    if not isinstance(receipt, Mapping) or set(receipt) != RECEIPT_KEYS:
        raise ValueError("receipt has missing or extra fields")
    if receipt["receipt_version"] != RECEIPT_VERSION:
        raise ValueError("unsupported receipt version")
    if receipt_hash(receipt) != receipt["receipt_hash"]:
        raise ValueError("receipt hash does not match its content")
    verify_protocol(protocol)
    if (receipt["protocol_id"], receipt["protocol_hash"]) != (protocol["protocol_id"], protocol["protocol_hash"]):
        raise ValueError("receipt is bound to another protocol")
    if dict(receipt["protocol_ref"]) != dict(protocol_ref):
        raise ValueError("receipt is bound to another protocol commit or path")
    if receipt["eligibility"] != protocol["eligibility"]:
        raise ValueError("receipt eligibility differs from the frozen protocol")
    if receipt["decision"] not in RECEIPT_DECISIONS:
        raise ValueError("unknown receipt decision")
    if receipt["decision"] == "SELECT_ARBITER":
        raise ValueError("an arbiter receipt has no S1 provenance support; S1 needs one standalone teacher")
    sel = receipt["selected"]
    if not isinstance(sel, Mapping) or set(sel) != {"name", "policy", "version", "config_hash", "code_fingerprint"}:
        raise ValueError("selected identity needs name, policy, version, config_hash and code_fingerprint")
    if receipt["decision"] == "RETAIN_T1_NO_IMPROVEMENT" and sel["name"] != "T1":
        raise ValueError("RETAIN_T1_NO_IMPROVEMENT must select T1")
    key = {k: sel[k] for k in ("name", "policy", "version", "config_hash")}
    cand = next((c for c, e in protocol["eligibility"].items()
                 if e.get("status") == reg.ELIGIBLE and dict(e["identity"]) == key), None)
    if cand is None:
        raise ValueError(f"selected {key} is not an eligible identity of the protocol")
    try:
        cls = reg.teacher_class(sel["name"], sel["version"])
    except reg.RegistryError as e:
        raise ValueError(str(e)) from None
    if pv.code_fingerprint(cls) != sel["code_fingerprint"]:
        raise ValueError("selected code fingerprint differs from the current frozen code")
    conf = receipt["confirmation"]
    if not isinstance(conf, Mapping) or set(conf) != CONFIRMATION_KEYS or not all(
            isinstance(conf[k], list) and conf[k] for k in ("family_ids", "lineage_ids", "scenario_hashes")):
        raise ValueError("receipt confirmation needs namespace, hashes and nonempty family/lineage/scenario lists")
    rv = receipt["review"]
    if not (isinstance(rv, Mapping) and rv.get("status") == "approved" and isinstance(rv.get("reviewers"), list)
            and rv["reviewers"] and all(isinstance(r, str) and r.strip() for r in rv["reviewers"])):
        raise ValueError("receipt is not reviewed (review.status approved with named reviewers)")
    return {"status": "verified", "receipt_hash": receipt["receipt_hash"], "decision": receipt["decision"],
            "selected": dict(sel), "candidate": cand, "confirmation": dict(conf)}


# ----- leakage against held-out sets ----------------------------------------------------------------------
def heldout_overlap(records: Sequence[Mapping[str, Any]], heldout: Mapping[str, Mapping[str, set]]) -> dict[str, Any]:
    """Family, lineage and scenario overlap of training provenance records with each held-out set."""
    out = {}
    for name, sets in heldout.items():
        hit = {"family": sorted({r["scenario"]["family_id"] for r in records} & sets["family"]),
               "lineage": sorted({r["scenario"]["lineage_id"] for r in records} & sets["lineage"]),
               "scenario": sorted({r["scenario"]["scenario_hash"] for r in records} & sets["scenario"])}
        out[name] = hit
    return {"overlap": out, "pass": not any(v for hit in out.values() for v in hit.values())}


def _manifest_sets(m: Mapping[str, Any]) -> dict[str, set]:
    sc = m.get("scenarios", ())
    return {"family": {s["family_id"] for s in sc}, "lineage": {s["lineage_id"] for s in sc},
            "scenario": {s["scenario_hash"] for s in sc}}


# ----- entrypoint -----------------------------------------------------------------------------------------
def build_s1_training_set(sources: Sequence[tuple[str | Path, str]], *, repo: str | Path,
                          protocol_ref: Mapping[str, Any], receipt_ref: Mapping[str, Any],
                          manifests: Mapping[str, Mapping[str, Any]],
                          trusted_ref: str = "origin/main") -> S1TrainingSet:
    """The only S1 entrypoint. Return an `S1TrainingSet` or raise `S1GateError` with every reason."""
    reasons: list[str] = []
    report: dict[str, Any] = {"s1_gate_version": S1_GATE_VERSION, "trusted_ref": trusted_ref}

    def blocked(extra: str | None = None) -> S1GateError:
        if extra:
            reasons.append(extra)
        return S1GateError({**report, "decision": "BULK_S1_BLOCKED", "reasons": reasons})

    try:
        protocol, p_trust = load_trusted_json(repo, protocol_ref, trusted_ref=trusted_ref)
        receipt, r_trust = load_trusted_json(repo, receipt_ref, trusted_ref=trusted_ref)
    except (ValueError, OSError, subprocess.SubprocessError) as e:
        raise blocked(f"trust root: {e}") from None
    report["trust"] = {"protocol": p_trust, "receipt": r_trust}
    try:
        selection = verify_receipt(receipt, protocol, protocol_ref)
    except (ValueError, KeyError, TypeError) as e:
        raise blocked(f"selection receipt: {e}") from None
    report["selection"] = selection
    t4 = protocol["eligibility"]["T4"]

    # manifests
    deps = {"t3_23": protocol["eligibility"]["T3"].get("status") in (reg.ELIGIBLE, reg.EXCLUDED),
            "t4_24_resolved": t4["status"] == reg.ELIGIBLE or t4["blocked_identities"] == reg.blocked_records("T4"),
            "selection_gate_25": True}
    try:
        if set(REQUIRED_MANIFESTS) - set(manifests):
            raise ValueError(f"manifests {REQUIRED_MANIFESTS} are all required")
        for s in REQUIRED_MANIFESTS:
            check_manifest(manifests[s])
            if manifests[s].get("split") != s:
                raise ValueError(f"manifest for {s!r} is the {manifests[s].get('split')!r} split")
        if len({(manifests[s]["plan_hash"], manifests[s]["spec_hash"]) for s in REQUIRED_MANIFESTS}) != 1:
            raise ValueError("manifests come from different split plans")
        deps["generator_splits_6"] = True
    except (ValueError, KeyError, TypeError) as e:
        deps["generator_splits_6"] = False
        reasons.append(f"manifests: {e}")
    drift = reg.check_pins() + reg.check_registry()
    deps["teachers_7_8_9"] = not drift
    reasons.extend(f"identity drift: {d}" for d in drift)
    if reasons:
        raise blocked()

    # ingestion with mandatory replay + provenance
    records: list[dict[str, Any]] = []
    src_hashes = []
    failures = 0
    for path, source_id in sources:
        try:
            recs, entry = ingest_zip(Path(path), source_id=source_id, quality="teacher_verified",
                                     manifests={"train": manifests["train"]}, allowed_splits=("train",))
        except (ValueError, KeyError, TypeError, OSError) as e:
            failures += 1
            reasons.append(f"source {source_id}: {e}")
            continue
        src_hashes.append(entry["sha256"])
        records.extend(recs)
    sel = selection["selected"]
    for r in records:
        t = r["provenance_record"]["teacher"]
        if (r.get("replay") or {}).get("status") != "complete" or r["provenance"].get("status") != "verified":
            failures += 1
        elif {k: t[k] for k in ("name", "policy", "version", "config_hash")} != \
                {k: sel[k] for k in ("name", "policy", "version", "config_hash")} or \
                r["provenance_record"]["code_fingerprint"] != sel["code_fingerprint"]:
            failures += 1
            reasons.append(f"record from {t.get('name')} {t.get('version')} is not the selected identity")
            break
    report["records"] = len(records)

    # regeneration spot check
    items = [{"record": r["provenance_record"], "observation": r["observation"], "actions": r["actions"],
              "values": r.get("values")} for r in records]
    k = max(MIN_SPOT_CHECKS, math.ceil(SPOT_FRACTION * len(items)))
    spot = (pv.regeneration_spot_check(items, k=k, seed=digest([selection["receipt_hash"], sorted(src_hashes)]))
            if items else None)
    report["spot_check"] = None if spot is None else {x: spot[x] for x in ("checked", "population", "pass")}

    # leakage
    prov = [r["provenance_record"] for r in records]
    conf = selection["confirmation"]
    heldout = {"dev": _manifest_sets(manifests["dev"]), "calibration": _manifest_sets(manifests["calibration"]),
               "selection_confirmation": {"family": set(conf["family_ids"]), "lineage": set(conf["lineage_ids"]),
                                          "scenario": set(conf["scenario_hashes"])}}
    leak = pv.leakage_audit(prov) if prov else None
    over = heldout_overlap(prov, heldout)
    leakage = None if leak is None else {"pass": leak["pass"] and over["pass"], "audit": leak, "heldout": over}
    report["leakage"] = leakage

    gate = pv.bulk_s1_gate(selection=selection, dependencies=deps, provenance_failures=failures,
                           records_checked=len(records), spot_check=spot, leakage=leakage,
                           min_spot_checks=k)
    reasons.extend(gate["reasons"])
    report["dependencies"] = deps
    if reasons or gate["decision"] != "BULK_S1_AUTHORIZED":
        raise blocked()
    report.update(decision="BULK_S1_AUTHORIZED", reasons=[], sources=[str(p) for p, _ in sources],
                  source_sha256=src_hashes)
    return S1TrainingSet(tuple(records), report, _AUTH)


__all__ = ["S1_GATE_VERSION", "RECEIPT_VERSION", "MIN_SPOT_CHECKS", "S1GateError", "S1TrainingSet",
           "build_s1_training_set", "load_trusted_json", "protocol_hash", "receipt_hash", "verify_protocol",
           "verify_receipt", "heldout_overlap", "TRUST_ROOT_KINDS"]
