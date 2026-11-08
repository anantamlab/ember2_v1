"""Teacher selection protocol and gate (issue #25, docs/P1_TEACHER_SELECTION_PROTOCOL_DRAFT.md). Fail closed.

Five kinds of information stay separate:

1 retained identities: `registry.inventory()` (all five candidates; T4 v1.2.0
  retained with its admission record, T4 v1.0.0 and v1.1.0 kept as blocked
  records);
2 eligible teachers: the `eligibility` record of a frozen protocol, checked by
  `registry.validate_eligibility` (a blocked identity is never eligible);
3 blocked or excluded candidates: the same record, with reason and the exact
  blocked identities, verdicts, gates and report hashes;
4 historical exploratory evidence: `evidence_inventory()` of the committed
  p1-02-v1 DEV reports (P1-03..P1-06). It is descriptive only and can never
  count as confirmation;
5 prospective confirmation evidence: one run, after the protocol commit, on a
  new family-disjoint confirmation set (`selection_gate`).

`selection_gate` returns `NO_SELECTION` with every reason, or a decision
(`SELECT_STANDALONE` or `RETAIN_T1_NO_IMPROVEMENT`) with a receipt DRAFT whose
review is pending. Only a reviewed receipt committed to the trusted ref can
reach S1 (`ember_v1.s1`). Historical T1 is the provisional operational
baseline, not a confirmed selection. The joint-map arbiter needs its own
frozen protocol (`arbiter.py`); this gate compares standalone policies only.
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..gen.spec import GENERATOR_REVISION
from ..sim.canonical import digest
from .. import trust
from . import evaluate as ev
from . import registry as reg

SELECTION_VERSION = "teacher-selection@2.1.0"
PROTOCOL_VERSION = "teacher-selection-protocol@1.2.0"
EVIDENCE_VERSION = "teacher-selection-evidence@1.1.0"
FAMILYWISE_ALPHA = 0.05
SIGN_TEST = "two_sided_exact_family_sign_test"
HOLM_RULE = "holm_adjusted_p < familywise_alpha"
ENFORCEMENT = "live_supervisor_fail_closed"
WALL_S_CAP = 14400.0        # explicit wall-time safety limit of the one confirmatory run (new in protocol 1.2.0)
CANDIDATES = reg.CANDIDATES
BASELINE = "T1"
HISTORICAL, PROSPECTIVE = "historical_exploratory", "prospective_confirmation"
HISTORICAL_PLAN_IDS = ("p1-02-v1",)
DEFAULT_REPORTS = tuple(ev.REPO / f"data/baselines/{n}" for n in (
    "p1-03-t0-dev-v1.json", "p1-04-t1-dev-v1.json", "p1-05-t2-dev-v1.json", "p1-06-t3-dev-v1.json"))
CPU_CAP_USD = 15.0          # existing P1-P2 CPU cap (docs/P1_TEACHER_DIRECTION.md section 8)
REQUIRED_EXCLUDED_DATA = ("locked_test", "blind_test", "calibration", "emberbench")
REQUIRED_ISOLATION = ("p1-02-v1/train", "p1-02-v1/dev", "p1-02-v1/calibration", "p1-02-v1/locked_test",
                      "t2-pilot@1", "t3-pilot@1", "t4-pilot@1", "t4-pilot@2", "t4-explore@v12", "t4-pilot@3")
CONFIRMATION_KEYS = {"namespace", "families", "lineages_per_family", "generator_revision", "seed_namespace",
                     "isolation_against", "runs_per_identity", "manifest_hash", "policy_seeds"}
CONFIRMATION_MANIFEST = ev.REPO / "data/selection/p1-09-selection@1/manifest.json"
REQUIRED_STOP_RULES = ("freeze_commit_missing", "scenario_isolation_invalid", "unexpected_evaluation_error",
                       "identity_or_source_drift", "forbidden_information_read", "incomplete_matched_episodes",
                       "resource_cap_exceeded", "unverified_proposal_provenance", "rules_changed_after_results",
                       "blocked_t4_included", "working_tree_not_clean", "confirmation_attempt_exists",
                       "run_aborted_by_supervisor")
COMPUTE_KEYS = {"usd_cap", "cpu_s_cap", "per_decision_cpu_ms_cap", "replacement_cpu_s_cap", "wall_s_cap",
                "enforcement"}
EVIDENCE_KEYS = {"evidence_version", "evidence_class", "protocol_id", "protocol_hash", "protocol_commit",
                 "code_commit", "manifest", "rows", "complete", "errors", "attempt", "supervision", "evidence_hash"}
PROTOCOL_FIELDS = ("protocol_version", "protocol_id", "candidates", "eligibility", "t4_entry_rule", "baseline",
                   "historical_evidence", "confirmation", "primary", "safety", "compute", "tie_rule", "arbiter",
                   "excluded_data", "stop_rules", "frozen")
# Preregistered simplicity order (tie rule after lower CPU), fixed before any confirmation outcome:
# T0 one shortest path per person; T1 adds congestion-aware cooperative reservations; T3 adds a
# dynamic pheromone colony over T1-style routing; T4 adds risk-priority ordering, transactional repair
# and an exact vacate-infeasibility proof; T2 adds stochastic fire-forecast rollouts per action.
SIMPLICITY_ORDER = ("T0", "T1", "T3", "T4", "T2")
T4_ENTRY_KEYS = {"deadline", "requires", "after_freeze", "admitted", "blocked_identities"}
RESULT_KEYS = {"config_hash", "casualties", "evacuated", "unresolved", "people", "failure",
               "final_transition_hash", "policy_cpu_s", "max_decision_cpu_ms", "decisions",
               "budget_limited_decisions"}


class ProtocolError(ValueError):
    """A selection protocol is missing, ambiguous, unsafe or not frozen."""


# ----- 4 historical exploratory evidence --------------------------------------------------------------------
def evidence_inventory(paths: Sequence[str | Path] = DEFAULT_REPORTS) -> dict[str, Any]:
    """Descriptive inventory of committed DEV reports. Historical and exploratory; never confirmation."""
    per: dict[str, dict[str, Any]] = {c: {"reports": [], "config_hashes": set(), "rows": {}} for c in CANDIDATES}
    conflicts: list[dict[str, Any]] = []
    dev_hashes: set[str] = set()
    report_hashes = {}
    for path in paths:
        raw = Path(path).read_bytes()
        rep = json.loads(raw.decode("utf-8"))
        name = Path(path).name
        report_hashes[name] = trust.sha256_lf(raw)
        dev_hashes.add(digest(rep.get("dev")))
        for t, ident in rep.get("policies", {}).items():
            if t not in per:
                continue
            per[t]["reports"].append(name)
            per[t]["config_hashes"].add(ident["config_hash"])
            for r in rep["rows"]:
                x = r[t]
                cur = {"final_transition_hash": x.get("final_transition_hash"), "casualties": x["casualties"],
                       "evacuated": x["evacuated"], "unresolved": x["unresolved"], "people": x["people"],
                       "failure": x.get("failure"), "family_id": r.get("family_id"),   # absent in P1-03 rows
                       "policy_cpu_ms": (x.get("timing") or {}).get("policy_cpu_ms")}
                old = per[t]["rows"].get(r["position"])
                if old is None:
                    per[t]["rows"][r["position"]] = cur
                elif old["final_transition_hash"] != cur["final_transition_hash"]:
                    conflicts.append({"teacher": t, "position": r["position"], "report": name})
    fam = {pos: r["family_id"] for p in per.values() for pos, r in p["rows"].items() if r["family_id"]}
    out: dict[str, Any] = {}
    for t, p in per.items():
        rows = p["rows"]
        for pos, r in rows.items():
            r["family_id"] = r["family_id"] or fam.get(pos)
        blocked = reg.blocked_records(t)
        admitted = reg.admission_record(t)
        status = ("dev_evidence" if rows else "admitted_without_historical_evidence" if admitted
                  else "blocked" if blocked else "missing")
        out[t] = {"status": status, "reports": p["reports"], "blocked_identities": blocked, "admission": admitted,
                  "config_hashes": sorted(p["config_hashes"]), "scenarios": len(rows),
                  "families": len({r["family_id"] for r in rows.values()}),
                  "casualties": sum(r["casualties"] for r in rows.values()),
                  "evacuated": sum(r["evacuated"] for r in rows.values()),
                  "unresolved": sum(r["unresolved"] for r in rows.values()),
                  "people": sum(r["people"] for r in rows.values()),
                  "failures": sum(r["failure"] is not None for r in rows.values()),
                  "policy_cpu_s": (None if not rows or any(r["policy_cpu_ms"] is None for r in rows.values())
                                   else sum(r["policy_cpu_ms"] for r in rows.values()) / 1000),
                  "positions": sorted(rows)}
    pos = {t: tuple(v["positions"]) for t, v in out.items() if v["status"] == "dev_evidence"}
    return {"version": SELECTION_VERSION, "evidence_class": HISTORICAL, "counts_as_confirmation": False,
            "reports": [Path(p).name for p in paths], "report_sha256": report_hashes, "candidates": out,
            "cross_report_conflicts": conflicts, "same_dev_metadata": len(dev_hashes) == 1,
            "same_positions": len(set(pos.values())) <= 1,
            "note": "historical exploratory DEV inventory; separate reports are not one paired comparison and "
                    "never count as prospective confirmation"}


def summary_by_teacher(inventory: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    keys = ("status", "scenarios", "casualties", "evacuated", "unresolved", "policy_cpu_s")
    return {t: {k: c[k] for k in keys} for t, c in inventory["candidates"].items()}


# ----- protocol -------------------------------------------------------------------------------------------------
def draft_protocol() -> dict[str, Any]:
    """The P1-09 draft as data. NOT frozen: `frozen` is False until a reviewed commit freezes it."""
    inv = evidence_inventory()
    return {
        "protocol_version": PROTOCOL_VERSION, "protocol_id": "p1-09-teacher-selection",
        "candidates": list(CANDIDATES), "eligibility": reg.default_eligibility(),
        "t4_entry_rule": t4_entry_rule(),
        "baseline": {"teacher": BASELINE, "basis": "historical_exploratory_provisional"},
        "historical_evidence": {"use": "descriptive_only",
                                "reports": [{"report": k, "sha256": v} for k, v in sorted(inv["report_sha256"].items())]},
        "confirmation": {"namespace": "p1-09-selection@1", "families": 24, "lineages_per_family": 3,
                         "generator_revision": GENERATOR_REVISION, "seed_namespace": "p1-09-selection@1/seeds",
                         "isolation_against": list(REQUIRED_ISOLATION), "runs_per_identity": 1,
                         "manifest_hash": _committed_manifest_hash(), "policy_seeds": policy_seeds()},
        "primary": {"metric": "casualties", "comparison": "paired_vs_baseline", "unit": "family", "ci": 0.95,
                    "bootstrap_resamples": 10000, "bootstrap_seed": 20261109,
                    "multiplicity": multiplicity_rule(reg.default_eligibility())},
        "safety": {"casualties_plus_unresolved": "no_net_deterioration", "evacuated": "no_net_deterioration"},
        "compute": {"usd_cap": CPU_CAP_USD, "cpu_s_cap": 14400.0, "per_decision_cpu_ms_cap": 30000.0,
                    "replacement_cpu_s_cap": 7200.0, "wall_s_cap": WALL_S_CAP, "enforcement": ENFORCEMENT},
        "tie_rule": {"order": "lower_cpu_then_simplicity", "simplicity_order": list(SIMPLICITY_ORDER)},
        "arbiter": None,
        "excluded_data": list(REQUIRED_EXCLUDED_DATA),
        "stop_rules": list(REQUIRED_STOP_RULES),
        "frozen": False,
    }


def multiplicity_rule(eligibility: Mapping[str, Any]) -> dict[str, Any]:
    """Holm step-down over every eligible candidate-vs-T1 sign test (one familywise alpha)."""
    return {"method": "holm", "familywise_alpha": FAMILYWISE_ALPHA, "test": SIGN_TEST, "reject": HOLM_RULE,
            "hypotheses": [c for c in CANDIDATES if c != BASELINE and eligibility[c]["status"] == reg.ELIGIBLE]}


def holm_adjust(pvals: Mapping[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values: sorted ascending, p_(i) * (m - i + 1), made monotone, capped at 1."""
    order = sorted(pvals, key=lambda c: (pvals[c], c))
    m, run, out = len(order), 0.0, {}
    for i, c in enumerate(order):
        run = max(run, min(1.0, (m - i) * pvals[c]))
        out[c] = run
    return {c: out[c] for c in pvals}


def _committed_manifest_hash() -> str | None:
    """Hash of the committed p1-09-selection@1 manifest (built before any teacher outcome), or None."""
    if not CONFIRMATION_MANIFEST.is_file():
        return None
    return json.loads(CONFIRMATION_MANIFEST.read_text(encoding="utf-8")).get("manifest_hash")


def policy_seeds(eligibility: Mapping[str, Any] | None = None) -> dict[str, int]:
    """Frozen default seed of every eligible stochastic teacher (deterministic teachers have none)."""
    elig = eligibility or reg.default_eligibility()
    out = {}
    for c in CANDIDATES:
        if elig[c]["status"] == reg.ELIGIBLE:
            seed = reg.teacher_class(c, elig[c]["identity"]["version"])().seed
            if seed is not None:
                out[c] = seed
    return out


def t4_entry_rule() -> dict[str, Any]:
    """T4 admission at protocol freeze: the registry admission (or None) and every blocked T4 record."""
    return {"deadline": "protocol_freeze",
            "requires": "new T4 identity passing a new preregistered independent pilot gate",
            "after_freeze": "excluded for this protocol version",
            "admitted": reg.admission_record("T4"), "blocked_identities": reg.blocked_records("T4")}


def _num(x: Any, lo: float = 0.0, hi: float = math.inf, *, strict: bool = True) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and (lo < x if strict else lo <= x) and x <= hi


def validate_protocol(p: Any) -> dict[str, Any]:
    """Fail closed on a missing, ambiguous or unsafe protocol field. Return the body (no hash check)."""
    if not isinstance(p, Mapping):
        raise ProtocolError("protocol must be a mapping")
    missing = [f for f in PROTOCOL_FIELDS if f not in p]
    if missing:
        raise ProtocolError(f"protocol misses {missing}")
    extra = set(p) - set(PROTOCOL_FIELDS) - {"protocol_hash"}
    if extra:
        raise ProtocolError(f"protocol has unknown fields {sorted(extra)}")
    if p["protocol_version"] != PROTOCOL_VERSION:
        raise ProtocolError("unsupported protocol version")
    if tuple(p["candidates"]) != CANDIDATES:
        raise ProtocolError("protocol must name all five candidates T0..T4 separately")
    try:
        elig = reg.validate_eligibility(p["eligibility"])
    except reg.RegistryError as e:
        raise ProtocolError(f"eligibility: {e}") from None
    eligible = [c for c in CANDIDATES if elig[c]["status"] == reg.ELIGIBLE]
    t4r = p["t4_entry_rule"]
    if not (isinstance(t4r, Mapping) and set(t4r) == T4_ENTRY_KEYS and t4r.get("deadline") == "protocol_freeze"
            and t4r.get("after_freeze") == "excluded for this protocol version" and t4r.get("requires")):
        raise ProtocolError(f"t4_entry_rule must close T4 entry at protocol freeze with exactly {sorted(T4_ENTRY_KEYS)}")
    if t4r["blocked_identities"] != reg.blocked_records("T4"):
        raise ProtocolError("t4_entry_rule must list exactly the registry blocked T4 records (v1.0.0 G5, v1.1.0 G4)")
    admitted = t4r["admitted"]
    if elig["T4"]["status"] == reg.ELIGIBLE:
        want = reg.admission_record("T4")
        if want is None or admitted != want:
            raise ProtocolError("eligible T4 needs the exact registry admission record (pilot verdict, report hash, "
                                "freeze commit, namespace)")
        if {k: admitted[k] for k in ("name", "policy", "version", "config_hash")} != dict(elig["T4"]["identity"]):
            raise ProtocolError("T4 admission record names another identity than the eligible T4")
    elif admitted is not None:
        raise ProtocolError("excluded T4 must not carry an admission record")
    if p["baseline"] != {"teacher": BASELINE, "basis": "historical_exploratory_provisional"} or \
            BASELINE not in eligible:
        raise ProtocolError("baseline must be eligible T1 as the historical provisional baseline")
    h = p["historical_evidence"]
    if not (isinstance(h, Mapping) and h.get("use") == "descriptive_only" and isinstance(h.get("reports"), list)):
        raise ProtocolError("historical evidence must be descriptive_only with its report list")
    c = p["confirmation"]
    if not isinstance(c, Mapping) or set(c) != CONFIRMATION_KEYS:
        raise ProtocolError("confirmation design needs exactly its preregistered fields")
    if not isinstance(c["namespace"], str) or not c["namespace"] or \
            any(c["namespace"].startswith(h) for h in HISTORICAL_PLAN_IDS):
        raise ProtocolError("confirmation needs a new namespace, not a historical plan")
    if not (isinstance(c["families"], int) and c["families"] >= 10 and isinstance(c["lineages_per_family"], int)
            and c["lineages_per_family"] >= 1):
        raise ProtocolError("confirmation needs >= 10 families and >= 1 lineage per family")
    if c["generator_revision"] != GENERATOR_REVISION:
        raise ProtocolError("confirmation generator revision differs from the current generator")
    if set(REQUIRED_ISOLATION) - set(c["isolation_against"]):
        raise ProtocolError(f"confirmation isolation must cover {REQUIRED_ISOLATION}")
    if c["runs_per_identity"] != 1:
        raise ProtocolError("exactly one confirmatory run per identity")
    mh = c["manifest_hash"]
    if not (isinstance(mh, str) and mh.startswith("sha256:") and len(mh) == 71):
        raise ProtocolError("confirmation needs the hash of the manifest built before any teacher outcome")
    if c["policy_seeds"] != policy_seeds(elig):
        raise ProtocolError("confirmation policy seeds must be the frozen seeds of the eligible stochastic teachers")
    pr = p["primary"]
    if not (isinstance(pr, Mapping) and pr.get("metric") == "casualties" and pr.get("unit") == "family"
            and pr.get("comparison") == "paired_vs_baseline" and pr.get("ci") == 0.95
            and isinstance(pr.get("bootstrap_resamples"), int) and pr["bootstrap_resamples"] >= 1000
            and isinstance(pr.get("bootstrap_seed"), int)
            and set(pr) == {"metric", "unit", "comparison", "ci", "bootstrap_resamples", "bootstrap_seed",
                            "multiplicity"}):
        raise ProtocolError("primary endpoint must be paired family-level casualties vs T1, 95% CI, family sign "
                            "tests")
    if pr["multiplicity"] != multiplicity_rule(elig):
        raise ProtocolError("primary needs the Holm correction over every eligible candidate-vs-T1 sign test at "
                            f"familywise alpha {FAMILYWISE_ALPHA}")
    if p["safety"] != {"casualties_plus_unresolved": "no_net_deterioration", "evacuated": "no_net_deterioration"}:
        raise ProtocolError("safety guardrails must bound casualties+unresolved and evacuated")
    cm = p["compute"]
    if not (isinstance(cm, Mapping) and set(cm) == COMPUTE_KEYS and _num(cm.get("usd_cap"), 0, CPU_CAP_USD)
            and _num(cm.get("cpu_s_cap")) and _num(cm.get("per_decision_cpu_ms_cap"))
            and _num(cm.get("replacement_cpu_s_cap")) and _num(cm.get("wall_s_cap"))
            and cm.get("enforcement") == ENFORCEMENT):
        raise ProtocolError(f"compute needs positive caps (CPU, per decision, replacement, wall), live fail-closed "
                            f"enforcement and usd_cap within the ${CPU_CAP_USD:g} CPU cap")
    tr = p["tie_rule"]
    if not (isinstance(tr, Mapping) and set(tr) == {"order", "simplicity_order"}
            and tr.get("order") == "lower_cpu_then_simplicity"
            and list(tr.get("simplicity_order") or []) == [c for c in SIMPLICITY_ORDER if c in eligible]):
        raise ProtocolError(f"tie rule must order exactly the eligible candidates as {SIMPLICITY_ORDER}")
    arb = p["arbiter"]
    if arb is not None and not (isinstance(arb, Mapping) and set(arb) == {"protocol_ref", "protocol_hash"}):
        raise ProtocolError("an arbiter needs its own separately frozen protocol (protocol_ref, protocol_hash)")
    if set(REQUIRED_EXCLUDED_DATA) - set(p["excluded_data"]):
        raise ProtocolError(f"protocol must exclude {REQUIRED_EXCLUDED_DATA}")
    if set(REQUIRED_STOP_RULES) - set(p["stop_rules"]):
        raise ProtocolError(f"protocol needs stop rules {REQUIRED_STOP_RULES}")
    if p["frozen"] is not True:
        raise ProtocolError("protocol must be frozen")
    return {k: v for k, v in p.items() if k != "protocol_hash"}


def protocol_hash(p: Mapping[str, Any]) -> str:
    return digest({k: v for k, v in p.items() if k != "protocol_hash"})


def freeze_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a protocol and add its hash. Committing it to the trusted ref is the freeze."""
    body = validate_protocol(protocol)
    return {**body, "protocol_hash": digest(body)}


def verify_frozen_protocol(protocol: Any) -> dict[str, Any]:
    body = validate_protocol(protocol)
    if digest(body) != protocol.get("protocol_hash"):
        raise ProtocolError("protocol hash does not match its content")
    return dict(protocol)


# ----- 5 prospective confirmation -------------------------------------------------------------------------------
def _sets(entries: Sequence[Mapping[str, Any]]) -> dict[str, set]:
    return {k: {e[f] for e in entries if e.get(f) is not None}
            for k, f in (("family", "family_id"), ("lineage", "lineage_id"), ("scenario", "scenario_hash"),
                         ("core", "core_hash"))}


def isolation_audit(manifest: Mapping[str, Any], historical: Mapping[str, Sequence[Mapping[str, Any]]]
                    ) -> dict[str, Any]:
    """Family, lineage, scenario and core-geometry overlap of the confirmation set with each historical set."""
    mine = _sets(manifest["scenarios"])
    over = {}
    for name, entries in historical.items():
        theirs = _sets(entries)
        over[name] = {k: sorted(mine[k] & theirs[k]) for k in mine}
    return {"overlap": over, "pass": not any(v for o in over.values() for v in o.values())}


def _family_stats(rows: Sequence[Mapping[str, Any]], cand: str, base: str, primary: Mapping[str, Any]
                  ) -> dict[str, Any]:
    by_f: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_f[r["family_id"]].append(r["results"][cand]["casualties"] - r["results"][base]["casualties"])
    fams = sorted(by_f)
    n_s = sum(len(v) for v in by_f.values())
    est = sum(sum(v) for v in by_f.values()) / n_s
    rnd = random.Random(primary["bootstrap_seed"])
    boots = []
    for _ in range(primary["bootstrap_resamples"]):
        pick = [fams[rnd.randrange(len(fams))] for _ in fams]
        boots.append(sum(sum(by_f[f]) for f in pick) / sum(len(by_f[f]) for f in pick))
    boots.sort()
    a = (1 - primary["ci"]) / 2
    lo = boots[int(math.floor(a * (len(boots) - 1)))]
    hi = boots[int(math.ceil((1 - a) * (len(boots) - 1)))]
    fam_sum = {f: sum(by_f[f]) for f in fams}
    neg = sum(v < 0 for v in fam_sum.values())
    pos = sum(v > 0 for v in fam_sum.values())
    n = neg + pos
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(neg, pos) + 1)) / 2 ** n)
    # Raw (unadjusted) two-sided p; `selection_gate` applies the Holm correction across all hypotheses.
    return {"mean_paired_casualty_diff": est, "ci95": [lo, hi], "families": len(fams),
            "families_fewer": neg, "families_more": pos, "families_tied": len(fams) - n, "sign_test_p": p,
            "ci_excludes_zero_favourably": hi < 0, "direction_favours_candidate": neg > pos}


def selection_gate(protocol: Any, evidence: Any, *, protocol_ref: Any, repo: str | Path,
                   historical: Mapping[str, Sequence[Mapping[str, Any]]],
                   trusted_ref: str = "origin/main") -> dict[str, Any]:
    """`NO_SELECTION` with reasons, or a decision with a receipt draft (review pending). Never final."""
    reasons: list[str] = []
    out: dict[str, Any] = {"version": SELECTION_VERSION, "decision": "NO_SELECTION", "reasons": reasons,
                           "selected": None, "receipt_draft": None}
    try:
        p = verify_frozen_protocol(protocol)
    except (ProtocolError, AttributeError) as e:
        reasons.append(f"protocol: {e}")
        return out
    try:
        trusted, _ = trust.load_trusted_json(repo, protocol_ref, trusted_ref=trusted_ref)
        if trusted != json.loads(json.dumps(p)):
            raise ValueError("protocol differs from the committed protocol")
    except (ValueError, OSError) as e:
        reasons.append(f"protocol freeze commit: {e}")
        return out
    if not isinstance(evidence, Mapping):
        reasons.append("no prospective confirmation evidence")
        return out
    if evidence.get("evidence_class") != PROSPECTIVE or "dev" in evidence or \
            (evidence.get("manifest") or {}).get("plan_id") in HISTORICAL_PLAN_IDS:
        reasons.append("historical DEV evidence is exploratory; it is not prospective confirmation")
        return out
    if set(evidence) != EVIDENCE_KEYS:
        reasons.append("evidence has missing or extra fields")
        return out
    if digest({k: v for k, v in evidence.items() if k != "evidence_hash"}) != evidence["evidence_hash"]:
        reasons.append("evidence hash does not match its content")
    if evidence["evidence_version"] != EVIDENCE_VERSION:
        reasons.append("unsupported evidence version")
    if (evidence["protocol_id"], evidence["protocol_hash"]) != (p["protocol_id"], p["protocol_hash"]):
        reasons.append("evidence was produced under another protocol")
    if evidence["protocol_commit"] != protocol_ref["commit"]:
        reasons.append("evidence names another protocol commit")
    elif not trust.is_ancestor(repo, protocol_ref["commit"], evidence["code_commit"]):
        reasons.append("evidence code commit does not descend from the protocol freeze commit")
    if evidence["complete"] is not True or evidence["errors"]:
        reasons.append("confirmation run incomplete or with errors")
    sup, att = evidence["supervision"], evidence["attempt"]
    if not (isinstance(sup, Mapping) and sup.get("stage") == "completed" and sup.get("abort") is None
            and sup.get("limits") == {k: p["compute"][k] for k in ("cpu_s_cap", "per_decision_cpu_ms_cap",
                                                                    "wall_s_cap")}):
        reasons.append(f"confirmation run not completed under the protocol limits by the supervisor "
                       f"(stage {sup.get('stage') if isinstance(sup, Mapping) else None!r})")
    if not (isinstance(att, Mapping) and att.get("protocol_hash") == p["protocol_hash"]
            and att.get("protocol_commit") == protocol_ref["commit"]
            and att.get("code_commit") == evidence["code_commit"]):
        reasons.append("evidence lacks the attempt marker of this protocol and code commit")

    conf = p["confirmation"]
    m = evidence["manifest"]
    try:
        trust.check_manifest(m)
        if m["plan_id"] != conf["namespace"] or m["generator_revision"] != conf["generator_revision"]:
            raise ValueError("confirmation manifest is not the preregistered namespace/generator")
        fams = {s["family_id"] for s in m["scenarios"]}
        per_f = defaultdict(set)
        for s in m["scenarios"]:
            per_f[s["family_id"]].add(s["lineage_id"])
        if len(fams) != conf["families"] or any(len(v) != conf["lineages_per_family"] for v in per_f.values()):
            raise ValueError(f"confirmation set is not {conf['families']} families x "
                             f"{conf['lineages_per_family']} lineages")
        if m["manifest_hash"] != conf["manifest_hash"]:
            raise ValueError("confirmation manifest is not the manifest preregistered in the protocol")
    except (ValueError, KeyError, TypeError) as e:
        reasons.append(f"confirmation manifest: {e}")
        return out
    missing_iso = set(conf["isolation_against"]) - set(historical)
    if missing_iso:
        reasons.append(f"isolation audit misses historical sets {sorted(missing_iso)}")
    iso = isolation_audit(m, historical)
    out["isolation"] = iso
    if not iso["pass"]:
        reasons.append("confirmation set overlaps historical families, lineages, scenarios or geometry")

    elig = p["eligibility"]
    eligible = [c for c in CANDIDATES if elig[c]["status"] == reg.ELIGIBLE]
    rows = evidence["rows"]
    want = {s["scenario_id"]: s for s in m["scenarios"]}
    seen = set()
    cpu = defaultdict(float)
    for r in rows if isinstance(rows, list) else []:
        sid = r.get("scenario_id")
        ent = want.get(sid)
        if ent is None or sid in seen or (r.get("family_id"), r.get("scenario_hash")) != (ent["family_id"],
                                                                                         ent["scenario_hash"]):
            reasons.append(f"row {sid!r} is not one confirmation scenario")
            continue
        seen.add(sid)
        res = r.get("results") or {}
        for c in sorted(set(res) - set(eligible)):
            reasons.append(f"excluded candidate {c} appears in confirmation evidence")
        for c in eligible:
            x = res.get(c)
            if not isinstance(x, Mapping) or set(x) != RESULT_KEYS:
                reasons.append(f"{sid}: {c} result missing or malformed")
                continue
            if x["config_hash"] != elig[c]["identity"]["config_hash"]:
                reasons.append(f"{sid}: {c} identity differs from the protocol")
            if x["failure"] is not None:
                reasons.append(f"{sid}: {c} failed episode")
            if x["max_decision_cpu_ms"] > p["compute"]["per_decision_cpu_ms_cap"]:
                reasons.append(f"{sid}: {c} decision CPU over cap")
            cpu[c] += x["policy_cpu_s"]
    if seen != set(want):
        reasons.append(f"incomplete matched episodes: {len(set(want) - seen)} scenarios missing")
    if sum(cpu.values()) > p["compute"]["cpu_s_cap"]:
        reasons.append("total confirmation CPU over cap")
    if reasons:
        return out

    analyses = {}
    tot = {c: {k: sum(r["results"][c][k] for r in rows) for k in ("casualties", "unresolved", "evacuated")}
           for c in eligible}
    mult = p["primary"]["multiplicity"]
    stats = {c: _family_stats(rows, c, BASELINE, p["primary"]) for c in mult["hypotheses"]}
    adj = holm_adjust({c: st["sign_test_p"] for c, st in stats.items()})
    for c, st in stats.items():
        sup_ok = st["ci_excludes_zero_favourably"] and st["direction_favours_candidate"] and \
            adj[c] < mult["familywise_alpha"]
        safe = (tot[c]["casualties"] + tot[c]["unresolved"] <= tot[BASELINE]["casualties"] + tot[BASELINE]["unresolved"]
                and tot[c]["evacuated"] >= tot[BASELINE]["evacuated"])
        feasible = cpu[c] <= p["compute"]["replacement_cpu_s_cap"]
        analyses[c] = {**st, "sign_test_p_holm": adj[c], "superior": sup_ok, "totals": tot[c], "safety_pass": safe,
                       "budget_limited_decisions": sum(r["results"][c]["budget_limited_decisions"] for r in rows),
                       "cpu_s": cpu[c], "compute_feasible": feasible, "qualifies": sup_ok and safe and feasible}
    out["analyses"] = analyses
    out["multiplicity"] = {"method": mult["method"], "familywise_alpha": mult["familywise_alpha"],
                           "hypotheses": list(mult["hypotheses"]),
                           "raw_p": {c: st["sign_test_p"] for c, st in stats.items()}, "holm_adjusted_p": adj}
    out["baseline_totals"] = tot[BASELINE]
    q = [c for c, a in analyses.items() if a["qualifies"]]
    order = p["tie_rule"]["simplicity_order"]
    if q:
        pick = min(q, key=lambda c: (cpu[c], order.index(c)))
        decision = "SELECT_STANDALONE"
    else:
        pick, decision = BASELINE, "RETAIN_T1_NO_IMPROVEMENT"
    out.update(decision=decision, selected=pick,
               receipt_draft=receipt_draft(p, protocol_ref, decision, pick, m, evidence["evidence_hash"]))
    return out


def receipt_draft(p: Mapping[str, Any], protocol_ref: Mapping[str, Any], decision: str, pick: str,
                  manifest: Mapping[str, Any], evidence_hash: str) -> dict[str, Any]:
    """An S1 receipt (`ember_v1.s1`) with review pending. S1 refuses it until reviewers approve and commit it."""
    from .. import provenance as pv           # lazy: provenance is not needed for the descriptive functions
    from .. import s1
    ident = p["eligibility"][pick]["identity"]
    sc = manifest["scenarios"]
    body = {"receipt_version": s1.RECEIPT_VERSION, "protocol_id": p["protocol_id"],
            "protocol_hash": p["protocol_hash"], "protocol_ref": dict(protocol_ref), "decision": decision,
            "selected": {**ident, "code_fingerprint": pv.code_fingerprint(reg.teacher_class(ident["name"],
                                                                                        ident["version"]))},
            "eligibility": p["eligibility"],
            "confirmation": {"namespace": manifest["plan_id"], "manifest_hash": manifest["manifest_hash"],
                             "evidence_hash": evidence_hash,
                             "family_ids": sorted({s["family_id"] for s in sc}),
                             "lineage_ids": sorted({s["lineage_id"] for s in sc}),
                             "scenario_hashes": sorted({s["scenario_hash"] for s in sc})},
            "review": {"status": "pending", "reviewers": []}}
    return {**body, "receipt_hash": s1.receipt_hash(body)}


def status(protocol: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The five separated views. Today: no frozen protocol, no prospective evidence, NO_SELECTION."""
    p = protocol or draft_protocol()
    elig = reg.validate_eligibility(p["eligibility"])
    return {"retained_identities": reg.inventory(),
            "eligible": [c for c in CANDIDATES if elig[c]["status"] == reg.ELIGIBLE],
            "excluded": {c: e for c, e in elig.items() if e["status"] == reg.EXCLUDED},
            "historical_exploratory_evidence": summary_by_teacher(evidence_inventory()),
            "prospective_confirmation_evidence": None,
            "protocol_frozen": p.get("frozen") is True, "operational_baseline": BASELINE,
            "operational_baseline_basis": "historical_exploratory_provisional",
            "decision": "NO_SELECTION"}
