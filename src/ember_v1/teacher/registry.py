"""Shared frozen teacher registry for provenance (#10) and teacher selection (#25).

One place records, for the five candidate families T0..T4:

- every retained frozen identity (name, policy, version, config hash and the
  class that implements it), pinned here so code drift is detected;
- every blocked identity with its failed preregistered gate and the hash of
  the committed report that blocked it (T4 v1.0.0 G5, T4 v1.1.0 G4);
- the teachers that really compute per-action values (`VALUE_TEACHERS`).

Retained is not eligible. Eligibility for one selection is a decision of a
frozen #25 protocol; `validate_eligibility` checks that decision against this
registry. A blocked identity can never be eligible, and a candidate that is
excluded must carry an explicit reason and the exact blocked records. Nothing
here selects a teacher or authorizes S1.

Classes are imported lazily from their dotted path, so this module imports no
teacher code and has no circular imports with `ember_v1.provenance` or
`ember_v1.teacher.selection`.
"""
from __future__ import annotations

import hashlib
import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REGISTRY_VERSION = "teacher-registry@1.0.0"
CANDIDATES = ("T0", "T1", "T2", "T3", "T4")
VALUE_TEACHERS = frozenset({"T2"})        # only T2 computes per-action values (docs/P1_TEACHER_DIRECTION.md)
RETAINED, BLOCKED = "retained", "blocked"
ELIGIBLE, EXCLUDED = "eligible", "excluded"
EXCLUSION_REASONS = ("blocked_by_preregistered_gate", "excluded_by_protocol_rule")
REPO = Path(__file__).resolve().parents[3]


class RegistryError(ValueError):
    """Unknown, blocked or drifted teacher identity, or an invalid eligibility record."""


@dataclass(frozen=True)
class Block:
    verdict: str
    failed_gate: str
    report: str               # repository path of the committed report
    report_sha256: str        # sha256 of the LF-normalized report bytes
    record: str               # repository path of the written verdict


@dataclass(frozen=True)
class Identity:
    candidate: str
    name: str
    policy: str
    version: str
    config_hash: str
    cls_path: str             # "module:Class"
    status: str
    block: Block | None = None

    def key(self) -> dict[str, str]:
        return {"name": self.name, "policy": self.policy, "version": self.version, "config_hash": self.config_hash}

    def blocked_record(self) -> dict[str, str]:
        if self.block is None:
            raise RegistryError(f"{self.name} {self.version} is not blocked")
        return {**self.key(), "verdict": self.block.verdict, "failed_gate": self.block.failed_gate,
                "report": self.block.report, "report_sha256": self.block.report_sha256}


IDENTITIES: tuple[Identity, ...] = (
    Identity("T0", "T0", "fire_aware_shortest_path", "1.0.0",
             "sha256:bdc4d3580bcf3889348c563aaf622334d135a535d9cb2aaeedc3c0528f10208a",
             "ember_v1.teacher.t0:T0ShortestPath", RETAINED),
    Identity("T1", "T1", "congestion_aware_cooperative_planner", "1.0.0",
             "sha256:5047e942db3326706b6184c9f29e0b992078920f03c8a53adc8cc24258345361",
             "ember_v1.teacher.t1:T1CongestionAware", RETAINED),
    Identity("T2", "T2", "stochastic_fire_forecast_rollout", "1.0.0",
             "sha256:302139e18ba05fdcdc5561b7390b95f725600ec8791e336d065f77cea0c507e5",
             "ember_v1.teacher.t2:T2StochasticRollout", RETAINED),
    Identity("T3", "T3", "dynamic_ant_colony_optimization", "1.0.0",
             "sha256:5a8d4985b1f65d5de18a06c86d8dd522e2ca95e2accd583165b7e7d7ea1d02e0",
             "ember_v1.teacher.t3:T3DynamicACO", RETAINED),
    Identity("T4", "T4", "risk_based_priority", "1.0.0",
             "sha256:dbcd0af4752ffcaad00e89a38ffcba5d3867fa3e4de0cce9fd560533d4794195",
             "ember_v1.teacher.t4:T4RiskPriority", BLOCKED,
             Block("BLOCKED_BY_MECHANISM_CHECK", "G5", "data/baselines/p1-08-t4-pilot.json",
                   "sha256:3aececbd715591933574c534e9845fbcdf097903667b86a75186355cc3508377",
                   "docs/P1_T4_RISK_PRIORITY.md")),
    Identity("T4", "T4", "risk_based_priority", "1.1.0",
             "sha256:67a3517de6fd109d128c0cf31836867fa0a1ec30cb09844e7a1541acae237ad4",
             "ember_v1.teacher.t4_v11:T4RiskPriorityV11", BLOCKED,
             Block("T4_V2_BLOCKED", "G4", "data/baselines/p1-08-t4-v2-pilot.json",
                   "sha256:fc172cbc3d1382294bb0c927cbe0d53baca1f9f824b6dda201708a6e54b5a872",
                   "docs/overnight/T4_V2_VERDICT.md")),
)


def identities(candidate: str | None = None) -> tuple[Identity, ...]:
    return tuple(i for i in IDENTITIES if candidate is None or i.candidate == candidate)


def identity(name: str, version: str | None = None) -> Identity:
    """The pinned identity. Without `version`, the single retained identity of `name`."""
    same = [i for i in IDENTITIES if i.name == name]
    if not same:
        raise RegistryError(f"unknown teacher {name!r}")
    if version is not None:
        for i in same:
            if i.version == version:
                return i
        raise RegistryError(f"unknown teacher version {name} {version!r}")
    kept = [i for i in same if i.status == RETAINED]
    if len(kept) != 1:
        blocked = ", ".join(f"{i.version} {i.block.verdict} ({i.block.failed_gate})" for i in same if i.block)
        raise RegistryError(f"{name} has no single retained identity; blocked: {blocked or 'none'}")
    return kept[0]


def _load(i: Identity) -> type:
    mod, cls = i.cls_path.split(":")
    return getattr(importlib.import_module(mod), cls)


def teacher_class(name: str, version: str | None = None) -> type:
    """Class of a retained identity. A blocked identity raises with its verdict (never a silent skip)."""
    i = identity(name, version)
    if i.status != RETAINED:
        raise RegistryError(f"{name} {i.version} is blocked ({i.block.verdict}, gate {i.block.failed_gate}); "
                            "it is not a selection or training teacher")
    return _load(i)


def retained_classes() -> dict[str, type]:
    """name -> class for every retained identity (one per name)."""
    return {i.name: _load(i) for i in IDENTITIES if i.status == RETAINED}


def check_pins() -> list[str]:
    """Differences between pinned identities and the current code (empty list = no drift)."""
    errors = []
    for i in IDENTITIES:
        ident = _load(i)().identity
        got = {"name": ident.name, "policy": ident.policy, "version": ident.version, "config_hash": ident.config_hash}
        if got != i.key():
            errors.append(f"{i.cls_path}: pinned {i.key()} != code {got}")
    return errors


def report_sha256(path: str | Path) -> str:
    raw = (REPO / path).read_bytes().replace(b"\r\n", b"\n")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def blocked_records(candidate: str) -> list[dict[str, str]]:
    return [i.blocked_record() for i in identities(candidate) if i.status == BLOCKED]


def default_eligibility() -> dict[str, dict[str, Any]]:
    """Every retained candidate eligible; every candidate without a retained identity excluded as blocked."""
    out: dict[str, dict[str, Any]] = {}
    for c in CANDIDATES:
        kept = [i for i in identities(c) if i.status == RETAINED]
        if kept:
            out[c] = {"status": ELIGIBLE, "identity": kept[0].key()}
        else:
            out[c] = {"status": EXCLUDED, "reason": "blocked_by_preregistered_gate",
                      "blocked_identities": blocked_records(c)}
    return out


def validate_eligibility(elig: Any) -> dict[str, dict[str, Any]]:
    """Fail closed on a missing, ambiguous or wrong eligibility record. Return it unchanged when valid.

    Rules: exactly the five candidates; `eligible` names a pinned retained
    identity; `excluded` has a known reason and lists exactly the registry's
    blocked identities of that candidate (`blocked_by_preregistered_gate` needs
    at least one; `excluded_by_protocol_rule` also needs a written `rule`);
    at least one candidate is eligible.
    """
    if not isinstance(elig, Mapping) or set(elig) != set(CANDIDATES):
        raise RegistryError(f"eligibility must list exactly {CANDIDATES}")
    eligible = 0
    for c in CANDIDATES:
        e = elig[c]
        if not isinstance(e, Mapping):
            raise RegistryError(f"{c}: eligibility entry must be a mapping")
        if e.get("status") == ELIGIBLE:
            if set(e) != {"status", "identity"} or not isinstance(e["identity"], Mapping):
                raise RegistryError(f"{c}: eligible entry needs exactly status and identity")
            idn = dict(e["identity"])
            match = [i for i in identities(c) if i.key() == idn]
            if not match:
                raise RegistryError(f"{c}: eligible identity {idn} is not a pinned identity of {c}")
            if match[0].status != RETAINED:
                raise RegistryError(f"{c} {match[0].version} is blocked ({match[0].block.verdict}); "
                                    "it cannot be eligible")
            eligible += 1
        elif e.get("status") == EXCLUDED:
            reason = e.get("reason")
            if reason not in EXCLUSION_REASONS:
                raise RegistryError(f"{c}: excluded entry needs a reason in {EXCLUSION_REASONS}")
            want_keys = {"status", "reason", "blocked_identities"} | ({"rule"} if reason == EXCLUSION_REASONS[1]
                                                                      else set())
            if set(e) != want_keys:
                raise RegistryError(f"{c}: excluded entry needs exactly {sorted(want_keys)}")
            if reason == EXCLUSION_REASONS[1] and not (isinstance(e["rule"], str) and e["rule"].strip()):
                raise RegistryError(f"{c}: excluded_by_protocol_rule needs a written rule")
            want = blocked_records(c)
            got = [dict(b) for b in e["blocked_identities"]] if isinstance(e["blocked_identities"], list) else None
            if got != want:
                raise RegistryError(f"{c}: blocked_identities must equal the registry blocked records {want}")
            if reason == EXCLUSION_REASONS[0] and not want:
                raise RegistryError(f"{c}: has no blocked identity; blocked_by_preregistered_gate is false")
        else:
            raise RegistryError(f"{c}: status must be {ELIGIBLE!r} or {EXCLUDED!r}")
    if not eligible:
        raise RegistryError("no eligible candidate")
    return {c: dict(elig[c]) for c in CANDIDATES}


def inventory() -> dict[str, Any]:
    """All five candidates with every retained and blocked identity (descriptive)."""
    return {"registry_version": REGISTRY_VERSION, "value_teachers": sorted(VALUE_TEACHERS),
            "candidates": {c: [{**i.key(), "status": i.status,
                                **({"block": i.blocked_record()} if i.block else {})} for i in identities(c)]
                           for c in CANDIDATES}}
