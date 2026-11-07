"""Teacher data provenance and trust gates (issue #10). Fail closed.

Full-episode replay (`sources.verify_teacher_decision`) shows only that the
recorded actions produce the recorded transitions. This module adds the
protections that replay does not give:

1 `provenance_record` binds one decision to the frozen teacher identity
  (name, policy, version, config hash), a fingerprint of the teacher's source
  code and of every `ember_v1` module it imports, the shared-utility
  versions, the policy seed, the observation and action digests, the
  per-action values digest (only for teachers that compute values), and the
  scenario's entry in an immutable split manifest (#6).
2 `verify_provenance` re-derives every field from the current code, the
  trace and the manifests, and raises `ProvenanceError` on any difference.
3 `regeneration_spot_check` re-runs the named frozen teacher on sampled
  observations and requires identical actions (and values when computed).
4 `leakage_audit` finds duplicates and cross-split leaks (observation,
  scenario, lineage, family).
5 `bulk_s1_gate` returns `BULK_S1_BLOCKED` unless every gate holds,
  including a frozen #25 selection. There is no frozen selection yet.

The digest is a hash binding, not a signature: it detects accidental and
inconsistent edits, and a forger who does not also match the frozen code,
config, seed and regeneration check. It cannot authenticate a party that
controls the repository; that needs a signing key, which is out of scope here.
"""
from __future__ import annotations

import ast
import hashlib
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .contract import validate_actions, validate_observation
from .gen.spec import GENERATOR_REVISION
from .gen.splits import LOCKED, MANIFEST_VERSION
from .sim.canonical import digest
from .teacher.grid import GRID_UTILS_VERSION
from .teacher.interface import POLICY_INTERFACE_VERSION, PolicyDecision, ReferencePolicy
from .teacher.t0 import T0ShortestPath
from .teacher.t1 import T1CongestionAware
from .teacher.t2 import T2StochasticRollout
from .teacher.t3 import T3DynamicACO

PROVENANCE_VERSION = "teacher-provenance@1.0.0"
REGISTRY: dict[str, type[ReferencePolicy]] = {"T0": T0ShortestPath, "T1": T1CongestionAware,
                                              "T2": T2StochasticRollout, "T3": T3DynamicACO}
VALUE_TEACHERS = frozenset({"T2"})        # only T2 computes per-action values (docs/P1_TEACHER_DIRECTION.md)
TRAINING_SPLITS = ("train",)
_PKG = Path(__file__).resolve().parent
_RECORD_KEYS = {"provenance_version", "teacher", "code_fingerprint", "utilities", "policy_seed", "step",
                "observation_digest", "actions_digest", "values_digest", "scenario", "split", "record_hash"}


class ProvenanceError(ValueError):
    """A provenance record does not match the frozen teacher, the trace or the manifest."""


# ----- code fingerprint --------------------------------------------------------------------------------
def _module_path(mod: str) -> Path | None:
    parts = mod.split(".")
    if parts[0] != "ember_v1":
        return None
    base = _PKG.joinpath(*parts[1:])
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    return None


def _imports(mod: str, path: Path) -> set[str]:
    pkg = mod if path.name == "__init__.py" else mod.rsplit(".", 1)[0]
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names if a.name.startswith("ember_v1"))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = pkg.split(".")
                base = ".".join(parts[: len(parts) - node.level + 1])
                full = f"{base}.{node.module}" if node.module else base
            elif node.module and node.module.startswith("ember_v1"):
                full = node.module
            else:
                continue
            out.add(full)
            out.update(f"{full}.{a.name}" for a in node.names if _module_path(f"{full}.{a.name}"))
    return {m for m in out if _module_path(m)}


def code_fingerprint(cls: type[ReferencePolicy]) -> str:
    """Digest of the policy module and its transitive `ember_v1` imports (LF-normalized source bytes)."""
    seen: dict[str, str] = {}
    todo = [cls.__module__]
    while todo:
        mod = todo.pop()
        path = _module_path(mod)
        if mod in seen or path is None:
            continue
        raw = path.read_bytes().replace(b"\r\n", b"\n")
        seen[mod] = "sha256:" + hashlib.sha256(raw).hexdigest()
        todo.extend(_imports(mod, path) - set(seen))
    return digest(dict(sorted(seen.items())))


# ----- records -----------------------------------------------------------------------------------------
def frozen_teacher(name: str, seed: int | None) -> ReferencePolicy:
    """The registered frozen teacher with the recorded seed (deterministic teachers take no seed)."""
    cls = REGISTRY.get(name)
    if cls is None:
        raise ProvenanceError(f"unknown teacher {name!r}")
    try:
        pol = cls() if seed is None else cls(seed=seed)
    except (TypeError, ValueError) as e:
        raise ProvenanceError(f"{name}: recorded seed {seed!r} is not valid for this teacher") from e
    if pol.seed != seed:
        raise ProvenanceError(f"{name}: recorded seed {seed!r} differs from the teacher seed {pol.seed!r}")
    return pol


def decision_values(name: str, decision_or_diag: PolicyDecision | Mapping[str, Any]) -> dict[str, Any] | None:
    """Per-action values that the teacher really computed, else None."""
    if name not in VALUE_TEACHERS:
        return None
    diag = decision_or_diag.diagnostics if isinstance(decision_or_diag, PolicyDecision) else decision_or_diag
    people = diag.get("people", {}) if isinstance(diag, Mapping) else {}
    vals = {pid: p.get("values") for pid, p in sorted(people.items()) if p.get("values") is not None}
    return vals or None


def _manifest_scenario(manifest: Mapping[str, Any], scenario_id: str) -> Mapping[str, Any]:
    for s in manifest.get("scenarios", ()):
        if s["scenario_id"] == scenario_id:
            return s
    raise ProvenanceError(f"scenario {scenario_id!r} is not in the {manifest.get('split')!r} manifest")


def _check_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise ProvenanceError("unsupported manifest version")
    body = {k: v for k, v in manifest.items() if k != "manifest_hash"}
    if digest(body) != manifest.get("manifest_hash"):
        raise ProvenanceError("manifest hash does not match its content")
    if manifest.get("generator_revision") != GENERATOR_REVISION:
        raise ProvenanceError("manifest generator revision differs from the current generator")


def provenance_record(policy: ReferencePolicy, decision: PolicyDecision, *, manifest: Mapping[str, Any],
                      scenario_id: str) -> dict[str, Any]:
    """Hash-bound provenance for one decision of a registered frozen teacher."""
    name = policy.identity.name
    if REGISTRY.get(name) is not type(policy):
        raise ProvenanceError(f"{name!r} is not a registered frozen teacher class")
    _check_manifest(manifest)
    if manifest["split"] == LOCKED:
        raise ProvenanceError("the locked test split never produces teacher records")
    entry = _manifest_scenario(manifest, scenario_id)
    vals = decision_values(name, decision)
    ident = policy.identity
    body = {
        "provenance_version": PROVENANCE_VERSION,
        "teacher": {"name": ident.name, "policy": ident.policy, "version": ident.version,
                    "config_hash": ident.config_hash},
        "code_fingerprint": code_fingerprint(type(policy)),
        "utilities": {"interface": decision.reproducibility["interface"], **decision.reproducibility["utilities"]},
        "policy_seed": policy.seed, "step": decision.reproducibility["step"],
        "observation_digest": decision.reproducibility["observation_digest"],
        "actions_digest": digest(decision.actions), "values_digest": None if vals is None else digest(vals),
        "scenario": {k: entry[k] for k in ("scenario_id", "scenario_hash", "lineage_id", "family_id")},
        "split": {"split": manifest["split"], "plan_id": manifest["plan_id"], "plan_hash": manifest["plan_hash"],
                  "manifest_hash": manifest["manifest_hash"], "generator_revision": manifest["generator_revision"],
                  "spec_hash": manifest["spec_hash"]},
    }
    return {**body, "record_hash": digest(body)}


def verify_provenance(record: Mapping[str, Any], *, observation: Mapping[str, Any], actions: Mapping[str, str],
                      values: Mapping[str, Any] | None, scenario_hash: str | None,
                      manifests: Mapping[str, Mapping[str, Any]],
                      allowed_splits: Sequence[str] = TRAINING_SPLITS) -> dict[str, Any]:
    """Re-derive every provenance field. Return a summary or raise `ProvenanceError` (fail closed)."""
    if not isinstance(record, Mapping) or set(record) != _RECORD_KEYS:
        raise ProvenanceError("provenance record has missing or extra fields")
    body = {k: v for k, v in record.items() if k != "record_hash"}
    if digest(body) != record["record_hash"]:
        raise ProvenanceError("record hash does not match the record")
    if record["provenance_version"] != PROVENANCE_VERSION:
        raise ProvenanceError("unsupported provenance version")
    t = record["teacher"]
    cls = REGISTRY.get(t.get("name"))
    frozen = frozen_teacher(t.get("name"), record["policy_seed"]).identity
    if (t.get("policy"), t.get("version"), t.get("config_hash")) != (frozen.policy, frozen.version, frozen.config_hash):
        raise ProvenanceError(f"{t['name']}: identity differs from the frozen teacher")
    if record["code_fingerprint"] != code_fingerprint(cls):
        raise ProvenanceError(f"{t['name']}: code fingerprint differs from the current frozen code")
    if record["utilities"] != {"interface": POLICY_INTERFACE_VERSION, "grid": GRID_UTILS_VERSION}:
        raise ProvenanceError("shared-utility versions differ")
    obs = validate_observation(observation)
    if digest(obs) != record["observation_digest"] or obs["step"] != record["step"]:
        raise ProvenanceError("observation does not match the record")
    if digest(validate_actions(obs, dict(actions))) != record["actions_digest"]:
        raise ProvenanceError("actions do not match the record")
    if t["name"] not in VALUE_TEACHERS and (values is not None or record["values_digest"] is not None):
        raise ProvenanceError(f"{t['name']} computes no per-action values; values are fabricated")
    if (None if values is None else digest(dict(values))) != record["values_digest"]:
        raise ProvenanceError("per-action values do not match the record")
    sp = record["split"]
    if sp.get("split") == LOCKED or sp.get("split") not in allowed_splits:
        raise ProvenanceError(f"split {sp.get('split')!r} is not allowed here")
    m = manifests.get(sp["split"])
    if m is None:
        raise ProvenanceError(f"no manifest for split {sp['split']!r}")
    _check_manifest(m)
    if {k: m[k] for k in ("split", "plan_id", "plan_hash", "manifest_hash", "generator_revision", "spec_hash")} != sp:
        raise ProvenanceError("record is bound to a different manifest")
    entry = _manifest_scenario(m, record["scenario"]["scenario_id"])
    if {k: entry[k] for k in ("scenario_id", "scenario_hash", "lineage_id", "family_id")} != record["scenario"]:
        raise ProvenanceError("scenario lineage/family/hash differs from the manifest")
    if scenario_hash is not None and scenario_hash != entry["scenario_hash"]:
        raise ProvenanceError("trace scenario differs from the manifest scenario")
    return {"status": "verified", "record_hash": record["record_hash"], "teacher": t["name"],
            "split": sp["split"], "family_id": entry["family_id"]}


# ----- regeneration spot check -------------------------------------------------------------------------
def regeneration_spot_check(items: Sequence[Mapping[str, Any]], *, k: int, seed: str) -> dict[str, Any]:
    """Re-run the named frozen teacher on k sampled items {record, observation, actions, values}.

    The sample is fixed by `seed` (a string) and the item order. Every sampled
    decision must give the same actions (and values digest) as recorded.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    rnd = random.Random(int.from_bytes(hashlib.sha256(f"spot-check/{seed}".encode()).digest()[:8], "big"))
    idx = sorted(rnd.sample(range(len(items)), min(k, len(items))))
    mismatches = []
    for i in idx:
        it = items[i]
        rec = it["record"]
        try:
            pol = frozen_teacher(rec["teacher"]["name"], rec["policy_seed"])
        except ProvenanceError as e:
            mismatches.append({"index": i, "reason": str(e)})
            continue
        d = pol.decide(it["observation"])
        vals = decision_values(pol.identity.name, d)
        if digest(d.actions) != rec["actions_digest"] or digest(dict(it["actions"])) != rec["actions_digest"]:
            mismatches.append({"index": i, "reason": "actions differ on regeneration"})
        elif (None if vals is None else digest(vals)) != rec["values_digest"]:
            mismatches.append({"index": i, "reason": "values differ on regeneration"})
    return {"seed": seed, "population": len(items), "checked": len(idx), "sampled": idx, "mismatches": mismatches,
            "pass": bool(idx) and not mismatches}


# ----- leakage and duplicates ----------------------------------------------------------------------------
def leakage_audit(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Cross-split leaks and duplicates over verified provenance records."""
    where: dict[str, dict[str, set[str]]] = {k: defaultdict(set) for k in ("observation", "scenario", "lineage",
                                                                           "family")}
    dup: dict[tuple[str, str, str], int] = defaultdict(int)
    n = 0
    for r in records:
        n += 1
        sp = r["split"]["split"]
        sc = r["scenario"]
        where["observation"][r["observation_digest"]].add(sp)
        where["scenario"][sc["scenario_hash"]].add(sp)
        where["lineage"][sc["lineage_id"]].add(sp)
        where["family"][sc["family_id"]].add(sp)
        dup[(r["teacher"]["name"], r["observation_digest"], r["actions_digest"])] += 1
    leaks = {k: sorted(key for key, sps in v.items() if len(sps) > 1) for k, v in where.items()}
    dups = sum(c - 1 for c in dup.values() if c > 1)
    return {"records": n, "cross_split_leaks": leaks, "duplicate_records": dups,
            "pass": n > 0 and not any(leaks.values()) and dups == 0}


# ----- bulk S1 gate --------------------------------------------------------------------------------------
REQUIRED_DEPENDENCIES = ("generator_splits_6", "teachers_7_8_9", "t3_23", "t4_24", "selection_gate_25")


def bulk_s1_gate(*, selection: Mapping[str, Any] | None, dependencies: Mapping[str, bool],
                 provenance_failures: int, records_checked: int, spot_check: Mapping[str, Any] | None,
                 leakage: Mapping[str, Any] | None, min_spot_checks: int = 50) -> dict[str, Any]:
    """`BULK_S1_AUTHORIZED` only if every gate holds; otherwise `BULK_S1_BLOCKED` with all reasons."""
    reasons = []
    for dep in REQUIRED_DEPENDENCIES:
        if dependencies.get(dep) is not True:
            reasons.append(f"dependency {dep} not met")
    if not selection:
        reasons.append("no frozen #25 teacher selection")
    else:
        body = {k: v for k, v in selection.items() if k != "selection_hash"}
        if selection.get("frozen") is not True or digest(body) != selection.get("selection_hash"):
            reasons.append("#25 selection is not frozen or its hash does not match")
        if selection.get("selected_teacher") not in REGISTRY:
            reasons.append("selected teacher is not a registered frozen teacher")
    if records_checked < 1 or provenance_failures:
        reasons.append(f"provenance: {provenance_failures} failures in {records_checked} records")
    if not spot_check or not spot_check.get("pass") or spot_check.get("checked", 0) < min_spot_checks:
        reasons.append(f"regeneration spot check missing, failed or < {min_spot_checks} decisions")
    if not leakage or not leakage.get("pass"):
        reasons.append("leakage audit missing or failed")
    return {"decision": "BULK_S1_BLOCKED" if reasons else "BULK_S1_AUTHORIZED", "reasons": reasons}
