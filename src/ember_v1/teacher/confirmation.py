"""Prospective five-teacher confirmation set `p1-09-selection@1` and its matched-seed runner (issue #25).

Build (deterministic, CPU, no teacher runs):

1 historical sets: every earlier split and pilot is regenerated from its public
  plan or cell list. `p1-02-v1` train/dev/calibration must equal the committed
  manifests and the regenerated locked test must equal its index commitment,
  else the build stops. Pilot sets: T2 `t2-pilot@1`, T3 `t3-pilot@1`, T4
  `t4-pilot@1`, `t4-pilot@2`, `t4-pilot@3`, `t4-explore@v12` (every defined cell).
2 family pool: pool slot i uses `derive_seed(MASTER_SEED, SEED_NAMESPACE, "family", i)`.
  A family is rejected if its map duplicates or nearly duplicates (also the
  exit-free wall rule) any `p1-02-v1` family or an accepted one.
3 lineages: `derive_seed(family_seed, "lineage", j)`, j < LINEAGES. A family is
  excluded (recorded, never repaired) if a lineage fails to generate or any
  lineage shares a family ID, lineage ID, scenario hash, core-geometry hash or
  simulator seed with a historical set. The next slot is then tried.
4 stop after FAMILIES complete families or MAX_SLOTS slots (then the build fails).

The manifest holds identities, seeds, hashes and diagnostics only; teacher
outcomes are never computed here. The runner (`run`) refuses to start unless
the frozen protocol and this manifest are committed on the trusted ref, the
code is clean and descends from the protocol commit, and no evidence exists.
Every eligible teacher then runs every scenario once from a fresh instance,
with the simulator seed of the scenario (matched across teachers) and its
preregistered policy seed.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..gen import analysis as A
from ..gen import splits as S
from ..gen.generator import Family, GenerationError, generate_family, generate_lineage, sample_target
from ..gen.rng import derive_seed
from ..gen.spec import DEFAULT_SPEC, GENERATOR_REVISION, GeneratorSpec
from ..sim import Scenario
from ..sim.canonical import digest
from .. import trust
from . import evaluate as ev
from . import registry as reg

CONF_VERSION = "p1-09-confirmation-set@1.0.0"
NAMESPACE = "p1-09-selection@1"
SEED_NAMESPACE = "p1-09-selection@1/seeds"
MASTER_SEED = 20261108
FAMILIES, LINEAGES = 24, 3
MAX_SLOTS = 3 * FAMILIES
SPLIT = "selection_confirmation"
OUT_DIR = ev.REPO / "data/selection/p1-09-selection@1"
MANIFEST_PATH = OUT_DIR / "manifest.json"
AUDIT_PATH = OUT_DIR / "isolation_audit.json"
EVIDENCE_PATH = OUT_DIR / "evidence.json"
HISTORICAL = ("p1-02-v1/train", "p1-02-v1/dev", "p1-02-v1/calibration", "p1-02-v1/locked_test",
              "t2-pilot@1", "t3-pilot@1", "t4-pilot@1", "t4-pilot@2", "t4-explore@v12", "t4-pilot@3")
ID_KEYS = ("family_id", "lineage_id", "scenario_hash", "core_hash", "scenario_seed")
# CPU estimate for the preflight: committed DEV CPU (48 scenarios) x scenarios/48 x safety factor.
CPU_SAFETY = 1.5
DEV_SCENARIOS = 48
T4_V12_DEV_CPU_REPORT = "data/baselines/p1-08-t4-v12-pilot.json"


class ConfirmationError(RuntimeError):
    """The confirmation set cannot be built, audited or run under its preregistered rules."""


def plan() -> dict[str, Any]:
    return {"plan_id": NAMESPACE, "conf_version": CONF_VERSION, "master_seed": MASTER_SEED,
            "seed_namespace": SEED_NAMESPACE, "spec": DEFAULT_SPEC, "families": FAMILIES,
            "lineages_per_family": LINEAGES, "max_slots": MAX_SLOTS, "reference_plan": S.DEFAULT_PLAN["plan_id"],
            "rules": {"family": "reject topology duplicate, near duplicate or wall near duplicate of p1-02-v1 "
                                "or an accepted family",
                      "lineage": "exclude the family if any lineage fails or collides with a historical set",
                      "isolation_keys": list(ID_KEYS), "historical_sets": list(HISTORICAL)}}


def plan_hash() -> str:
    return digest({"generator_revision": GENERATOR_REVISION, "plan": plan()})


# ----- 1 historical sets -------------------------------------------------------------------------------------
def _pilot_entry(ns: str, s: Scenario) -> dict[str, Any]:
    return {"family_id": f"{ns}/{s.scenario_id}", "lineage_id": f"{ns}/{s.scenario_id}",
            "scenario_hash": s.content_hash(), "core_hash": A.core_hash(s), "scenario_seed": s.seed}


def _pilot_scenarios() -> dict[str, list[Scenario]]:
    from . import t2_pilot, t3_pilot, t4_pilot, t4_v12_pilot
    from . import t4_v2_pilot as v2
    return {
        "t2-pilot@1": [t2_pilot.pilot_scenario(t, n, m) for t, n, m in t2_pilot.pilot_cells()],
        "t3-pilot@1": [t3_pilot.pilot_scenario(c["topology"], c["n"], c["model"], c["grid"], c["exits"])
                       for c in t3_pilot.pilot_cells()],
        "t4-pilot@1": [t4_pilot.pilot_scenario(c) for c in t4_pilot.pilot_cells()],
        "t4-pilot@2": [v2.pilot_scenario_v2(c) for c in v2.pilot_cells_v2()],
        "t4-pilot@3": [t4_v12_pilot.scenario(c, t4_v12_pilot.NAMESPACE) for c in v2.pilot_cells_v2()],
        "t4-explore@v12": [t4_v12_pilot.scenario(c, t4_v12_pilot.EXPLORATORY_NAMESPACE)
                           for c in v2.pilot_cells_v2()],
    }


def historical_sets() -> tuple[dict[str, list[dict[str, Any]]], list[Family], dict[str, Any]]:
    """(entries per historical set, the p1-02-v1 families, provenance of the regeneration)."""
    build = S.build_splits(S.SplitPlan.default())
    idx = json.loads((ev.REPO / "data/splits/p1-02-v1/index.json").read_text(encoding="utf-8"))
    entries: dict[str, list[dict[str, Any]]] = {}
    checked = {}
    for split in S.SPLITS:
        m = S.manifest(build, split)
        want = idx["splits"][split]["manifest_hash"]
        if split in S.PUBLIC_SPLITS:
            committed = json.loads((ev.REPO / f"data/splits/p1-02-v1/{split}.json").read_text(encoding="utf-8"))
            want = committed["manifest_hash"]
        if m["manifest_hash"] != want:
            raise ConfirmationError(f"regenerated p1-02-v1/{split} differs from its committed hash")
        checked[f"p1-02-v1/{split}"] = {"manifest_hash": want, "source": "committed manifest" if split in
                                        S.PUBLIC_SPLITS else "index commitment (sealed split)"}
        entries[f"p1-02-v1/{split}"] = [{k: e[k] for k in ID_KEYS} for e in m["scenarios"]]
    for ns, scs in _pilot_scenarios().items():
        entries[ns] = [_pilot_entry(ns, s) for s in scs]
        checked[ns] = {"cells": len(scs), "source": "regenerated from the committed pilot cell list"}
    if set(entries) != set(HISTORICAL):
        raise ConfirmationError("historical set list differs from HISTORICAL")
    families = [f for fams in build.families.values() for f, _ in fams]
    return entries, families, {"sets": checked, "reference_families": len(families)}


# ----- 2-4 confirmation set ----------------------------------------------------------------------------------
def _scenario_entry(g, j: int) -> dict[str, Any]:
    return {"lineage_id": g.lineage_id, "family_id": g.family.family_id, "lineage_index": j,
            "lineage_seed": g.lineage_seed, "attempt": g.attempt, "scenario_id": g.scenario.scenario_id,
            "scenario_seed": g.scenario.seed, "scenario_hash": g.scenario.content_hash(), "core_hash": g.core_hash(),
            "target": g.target, "label": g.label, "diagnostics": g.diagnostics, "probe": g.probe}


def build_manifest(entries: Mapping[str, Sequence[Mapping[str, Any]]], reference: Sequence[Family],
                   spec: GeneratorSpec | None = None, families: int = FAMILIES, lineages: int = LINEAGES,
                   max_slots: int = MAX_SLOTS) -> dict[str, Any]:
    spec = spec or GeneratorSpec.from_dict(DEFAULT_SPEC)
    taken = {k: {e[k] for es in entries.values() for e in es} for k in ID_KEYS}
    owner = {(k, e[k]): n for n, es in entries.items() for e in es for k in ID_KEYS}
    hashes = {f.topology_hash for f in reference}
    accepted: list[Family] = []

    def reject(f: Family) -> str | None:
        if f.topology_hash in hashes:
            return "duplicate_family"
        for g in (*reference, *accepted):
            if A.near_duplicate(f.signature, g.signature)[0] or A.wall_near_duplicate(f.signature, g.signature)[0]:
                return "near_duplicate_family"
        return None

    fams, scen, excluded, failures = [], [], [], []
    rej: Counter = Counter()
    lrej: Counter = Counter()
    slot = 0
    while len(accepted) < families:
        if slot >= max_slots:
            raise ConfirmationError(f"confirmation pool not filled: {len(accepted)}/{families} in {slot} slots")
        fseed = derive_seed(MASTER_SEED, SEED_NAMESPACE, "family", slot)
        try:
            fam, reasons = generate_family(spec, fseed, reject)
        except GenerationError as exc:
            failures.append({"pool_slot": slot, "family_seed": fseed, "reasons": dict(sorted(exc.reasons.items()))})
            rej.update(exc.reasons)
            slot += 1
            continue
        rej.update(reasons)
        got, why = [], None
        for j in range(lineages):
            lseed = derive_seed(fam.family_seed, "lineage", j)
            try:
                g, lr = generate_lineage(spec, fam, lseed)
            except GenerationError as exc:
                lrej.update(exc.reasons)
                why = {"reason": "lineage_generation_failed", "lineage_index": j, "lineage_seed": lseed,
                       "target": sample_target(spec, lseed), "attempts": exc.attempts,
                       "reasons": dict(sorted(exc.reasons.items()))}
                break
            lrej.update(lr)
            e = _scenario_entry(g, j)
            hit = next(((k, e[k]) for k in ID_KEYS if e[k] in taken[k]), None)
            if hit:
                why = {"reason": "isolation_collision", "lineage_index": j, "key": hit[0],
                       "historical_set": owner[hit]}
                break
            got.append(e)
        if why:
            excluded.append({"pool_slot": slot, "family_id": fam.family_id, "family_seed": fam.family_seed, **why})
        else:
            accepted.append(fam)
            hashes.add(fam.topology_hash)
            fams.append({**fam.summary(), "pool_slot": slot, "rejections": dict(sorted(reasons.items()))})
            scen.extend(got)
        slot += 1
    body = {"manifest_version": S.MANIFEST_VERSION, "split": SPLIT, "plan_id": NAMESPACE, "plan_hash": plan_hash(),
            "plan": plan(), "generator_revision": GENERATOR_REVISION, "spec_hash": spec.spec_hash(),
            "families": fams, "scenarios": scen, "failed_lineages": [x for x in excluded
                                                                     if x["reason"] == "lineage_generation_failed"],
            "excluded_families": excluded,
            "pool": {"slots_used": slot, "family_failures": failures, "family_rejections": dict(sorted(rej.items()))},
            "lineage_rejections": dict(sorted(lrej.items()))}
    return {**body, "manifest_hash": digest(body)}


def isolation_report(manifest: Mapping[str, Any], entries: Mapping[str, Sequence[Mapping[str, Any]]],
                     provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Overlap of every isolation key with every historical set (all must be empty)."""
    mine = {k: {s[k] for s in manifest["scenarios"]} for k in ID_KEYS}
    over = {n: {k: sorted(map(str, mine[k] & {e[k] for e in es})) for k in ID_KEYS} for n, es in entries.items()}
    fam = Counter(s["family_id"] for s in manifest["scenarios"])
    body = {"conf_version": CONF_VERSION, "namespace": NAMESPACE, "manifest_hash": manifest["manifest_hash"],
            "historical": dict(provenance), "historical_sets": sorted(entries), "overlap": over,
            "families": len(fam), "lineages_per_family": sorted(set(fam.values())),
            "scenarios": len(manifest["scenarios"]),
            "unique": {k: len(mine[k]) == len(manifest["scenarios"]) for k in ("lineage_id", "scenario_hash",
                                                                               "core_hash", "scenario_seed")},
            "not_checked": ["blind_test (secret seeds; family IDs and seeds derive from another root)",
                            "near-duplicate geometry against pilot maps (exact core hash only)"]}
    body["pass"] = (not any(v for o in over.values() for v in o.values()) and all(body["unique"].values())
                    and body["families"] == FAMILIES and body["lineages_per_family"] == [LINEAGES])
    return {**body, "audit_hash": digest(body)}


def materialize(manifest: Mapping[str, Any]) -> list[tuple[dict[str, Any], Scenario]]:
    """Regenerate every confirmation scenario and check its hashes (the manifest holds no scenarios)."""
    trust.check_manifest(manifest)
    if manifest.get("plan_id") != NAMESPACE or manifest.get("plan_hash") != plan_hash():
        raise ConfirmationError("manifest is not the preregistered p1-09-selection@1 plan")
    spec = GeneratorSpec.from_dict(manifest["plan"]["spec"])
    fams = {}
    out = []
    for e in manifest["scenarios"]:
        fseed = next(f["family_seed"] for f in manifest["families"] if f["family_id"] == e["family_id"])
        if e["family_id"] not in fams:
            fams[e["family_id"]] = generate_family(spec, fseed)[0]
        g, _ = generate_lineage(spec, fams[e["family_id"]], e["lineage_seed"])
        if (g.scenario.content_hash(), g.core_hash(), g.lineage_id) != (e["scenario_hash"], e["core_hash"],
                                                                         e["lineage_id"]):
            raise ConfirmationError(f"{e['scenario_id']}: regenerated scenario differs from the manifest")
        out.append((dict(e), g.scenario))
    return out


# ----- compute preflight -------------------------------------------------------------------------------------
def cpu_estimate(scenarios: int = FAMILIES * LINEAGES) -> dict[str, Any]:
    """Upper CPU estimate per teacher from committed DEV CPU (one worker per teacher)."""
    per: dict[str, float] = {}
    for name in ("p1-04-t1-dev-v1.json", "p1-05-t2-dev-v1.json", "p1-06-t3-dev-v1.json"):
        rep = json.loads((ev.REPO / "data/baselines" / name).read_text(encoding="utf-8"))
        for t in rep.get("policies", {}):
            ms = [(r[t].get("timing") or {}).get("policy_cpu_ms") for r in rep["rows"]]
            if t in reg.CANDIDATES and ms and all(x is not None for x in ms):
                per[t] = max(per.get(t, 0.0), sum(ms) / 1000)
    t4 = json.loads((ev.REPO / T4_V12_DEV_CPU_REPORT).read_text(encoding="utf-8"))
    # Conservative: the whole upper DEV comparison estimate (three T4 variants + T1 + evaluator) stands for T4.
    per["T4"] = float(t4["estimate"]["upper"]["comparison_cpu_s_total"])
    scale = scenarios / DEV_SCENARIOS * CPU_SAFETY
    est = {t: (None if v is None else round(v * scale, 1)) for t, v in sorted(per.items())}
    return {"basis": "committed DEV policy CPU x scenarios/48 x 1.5 (T4: whole upper DEV comparison estimate of "
                     "the v1.2 pilot)",
            "dev_cpu_s": per, "estimated_cpu_s": est,
            "total_s": None if None in est.values() else round(sum(est.values()), 1)}


def compute_preflight(protocol: Mapping[str, Any]) -> dict[str, Any]:
    est = cpu_estimate()
    elig = [c for c in reg.CANDIDATES if protocol["eligibility"][c]["status"] == reg.ELIGIBLE]
    cm = protocol["compute"]
    missing = [c for c in elig if est["estimated_cpu_s"].get(c) is None]
    total = sum(est["estimated_cpu_s"].get(c) or 0.0 for c in elig)
    over = [c for c in elig if (est["estimated_cpu_s"].get(c) or 0.0) > cm["replacement_cpu_s_cap"]]
    return {**est, "eligible": elig, "missing_estimates": missing, "eligible_total_s": round(total, 1),
            "cpu_s_cap": cm["cpu_s_cap"], "replacement_cpu_s_cap": cm["replacement_cpu_s_cap"],
            "over_replacement_cap": over, "pass": not missing and total <= cm["cpu_s_cap"]}


# ----- runner ------------------------------------------------------------------------------------------------
def result_of(rec: Mapping[str, Any], config_hash: str) -> dict[str, Any]:
    t = rec["timing"]
    return {"config_hash": config_hash, "casualties": rec["casualties"], "evacuated": rec["evacuated"],
            "unresolved": rec["unresolved"], "people": rec["people"], "failure": rec["failure"],
            "final_transition_hash": rec["final_transition_hash"], "policy_cpu_s": t["policy_cpu_ms_total"] / 1000,
            "max_decision_cpu_ms": t["decision_cpu_ms_max"]}


def run_teacher(identity: Mapping[str, str], seed: int | None, scenarios: Sequence[tuple[dict, Scenario]]
                ) -> dict[str, Any]:
    """One teacher on every scenario: a fresh frozen instance per episode, simulator seed = scenario seed."""
    cls = reg.teacher_class(identity["name"], identity["version"])
    out, errors = {}, []
    for e, s in scenarios:
        pol = cls() if seed is None else cls(seed=seed)
        if pol.seed != seed or pol.identity.config_hash != identity["config_hash"]:
            raise ConfirmationError(f"{identity['name']}: seed or identity differs from the protocol")
        try:
            rec, _ = ev.run_episode(s, pol)
        except Exception as exc:          # recorded, never repaired; the gate stops on errors
            errors.append({"scenario_id": e["scenario_id"], "error": type(exc).__name__})
            continue
        out[e["scenario_id"]] = result_of(rec, identity["config_hash"])
    return {"results": out, "errors": errors}


def _worker(args: tuple) -> tuple[str, dict[str, Any]]:
    name, identity, seed, manifest = args
    t0 = time.perf_counter()
    r = run_teacher(identity, seed, materialize(manifest))
    return name, {**r, "wall_s": round(time.perf_counter() - t0, 1)}


def run_rows(protocol: Mapping[str, Any], manifest: Mapping[str, Any], *, processes: int | None = None
             ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Matched rows for every eligible teacher (one process per teacher; T2 dominates the wall time)."""
    elig = protocol["eligibility"]
    seeds = protocol["confirmation"]["policy_seeds"]
    jobs = [(c, dict(elig[c]["identity"]), seeds.get(c), dict(manifest)) for c in reg.CANDIDATES
            if elig[c]["status"] == reg.ELIGIBLE]
    with mp.get_context("spawn").Pool(processes or len(jobs)) as pool:
        done = dict(pool.map(_worker, jobs))
    rows, errors = [], []
    for e in manifest["scenarios"]:
        res = {c: done[c]["results"][e["scenario_id"]] for c in done if e["scenario_id"] in done[c]["results"]}
        rows.append({"scenario_id": e["scenario_id"], "family_id": e["family_id"],
                     "scenario_hash": e["scenario_hash"], "results": res})
    for c, d in done.items():
        errors.extend({"teacher": c, **x} for x in d["errors"])
    return rows, errors, {c: d["wall_s"] for c, d in done.items()}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def run_preconditions(repo: Path, protocol_ref: Mapping[str, Any], manifest_ref: Mapping[str, Any], *,
                      trusted_ref: str = "origin/main", evidence_path: Path = EVIDENCE_PATH) -> dict[str, Any]:
    """Every check before any teacher outcome exists. Raise ConfirmationError with all reasons."""
    from . import selection as sel
    reasons: list[str] = []
    protocol = manifest = None
    try:
        protocol, _ = trust.load_trusted_json(repo, protocol_ref, trusted_ref=trusted_ref)
        sel.verify_frozen_protocol(protocol)
    except (ValueError, OSError, subprocess.SubprocessError, sel.ProtocolError) as e:
        reasons.append(f"protocol: {e}")
    try:
        manifest, _ = trust.load_trusted_json(repo, manifest_ref, trusted_ref=trusted_ref)
        trust.check_manifest(manifest)
    except (ValueError, OSError, subprocess.SubprocessError) as e:
        reasons.append(f"manifest: {e}")
    if protocol and manifest:
        c = protocol["confirmation"]
        if (c["namespace"], c["manifest_hash"]) != (NAMESPACE, manifest["manifest_hash"]):
            reasons.append("manifest is not the one preregistered in the frozen protocol")
        pre = compute_preflight(protocol)
        if not pre["pass"]:
            reasons.append(f"compute preflight failed: {pre}")
    head = _git(repo, "rev-parse", "HEAD")
    if _git(repo, "status", "--porcelain", "--untracked-files=no"):
        reasons.append("working tree has uncommitted changes")
    if protocol and not trust.is_ancestor(repo, protocol_ref.get("commit", ""), head):
        reasons.append("code commit does not descend from the protocol freeze commit")
    if reg.check_pins() or reg.check_registry():
        reasons.append("teacher identity drift")
    if evidence_path.exists():
        reasons.append("evidence already exists: exactly one confirmatory run per identity")
    if reasons:
        raise ConfirmationError("; ".join(reasons))
    return {"protocol": protocol, "manifest": manifest, "code_commit": head}


def run(repo: Path, protocol_ref: Mapping[str, Any], manifest_ref: Mapping[str, Any], *,
        trusted_ref: str = "origin/main", evidence_path: Path = EVIDENCE_PATH) -> dict[str, Any]:
    from . import selection as sel
    pre = run_preconditions(repo, protocol_ref, manifest_ref, trusted_ref=trusted_ref, evidence_path=evidence_path)
    p, m = pre["protocol"], pre["manifest"]
    rows, errors, wall = run_rows(p, m)
    body = {"evidence_version": sel.EVIDENCE_VERSION, "evidence_class": sel.PROSPECTIVE,
            "protocol_id": p["protocol_id"], "protocol_hash": p["protocol_hash"],
            "protocol_commit": protocol_ref["commit"], "code_commit": pre["code_commit"], "manifest": m,
            "rows": rows, "complete": not errors and all(len(r["results"]) == len(wall) for r in rows),
            "errors": errors}
    ev_doc = {**body, "evidence_hash": digest(body)}
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    with open(evidence_path, "x", encoding="utf-8", newline="\n") as f:        # never overwrite
        f.write(json.dumps(ev_doc, indent=1, sort_keys=True) + "\n")
    return {"evidence": str(evidence_path), "teacher_wall_s": wall, "complete": body["complete"]}


# ----- CLI ---------------------------------------------------------------------------------------------------
def _write_once(path: Path, doc: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, indent=1, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ConfirmationError(f"{path} exists with other content; it is frozen")
        return
    path.write_text(text, encoding="utf-8", newline="\n")


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m ember_v1.teacher.confirmation")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="build and audit the confirmation manifest (no teacher runs)")
    r = sub.add_parser("run", help="the one confirmatory run (needs the frozen protocol on the trusted ref)")
    r.add_argument("--protocol-commit", required=True)
    r.add_argument("--trusted-ref", default="origin/main")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        entries, fams, prov = historical_sets()
        m = build_manifest(entries, fams)
        audit = isolation_report(m, entries, prov)
        if not audit["pass"]:
            raise ConfirmationError("isolation audit failed")
        _write_once(MANIFEST_PATH, m)
        _write_once(AUDIT_PATH, audit)
        print(json.dumps({"manifest_hash": m["manifest_hash"], "audit_hash": audit["audit_hash"],
                          "families": audit["families"], "scenarios": audit["scenarios"],
                          "excluded_families": len(m["excluded_families"]), "slots_used": m["pool"]["slots_used"]}))
    else:
        rel = lambda p: p.relative_to(ev.REPO).as_posix()      # noqa: E731
        ref = {"kind": "git", "commit": a.protocol_commit}
        print(json.dumps(run(ev.REPO, {**ref, "path": rel(OUT_DIR / "protocol.json")},
                             {**ref, "path": rel(MANIFEST_PATH)}, trusted_ref=a.trusted_ref)))


if __name__ == "__main__":
    main(sys.argv[1:])
