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
3 pilot maps: a generated family is excluded as a whole (recorded, never
  repaired) if its map duplicates or nearly duplicates any historical T2/T3/T4
  pilot map: same symmetry-invariant topology hash, the generator near-duplicate
  rules, or the same rules without the scale and aspect gates over all 8 grid
  symmetries (`pilot_match`). The next slot is then tried.
4 lineages: `derive_seed(family_seed, "lineage", j)`, j < LINEAGES. A family is
  excluded (recorded, never repaired) if a lineage fails to generate or any
  lineage shares a family ID, lineage ID, scenario hash, core-geometry hash or
  simulator seed with a historical set. The next slot is then tried.
5 stop after FAMILIES complete families or MAX_SLOTS slots (then the build fails).

The manifest holds identities, seeds, hashes and diagnostics only; teacher
outcomes are never computed here. The runner (`run`) refuses to start unless
the frozen protocol and this manifest are committed on the trusted ref, the
checkout is clean (also no untracked or unexpected ignored file), the code
descends from the protocol commit, and no attempt marker or evidence exists.
It then writes the attempt marker atomically, before the first scenario.
Every eligible teacher runs every scenario once from a fresh instance, with the
simulator seed of the scenario (matched across teachers) and its preregistered
policy seed, in a supervised worker process. The supervisor enforces the total
CPU, per-decision CPU and wall-time limits live and stops every worker at the
first breach; the evidence is then recorded as incomplete with the stage.
"""
from __future__ import annotations

import argparse
import functools
import json
import multiprocessing as mp
import os
import platform
import queue as queue_mod
import re
import subprocess
import sys
import threading
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

CONF_VERSION = "p1-09-confirmation-set@1.1.0"
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
ATTEMPT_NAME, LOG_NAME = "attempt.json", "attempt_log.jsonl"     # beside the evidence file
ATTEMPT_VERSION = "p1-09-attempt@1.0.0"
PILOT_SETS = ("t2-pilot@1", "t3-pilot@1", "t4-pilot@1", "t4-pilot@2", "t4-pilot@3", "t4-explore@v12")
# Supervisor (protocol compute limits are enforced live; these are mechanics, not budgets).
POLL_S = 0.25              # supervisor poll interval
GRACE_S = 5.0              # terminate -> kill grace per worker
STALL_S = 60.0             # worker heartbeat age that counts as unresponsive (only without OS CPU readings)
# Ignored paths a confirmation checkout may contain (caches and build output only). Every other
# untracked, modified or ignored path stops the run. Documented in docs/P1_SELECTION_CONFIRMATION_SET.md.
CLEAN_ALLOWLIST = (re.compile(r"\.venv/"), re.compile(r"\.pytest_cache/"), re.compile(r"(?:[^/]+/)*__pycache__/"),
                   re.compile(r"src/[A-Za-z0-9_.-]+\.egg-info/"), re.compile(r"build/"), re.compile(r"dist/"))
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
                      "pilot_maps": "exclude the family if its map has the topology hash of, or is a near or wall "
                                    "near duplicate (frozen generator thresholds, also without scale and aspect "
                                    "gates over all 8 grid symmetries) of, any map of the pilot sets",
                      "pilot_sets": list(PILOT_SETS),
                      "lineage": "exclude the family if any lineage fails or collides with a historical set",
                      "isolation_keys": list(ID_KEYS), "historical_sets": list(HISTORICAL)}}


def plan_hash() -> str:
    return digest({"generator_revision": GENERATOR_REVISION, "plan": plan()})


# ----- 1 historical sets -------------------------------------------------------------------------------------
def _pilot_entry(ns: str, s: Scenario) -> dict[str, Any]:
    return {"family_id": f"{ns}/{s.scenario_id}", "lineage_id": f"{ns}/{s.scenario_id}",
            "scenario_hash": s.content_hash(), "core_hash": A.core_hash(s), "scenario_seed": s.seed}


@functools.lru_cache(maxsize=1)
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


class PilotMap:
    """Static map of one historical pilot scenario (walls and exits only; no outcome)."""

    __slots__ = ("set", "scenario_id", "topology_hash", "signature")

    def __init__(self, ns: str, s: Scenario) -> None:
        ex = [p for _, p in s.exits]
        self.set, self.scenario_id = ns, s.scenario_id
        self.topology_hash = A.topology_hash(s.height, s.width, s.obstacles, ex)
        self.signature = A.MapSignature(s.height, s.width, s.obstacles, ex)


def pilot_maps() -> list[PilotMap]:
    """Every map of the historical T2/T3/T4 pilot sets (regenerated from the committed cell lists)."""
    scs = _pilot_scenarios()
    if set(scs) != set(PILOT_SETS):
        raise ConfirmationError("pilot set list differs from PILOT_SETS")
    return [PilotMap(ns, s) for ns in PILOT_SETS for s in scs[ns]]


def _free_distance(a: A.MapSignature, b: A.MapSignature) -> tuple[float, float]:
    """(wall distance, exit distance) at the best of all 8 symmetries, without scale or aspect gates."""
    n = len(a.mats[0])
    return min((sum(abs(x - y) for x, y in zip(a.mats[0], b.mats[k])) / n, A._chamfer(a.exits[0], b.exits[k]))
               for k in range(8))


def pilot_match(fam: Family, pm: PilotMap) -> str | None:
    """The rule by which a family map duplicates a pilot map, or None."""
    a, b = fam.signature, pm.signature
    if fam.topology_hash == pm.topology_hash:
        return "topology_duplicate"
    if A.near_duplicate(a, b)[0]:
        return "near_duplicate"
    if A.wall_near_duplicate(a, b)[0]:
        return "wall_near_duplicate"
    n = len(a.mats[0])
    walls = [sum(abs(x - y) for x, y in zip(a.mats[0], b.mats[k])) / n for k in range(8)]
    if abs(a.walls_frac - b.walls_frac) <= A.NEAR_WALL_MAD:
        if any(w <= A.NEAR_WALL_MAD and A._chamfer(a.exits[0], b.exits[k]) <= A.NEAR_EXIT
               for k, w in enumerate(walls)):
            return "scale_free_near_duplicate"
        if min(a.walls_frac, b.walls_frac) >= A.NEAR_WALL_MIN_FRAC and min(walls) <= A.NEAR_WALL_MAD:
            return "scale_free_wall_near_duplicate"
    return None


def pilot_screen(manifest: Mapping[str, Any], pilots: Sequence[PilotMap]) -> dict[str, Any]:
    """Audit view: every accepted family against every pilot map, with the nearest map per family."""
    spec = GeneratorSpec.from_dict(manifest["plan"]["spec"])
    hits, nearest, comparable = [], {}, 0
    for f in manifest["families"]:
        fam = generate_family(spec, f["family_seed"])[0]
        best = None
        for pm in pilots:
            comparable += A._comparable(fam.signature, pm.signature) is not None
            rule = pilot_match(fam, pm)
            if rule:
                hits.append({"family_id": f["family_id"], "pilot_set": pm.set, "scenario_id": pm.scenario_id,
                             "rule": rule})
            wall, ex = _free_distance(fam.signature, pm.signature)
            if best is None or (wall, ex) < best[:2]:
                best = (wall, ex, pm)
        nearest[f["family_id"]] = {"pilot_set": best[2].set, "scenario_id": best[2].scenario_id,
                                   "scale_free_wall_distance": round(best[0], 6),
                                   "exit_distance": round(best[1], 6)} if best else None
    by_set = Counter(pm.set for pm in pilots)
    return {"pilot_maps": len(pilots), "maps_per_set": dict(sorted(by_set.items())),
            "unique_topologies": len({pm.topology_hash for pm in pilots}),
            "pairs": len(pilots) * len(manifest["families"]), "pairs_within_scale_and_aspect_gates": comparable,
            "thresholds": {"wall_mad": A.NEAR_WALL_MAD, "exit_chamfer": A.NEAR_EXIT,
                           "wall_rule_min_fraction": A.NEAR_WALL_MIN_FRAC, "scale": A.NEAR_SCALE,
                           "aspect": A.NEAR_ASPECT},
            "symmetries": "all 8 rotations/reflections of the grid",
            "hits": hits, "nearest_by_family": nearest}


# ----- 2-4 confirmation set ----------------------------------------------------------------------------------
def _scenario_entry(g, j: int) -> dict[str, Any]:
    return {"lineage_id": g.lineage_id, "family_id": g.family.family_id, "lineage_index": j,
            "lineage_seed": g.lineage_seed, "attempt": g.attempt, "scenario_id": g.scenario.scenario_id,
            "scenario_seed": g.scenario.seed, "scenario_hash": g.scenario.content_hash(), "core_hash": g.core_hash(),
            "target": g.target, "label": g.label, "diagnostics": g.diagnostics, "probe": g.probe}


def build_manifest(entries: Mapping[str, Sequence[Mapping[str, Any]]], reference: Sequence[Family],
                   spec: GeneratorSpec | None = None, families: int = FAMILIES, lineages: int = LINEAGES,
                   max_slots: int = MAX_SLOTS, pilots: Sequence[PilotMap] = ()) -> dict[str, Any]:
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
        hit = next(((pm, r) for pm in pilots if (r := pilot_match(fam, pm))), None)
        if hit:
            why = {"reason": "pilot_map_duplicate", "rule": hit[1], "pilot_set": hit[0].set,
                   "pilot_scenario_id": hit[0].scenario_id}
        for j in range(0 if why else lineages):
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
                     provenance: Mapping[str, Any], pilots: Sequence[PilotMap]) -> dict[str, Any]:
    """Overlap of every isolation key with every historical set and pilot-map duplicates (all must be empty)."""
    mine = {k: {s[k] for s in manifest["scenarios"]} for k in ID_KEYS}
    over = {n: {k: sorted(map(str, mine[k] & {e[k] for e in es})) for k in ID_KEYS} for n, es in entries.items()}
    fam = Counter(s["family_id"] for s in manifest["scenarios"])
    body = {"conf_version": CONF_VERSION, "namespace": NAMESPACE, "manifest_hash": manifest["manifest_hash"],
            "historical": dict(provenance), "historical_sets": sorted(entries), "overlap": over,
            "families": len(fam), "lineages_per_family": sorted(set(fam.values())),
            "scenarios": len(manifest["scenarios"]),
            "unique": {k: len(mine[k]) == len(manifest["scenarios"]) for k in ("lineage_id", "scenario_hash",
                                                                               "core_hash", "scenario_seed")},
            "pilot_geometry": pilot_screen(manifest, pilots),
            "not_checked": ["blind_test (secret seeds; family IDs and seeds derive from another root)",
                            "pilot maps: dynamic content (people, ignitions, fire model) is compared by exact "
                            "core hash only; the geometry screen compares walls and exits",
                            "geometry screen: coarse 16x16 wall-fraction grid; local edits below the wall "
                            "threshold, crops, translations and partial overlaps are not detected"]}
    body["pass"] = (not any(v for o in over.values() for v in o.values()) and all(body["unique"].values())
                    and body["families"] == FAMILIES and body["lineages_per_family"] == [LINEAGES]
                    and body["pilot_geometry"]["pilot_maps"] > 0 and not body["pilot_geometry"]["hits"])
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
def result_of(rec: Mapping[str, Any], config_hash: str, budget_limited: int = 0) -> dict[str, Any]:
    t = rec["timing"]
    return {"config_hash": config_hash, "casualties": rec["casualties"], "evacuated": rec["evacuated"],
            "unresolved": rec["unresolved"], "people": rec["people"], "failure": rec["failure"],
            "final_transition_hash": rec["final_transition_hash"], "policy_cpu_s": t["policy_cpu_ms_total"] / 1000,
            "max_decision_cpu_ms": t["decision_cpu_ms_max"], "decisions": rec["decisions"],
            "budget_limited_decisions": budget_limited}


def budget_limited(diagnostics: Any) -> bool:
    """True if a teacher reports that its frozen internal search budget cut this decision short.

    T1 `budget_exhausted`; T2/T3/T4 `budget.limit_hit`; T3 `status == "budget_limited"`. Descriptive only:
    the internal budgets are part of each frozen configuration, not a failure.
    """
    if not isinstance(diagnostics, Mapping):
        return False
    b = diagnostics.get("budget")
    return (bool(diagnostics.get("budget_exhausted")) or diagnostics.get("status") == "budget_limited"
            or (isinstance(b, Mapping) and bool(b.get("limit_hit"))))


# Shared slot of one worker: [worker-reported process CPU s, heartbeat (time.time), decision start CPU s or -1].
CPU, BEAT, DEC = 0, 1, 2


class _Probe:
    """One fresh policy, unchanged (same observation in, same decision out).

    Marks each decision's start CPU in the shared slot for the supervisor (if any) and counts the
    decisions whose frozen internal budget was hit.
    """

    def __init__(self, policy: Any, slot: Any = None) -> None:
        self._policy, self._slot, self.budget_limited = policy, slot, 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._policy, name)

    def decide(self, obs: Any) -> Any:
        if self._slot is not None:
            self._slot[DEC] = time.process_time()
        try:
            d = self._policy.decide(obs)
        finally:
            if self._slot is not None:
                self._slot[DEC] = -1.0
        self.budget_limited += budget_limited(getattr(d, "diagnostics", None))
        return d


def run_teacher(identity: Mapping[str, str], seed: int | None, scenarios: Sequence[tuple[dict, Scenario]], *,
                slot: Any = None, on_result: Any = None, on_error: Any = None) -> dict[str, Any]:
    """One teacher on every scenario: a fresh frozen instance per episode, simulator seed = scenario seed."""
    cls = reg.teacher_class(identity["name"], identity["version"])
    out, errors = {}, []
    for e, s in scenarios:
        pol = cls() if seed is None else cls(seed=seed)
        if pol.seed != seed or pol.identity.config_hash != identity["config_hash"]:
            raise ConfirmationError(f"{identity['name']}: seed or identity differs from the protocol")
        probe = _Probe(pol, slot)
        try:
            rec, _ = ev.run_episode(s, probe)
        except Exception as exc:          # recorded, never repaired; the gate stops on errors
            err = {"scenario_id": e["scenario_id"], "error": type(exc).__name__}
            errors.append(err)
            if on_error:
                on_error(err)
            continue
        out[e["scenario_id"]] = result_of(rec, identity["config_hash"], probe.budget_limited)
        if on_result:
            on_result(e["scenario_id"], out[e["scenario_id"]])
    return {"results": out, "errors": errors}


def _heartbeat(slot: Any) -> None:
    """Worker thread: publish own CPU and liveness; exit the worker if the supervisor process is gone."""
    parent = mp.parent_process()
    while True:
        slot[CPU], slot[BEAT] = time.process_time(), time.time()
        if parent is not None and not parent.is_alive():
            os._exit(4)
        time.sleep(0.2)


def _worker_main(name: str, identity: Mapping[str, str], seed: int | None, manifest: Mapping[str, Any],
                 q: Any, slot: Any, loader: Any = None) -> None:
    """Supervised worker: materialize the manifest, run one teacher, stream every result to the supervisor.

    `loader` (default `materialize`) turns the manifest into scenarios; tests and microbenchmarks pass a
    loader of synthetic non-confirmation scenarios.
    """
    threading.Thread(target=_heartbeat, args=(slot,), daemon=True).start()
    code = 0
    try:
        q.put(("start", name, os.getpid()))
        r = run_teacher(identity, seed, (loader or materialize)(manifest), slot=slot,
                        on_result=lambda sid, res: q.put(("result", name, sid, res)),
                        on_error=lambda err: q.put(("error", name, err)))
        q.put(("done", name, time.process_time(), len(r["results"])))
    except BaseException as exc:          # noqa: BLE001 - reported to the supervisor, never repaired
        q.put(("fatal", name, f"{type(exc).__name__}: {exc}"[:500]))
        code = 3
    q.close()
    q.join_thread()
    sys.exit(code)


def process_cpu_s(pid: int) -> float | None:
    """User + system CPU seconds of a live process as the OS accounts them, or None if unreadable."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.OpenProcess.restype = wintypes.HANDLE
            k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            k32.GetProcessTimes.argtypes = (wintypes.HANDLE,) + (ctypes.POINTER(wintypes.FILETIME),) * 4
            k32.CloseHandle.argtypes = (wintypes.HANDLE,)
            h = k32.OpenProcess(0x1000, False, pid)          # PROCESS_QUERY_LIMITED_INFORMATION
            if not h:
                return None
            try:
                ft = [wintypes.FILETIME() for _ in range(4)]
                if not k32.GetProcessTimes(h, *(ctypes.byref(f) for f in ft)):
                    return None
                return sum((f.dwHighDateTime << 32 | f.dwLowDateTime) / 1e7 for f in ft[2:])
            finally:
                k32.CloseHandle(h)
        with open(f"/proc/{pid}/stat", "rb") as f:
            rest = f.read().rsplit(b")", 1)[1].split()
        return (int(rest[11]) + int(rest[12])) / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError, AttributeError):
        return None


class AttemptLog:
    """Append-only JSON-lines log of one attempt; every line is flushed and synced to disk."""

    def __init__(self, path: Path | None) -> None:
        self.path = path

    def __call__(self, event: str, **fields: Any) -> None:
        if self.path is None:
            return
        line = json.dumps({"t_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": event, **fields},
                          sort_keys=True)
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())


def _stop_all(procs: Mapping[str, Any]) -> dict[str, int | None]:
    """Terminate every live worker, kill it after the grace period, and reap it."""
    for p in procs.values():
        if p.is_alive():
            p.terminate()
    for p in procs.values():
        p.join(GRACE_S)
        if p.is_alive():
            p.kill()
            p.join(GRACE_S)
    return {n: p.exitcode for n, p in procs.items()}


def supervise(jobs: Sequence[tuple[str, Mapping[str, str], int | None, Mapping[str, Any]]],
              limits: Mapping[str, float], *, log: AttemptLog | None = None, target: Any = None,
              loader: Any = None, poll_s: float = POLL_S) -> dict[str, Any]:
    """Run one worker per job under live limits. Stop all workers at the first breach or failure.

    limits: cpu_s_cap (sum of worker process CPU), per_decision_cpu_ms_cap, wall_s_cap.
    Returns results, errors and the supervision report; `stage` is "completed" only if every worker
    reported done and exited 0 with no breach.
    """
    log = log or AttemptLog(None)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs, slots = {}, {}
    names = [j[0] for j in jobs]
    results: dict[str, dict[str, Any]] = {n: {} for n in names}
    errors: list[dict[str, Any]] = []
    done: dict[str, Any] = {}
    cpu = {n: 0.0 for n in names}
    via = {n: "worker_reported" for n in names}
    peak_decision_ms = {n: 0.0 for n in names}
    abort: dict[str, Any] | None = None
    t0 = time.monotonic()

    def handle(msg: tuple) -> None:
        nonlocal abort
        kind, n = msg[0], msg[1]
        if kind == "result":
            results[n][msg[2]] = msg[3]
            log("result", teacher=n, scenario_id=msg[2])
        elif kind == "error":
            errors.append({"teacher": n, **msg[2]})
            log("episode_error", teacher=n, **msg[2])
        elif kind == "done":
            done[n] = msg[3]
            cpu[n] = max(cpu[n], msg[2])
            log("worker_done", teacher=n, process_cpu_s=msg[2], results=msg[3])
        elif kind == "fatal":
            abort = abort or {"stage": "worker_error", "teacher": n, "error": msg[2]}
        elif kind == "start":
            log("worker_start", teacher=n, pid=msg[2])

    def drain(timeout: float) -> None:
        try:
            handle(q.get(timeout=timeout))
            while True:
                handle(q.get_nowait())
        except queue_mod.Empty:
            pass

    try:
        for name, identity, seed, manifest in jobs:
            slots[name] = ctx.Array("d", [0.0, 0.0, -1.0], lock=False)
            procs[name] = ctx.Process(target=target or _worker_main, name=f"confirmation-{name}", daemon=True,
                                      args=(name, dict(identity), seed, dict(manifest), q, slots[name], loader))
            procs[name].start()
        log("supervisor_start", limits=dict(limits), pids={n: p.pid for n, p in procs.items()})
        while abort is None:
            drain(poll_s)
            if abort is not None:
                break
            for n, p in procs.items():
                alive = p.is_alive()
                ext = process_cpu_s(p.pid) if alive else None
                start = slots[n][DEC]
                if ext is not None:
                    via[n] = "os"
                now_cpu = ext if ext is not None else slots[n][CPU]
                cpu[n] = max(cpu[n], now_cpu, slots[n][CPU])
                if alive and start >= 0:
                    d_ms = (now_cpu - start) * 1000
                    peak_decision_ms[n] = max(peak_decision_ms[n], d_ms)
                    if d_ms > limits["per_decision_cpu_ms_cap"]:
                        abort = {"stage": "per_decision_cpu_cap", "teacher": n, "decision_cpu_ms": round(d_ms, 1),
                                 "scenarios_done": len(results[n])}
                        break
                if alive and ext is None and slots[n][BEAT] > 0 and time.time() - slots[n][BEAT] > STALL_S:
                    abort = {"stage": "worker_unresponsive", "teacher": n}
                    break
            if abort is not None:
                break
            total = sum(cpu.values())
            if total > limits["cpu_s_cap"]:
                abort = {"stage": "total_cpu_cap", "process_cpu_s": round(total, 1)}
                break
            if time.monotonic() - t0 > limits["wall_s_cap"]:
                abort = {"stage": "wall_time_cap", "wall_s": round(time.monotonic() - t0, 1)}
                break
            exited = [n for n, p in procs.items() if not p.is_alive()]
            if len(exited) == len(procs):
                drain(0.05)
                missing = [n for n in names if n not in done or procs[n].exitcode != 0]
                if missing and abort is None:
                    abort = {"stage": "worker_crashed", "teachers": missing,
                             "exit_codes": {n: procs[n].exitcode for n in missing}}
                break
            for n in exited:
                if procs[n].exitcode != 0 and abort is None:
                    drain(0.05)
                    abort = abort or {"stage": "worker_crashed", "teacher": n, "exit_code": procs[n].exitcode}
    except KeyboardInterrupt:
        abort = abort or {"stage": "interrupted"}
    except Exception as exc:              # noqa: BLE001 - supervisor fault: fail closed, keep evidence
        abort = abort or {"stage": "supervisor_error", "error": f"{type(exc).__name__}: {exc}"[:500]}
    finally:
        codes = _stop_all(procs)
        q.close()
        q.cancel_join_thread()
    stage = "completed" if abort is None else abort["stage"]
    if abort:
        log("abort", **abort)
    report = {"limits": {k: limits[k] for k in ("cpu_s_cap", "per_decision_cpu_ms_cap", "wall_s_cap")},
              "stage": stage, "abort": abort, "wall_s": round(time.monotonic() - t0, 3),
              "process_cpu_s": {n: round(v, 3) for n, v in cpu.items()},
              "total_process_cpu_s": round(sum(cpu.values()), 3), "cpu_measurement": via,
              "peak_live_decision_cpu_ms": {n: round(v, 1) for n, v in peak_decision_ms.items()},
              "exit_codes": codes, "results_received": {n: len(results[n]) for n in names}}
    log("supervisor_end", stage=stage, exit_codes=codes)
    return {"results": results, "errors": errors, "report": report}


def run_rows(protocol: Mapping[str, Any], manifest: Mapping[str, Any], *, log: AttemptLog | None = None
             ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Matched rows for every eligible teacher (one supervised process per teacher; T2 dominates wall time)."""
    elig = protocol["eligibility"]
    seeds = protocol["confirmation"]["policy_seeds"]
    jobs = [(c, dict(elig[c]["identity"]), seeds.get(c), dict(manifest)) for c in reg.CANDIDATES
            if elig[c]["status"] == reg.ELIGIBLE]
    out = supervise(jobs, protocol["compute"], log=log)
    rows = []
    for e in manifest["scenarios"]:
        res = {c: out["results"][c][e["scenario_id"]] for c, *_ in jobs if e["scenario_id"] in out["results"][c]}
        rows.append({"scenario_id": e["scenario_id"], "family_id": e["family_id"],
                     "scenario_hash": e["scenario_hash"], "results": res})
    return rows, out["errors"], out["report"]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def tree_problems(repo: Path) -> list[str]:
    """Every modified, staged, untracked or non-allowlisted ignored path of the checkout (empty if clean)."""
    raw = subprocess.run(["git", "-C", str(repo), "status", "--porcelain=v1", "-z", "--untracked-files=all",
                          "--ignored=matching"], capture_output=True, check=True).stdout.decode("utf-8", "replace")
    items, out, i = raw.split("\0"), [], 0
    while i < len(items):
        e = items[i]
        i += 1
        if not e:
            continue
        xy, path = e[:2], e[3:]
        if xy[0] in "RC":
            i += 1                        # the rename/copy source path follows
        if xy == "!!":
            if not any(r.fullmatch(path) for r in CLEAN_ALLOWLIST):
                out.append(f"ignored {path}")
        elif xy == "??":
            out.append(f"untracked {path}")
        else:
            out.append(f"modified {path}")
    return out


def attempt_paths(evidence_path: Path) -> tuple[Path, Path]:
    return evidence_path.with_name(ATTEMPT_NAME), evidence_path.with_name(LOG_NAME)


def create_attempt_marker(path: Path, doc: Mapping[str, Any]) -> dict[str, Any]:
    """Create the attempt marker exclusively (atomic: fails if it exists) and sync it to disk."""
    body = {"attempt_version": ATTEMPT_VERSION, **doc}
    out = {**body, "attempt_hash": digest(body)}
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        f = open(path, "x", encoding="utf-8", newline="\n")
    except FileExistsError:
        raise ConfirmationError(f"{path} exists: a confirmatory attempt already started") from None
    with f:
        f.write(json.dumps(out, indent=1, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return out


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
    dirty = tree_problems(repo)
    if dirty:
        reasons.append(f"working tree not clean (uncommitted, untracked or unexpected ignored files): "
                       f"{dirty[:20]}{' ...' if len(dirty) > 20 else ''}")
    if protocol and not trust.is_ancestor(repo, protocol_ref.get("commit", ""), head):
        reasons.append("code commit does not descend from the protocol freeze commit")
    if reg.check_pins() or reg.check_registry():
        reasons.append("teacher identity drift")
    if evidence_path.exists():
        reasons.append("evidence already exists: exactly one confirmatory run per identity")
    marker, log = attempt_paths(evidence_path)
    if marker.exists() or log.exists():
        reasons.append("a previous confirmatory attempt exists (attempt marker or log; possibly interrupted): "
                       "re-execution needs a separately documented scientific authorization")
    if reasons:
        raise ConfirmationError("; ".join(reasons))
    return {"protocol": protocol, "manifest": manifest, "code_commit": head}


def run(repo: Path, protocol_ref: Mapping[str, Any], manifest_ref: Mapping[str, Any], *,
        trusted_ref: str = "origin/main", evidence_path: Path = EVIDENCE_PATH) -> dict[str, Any]:
    """The one confirmatory attempt. Preconditions, then the marker, then supervised teachers, then evidence.

    Any limit breach, worker failure or interruption after the marker yields evidence with
    `complete: false` and the failure stage; the marker stays and blocks every later attempt.
    """
    from . import selection as sel
    pre = run_preconditions(repo, protocol_ref, manifest_ref, trusted_ref=trusted_ref, evidence_path=evidence_path)
    p, m = pre["protocol"], pre["manifest"]
    marker_path, log_path = attempt_paths(evidence_path)
    limits = {k: p["compute"][k] for k in ("cpu_s_cap", "per_decision_cpu_ms_cap", "wall_s_cap")}
    marker = create_attempt_marker(marker_path, {
        "protocol_id": p["protocol_id"], "protocol_hash": p["protocol_hash"], "protocol_commit": protocol_ref["commit"],
        "code_commit": pre["code_commit"], "manifest_hash": m["manifest_hash"], "limits": limits,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": {"system": platform.system(), "machine": platform.machine(),
                     "python": platform.python_version(), "cpu_count": os.cpu_count()},
        "rerun_policy": "one attempt only; a later attempt needs a separately documented scientific "
                        "authorization (a new protocol version)"})
    log = AttemptLog(log_path)
    log("attempt_start", attempt_hash=marker["attempt_hash"])
    interrupted = False
    try:
        rows, errors, sup = run_rows(p, m, log=log)
    except BaseException as exc:          # noqa: BLE001 - fail closed: record, never retry
        interrupted = isinstance(exc, KeyboardInterrupt)
        rows, errors = [], []
        sup = {"limits": limits, "stage": "interrupted" if interrupted else "supervisor_error",
               "abort": {"stage": "interrupted" if interrupted else "supervisor_error",
                         "error": f"{type(exc).__name__}: {exc}"[:500]}}
    eligible = [c for c in reg.CANDIDATES if p["eligibility"][c]["status"] == reg.ELIGIBLE]
    complete = (sup["stage"] == "completed" and sup["abort"] is None and not errors
                and len(rows) == len(m["scenarios"]) and all(set(r["results"]) == set(eligible) for r in rows))
    body = {"evidence_version": sel.EVIDENCE_VERSION, "evidence_class": sel.PROSPECTIVE,
            "protocol_id": p["protocol_id"], "protocol_hash": p["protocol_hash"],
            "protocol_commit": protocol_ref["commit"], "code_commit": pre["code_commit"], "manifest": m,
            "rows": rows, "complete": complete, "errors": errors, "attempt": marker, "supervision": sup}
    ev_doc = {**body, "evidence_hash": digest(body)}
    with open(evidence_path, "x", encoding="utf-8", newline="\n") as f:        # never overwrite
        f.write(json.dumps(ev_doc, indent=1, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    log("evidence_written", complete=complete, stage=sup["stage"], evidence_hash=ev_doc["evidence_hash"])
    if interrupted:
        raise KeyboardInterrupt
    return {"evidence": str(evidence_path), "complete": complete, "stage": sup["stage"],
            "supervision": {k: v for k, v in sup.items() if k != "limits"}}


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
        pilots = pilot_maps()
        m = build_manifest(entries, fams, pilots=pilots)
        audit = isolation_report(m, entries, prov, pilots)
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
        out = run(ev.REPO, {**ref, "path": rel(OUT_DIR / "protocol.json")}, {**ref, "path": rel(MANIFEST_PATH)},
                  trusted_ref=a.trusted_ref)
        print(json.dumps(out))
        if not out["complete"]:
            raise SystemExit(f"confirmation run INCOMPLETE at stage {out['stage']}; evidence recorded, no rerun")


if __name__ == "__main__":
    main(sys.argv[1:])
