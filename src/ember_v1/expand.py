"""P1-18 Stage 2: 1,200 NEW scenarios from 300 NEW families, isolated from every historical set.

Two strata, each with its own seed namespace and generator spec (frozen here, before any teacher runs):
* core   (N <= 120): 225 families x 4 lineages = 900 scenarios. DEFAULT_SPEC with targets moved toward
  `difficult` (feasible 4 / difficult 5 / unreachable 1).
* stress (N 121-200): 75 families x 4 lineages = 300 scenarios. Larger grids, crowd count [121, 200].

Isolation (the P1-09 confirmation rules, `teacher.confirmation`, extended):
* family: reject a topology duplicate, near duplicate or wall near duplicate of any p1-02-v1 family (TRAIN, DEV,
  calibration, locked test), any p1-09-selection@1 family, any family accepted earlier in this plan (both strata);
* pilot maps: exclude a family whose map duplicates any historical pilot map (8 symmetries, scale-free rules);
* scenario: exclude the family if any lineage collides on family_id / lineage_id / scenario_hash / core_hash /
  scenario_seed with p1-02-v1 (all splits), the pilot sets or p1-09-selection@1.
EMBERBench hidden data and its blind seeds are not read (not available; they derive from another root).

Partitions are by family and fixed here, before any teacher outcome exists: inside each stratum the families are
ranked by sha256(NAMESPACE/partition|family_id); the first ranks are TRAIN, then val, then diagnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .gen import analysis as A
from .gen import splits as S
from .gen.generator import Family, GenerationError, generate_family, generate_lineage, sample_target
from .gen.rng import derive_seed
from .gen.spec import DEFAULT_SPEC, GENERATOR_REVISION, GeneratorSpec, _merge
from .sim import Scenario
from .sim.canonical import digest
from .teacher import confirmation as C

VERSION = "p1-18-expand@1.0.0"
NAMESPACE = "p1-18-expand@1"
MASTER_SEED = 20261110
LINEAGES = 4
PARTITIONS = ("train", "val", "diagnostic")
SELECTION_MANIFEST = "data/selection/p1-09-selection@1/manifest.json"
OUT_DIR = "data/expansion/p1-18-expand@1"
TARGET_MIX = {"feasible": 4, "difficult": 5, "unreachable": 1}
STRATA: dict[str, dict[str, Any]] = {
    "core": {"families": 225, "partition": {"train": 165, "val": 20, "diagnostic": 40},
             "overrides": {"targets": {"mix": TARGET_MIX}}},
    "stress": {"families": 75, "partition": {"train": 55, "val": 5, "diagnostic": 15},
               "overrides": {"grid": {"short_side": [16, 34], "aspect_pct": [100, 220]},
                             "obstacles": {"density_pct": [0, 15]},
                             "crowd": {"count": [121, 200], "density_permille": [150, 400]},
                             "targets": {"mix": TARGET_MIX}}},
}
HISTORICAL = (*C.HISTORICAL, "p1-09-selection@1")


class ExpandError(RuntimeError):
    """The expansion set cannot be built or audited under its preregistered rules."""


def spec_of(stratum: str) -> GeneratorSpec:
    return GeneratorSpec.from_dict(_merge(DEFAULT_SPEC, STRATA[stratum]["overrides"]))


def plan() -> dict[str, Any]:
    return {"plan_id": NAMESPACE, "version": VERSION, "master_seed": MASTER_SEED, "lineages_per_family": LINEAGES,
            "strata": {k: {"families": v["families"], "partition": v["partition"], "seed_namespace":
                           f"{NAMESPACE}/{k}/seeds", "max_slots": 4 * v["families"],
                           "spec": json.loads(spec_of(k).canonical), "spec_hash": spec_of(k).spec_hash()}
                       for k, v in STRATA.items()},
            "partition_rule": f"rank by sha256('{NAMESPACE}/partition|' + family_id) within the stratum; "
                              "first ranks train, then val, then diagnostic",
            "rules": C.plan()["rules"] | {"historical_sets": list(HISTORICAL), "stratum_crowd": "a family is rejected if its free main region cannot hold the stratum minimum N, and excluded if any lineage N falls outside the stratum crowd range",
                                          "families_also_rejected_against": ["p1-09-selection@1 families",
                                                                             "families accepted earlier here"]}}


def plan_hash() -> str:
    return digest({"generator_revision": GENERATOR_REVISION, "plan": plan()})


# ----- historical sets -----------------------------------------------------------------------------------------
def historical(repo: Path) -> tuple[dict[str, list[dict[str, Any]]], list[Family], dict[str, Any]]:
    """(entries per historical set, reference families, provenance): P1-09's sets + the selection set."""
    entries, families, prov = C.historical_sets()
    sel = json.loads((repo / SELECTION_MANIFEST).read_text(encoding="utf-8"))
    if sel.get("manifest_hash") != digest({k: v for k, v in sel.items() if k != "manifest_hash"}):
        raise ExpandError("p1-09-selection@1 manifest hash does not verify")
    entries["p1-09-selection@1"] = [{k: e[k] for k in C.ID_KEYS} for e in sel["scenarios"]]
    spec = GeneratorSpec.from_dict(sel["plan"]["spec"])
    sel_fams = [generate_family(spec, f["family_seed"])[0] for f in sel["families"]]
    if [f.family_id for f in sel_fams] != [f["family_id"] for f in sel["families"]]:
        raise ExpandError("regenerated p1-09-selection@1 families differ from the manifest")
    prov["sets"]["p1-09-selection@1"] = {"manifest_hash": sel["manifest_hash"], "source": "committed manifest"}
    prov["reference_families"] = len(families) + len(sel_fams)
    if set(entries) != set(HISTORICAL):
        raise ExpandError("historical set list differs from HISTORICAL")
    return entries, [*families, *sel_fams], prov


# ----- manifest ------------------------------------------------------------------------------------------------
def _stratum(stratum: str, entries: Mapping[str, Sequence[Mapping[str, Any]]], reference: list[Family],
             pilots: Sequence[C.PilotMap], taken: dict[str, set], owner: dict, *, families: int | None = None
             ) -> dict[str, Any]:
    """Fill one stratum (P1-09 `build_manifest` rules). Accepted families are appended to `reference`."""
    cfg = plan()["strata"][stratum]
    spec, ns = spec_of(stratum), cfg["seed_namespace"]
    want = cfg["families"] if families is None else families
    hashes = {f.topology_hash for f in reference}
    accepted: list[Family] = []

    n_lo, n_hi = spec.get("crowd", "count")

    def reject(f: Family) -> str | None:
        if f.topology_hash in hashes:
            return "duplicate_family"
        if len(f.main_region) - len(f.exits) <= n_lo:      # a lineage could not hold the stratum minimum N
            return "crowd_capacity_below_stratum"
        for g in reference:
            if A.near_duplicate(f.signature, g.signature)[0] or A.wall_near_duplicate(f.signature, g.signature)[0]:
                return "near_duplicate_family"
        return None

    fams, scen, excluded, failures = [], [], [], []
    rej: Counter = Counter()
    lrej: Counter = Counter()
    slot = 0
    while len(accepted) < want:
        if slot >= cfg["max_slots"]:
            raise ExpandError(f"{stratum} pool not filled: {len(accepted)}/{want} in {slot} slots")
        fseed = derive_seed(MASTER_SEED, ns, "family", slot)
        try:
            fam, reasons = generate_family(spec, fseed, reject)
        except GenerationError as exc:
            failures.append({"pool_slot": slot, "family_seed": fseed, "reasons": dict(sorted(exc.reasons.items()))})
            rej.update(exc.reasons)
            slot += 1
            continue
        rej.update(reasons)
        got, why = [], None
        hit = next(((pm, r) for pm in pilots if (r := C.pilot_match(fam, pm))), None)
        if hit:
            why = {"reason": "pilot_map_duplicate", "rule": hit[1], "pilot_set": hit[0].set,
                   "pilot_scenario_id": hit[0].scenario_id}
        for j in range(0 if why else LINEAGES):
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
            if not n_lo <= len(g.scenario.people) <= n_hi:
                why = {"reason": "crowd_out_of_stratum", "lineage_index": j, "n_people": len(g.scenario.people)}
                break
            e = C._scenario_entry(g, j)
            hit2 = next(((k, e[k]) for k in C.ID_KEYS if e[k] in taken[k]), None)
            if hit2:
                why = {"reason": "isolation_collision", "lineage_index": j, "key": hit2[0],
                       "historical_set": owner.get(hit2, "this plan")}
                break
            got.append({**e, "stratum": stratum, "n_people": len(g.scenario.people)})
        if why:
            excluded.append({"pool_slot": slot, "family_id": fam.family_id, "family_seed": fam.family_seed, **why})
        else:
            accepted.append(fam)
            reference.append(fam)
            hashes.add(fam.topology_hash)
            for e in got:
                for k in C.ID_KEYS:
                    taken[k].add(e[k])
            fams.append({**fam.summary(), "stratum": stratum, "pool_slot": slot,
                         "rejections": dict(sorted(reasons.items()))})
            scen.extend(got)
        slot += 1
    part = partition([f["family_id"] for f in fams], STRATA[stratum]["partition"] if families is None
                     else _scaled(STRATA[stratum]["partition"], want))
    for f in fams:
        f["partition"] = part[f["family_id"]]
    for e in scen:
        e["partition"] = part[e["family_id"]]
    return {"families": fams, "scenarios": scen, "excluded_families": excluded,
            "pool": {"slots_used": slot, "family_failures": failures, "family_rejections": dict(sorted(rej.items()))},
            "lineage_rejections": dict(sorted(lrej.items()))}


def _scaled(sizes: Mapping[str, int], total: int) -> dict[str, int]:
    """Partition sizes for a smaller test plan (same order; at least one per partition when possible)."""
    out = {k: max(1, total * v // sum(sizes.values())) for k, v in sizes.items()}
    out["train"] = total - out["val"] - out["diagnostic"]
    return out


def partition(family_ids: Sequence[str], sizes: Mapping[str, int]) -> dict[str, str]:
    if len(family_ids) != sum(sizes.values()) or len(set(family_ids)) != len(family_ids):
        raise ExpandError("partition sizes do not match the family list")
    ranked = sorted(family_ids, key=lambda f: hashlib.sha256(f"{NAMESPACE}/partition|{f}".encode()).hexdigest())
    out, i = {}, 0
    for p in PARTITIONS:
        for f in ranked[i:i + sizes[p]]:
            out[f] = p
        i += sizes[p]
    return out


def build_manifest(entries: Mapping[str, Sequence[Mapping[str, Any]]], reference: Sequence[Family],
                   pilots: Sequence[C.PilotMap], *, families: Mapping[str, int] | None = None) -> dict[str, Any]:
    """Both strata, core first. `families` (tests only) shrinks the per-stratum family counts."""
    taken = {k: {e[k] for es in entries.values() for e in es} for k in C.ID_KEYS}
    owner = {(k, e[k]): n for n, es in entries.items() for e in es for k in C.ID_KEYS}
    ref = list(reference)
    strata = {s: _stratum(s, entries, ref, pilots, taken, owner,
                          families=None if families is None else families[s]) for s in STRATA}
    body = {"manifest_version": S.MANIFEST_VERSION, "plan_id": NAMESPACE, "plan_hash": plan_hash(), "plan": plan(),
            "generator_revision": GENERATOR_REVISION, "test_scale": families is not None,
            "families": [f for s in strata.values() for f in s["families"]],
            "scenarios": [e for s in strata.values() for e in s["scenarios"]],
            "excluded_families": {k: s["excluded_families"] for k, s in strata.items()},
            "pool": {k: s["pool"] for k, s in strata.items()},
            "lineage_rejections": {k: s["lineage_rejections"] for k, s in strata.items()}}
    return {**body, "manifest_hash": digest(body)}


def isolation_report(manifest: Mapping[str, Any], entries: Mapping[str, Sequence[Mapping[str, Any]]],
                     provenance: Mapping[str, Any], pilots: Sequence[C.PilotMap]) -> dict[str, Any]:
    """Overlap of every isolation key with every historical set, uniqueness, counts and the pilot-map screen."""
    mine = {k: {s[k] for s in manifest["scenarios"]} for k in C.ID_KEYS}
    over = {n: {k: sorted(map(str, mine[k] & {e[k] for e in es})) for k in C.ID_KEYS} for n, es in entries.items()}
    fam = Counter(s["family_id"] for s in manifest["scenarios"])
    hits = []
    for f in manifest["families"]:
        g = generate_family(spec_of(f["stratum"]), f["family_seed"])[0]
        hits += [{"family_id": f["family_id"], "pilot_set": pm.set, "scenario_id": pm.scenario_id, "rule": r}
                 for pm in pilots if (r := C.pilot_match(g, pm))]
    n = Counter((s["stratum"], "le120" if s["n_people"] <= 120 else "121-200") for s in manifest["scenarios"])
    parts = Counter((f["stratum"], f["partition"]) for f in manifest["families"])
    body = {"version": VERSION, "namespace": NAMESPACE, "manifest_hash": manifest["manifest_hash"],
            "historical": dict(provenance), "historical_sets": sorted(entries), "overlap": over,
            "families": len(fam), "lineages_per_family": sorted(set(fam.values())),
            "scenarios": len(manifest["scenarios"]),
            "unique": {k: len(mine[k]) == len(manifest["scenarios"]) for k in ("lineage_id", "scenario_hash",
                                                                               "core_hash", "scenario_seed")},
            "n_people_bins": {f"{a}/{b}": c for (a, b), c in sorted(n.items())},
            "family_partitions": {f"{a}/{b}": c for (a, b), c in sorted(parts.items())},
            "pilot_maps": len(pilots), "pilot_hits": hits,
            "not_checked": ["EMBERBench blind/hidden sets (not read; their seeds derive from another root)",
                            "geometry screen: coarse 16x16 wall-fraction grid (P1-09 limits apply)"]}
    want_n = {"core/le120": STRATA["core"]["families"] * LINEAGES,
              "stress/121-200": STRATA["stress"]["families"] * LINEAGES}
    body["pass"] = (not any(v for o in over.values() for v in o.values()) and all(body["unique"].values())
                    and body["lineages_per_family"] == [LINEAGES] and not hits and len(pilots) > 0
                    and (manifest["test_scale"] or body["n_people_bins"] == want_n))
    return {**body, "audit_hash": digest(body)}


def materialize(manifest: Mapping[str, Any], *, partitions: Sequence[str] = PARTITIONS
                ) -> list[tuple[dict[str, Any], Scenario]]:
    """Regenerate the scenarios (of the given partitions) and check every hash against the manifest."""
    body = {k: v for k, v in manifest.items() if k != "manifest_hash"}
    if manifest.get("manifest_hash") != digest(body) or manifest.get("plan_hash") != plan_hash():
        raise ExpandError("manifest is not the preregistered p1-18-expand@1 plan")
    seeds = {f["family_id"]: f for f in manifest["families"]}
    fams: dict[str, Family] = {}
    out = []
    for e in manifest["scenarios"]:
        if e["partition"] not in partitions:
            continue
        f = seeds[e["family_id"]]
        if e["family_id"] not in fams:
            fams[e["family_id"]] = generate_family(spec_of(f["stratum"]), f["family_seed"])[0]
        g, _ = generate_lineage(spec_of(f["stratum"]), fams[e["family_id"]], e["lineage_seed"])
        if (g.scenario.content_hash(), g.core_hash(), g.lineage_id) != (e["scenario_hash"], e["core_hash"],
                                                                         e["lineage_id"]):
            raise ExpandError(f"{e['scenario_id']}: regenerated scenario differs from the manifest")
        out.append((dict(e), g.scenario))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ember_v1.expand")
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    out = a.out or a.repo / OUT_DIR
    entries, reference, prov = historical(a.repo)
    pilots = C.pilot_maps()
    m = build_manifest(entries, reference, pilots)
    audit = isolation_report(m, entries, prov, pilots)
    out.mkdir(parents=True, exist_ok=True)
    for name, doc in (("manifest.json", m), ("isolation_audit.json", audit)):
        (out / name).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"manifest_hash": m["manifest_hash"], "audit_pass": audit["pass"],
                      "families": audit["families"], "scenarios": audit["scenarios"],
                      "n_people_bins": audit["n_people_bins"], "family_partitions": audit["family_partitions"]}))
    return 0 if audit["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
