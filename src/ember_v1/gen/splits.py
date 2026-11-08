"""Immutable TRAIN / DEV / CALIBRATION / LOCKED TEST manifests and the leakage audit.

Build order (all deterministic from the plan):
  1 family pool: pool slot i uses `derive_seed(master_seed, plan_id, "family", i)`.
    A candidate whose map has the same topology hash as, or is a near-duplicate
    of, an accepted family is rejected (`duplicate_family`, `near_duplicate_family`),
    so no two families in the whole pool, in any split, look alike.
  2 split assignment by family: families sorted by (archetype, family_id) and
    dealt to splits in proportion to the plan counts (archetype-stratified).
    All lineages of a family stay in the family's split.
  3 lineages: `lineages_per_family` per family, `derive_seed(family_seed, "lineage", j)`.

A manifest holds identities, seeds, hashes, labels and diagnostics, never the
scenarios. `materialize` regenerates each scenario and checks its hashes.

Locked test. The index file records only a commitment (hash and counts) for
the locked-test manifest. `write_split_files` writes the locked manifest to a
separate sealed directory, outside the public split directory.
`load_manifest` refuses the locked split unless the caller passes
`purpose=LOCKED_PURPOSE`. This is a procedural guard, not secrecy: the plan
is public, so anyone can regenerate the locked split. See the design note.
"""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ember_v1.sim import Scenario
from ember_v1.sim.canonical import digest

from . import analysis as A
from .generator import (Family, GenerationError, Generated, build_family, build_lineage, generate_family,
                        generate_lineage, sample_target)
from .rng import derive_seed
from .spec import DEFAULT_SPEC, GENERATOR_REVISION, GeneratorSpec

SPLITS = ("train", "dev", "calibration", "locked_test")
PUBLIC_SPLITS = ("train", "dev", "calibration")
LOCKED = "locked_test"
LOCKED_PURPOSE = "final_locked_evaluation"
MANIFEST_VERSION = "1.0"
INDEX_VERSION = "1.0"

DEFAULT_PLAN: dict[str, Any] = {
    "plan_id": "p1-02-v1", "master_seed": 20261106, "spec": DEFAULT_SPEC,
    "families": {"train": 64, "dev": 16, "calibration": 16, "locked_test": 24},
    "lineages_per_family": 3,
}


class ManifestError(ValueError):
    """Manifest hash mismatch, wrong version, regeneration mismatch or overwrite attempt."""


class LockedSplitError(PermissionError):
    """The locked test split was requested without the final-evaluation purpose."""


class PlanError(RuntimeError):
    """The family pool could not be filled within its bounded slots."""


@dataclass(frozen=True)
class SplitPlan:
    plan_id: str
    master_seed: int
    spec: GeneratorSpec
    families: tuple[tuple[str, int], ...]
    lineages_per_family: int

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SplitPlan":
        if not isinstance(d, Mapping) or set(d) != {"plan_id", "master_seed", "spec", "families",
                                                     "lineages_per_family"}:
            raise ValueError("plan must contain plan_id, master_seed, spec, families, lineages_per_family")
        pid = d["plan_id"]
        if not isinstance(pid, str) or not pid or len(pid) > 64 or not all(ch.isalnum() or ch in "-_." for ch in pid):
            raise ValueError("plan_id must be 1-64 characters of [A-Za-z0-9._-]")
        seed = d["master_seed"]
        if type(seed) is not int or not 0 <= seed < 2 ** 64:
            raise ValueError("master_seed must be an integer in [0, 2**64)")
        fam = d["families"]
        if not isinstance(fam, Mapping) or set(fam) != set(SPLITS):
            raise ValueError(f"families must give a count for each of {list(SPLITS)}")
        if any(type(fam[s]) is not int or not 1 <= fam[s] <= 100000 for s in SPLITS):
            raise ValueError("family counts must be integers in [1, 100000]")
        lpf = d["lineages_per_family"]
        if type(lpf) is not int or not 1 <= lpf <= 1000:
            raise ValueError("lineages_per_family must be an integer in [1, 1000]")
        return cls(pid, seed, GeneratorSpec.from_dict(d["spec"]), tuple((s, fam[s]) for s in SPLITS), lpf)

    @classmethod
    def default(cls) -> "SplitPlan":
        return cls.from_dict(DEFAULT_PLAN)

    def to_dict(self) -> dict[str, Any]:
        return {"plan_id": self.plan_id, "master_seed": self.master_seed, "spec": self.spec.to_dict(),
                "families": dict(self.families), "lineages_per_family": self.lineages_per_family}

    def plan_hash(self) -> str:
        return digest({"generator_revision": GENERATOR_REVISION, "plan": self.to_dict()})


@dataclass
class SplitBuild:
    plan: SplitPlan
    families: dict[str, list[tuple[Family, dict[str, Any]]]] = field(default_factory=dict)
    scenarios: dict[str, list[Generated]] = field(default_factory=dict)
    lineage_rejections: dict[str, dict[str, int]] = field(default_factory=dict)
    failed_lineages: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    lineage_index: dict[str, int] = field(default_factory=dict)
    pool: dict[str, Any] = field(default_factory=dict)


def _assign(families: list[Family], counts: Sequence[tuple[str, int]]) -> dict[str, list[Family]]:
    total = sum(n for _, n in counts)
    out: dict[str, list[Family]] = {s: [] for s, _ in counts}
    ordered = sorted(families, key=lambda f: (f.archetype, f.family_id))
    for k, fam in enumerate(ordered):
        best = max((n * (k + 1) / total - len(out[s]), -i, s) for i, (s, n) in enumerate(counts)
                   if len(out[s]) < n)
        out[best[2]].append(fam)
    return out


def fill_family_pool(spec: GeneratorSpec, total: int, seed_of: Callable[[int], int],
                     reference: Sequence[Family] = (), wall_rule: bool = False
                     ) -> tuple[list[Family], dict[str, dict[str, Any]], dict[str, Any]]:
    """Accept `total` families from pool slots 0, 1, ... (family seed `seed_of(slot)`).

    A candidate is rejected if its map duplicates or nearly duplicates a family
    in `reference` or an already accepted one. `wall_rule` adds the exit-free
    wall-geometry rule. The `p1-02-v1` plan uses no reference and no wall rule.
    """
    pool: list[Family] = []
    hashes: set[str] = {f.topology_hash for f in reference}
    fam_meta: dict[str, dict[str, Any]] = {}
    pool_rejections: Counter = Counter()
    failures = []
    slot = 0

    def reject(f: Family) -> str | None:
        if f.topology_hash in hashes:
            return "duplicate_family"
        for g in (*reference, *pool):
            if A.near_duplicate(f.signature, g.signature)[0] or (
                    wall_rule and A.wall_near_duplicate(f.signature, g.signature)[0]):
                return "near_duplicate_family"
        return None

    while len(pool) < total:
        if slot >= 3 * total:
            raise PlanError(f"family pool not filled: {len(pool)}/{total} in {slot} slots")
        fseed = seed_of(slot)
        try:
            fam, reasons = generate_family(spec, fseed, reject)
        except GenerationError as exc:
            failures.append({"pool_slot": slot, "family_seed": fseed, "reasons": dict(sorted(exc.reasons.items()))})
            pool_rejections.update(exc.reasons)
        else:
            pool.append(fam)
            hashes.add(fam.topology_hash)
            pool_rejections.update(reasons)
            fam_meta[fam.family_id] = {"pool_slot": slot, "rejections": dict(sorted(reasons.items()))}
        slot += 1
    info = {"slots_used": slot, "family_failures": failures,
            "family_rejections": dict(sorted(pool_rejections.items()))}
    return pool, fam_meta, info


def generate_lineages(spec: GeneratorSpec, families: Sequence[Family], lineages_per_family: int
                      ) -> tuple[list[tuple[Generated, int]], list[dict[str, Any]], dict[str, int]]:
    """Lineages 0..n-1 of each family: (accepted with index, failed lineages, rejection counts)."""
    accepted: list[tuple[Generated, int]] = []
    failed: list[dict[str, Any]] = []
    rej: Counter = Counter()
    for fam in families:
        for j in range(lineages_per_family):
            lseed = derive_seed(fam.family_seed, "lineage", j)
            try:
                g, reasons = generate_lineage(spec, fam, lseed)
            except GenerationError as exc:
                failed.append({"family_id": fam.family_id, "lineage_index": j, "lineage_seed": lseed,
                               "target": sample_target(spec, lseed), "attempts": exc.attempts,
                               "reasons": dict(sorted(exc.reasons.items()))})
                rej.update(exc.reasons)
            else:
                accepted.append((g, j))
                rej.update(reasons)
    return accepted, failed, dict(sorted(rej.items()))


def build_splits(plan: SplitPlan) -> SplitBuild:
    spec = plan.spec
    total = sum(n for _, n in plan.families)
    pool, fam_meta, info = fill_family_pool(
        spec, total, lambda slot: derive_seed(plan.master_seed, plan.plan_id, "family", slot))
    build = SplitBuild(plan)
    build.pool = info
    for split, fams in _assign(pool, plan.families).items():
        build.families[split] = [(f, fam_meta[f.family_id]) for f in sorted(fams, key=lambda f: f.family_id)]
        accepted, build.failed_lineages[split], build.lineage_rejections[split] = generate_lineages(
            spec, [f for f, _ in build.families[split]], plan.lineages_per_family)
        build.scenarios[split] = [g for g, _ in accepted]
        build.lineage_index.update((g.lineage_id, j) for g, j in accepted)
    return build


# ----- manifests ------------------------------------------------------------

def _scenario_entry(g: Generated, lineage_index: int) -> dict[str, Any]:
    return {"lineage_id": g.lineage_id, "family_id": g.family.family_id,
            "lineage_index": lineage_index, "lineage_seed": g.lineage_seed,
            "attempt": g.attempt, "scenario_id": g.scenario.scenario_id, "scenario_seed": g.scenario.seed,
            "scenario_hash": g.scenario.content_hash(), "core_hash": g.core_hash(),
            "target": g.target, "label": g.label, "diagnostics": g.diagnostics, "probe": g.probe}


def _body(build: SplitBuild, split: str) -> dict[str, Any]:
    plan = build.plan
    fams = []
    for fam, meta in build.families[split]:
        entry = fam.summary()
        entry.update(meta)
        fams.append(entry)
    return {"manifest_version": MANIFEST_VERSION, "split": split, "plan_id": plan.plan_id,
            "plan_hash": plan.plan_hash(), "generator_revision": GENERATOR_REVISION,
            "spec_hash": plan.spec.spec_hash(), "families": fams,
            "scenarios": [_scenario_entry(g, build.lineage_index[g.lineage_id]) for g in build.scenarios[split]],
            "failed_lineages": build.failed_lineages[split],
            "lineage_rejections": build.lineage_rejections[split]}


def manifest(build: SplitBuild, split: str) -> dict[str, Any]:
    body = _body(build, split)
    return {**body, "manifest_hash": digest(body)}


def _bucket(v: float | int | None, edges: Sequence[int]) -> str:
    if v is None:
        return "none"
    for lo, hi in zip(edges, edges[1:]):
        if lo <= v < hi:
            return f"{lo}-{hi - 1}"
    return f">={edges[-1]}" if v >= edges[-1] else f"<{edges[0]}"


def coverage(m: Mapping[str, Any]) -> dict[str, Any]:
    """Counts per scenario dimension and per situation flag for one manifest."""
    c: dict[str, Counter] = {k: Counter() for k in (
        "archetype", "label", "difficulty_reasons", "unreachable_mechanism", "short_side", "aspect", "area",
        "wall_fraction_pct", "exits", "people", "crowd_placement", "ignitions", "ignition_placement",
        "fire_model", "spread_interval", "spread_probability", "exits_flammable", "enter_fire", "max_steps",
        "bottleneck_cells", "flags")}
    for s in m["scenarios"]:
        d = s["diagnostics"]
        h, w = d["grid"]
        c["archetype"][d["archetype"]] += 1
        c["label"][s["label"]] += 1
        for r in d["difficulty_reasons"]:
            c["difficulty_reasons"][r] += 1
        if d["unreachable_mechanism"]:
            c["unreachable_mechanism"][d["unreachable_mechanism"]] += 1
        c["short_side"][_bucket(min(h, w), [8, 12, 16, 20, 25, 31])] += 1
        c["aspect"][_bucket((100 * max(h, w)) // min(h, w), [100, 125, 150, 200, 251])] += 1
        c["area"][_bucket(h * w, [0, 150, 300, 600, 1200, 2400])] += 1
        c["wall_fraction_pct"][_bucket(d["wall_fraction_pct"], [0, 5, 15, 30, 50, 70])] += 1
        c["exits"][str(d["exits"])] += 1
        c["people"][_bucket(d["people"], [1, 2, 6, 16, 41, 81])] += 1
        c["crowd_placement"][d["crowd_placement"]] += 1
        c["ignitions"][str(d["ignitions"])] += 1
        c["ignition_placement"][d["ignition_placement"]] += 1
        c["fire_model"][d["fire_model"]] += 1
        c["spread_interval"][str(d["spread_interval"])] += 1
        p = d["spread_probability"]
        c["spread_probability"]["deterministic" if p is None else _bucket(round(100 * p), [0, 25, 50, 75, 101])] += 1
        c["exits_flammable"][str(d["exits_flammable"]).lower()] += 1
        c["enter_fire"][d["enter_fire"]] += 1
        c["max_steps"][_bucket(d["max_steps"], [0, 40, 80, 120, 200, 301])] += 1
        c["bottleneck_cells"][_bucket(d["bottleneck_cells"], [0, 1, 3, 10, 30])] += 1
        f = d["flags"]
        for k in ("stochastic_fire", "nearest_exit_overload", "bottleneck_on_route", "nearest_exit_cut_by_fire",
                  "heterogeneous_urgency"):
            c["flags"][k] += bool(f[k])
        c["flags"]["multiple_reachable_exits"] += f["people_with_multiple_reachable_exits"] > 0
        c["flags"]["isolated_people"] += f["isolated_people"] > 0
    return {"families": len(m["families"]), "scenarios": len(m["scenarios"]),
            "failed_lineages": len(m["failed_lineages"]),
            **{k: dict(sorted(v.items())) for k, v in c.items()}}


def index(build: SplitBuild, manifests: Mapping[str, Mapping[str, Any]], audit_report: Mapping[str, Any]) -> dict[str, Any]:
    plan = build.plan
    splits = {}
    for s in SPLITS:
        m = manifests[s]
        splits[s] = {"file": None if s == LOCKED else f"{s}.json", "sealed": s == LOCKED,
                     "manifest_hash": m["manifest_hash"], "families": len(m["families"]),
                     "scenarios": len(m["scenarios"]), "failed_lineages": len(m["failed_lineages"])}
    rej: Counter = Counter()
    for s in SPLITS:
        rej.update(build.lineage_rejections[s])
    body = {"index_version": INDEX_VERSION, "plan": plan.to_dict(), "plan_hash": plan.plan_hash(),
            "generator_revision": GENERATOR_REVISION, "spec_hash": plan.spec.spec_hash(), "splits": splits,
            "pool": build.pool, "lineage_rejections_all_splits": dict(sorted(rej.items())),
            "audit": audit_report, "coverage": {s: coverage(manifests[s]) for s in PUBLIC_SPLITS}}
    return {**body, "index_hash": digest(body)}


# ----- leakage audit --------------------------------------------------------

@dataclass(frozen=True)
class AuditItem:
    """One scenario as the audit sees it. The map comes from the scenario itself."""
    split: str
    family_id: str
    lineage_id: str
    seeds: tuple[int, ...]
    scenario: Scenario


def audit_items(build: SplitBuild) -> list[AuditItem]:
    return [AuditItem(s, g.family.family_id, g.lineage_id,
                      (g.family.family_seed, g.lineage_seed, g.scenario.seed), g.scenario)
            for s in SPLITS for g in build.scenarios[s]]


def audit(items: Sequence[AuditItem], max_listed: int = 10,
          splits: Sequence[str] = SPLITS) -> dict[str, Any]:
    """Cross-split isolation checks. Every check must have zero violations."""
    for it in items:
        if it.split not in splits:
            raise ValueError(f"unknown split {it.split!r}")
    checks: dict[str, list[Any]] = {k: [] for k in (
        "family_ids_disjoint", "lineage_ids_disjoint", "lineage_ids_unique", "seeds_disjoint",
        "scenario_seeds_unique", "scenario_hashes_disjoint", "core_hashes_disjoint", "topology_disjoint",
        "near_duplicate_maps", "family_single_split")}

    def cross(name: str, key) -> None:
        where: dict[Any, set[str]] = {}
        for it in items:
            for k in key(it):
                where.setdefault(k, set()).add(it.split)
        checks[name] = sorted([str(k), sorted(v)] for k, v in where.items() if len(v) > 1)

    cross("family_ids_disjoint", lambda it: [it.family_id])
    cross("lineage_ids_disjoint", lambda it: [it.lineage_id])
    cross("seeds_disjoint", lambda it: sorted(set(it.seeds) | {it.scenario.seed}))
    cross("scenario_hashes_disjoint", lambda it: [it.scenario.content_hash()])
    cross("core_hashes_disjoint", lambda it: [A.core_hash(it.scenario)])
    lin = Counter(it.lineage_id for it in items)
    checks["lineage_ids_unique"] = sorted(k for k, n in lin.items() if n > 1)
    seeds = Counter(it.scenario.seed for it in items)
    checks["scenario_seeds_unique"] = sorted(str(k) for k, n in seeds.items() if n > 1)
    fam_split: dict[str, set[str]] = {}
    for it in items:
        fam_split.setdefault(it.family_id, set()).add(it.split)
    checks["family_single_split"] = sorted(k for k, v in fam_split.items() if len(v) > 1)

    maps: dict[str, tuple[Scenario, set[str]]] = {}
    for it in items:
        s = it.scenario
        th = A.topology_hash(s.height, s.width, s.obstacles, [p for _, p in s.exits])
        maps.setdefault(th, (s, set()))[1].add(it.split)
    checks["topology_disjoint"] = sorted(th for th, (_, sp) in maps.items() if len(sp) > 1)
    sigs = [(th, sp, A.MapSignature(s.height, s.width, s.obstacles, [p for _, p in s.exits]))
            for th, (s, sp) in sorted(maps.items())]
    near = []
    for i in range(len(sigs)):
        for j in range(i + 1, len(sigs)):
            (ta, sa, ga), (tb, sb, gb) = sigs[i], sigs[j]
            if sa == sb and len(sa) == 1:
                continue        # same single split: not leakage
            hit, mad, ex = A.near_duplicate(ga, gb)
            if hit:
                near.append([ta, tb, sorted(sa | sb), round(mad, 4), round(ex, 4)])
    checks["near_duplicate_maps"] = near
    out = {k: {"passed": not v, "violations": len(v), "examples": v[:max_listed]} for k, v in checks.items()}
    return {"passed": all(c["passed"] for c in out.values()), "items": len(items),
            "distinct_maps": len(maps), "map_pairs_compared": len(sigs) * (len(sigs) - 1) // 2,
            "near_duplicate_thresholds": {"area_ratio": A.NEAR_SCALE, "aspect_ratio": A.NEAR_ASPECT,
                                          "wall_mad": A.NEAR_WALL_MAD, "exit_chamfer": A.NEAR_EXIT,
                                          "coarse_grid": A.SIG_K},
            "checks": out}


def wall_geometry_audit(items: Sequence[AuditItem], max_listed: int = 10) -> dict[str, Any]:
    """Supplementary cross-split check: one wall geometry (>= NEAR_WALL_MIN_FRAC walls) with any exits.

    `audit` matches maps with their exits, so one building reused with other
    exits passes it. This check ignores exits. It is not part of the frozen
    `p1-02-v1` index (that would change its hash); `build_all` and
    `verify_split_files` enforce it for every plan.
    """
    walls: dict[str, tuple[Scenario, set[str]]] = {}
    for it in items:
        s = it.scenario
        walls.setdefault(A.topology_hash(s.height, s.width, s.obstacles, []), (s, set()))[1].add(it.split)
    hits = [[th, sorted(sp), 0.0] for th, (s, sp) in sorted(walls.items())
            if len(sp) > 1 and len(s.obstacles) >= A.NEAR_WALL_MIN_FRAC * s.height * s.width]
    sigs = [(th, sp, A.MapSignature(s.height, s.width, s.obstacles, []))
            for th, (s, sp) in sorted(walls.items())]
    for i in range(len(sigs)):
        for j in range(i + 1, len(sigs)):
            (ta, sa, ga), (tb, sb, gb) = sigs[i], sigs[j]
            if sa == sb and len(sa) == 1:
                continue
            hit, mad = A.wall_near_duplicate(ga, gb)
            if hit:
                hits.append([f"{ta}|{tb}", sorted(sa | sb), round(mad, 4)])
    return {"passed": not hits, "violations": len(hits), "examples": hits[:max_listed],
            "rule": {"min_wall_fraction": A.NEAR_WALL_MIN_FRAC, "wall_mad": A.NEAR_WALL_MAD,
                     "area_ratio": A.NEAR_SCALE, "aspect_ratio": A.NEAR_ASPECT, "exits": "ignored"}}


# ----- files ----------------------------------------------------------------

def _dump(obj: Any) -> str:
    return json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"


def _write_once(path: Path, text: str) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != json.loads(text):
            raise ManifestError(f"{path} exists with different content; manifests are immutable")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def _inside(a: Path, b: Path) -> bool:
    a, b = a.resolve(), b.resolve()
    return a == b or b in a.parents


def build_all(plan: SplitPlan) -> tuple[SplitBuild, dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    build = build_splits(plan)
    manifests = {s: manifest(build, s) for s in SPLITS}
    items = audit_items(build)
    report = audit(items)
    if not report["passed"]:
        raise ManifestError(f"split audit failed: {[k for k, v in report['checks'].items() if not v['passed']]}")
    walls = wall_geometry_audit(items)
    if not walls["passed"]:
        raise ManifestError(f"wall geometry audit failed: {walls['examples']}")
    return build, manifests, report, index(build, manifests, report)


def write_split_files(plan: SplitPlan, public_dir: str | Path, sealed_dir: str | Path | None) -> dict[str, Any]:
    """Build, audit and write. The locked manifest goes only to `sealed_dir` (outside `public_dir`)."""
    public_dir = Path(public_dir)
    if sealed_dir is not None:
        sealed_dir = Path(sealed_dir)
        if _inside(sealed_dir, public_dir) or _inside(public_dir, sealed_dir):
            raise ManifestError("sealed_dir and public_dir must not contain each other")
    _, manifests, _, idx = build_all(plan)
    for s in PUBLIC_SPLITS:
        _write_once(public_dir / f"{s}.json", _dump(manifests[s]))
    _write_once(public_dir / "index.json", _dump(idx))
    if sealed_dir is not None:
        _write_once(sealed_dir / f"{LOCKED}.json", _dump(manifests[LOCKED]))
    return idx


def check_hash(data: Mapping[str, Any], field_name: str) -> None:
    body = {k: v for k, v in data.items() if k != field_name}
    if data.get(field_name) != digest(body):
        raise ManifestError(f"{field_name} does not match the content")


def load_index(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("index_version") != INDEX_VERSION:
        raise ManifestError("unsupported index_version")
    check_hash(data, "index_hash")
    return data


def load_manifest(path: str | Path, *, index_data: Mapping[str, Any] | None = None,
                  purpose: str | None = None) -> dict[str, Any]:
    """Load and verify one manifest. The locked split needs `purpose=LOCKED_PURPOSE`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("manifest_version") != MANIFEST_VERSION or data.get("split") not in SPLITS:
        raise ManifestError("unsupported manifest_version or split")
    if data["split"] == LOCKED and purpose != LOCKED_PURPOSE:
        raise LockedSplitError("the locked test split is sealed; it is only for the final locked evaluation")
    check_hash(data, "manifest_hash")
    if index_data is not None:
        want = index_data["splits"][data["split"]]["manifest_hash"]
        if want != data["manifest_hash"] or index_data["plan_hash"] != data["plan_hash"]:
            raise ManifestError("manifest is not the one committed in the index")
    return data


def plan_from_index(index_data: Mapping[str, Any]) -> SplitPlan:
    plan = SplitPlan.from_dict(index_data["plan"])
    if plan.plan_hash() != index_data["plan_hash"]:
        raise ManifestError("index plan does not match its plan_hash")
    return plan


def materialize(m: Mapping[str, Any], spec: GeneratorSpec, *,
                purpose: str | None = None) -> Iterator[tuple[dict[str, Any], Generated]]:
    """Regenerate every scenario of a loaded manifest and check its hashes and probe witness.

    Only public splits, and the locked split with `purpose=LOCKED_PURPOSE`. Any
    other split (for example a private blind manifest) is refused.
    """
    split = m.get("split")
    if split not in PUBLIC_SPLITS and not (split == LOCKED and purpose == LOCKED_PURPOSE):
        raise LockedSplitError(f"split {split!r} is sealed or unknown; it cannot be materialized here")
    return _regenerate(m, spec)


def _regenerate(m: Mapping[str, Any], spec: GeneratorSpec) -> Iterator[tuple[dict[str, Any], Generated]]:
    """The checked regeneration behind `materialize`, without the split gate (for `blind.py` only)."""
    check_hash(m, "manifest_hash")
    if spec.spec_hash() != m["spec_hash"] or m["generator_revision"] != GENERATOR_REVISION:
        raise ManifestError("manifest was made by a different spec or generator revision")
    fams = {f["family_id"]: f for f in m["families"]}
    cache: dict[str, Family] = {}
    for entry in m["scenarios"]:
        fid = entry["family_id"]
        if fid not in cache:
            f = fams[fid]
            fam = build_family(spec, f["family_seed"], f["attempt"])
            if not isinstance(fam, Family) or fam.family_id != fid or fam.topology_hash != f["topology_hash"]:
                raise ManifestError(f"family {fid} does not regenerate")
            cache[fid] = fam
        g = build_lineage(spec, cache[fid], entry["lineage_seed"], entry["attempt"], entry["target"])
        if (not isinstance(g, Generated) or g.lineage_id != entry["lineage_id"]
                or g.scenario.content_hash() != entry["scenario_hash"] or g.probe != entry["probe"]):
            raise ManifestError(f"lineage {entry['lineage_id']} does not regenerate")
        yield entry, g


def verify_split_files(public_dir: str | Path, sealed_dir: str | Path | None = None) -> dict[str, Any]:
    """Regenerate the plan in the index and compare every committed file with the rebuild.

    Checks: index hash, manifest hashes, index-to-manifest binding, and equal
    content after a full rebuild (deterministic regeneration). The locked
    manifest is compared only when `sealed_dir` is given; otherwise only its
    commitment in the index is compared.
    """
    public_dir = Path(public_dir)
    idx = load_index(public_dir / "index.json")
    plan = plan_from_index(idx)
    _, manifests, _, fresh = build_all(plan)
    if fresh != idx:
        raise ManifestError("index.json differs from a fresh rebuild of its plan")
    for s in PUBLIC_SPLITS:
        if load_manifest(public_dir / f"{s}.json", index_data=idx) != manifests[s]:
            raise ManifestError(f"{s}.json differs from a fresh rebuild")
    locked = "commitment_only"
    if sealed_dir is not None:
        if load_manifest(Path(sealed_dir) / f"{LOCKED}.json", index_data=idx, purpose=LOCKED_PURPOSE) != manifests[LOCKED]:
            raise ManifestError("locked test manifest differs from a fresh rebuild")
        locked = "verified"
    return {"plan_id": plan.plan_id, "plan_hash": plan.plan_hash(), "index_hash": idx["index_hash"],
            "public_splits": "verified", "locked_test": locked,
            # The plan is public, so the locked split can be rebuilt: procedurally locked, not blind.
            "locked_test_secrecy": "procedural_not_blind", "wall_geometry_audit": "passed"}
