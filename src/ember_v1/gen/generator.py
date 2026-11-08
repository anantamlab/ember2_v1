"""Seeded scenario generator with bounded rejection sampling.

Two levels:
  family   a static map (grid, walls, exits): the topology. Seeded by
           `family_seed`; attempt k uses `derive_seed(family_seed, "family-attempt", k)`.
  lineage  one scenario on a family: crowd, ignitions, fire rules, horizon and
           simulator seed. Seeded by `lineage_seed`; attempt k uses
           `derive_seed(lineage_seed, "lineage-attempt", k)`.

Each attempt either returns a candidate or a rejection reason. A caller gets a
result only after it passes the strict schema (`Scenario.from_dict`), the
requested label and the probe episode. After the bounded attempts are used,
`GenerationError` carries the reason counts. Nothing invalid is returned.

Everything here is generator-private (seeds, lineage, labels, diagnostics).
Only `Simulator(scenario).observe()` may reach a policy.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ember_v1.contract import validate_observation
from ember_v1.sim import Scenario, Simulator
from ember_v1.sim.canonical import digest

from . import analysis as A
from .layouts import BUILDERS, add_clutter, components
from .rng import GenRng, derive_seed
from .spec import GENERATOR_REVISION, GeneratorSpec

Pos = tuple[int, int]
_MAX_SIDE = 1024
PROBE_STEPS = 20   # probe episode length bound (full-horizon fire equality is tested separately)


class GenerationError(RuntimeError):
    """Bounded attempts used without an accepted candidate."""

    def __init__(self, what: str, attempts: int, reasons: Counter) -> None:
        super().__init__(f"{what}: no accepted candidate in {attempts} attempts; reasons {dict(reasons)}")
        self.attempts = attempts
        self.reasons = reasons


class GeneratorBug(AssertionError):
    """An accepted candidate failed an internal consistency check (never a rejection)."""


@dataclass(frozen=True)
class Family:
    family_id: str
    family_seed: int
    attempt: int
    archetype: str
    height: int
    width: int
    walls: frozenset[Pos]
    exits: tuple[Pos, ...]
    params: dict[str, Any]
    topology_hash: str
    main_region: frozenset[Pos]
    pockets: frozenset[Pos]
    cut_cells: tuple[Pos, ...]
    signature: A.MapSignature = field(compare=False, repr=False)

    def summary(self) -> dict[str, Any]:
        return {"family_id": self.family_id, "family_seed": self.family_seed, "attempt": self.attempt,
                "archetype": self.archetype, "grid": [self.height, self.width], "walls": len(self.walls),
                "exits": len(self.exits), "main_region": len(self.main_region), "pocket_cells": len(self.pockets),
                "cut_cells": len(self.cut_cells), "params": self.params, "topology_hash": self.topology_hash}


@dataclass(frozen=True)
class Generated:
    scenario: Scenario
    family: Family
    lineage_id: str
    lineage_seed: int
    attempt: int
    target: str
    label: str
    diagnostics: dict[str, Any]
    probe: dict[str, Any]

    def core_hash(self) -> str:
        return A.core_hash(self.scenario)


def family_id(spec: GeneratorSpec, family_seed: int, attempt: int) -> str:
    h = digest({"spec_hash": spec.spec_hash(), "family_seed": family_seed, "attempt": attempt})
    return "fam-" + h.split(":")[1][:16]


def lineage_id(fam: Family, lineage_seed: int, attempt: int) -> str:
    h = digest({"family_id": fam.family_id, "lineage_seed": lineage_seed, "attempt": attempt})
    return "lin-" + h.split(":")[1][:16]


# ----- family ---------------------------------------------------------------

def _place_exits(h: int, w: int, main: list[Pos], n: int, placement: str, interior_pct: int,
                 rng: GenRng) -> list[Pos] | str:
    main_set = set(main)
    boundary = [p for p in main if p[0] in (0, h - 1) or p[1] in (0, w - 1)]
    interior = [p for p in main if p not in set(boundary)]
    n_interior = sum(1 for _ in range(n) if rng.chance(interior_pct / 100))
    n_interior = min(n_interior, len(interior))
    n_boundary = n - n_interior
    if len(boundary) < n_boundary:
        return "insufficient_exit_cells"
    manhattan = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])  # noqa: E731
    if placement == "random" or n_boundary == 0:
        chosen = rng.sample(boundary, n_boundary)
    elif placement == "spread":
        chosen = [rng.choice(boundary)]
        while len(chosen) < n_boundary:
            order = list(boundary)
            rng.shuffle(order)
            chosen.append(max((p for p in order if p not in chosen), key=lambda p: min(manhattan(p, q) for q in chosen)))
    else:  # clustered: the n nearest boundary cells to an anchor (ties in random order)
        anchor = rng.choice(boundary)
        order = list(boundary)
        rng.shuffle(order)
        order.sort(key=lambda p: manhattan(p, anchor))
        chosen = order[:n_boundary]
    chosen += rng.sample(interior, n_interior)
    assert set(chosen) <= main_set
    return sorted(chosen)


def build_family(spec: GeneratorSpec, family_seed: int, attempt: int) -> Family | str:
    """One family attempt: a `Family` or a rejection reason."""
    rng = GenRng(derive_seed(family_seed, "family-attempt", attempt))
    lay = spec.get("layout", "mix")
    archetype = rng.weighted([(k, lay[k]) for k in sorted(lay)])
    short = rng.integer(*spec.get("grid", "short_side"))
    aspect = rng.integer(*spec.get("grid", "aspect_pct"))
    long = min(_MAX_SIDE, max(short, (short * aspect) // 100))
    h, w = (short, long) if rng.chance(0.5) else (long, short)
    params: dict[str, Any] = {
        "short_side": short, "aspect_pct": aspect,
        "corridor_width": rng.integer(*spec.get("layout", "corridor_width")),
        "door_width": rng.integer(*spec.get("layout", "door_width")),
        "min_room_side": rng.integer(*spec.get("layout", "min_room_side")),
        "extra_door_pct": spec.get("layout", "extra_door_pct"),
        "sealed_room_pct": spec.get("layout", "sealed_room_pct"),
        "cave_fill_pct": rng.integer(*spec.get("layout", "cave_fill_pct")),
        "cave_smoothing": rng.integer(*spec.get("layout", "cave_smoothing")),
    }
    walls = BUILDERS[archetype](h, w, rng, params)
    density = rng.integer(*spec.get("obstacles", "density_pct"))
    con = spec.get("obstacles", "connectivity")
    connectivity = rng.weighted([(k, con[k]) for k in sorted(con)])
    walls, added = add_clutter(h, w, walls, density, connectivity, rng)
    params.update({"clutter_density_pct": density, "clutter_connectivity": connectivity, "clutter_cells": added})
    # keep only the parameters that apply to this archetype
    used = {"open_room": (), "corridor": ("corridor_width", "corridor_width_used", "extra_door_pct"),
            "multi_room": ("door_width", "min_room_side", "extra_door_pct", "sealed_room_pct", "rooms",
                           "sealed_rooms"),
            "irregular": ("cave_fill_pct", "cave_smoothing")}[archetype]
    params = {k: v for k, v in params.items()
              if k in used or k in ("short_side", "aspect_pct") or k.startswith("clutter_")}
    comps = components(h, w, walls)
    free = sum(len(c) for c in comps)
    if not comps or free < 8:
        return "too_little_free_space"
    main = comps[0]
    if 2 * len(main) < free:
        return "fragmented_layout"
    n_exits = rng.integer(*spec.get("exits", "count"))
    pl = spec.get("exits", "placement")
    placement = rng.weighted([(k, pl[k]) for k in sorted(pl)])
    exits = _place_exits(h, w, main, n_exits, placement, spec.get("exits", "interior_pct"), rng)
    if isinstance(exits, str):
        return exits
    exit_set = set(exits)
    if len(main) - len(exits) < 4:
        return "too_little_free_space"
    for e in exits:
        if not any(n not in walls and n not in exit_set for n in A.neighbours4(e, h, w)):
            return "exit_inaccessible"
    params["exit_placement"] = placement
    main_set = frozenset(main)
    pockets = frozenset(p for c in comps[1:] for p in c)
    fam = Family(family_id(spec, family_seed, attempt), family_seed, attempt, archetype, h, w, walls,
                 tuple(exits), params, A.topology_hash(h, w, walls, exits), main_set, pockets,
                 tuple(A.cut_cells(h, w, walls, exits)), A.MapSignature(h, w, walls, exits))
    return fam


def generate_family(spec: GeneratorSpec, family_seed: int,
                    reject: Callable[[Family], str | None] | None = None) -> tuple[Family, Counter]:
    """First accepted family attempt. `reject` adds caller rules (e.g. near-duplicate)."""
    reasons: Counter = Counter()
    limit = spec.get("attempts", "family")
    for attempt in range(limit):
        out = build_family(spec, family_seed, attempt)
        if isinstance(out, Family) and reject is not None:
            why = reject(out)
            if why:
                out = why
        if isinstance(out, Family):
            return out, reasons
        reasons[out] += 1
    raise GenerationError(f"family seed {family_seed}", limit, reasons)


# ----- lineage --------------------------------------------------------------

def _crowd(fam: Family, n: int, placement: str, clusters: int, rng: GenRng) -> list[Pos]:
    exits = set(fam.exits)
    avail = sorted(fam.main_region - exits)
    if placement == "uniform":
        return rng.sample(avail, n)
    if placement == "clustered":
        k = min(clusters, n)
        centres = rng.sample(avail, k)
        blocked = fam.walls | exits | fam.pockets
        orders = []
        for c in centres:
            d = A.bfs(fam.height, fam.width, blocked, [c])
            cells = sorted(d, key=lambda p: (d[p], p))
            orders.append(cells)
        chosen: list[Pos] = []
        taken: set[Pos] = set()
        idx = [0] * k
        while len(chosen) < n:
            progressed = False
            for i in range(k):
                while idx[i] < len(orders[i]) and orders[i][idx[i]] in taken:
                    idx[i] += 1
                if idx[i] < len(orders[i]) and len(chosen) < n:
                    p = orders[i][idx[i]]
                    taken.add(p)
                    chosen.append(p)
                    progressed = True
            if not progressed:
                break
        return chosen
    d = A.bfs(fam.height, fam.width, fam.walls, list(fam.exits), stop=exits)
    order = list(avail)
    rng.shuffle(order)
    order.sort(key=lambda p: d.get(p, 0), reverse=(placement == "far_from_exits"))
    half = order[: max(n, len(order) // 2)]
    return rng.sample(half, n)


def _ignitions(fam: Family, people: list[Pos], count: int, placement: str, exits_flammable: bool,
               rng: GenRng) -> list[Pos] | str:
    exits = set(fam.exits)
    occupied = set(people)
    free = [(r, c) for r in range(fam.height) for c in range(fam.width)
            if (r, c) not in fam.walls and (r, c) not in occupied]
    floor = [p for p in free if p not in exits]
    if placement == "random":
        cand = floor
    elif placement == "near_crowd":
        d = A.bfs(fam.height, fam.width, fam.walls, people)
        cand = [p for p in floor if 1 <= d.get(p, 99) <= 3]
    elif placement == "near_exit":
        d = A.bfs(fam.height, fam.width, fam.walls, list(fam.exits))
        cand = [p for p in free if d.get(p, 99) <= 3 and (exits_flammable or p not in exits)]
    elif placement == "bottleneck":
        cand = [p for p in fam.cut_cells if p not in occupied]
    else:  # far_from_crowd
        d = A.bfs(fam.height, fam.width, fam.walls, people)
        reach = sorted((d[p], p) for p in floor if p in d)
        cand = [p for _, p in reach[(3 * len(reach)) // 4:]] if reach else []
    if not cand:
        return f"no_ignition_cells:{placement}"
    return sorted(rng.sample(cand, min(count, len(cand))))


def _flags(s: Scenario, people: dict[str, Pos], arrival: dict[Pos, int], L: dict[Pos, float],
           trapped: set[str], routes: dict[str, Any]) -> dict[str, Any]:
    """Descriptive situation flags where reference policies may disagree. Not optimization targets."""
    exits = [(i, p) for i, p in s.exits if p not in set(s.ignitions)]
    blocked = set(s.obstacles) | set(s.ignitions)
    stop = {p for _, p in s.exits}
    per_exit = {i: A.bfs(s.height, s.width, blocked, [p], stop=stop) for i, p in exits}
    reach_people = [pid for pid in sorted(people) if pid not in trapped]
    multi = load = fair = 0
    loads: Counter = Counter()
    cut_nearest = False
    for pid in reach_people:
        p = people[pid]
        ds = sorted((d[p], i) for i, d in per_exit.items() if p in d)
        if len(ds) >= 2:
            multi += 1
        if ds:
            dmin, eid = ds[0]
            loads[eid] += 1
            e_pos = dict(s.exits)[eid]
            if arrival.get(e_pos, A.INF) <= dmin and L.get(p, -1) >= 0:
                cut_nearest = True
    usable = sum(1 for d in per_exit.values() if any(people[pid] in d for pid in reach_people))
    if loads and usable >= 2:
        fair = -(-len(reach_people) // usable)
        load = max(loads.values())
    slack = sorted(L[people[pid]] for pid in reach_people if people[pid] in L)
    return {
        "stochastic_fire": s.fire_model == "stochastic",
        "people_with_multiple_reachable_exits": multi,
        "nearest_exit_overload": bool(loads and usable >= 2 and load >= 2 * fair),
        "bottleneck_on_route": bool(routes["bottleneck_cells"]),
        "nearest_exit_cut_by_fire": cut_nearest,
        "heterogeneous_urgency": bool(slack and slack[0] <= 3 and slack[-1] >= 15),
        "isolated_people": len(trapped),
        "min_slack": None if not slack else int(slack[0]),
        "median_slack": None if not slack else int(slack[len(slack) // 2]),
    }


def _probe(s: Scenario, arrival: dict[Pos, int], trapped: set[str]) -> dict[str, Any]:
    """Stay-policy episode in the simulator for at most PROBE_STEPS steps.

    Checks: the strict observation contract, the step-0 trapped set, and the
    analysis fire arrival against the simulator's ignitions at every step run.
    The chain hash is a reproducibility witness for the manifest.
    """
    sim = Simulator(s)
    if set(sim.private_state()["trapped"]) != trapped:
        raise GeneratorBug(f"{s.scenario_id}: static trapped set differs from the simulator")
    validate_observation(sim.observe())
    by_step: dict[int, list[list[int]]] = {}
    for p, t in sorted(arrival.items()):
        by_step.setdefault(t, []).append(list(p))
    steps = 0
    while not sim.terminated and steps < PROBE_STEPS:
        rec = sim.step({pid: "stay" for pid in sim.active_ids()})
        steps += 1
        if rec["ignitions"] != by_step.get(rec["step"], []):
            raise GeneratorBug(f"{s.scenario_id}: fire arrival differs from the simulator at step {rec['step']}")
    if steps > s.max_steps:
        raise GeneratorBug(f"{s.scenario_id}: probe exceeded max_steps")
    return {"steps": steps, "termination": sim.termination_reason, "chain_hash": sim.chain_hash}


def build_lineage(spec: GeneratorSpec, fam: Family, lineage_seed: int, attempt: int,
                  target: str) -> Generated | str:
    """One lineage attempt: a `Generated` (schema-valid, label == target, probe passed) or a reason."""
    rng = GenRng(derive_seed(lineage_seed, "lineage-attempt", attempt))
    fm = spec.get("fire", "model")
    model = rng.weighted([(k, fm[k]) for k in sorted(fm)])
    interval = rng.integer(*spec.get("fire", "spread_interval"))
    prob_pct = rng.integer(*spec.get("fire", "spread_probability_pct"))
    flammable = rng.chance(spec.get("fire", "exits_flammable_pct") / 100)
    ef = spec.get("fire", "enter_fire")
    enter_fire = rng.weighted([(k, ef[k]) for k in sorted(ef)])
    fire = {"model": model, "spread_interval": interval, "exits_flammable": flammable}
    if model == "stochastic":
        fire["spread_probability"] = prob_pct / 100
    lo, hi = spec.get("horizon", "max_steps")
    max_steps = max(lo, min(hi, ((fam.height + fam.width) * rng.integer(*spec.get("horizon", "steps_per_side_pct"))) // 100))

    avail = len(fam.main_region) - len(fam.exits)
    permille = rng.integer(*spec.get("crowd", "density_permille"))
    c_lo, c_hi = spec.get("crowd", "count")
    n = max(c_lo, min(c_hi, (avail * permille) // 1000))
    n = min(n, avail - 1)
    if n < 1:
        return "crowd_capacity"
    cp = spec.get("crowd", "placement")
    placement = rng.weighted([(k, cp[k]) for k in sorted(cp)])
    people = _crowd(fam, n, placement, rng.integer(*spec.get("crowd", "clusters")), rng)
    if len(people) < n:
        return "crowd_capacity"
    mechanism = None
    if target == "unreachable":
        mechanism = "pocket" if fam.pockets and rng.chance(0.5) else "fire_cut"
        if mechanism == "pocket":
            m = rng.integer(1, max(1, min(len(fam.pockets), n // 4)))
            keep = rng.sample(people, n - m) if n > m else []
            people = sorted(keep + rng.sample(sorted(fam.pockets), m))
    ip = spec.get("ignition", "placement")
    if not fam.cut_cells:   # "fire at a bottleneck" is undefined on a map without one
        ip = {k: v for k, v in ip.items() if k != "bottleneck"} or {"random": 1}
    if mechanism == "fire_cut":
        ign_place = rng.choice(["bottleneck", "near_exit"] if fam.cut_cells else ["near_exit"])
    else:
        ign_place = rng.weighted([(k, ip[k]) for k in sorted(ip)])
    ignitions = _ignitions(fam, people, rng.integer(*spec.get("ignition", "count")), ign_place, flammable, rng)
    if isinstance(ignitions, str):
        return ignitions
    order = list(people)
    rng.shuffle(order)          # person IDs carry no spatial order
    lid = lineage_id(fam, lineage_seed, attempt)
    data = {
        "schema_version": "1.0", "scenario_id": "g-" + lid[4:],
        "grid": {"width": fam.width, "height": fam.height},
        "obstacles": [list(p) for p in sorted(fam.walls)],
        "exits": [{"id": f"E{i}", "position": list(p)} for i, p in enumerate(fam.exits)],
        "people": [{"id": f"P{i}", "position": list(p)} for i, p in enumerate(order)],
        "ignitions": [list(p) for p in ignitions], "fire": fire,
        "movement": {"conflict_resolution": "lowest_id_wins", "enter_fire": enter_fire},
        "termination": {"max_steps": max_steps},
        "seed": derive_seed(lineage_seed, "scenario-seed", attempt),
    }
    try:
        s = Scenario.from_dict(data)
    except ValueError as exc:
        return f"schema_rejected:{exc}"

    pos = dict(s.people)
    region = A.exit_region(s)
    trapped = {pid for pid, p in pos.items() if p not in region}
    arrival = A.fire_arrival(s)
    L = A.latest_departure(s, arrival)
    lone_fail = {pid for pid, p in pos.items() if p not in L}
    capacity = A.capacity_feasible(s, arrival, [p for pid, p in pos.items() if pid not in trapped])
    reasons = []
    if trapped:
        label = "unreachable"
    elif lone_fail or not capacity:
        label = "difficult"
        reasons = (["fire_race"] if lone_fail else []) + ([] if capacity else ["exit_capacity"])
    else:
        label = "feasible"
    if label != target:
        return f"label_mismatch:{label}"
    routes = A.route_structure(s, [p for pid, p in sorted(pos.items()) if pid not in trapped])
    probe = _probe(s, arrival, trapped)
    exit_burn = [arrival[p] for _, p in s.exits if p in arrival]
    diag = {
        "grid": [s.height, s.width], "archetype": fam.archetype, "walls": len(s.obstacles),
        "wall_fraction_pct": (100 * len(s.obstacles)) // (s.height * s.width),
        "exits": len(s.exits), "people": len(s.people), "crowd_placement": placement,
        "ignitions": len(s.ignitions), "ignition_placement": ign_place, "unreachable_mechanism": mechanism,
        "fire_model": s.fire_model, "spread_interval": s.spread_interval,
        "spread_probability": s.spread_probability, "exits_flammable": s.exits_flammable,
        "enter_fire": s.enter_fire, "max_steps": s.max_steps,
        "difficulty_reasons": reasons, "lone_escape_failures": len(lone_fail), "trapped_at_start": len(trapped),
        "first_exit_burn_step": min(exit_burn) if exit_burn else None,
        "bottleneck_cells": len(routes["bottleneck_cells"]),
        "people_with_disjoint_alternatives": routes["people_with_disjoint_alternatives"],
        "people_single_exit": routes["people_single_exit"],
        "flags": _flags(s, pos, arrival, L, trapped, routes),
    }
    return Generated(s, fam, lid, lineage_seed, attempt, target, label, diag, probe)


def sample_target(spec: GeneratorSpec, lineage_seed: int) -> str:
    mix = spec.get("targets", "mix")
    return GenRng(derive_seed(lineage_seed, "target")).weighted([(k, mix[k]) for k in sorted(mix)])


def generate_lineage(spec: GeneratorSpec, fam: Family, lineage_seed: int,
                     target: str | None = None) -> tuple[Generated, Counter]:
    """First accepted lineage attempt for `target` (sampled from the spec mix if None)."""
    target = sample_target(spec, lineage_seed) if target is None else target
    reasons: Counter = Counter()
    limit = spec.get("attempts", "lineage")
    for attempt in range(limit):
        out = build_lineage(spec, fam, lineage_seed, attempt, target)
        if isinstance(out, Generated):
            return out, reasons
        reasons["schema_rejected" if out.startswith("schema_rejected") else out] += 1
    raise GenerationError(f"lineage seed {lineage_seed} (target {target})", limit, reasons)


def provenance(g: Generated, spec: GeneratorSpec) -> dict[str, Any]:
    """Generator-private provenance of one scenario. Never part of an observation."""
    return {"generator_revision": GENERATOR_REVISION, "spec_hash": spec.spec_hash(),
            "family_id": g.family.family_id, "family_seed": g.family.family_seed,
            "family_attempt": g.family.attempt, "lineage_id": g.lineage_id, "lineage_seed": g.lineage_seed,
            "lineage_attempt": g.attempt, "scenario_seed": g.scenario.seed, "scenario_id": g.scenario.scenario_id}
