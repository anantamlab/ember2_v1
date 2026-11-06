"""Generator: reproducibility, strict schema, labels, occupancy, diversity, bounds, leakage."""
from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import replace

import pytest

from ember_v1.contract import validate_observation
from ember_v1.gen import (GENERATOR_REVISION, GenerationError, Generated, GeneratorSpec,
                          generate_family, generate_lineage, provenance)
from ember_v1.gen import analysis as A
from ember_v1.gen.rng import GenRng, derive_seed
from ember_v1.sim import Scenario, Simulator, run_episode, verify_replay
from ember_v1.sim.movement import ACTIONS
from gen_helpers import SMALL_SPEC


def _batch(n_fam: int = 16, per: int = 3, spec: GeneratorSpec = SMALL_SPEC, base: int = 1):
    out = []
    for i in range(n_fam):
        fam, _ = generate_family(spec, derive_seed(base, "fam", i))
        for j in range(per):
            try:
                out.append(generate_lineage(spec, fam, derive_seed(fam.family_seed, "lin", j))[0])
            except GenerationError:
                pass
    return out


@pytest.fixture(scope="module")
def batch():
    return _batch()


def test_seed_derivation_and_rng_are_pinned():
    # Golden values, same on every supported Python. A change here changes every manifest.
    assert derive_seed(0, "x") == 0xA339EFA2606D5B2F
    assert derive_seed(1, "family", 0) == 0x0D18A95BE23B3932
    assert derive_seed(1, "family", 0) != derive_seed("1", "family", 0)
    r = GenRng(5)
    assert [r.below(10) for _ in range(8)] == [8, 4, 3, 9, 1, 6, 9, 5]
    assert r.integer(3, 9) == 9 and r.unit() == 0.6034405546675778
    for bad in ((True,), (), (-1,), ("",), (1.5,)):
        with pytest.raises(ValueError):
            derive_seed(*bad)


def test_generation_is_reproducible_from_seed_and_revision(batch):
    again = _batch()
    assert [g.scenario.to_json() for g in again] == [g.scenario.to_json() for g in batch]
    assert [g.probe for g in again] == [g.probe for g in batch]
    assert [g.diagnostics for g in again] == [g.diagnostics for g in batch]
    other = _batch(base=2)
    assert {g.scenario.content_hash() for g in other}.isdisjoint(g.scenario.content_hash() for g in batch)
    assert provenance(batch[0], SMALL_SPEC)["generator_revision"] == GENERATOR_REVISION


def test_spec_change_changes_ids_and_hash():
    other = GeneratorSpec.with_overrides({"crowd": {"count": [1, 29]}})
    assert other.spec_hash() != SMALL_SPEC.spec_hash()
    a, _ = generate_family(SMALL_SPEC, 99)
    b, _ = generate_family(other, 99)
    assert a.family_id != b.family_id


def test_spec_is_strict():
    bad = [{"grid": {"short_side": [2, 10]}}, {"grid": {"short_side": [10, 5]}}, {"grid": {"oops": 1}},
           {"layout": {"mix": {"castle": 1}}}, {"layout": {"mix": {"open_room": 0}}},
           {"fire": {"spread_probability_pct": [0, 101]}}, {"exits": {"interior_pct": 1.5}},
           {"attempts": {"family": 0}}, {"crowd": {"count": [1.0, 3]}}, {"nope": {}}]
    for override in bad:
        with pytest.raises(ValueError):
            GeneratorSpec.with_overrides(override)
    with pytest.raises(ValueError):
        GeneratorSpec.from_dict({"grid": {}})


def test_every_scenario_passes_the_strict_schema_and_round_trips(batch):
    for g in batch:
        s = g.scenario
        again = Scenario.from_dict(json.loads(s.to_json()))
        assert again == s and again.content_hash() == s.content_hash()


def test_occupancy_and_fire_placement_are_valid(batch):
    for g in batch:
        s = g.scenario
        walls, exits = set(s.obstacles), {p for _, p in s.exits}
        people = [p for _, p in s.people]
        assert len(set(people)) == len(people)
        assert not (set(people) & (walls | exits | set(s.ignitions)))
        assert not set(s.ignitions) & walls
        if not s.exits_flammable:
            assert not set(s.ignitions) & exits
        assert all(0 <= r < s.height and 0 <= c < s.width for r, c in walls | exits | set(people))
        # every exit is next to free floor in the main region of its family
        for e in exits:
            assert any(n not in walls and n not in exits for n in A.neighbours4(e, s.height, s.width))


def test_labels_match_independent_checks(batch):
    labels = Counter(g.label for g in batch)
    assert labels["feasible"] and labels["difficult"] and labels["unreachable"]
    for g in batch:
        s = g.scenario
        assert g.label == g.target
        trapped = set(Simulator(s).private_state()["trapped"])
        arrival = A.fire_arrival(s)
        L = A.latest_departure(s, arrival)
        if g.label == "feasible":
            region = A.exit_region(s)
            assert not trapped and all(p in region for _, p in s.people)
            assert all(p in L for _, p in s.people)
            assert A.capacity_feasible(s, arrival, [p for _, p in s.people])
        elif g.label == "unreachable":
            assert trapped and g.diagnostics["trapped_at_start"] == len(trapped)
            assert g.diagnostics["unreachable_mechanism"] in ("pocket", "fire_cut")
        else:
            assert not trapped
            assert g.diagnostics["difficulty_reasons"]
            assert (any(p not in L for _, p in s.people)
                    or not A.capacity_feasible(s, arrival, [p for _, p in s.people]))


def test_unreachable_cases_are_explicit_in_both_mechanisms():
    spec = SMALL_SPEC
    seen = Counter()
    for i in range(30):
        fam, _ = generate_family(spec, derive_seed(3, "fam", i))
        try:
            g, _ = generate_lineage(spec, fam, derive_seed(fam.family_seed, "u"), target="unreachable")
        except GenerationError:
            continue
        seen[g.diagnostics["unreachable_mechanism"]] += 1
        s = g.scenario
        sim = Simulator(s)
        assert sim.private_state()["trapped"]        # the simulator agrees at step 0
    assert seen["pocket"] and seen["fire_cut"]


def test_diverse_layouts_obstacles_exits_and_fire(batch):
    d = [g.diagnostics for g in batch]
    assert {x["archetype"] for x in d} == {"open_room", "corridor", "multi_room", "irregular"}
    assert len({x["exits"] for x in d}) >= 3
    assert len({g.family.params["exit_placement"] for g in batch}) == 3
    assert len({x["wall_fraction_pct"] // 10 for x in d}) >= 4
    assert {x["fire_model"] for x in d} == {"deterministic", "stochastic"}
    assert {x["enter_fire"] for x in d} == {"blocked", "casualty"}
    assert len({x["crowd_placement"] for x in d}) == 4
    assert len({x["ignition_placement"] for x in d}) >= 4
    assert any(x["grid"][0] != x["grid"][1] for x in d) and any(x["bottleneck_cells"] for x in d)
    assert any(x["people_with_disjoint_alternatives"] for x in d)


def test_bounded_attempts_raise_with_reasons():
    # every exit cell must be interior on a tiny map with 16 exits: impossible
    spec = GeneratorSpec.with_overrides({"grid": {"short_side": [3, 3], "aspect_pct": [100, 100]},
                                         "exits": {"count": [16, 16], "interior_pct": 0},
                                         "attempts": {"family": 7, "lineage": 5}})
    with pytest.raises(GenerationError) as exc:
        generate_family(spec, 1)
    assert exc.value.attempts == 7 and sum(exc.value.reasons.values()) == 7
    # a lineage target that the map cannot meet: no pockets and fire cannot cut an open room with 4 exits
    spec2 = GeneratorSpec.with_overrides({"layout": {"mix": {"open_room": 1}}, "obstacles": {"density_pct": [0, 0]},
                                          "exits": {"count": [4, 4]}, "ignition": {"count": [1, 1]},
                                          "fire": {"exits_flammable_pct": 0}, "attempts": {"family": 5, "lineage": 6}})
    fam, _ = generate_family(spec2, 5)
    with pytest.raises(GenerationError) as exc:
        generate_lineage(spec2, fam, 5, target="unreachable")
    assert exc.value.attempts == 6 and sum(exc.value.reasons.values()) == 6


def test_custom_rejection_rule_is_counted():
    fam, reasons = generate_family(SMALL_SPEC, 8, reject=lambda f: "veto" if f.attempt < 3 else None)
    assert fam.attempt == 3 and reasons["veto"] == 3


def _private_values(g: Generated) -> list[str]:
    p = provenance(g, SMALL_SPEC)
    values = [str(v) for k, v in p.items() if k not in ("generator_revision", "family_attempt", "lineage_attempt")]
    values += [g.family.topology_hash, g.lineage_id, g.scenario.content_hash()]
    assert all(len(v) >= 8 for v in values)   # seeds are 64-bit; short substrings would match by chance
    return values


def test_observation_carries_no_private_provenance(batch):
    for g in batch[:20]:
        sim = Simulator(g.scenario)
        obs = sim.observe()
        validate_observation(obs)
        text = json.dumps(obs)
        for secret in _private_values(g):
            assert secret not in text
        for key in ("seed", "scenario_id", "family", "lineage", "label", "target", "trapped", "rng"):
            assert key not in text
        assert all(p["id"].startswith("P") and p["id"][1:].isdigit() for p in obs["people"])


def test_observation_does_not_depend_on_seed_or_id(batch):
    for g in batch[:20]:
        s = g.scenario
        twin = replace(s, seed=(s.seed + 1) % 2 ** 64, scenario_id="other")
        assert Simulator(s).observe() == Simulator(twin).observe()


def test_person_ids_do_not_encode_position(batch):
    # IDs are assigned after a shuffle: id order is not row-major order across the batch
    row_major = 0
    total = 0
    for g in batch:
        if len(g.scenario.people) < 4:
            continue
        ids = sorted(g.scenario.people, key=lambda x: int(x[0][1:]))
        positions = [p for _, p in ids]
        total += 1
        row_major += positions == sorted(positions)
    assert total >= 10 and row_major < total / 4


def test_bounded_cpu_episodes_replay_completely(batch):
    rnd = random.Random(21)
    for g in batch[:25]:
        s = g.scenario
        policy = lambda obs: {p["id"]: rnd.choice(ACTIONS) for p in obs["people"] if p["status"] == "active"}  # noqa: E731
        records = run_episode(s, policy)
        assert 1 <= len(records) <= s.max_steps
        assert verify_replay(s, None, records).complete
        assert g.probe["steps"] <= min(s.max_steps, 20)
