# P1-02 (issue #6): independent scenario generator and leakage-safe splits

**Status:** implemented and tested on CPU. Manifests `p1-02-v1` are frozen in `data/splits/p1-02-v1/`. The locked test manifest is not in the repository, but its plan is public: it is **procedurally locked, not blind** (§5, §11). **Blind evaluation: BLOCKED** until a private salt is provisioned (§11). No teacher is implemented or evaluated. No GPU, Modal, Qwen or EMBERBench use.

## 1. Scope and provenance

- The generator makes `ember_v1.sim.Scenario` objects (schema 1.0) for our own simulator (issue #5).
- Its dimensions come from general evacuation factors: building form, room scale, exits, obstacles, crowd size and position, ignition position and fire speed.
- It does **not** reproduce any EMBERBench scenario family or distribution. It does not use EMBERBench code, fixtures, evaluation scenarios, hidden tests, results or leaderboard feedback. No range was chosen to fill a benchmark coverage gap.
- The simulator is not changed. No simulator bug was found (§10).

## 2. Architecture

Package `ember_v1.gen` (standard library only):

| Module | Job |
|---|---|
| `rng.py` | `derive_seed(*parts)` (SHA-256 over a label path, prefix `ember_v1/gen`), `GenRng` (SplitMix64 with own `below`/`sample`/`shuffle`). Python's `random` is not used, because its methods changed between releases. |
| `spec.py` | `GeneratorSpec`: strict, integer-only spec. `GENERATOR_REVISION = "gen-1.0.0"`. `spec_hash()` binds the revision and the spec. |
| `layouts.py` | Wall builders per archetype; clutter with an optional local connectivity guard; free-space components. |
| `analysis.py` | Private analysis: exit region, fire arrival `F(c)`, lone-escape latest departure `L(c)`, route graph cut cells (Tarjan), exit capacity bound, symmetry-invariant topology hash, near-duplicate signatures. |
| `generator.py` | `build_family` / `build_lineage` (one attempt each → candidate or rejection reason), `generate_family` / `generate_lineage` (bounded rejection sampling), probe episode, labels, diagnostics. |
| `splits.py` | `SplitPlan`, family pool (`fill_family_pool`), lineages (`generate_lineages`), split assignment, manifests, index, leakage audit, wall geometry audit, sealed write, verified load, `materialize`, `verify_split_files`. |
| `blind.py` | Blind test set from a private salt: guarded salt loading, disjointness against every public split, private manifest, public commitment (§11). Not imported by any other module. |

Two levels of identity:

- **Family** = one static map (grid, walls, exits). Seed `family_seed`; attempt `k` uses `derive_seed(family_seed, "family-attempt", k)`. ID `fam-<16 hex>` = digest of (spec hash, family seed, attempt). `topology_hash` = digest of the map in its canonical form over the 8 grid symmetries (exit IDs ignored).
- **Lineage** = one scenario on a family (crowd, ignitions, fire rules, horizon, simulator seed). Seed `derive_seed(family_seed, "lineage", j)`; attempt `k` uses `derive_seed(lineage_seed, "lineage-attempt", k)`. ID `lin-<16 hex>` = digest of (family ID, lineage seed, attempt). Simulator seed = `derive_seed(lineage_seed, "scenario-seed", k)`.

Acceptance of one lineage attempt, in order: strict `Scenario.from_dict` → label equals the requested target → probe episode passes. A probe failure raises `GeneratorBug` (it is never a silent rejection).

**Probe episode.** A stay policy runs in the simulator for `min(max_steps, 20)` steps (bounded). It checks the strict observation contract (`ember_v1.contract.validate_observation`), that the simulator's step-0 trapped set equals the analysis, and that the analysis fire arrival equals the simulator's ignitions at every step. The final chain hash is stored as a reproducibility witness. Full-horizon fire equality is tested separately against `ember_v1.sim.fire.spread` (400 random scenarios).

## 3. Scenario dimensions (`GeneratorSpec`, all independent)

| Section.field | Default | Meaning |
|---|---|---|
| `grid.short_side`, `grid.aspect_pct` | [8, 30], [100, 250] | Grid scale and aspect ratio; orientation is random. |
| `layout.mix` | open_room 1, corridor 1, multi_room 1, irregular 1 | Archetype: open hall; carved corridor network; binary-space-partition rooms with doors; cellular-automaton cave. |
| `layout.corridor_width`, `layout.door_width` | [1, 3], [1, 3] | Corridor and door width (bottleneck strength). |
| `layout.min_room_side` | [3, 6] | Room size in `multi_room`. |
| `layout.extra_door_pct` | 35 | Chance of a second door or second corridor spine (loops, alternative paths). |
| `layout.sealed_room_pct` | 15 | Chance of a room with no door (isolated area). |
| `layout.cave_fill_pct`, `layout.cave_smoothing` | [38, 46], [2, 5] | Cave roughness. |
| `obstacles.density_pct` | [0, 20] | Clutter as % of free cells. |
| `obstacles.connectivity` | preserve 3, free 1 | `preserve`: 1×1 obstacles that cannot split free space (local 8-ring test). `free`: 1×1–2×2 blocks, may make pockets. |
| `exits.count`, `exits.placement`, `exits.interior_pct` | [1, 4]; random/spread/clustered; 10 | Exit number, placement on the boundary of the main region, chance per exit of an interior exit (stair). |
| `crowd.density_permille`, `crowd.count` | [10, 200], [1, 120] | Crowd size from density of the main region, clamped by count. |
| `crowd.placement`, `crowd.clusters` | uniform 2, clustered 2, far_from_exits 1, near_exits 1; [1, 3] | Initial distribution. |
| `ignition.count`, `ignition.placement` | [1, 3]; random 2, near_crowd 1, near_exit 1, bottleneck 1, far_from_crowd 1 | Fire start cells. `bottleneck` uses map cut cells; it is dropped for maps without one. |
| `fire.model`, `fire.spread_interval`, `fire.spread_probability_pct` | det 1, stoch 1; [1, 4]; [10, 90] | Deterministic or stochastic spread, speed and probability. |
| `fire.exits_flammable_pct`, `fire.enter_fire` | 50; blocked 1, casualty 1 | Simulator rules. |
| `horizon.steps_per_side_pct`, `horizon.max_steps` | [150, 400], [20, 300] | `max_steps = (h + w) · pct / 100`, clamped. |
| `targets.mix` | feasible 6, difficult 3, unreachable 1 | Requested label per lineage. |
| `attempts.family`, `attempts.lineage` | 40, 60 | Rejection-sampling bounds. |

Person IDs `P0…` are given after a shuffle, so ID order does not follow position. Exit IDs `E0…` follow row-major position.

## 4. Labels and diagnostics (generator-private)

Escape model: `F(c)` = first step after which cell `c` burns (true trajectory for the scenario seed). A lone person at `c` at time `t` can escape iff a move sequence reaches an exit, never entering a cell that burns at the start of the step, and never ending a step on a cell that burns in that step. `L(c)` = latest such `t` (max-bottleneck Dijkstra). Tests compare it with a time-expanded BFS on every cell of 250 random scenarios, and replay each BFS plan in the simulator, which evacuates the person.

| Label | Definition |
|---|---|
| `unreachable` | At step 0 some person has no path to a non-burning exit (the simulator's trapped set). Mechanism `pocket` (person in an isolated area) or `fire_cut` (ignition at a bottleneck or exit). |
| `difficult` | Nobody is trapped, but `fire_race` (some person cannot escape even alone) or `exit_capacity` (one person per exit per step inside each exit's usable window is less than the crowd). |
| `feasible` | Every person reaches an exit at step 0, can escape alone under the true fire, and the exit capacity bound holds. This is necessary, not sufficient, for zero casualties: it ignores collisions and queues. |

Diagnostics per scenario include grid, wall fraction, counts, placements, fire rules, first exit burn step, cut cells on routes, people with two vertex-disjoint routes, people with a single exit, and **situation flags** where reference policies (T0–T4) may disagree: `stochastic_fire`, `multiple_reachable_exits`, `nearest_exit_overload` (nearest-exit assignment loads one exit at least 2× its share), `bottleneck_on_route`, `nearest_exit_cut_by_fire` (the nearest exit burns before arrival, another route works), `heterogeneous_urgency` (lone-escape slack ≤ 3 for one person and ≥ 15 for another). The flags are descriptive. They are counted, never used as acceptance targets, and no scenario is designed for one teacher to win.

## 5. Splits and leakage audit

Plan `p1-02-v1` (`splits.DEFAULT_PLAN`): master seed 20261106, default spec, families train 64 / dev 16 / calibration 16 / locked test 24, 3 lineages per family.

1. **Pool.** Families are built in slot order. A candidate with the same topology hash as an accepted family is rejected (`duplicate_family`); a near-duplicate of any accepted family is rejected (`near_duplicate_family`). So no two families in any split look alike.
2. **Assignment.** By family only. Families are sorted by (archetype, family ID) and dealt in proportion to the split counts (archetype-stratified). All lineages stay with their family.
3. **Audit** (`splits.audit`, recomputed from the scenarios themselves, not from manifest claims). All must have zero violations:

| Check | Rule |
|---|---|
| `family_ids_disjoint`, `family_single_split` | A family ID is in one split only. |
| `lineage_ids_disjoint`, `lineage_ids_unique` | A lineage ID is in one split and occurs once. |
| `seeds_disjoint`, `scenario_seeds_unique` | Family, lineage and simulator seeds never repeat across splits; simulator seeds are globally unique. |
| `scenario_hashes_disjoint`, `core_hashes_disjoint` | Same content, also with `scenario_id` and `seed` removed. |
| `topology_disjoint` | Symmetry-invariant map hash in one split only (catches rotated, mirrored and transposed copies and same-map variants with a new crowd, fire or seed). |
| `near_duplicate_maps` | Map pairs in different splits: area ratio ≤ 1.3, aspect ratio ratio ≤ 1.15, mean absolute wall-fraction difference ≤ 0.03 on a 16×16 area-weighted grid (best of the 8 symmetries), and exit chamfer distance ≤ 0.1 (unit square, Chebyshev). |
| `wall_geometry_audit` (supplementary) | Map pairs in different splits with the same scale and aspect gates and wall distance ≤ 0.03, **exits ignored**, if both maps have ≥ 5 % wall cells. Added in review: `near_duplicate_maps` let one building pass with other exits (107 of 108 structured `p1-02-v1` maps with exits moved far were not caught). It is not stored in the frozen `p1-02-v1` index (that would change its hash); `build_all` and `verify_split_files` enforce it for every plan, so CI checks it on the committed files. `p1-02-v1` has zero violations. |

Deliberate leak tests (`tests/test_gen_splits.py`) insert into the locked split: an exact copy with new IDs and seed; the same map with a new crowd, fire rules, horizon and seed; all 7 non-identity symmetries; 1 % wall edits; a one-cell exit move; the same structured walls (also mirrored) with one far exit; a reused family ID, lineage ID, family seed or simulator seed; and a family split across two partitions. Each is caught by the expected check, and a clean unrelated map passes.

**Similarity classes** (the audit reads scenario content, not IDs, so a family ID never hides a copy):

| Similarity across splits | Result | Check |
|---|---|---|
| Exact copy, any IDs or seeds; same parent map under another family ID | prevented | `core_hashes_disjoint`, `topology_disjoint` |
| Same map with another crowd, ignition, fire rules, horizon or seed | prevented | `topology_disjoint` |
| Any of the 8 grid symmetries | prevented | `topology_disjoint` |
| About ≤ 3 % wall edits, or a small exit move, same exits otherwise | prevented | `near_duplicate_maps` |
| Same or nearly same walls (≥ 5 % walls) with different exits | prevented | `wall_geometry_audit` |
| Sparse halls (< 5 % walls) with exits far apart (chamfer > 0.1) | allowed: in a hall the exits are the structure; two halls are not one building because both have few walls | — |
| Independent maps of one archetype (similar corridors, similar room graphs, different geometry) | allowed: this is the generalization target | — |
| Not detected: whole-layout shifts by ≥ 1 cell on small maps, area change > 1.3×, edits > about 3 % of cells | allowed by threshold; independent family seeds make them very unlikely | — |

Nearest cross-split pairs in `p1-02-v1` (all 120 families compared, gated pairs only): (1) train open room 29×51 with 2 % clutter vs locked open room 26×50 with no walls: wall distance 0.020, exits 0.49 apart, so allowed; (2) the same train room vs locked open room 26×52, 2 % walls: wall distance 0.032; (3) dev vs locked open rooms, 4 % walls: wall distance 0.064, exits 0.08. The nearest cross-split pair of structured maps (≥ 5 % walls) has wall distance 0.128, four times the threshold. No two families have identical walls up to symmetry.

**Locked test seal (procedural).** `index.json` holds only the locked manifest hash and counts (a commitment). `write_split_files` writes `locked_test.json` only to a separate sealed directory and refuses a sealed directory inside the public one (or the reverse). `load_manifest` and `materialize` raise `LockedSplitError` for the locked split unless `purpose=LOCKED_PURPOSE`; `materialize` also refuses every non-public split (for example a blind manifest). A test asserts that only `gen/splits.py` uses that token, and that no `locked_test*.json` exists in the repository (`data/sealed/` is git-ignored). Locked-test coverage is not published in the index.

**Security classification.** The `p1-02-v1` locked test is a **procedurally locked, reproducible development artifact**. It is not secret and not blind: the public plan rebuilds it, and `verify_split_files` (CI) rebuilds it to check its commitment. The guards stop accidental use by code, not a person. Use it as a held-out split for development checks. Do not report it as a blind result. A blind result needs §11. The issue #6 review also inspected its maps and failed lineages (§5 nearest pairs, §9), so it is inspected as well as reproducible.

**Policy visibility.** Seeds, IDs, lineage, labels, diagnostics and provenance live only in the generator and manifests. Tests check that `Simulator(scenario).observe()` passes the observation whitelist, contains none of these values, and is identical for two scenarios that differ only in seed and scenario ID.

**Immutability.** Each manifest carries `manifest_hash` (digest of its body); the index carries `index_hash` and binds each manifest hash and the plan hash. `write_split_files` refuses to overwrite a file with different content. `verify_split_files` rebuilds the plan and compares every file; CI runs it on the committed files.

## 6. Reproducibility evidence

- `test_seed_derivation_and_rng_are_pinned`: golden `derive_seed` and `GenRng` values.
- `test_generation_is_reproducible_from_seed_and_revision`: two runs give equal scenario JSON, diagnostics and probe chain hashes; other seeds give disjoint hashes.
- `test_manifest_regeneration_is_deterministic`: two builds of a plan give equal manifests and index.
- `test_committed_manifests_regenerate_exactly`: the committed `p1-02-v1` files were built on Python 3.14.7 (Windows); CI rebuilds and compares them on Python 3.11 and 3.12 (Linux).
- Hashes: index `sha256:277f098d…aea74a1c`; train `sha256:96a56481…d44b442d`; dev `sha256:b08c2b13…6b6d1347`; calibration `sha256:20af10dd…d27736bd`; locked test commitment `sha256:e1eb5d54…2201c56a`.

## 7. Coverage (`p1-02-v1`, train / dev / calibration scenario counts)

Families 64 / 16 / 16; scenarios 189 / 48 / 48 (locked test: 24 families, 70 scenarios, coverage sealed).

| Dimension | Counts |
|---|---|
| archetype | corridor 41/12/12; irregular 57/12/12; multi_room 36/9/9; open_room 55/15/15 |
| label | feasible 109/28/33; difficult 59/14/12; unreachable 21/6/3 |
| difficulty reason | fire_race 52/13/11; exit_capacity 29/2/4 |
| unreachable mechanism | fire_cut 13/6/2; pocket 8/0/1 |
| short side | 8–11: 23/3/9; 12–15: 21/18/15; 16–19: 11/9/3; 20–24: 71/9/12; 25–30: 63/9/9 |
| aspect (×100) | 100–124: 29/3/3; 125–149: 32/6/6; 150–199: 68/27/24; 200–250: 60/12/15 |
| wall fraction % | 0–4: 19/3/0; 5–14: 27/9/12; 15–29: 48/12/15; 30–49: 42/9/6; 50–69: 23/9/3; ≥70: 30/6/12 |
| exits | 1: 48/12/12; 2: 44/15/18; 3: 60/6/9; 4: 37/15/9 |
| people | 1: 8/1/2; 2–5: 14/7/7; 6–15: 34/11/7; 16–40: 51/13/23; 41–80: 43/10/7; ≥81: 39/6/2 |
| crowd placement | uniform 58/15/14; clustered 66/17/17; far_from_exits 31/5/6; near_exits 34/11/11 |
| ignitions | 1: 73/18/19; 2: 55/10/16; 3: 61/20/13 |
| ignition placement | random 72/20/18; near_crowd 28/7/9; near_exit 30/4/7; bottleneck 25/10/5; far_from_crowd 34/7/9 |
| fire model | deterministic 82/19/21; stochastic 107/29/27 |
| spread interval | 1: 51/14/12; 2: 61/11/15; 3: 44/11/10; 4: 33/12/11 |
| spread probability | 0–0.24: 19/3/3; 0.25–0.49: 27/7/15; 0.50–0.74: 38/11/5; 0.75–0.90: 23/8/4 |
| exits flammable | true 100/24/28; false 89/24/20 |
| enter_fire | blocked 80/29/23; casualty 109/19/25 |
| bottleneck cells on routes | 0: 96/23/27; 1–2: 29/8/9; 3–9: 44/9/6; 10–29: 17/8/6; ≥30: 3/0/0 |
| flags | multiple_reachable_exits 129/33/34; bottleneck_on_route 93/25/21; heterogeneous_urgency 51/10/12; nearest_exit_overload 43/18/4; nearest_exit_cut_by_fire 24/3/7; isolated_people 21/6/3 |

Full per-split counts: `index.json` → `coverage`.

## 8. CPU cost

Full `p1-02-v1` build, Python 3.14.7, Windows 11, one core: **30.3 s** for 120 families and 355 scenarios. 1395 lineage attempts; accepted attempt mean 48 ms (p95 121 ms, includes the probe), rejected attempt mean 10.5 ms. The probe dominates the accepted cost (simulator `_compute_trapped` per step); the probe bound of 20 steps keeps it small. `verify_split_files` costs one full build (CI: about 30 s per Python version). No GPU or Modal call. Spend: $0.

## 9. Rejections and failures (`p1-02-v1`)

- Family pool: 120 slots for 120 families; 19 rejected attempts (`insufficient_exit_cells` 9, `exit_inaccessible` 7, `fragmented_layout` 3); no `duplicate_family` or `near_duplicate_family` in this plan (the mechanism is tested with a forced look-alike plan).
- Lineages: 1040 rejected attempts, all recorded: `label_mismatch:feasible` 564, `label_mismatch:difficult` 369, `label_mismatch:unreachable` 103, `no_ignition_cells:bottleneck` 4.
- Failed lineages (bounded attempts used, listed in the manifests, not replaced): 3 in train, 2 in locked test. Split sizes are therefore 189 and 70, not 192 and 72. `test_failed_lineages_are_kept_with_reasons_and_counts` pins these counts.

| Split | Family | Map | Exits | Pocket / cut cells | Reasons (60 attempts) | Extra diagnostic attempts 60–1059 |
|---|---|---|---|---|---|---|
| train | `fam-8280f8fcac9878ea` | open room 9×14 | 4 | 0 / 0 | feasible 53, difficult 7 | 0 / 1000 accepted |
| train | `fam-a0a3780b6d702f75` | corridor 27×19 | 2 | 0 / 0 | feasible 30, difficult 30 | 9 / 1000 |
| train | `fam-cb18a7ae2d90994a` | open room 23×22 | 4 | 0 / 0 | feasible 42, difficult 18 | 0 / 1000 |
| locked | (sealed) | open room 26×52 | 4 | 0 / 0 | feasible 32, difficult 28 | 0 / 1000 |
| locked | (sealed) | corridor 11×24 | 4 | 0 / 0 | feasible 53, difficult 7 | 1 / 1000 |

**Root cause: a near-infeasible family and target pair, not a generator defect.** The target label is sampled per lineage before the family is known. On a map with no pocket, `unreachable` must come from `fire_cut`. With no cut cell, `fire_cut` can only use `near_exit` ignitions (1–3 cells within distance 3 of an exit). A step-0 trap then needs either every exit on fire (impossible with 4 exits and at most 3 ignitions; with 2 exits it needs flammable exits and both exit cells drawn), or an exact 2–3 cell ring around a corner or dead end that holds a person. An exhaustive search confirms that such rings exist on all 5 maps (smallest: 2 cells, cutting off 1 cell; 3 cells for the 26×52 hall), so the request is feasible in principle. The measured acceptance rate is 0–0.9 % per attempt, so 60 attempts fail as expected. No wrong label or invalid scenario was produced. The extra attempts were diagnostic only; nothing was added to any manifest and no seed was changed.

**Accepted limitation (coverage).** Of 46 accepted `unreachable` scenarios, 45 are on maps with a pocket or a cut cell; 1 is on a pocket-free, cut-free map with one exit. Open, well-connected maps with 2 or more exits have no `unreachable` example. A future version could choose the target after the family, or add a fire-cut mechanism that places a ring of ignitions. Either changes generation semantics and needs a new plan version (§10).

## 10. Limitations and open questions

1. **Locked test is procedural, not blind.** The plan is public, so anyone can rebuild the locked split. The public split stays as an auditable development artifact. A blind set uses a private salt (§11); it is BLOCKED until the salt and the evaluator environment exist.
2. **Near-duplicate rules are threshold rules** (§5 similarity classes). Sparse halls (< 5 % walls) are compared only with their exits. In a v2 plan, record `wall_geometry_audit` in the index and use the wall rule in the family pool too (`fill_family_pool(wall_rule=True)`); both change the frozen output, so they are not applied to `p1-02-v1`.
3. **`unreachable` coverage is confined** to maps with a pocket, a cut cell or one exit (§9).
4. **Labels ignore congestion** beyond the exit capacity bound. `feasible` does not mean zero casualties for any policy.
5. **Labels use the true fire trajectory** for the scenario seed. They are evaluator-private and never policy inputs.
6. **Probe is bounded** to 20 steps; longer-horizon fire equality relies on the `spread` equivalence test.
7. **In-distribution splits.** All splits share one spec; generalization is across unseen families of the same archetypes. An out-of-distribution slice (larger grids, new archetype) is not in `p1-02-v1`.
8. **Grid scale** stops at short side 30 for CPU cost. Larger maps are supported by the spec.
9. No simulator bug was found. Observation: `Simulator._compute_trapped` runs a full BFS per step and dominates probe cost; it is a performance note, not a rule change.
10. `PHASE_CHECKPOINT.json` and `docs/HLD.md` are not edited here, to avoid conflicts with open PR #22.

## 11. Blind evaluation (readiness: BLOCKED)

`ember_v1.gen.blind` implements a blind test set. No salt exists. None is in the repository, CI or any test fixture; each test makes a throw-away random salt in a temporary directory. Readiness stays **BLOCKED** until steps 1–3 below are done by a named evaluator.

**Mechanism.** Blind family seeds are `HMAC-SHA256(salt, "ember_v1/blind/<plan_id>/family/<slot>")` (first 8 bytes). Lineage and simulator seeds follow from the family seed as in §2. The spec is the public plan's spec (in-distribution). Without the salt, nobody can rebuild the blind scenarios.

**Guards in code.** `load_secret` refuses a salt file inside the repository, refuses to run when `CI` or `GITHUB_ACTIONS` is set, needs ≥ 32 bytes, and has no default. `BlindSecret` has a redacted `repr` and cannot be pickled. `_write_private` refuses a private manifest path inside the repository and never overwrites. `splits.load_manifest` and `splits.materialize` refuse blind manifests; only `blind.materialize_blind` with the matching salt regenerates them. A test asserts that no source module imports `ember_v1.gen.blind`. `.gitignore` blocks `*.salt`, `*.salt.hex`, `data/blind/` and `blind_private*.json`.

**Independent disjointness check** (`build_blind`). It rebuilds the public plan from `index.json`, requires every rebuilt manifest hash (all four public splits, including the procedural locked split) to equal the committed one, rejects blind family candidates that duplicate a public or earlier blind family (topology hash, exit-aware rule and wall rule), then runs `audit` and `wall_geometry_audit` over public plus blind scenarios. Any violation stops the build.

**What may be published.** Only the commitment that `build_blind` returns: blind version, plan ID, generator revision, spec hash, public index hash, salt commitment (`sha256` of the salt with a prefix), private manifest hash, family / scenario / failed-lineage counts, the disjointness result, and the number of maps audited. No IDs, seeds, labels, coverage or per-scenario data. After the final evaluation, only preregistered aggregate metrics may be published. Revealing the salt afterwards lets anyone check the commitment.

**Future setup (exact steps).**

1. The evaluator makes the salt on the evaluator machine, outside the repository: `python -c "import secrets; print(secrets.token_hex(32))" > /secure/ember_blind_v1.salt.hex`; restrict it to the evaluator account (`chmod 600`). Never paste it into a chat, issue, CI secret, notebook, log or training environment.
2. `python -m ember_v1.gen.blind --salt-file /secure/ember_blind_v1.salt.hex --public-index data/splits/p1-02-v1/index.json --plan-id blind-v1 --families <n> --lineages-per-family <k> --private-out /secure/blind-v1/private.json`. Commit only the printed commitment (for example `data/splits/blind-v1.commitment.json`) and record it in the ledger below.
3. Keep `private.json` and the salt on the evaluator machine. Teachers, training jobs, distillation and hyperparameter selection never run there and never get these files.
4. Run the final evaluation once, after all choices are frozen: `verify_blind`, then `materialize_blind`, then the preregistered metrics. Publish aggregates only.

**Exposure rule.** A blind set is exposed when any person or job inspects its scenarios or per-scenario results, or uses any of its results to choose a model, teacher, checkpoint, threshold or hyperparameter. An exposed set is never reported as a fresh blind result. Mark it exposed in the ledger and make the next blind set with a new salt and a new plan ID.

| Blind plan | Commitment | Status |
|---|---|---|
| (none) | — | BLOCKED: no salt provisioned |

## 12. Next step

Review this PR. After merge, teacher work (#7–#9, #23, #24) may load `dev` with `load_manifest` + `materialize`. The procedural locked split stays sealed for development checks only. A blind result needs §11. Bulk S1 generation stays blocked (#25, #10).
