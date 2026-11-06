# P1-01 (issue #5): independent fire-evacuation simulator — validation

**Objective (changed 2026-11-06):** an independently validated, general-purpose fire-evacuation simulator.
It is **not** a simulator tuned to reproduce EMBERBench bit for bit.
EMBERBench's public documents define only the public observation/action interface and the essential movement rules that we keep compatible.

| Claim | Status | Evidence |
|---|---|---|
| Internal correctness (movement, collisions, fire, RNG, replay, invariants, termination) | **Tested** (CPU, CI) | §3, §4 |
| Public observation / action interface compatibility | **Tested where applicable** | §3 rows "Observation", "Exactly one action"; `ember_v1.contract` accepts every observation unchanged |
| Exact EMBERBench implementation equivalence | **NOT CLAIMED** | §7 |
| Real-world physical fidelity (crowd or fire physics) | **NOT ESTABLISHED** | The model is a discrete grid abstraction (§5) |
| Independent cross-environment generalization | **TO BE EVALUATED** | Needs independently defined variations (§6, issue #6 and later) |

## 1. Scope and provenance

- Code: `src/ember_v1/sim/` (package `ember_v1.sim`). Standard library only. It imports nothing from EMBERBench, the decision model (`agent`, `p0b`, `encoding`, ...) or Modal. Tests check the imports.
- Specification of the compatible interface and V1 rules: the public EMBERBench documents at commit `423143a2a9edc9382c58a2e4a10700252f7b38c5` (`docs/simulation-contract.md`, `docs/v1-fairness-spec.md`, `docs/scenario-schema.md`, `docs/model-interface.md` §8). We use them as a written rule set, not as an implementation to copy.
- Not used for the implementation or the tests: EMBERBench source code, tests, fixtures, golden traces, benchmark scenarios, hidden or private data, evaluation runs or leaderboard results.
- SplitMix64 follows the published reference algorithm (Steele, Lea and Flood, 2014). A test checks it against the published reference outputs for state `1234567`.

## 2. Architecture

| Module | Role |
|---|---|
| `rng.py` | `SplitMix64` (one `next_u64()` = one draw), `bernoulli(p)` = `(u >> 11) / 2**53 < p`, `stream_seed(seed, label)` = `sha256("emberbench/v1/{seed}/{label}")[:8]` big-endian, `draws_between()` (exact draw count from two states). |
| `scenario.py` | Strict schema 1.0 parser, fail closed. Canonical form (sorted lists), `content_hash()`. |
| `fire.py` | `Terrain`, frontier (candidates and edges in row-major order), deterministic and stochastic spread. Sees only terrain, burning set, step and the fire stream. |
| `movement.py` | `resolve_moves()`: phase 3 validation, then collision rules 1–4. Pure function, O(N) per step. |
| `simulator.py` | `Simulator`: `observe()`, `step(actions)` (phases 2–9), trapped set, termination, step records, SHA-256 state and transition hash chain. `run_episode`, `replay`, `verify_replay` (full episode), `first_divergence` (prefix only). |
| `canonical.py` | Canonical JSON (sorted keys, no whitespace, ASCII) and `sha256:` digests. |
| `bench.py` | `python -m ember_v1.sim.bench`: CPU step time. |

A step record holds `step`, `actions`, `movements` (`outcome`, `from`, `to`), `ignitions`, `fire_draws`, `evacuations` (ID → exit), `casualties` (ID → cause), `termination_reason` (null until the last step), `state_hash` and `transition_hash`. `transition_hash = sha256(canonical({prev, record}))`. The chain starts from the scenario hash, the seed and the initial state. These hashes identify runs of this simulator only.

## 3. Rules and evidence

Tests: `tests/test_sim_movement_golden.py` (G-M), `tests/test_sim_rules_golden.py` (G-S), `tests/test_sim_properties.py` (P), `tests/test_sim_contract.py` (C), `tests/test_sources.py` (S). Golden expectations were derived by hand from the written rules, not from any implementation.

| Rule | Implementation | Evidence |
|---|---|---|
| Phase 3: stay, boundary, wall, `enter_fire=blocked` | `movement.py` | G-M `stay`, `boundary*`, `wall`, `fire_blocked`, `fire_casualty_moves` |
| Rule 1: target holds a non-mover → `blocked_occupied` | one pass (A2) | G-M `occupied_by_*`, `contest_by_two_on_stayer_is_occupied` |
| Rule 2: lowest ID wins, string order, intent only, loser not re-awarded | `min(ids)` | G-M `contest_*` (incl. `P10` < `P2`), `loser_not_reawarded_*` |
| Rule 3: follow chains | chain walk | G-M `follow_chain_*`, `chain_ends_at_*`, `exit_queue_flows` |
| Rule 4: swaps and rotations blocked; lower-ID outsider breaks a cycle | cycle detection | G-M `swap_*`, `rotation_4/6_blocked`, `*_outsider_*` |
| Independent movement oracle | least-fixpoint formulation (different algorithm) | P `test_resolver_matches_fixpoint_oracle` (3000 dense random steps) |
| Order independence, rename invariance | sorted iteration | G-M reversed-order copy of every case; P shuffled order and order-preserving rename (3000 cases) |
| Exits: one person per exit per step, queues flow | phase 6 | G-S `test_exit_queue_*`, `test_two_people_*`, `test_two_adjacent_exits_*` |
| Evacuation only on exits not burning at step start | `burning0` check | G-S `test_entering_burning_exit_*`, `test_exit_reached_in_the_step_fire_reaches_it_*` |
| `enter_fire` blocked/casualty; cause `entered_fire` vs `fire_spread` | phases 3, 8 | G-S `test_burning_exit_blocked_policy_*`, `test_follow_chain_into_fire_*`, `test_stay_on_igniting_cell_*` |
| Spread only on `t % interval == 0`; no burn-out | `fire.spread` | G-S `test_spread_interval_and_no_burn_out`; P `test_episode_invariants` |
| Fire locality: a cell ignites only next to a cell burning at step start | frontier | P `test_fire_locality_people_conservation_and_termination` (300 episodes) |
| Walls never burn; non-flammable exits never ignite | `Terrain.flammable` | G-S `test_deterministic_frontier_*`; P invariants |
| Stochastic: edge order, every edge draws, `p=0`/`p=1`, draws = edges on spread steps, else 0 | `fire.spread` | G-S `test_stochastic_*`, `test_deterministic_model_never_draws`; P `fire_draws == edge_count` per step |
| Exact Bernoulli threshold, one draw per call | `rng.py` | G-S `test_bernoulli_*`, `test_splitmix64_reference_vector`, `test_stream_seed_formula` |
| Fire is exogenous (common random numbers) | fire never sees people | P `test_fire_is_exogenous_*` (60 scenarios × 3 policies) |
| Physical/logical invariants: no shared cells, moves ≤ 1 cell, resolved people frozen, monotone burning, people conserved | phases 5–8 | P `test_episode_invariants` (300 episodes), `test_fire_locality_*` |
| No future-information leakage | `observe()` before the step | P `test_no_future_information_in_observations`: observations equal across seeds while fire histories agree, and `burning` equals the recorded history (150 episode pairs); C `test_observation_*` |
| `observe()` is pure and returns a copy | fresh dict | P `test_observe_is_pure` |
| Trapped set: private, ignores people | BFS from non-burning exits | G-S `test_trapped_is_private_and_ignores_people` |
| Termination: `all_resolved` wins over `max_steps`; a terminated simulator raises | phase 9 | G-S `test_all_resolved_*`, `test_max_steps_*`; C `test_terminated_*`; P `test_fire_locality_*` |
| Exactly one action per active person; failed step changes nothing | `_validate_actions` | C `test_invalid_actions_*`, `test_actions_for_resolved_people_are_rejected` |
| Observation: exact whitelist, canonical order, no seed, RNG, scenario ID, description or trapped | `observe()` | C `test_observation_*` (accepted unchanged by `ember_v1.contract.validate_observation`), `test_policy_receives_only_the_observation` |
| Deterministic replay and hash chain | `replay` | C `test_same_seed_*` (40 random episodes), `test_records_are_*`, `test_seed_changes_*` |
| Full-episode replay verification | `verify_replay` | §4 |
| Scenario schema 1.0, fail closed (supported domain) | `Scenario.from_dict` | C `test_scenario_rejects_invalid_input` (36 cases), canonical-order test |

Six deliberate mutants of the rules (highest ID wins, cycles allowed, no rule 1, evacuation from burning exits, one draw per candidate, single casualty cause) each fail the suite.

## 4. Replay completeness

`first_divergence()` is a prefix check. It returns `None` for a valid prefix, so it can not prove that a whole episode was replayed. It keeps that behaviour and its docstring now says so.

`verify_replay(scenario, seed, records, partial=False)` returns a `ReplayVerdict`:

| Status | Meaning |
|---|---|
| `complete` | Every record matches type-strictly (canonical JSON) and the replay terminates exactly at the last record. |
| `partial` | Only with `partial=True`: every record matches and the episode has not terminated. Never counts as complete. |
| `diverged` | `reason` = `incomplete` (records end before termination), `step_after_termination`, `record_mismatch`, `invalid_actions` or `malformed_record`; `step` is the first failing step. |

Regression tests (C): complete episode; missing final step (prefix check accepts it, full check rejects it); extra step after termination; modified intermediate step (4 fields); wrong or early termination reason; incorrect transition hash; float-for-int record value; malformed records; legitimate partial trace marked partial.

**Teacher gate in code:** `ember_v1.sources.ingest_zip(quality="teacher_verified")` now requires each trace to carry its whole episode (`episode = {scenario, seed, records}`, `step`). The episode must pass `verify_replay(partial=False)`; the trace observation and actions must equal the replayed observation and actions of that step. `training_records()` raises on a `teacher_verified` record without complete replay evidence. Tests: S `test_teacher_verified_*` (14 failure cases), `test_training_records_rejects_*`.

**Trust boundary.** Replay consistency proves only that the recorded actions, under the recorded scenario and seed, produce the recorded transitions in this simulator. It does **not** authenticate teacher identity (any program can write a replay-consistent trace with any teacher name), label quality (a poor policy also replays) or data provenance (frozen teacher code, config and seed; approved generator split). Those protections are open work under issue #10. Replay verification alone does not authorize bulk S1 generation.

## 5. Assumptions and uncertainty

The simulator is a discrete abstraction. It makes these assumptions explicitly:

- 4-connected grid, one person per cell, unit moves, synchronous steps. No speed, body size, panic, group or crowd-pressure model.
- Fire: binary burning state, 4-neighbour spread, no burn-out, no smoke, no heat, static terrain. Fire is exogenous to people.
- Randomness only in the fire stream, seeded per run. Movement is deterministic.
- Full observability in V1: every burning cell is visible. No sensor noise.

Reading choices where the written rules leave room (each one is our decision, documented, and tested; none is claimed to match another implementation):

| # | Item | Decision | Effect |
|---|---|---|---|
| A1 | Stream seed text and SplitMix64 output function | Seed in decimal; standard SplitMix64 mix. | Fire trajectories are defined by our documented formula. Bit-equality with any other simulator is not claimed. |
| A2 | Collision rule 1 | One pass over phase-3 non-movers. | Labels only: P `test_rule1_reading_does_not_change_positions` shows identical positions on 3000 cases for the iterative reading; some `blocked_conflict` labels differ. |
| A3 | Hashes | This simulator's canonical form. | Hashes identify our runs only. |
| A4 | Integer `spread_probability` (e.g. `1`) | Becomes `1.0` in the observation. | Observation bytes only. |
| A5 | Person and exit IDs | Must match `^[A-Za-z0-9_-]{1,64}$`. | Stricter than the minimum; fail closed; same pattern as `ember_v1.contract`. |
| A6 | Action type | Plain strings only. | Interface choice; enums are rejected. |
| A7 | Spread computed before movement is applied | Applied in phase 7. | None, because fire is exogenous (P test). |

## 6. Extensibility review (no generator in this issue)

| Variation | Current support | Needed for a future extension |
|---|---|---|
| Fire spread probabilities | One scalar `p` per scenario, any value in [0, 1]. | Per-cell or per-edge probability maps: a new fire model name and schema field. `draws_between` already counts any number of draws. |
| Fire patterns and models | `spread()` dispatches on the model name (`deterministic`, `stochastic`). | Replace the string dispatch with a fire-model interface `(terrain, burning, step, rng) -> ignited`. Keep the exogeneity and draw-accounting invariants, and give each model its own golden tests. Burn-out, 8-neighbour or wind-biased spread each need a new model name. |
| Obstacles, layouts, multiple exits | Any grid up to 1024 × 1024, any walls, any number of exits. | None for static layouts. |
| Crowd size and congestion | O(N) resolver; measured to 500 people (§8). Exit throughput is one person per exit per step. | Exit capacity > 1 or per-exit capacity needs a rule change in phase 6 and new tests. |
| Dynamic hazards, blocked routes | Not supported: `Terrain` is frozen for the episode; only fire changes. | Time-varying terrain in the private state and hash, and a new observation field. This changes observation contract 1.0, so it needs a versioned contract. |
| Fire-model assumptions | Exogenous, fully observed. | Partial observability or people-dependent fire would break the common-random-numbers property; the test suite would need to state the new contract. |

These variations must be defined independently (issue #6 and later). They must not copy EMBERBench's scenario distribution, seeds or random streams.

## 7. Record: stopped EMBERBench comparison work

Before the change of objective, work on a side-by-side comparison started on 2026-11-06. It was stopped and removed. What happened:

- We located the public repository `anantamlab/EMBERBench` at commit `423143a2a9…` (Apache-2.0, package `1.0.0rc1`, simulator `0.2.0`). We read parts of its source to learn its public API (`simulator.py`, `rng.py`, and the parts of `movement.py`, `scenario.py`, `events.py`, `state.py` and `observation.py` that define enums, validation and `to_dict`).
- An uncommitted harness installed that package in a temporary CPU-only venv outside the repository. It ran locally generated synthetic cases through both simulators. One local run reported no mismatch.
- The harness, the clone, the venv and the results were deleted. Nothing from them is in this branch.
- We did not change any movement, fire or RNG code because of that work. The changes in this branch (`verify_replay`, `termination_reason` in step records, type-strict replay comparison, the teacher gate) are independent replay-integrity work.
- That run is not evidence. We do not use it to support any claim, and we do not use it to tune the simulator or teachers.

## 8. Gates

| Gate | State |
|---|---|
| G1 internal correctness suite (§3) on CPU in CI | PASS at this PR head (see PR #26 checks) |
| G2 full-episode replay verification required for `teacher_verified` (§4) | Implemented and tested |
| G3 supported domain: schema 1.0, V1 rules, fail-closed parsing | Implemented and tested |
| Exact EMBERBench parity | **Not a prerequisite. Not claimed.** |
| Teacher-generated training data | **BLOCKED** until G1–G3 pass on the reviewed and merged issue #5 code. G2 is necessary, not sufficient: bulk S1 generation also needs the generator (#6), the five reference policies (#7–#9, #23, #24), the five-teacher gate (#25) and the provenance protections (#10). **Not authorized by this PR.** |

## 9. CPU performance

`python -m ember_v1.sim.bench` (random actions, stochastic fire, includes `observe()`), Windows 11, Python 3.14.7, one core:

| Grid | People | Steps | Mean active | ms / step |
|---|---|---|---|---|
| 12×12 | 10 | 42 | 5.2 | 0.34 |
| 24×24 | 50 | 90 | 27.3 | 1.18 |
| 32×32 | 200 | 100 | 135.9 | 2.76 |
| 64×64 | 500 | 100 | 430.0 | 9.41 |

Single measurements, not a benchmark claim. T2 rollouts (K futures × H steps × 5 actions × N people) will multiply this cost; measure T2 per step in P1.

## 10. Next step

Review and merge issue #5. Then issue #6 (independent scenario generator) can define the variations in §6.
