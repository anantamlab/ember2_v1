# P1-03: T0 fire-aware shortest-path baseline (issue #7)

**Status:** implemented and tested on CPU. T0 is a simple diagnostic baseline. It is **not** a claim of teacher optimality. T1–T4 and the multi-teacher coordinator (#25) are not implemented. No GPU, Modal, model training, distillation, S1 generation or EMBERBench use. The frozen `p1-02-v1` split manifests are unchanged.

## 1. Architecture

Package `ember_v1.teacher` (standard library only):

| Module | Content |
|---|---|
| `interface.py` | `ReferencePolicy` base class, `PolicyIdentity`, `PolicyDecision`, `POLICY_INTERFACE_VERSION = policy-interface@1.0.0`. T1–T4 implement the same interface later. |
| `grid.py` | Shared public-grid utility, `GRID_UTILS_VERSION = grid-utils@1.0.0`: `PublicGrid` (from the observation only) and `exit_distance_field` (multi-source BFS). |
| `t0.py` | `T0ShortestPath`: name `T0`, policy `fire_aware_shortest_path`, version `1.0.0`. |
| `controls.py` | Baseline controls `C-stay`, `C-random`, `C-greedy`. Controls are not teachers. |
| `evaluate.py` | Closed-loop DEV harness: `python -m ember_v1.teacher.evaluate`. |
| `bench.py` | Decision latency: `python -m ember_v1.teacher.bench`. |

Policy modules (`interface`, `grid`, `t0`, `controls`) import only `ember_v1.contract`, `ember_v1.sim.movement` (the public resolver) and `ember_v1.sim.canonical`. A test asserts that they import nothing from `ember_v1.gen`, `sim.simulator`, `sim.scenario`, `sim.rng` or `sim.fire`. Only the harness (`evaluate.py`) builds simulators and loads DEV.

### 1.1 Decision record

`policy.decide(observation)` returns a `PolicyDecision`:

| Field | Content |
|---|---|
| `actions` | Complete map: every active person ID → one allowed action. Checked with `contract.validate_actions`; never repaired. |
| `teacher` | `name`, `policy`, `version`, `config`, `config_hash` (SHA-256 over name, policy, version and configuration). Each policy has its own identity. |
| `diagnostics` | Policy-specific. T0: per person `action`, `route_found`, `reason`, `target_exit`, `distance`, `equal_first_moves`, `predicted_outcome`; summary `safe_exits`, `no_route`, `target_exit_load`, `predicted_outcomes`. |
| `reproducibility` | `interface`, `utilities` (versions), `step`, `observation_digest`, `policy_seed` (`null` for T0), `deterministic: true`. |
| `runtime` | `wall_ms` (`perf_counter`), `cpu_ms` (`thread_time`; OS tick resolution, about 15.6 ms on Windows, so use it summed, not per decision). |

`deterministic_dict()` gives every field except `runtime`. `act(obs)` (and `__call__`) gives the action map only, so a policy plugs into `sim.run_episode`.

## 2. T0 algorithm

1. Validate the observation with `contract.validate_observation` (strict whitelist).
2. Safe exits: exits that are not burning now.
3. `exit_distance_field`: BFS from all safe exits over 4-neighbour cells that are in the grid, not walls and not burning now. Each cell gets `(distance, exit_id)`. `exit_id` is the lowest ID among the nearest exits (layer-by-layer BFS; a cell takes the lowest label of its neighbours one layer closer).
4. Each active person, independently:
   - target exit: the nearest safe exit; equal distance → lowest exit ID (string order), not position;
   - action: the first move in the order **north, south, east, west** whose cell is one step closer to that target exit;
   - no path to any safe exit → **`stay`** with `route_found: false` and `reason: no_safe_exit_reachable`. `stay` is the documented safe default: T0 never walks into fire, a wall or off the grid to "do something".
5. Joint prediction: T0 runs the full action map through the exact resolver `sim.movement.resolve_moves` with the observed positions and burning set. It records `predicted_outcome` per person. **T0 does not change any action from this prediction.**

T0 has no tunable parameters. Its configuration is fixed and recorded (`T0ShortestPath({...})` raises).

**Not in T0 (by design, for T1–T4):** reservations, congestion or exit-load balancing, other people as obstacles, fire forecast, urgency order, stochastic rollouts, ACO. The optional spread-risk estimate of issue #7 is **not** in T0 v1. T0 avoids only cells that burn now. (HLD §4 D4 and P1_TEACHER_DIRECTION §1 said "soon-burning cells" for T0; both now say current burning cells.)

## 3. Joint movement

T0 plans each person alone. A set of shortest paths is not a valid group plan. The simulator rules are unchanged; blocked moves are recorded as baseline limitations.

Facts that follow from the construction (tested on 300 random closed-loop episodes, > 1000 steps):

- Every intended move goes to a cell with distance one less. The intended-move graph has no cycles, so T0 **never causes `blocked_cycle`** (no swaps or rotations).
- Every move target is in the grid, not a wall and not burning now: T0 never causes `blocked_boundary`, `blocked_wall`, `blocked_fire` or an `entered_fire` casualty.
- `predicted_outcome` equals the simulator's movement outcome for every person and step (0 mismatches on DEV; the resolver sees the same burning set, and fire spread comes after movement).
- `blocked_conflict` and `blocked_occupied` do happen. They are T0's main group limitation.

Unit tests with the resolver (`tests/test_teacher_t0.py`): two people with one target cell (lowest ID wins, the other `blocked_conflict`); a chain behind a conflict loser (`blocked_occupied`); follow chains in both ID orders (all `moved`); multiple exits (the group splits); an unreachable person (`stayed`) next to a mover; a door bottleneck; a hand-made swap and 4-rotation (resolver gives `blocked_cycle`) and the matching T0 case (each person goes to its own exit, no swap).

Routing tests: move tie-break order; exit tie-break by ID (not by position); detour around burning cells; burning exit skipped; four no-route cases (only exit burning, exit walled off, only path burning, exit sealed); exact active IDs after an evacuation.

## 4. Observation-only and deterministic guarantees

| Guarantee | Mechanism | Test |
|---|---|---|
| Only `Observation.to_dict()` fields | `validate_observation` rejects any extra key at the root, in `map`, `rules` or a person (seed, RNG state, scenario ID, trapped, difficulty, label, lineage, family, split, future burning, teacher outcome) | `test_private_fields_are_rejected_not_ignored` (all four policies × 11 keys × 4 places) |
| Harness passes only `observe()` | `evaluate.run_episode` calls `policy.decide(sim.observe())`; private state is read only for metrics after the step | `test_harness_passes_only_the_public_observation` |
| No simulator or generator imports in policy code | AST import audit | `test_policy_modules_do_not_import_simulator_internals_or_generator` |
| Same public observation → same decision | T0 has no state and no randomness | three seeds and scenario IDs with deterministic fire give identical full episode logs; stochastic fire with four seeds gives different trajectories, an identical step-0 decision, and a fresh T0 repeats every logged decision |
| Controls | `C-random` RNG = f(policy seed, observation step) | `test_random_control_is_a_function_of_seed_and_observation` |
| Results reproduce | `deterministic_digest` over all rows without timing | `test_committed_rows_regenerate_exactly` (first 3 DEV scenarios, all policies), digest consistency test |

## 5. Independent DEV comparison

**Setup.** Plan `p1-02-v1`, split `dev` only (16 families, 48 scenarios, 1734 people). `load_dev` verifies the index and manifest hashes, regenerates every scenario and refuses any other split (`PermissionError`). Train, calibration, the procedurally locked test, blind sets and EMBERBench are not read. Each scenario runs once per policy with the scenario's own simulator seed, so the fire trajectory is identical for all policies (fire never depends on actions). `C-random` seed: `int(sha256({control, plan_id, position})[0:16 hex])`. Code revision `fcabdd9` (clean `src`), Python 3.14.7, Windows 11, AMD64. Results: [`data/baselines/p1-03-t0-dev-v1.json`](../data/baselines/p1-03-t0-dev-v1.json) (per scenario and aggregates), deterministic digest `sha256:a9efeff6…5278838d`.

Controls:

- `C-stay`: every active person stays.
- `C-random`: uniform over the five allowed actions.
- `C-greedy`: target = Manhattan-nearest safe exit (ties: lowest ID); first Manhattan-reducing move (larger axis gap first) to a cell that is in the grid, not a wall and not burning now; else stay. No path search.

### 5.1 Aggregates (48 scenarios)

| Metric | T0 | C-greedy | C-random | C-stay |
|---|---|---|---|---|
| People | 1734 | 1734 | 1734 | 1734 |
| Evacuated | 1210 | 814 | 93 | 0 |
| Casualties | 507 | 840 | 1510 | 1581 |
| Unresolved at termination | 17 | 80 | 131 | 153 |
| Trapped at termination | 17 | 78 | 121 | 135 |
| Mean evacuated fraction | 0.722 | 0.485 | 0.083 | 0.000 |
| Mean casualty fraction | 0.273 | 0.464 | 0.820 | 0.866 |
| Episodes all resolved (of 48) | 46 | 40 | 38 | 35 |
| Mean completion step (resolved episodes) | 31.6 | 40.6 | 56.9 | 55.2 |
| Mean episode steps | 34.4 | 59.4 | 73.1 | 75.1 |
| Blocked moves | 14294 | 29266 | 24134 | 0 |
| Collisions (occupied + conflict + cycle) | 14294 | 29266 | 9339 | 0 |
| Blocked per 100 attempted moves | 38.0 | 66.9 | 35.5 | – |
| `blocked_conflict` | 3512 | 3063 | 2031 | 0 |
| `blocked_occupied` | 10782 | 26203 | 5358 | 0 |
| `blocked_cycle` | 0 | 0 | 1950 | 0 |
| `blocked_wall` / `blocked_boundary` / `blocked_fire` | 0 / 0 / 0 | 0 / 0 / 0 | 11661 / 2338 / 796 | 0 / 0 / 0 |
| Casualties `entered_fire` / `fire_spread` | 0 / 507 | 0 / 840 | 316 / 1194 | 0 / 1581 |
| Decisions | 1653 | 2853 | 3511 | 3604 |
| Failures / invalid-action episodes | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Decision wall ms mean / p50 / p95 / max | 2.32 / 1.99 / 5.43 / 17.3 | 1.90 / 1.36 / 5.00 / 13.3 | 1.62 / 1.06 / 4.54 / 12.6 | 1.49 / 1.02 / 4.21 / 21.3 |
| Episode CPU s mean / max (policy + simulator) | 0.145 / 0.672 | 0.229 / 2.047 | 0.248 / 2.016 | 0.240 / 1.859 |

`C-stay` latency (1.0 ms p50) is the interface cost: observation validation and the observation digest. T0 episodes are shorter, so T0 uses less CPU per episode than the controls.

### 5.2 Paired comparison (same scenario, same seed)

| Pair | Metric | Mean diff (T0 − control) | T0 better / equal / worse | Sign test p (two-sided) |
|---|---|---|---|---|
| T0 vs C-greedy | casualties | −6.938 | 32 / 13 / 3 | 4.2e-07 |
| T0 vs C-greedy | evacuated fraction | +0.237 | 33 / 13 / 2 | 3.7e-08 |
| T0 vs C-greedy | unresolved | −1.312 | 7 / 41 / 0 | 0.016 |
| T0 vs C-random | casualties | −20.896 | 42 / 6 / 0 | 4.5e-13 |
| T0 vs C-random | evacuated fraction | +0.639 | 44 / 4 / 0 | 1.1e-13 |
| T0 vs C-random | unresolved | −2.375 | 10 / 38 / 0 | 0.002 |
| T0 vs C-stay | casualties | −22.375 | 41 / 7 / 0 | 9.1e-13 |
| T0 vs C-stay | evacuated fraction | +0.722 | 45 / 3 / 0 | 5.7e-14 |
| T0 vs C-stay | unresolved | −2.833 | 13 / 35 / 0 | 0.00024 |

The p-values are descriptive (one DEV set, no correction for multiple comparisons). They are not used to select anything.

Mean evacuated fraction by generator label (evaluator-side only; the label never reaches a policy):

| Label | Scenarios | T0 | C-greedy | C-random | C-stay |
|---|---|---|---|---|---|
| feasible | 28 | 0.922 | 0.584 | 0.103 | 0.000 |
| difficult | 14 | 0.356 | 0.294 | 0.051 | 0.000 |
| unreachable | 6 | 0.642 | 0.468 | 0.070 | 0.000 |

### 5.3 Where T0 loses people

T0's 507 casualties, by T0's last decision for that person (`casualty_context`):

| Context | Casualties | Meaning |
|---|---|---|
| `no_route/stayed` | 237 | Fire cut every safe exit path; T0 stayed (default). 40 people were trapped at step 0; the rest were cut off during the episode. |
| `route/moved` | 156 | The person moved into a cell that ignited in the same step. T0 has no fire forecast. |
| `route/blocked_occupied` | 66 | Queued behind a person who did not move. |
| `route/blocked_conflict` | 48 | Lost a same-cell contest (lowest ID wins). |

T0 is worse than `C-greedy` on casualties or evacuated fraction in 3 scenarios (positions 1, 4, 32). In positions 1 and 32 (stochastic fire, `enter_fire = casualty`) most T0 casualties are `no_route/stayed`. In position 4, T0 evacuates more people (31 vs 13) but has more casualties (15 vs 10), because `C-greedy` stalls more people who stay unresolved.

## 6. CPU decision latency

`python -m ember_v1.teacher.bench`: `T0.decide` (validation, BFS, resolver diagnostics) on open rooms (`sim.bench.room`, two exits, one ignition, stochastic fire). For each grid and N: 5 layouts × up to 10 closed-loop observations × 5 repetitions. Wall-clock ms, measured on the machine in §5. Raw output: [`data/baselines/p1-03-t0-latency.jsonl`](../data/baselines/p1-03-t0-latency.jsonl).

| Grid | N | Mean active | Samples | Median | Mean | p95 | Min | Max |
|---|---|---|---|---|---|---|---|---|
| 32×32 | 1 | 1.0 | 235 | 2.28 | 2.14 | 2.74 | 1.43 | 3.60 |
| 32×32 | 10 | 9.1 | 250 | 2.45 | 2.28 | 3.10 | 1.52 | 4.99 |
| 32×32 | 50 | 46.1 | 250 | 2.81 | 2.83 | 3.80 | 1.94 | 4.56 |
| 32×32 | 100 | 94.7 | 250 | 4.12 | 3.98 | 5.34 | 2.55 | 6.76 |
| 32×32 | 200 | 191.7 | 250 | 5.93 | 5.66 | 7.23 | 3.73 | 7.95 |
| 64×64 | 1 | 1.0 | 250 | 9.10 | 8.65 | 11.11 | 5.80 | 15.05 |
| 64×64 | 10 | 9.6 | 250 | 8.35 | 8.53 | 11.32 | 5.80 | 12.78 |
| 64×64 | 50 | 49.2 | 250 | 10.52 | 10.14 | 12.65 | 6.58 | 19.72 |
| 64×64 | 100 | 98.6 | 250 | 11.35 | 10.92 | 13.85 | 7.18 | 20.11 |
| 64×64 | 200 | 197.1 | 250 | 13.64 | 13.35 | 16.69 | 8.43 | 23.77 |

Cost grows mainly with grid area (one BFS per decision) and slowly with N. For 32×32, N = 1, one layout ends after fewer than 10 decisions, so that row has 47 observations.

## 7. Limitations

1. **No coordination.** Several people target the same cell or queue behind each other: 38 blocked moves per 100 attempted moves on DEV, and 114 casualties were blocked on their last step.
2. **No fire forecast.** 156 casualties moved into a cell that ignited in the same step.
3. **No-route default is `stay`.** A person with no safe exit path waits; that person is never moved toward a possible future opening. 237 casualties were in this state.
4. **Current-fire distance only.** Exits on a short path near the fire are preferred over safer, longer paths.
5. **Single-person tie-breaks are fixed** (north, south, east, west; lowest exit ID). They are deterministic but can send many people the same way.
6. **One DEV set, one platform.** Latency comes from one Windows machine with Python 3.14.7. The comparison is diagnostic; it is not evidence about T1–T4 or EMBERBench.

## 8. Reproduce

```
python -m pip install -e '.[test]'
python -m ember_v1.teacher.evaluate            # writes data/baselines/p1-03-t0-dev-v1.json (~45 s)
python -m ember_v1.teacher.bench                # prints the latency JSON lines (~30 s)
python -m pytest -q tests/test_teacher_t0.py tests/test_teacher_eval.py
```

Timing fields differ between runs. Every other field must match (`deterministic_digest`).
