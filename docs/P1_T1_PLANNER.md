# P1-04: T1 congestion-aware cooperative planner (issue #8)

**Status:** implemented and tested on CPU. Configuration frozen and preregistered (§3) before any DEV run. DEV results: §6. T1 is a reference planner, not a claim of optimality. T2–T4 and the #25 coordinator are not implemented. No GPU, Modal, model training, distillation, S1 generation or EMBERBench use. The frozen `p1-02-v1` split manifests are unchanged.

## 1. Architecture

| Module | Content |
|---|---|
| `teacher/t1.py` | `T1CongestionAware`: name `T1`, policy `congestion_aware_cooperative_planner`, version `1.0.0`. Implements `ReferencePolicy` (`policy-interface@1.0.0`). |
| `teacher/t1_eval.py` | Closed-loop paired DEV harness, T1 vs frozen T0: `python -m ember_v1.teacher.t1_eval`. |
| `teacher/bench.py` | Latency bench, now `--policy T0|T1` (default T0, unchanged). |

`t1.py` imports only `sim.movement` (public resolver), `teacher.grid.PublicGrid` and `teacher.interface`. It imports no T0 code, no simulator internals (`sim.simulator`, `sim.scenario`, `sim.rng`, `sim.fire`) and nothing from `gen` (AST test). T0 is unchanged; its DEV episodes are bit-identical to P1-03 (test compares final transition hashes).

T1 is stateless: every decision is a function of the validated observation and the frozen configuration. There is no memory between steps, no RNG and no seed (`policy_seed: null`).

## 2. Algorithm (one decision)

1. **Validate** the observation (`contract.validate_observation`, strict whitelist). Any private field is rejected.
2. **Public fire forecast.** Layer BFS from the burning cells over flammable cells (not wall; exits only if `exits_flammable`). Layer *l* is the earliest spread step that can ignite the cell. Spread steps follow the public rule `t % spread_interval == 0`. *Q(c, τ)* = probability that *c* burns after τ more steps:
   - deterministic: 1 if the *l*-th spread step is ≤ τ, else 0. This is exact for the public rule (test against the simulator);
   - stochastic, *l* = 1: 1 − (1 − p)^(k·n), *k* = burning neighbours now, *n* = spread steps in τ;
   - stochastic, *l* ≥ 2: P(Binomial(n, p) ≥ l), a single-front **lower bound** (optimistic).

   Hazard of being on cell *c* at the end of relative step τ is *Q(c, τ)*; for a safe exit it is *Q(c, τ−1)*, because evacuation (phase 6) comes before spread (phase 7). Nothing is sampled. No seed, RNG state or future fire is read.
3. **Static per-exit fields.** BFS from each safe exit over cells in the grid, not walls and not burning now. Other exits are absorbing (a person evacuates on any safe exit), so a field never passes through another exit.
4. **Congestion-aware exit assignment**, ascending person ID. For each of the `candidate_exits` nearest safe exits, walk the static shortest path (fixed N/S/E/W descent). On *congestion cells* (safe exits, and open cells with ≤ 2 open neighbours) a nominal space-time table holds one person per (cell, step). A later (higher-ID) person waits until a free step. This models exit throughput (one person per exit per step) and corridor queues in the simulator's lowest-ID-wins order. Cost of exit *e*:

   `arrival_with_delays + alt_exit_margin·[e ≠ nearest] + risk_weight·Σ hazard along the delayed path`

   An exit whose maximum path hazard is ≥ `risk_hard` is infeasible. T1 takes the best feasible exit (ties: static distance, then exit ID). If no exit is feasible, the least costly exit is still the target with `fire_feasible: false`; no route is invented. The chosen path is committed to the nominal table.
5. **Windowed cooperative space-time A\***, ascending person ID (*planning order*). State (cell, τ), τ ≤ `reservation_horizon`. Successors: north, south, east, west, stay. A successor is pruned if the cell is not open now, the vertex (cell, τ) is reserved by an earlier planner, the reverse edge is reserved (swap), or its hazard is ≥ `risk_hard`. Cost: 1 per step, + `risk_weight`·hazard, + `move_extra_cost` per move (< 1, so it never trades a step of time; it prefers waiting over equal-time shuffles). Heuristic: static distance to the assigned exit × (1 + `move_extra_cost`) (admissible). Goals: any safe exit (absorbing; + `alt_exit_margin` if not the assigned exit), or τ = horizon with that cost-to-go. Heap order (f, h, generation order).
   - **Follow chains in any ID order.** A person's current cell is free to earlier planners at τ = 1. If an earlier planner reserved my cell at τ = 1 (it follows me), I first search for a plan that leaves the cell (the earlier planner has precedence). Only if no full plan exists may I stay on my own cell at τ = 1. Step 6 checks that every leader really leaves.
   - **Budgets.** `node_budget_per_person`, `node_budget_total` (expansions). Per-person budget hit: best partial path (min f, then deepest), status `budget_partial`. Total budget gone: reservation-consistent greedy descent, `budget_fallback`. No conflict-free path to the horizon inside the window: best partial path, `boxed_partial` (a complete search result, not a failure). No admissible successor at all (every action reserved or hazard ≥ `risk_hard`): `no_successor_greedy` or `no_successor_least_hazard`.
   - **No route.** A person with no static path to a safe exit gets a *shelter* plan: move toward the reachable cells with the largest fire layer (latest earliest-ignition). `route_found: false`, `reason: no_safe_exit_reachable`, `target_exit: null`.
6. **Joint verification with the public resolver.** The full action map goes through `sim.movement.resolve_moves`, the simulator's own movement rule. *Planning order is not collision resolution:* the simulator always resolves contested targets, follow chains, occupied cells and cycles by its person-ID rules, and T1 never overrides them. A blocked planned move is repaired for at most `repair_rounds` rounds: a conflict loser, a person directly behind a stayer, or the highest ID of a closed cycle (swap or rotation) is re-planned with that τ = 1 cell forbidden. A move that is still predicted blocked after the last round becomes `stay`. End positions are asserted to be unchanged by that conversion (the resolver gives the same end cells).

**Complexity per decision.** Grid area *A*, exits *E*, people *N*, path length *L*, horizon *W*: fire BFS O(A); exit fields O(E·A); assignment O(N·K·(L + queue waits)); A\* bounded by `node_budget_total` expansions × 5 successors; repair ≤ `repair_rounds` × O(N) resolver calls. Measured latency: §7.

**What T1 does not do.** No stochastic rollouts or Monte Carlo (T2). No ACO (T3). No risk-based evacuation priority: the planning order is always ascending ID (T4). No learned parts. No memory between steps.

## 3. Preregistered frozen configuration

Frozen in `T1CongestionAware.default_config()` before any DEV run. Config hash (name + policy + version + configuration):

**`sha256:5047e942db3326706b6184c9f29e0b992078920f03c8a53adc8cc24258345361`**

A test pins this hash. Descriptive keys cannot be changed. Numeric keys can be overridden only to make a new identity (a new hash); any such run is exploratory and must be declared as such.

| Key | Value |
|---|---|
| `planning_order` | `ascending_person_id` |
| `reservation_horizon` | 8 steps |
| `candidate_exits` (candidate-route limit) | 3 nearest safe exits |
| `exit_cost` | nominal space-time arrival + `alt_exit_margin`·[not nearest] + `risk_weight`·Σ hazard |
| `alt_exit_margin` | 2.0 steps |
| `congestion_cells` | safe exits and corridor cells with ≤ 2 open neighbours |
| `risk_weight` | 20.0 (steps per unit probability) |
| `risk_hard` | 0.5 (prune threshold) |
| `move_extra_cost` | 0.25 per move |
| `node_budget_per_person` | 256 expansions |
| `node_budget_total` | 16000 expansions per decision |
| `repair_rounds` | 6 |
| `fire_forecast` | public-rule layer BFS; deterministic exact; stochastic single-front binomial lower bound |
| `search` | windowed space-time A\*, vertex and edge reservations |
| `move_order` / `search_tie_break` | N, S, E, W, stay / (f, h, generation order) |
| `follow_dependency` | leader vacates if a full plan exists, else may stay |
| `no_route_mode` | shelter toward max fire layer |
| `budget_fallback` / `no_successor_fallback` / `unrepaired_blocked_moves` | greedy descent / least-hazard action / `stay` |

**Provenance of the values.** They were chosen on synthetic data only: `sim.bench.room` open rooms (32×32 with N = 50, 100; 64×64 with N = 200; layout seeds 101–103; both fire models), the random scenarios of `tests/gen_helpers.random_scenario`, and the hand-made layouts of `tests/test_teacher_t1.py`. No DEV, train, calibration, test, blind or EMBERBench data was used. Two changes were made during that synthetic development: `move_extra_cost` 0 → 0.25 (it about halved A→B→A reversals in crowds with identical evacuation counts) and the follow-dependency rule (it removed a livelock in a one-wide corridor with a passing bay). A separate push-penalty idea had no effect and was removed.

## 4. Preregistered DEV analysis

- **Data.** `p1-02-v1` `dev` only: 48 scenarios, 16 topology families × 3 lineages. T0 (frozen, P1-03) and T1 run each scenario once with the scenario's own simulator seed. The fire trajectory is identical for both (fire never depends on actions).
- **Primary metric.** Casualties per scenario, paired difference T1 − T0.
- **Secondary.** Evacuated fraction, unresolved at termination, episode steps, completion step (episodes that both policies resolve), blocked moves per 100 attempts, collisions, non-progress rate ((stayed + blocked) / active person-steps), A→B→A reversals per 100 moves, exit utilization (evacuations / open-exit-steps), policy CPU ms per episode. Also reported: waiting rate, bottleneck-blocked moves, busiest-exit share, planning failure rate, plan statuses, decision latency p50/p95/p99.
- **Uncertainty.** Lineages of one family share topology, so scenarios are not independent. 95 % CI of the mean scenario difference: family-cluster bootstrap (resample the 16 families with replacement, 10 000 resamples, seed `t1-vs-t0-family-cluster-bootstrap@1/<metric>`, percentile interval). Also: scenario-level and family-level (family mean difference) exact two-sided sign tests.
- **Decision rule.** "T1 better (or worse) than T0 on a metric" only if the family-cluster CI excludes 0 *and* the family-level sign test has p < 0.05. Otherwise: "no evidence of a difference". No correction for the 11 metrics; p-values are descriptive. Every scenario where T1 is worse than T0 on casualties, evacuated fraction or unresolved is listed.
- **Planning failure rate** = (decisions with a budget status or budget exhaustion + policy errors) / decisions. `boxed_partial` and `no_successor_*` are reported separately: they describe the state (blocked window, no safe action), not a search failure.
- **Not done.** No parameter changes after DEV. No train/calibration/test use. No EMBERBench.

## 5. Tests (`tests/test_teacher_t1.py`, `tests/test_teacher_t1_eval.py`)

| Case | Test |
|---|---|
| Two people, one cell | lowest ID moves, the other waits (`stayed`, not `blocked_conflict`; T0 gets `blocked_conflict`) |
| Exit capacity | one evacuation per exit per step |
| Long follow chains | 10-person corridor in ascending, descending and mixed ID order: all move every step |
| Chain behind a waiting person | followers wait, no blocked attempts |
| Swaps and rotations | `repair_roots`: 4-rotation → highest ID re-planned; direct swap and queued-behind-stayer roots |
| Narrow corridor, opposing movement | forced opposite targets (nearest-exit routing never makes opposing flow on one edge): with a passing bay both evacuate through their own exits, the lower ID never oscillates; without a bay the higher ID yields; never a swap, conflict or cycle |
| Uneven exit queues, congestion rerouting | 14 people nearest to one exit: ≥ 3 rerouted, fewer steps than T0, no blocked moves |
| Exit becoming unsafe | fire reaches the nearest exit before arrival: T1 picks the other exit (7 steps vs T0 11) |
| Burning exit / imminent fire | burning exit never a target; path next to fire avoided |
| Unreachable people | shelter plan away from fire, `route_found: false`, no fake target; others unaffected |
| Fire forecast | deterministic forecast equals the simulator's burning set for 6 steps (> 1000 cell checks); stochastic frontier formula |
| N = 1, 10, 50, 100, 200 | full valid maps, repeatable decisions, no blocked outcome |
| Budgets | total budget 0 → greedy fallback; per-person budget 2 → `budget_partial`; still valid and conflict-free |
| Determinism | same public observation → same decision for any private seed or scenario ID; shuffled people/exit order → same decision |
| Leakage | 11 private keys × 4 places rejected; invalid observations rejected; AST import audit |
| Full action-set validity | 250 random closed-loop episodes (> 1000 steps): exact IDs, allowed actions, every movement `moved` or `stayed`, no `entered_fire`, `no_route` = simulator trapped set (evaluator-side oracle) |
| Harness | DEV-only gate, observation-only loop, family-cluster bootstrap, frozen config hash, committed results regenerate |

**Internal consistency, not independent evidence.** T1's `predicted_outcome` is computed with the same public resolver that the simulator uses, so "predicted = actual" checks only that T1 calls the resolver with the right inputs. It does not validate the resolver.

## 6. DEV results

Run after the preregistration commit `34c7a5a` (clean tree, frozen config above), with `python -m ember_v1.teacher.t1_eval`. Results: `data/baselines/p1-04-t1-dev-v1.json`, deterministic digest `sha256:87b409703c2f926f51e8b7f43224f826f2a3620b715ffb0f2d2800af9118d30a`. 48 scenarios, 16 families, 1734 people. T0 episodes are identical to P1-03 (same final transition hashes). No parameter changed after this run.

### 6.1 Totals

| Metric | T0 | T1 |
|---|---|---|
| Evacuated | 1210 | 1314 |
| Casualties | 507 | 345 |
| Unresolved at termination | 17 | 75 |
| Episodes with all people resolved | 46 | 44 |
| Episode steps (mean) | 34.4 | 41.2 |
| Blocked moves (per 100 attempts) | 14294 (38.0) | 0 (0.0) |
| Bottleneck-blocked moves | 1013 | 0 |
| Waiting rate | 0.092 | 0.310 |
| Non-progress rate | 0.437 | 0.310 |
| A→B→A reversals (per 100 moves) | 57 (0.24) | 1362 (3.79) |
| Exit utilization | 0.341 | 0.319 |
| Busiest-exit share | 0.80 | 0.75 |
| Predicted ≠ actual outcome | 0 | 0 |
| Policy errors / planning failure rate | 0 / 0.0 | 0 / 0.0 |
| Decisions | 1653 | 1978 |

T1 plan statuses (person-decisions): `window` 38081, `goal_exit` 9356, `boxed_partial` 4219, `no_successor_least_hazard` 378, budget statuses 0. Repair rounds 2854. Blocked moves converted to stay 817.

Casualty context (route found / action outcome):

| Context | T0 | T1 |
|---|---|---|
| no_route / stayed | 237 | 259 |
| no_route / moved | 0 | 9 |
| route / moved | 156 | 5 |
| route / stayed | 0 | 72 |
| route / blocked_occupied | 66 | 0 |
| route / blocked_conflict | 48 | 0 |

### 6.2 Paired T1 − T0 (per scenario)

CI: family-cluster bootstrap, 95 %. Sign counts: T1 better / worse / equal.

| Metric | Mean | CI95 | Scenarios b/w/= | Scenario p | Families b/w | Family p | Verdict |
|---|---|---|---|---|---|---|---|
| Casualties (primary) | −3.375 | [−5.71, −1.29] | 21/1/26 | 1.1e-5 | 12/0 | 4.9e-4 | **T1 better** |
| Evacuated fraction | +0.0465 | [+0.020, +0.080] | 20/0/28 | 1.9e-6 | 11/0 | 9.8e-4 | **T1 better** |
| Unresolved | +1.21 | [−0.38, +3.38] | 1/3/44 | 0.62 | 1/2 | 1.0 | no evidence |
| Episode steps | +6.77 | [−0.42, +16.7] | 8/23/17 | 0.011 | 4/11 | 0.12 | no evidence |
| Completion step (n = 44) | +3.02 | [−1.2, +8.5] | 8/21/15 | 0.024 | 4/10 | 0.18 | no evidence |
| Blocked per 100 attempts | −17.9 | [−24.6, −12.3] | 38/0/10 | 7.3e-12 | 16/0 | 3.1e-5 | **T1 better** |
| Collisions | −297.8 | [−591, −99] | 38/0/10 | 7.3e-12 | 16/0 | 3.1e-5 | **T1 better** |
| Non-progress rate | −0.075 | [−0.118, −0.036] | 25/11/12 | 0.029 | 12/3 | 0.035 | **T1 better** |
| Reversals per 100 moves | +1.75 | [+0.88, +2.79] | 3/29/16 | 2.6e-6 | 1/14 | 9.8e-4 | **T1 worse** |
| Exit utilization | +0.026 | [−0.002, +0.061] | 16/11/21 | 0.44 | 8/4 | 0.39 | no evidence |
| Policy CPU ms / episode | +339.5 | [+170, +549] | 0/45/3 | 5.7e-14 | 0/16 | 3.1e-5 | **T1 worse** |

p-values are descriptive (no multiplicity correction, §4). Steps and completion step have scenario-level p < 0.05 against T1, but the family-level test does not pass the decision rule.

### 6.3 Strata (evaluator-side labels; T1 never sees them)

| Stratum | Scenarios | T0 casualties | T1 casualties |
|---|---|---|---|
| label `feasible` | 14 | 99 | 0 |
| label `difficult` | 14 | 335 | 283 |
| fire deterministic | 19 | 280 | 179 |
| fire stochastic | 29 | 227 | 166 |
| enter_fire `blocked` | 29 | 277 | 164 |
| enter_fire `casualty` | 19 | 230 | 181 |

### 6.4 Scenarios where T1 is worse than T0

On casualties, evacuated fraction or unresolved (4 of 48):

| Pos | Fire | People | T0 cas / evac / unres / steps | T1 cas / evac / unres / steps | Cause |
|---|---|---|---|---|---|
| 4 | deterministic | 46 | 15 / 31 / 0 / 87 | 0 / 31 / 15 / 102 | shelter mode: no-route people survive to the step limit, not evacuated |
| 5 | deterministic | 34 | 28 / 1 / 5 / 90 | 4 / 1 / 29 / 90 | same as pos 4 |
| 30 | stochastic | 120 | 9 / 99 / 12 / 110 | 11 / 106 / 3 / 110 | 9 casualties with a route while waiting in a queue (`route/stayed`) |
| 38 | deterministic | 71 | 31 / 40 / 0 / 56 | 0 / 43 / 28 / 233 | shelter survivors keep the episode open to the step limit |

Pos 30 is the only scenario where T1 has more casualties.

### 6.5 Conclusion

- Under the preregistered rule, T1 is better than T0 on the primary metric (casualties) and on evacuated fraction, blocked moves, collisions and non-progress.
- T1 is worse than T0 on reversals and CPU cost.
- No evidence of a difference on unresolved, episode steps, completion step or exit utilization.
- Known regressions:
  - **Reversals.** The rolling-horizon replan changes targets and paths between steps (3.79 vs 0.24 per 100 moves).
  - **Waiting.** Waiting rate rises from 0.092 to 0.310. T1 converts blocked attempts into planned waits. 72 casualties occur while a routed person waits (`route/stayed`).
  - **Unresolved / long episodes.** Shelter mode turns many no-route casualties into survivors that are unresolved at the step limit (17 → 75). Some episodes run much longer (pos 38: 56 → 233 steps).
  - **CPU.** About 7× T0 per episode (§7).
- These are DEV results for one split (48 scenarios, 16 families). They are not a claim about TEST or EMBERBench.

## 7. CPU latency

### 7.1 Synthetic bench

`python -m ember_v1.teacher.bench --policy T1`, one CPU process. Output: `data/baselines/p1-04-t1-latency.jsonl`. Wall ms per `decide()` call, 250 samples each (235 for 32×32 N=1). T0 reference: `data/baselines/p1-03-t0-latency.jsonl`.

| Grid | N | T1 p50 | T1 p95 | T1 p99 | T1 max | T0 p50 | T0 p95 |
|---|---|---|---|---|---|---|---|
| 32×32 | 1 | 5.4 | 7.7 | 9.2 | 9.9 | 2.3 | 2.7 |
| 32×32 | 10 | 7.0 | 8.7 | 10.3 | 10.8 | 2.4 | 3.1 |
| 32×32 | 50 | 12.7 | 17.8 | 20.0 | 23.0 | 2.8 | 3.8 |
| 32×32 | 100 | 21.4 | 28.5 | 32.1 | 34.5 | 4.1 | 5.3 |
| 32×32 | 200 | 45.5 | 56.8 | 66.9 | 72.4 | 5.9 | 7.2 |
| 64×64 | 1 | 23.7 | 32.8 | 39.1 | 43.2 | 9.1 | 11.1 |
| 64×64 | 10 | 27.9 | 39.7 | 47.1 | 55.5 | 8.3 | 11.3 |
| 64×64 | 50 | 33.5 | 53.2 | 64.6 | 88.9 | 10.5 | 12.6 |
| 64×64 | 100 | 43.1 | 63.2 | 77.4 | 86.8 | 11.3 | 13.9 |
| 64×64 | 200 | 68.5 | 98.8 | 107.4 | 114.7 | 13.6 | 16.7 |

The total node budget (16000) bounds search work. On DEV, no decision had a budget status (§6.1). The bench does not record plan statuses. The fixed cost at N = 1 comes from the fire forecast and one BFS field per safe exit. HLD §4 estimates 10–100 ms for T1. p95 is in that range in all cells. p99 (107 ms) and max (115 ms) at 64×64, N = 200 are above it.

### 7.2 DEV

| | T0 | T1 |
|---|---|---|
| Decision wall ms p50 / p95 / p99 / max | 1.48 / 3.75 / 4.86 / 8.9 | 8.64 / 21.8 / 31.7 / 47.8 |
| Policy CPU ms per episode, mean / max | 57 / 281 | 396 / 3172 |
| Policy CPU total, 48 episodes | 2.7 s | 19.0 s |

CPU only. No GPU, no Modal, no network calls.
