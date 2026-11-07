# P1-06: T3 dynamic Ant Colony Optimization reference agent (issue #23)

**Status:** implemented and tested on CPU. Configuration, CPU pilot gate and DEV analysis are preregistered in this document (§3, §7, §8) **before** the pilot runs and before any T3 decision on DEV. Pilot results: §9; gate decision `FULL_DEV_FEASIBLE` (upper full-DEV comparison estimate 188 CPU-s). Pre-result amendments: §8.1. The DEV comparison (§8) was **run once and is complete** (§10): T3 has fewer casualties than frozen T0 (decision rule passes); there is **no evidence of a casualty difference** against frozen T1 (point estimate unfavourable to T3). T3 is **not selected**; selection stays gated by #25. T3 is a reference agent, not a claim of optimality or superiority. T0, T1 and T2 are unchanged. T4 and the #25 coordinator are not implemented. No GPU, Modal, model training, distillation, S1 generation or EMBERBench use. The frozen `p1-02-v1` split manifests are unchanged.

## 1. Architecture

| Module | Content |
|---|---|
| `teacher/t3.py` | `T3DynamicACO`: name `T3`, policy `dynamic_ant_colony_optimization`, version `1.0.0`. Implements `ReferencePolicy` (`policy-interface@1.0.0`). |
| `teacher/t3_pilot.py` | Bounded CPU feasibility pilot and full-DEV cost gate (§7): `python -m ember_v1.teacher.t3_pilot`. |
| `teacher/t3_eval.py` | Gated paired DEV harness, T3 vs frozen T0 and T1 (§8): `python -m ember_v1.teacher.t3_eval --pilot-report PATH`. Refuses to run without a matching `FULL_DEV_FEASIBLE` pilot report. |

**Independence.** `t3.py` imports only `sim.canonical` (digest), `sim.movement` (public resolver), `teacher.grid` (`PublicGrid`, `exit_distance_field`, the move order) and `teacher.interface`. It imports no T0, T1 or T2 code, no simulator internals (`sim.simulator`, `sim.scenario`, `sim.rng`, `sim.fire`) and nothing from `gen` (AST allow-list test). The fire-risk model, route costs, colony, coordination and fallback are T3's own code. T3 has its own name, version, configuration, config hash, teacher seed and evaluation; its hash differs from T0, T1 and T2 (test). The fallback is a static BFS descent on the shared `grid-utils` field, labeled in the diagnostics; T3 never calls T0, T1 or T2.

## 2. Algorithm (one decision)

Input: `Observation.to_dict()` only, validated by the strict whitelist (`contract.validate_observation`). Any private field is rejected.

1. **Graph.** Vertices: cells in the grid that are not walls and do not burn now. Edges: 4-neighbour moves. Destinations: exits that do not burn now. They are absorbing (an agent evacuates on any of them).
2. **Public fire risk** (T3's own model, `FireRisk`). Layer BFS from the burning set over flammable cells (not wall; exits only if `exits_flammable`) gives *l(c)*, the least number of spread steps that can ignite *c*. Spread happens in absolute step *t* only if *t* % `spread_interval` = 0; *n(k)* = spread steps among the next *k* steps. hazard(*c*, *k*), the risk that *c* burns at the end of relative step *k*:
   - deterministic: 1 if *n(k)* ≥ *l(c)*, else 0. This is exact for the public rule (test against the simulator).
   - stochastic, *l* = 1: 1 − (1 − *p*)^(*b*·*n*), *b* = burning neighbours now.
   - stochastic, *l* ≥ 2: P(Binomial(*n*, *p*) ≥ *l*), a single-front lower bound.
   - on a safe exit: hazard(*c*, *k* − 1), because evacuation (phase 6) comes before spread (phase 7).

   **Forecasts are not states.** Stochastic hazards are soft costs only; they never remove a cell or make a route infeasible. Only cells that burn now are removed. A deterministic hazard of 1 is certain under the public rule, so that (cell, step) pair is pruned. No seed, RNG state or future burning set is read.
3. **Route cost** of a path *c*₁…*c*ₘ to a safe exit *c*ₘ (arrival step *k* = path index, no waiting):

   C = Σₖ [1 + `w_fire`·hazard(*c*ₖ, *k*) + `w_exposure`·(burning neighbours of *c*ₖ now) + `w_congestion`·load(*c*ₖ, *k*) + `w_bottleneck`·load(*c*ₖ, *k*)·[*c*ₖ bottleneck] + `w_occupied`·[*k* = 1 and *c*₁ occupied by another person now]] + `w_queue`·exit_wait(*c*ₘ, *m*) + `w_unsafe`·[max hazard ≥ `unsafe_threshold`]

   - load(*c*, *k*): number of *other* people whose current best route is on *c* at step *k* (exits excluded). Iteration 0 uses each person's static BFS descent route as the initial flow estimate.
   - exit_wait: a FIFO queue with throughput 1 per step (one evacuation per exit per step) over the other people's arrivals at that exit. The person's own previous entry is removed approximately (−1, clamped at 0).
   - bottleneck: an open cell with ≤ 2 open 4-neighbours (narrow corridors, doors).
   - Waiting enters as a cost, not as a time shift of later hazards (limitation, §11).
4. **Colony** (one pheromone table on directed edges, shared by all ants of the decision, reset at every decision):
   - **Ant allocation.** Per iteration, each routable person (static route to a safe exit, ID order) gets `ants_per_person` ants, at most `max_ants_per_iteration` in total. If the cap binds, a rotating window (start = iteration × cap mod slots) chooses the ants.
   - **Walk.** From the person's cell, no revisits (tabu). Candidate successors: open, unvisited neighbours with a static route to an exit, not certain to burn (deterministic). Desirability *w* = τ(*u*,*v*)^α · η^β with the reduced-cost heuristic η = 1 / (step_cost(*v*, *k*) + *h(v)* − *h(u)* + 1), *h* = static exit distance (≥ 1 in the denominator).
   - **Exploration and exploitation** (ACS pseudo-random proportional rule): with probability `q0` take the argmax of *w* (ties: N, S, E, W); else a roulette draw proportional to *w*. An ant dies at a dead end or after *L*max = min(`max_path_length`, ⌊`max_path_factor`·*d*₀⌋ + `max_path_extra`) steps, *d*₀ = static distance.
   - **After the ants of an iteration:** candidate routes are re-scored with the current tables; **evaporation** τ ← max(`tau_min`, (1 − ρ)·τ) on every edge (stored edges and the unseen default); **deposits:** the iteration-best feasible ant of each person adds `deposit_q`·*d*₀/*C* to each of its edges, and the best-so-far route of each person adds `elite_weight`·`deposit_q`·*d*₀/*C*; τ ≤ `tau_max` (MAX–MIN bounds). *d*₀/*C* ∈ (0, 1] is the route quality: 1 for a shortest, safe, uncongested route. Only routes that reach a safe exit deposit. Then load and exit queues are rebuilt from the new best routes, so route costs are **recomputed** every iteration from the current fire and congestion estimate.
   - **Final exploitation ant.** After the last iteration, each routable person whose ants found no route (e.g. no ant under the cap) gets one greedy ant (`q0` = 1, no random draws) on the final pheromone.
5. **Candidates.** Per person, the best route for each first move (≤ 4) is kept, re-scored with the final tables and ranked by (cost, length, move order, path).
6. **Joint action** (deterministic):
   - **Priority:** people with a route by (best route length, ID), then people without a route.
   - **One-step reservations.** In priority order, a person takes the first candidate first move whose target cell no one has claimed, whose occupant has not decided to stay, and that is not a swap with its occupant. Else the next candidate first move, else `stay`. One claim per cell means at most one person enters each exit per step (exit throughput).
   - **Public resolver.** The full map goes through `sim.movement.resolve_moves`: lowest-ID precedence, follow chains (in any ID order), occupied cells, swaps and rotations (cycles). T3 never overrides it. A blocked mover tries its next option, for up to `repair_rounds` rounds. A move still predicted blocked becomes `stay`; the end positions are asserted unchanged.
7. **No viable route.** A person with no static route to a safe exit (`no_route_shelter`), or, with deterministic fire, whose ants found no route and whose static route is certain to burn (`no_feasible_route_shelter`), shelters: stay or move to an open, unclaimed neighbour with the least step-1 hazard, then the latest fire layer. `route_found: false`, `target_exit: null`.
8. **Budgets and fallback.** `max_ant_steps` (deterministic cap, checked before each ant with its worst-case length) and `cpu_budget_ms` (thread CPU, checked after each ant) stop the colony; status `budget_limited`, `limit_hit` names the limit. A person with a static route but no ACO route gets the first step of the static BFS descent: `reason: fallback_bfs_descent`, `fallback: true`, listed in `fallback_people`. A missing ACO route is never filled with another teacher's action.

**Diagnostics per decision** (trace only, never model input): status, colony (iterations run, ants, ants that reached an exit, success rate, ant failures by reason, ant steps, pheromone edges, mean best cost and number of changed best routes per iteration, converged people), budget, ACO-route / fallback / no-route / unsafe-route people, target exit load, coordination (repair rounds, converted to stay, first-choice and alternative moves, stays), predicted outcomes, per person (action, reason, route found, target exit, route length/cost/max hazard/unsafe, static distance, candidates, ants, option rank, predicted outcome), and **provenance** (`stream`, teacher seed, ACO seed, canonical public observation digest).

**Complexity per decision.** Grid area *A*, people *N*, routable *R*, path length *L*: fire BFS and fields O(*A*); colony ≤ `iterations` × min(`ants_per_person`·*R*, `max_ants_per_iteration`) ants × *L*max steps × 4 successors, bounded by `max_ant_steps`; re-scoring O(`iterations` × 4*R* × *L*); coordination ≤ (`repair_rounds` + 2) resolver calls.

## 3. Preregistered frozen configuration

Frozen in `T3DynamicACO.default_config()` before the pilot and before any T3 decision on DEV. Config hash (name + policy + version + configuration), pinned by a test:

**`sha256:5a8d4985b1f65d5de18a06c86d8dd522e2ca95e2accd583165b7e7d7ea1d02e0`**

Teacher seed (not in the hash; recorded in every decision): `20261123`. ACO stream: `ember_v1/t3/aco@1`.

| Key | Value | Role |
|---|---|---|
| `iterations` | 10 | ACO iterations per decision |
| `ants_per_person` | 2 | ants per routable person per iteration |
| `max_ants_per_iteration` | 64 | colony size cap per iteration |
| `alpha` | 1.0 | pheromone influence |
| `beta` | 2.0 | heuristic influence |
| `evaporation` (ρ) | 0.2 | per iteration |
| `q0` | 0.5 | exploitation probability (exploration rule) |
| `tau0` / `tau_min` / `tau_max` | 1.0 / 0.1 / 5.0 | initial value and MAX–MIN bounds |
| `deposit_q` / `elite_weight` | 1.0 / 1.0 | deposit = `deposit_q`·*d*₀/*C* (iteration best), `elite_weight`·`deposit_q`·*d*₀/*C* (best so far) |
| `max_path_factor` / `max_path_extra` / `max_path_length` | 2.0 / 12 / 400 | maximum candidate path length |
| `w_fire` | 20.0 | per unit hazard per step |
| `w_exposure` | 1.0 | per burning neighbour now |
| `w_congestion` / `w_bottleneck` | 1.0 / 1.0 | per other person on (cell, step); extra on bottleneck cells |
| `w_occupied` | 0.5 | first cell occupied now |
| `w_queue` | 1.0 | per expected exit-queue step |
| `w_unsafe` / `unsafe_threshold` | 20.0 / 0.5 | unsafe-route penalty |
| `repair_rounds` | 6 | coordination repair rounds |
| `max_ant_steps` | 100 000 | deterministic search budget per decision |
| `cpu_budget_ms` | 3000 ms | thread-CPU budget per decision |
| descriptive keys | graph, fire risk, route cost, arrival time, congestion model, pheromone, transition rule, heuristic, tabu, evaporation and deposit rules, final ant, ant allocation, path length limit, feasibility, candidates, coordination, no-route mode, fallback, RNG, memory | see `_DESCRIPTIVE` in `t3.py`; not configurable |

**Provenance of the values.** Chosen by algorithmic reasoning and synthetic smoke runs only (`sim.bench.room` 16–64 cells per side, N = 10–200, both fire models; `tests/gen_helpers.random_scenario`; the hand-made layouts in `tests/test_teacher_t3.py`). No DEV, train, calibration, test, blind or EMBERBench outcome was used, and no T2 result was used.
- `beta` = 2 with the reduced-cost heuristic gives forward / sideways / backward weights 1 / 0.25 / 0.11 on an empty floor, so ants are guided but still explore; `q0` = 0.5 splits exploitation and exploration evenly; ρ = 0.2 and [0.1, 5] bounds keep the shared field from stagnating in 10 iterations (MMAS practice).
- `w_fire` = 20 makes a step with hazard 0.5 cost as much as a 10-step detour; `w_unsafe` = 20 adds the same again for a route whose maximum hazard is ≥ 0.5. Congestion weights are 1 step per expected waiting person.
- A first draft had 8 iterations and a cap of 48 ants. In a synthetic smoke run, people without an ant under the cap fell back to BFS at N = 200 (76 fallback person-decisions in 60 steps). The final exploitation ant was added and the colony was set to 10 iterations × 64 ants for that reason only. A smoke variant with `q0` = 0.75 gave the same synthetic evacuation counts within noise; `q0` stayed at 0.5.
- The DEV workload summary (grid sizes, people counts) of `p1-02-v1` was read for the cost model only (it is public in P1-02); no T3 decision was made on DEV.
- A harness smoke run (8 s of pilot cells, scratch output, not committed) checked that the pilot code runs. It was after the hash was pinned; nothing was changed after it.

**Changing values.** Numeric keys can be overridden only to make a new identity (new hash). Any run with another value is exploratory and needs a new preregistered, independent confirmation.

## 4. Dynamic replanning and pheromone memory

T3 replans from scratch at every simulator step on the fresh public observation: the fire risk, graph, distance fields, load, exit queues and pheromone are rebuilt. This handles fire spread, routes that become unsafe, exits that burn or become inaccessible, congestion changes, and people who evacuate or become casualties (they leave the active set and the load).

**No cross-step pheromone memory (v1).** Pheromone persists across the iterations of one decision only, and is reset at every decision. T3 keeps no state between decisions, so a decision is a function of (observation, configuration, teacher seed) and repeated decisions are identical (test). A cross-step warm start is not implemented. It would need: explicit state in the trace, a defined reset per episode, invalidation of edges that burn, and replay-safe seeding; it would also make a decision depend on the history of earlier observations. It is left for a later, separately preregistered identity.

## 5. Observation consistency and reproducibility

| Claim | Evidence |
|---|---|
| T3 receives only `Observation.to_dict()` | `ReferencePolicy.decide` validates with the strict whitelist; 11 private keys × 4 places are rejected (test) |
| No private state, no other teacher | AST allow-list: only `sim.canonical`, `sim.movement`, `teacher.grid`, `teacher.interface` and the standard library; no `sim.simulator/scenario/rng/fire`, `gen`, T0, T1, T2; no `emberbench/v1`, `stream_seed` or `private_state` in the source (test) |
| ACO randomness is teacher-owned | `random.Random(seed)`, seed = first 8 bytes of sha256(`ember_v1/t3/aco@1/<teacher seed>/<canonical public digest>`); recorded in `diagnostics.provenance`; another teacher seed changes the ant samples (test) |
| Exploration is the only source of randomness | with `q0` = 1 no draw is made, and two teacher seeds give the same decision (test) |
| Ordering invariance | shuffling people, exits and walls in the observation gives the same actions and diagnostics (test) |
| Evaluator seeds and scenario IDs have no effect | four simulator seeds / scenario IDs → identical T3 decisions, while the realized fire trajectories differ (test) |
| Repeated decisions | same observation → identical deterministic decision, before and after another decision; a fresh policy gives the same decision (tests) |
| Fire risk follows the public rule | deterministic hazard = simulator trajectory (40 random scenarios, > 1000 cell checks); stochastic layer-1 hazard matches the simulator's spread rule (4000 draws), layer-2 is a lower bound, hazards are bounded and monotone; interval and exit phase order (tests) |

**Determinism.** Same observation + teacher seed + configuration → identical decision, unless `cpu_budget_ms` stops the colony (machine dependent; labeled `limit_hit: cpu_budget_ms`). `max_ant_steps` is deterministic.

## 6. Tests (`tests/test_teacher_t3.py`, `tests/test_teacher_t3_pilot.py`)

| Case | Test |
|---|---|
| Identity | name/policy/version, frozen hash pinned, hash differs from T0–T2, descriptive and numeric config validation, seed required |
| Independence and privacy | AST allow-list; private fields rejected |
| Reproducibility | repeated and stateless decisions; provenance; teacher seed changes ants; `q0` = 1 draws nothing; ordering invariance; evaluator seeds |
| Fire risk | deterministic exact; stochastic bounds, monotone, layer-1 frequency, layer-2 lower bound; spread interval; exit *k* − 1 rule |
| Pheromone | evaporation without deposits decays every edge to max(τmin, τ0(1 − ρ)^I); deposits bounded by τmax; best routes carry more pheromone than the unseen default; stored costs equal a re-score with current tables; routes within *L*max |
| Dynamic replanning | burning cell on the nearest route → other exit; burning exit is not a destination; certain forecast blockage avoided (T0 goes into it); stochastic risk is soft (p = 0.3 keeps the near exit, p = 0.9 switches); exit queue spreads people over two exits (T0 uses one); closed loop with shrinking active set and fresh-policy equality; realized fire makes T3 switch exit mid-episode in some evaluator seeds and not in others |
| Joint action | swap never attempted; rotation cycle repaired to legal moves; lowest-ID precedence on a contested exit; follow chain of 4 in reverse ID order all move; narrow corridor with 7 people evacuates with no blocked move; no route and no feasible route shelter and are labeled |
| N = 1, 10, 50, 100, 200 | complete valid maps, no predicted blocked move, repeatable, status complete |
| Closed loop | 60 random episodes (> 300 steps): exact IDs, every movement `moved` or `stayed`, predicted = actual |
| Budgets and fallback | `max_ant_steps` 0 → no ants, every routable person `fallback_bfs_descent` with the BFS first move, deterministic; `cpu_budget_ms` 0 → colony stops after one ant, fallbacks labeled |
| T3 vs T1 on synthetic | random stochastic scenarios where T3 has more casualties than T1, and others with fewer (closed loop) |
| Pilot and gate | pilot scenarios synthetic, repeatable, cover N, grids, fire models, exit counts; work model; estimate excludes budget-limited decisions and caps; gate limiting factors G1–G4; DEV harness refuses a blocked, mismatched or missing pilot report and writes no output; harness runs T0, T1, T3 and not T2; committed pilot report recomputes to the same gate |
| DEV protocol (§8.1) | wall limit min(6 h, max(900 s, 1.5 × estimate)); decision rule; committed DEV report integrity (complete, 48/16, identity, hash, seed, manifest/index, digest, paired statistics and verdicts recomputed, frozen T0/T1 transition hashes, valid episodes); CI never reruns T3 on DEV |

## 7. Preregistered CPU feasibility pilot and gate

`python -m ember_v1.teacher.t3_pilot`, one CPU process, no multiprocessing. Output `data/baselines/p1-06-t3-pilot.json`.

1. **Microbenchmark:** one T3 decision on `sim.bench.room` 32×32, N = 50, stochastic, step 3 (best of 5); wall µs per ant step.
2. **Pilot decisions, synthetic only.** `gen.layouts` builders with pilot-only seeds (`t3-pilot@1/...`; not a generator plan, family or split), one ignition, `spread_interval` 2, stochastic *p* = 0.5, `enter_fire` blocked, `max_steps` 2(*h* + *w*). 42 cells:
   - stochastic: grids 16×20 and 48×64 × {`open_room`, `corridor` (bottlenecks), `multi_room`} × N ∈ {1, 10, 50, 100, 200}, 3 exits;
   - deterministic: both grids × {`open_room`, `corridor`} × N ∈ {10, 100}, 3 exits;
   - exit configurations: 48×64 stochastic × {`open_room`, `multi_room`} × N = 50 with 1 and 6 exits.

   Small-grid cells run full closed-loop T3 episodes (route quality: evacuated, casualties, unresolved, blocked moves, prediction mismatches). Large-grid cells run the first 12 closed-loop T3 decisions. Every decision is recorded: wall and CPU ms, status, limit, iterations, ants, ant steps, ant success rate, final greedy ants, converged people, ACO-route / fallback / no-route / unsafe people, mean route stretch (ACO route length / static distance), repair rounds, conversions, predicted blocked outcomes, prediction mismatches.
3. **DEV workload profile.** Frozen T1 is replayed on the 48 `p1-02-v1` DEV scenarios (T1 DEV results are public, P1-04). Per decision: active and routable people, mean static exit distance, grid area (public observation only); T0 and T1 decision time. **T3 is not run on DEV.**
4. **Cost model.** *W*_d = `iterations`·(min(`ants_per_person`·*R*_d, `max_ants_per_iteration`) + *R*_d)·(*d̄*_d + 1) + area_d; est_d = α·*W*_d wall ms, capped at `cpu_budget_ms` + 500 ms. α from pilot decisions without a budget hit. Central: median α. Upper: 90th-percentile α × episode-length factor 1.25. Hard bound: every DEV decision at the cap × 1.25. Full DEV comparison cost = T3 estimate + measured T0 + T1 time (× the same factor).
5. **Limits.** Pilot wall-clock limit 2700 s for the whole run. One worker.
6. **Gate** (all must hold):
   - **G1 bounded:** every pilot cell ran inside the wall limit; every map complete with no predicted blocked move and 0 prediction mismatches; max decision CPU ≤ `cpu_budget_ms` + 500 ms; process peak RSS ≤ 1024 MB.
   - **G2 cost:** upper full-DEV comparison CPU (T0 + T1 + T3) ≤ 2 CPU-hours (7200 s) on one worker.
   - **G3 cap hits:** predicted fraction of DEV decisions at the cap (upper α) ≤ 5 %.
   - **G4 ACO health:** pilot fallback person-decisions ≤ 5 % of routable person-decisions, and budget-limited pilot decisions ≤ 5 %.

   Decision: `FULL_DEV_FEASIBLE` if G1–G4 pass, else `BLOCKED_BY_COMPUTE` with the limiting factors. If blocked, the DEV run does not start and the work stops for review. No configuration is changed to pass the gate without a new preregistered identity.

## 8. Preregistered DEV analysis

- **Data.** `p1-02-v1` `dev` only: 48 scenarios, 16 families × 3 lineages. Frozen T0 (P1-03), frozen T1 (P1-04) and T3 run each scenario once with the scenario's own simulator seed (identical fire trajectory for all three). T3 uses teacher seed `20261123`. T2 is not run; its P1-05 results are not used for T3.
- **Primary metric.** Casualties per scenario, paired differences T3 − T1 and T3 − T0.
- **Secondary.** Evacuated fraction, unresolved at termination, trapped at end, episode steps, completion step (both resolved), blocked moves per 100 attempts, collisions, non-progress rate, reversals per 100 moves, exit utilization, policy CPU ms per episode; decision latency p50/p95/p99/max; T3 statuses, budget hits, fallback rate, ant success rate, converged fraction (ACO convergence), unsafe routes, repair rounds and conversions.
- **Uncertainty.** Family-cluster bootstrap 95 % CI of the mean scenario difference (16 families resampled, 10 000 resamples, seeds `t3-vs-t1-family-cluster-bootstrap@1/<metric>` and `t3-vs-t0-family-cluster-bootstrap@1/<metric>`, percentile); scenario-level and family-level exact two-sided sign tests.
- **Decision rule.** "T3 better (or worse) than T1 (or T0) on a metric" only if the family-cluster CI excludes 0 *and* the family-level sign test has p < 0.05. Otherwise "no evidence of a difference". p-values are descriptive (12 metrics × 2 references, no correction). Every scenario where T3 is worse than T1 or T0 on casualties, evacuated fraction or unresolved is listed (`t3_worse_than_t1`, `t3_worse_than_t0`). No claim that ACO is superior is made without passing paired evidence.
- **Run budget.** One worker; wall limit = min(6 h, max(900 s, 1.5 × the pilot's upper comparison estimate)) (amendment A1, §8.1), checked between episodes. A run that reaches the limit stops, is marked `complete: false`, and supports no comparison claim.
- **Not done.** No parameter change after the pilot or DEV. No train, calibration, procedural locked test, private blind test or EMBERBench data.

### 8.1 Pre-result protocol amendments (before any T3 decision on DEV)

Made after review of the pilot (§9) and **before** the DEV run. No T3 DEV outcome existed or was seen. They change no policy, reward, heuristic, ACO parameter, budget, seed, DEV data, metric or statistical decision rule. T3's code, configuration (hash `sha256:5a8d…02e0`) and teacher seed `20261123` are unchanged.

| ID | Change | Why it is outcome-independent |
|---|---|---|
| A1 wall guard | Wall limit = min(6 h, max(**900 s**, 1.5 × upper estimate)) = **900 s** for this pilot (was 1.5 × 187.6 s = 281 s). `t3_eval.wall_limit`; regression `test_dev_wall_limit_amendment_a1`. | Operational only. The estimate is policy CPU time; the guard measures elapsed wall time, which also includes simulator stepping, metrics and process overhead. A 281 s guard could mark a valid run incomplete for reasons unrelated to T3. The guard decides only whether the run is complete, never a result. The 6 h maximum, the 3000 ms per-decision CPU budget and all ACO parameters are unchanged. |
| A2 committed-result test | The temporary assertion "no DEV report exists" is replaced by `test_committed_dev_report_integrity_if_present`: skips without a report; otherwise checks `complete: true`, 48 scenarios in 16 families of 3, T3 identity, config hash and seed, DEV manifest and index hashes, clean `src`, wall limit, deterministic digest, paired statistics and verdicts recomputed from the committed rows, worse-case lists and strata, frozen T0/T1 transition hashes equal to P1-03/P1-04, and valid episodes without policy failures. CI never reruns T3 on DEV. | Reads the committed file only. It checks integrity, not a direction of the result. |
| A3 reporting | The report also records the elapsed wall time, the §8 decision-rule verdict per metric (`decision_rule`, `t3_eval.verdict`; unit test `test_preregistered_decision_rule`), and trace-only route diagnostics per T3 episode: target-exit switches between consecutive decisions, route-lost person-decisions, ACO route length vs static distance, detour person-decisions. | Mechanical application of the preregistered rule and extra counters from the existing per-decision diagnostics. The recorder wraps T3 and does not change its input or output. |

## 9. Pilot results

Run once after the protocol commit `51a601e` (clean `src`, frozen configuration), with `python -m ember_v1.teacher.t3_pilot`. Report: [`data/baselines/p1-06-t3-pilot.json`](../data/baselines/p1-06-t3-pilot.json). One process, Windows 11, AMD64, Python 3.14.7. Timing values are machine dependent; Windows thread-CPU time has a 15.6 ms resolution, so latencies below are wall ms of the single policy thread. No T3 decision was made on DEV. No parameter changed after this run.

### 9.1 Resources

| | Value |
|---|---|
| Whole run wall / process CPU | 106.8 s / 105.1 s (limit 2700 s) |
| Pilot part (782 decisions) wall / T3 CPU | 76.9 s / 70.2 s |
| Peak RSS | 39.2 MB (limit 1024 MB) |
| Workers | 1 |
| Cells | 42 planned, 42 run; 782 decisions (627 stochastic, 155 deterministic) |

### 9.2 Microbenchmark (32×32 room, N = 50, stochastic, step 3)

One decision: 88.9 ms wall (best of 5), 640 ants, 8121 ant steps, 10.95 µs per ant step.

### 9.3 Pilot decisions by grid and N (all fire models and exit counts)

| Grid, N | Decisions | Mean active | Wall ms p50 / p95 / p99 / max | Mean ants | Ant success | Route stretch | Fallback fraction |
|---|---|---|---|---|---|---|---|
| 16×20, 1 | 7 | 1.0 | 3.2 / 4.4 / 4.4 / 4.4 | 20 | 0.993 | 1.000 | 0 |
| 16×20, 10 | 94 | 5.1 | 8.4 / 20.8 / 28.9 / 28.9 | 102 | 0.982 | 1.054 | 0 |
| 16×20, 50 | 82 | 21.4 | 24.9 / 66.2 / 76.7 / 76.7 | 369 | 0.994 | 1.027 | 0 |
| 16×20, 100 | 167 | 39.1 | 27.1 / 88.5 / 118.9 / 128.3 | 430 | 0.986 | 1.039 | 0.0007 |
| 16×20, 200 | 156 | 73.6 | 14.6 / 183.2 / 208.4 / 218.8 | 311 | 0.922 | 1.064 | 0.0013 |
| 48×64, 1 | 36 | 1.0 | 25.5 / 37.2 / 37.6 / 37.6 | 20 | 0.962 | 1.000 | 0 |
| 48×64, 10 | 60 | 9.3 | 78.0 / 179.6 / 190.4 / 190.4 | 186 | 0.926 | 1.051 | 0 |
| 48×64, 50 | 84 | 47.4 | 198.6 / 455.9 / 479.7 / 479.7 | 640 | 0.950 | 1.027 | 0 |
| 48×64, 100 | 60 | 95.0 | 250.5 / 425.9 / 433.1 / 433.1 | 647 | 0.862 | 1.049 | 0 |
| 48×64, 200 | 36 | 189.8 | 268.5 / 701.6 / 711.4 / 711.4 | 640 | 0.951 | 1.069 | 0 |

Route stretch = ACO route length / static shortest distance (mean over people with an ACO route). All 782 decisions: wall p50 38.1, p95 323.1, p99 573.6, max 711.4 ms; max decision CPU 703 ms (limit 3500 ms); stochastic p50 42.9 / max 711.4 ms, deterministic p50 21.2 / max 352.5 ms. Exit configurations (48×64, N = 50): 1 exit median 365.7 ms, 6 exits 171.1 ms, no fallback in either.

**ACO behaviour.** Mean ant success rate 0.955 (min 0.439 in one decision). Mean converged fraction 0.954 (people whose best route did not change in the last iteration). Fallback person-decisions: 14 of 33 096 routable (0.04 %). Final greedy ants: 548. Budget-limited decisions: 0. Unsafe-route person-decisions: 2389; no-route person-decisions: 4700 (people cut off by fire). Repair rounds: 366 in total; 0 moves converted to `stay`; 0 predicted blocked moves; 0 prediction mismatches.

### 9.4 Pilot closed-loop episodes (16×20, synthetic route quality)

19 episodes, all `all_resolved`, 1077 people: 740 evacuated, 337 casualties, 0 unresolved, 0 blocked moves, 0 prediction mismatches. The casualties come mostly from the dense cells (open_room N = 200: 152 of 200; multi_room N = 200: 105; multi_room N = 100: 26), where one ignition in a 16×20 grid with 100–200 people cuts off much of the room. All N ≤ 10 episodes evacuate everyone. These are synthetic smoke outcomes. They are not compared with any other policy and were not used to change anything.

### 9.5 Full-DEV cost estimate

DEV workload from the frozen-T1 replay: 1978 decisions (903 stochastic), mean 26.3 active, 19.1 routable. Measured on that replay: T0 3.95 s, T1 19.7 s. α (wall ms per work unit, 782 unbudgeted pilot decisions): median 0.00655, p90 0.00972, max 0.0207.

| | Central (median α) | Upper (p90 α, ×1.25 episodes) |
|---|---|---|
| T3 CPU, 48 episodes, 1 worker | 85.1 s | 158.0 s |
| Full comparison T0 + T1 + T3 | 108.8 s (0.030 h) | 187.6 s (0.052 h) |
| Max one T3 episode | 9.1 s | 16.8 s |
| DEV decisions predicted at the cap | 0 | 0 |

Hard bound (every DEV decision at `cpu_budget_ms` + 500 ms, ×1.25): 8654 s (2.4 h). The full comparison uses no paid compute; it is CPU work inside the existing P1–P2 cap.

### 9.6 Gate

| Gate | Result | Limit | Pass |
|---|---|---|---|
| G1 bounded | 42/42 cells, valid maps, 0 predicted blocked, 0 mismatches, max CPU 703 ms, peak RSS 39.2 MB | 3500 ms, 1024 MB | yes |
| G2 DEV comparison CPU (upper) | 187.6 s | 7200 s | yes |
| G3 cap hits (upper) | 0 % | 5 % | yes |
| G4 ACO health | fallback 0.04 %, budget-limited 0 % | 5 %, 5 % | yes |

**Decision: `FULL_DEV_FEASIBLE`.** The DEV comparison (§8) is permitted by the gate. Its original wall limit, min(6 h, 1.5 × 187.6 s) = 281 s, was replaced before the run by amendment A1 (§8.1): 900 s.

## 10. DEV results (run once, complete)

Run once after amendment commit `cb8d901` (clean `src`, frozen T3 code, configuration and seed), with `python -m ember_v1.teacher.t3_eval --pilot-report data/baselines/p1-06-t3-pilot.json --out data/baselines/p1-06-t3-dev-v1.json`. No retry, no subset run, no change after the run. Report: [`data/baselines/p1-06-t3-dev-v1.json`](../data/baselines/p1-06-t3-dev-v1.json); log (stdout, stderr, exit code, resources): [`p1-06-t3-dev-v1.log`](../data/baselines/p1-06-t3-dev-v1.log). Deterministic digest `sha256:2a3db9e199d82eea422d80946185a21aa94e386aa197086999b134fc56122fe9`. Claims are restricted to the independent `p1-02-v1` DEV split (48 scenarios, 16 families). They are not EMBERBench, test or blind-set claims.

### 10.1 Run

| | Value |
|---|---|
| Status | `complete: true`, 48/48 scenarios, 16 families, exit code 0, stderr empty, 1 attempt |
| Wall (process / harness loop) | 107.3 s / 104.7 s; limit 900 s; no overrun |
| CPU (job: launcher + interpreter) | 101.3 s user + 4.3 s kernel |
| Peak memory | 39.9 MB working set, 26.3 MB commit |
| Policy CPU | T0 2.6 s, T1 19.5 s, T3 69.0 s (pilot estimate for T3: central 85.1 s, upper 158.0 s) |
| Integrity | frozen T0 and T1 transition hashes equal P1-03 / P1-04 in all 48 scenarios; 0 policy failures; 0 prediction mismatches for all three policies |

Windows 11, AMD64, Python 3.14.7, one worker. Timings are machine dependent.

### 10.2 Primary: casualties per scenario (matched scenario seeds)

| | T0 | T1 | T3 |
|---|---|---|---|
| Total casualties (1734 people) | 507 | 345 | 403 |

| Paired | Mean diff | Family-cluster 95 % CI | Scenarios better / equal / worse (p) | Families better / equal / worse (p) | §8 verdict |
|---|---|---|---|---|---|
| T3 − T1 | +1.21 | [−0.13, +3.02] | 2 / 39 / 7 (0.18) | 2 / 10 / 4 (0.69) | **no evidence of a difference** |
| T3 − T0 | −2.17 | [−3.67, −0.94] | 19 / 27 / 2 (0.00022) | 12 / 4 / 0 (0.00049) | **T3 better** |

"Better" means fewer casualties for T3. The T3 − T1 point estimate favours T1, but the CI includes 0 and the family sign test is far from 0.05. It is not evidence that T3 is better or worse than T1.

### 10.3 Secondary (exploratory, unadjusted for 12 metrics × 2 references)

| Metric | T0 | T1 | T3 |
|---|---|---|---|
| Evacuated (mean evacuated fraction) | 1210 (0.722) | 1314 (0.768) | 1307 (0.759) |
| Unresolved at end = trapped at end (trapped at start: 40) | 17 | 75 | 24 |
| Episodes `all_resolved` / `max_steps` | 46 / 2 | 44 / 4 | 45 / 3 |
| Steps mean / completion step mean | 34.4 / 31.6 | 41.2 / 32.8 | 37.1 / 32.9 |
| Blocked moves = collisions (per 100 attempts) | 14 294 (38.0) | 0 | 0 |
| Waiting rate / non-progress rate | 0.092 / 0.437 | 0.310 / 0.310 | 0.328 / 0.328 |
| Reversals (per 100 moves) | 57 (0.24) | 1362 (3.79) | 1947 (6.37) |
| Exit utilization / busiest-exit share | 0.341 / 0.802 | 0.319 / 0.753 | 0.337 / 0.745 |
| Policy CPU ms per episode, mean / max | 54 / 297 | 405 / 3625 | 1437 / 7906 |
| Decision wall ms p50 / p95 / p99 / max | 1.5 / 3.8 / 5.6 / 8.9 | 8.7 / 22.5 / 30.1 / 51.7 | 17.9 / 135.8 / 200.3 / 281.6 |

Decision-rule verdicts on secondary metrics (CI excludes 0 and family p < 0.05):

| Metric | T3 − T1 | T3 − T0 |
|---|---|---|
| Evacuated fraction | no evidence (−0.009, CI [−0.034, +0.011]) | **better** (+0.038, CI [+0.015, +0.067], families 11/0) |
| Unresolved, trapped at end | no evidence (−1.06, CI [−2.67, 0.00]) | no evidence (+0.15) |
| Steps; completion step | no evidence (−4.1 and −4.0; CIs exclude 0, family p 0.092) | no evidence |
| Blocked moves, collisions | no evidence (0 = 0) | **better** (all 16 families) |
| Non-progress rate; exit utilization | no evidence | no evidence (CIs exclude 0; family p 0.45 and 0.092) |
| Reversals per 100 moves | **worse** (+1.18, families 3/13, p 0.021) | **worse** (+2.93, families 0/15) |
| Policy CPU | **worse** (+1031 ms per episode, families 0/16) | **worse** (+1382 ms, families 0/16) |

**Every scenario where T3 is worse** on casualties, evacuated fraction or unresolved (`t3_worse_than_t1`, `t3_worse_than_t0`). Values are evacuated / casualties / unresolved.

| Position | Family | Label, fire, enter | N | Reference | T3 |
|---|---|---|---|---|---|
| 5 | 1074f7… | difficult, det., blocked | 34 | T1 1 / 4 / 29 | 1 / 26 / 7 |
| 9 | 438ad8… | difficult, det., blocked | 106 | T1 51 / 55 / 0 | 49 / 57 / 0 |
| 10 | 438ad8… | feasible, det., blocked | 120 | T1 120 / 0 / 0 | 119 / 1 / 0 |
| 11 | 438ad8… | difficult, det., casualty | 79 | T1 71 / 8 / 0 | 67 / 12 / 0 |
| 37 | c3ffa5… | difficult, stoch., casualty | 73 | T1 38 / 35 / 0 | 35 / 38 / 0 |
| 38 | c3ffa5… | feasible, det., blocked | 71 | T1 43 / 0 / 28; T0 40 / 31 / 0 | 39 / 32 / 0 |
| 40 | d5996b… | feasible, stoch., blocked | 2 | T1 2 / 0 / 0 | 1 / 1 / 0 |
| 4 | 1074f7… | feasible, det., blocked | 46 | T0 31 / 15 / 0 | 31 / 0 / 15 (worse on unresolved only) |
| 5 | 1074f7… | difficult, det., blocked | 34 | T0 1 / 28 / 5 | 1 / 26 / 7 (worse on unresolved only) |
| 30 | b6ee54… | unreachable, stoch., casualty | 120 | T0 99 / 9 / 12 | 108 / 10 / 2 |

T3 is never worse than T1 on unresolved people. Strata (exploratory): T3 − T1 casualties are +61 on the 19 deterministic-fire scenarios and −3 on the 29 stochastic ones. On `feasible` scenarios T1 has 0 casualties and T3 has 34 (32 of them in position 38).

### 10.4 ACO behaviour on DEV

| | Value |
|---|---|
| Decisions | 1782, all status `complete`; CPU-budget hits 0; `max_ant_steps` hits 0 |
| Colony | 17 820 iterations, 493 005 ants, 5.995 M ant steps; ant success 0.878 (one episode had no successful ant) |
| Convergence | best route unchanged in the last iteration for 0.932 of routable person-decisions |
| Person-decisions (45 490) | ACO route 35 441; BFS fallback **15** (0.040 % of 37 477 routable); no-feasible-route shelter 2021; no-route shelter 8013 |
| Route shape | ACO route length / static distance 1.049; detour in 6423 ACO person-decisions (18.1 %); unsafe-route person-decisions 1157 |
| Route changes | target-exit switches between consecutive decisions 2723 (stochastic 2418, deterministic 305); route lost 147 |
| Coordination | repair rounds 956; alternative first moves 1968; converted to `stay` 0; predicted = actual for every movement |

### 10.5 Interpretation (DEV only)

1. **Against T0, the casualty reduction is mostly real evacuation.** Net, T3 has 104 fewer casualties, 97 more evacuations and 7 more unresolved people than T0. In position 4, 15 T0 casualties become 15 people who are still trapped (alive) at `max_steps` with T3. That part is replacement, not rescue.
2. **Against T1, the unfavourable casualty point estimate is mostly a difference in how trapped people end.** T3 has 58 more casualties, but only 7 fewer evacuations and 51 fewer unresolved people. 54 of the 58 come from positions 5 and 38. There, T1 leaves 29 and 28 people alive but trapped at `max_steps`; T3 leaves 7 and 0. Casualties + unresolved: T3 427, T1 420. Position 38 is also a real evacuation loss (39 vs 43 in a `feasible` scenario).
3. **The routes are ACO routes, not BFS.** Fallback is 0.04 % of routable person-decisions and no budget limit was hit. The result describes the frozen ACO, not its fallback.
4. **The gain over T0 is not shown to come from dynamic ACO routing.** T0 → T3 removes all 14 294 blocked moves (T0 has 114 casualties among people blocked while moving). Frozen T1 has no ACO; it removes them too and reaches the same outcome level. T3 reroutes actively (2723 exit switches, 18 % detours), but this gives no more evacuations than T1. The DEV evidence is consistent with a benefit from conflict-free, congestion-aware joint actions. It does not show a benefit from fire-risk-aware ACO routing. This run has no ablation, so it cannot formally separate the congestion and fire-risk contributions.
5. **Possible weakness (exploratory).** T3's excess casualties against T1 are concentrated in deterministic fire and in people without a feasible route who stay (casualty context `no_route/stayed`: T3 323, T1 259). This is consistent with the one-step shelter rule (§2 step 7) and with hazards that ignore waiting time (§11). It is a hypothesis for a new identity, not a tuned change.
6. **Cost.** T3 uses 3.5 × the policy CPU of T1. It has more reversals than T1 and T0 (the decision rule passes against both).

**Verdict (preregistered rule, casualties):** T3 better than T0; no evidence of a difference against T1. T3 is not superior to T1 on DEV. T3 is **not selected** as the training teacher. It stays an independent reference and may enter the #25 multi-teacher evaluation as a candidate. No bulk labels are generated. Any changed T3 (for example multi-step shelter or time-shifted hazards) needs a new identity and a new preregistered, independent confirmation.

## 11. Limitations

- Arrival time is the path index: waiting in a queue is a cost, not a time shift, so hazards after a wait are underestimated.
- The stochastic layer ≥ 2 hazard is a single-front lower bound (optimistic for cells reachable by several fronts).
- The exit-queue model removes the person's own previous entry approximately (−1).
- One shared pheromone field: people heading to different exits share edges. Congestion costs, not pheromone, separate flows.
- Coordination is one step deep; it does not plan multi-step space-time reservations.
- No cross-step pheromone memory (§4).
- The CPU budget makes rare decisions machine dependent; they are labeled.
- The DEV wall limit (900 s after amendment A1) is checked between episodes, so one long episode can overrun it; the elapsed time is recorded. If the run reaches the limit, it is marked `complete: false` and supports no claim.
- Pilot outcomes are synthetic smoke results. They show bounded cost and legal joint actions, not route quality relative to T0 or T1.
- DEV results (§10) cover 48 scenarios in 16 families. Against T1 only 6 families are not tied on casualties, so the family sign test has low power. There is no ablation of the fire-risk and congestion terms. Secondary p-values are unadjusted. No claim is made outside `p1-02-v1` DEV.
