# P1-05: T2 stochastic fire-forecasting rollout agent (issue #9)

**Status:** implemented and tested on CPU. Protocol, configuration, pilot gate and DEV analysis are preregistered in this document (§3, §6, §7) **before** any pilot run and before any T2 decision on DEV (protocol commit `ca97780`). Pilot results: §8; gate decision `FULL_DEV_FEASIBLE`. DEV comparison: §9, run once, complete; **no evidence of a casualty difference → T2 NOT SELECTED as the training teacher** (issue #9 gate). T2 stays an independent reference policy (issue #25). T2 is a reference agent, not a claim of optimality. T3, T4 and the #25 coordinator are not implemented. T0 and T1 are unchanged. No GPU, Modal, model training, S1 generation or EMBERBench use.

## 1. Architecture

| Module | Content |
|---|---|
| `teacher/t2.py` | `T2StochasticRollout`: name `T2`, policy `stochastic_fire_forecast_rollout`, version `1.0.0`. Implements `ReferencePolicy` (`policy-interface@1.0.0`). |
| `teacher/t2_pilot.py` | Bounded CPU feasibility pilot and full-DEV cost gate (§6): `python -m ember_v1.teacher.t2_pilot`. |
| `teacher/t2_eval.py` | Gated paired DEV harness, T2 vs frozen T1 (§7): `python -m ember_v1.teacher.t2_eval --pilot-report PATH`. Refuses to run without a matching `FULL_DEV_FEASIBLE` pilot report. |

`t2.py` imports `contract` (public validation), `sim.canonical` (digest), `sim.movement` (public resolver), `teacher.interface` and `teacher.t1` (the frozen continuation). It imports no simulator internals (`sim.simulator`, `sim.scenario`, `sim.rng`, `sim.fire`), nothing from `gen`, and no T0, evaluation or pilot code (AST test).

**Identity and separation.** T2 has its own name, version, configuration, config hash and teacher seed. T1 is used only as (a) the proposal for the joint map on the real observation and (b) the continuation policy inside T2's rollouts. T2 checks at construction that the continuation has the frozen T1 config hash `sha256:5047e942db3326706b6184c9f29e0b992078920f03c8a53adc8cc24258345361` and version `1.0.0`, and records the T1 source digest (sha256 of `t1.py` with LF line endings) `sha256:0365af65c598ce56ef64a6bd4032cb8ef4fefa5f912b004cdc023993e80d4dfa` in every decision. A test pins both values. T2's outputs, diagnostics, costs and evaluation are separate from T1's.

## 2. Algorithm (one decision)

Input: `Observation.to_dict()` only, validated by the strict whitelist (`contract.validate_observation`). Any private field is rejected.

1. **T1 proposal.** Frozen T1 plans the joint map *A* on the real observation. If T1 raises or returns an incomplete map, T2 fails closed (the error propagates; `validate_actions` rejects missing IDs).
2. **Planning state.** Built only from the observation: grid, walls, exits, burning set, active people, step, public rules (`fire_model`, `spread_interval`, `spread_probability`, `exits_flammable`, `enter_fire`, `max_steps`).
3. **Fire futures.** *K* futures of *H* steps (fewer if `max_steps` ends sooner). Public rule: spread in step *t* only if *t* % `spread_interval` = 0; frontier = flammable cells (not wall; exits only if `exits_flammable`) with a burning 4-neighbour, row-major; deterministic: every frontier cell ignites; stochastic: one Bernoulli(*p*) draw per (cell, burning neighbour) edge, every edge draws, the cell ignites if any draw hits. Draws come from the teacher's own stream `random.Random(seed_k)`, `seed_k` = first 8 bytes of sha256(`ember_v1/t2/fire-futures@1/<teacher seed>/<public fire digest>/<k>`). The public fire digest covers only `map`, `rules`, `step` and `burning`. Deterministic fire: one exact future (*K* = 1).
4. **Rollout model.** Public phase order: movement with the public resolver (`sim.movement.resolve_moves`, lowest ID wins, follow chains, cycles, one person per cell, so one evacuation per exit per step), evacuation on exits not burning at the start of the step, spread, casualties (`entered_fire` / `fire_spread`), termination (all resolved, or `max_steps`). Step 1 uses the evaluated joint map; steps 2..*H* use frozen T1 on an observation rebuilt from the rollout state. A test feeds the simulator's realized fire trajectory into the model and gets the same return as a real simulator run of T1 (80 random scenarios).
5. **Return** (sum over all people; team value, so effects on other people count):

   | Outcome | Value |
   |---|---|
   | evacuated at relative step τ | `evac_reward` − `time_cost`·τ |
   | casualty | −`casualty_penalty` |
   | unresolved when `max_steps` ends the rollout | `unresolved_at_limit_value` |
   | unresolved at the horizon, safe exit reachable in that future at BFS distance *d*, and *t*₀ + τ_end + *d* ≤ `max_steps` | `evac_reward` − `time_cost`·(τ_end + *d*) − `unresolved_route_penalty` |
   | unresolved at the horizon, reachable but too late for `max_steps` | `unresolved_at_limit_value` |
   | unresolved at the horizon, no safe exit reachable in that future (trapped) | −`trapped_terminal_penalty` |

   *d* is the BFS distance over cells that are not walls and not burning in that future at τ_end, from exits not burning at τ_end.
6. **Person selection.** Active people sorted by (BFS distance from the current burning set over non-wall cells, person ID); the first `max_scored_people` are scored. The others keep T1's action and get no values (`reason: not_selected`).
7. **Five-action scoring.** For each scored person *i* and each action in (`stay`, `north`, `south`, `east`, `west`): joint map = *A* with *i* → action (other people held to T1's proposal at step 1, T1 continuation afterwards). If the step-1 end positions equal those of *A* or of an action already evaluated for *i*, the rollouts are identical (T1 is deterministic and stateless; the futures are common) and the values are reused (`shared_with`). Otherwise *K* new rollouts on the **same** *K* futures (common random numbers). Futures that are identical to an earlier sample reuse its rollout (exact). A per-decision cache of T1 continuation calls keyed by (relative step, active positions, burning set) is exact: T1 reads only the active people, the burning set, the rules and the step. Recorded per action: mean, sample variance, standard error, sample count, step-1 outcome; per person: ties, best action, paired gain vs T1's action and its standard error.
8. **Tie-breaking.** Ties: means within `tie_tolerance` of the best. Choose T1's action if it is tied, else the first of `stay`, `north`, `south`, `east`, `west`.
9. **Joint-action construction.** A best action ≠ T1's action is a deviation candidate only if gain > `min_gain` and gain > `deviation_z`·SE(gain) (paired over the common futures). Candidates are added greedily (gain descending, then person ID) to *J* = *A*. A candidate is accepted only if the public resolver gives exactly the step-1 end positions that were evaluated (*A*'s end positions with the candidate's own changes) and the candidate changes no person that an accepted candidate already changed; otherwise `rejected_conflict`. If more than one deviation is accepted, *J* is evaluated on the same futures; *J* is kept only if mean V(*J*) − V(*A*) > `min_gain` and ≥ the best single gain, else T2 uses *A* plus only the best single deviation (its rollouts are already the evaluation of that exact map). Moves still predicted blocked become `stay` (end positions asserted unchanged). Final map: `validate_actions` (exact active IDs, allowed actions).
10. **Budgets and fallbacks.** `max_rollouts` (deterministic count of executed rollouts) and `cpu_budget_ms` (thread CPU, checked after every rollout step). A person whose rollouts do not all finish gets no values (`budget_limited`); scoring stops. If the *K* rollouts of *A* do not finish: status `budget_fallback_t1`, actions = *A*, no values. If a continuation call raises inside a rollout: status `continuation_error_fallback_t1`, actions = *A* (valid), all values dropped, error type recorded. Missing values are never filled.

**What the values mean.** Per-action values are conditional single-person Monte Carlo estimates with the other people held to T1. They are not joint-optimal values and not causal effects. The continuation is T1, so T2 is a one-step policy improvement over T1 under sampled futures. It cannot exploit a future in which T1 itself would act differently after step 1 (a test shows such a case).

**Provenance (trace only, never model input).** `diagnostics.provenance`: teacher seed, public fire digest, the *K* sample seeds. `diagnostics.continuation`: T1 name, version, config hash, source digest.

**Determinism.** Same observation + same teacher seed + same configuration → identical decision and values, unless the CPU budget stops the decision (machine dependent; labeled `limit_hit: cpu_budget_ms`). The rollout cap is deterministic.

**Complexity per decision.** Executed rollouts ≤ *K*·(1 + 4·*M*) + *K* (*M* = scored people), each ≤ *H* steps, each step one resolver call and (τ ≥ 2) one T1 call unless cached. Cost ≈ people-scored × 5 actions × *K* × *H* × T1 cost, reduced by the exact sharing above.

## 3. Preregistered frozen configuration

Frozen in `T2StochasticRollout.default_config()` before the pilot and before any T2 decision on DEV. Config hash (name + policy + version + configuration):

**`sha256:302139e18ba05fdcdc5561b7390b95f725600ec8791e336d065f77cea0c507e5`**

Teacher seed (not in the hash; recorded in every decision): `20261107`. Continuation: T1 `1.0.0`, config hash `sha256:5047e942db3326706b6184c9f29e0b992078920f03c8a53adc8cc24258345361`, source `sha256:0365af65c598ce56ef64a6bd4032cb8ef4fefa5f912b004cdc023993e80d4dfa`.

| Key | Value |
|---|---|
| `horizon` (*H*) | 8 steps (= T1 reservation horizon) |
| `samples_stochastic` (*K*) | 8 futures (deterministic fire: 1 exact future) |
| `max_scored_people` (*M*) | 4 per decision |
| `max_rollouts` | 144 = *K*·(1 + 4·*M*) + *K* (never binds before *M* people are scored) |
| `cpu_budget_ms` | 20000 ms thread CPU per decision |
| `evac_reward` | 1.0 |
| `time_cost` | 0.01 per step |
| `casualty_penalty` | 5.0 |
| `unresolved_route_penalty` | 0.25 |
| `trapped_terminal_penalty` | 2.5 |
| `unresolved_at_limit_value` | 0.0 |
| `deviation_z` | 1.0 |
| `min_gain` | 0.005 |
| `tie_tolerance` | 1e-9 |
| descriptive keys | sampler, value, terminal, selection, dedup, deviation rule, tie-break, joint construction, fallbacks (see `_DESCRIPTIVE` in `t2.py`); not configurable |

Memory bound per decision: *K* futures of ≤ *H* + 1 burning sets, ≤ `T1_CACHE_MAX` = 4096 cached continuation maps (cleared when full), values O(*M*·5·*K*). Rollouts run one after another in one thread.

**Provenance of the values.** Chosen before any pilot or DEV use, from synthetic smoke runs only (`sim.bench.room` 32×32, N = 10 and 50, and `tests/gen_helpers.random_scenario`). A first draft (*H* = 10, *M* = 6, 208 rollouts, 30 s) cost 11–19 s per stochastic decision at N = 10–50 in a smoke run; *H*, *M*, `max_rollouts` and `cpu_budget_ms` were reduced to the values above for that cost reason only. The value weights were set by reasoning (casualty 5× an evacuation; trapped at the horizon half a casualty, because T1's shelter mode often keeps such people alive; a small time cost), not by outcome comparison. No DEV, train, calibration, test, blind or EMBERBench data was used.

**Changing values.** Numeric keys can be overridden only to make a new identity (new hash). Any run with another value is exploratory and needs a new preregistered, independent confirmation.

## 4. Observation consistency

Every sampled future is a function of (public observation, public rules, teacher seed, configuration) only:

| Claim | Evidence |
|---|---|
| T2 receives only `Observation.to_dict()` | `ReferencePolicy.decide` validates with the strict whitelist; 11 private keys × 4 places are rejected (test) |
| No private state is read | AST import audit: no `sim.simulator`, `sim.scenario`, `sim.rng`, `sim.fire`, `gen`; the source contains no `emberbench/v1` seed label, no `stream_seed`, no `private_state` (test) |
| The teacher stream is not the evaluator stream | seeds come from `ember_v1/t2/fire-futures@1/...` (sha256), Mersenne Twister draws; the simulator uses SplitMix64 from `emberbench/v1/<seed>/fire` |
| Futures depend only on the public fire state and the teacher seed | moving people does not change the futures; a different teacher seed changes the seeds (test) |
| Evaluator seed and scenario ID have no effect | four simulator seeds and scenario IDs → identical T2 decisions, while the realized evaluator futures differ (test) |
| Futures follow the public rule | deterministic: equal to the simulator trajectory (40 random scenarios, > 100 step checks); stochastic: fed the same uniform stream as the rule implementation, the sampler ignites exactly the same cells (60 scenarios × 6 steps); frontier frequency 1 − (1 − *p*)^*k* (4000 draws); every sampled ignition has a burning neighbour and is flammable (tests) |
| Rollouts never get the real future | futures are generated inside the decision from the observation; the rollout model test injects the realized trajectory on the test side only |

## 5. Tests (`tests/test_teacher_t2.py`, `tests/test_teacher_t2_pilot.py`)

| Case | Test |
|---|---|
| Identity | name/policy/version, frozen config hash and T1 source digest pinned, config hash differs from T1, numeric/descriptive config validation |
| Determinism | same observation + seed → identical deterministic decision dict; other teacher seed → other sample seeds |
| Private seeds | different evaluator seeds / scenario IDs → same decision |
| Common random numbers | the futures are sampled once per decision; every evaluated joint map uses the same future objects |
| Fire rule | deterministic exact, stochastic edge-exact, frontier frequency, rule closure |
| Rollout model | equals a real simulator run of T1 on the realized future (80 random scenarios) |
| Different futures → different actions; forecast exit blockage | corridor, T1 goes east to E1 (fire 3 cells from E1, *p* = 0.3): with *K* = 1, every future in which E1 burns before Pa can arrive makes T2 go west; every future without fire at E1 keeps east; *K* = 8 keeps east (blockage rare) |
| Shared rollouts | wall moves have the same step-1 state as `stay` and share its values |
| Values | scored people: exactly five finite means, variance ≥ 0, SE = √(var/K), *K* samples each; unscored: `values: null`, no variance |
| Joint construction | two individually best moves into one cell: the higher gain is accepted, the other `rejected_conflict` |
| N = 1, 10, 50, 100, 200 | complete valid maps, no predicted blocked move, repeatable |
| Closed loop | 60 random episodes (> 300 steps): exact IDs, every movement `moved` or `stayed`, predicted = actual, deviations occur |
| Budgets | `cpu_budget_ms` 0 → `budget_fallback_t1` = T1 map, no values; rollout cap → `budget_limited`, values missing; 200 ms cap on N = 100 stops the decision |
| Fail closed | T1 proposal error → raises; incomplete T1 proposal → `ValueError`; continuation error → `continuation_error_fallback_t1` with the valid T1 map and no values |
| T2 worse than T1 | a random stochastic scenario where T2 has more casualties than T1 (closed loop); also one where T2 has fewer |
| Pilot and gate | pilot scenarios synthetic and repeatable; nominal-step model; estimate excludes budget-limited decisions and caps at the budget; gate decision and limiting factors; DEV harness refuses a blocked, mismatched or missing pilot report and writes no output |

## 6. Preregistered CPU feasibility pilot and gate

`python -m ember_v1.teacher.t2_pilot`, one CPU process, no multiprocessing. Output `data/baselines/p1-05-t2-pilot.json`.

1. **Microbenchmark** (`sim.bench.room` 32×32, N = 50, stochastic, step 3): sampler, resolver, one T1 continuation call, one rollout.
2. **Pilot decisions, synthetic only.** `gen.layouts` builders `open_room`, `corridor`, `multi_room`, `irregular` on 32×40 grids with pilot-only seeds (`t2-pilot@1/...`; not a generator plan, family or split), 3 border exits, one ignition, `spread_interval` 2, stochastic *p* = 0.5, `enter_fire` blocked. Cells: 4 topologies × N ∈ {1, 10, 50, 100, 200} stochastic, plus deterministic {open_room, multi_room} × {10, 100}. Observations at steps 0 and 4 of a frozen-T1 prefix: ≤ 48 decisions. Measured per decision: T2 CPU and wall ms, T1 decision ms, rollouts, rollout steps, T1 calls, cache hits, CPU ms per evaluated action, rollouts per CPU second, status.
3. **DEV workload profile.** Frozen T1 is replayed on the 48 `p1-02-v1` DEV scenarios (as in P1-04; T1 DEV results are already public) to record per decision: active count, step, `max_steps`, fire model, T1 decision ms. **T2 is not run on DEV.**
4. **Cost model.** est_d = c1_d·(1 + α·nominal_d), nominal_d = K_eff·(1 + 4·min(N_d, *M*))·min(*H*, `max_steps` − t_d). α is measured on pilot decisions without a budget hit, per fire model (pooled if fewer than 3). Central: median α. Upper: 90th-percentile α × episode-length factor 1.25 (T2 episodes may be longer than T1's). est_d is capped at `cpu_budget_ms` + c1_d.
5. **Limits.** Pilot wall-clock limit 2700 s for the whole run (no new decision starts after it). One worker.
6. **Gate** (all must hold):
   - **G1 bounded:** all planned pilot decisions ran inside the wall limit; every map complete with no predicted blocked move; max decision CPU ≤ `cpu_budget_ms` + 2000 ms; process peak RSS ≤ 1024 MB.
   - **G2 cost:** upper full-DEV T2 CPU ≤ 4 CPU-hours (14400 s) on one worker.
   - **G3 budget hits:** predicted fraction of DEV decisions that hit `cpu_budget_ms` (upper α) ≤ 5 %.
   
   Decision: `FULL_DEV_FEASIBLE` if G1–G3 pass, else `BLOCKED_BY_COMPUTE` with the limiting factors. If blocked, the full DEV run does not start and the work stops for review. No configuration is changed to pass the gate without a new preregistered identity.

## 7. Preregistered DEV analysis (runs only after a passing gate)

- **Data.** `p1-02-v1` `dev` only: 48 scenarios, 16 families × 3 lineages. Frozen T1 and T2 run each scenario once with the scenario's own simulator seed (identical fire trajectory for both). T2 uses teacher seed `20261107`.
- **Primary metric.** Casualties per scenario, paired difference T2 − T1.
- **Secondary.** Evacuated fraction, unresolved at termination, trapped at end, episode steps, completion step (both resolved), blocked moves per 100 attempts, collisions, non-progress rate, reversals per 100 moves, exit utilization, policy CPU ms per episode; decision latency p50/p95/p99/max; T2 statuses, budget hits, deviations, scored person-decisions, rollouts.
- **Uncertainty.** Family-cluster bootstrap 95 % CI of the mean scenario difference (16 families resampled, 10 000 resamples, seed `t2-vs-t1-family-cluster-bootstrap@1/<metric>`, percentile); scenario-level and family-level exact two-sided sign tests.
- **Decision rule.** "T2 better (or worse) than T1 on a metric" only if the family-cluster CI excludes 0 *and* the family-level sign test has p < 0.05. Otherwise "no evidence of a difference". p-values are descriptive (12 metrics, no correction). Every scenario where T2 is worse than T1 on casualties, evacuated fraction or unresolved is listed. If T2 shows no measurable benefit on casualties, T2 is not selected as the training teacher (issue #9, HLD §6 P1 gate).
- **Run budget.** One worker; wall limit = min(6 h, 1.5 × the pilot's upper estimate). A run that reaches the limit stops, is marked `complete: false`, and supports no comparison claim.
- **Not done.** No parameter change after DEV. No train, calibration, procedural locked test, private blind test or EMBERBench data.

## 8. Pilot results

Run after the protocol commit `ca97780` (clean tree, frozen configuration), with `python -m ember_v1.teacher.t2_pilot`. Report: `data/baselines/p1-05-t2-pilot.json`. One process, Windows 11, AMD64, Python 3.14.7. Timing values are machine dependent. No T2 decision was made on DEV. No parameter changed after this run.

### 8.1 Resources

| | Value |
|---|---|
| Pilot wall / process CPU | 442 s / 434 s (limit 2700 s) |
| Peak RSS | 45.6 MB (limit 1024 MB) |
| Workers | 1 |
| Pilot decisions | 48 planned, 48 run (40 stochastic, 8 deterministic) |

### 8.2 Microbenchmark (32×32 room, N = 50, stochastic)

| Part | ms |
|---|---|
| Sample 8 futures × 8 steps | 0.63 |
| Resolver call | 0.08 |
| One T1 continuation call | 12.8 |
| One rollout (8 steps, empty cache) | 91.7 |

The T1 continuation call is the dominant cost.

### 8.3 Pilot decisions (stochastic, 4 topologies × 2 observations per N)

| N (mean active) | T1 decision ms (median) | T2 CPU ms median / max | CPU ms per evaluated action (median) | Rollouts per CPU s (median) | Budget-limited |
|---|---|---|---|---|---|
| 1 (1.0) | 7.2 | 1156 / 2047 | 231 | 29.2 | 0/8 |
| 10 (9.9) | 9.6 | 5305 / 7547 | 265 | 21.2 | 0/8 |
| 50 (48.9) | 14.6 | 10000 / 12797 | 500 | 12.4 | 0/8 |
| 100 (96.9) | 22.6 | 15055 / 19922 | 753 | 8.7 | 0/8 |
| 200 (182.1) | 38.4 | 20039 / 20094 | 1337 | 4.2 | 6/8 |

Deterministic cells (N = 10 and 100, one exact future): 734–1953 ms per decision. Every pilot map was complete with no predicted blocked move. The 6 budget-limited N = 200 decisions are labeled `budget_limited` (`cpu_budget_ms`); their unscored people have no values. Max decision CPU 20094 ms (limit 22000 ms).

### 8.4 Full-DEV cost estimate

DEV workload from the frozen-T1 replay: 1978 decisions (903 stochastic), mean 26.3 active people, T1 total 18.7 s (P1-04: 19.0 s). α (relative cost of one nominal rollout step): stochastic median 0.597 / p90 0.785 (n = 34), deterministic 0.631 / 0.861 (n = 8).

| | Central (median α) | Upper (p90 α, ×1.25 episodes) |
|---|---|---|
| T2 CPU, 48 episodes, 1 worker | 5788 s (1.61 h) | 9387 s (2.61 h) |
| Max one episode | 1381 s | 2100 s |
| Decisions predicted to hit `cpu_budget_ms` | 5 (0.25 %) | 42 (2.1 %) |

### 8.5 Gate

| Gate | Result | Limit | Pass |
|---|---|---|---|
| G1 bounded | complete, valid maps, max CPU 20094 ms, peak RSS 45.6 MB | 22000 ms, 1024 MB | yes |
| G2 DEV CPU (upper) | 9387 s | 14400 s | yes |
| G3 budget hits (upper) | 2.1 % | 5 % | yes |

**Decision: `FULL_DEV_FEASIBLE`.** The DEV comparison (§7) is permitted. Its preregistered wall limit is min(6 h, 1.5 × 9387 s) = 14080 s. Results: §9.

## 9. DEV results (T2 vs frozen T1, run once)

### 9.1 Run

| | Value |
|---|---|
| Command | `python -m ember_v1.teacher.t2_eval --pilot-report data/baselines/p1-05-t2-pilot.json --out data/baselines/p1-05-t2-dev-v1.json` |
| Code | `c62fb89` (clean `src`), T2 hash `sha256:302139e1…c507e5`, teacher seed `20261107`, T1 `sha256:5047e942…345361` |
| Executions | 1 (no retry), 1 worker, CPU only, Windows 11, AMD64, Python 3.14.7 |
| Start / end (UTC) | 2026-11-07 08:07:31 / 09:05:51 |
| Exit status / `complete` | 0 / `true`, 48 scenarios, 16 families |
| Wall time | 3499 s (limit 14080 s; no overrun) |
| T2 policy CPU | 3384 s (pilot central estimate 5788 s, upper 9387 s); T1 policy CPU 20.6 s |
| Peak working set | 42 MB (sampled once per minute from the interpreter process) |
| Report / log | [`data/baselines/p1-05-t2-dev-v1.json`](../data/baselines/p1-05-t2-dev-v1.json), [`p1-05-t2-dev-v1.log`](../data/baselines/p1-05-t2-dev-v1.log) |
| Deterministic digest | `sha256:9309c325a79c489682fc72ea70c404493a27ff80050abc7cfe767e9530a36bc6` |

The T1 episodes in this run are identical to P1-04 (48/48 final transition hashes). The `peak_working_set_mb` value in the log footer (4.3 MB) is the venv launcher process, not the interpreter; use the sampled value above.

### 9.2 Primary metric: casualties per scenario, T2 − T1

| | Value |
|---|---|
| Total casualties | T1 345, T2 333 (1734 people) |
| Mean paired difference | −0.250 per scenario |
| Family-cluster bootstrap 95 % CI | [−0.688, +0.063] (includes 0) |
| Scenarios T2 better / equal / worse | 3 / 43 / 2; sign test p = 1.0 |
| Families T2 better / equal / worse | 3 / 11 / 2; family sign test p = 1.0 |

**Decision rule (§7): no evidence of a difference.** The CI includes 0 and the family sign test has p = 1.0. 9 of the 12 fewer casualties come from one scenario (position 30), in which T2 has 8 more unresolved people at `max_steps`.

### 9.3 Secondary metrics (exploratory, unadjusted)

| Metric | T1 | T2 | Mean T2 − T1 | Family-cluster 95 % CI | Scen. b/e/w | Fam. b/w, p |
|---|---|---|---|---|---|---|
| Evacuated fraction (mean) | 0.7684 | 0.7703 | +0.0019 | [−0.0043, +0.0100] | 3/43/2 | 3/2, 1.0 |
| Unresolved at end (total) | 75 | 83 | +0.167 | [0.000, +0.500] | 0/47/1 | 0/1, 1.0 |
| Trapped at end (total) | 75 | 83 | +0.167 | [0.000, +0.500] | 0/47/1 | 0/1, 1.0 |
| Episode steps (mean) | 41.21 | 41.21 | 0.000 | [−0.417, +0.458] | 7/38/3 | 5/3, 0.73 |
| Completion step (44 resolved pairs) | 32.80 | 32.80 | 0.000 | [−0.455, +0.489] | 7/34/3 | 5/3, 0.73 |
| Blocked per 100 attempts | 0 | 0 | 0 | [0, 0] | 0/48/0 | — |
| Collisions | 0 | 0 | 0 | [0, 0] | 0/48/0 | — |
| Non-progress (waiting) rate | 0.3102 | 0.3144 | −0.0034 | [−0.0079, +0.0016] | 17/21/10 | 10/4, 0.18 |
| Reversals per 100 moves | 3.79 | 3.56 | −0.059 | [−0.253, +0.128] | 15/24/9 | 6/6, 1.0 |
| Exit utilization | 0.3191 | 0.3206 | +0.0024 | [−0.0059, +0.0108] | 10/34/4 | 8/3, 0.23 |
| Policy CPU ms per episode (mean) | 430 | 70 509 | +70 080 | [+29 890, +123 421] | 0/0/48 | 0/16, 3.1e−5 |

Pooled totals in the T1/T2 columns; "b/e/w" = T2 better / equal / worse. The only metric that passes the decision rule is CPU cost (T2 worse). Decision wall latency (one thread): T1 p50 9.2, p95 24.0, p99 32.8, max 54.5 ms; T2 p50 601, p95 7768, p99 13 551, max 18 541 ms. No episode failed; prediction mismatches 0 for both.

### 9.4 T2 behaviour (realized decisions)

| | Value |
|---|---|
| Decisions | 1978, all status `scored`; 0 `budget_fallback_t1`, 0 `continuation_error_fallback_t1` |
| Budget hits | 0 `cpu_budget_ms`, 0 `max_rollouts` (pilot upper prediction: 42) |
| Scored person-decisions | 7403 |
| Person-decisions where T2's action ≠ T1's proposal | 339 = 0.65 % of 52 306 person-steps, 4.6 % of scored (215 stochastic, 124 deterministic) |
| Scenarios with any deviation | 27 of 48 (14 families); the realized episode differs from T1 in exactly these 27 |
| Joint construction | 1932 decisions with ≤ 1 accepted deviation; 46 with ≥ 2: 21 joint map kept, 25 best single |
| Rollouts | 82 705 total, mean 41.8 per decision; per-episode mean per decision: min 3.1, median 61.5, max 101.2 |
| T2 episode CPU / wall | 3388 s / 3471 s total; max one episode 810 s / 827 s (position 30) |

The harness records rollout counts per episode, not per decision, and decision CPU as an episode sum; per-decision CPU percentiles are not in the report (the wall percentiles above are a close proxy for one thread). `rejected_conflict` counts are not aggregated. The report holds realized episode outcomes only; T2's Monte Carlo values are internal estimates and are not stored or compared here.

### 9.5 Scenarios where T2 differs in outcome

| Pos. | Family | Label, fire, N | Casualties T1 → T2 | Unresolved | Evacuated fraction | Steps |
|---|---|---|---|---|---|---|
| 26 | fam-92b71e2df9 | difficult, stochastic, 26 | 17 → 13 | 0 → 0 | 0.346 → 0.500 | 45 → 48 |
| 37 | fam-c3ffa5d21c | difficult, stochastic, 73 | 35 → 34 | 0 → 0 | 0.521 → 0.534 | 68 → 68 |
| 30 | fam-b6ee546f8e | unreachable, stochastic, 120 | 11 → 2 | **3 → 11** | 0.883 → 0.892 | 110 → 110 (`max_steps`) |
| 9 | fam-438ad8fbaa | difficult, deterministic, 106 | **55 → 56** | 0 → 0 | 0.481 → 0.472 | 39 → 39 |
| 47 | fam-fc0f1986ff | feasible, stochastic, 13 | **0 → 1** | 0 → 0 | 1.000 → 0.923 | 9 → 9 |

T2 is worse on casualties, evacuated fraction or unresolved in positions 9, 30 and 47. In position 47 the casualty had a route and stayed (`route/stayed`, `fire_spread`). In position 30, T2 keeps 9 more people alive, but 8 of them stay trapped at the step limit; this is a shift from casualties to unresolved, not an evacuation gain (+1 evacuated). T2 also changes the episode length in 10 scenarios: shorter in 7 feasible scenarios with no casualty change (e.g. 28 → 25, 56 → 53), longer in 3 (positions 10: 38 → 45 and 16: 53 → 56, both feasible with no casualty change; 26: 45 → 48).

### 9.6 Gate verdict

**T2 NOT SELECTED** as the training teacher (issue #9, HLD §6 P1 gate): no measurable casualty benefit over frozen T1 on independent DEV under the preregistered rule. T2 costs ≈ 164× T1's policy CPU. T2 stays an independent reference policy for issue #25. The configuration, protocol and code are unchanged after this run. No parameter (*K*, *H*, rewards, thresholds, selection) is tuned on these results; any changed T2 is a new identity and needs a new preregistered, independent confirmation.

## 10. Limitations

- DEV has 48 scenarios in 16 families; most scenarios (43) have identical casualties under both policies, so the test has low power for small effects. The point estimate (−0.25) is not evidence of a benefit.
- One-step improvement over T1 with T1 continuation; at most *M* = 4 people are scored per decision.
- *K* = 8 gives coarse values; the deviation rule (z = 1) accepts some noise-driven deviations.
- The terminal value is a heuristic (BFS distance, no congestion) and was set by reasoning, not calibrated.
- Values are conditional single-person estimates with others held to T1, not joint or causal.
- The CPU budget makes rare decisions machine dependent; they are labeled.
