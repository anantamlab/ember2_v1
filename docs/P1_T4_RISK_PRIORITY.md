# P1-08: T4 risk-based priority reference agent (issue #24)

**Status: implemented and CPU tested. Protocol frozen before any T4 decision on DEV.** The CPU pilot (§7) runs on synthetic scenarios only, after the freeze commit. The DEV comparison (§8) is prepared and gated. It is **not run** in this change; it needs review of the pilot report first.

T4 is a standalone reference policy. It is not a T1 variant and does not import T0–T3. It does not change T0–T3, the frozen datasets or the budget. No GPU, Modal, training, distillation, S1 teacher data or EMBERBench is used.

## 1. Architecture

| Module | Role | Imports (enforced by an AST test) |
|---|---|---|
| `teacher/t4_risk.py` (`t4-risk@1.0.0`) | Public fire forecast, per-exit fields, per-person urgency and stratum | `teacher.grid` + stdlib |
| `teacher/t4.py` (`t4-priority-planner@1.0.0`) | `T4RiskPriority` (A2), `T4NoPriority` (A0), `T4DistancePriority` (A1); one shared planner | `sim.movement`, `teacher.grid`, `teacher.interface`, `teacher.t4_risk` + stdlib |
| `teacher/t4_pilot.py` | CPU feasibility pilot, DEV cost estimate, gate | — |
| `teacher/t4_eval.py` | Gated paired DEV comparison with frozen T1 (not run) | — |

All three policies implement `ReferencePolicy`. Input: one validated `Simulator.observe()` dict. Output: the full ID → action map and deterministic diagnostics. The policy is deterministic, seedless (`seed = None`) and stateless between decisions. Unknown or private observation keys raise `ValueError` (`contract.validate_observation`).

T4 reads only: map (size, walls, exits, exit flammability), public fire rules (model, spread interval, spread probability, `enter_fire`, `max_steps`), the current step, the current burning set and people (ID, position, status). It never reads future fire, RNG state, trapped flags, scenario IDs, labels, family IDs or evaluator seeds. Tests check invariance to the scenario seed, scenario ID and list order.

## 2. Urgency model (`t4_risk.py`)

### 2.1 Public fire forecast

- Layer `l(c)`: the least number of spread steps that can ignite cell `c` (BFS from the burning set over flammable cells; exits only if `exits_flammable`).
- Spread happens in absolute step `t` only if `t % spread_interval == 0`. `n(τ)` = spread steps among the next `τ` steps.
- P(c burns by relative step τ):
  - deterministic: `1` if `n(τ) ≥ l(c)` else `0` (exact; a test checks it against the simulator);
  - stochastic, `l = 1`: `1 − (1 − p)^(k·n)`, `k` = burning 4-neighbours now (exact);
  - stochastic, `l ≥ 2`: `P(Binomial(n, p) ≥ l)`, a single-front **lower bound** (optimistic when fronts converge).
- Arrival estimate `T_q(c)`: the first relative step with P ≥ q. Two quantiles: `q_early = 0.25` (pessimistic for the person) and `q_late = 0.75` (optimistic). A value after the episode end is `CAP` ("not before the end").

### 2.2 Per-exit fields (open cells now; other safe exits absorb)

- `d_e(c)`: static BFS distance to safe exit `e`.
- `L_e^q(c)`: latest departure. It is the latest relative step at which a person can be on `c` and still reach `e` with every visited cell unburnt: `L(e) = T_q(e)`, `L(c) = min(T_q(c) − 1, max_n L(n) − 1)`. A max-min Dijkstra computes it. `m_e^q(c)` is the length of that route. A test checks `L` against a brute-force time-expanded search.
- `G_e(c)` (DP over the BFS DAG, under `T_early`): a shortest route from `c` is fire-viable iff `G_e(c) > d_e(c)`.

People, waiting and queues are not in these fields.

### 2.3 Six signals per person (units: simulator steps)

| Signal | Definition | Missing / unreachable |
|---|---|---|
| Fire proximity | `l(c)` (spread steps) | `None` if fire cannot reach `c` |
| Time until exposure | `T_early(c)` | `CAP` |
| Shortest viable evacuation time `V` | `d_e(c)` on the nearest exit with a viable shortest route; else the latest-departure route length | `None` if no exit is viable |
| Congestion delay `Q` | FIFO queue at the person's nominal exit, by static distance. People at the same distance share the mean delay of their group, so `Q` does not depend on IDs | 0 |
| Time until viable routes are lost | `slack + route_len`, where `slack = max_e L_e^early(c)` (tie: shorter route, then exit ID) | `None` if `slack = −∞` |
| Independent escape options `K` | number of safe exits viable under `T_late` | 0 |

**Margin** `M = time_to_route_loss − evacuation_time = slack − Q`, with `evacuation_time = route_len + Q`. A test checks this identity.

### 2.4 Normalization and urgency

Each component is in [0, 1]; higher means more urgent. `ratio(s, x) = s / (s + max(0, x))`.

| Component | Formula | Weight | Scale |
|---|---|---|---|
| margin | `ratio(6, M)`; 1 if `M` is `None` | 0.35 | 6 |
| exposure | `ratio(8, T_early)`; 0 if `CAP` | 0.20 | 8 |
| proximity | `ratio(4, l − 1)`; 0 if `None` | 0.10 | 4 |
| options | `1 / K`; 1 if `K = 0` | 0.15 | — |
| congestion | `Q / (Q + 8)` | 0.10 | 8 |
| evacuation | `V / (V + 16)`; 0 if `None` | 0.10 | 16 |

`urgency = Σ w_k · u_k`, rounded to 6 digits so that float noise cannot change the order. The weights sum to 1 (validated). The margin has the largest weight because it combines route loss, travel and queueing.

### 2.5 Risk strata

| Stratum | Rule |
|---|---|
| `no_viable_route` | `K = 0` (no exit viable even under `T_late`) |
| `critical` | `slack < 0` (no route viable under `T_early`) or `M < 3` |
| `high` | `3 ≤ M < 10` |
| `elevated` | `10 ≤ M < 25` |
| `low` | `M ≥ 25` |

Viability label: `none` (`K = 0`), `uncertain` (`slack < 0`, `K > 0`), else `viable`.

### 2.6 Planning order (A2), ties and stability

Key: `(stratum index, −urgency, nominal distance, row, col)`. The stratum comes first, so a coarse class decides most of the order. Ties break by **position**, never by ID. A test permutes IDs and checks that the risk records and the planning order by position are unchanged. `no_viable_route` people come last: no route can save them under the public forecast. They shelter (§3.4) and do not take queue slots from people that can still escape.

Stability: the policy is stateless (F3), so there is no hysteresis. Coarse strata, rounding and position tie-breaks reduce order changes between steps. Oscillation is possible and is measured (reversals per 100 moves, §8).

### 2.7 Uncertainty and limits

- No time-to-death claim. `T_q` is a public-rule estimate. For stochastic fire with `l ≥ 2` it is optimistic.
- Congestion uses a FIFO static-distance model. It ignores corridor capacity and blocking between people.
- `K` counts exits, not disjoint paths. Two exits through the same corridor count as two options.
- Urgency is an ordering heuristic. The weights and scales are reasoned values (§5), not fitted.

### 2.8 Complexity per decision

Area `A`, safe exits `E`, active people `N`.
- Forecast: `O(A)`. Fields: `O(E · A log A)` (BFS + Dijkstra + DP).
- Person signals: `O(N · E)` plus the FIFO sort `O(N log N)`.
- Planner: at most `node_budget_per_person = 256` A* expansions per person and `node_budget_total = 20000` per decision; each expansion is `O(log)`. At most `repair_rounds + 1 = 7` resolver calls (`O(N)` each).
- CPU budget: 2000 ms per decision (§3.6).

## 3. Priority-controlled movement (`t4.py`)

The simulator's resolver (lowest ID wins a contested cell) is not changed. T4 never relies on the resolver to enforce priority. It plans so that a lower-priority person never contests the cell of a higher-priority person.

1. **Priority-aware exit assignment.** People are taken in planning order. Candidates: up to `candidate_exits = 3` exits viable under `T_late`, within `max_detour = 12` steps of the nearest viable exit (F1). A nominal descent on each exit field holds one person per (narrow cell, step); narrow = safe exit or an open cell with ≤ 2 open neighbours. A later person waits in this table, so its cost increases and it can be diverted to another exit (**alternative routing for lower-risk people**). Cost = arrival + `2.0·[not nearest viable]` + `20·Σ hazard` + `4·max(0, delay − L_early)`. Infeasible if delay > `L_late` or the maximum hazard ≥ 0.5.
2. **Space-time reservations.** Windowed cooperative A* (horizon 8) in planning order, with vertex and edge (swap) reservations. Step cost: 1 per step (move or stay), + `0.25` per move, + `20 · hazard(cell, τ)`; a cell with hazard ≥ 0.5 is excluded. Heuristic: static exit distance × 1.25 (admissible). Tie-break `(f, h, generation order)`.
3. **Deliberate yielding.** An earlier planner may reserve a later person's current cell at τ = 1. The later person must **vacate**: it searches for a full plan that leaves its cell (step aside or reroute). If none exists it stays, and the earlier planner is repaired. A later person never takes a (cell, τ) that an earlier planner holds, so it waits (**yielded_wait**) when the urgent person needs the cell.
4. **Verification and repair.** The planner runs the public resolver on the joint action. For up to 6 rounds: a cycle re-plans its lowest-priority member; a contest whose ID winner has lower priority than a loser re-plans the winner (it yields); else the blocked mover is re-planned. Moves still predicted blocked become `stay`. The planner asserts that end positions do not change.
5. **Shelter.** No viable exit: the person moves toward reachable cells with the latest `T_late` (label `sheltering`).
6. **Fallback.** CPU or node budget exhausted: reservation-consistent greedy descent (`cpu_budget_fallback`, `node_budget_fallback`). No admissible successor: least-hazard action. Each fallback has a reason label.

### 3.1 Fairness and starvation controls

- **F1** detour cap: an alternative exit is at most 12 steps longer than the nearest viable one.
- **F2** a `critical` person vacates only by a progress move, so priority cannot push an endangered person back.
- **F3** stateless: no aging and no memory. Starvation is bounded by fire progress: a waiting person's margin falls, so its stratum rises. Tests check that a low-risk person behind four urgent people still evacuates and waits at most four times.

### 3.2 Per-person diagnostics

Rank, priority class, protected flag, urgency components, stratum, signals and uncertainty (A2: `public_view`; A0/A1: a minimal view), target exit, diverted flag, intended move (static descent), actual action, predicted outcome (checked against the resolver), plan status, fallback flag and reason, priority effect (`moved_as_intended`, `moved_alternative`, `waited`, `yielded_wait`, `yielded_step_aside`, `yielded_reroute`, `vacate_refused`, `converted_to_stay`, `sheltering`), `yielded_to`, contested cell, `yield_over_id_rule` (the yielder has the lower ID), `beneficiary_of`, expansions and repairs. Decision level: order, classes, effect counts, yields, effective yields (the beneficiary is predicted to move into the contested cell), budget, repair rounds, resolver calls. Wall and CPU time are in `runtime` only.

## 4. Ablations (one planner, frozen before DEV)

| ID | Class | `priority_scheme` | Planning order | Classes |
|---|---|---|---|---|
| A0 | `T4NoPriority` (`T4-A0`) | `none` | ascending ID (the resolver's own order) | `neutral` |
| A1 | `T4DistancePriority` (`T4-A1`) | `fire_distance` | `(no static route last, fire layer, static distance, row, col)` | bands ≤ 3 / ≤ 8 / ≤ 16 layers |
| A2 | `T4RiskPriority` (`T4`) | `risk` | §2.6 | strata §2.5 |

Every other number and rule is identical (a test checks it). A0 never yields over the ID rule, because its order is the ID order. T1 is an external baseline only.

## 5. Frozen configuration

Shared numbers (`t4._FROZEN`):

| Group | Values |
|---|---|
| Quantiles | `q_early 0.25`, `q_late 0.75` |
| Weights | margin 0.35, exposure 0.20, proximity 0.10, options 0.15, congestion 0.10, evacuation 0.10 |
| Scales | margin 6, exposure 8, proximity 4, congestion 8, evacuation 16; `urgency_digits 6` |
| Strata (margin, steps) | critical < 3, high < 10, elevated < 25 |
| A1 bands (layers) | 3, 8, 16 |
| Planner | horizon 8, candidate exits 3, alt-exit margin 2.0, max detour 12, risk weight 20, risk hard 0.5, deadline weight 4.0, move extra cost 0.25 |
| Budgets | 256 nodes/person, 20000 nodes/decision, 6 repair rounds, 2000 ms CPU/decision |

Descriptive keys (`t4._DESCRIPTIVE`) name the models, tie-breaks and rules and are not configurable. Identities (`name`, `policy`, `version 1.0.0`) and pinned config hashes (`tests/test_teacher_t4.py`):

| Variant | Name / policy | Config hash |
|---|---|---|
| A2 | `T4` / `risk_based_priority` | `sha256:dbcd0af4752ffcaad00e89a38ffcba5d3867fa3e4de0cce9fd560533d4794195` |
| A0 | `T4-A0` / `risk_priority_ablation_no_priority` | `sha256:f561075ebc1c2cfbffeb55d196bc2675801b167d11e4c4b157079e33fc506243` |
| A1 | `T4-A1` / `risk_priority_ablation_fire_distance` | `sha256:e9a70eab344df4237a41efdda2522a7547ac2f94b3549ce821b96c5bae49b9ff` |

**Provenance.** The numbers come from reasoning about the step units of the public rule and from synthetic smoke tests (`sim.bench.room` and `tests/gen_helpers.random_scenario`). Rationale: `q_early 0.25` plans against an early but plausible fire arrival; scales are about half of typical route lengths, so the components separate near-term cases; strata thresholds are short, medium and long in steps. No T4 decision ran on DEV, train, calibration, test, blind or EMBERBench data. No number was tuned to the T1/T2/T3 DEV results. A synthetic check on 2000 random scenarios gave casualties A0 623, A1 622, T1 621, A2 620 (small differences; A2 worse than A0 in some scenarios). It is a smoke result only, not evidence.

## 6. Tests (`tests/test_teacher_t4.py`, `tests/test_teacher_t4_pilot.py`)

| Area | Checks |
|---|---|
| Identity | Pinned hashes; distinct from T0–T3; shared numbers across A0/A1/A2; invalid or descriptive config rejected |
| Independence | AST import allow-list for `t4.py` and `t4_risk.py`; no private names in code; private keys rejected |
| Reproducibility | Repeat and stateless; list-order invariance; scenario seed / ID invariance |
| Forecast | Deterministic arrival = simulator; stochastic layer-1 exact, quantiles ordered, probability monotone; spread interval |
| Fields | Latest departure = brute-force time-expanded search |
| Urgency | Components in [0, 1]; margin identity; strata thresholds; ID-permutation invariance; urgency rises with fire spread |
| Priority changes movement | Exit contest (`b.0.z*`): T1 and A0 lose `Pz`; A1/A2 save both; `Pb` yields over the ID rule and `Pz` evacuates. Alternative route (`1..b.0.zyxw*`): `Pb` diverts to `E1`, all saved. Step aside: lower-ID `Pa` vacates for critical `Pd`, who takes the cell. A0 never yields over the ID rule; A2 does |
| Safety | Two urgent people in one corridor; opposing flows (no blocked moves, no swaps); follow chain and cycle; exit that burns first is not targeted; no safe exit → shelter; low-risk starvation bound; 200 people at one exit |
| Budgets | CPU budget 0 and node budget 0 give valid, labeled fallbacks |
| Scale | Valid maps for N = 1, 10, 50, 100, 200 (all variants); closed-loop random episodes legal with predicted = actual |
| Negative | A2 has more casualties than A0 in some synthetic scenarios and fewer in others |
| Pilot / gate | 52 synthetic cells; high-ID risk ordering; unreachable cells; cost formula; estimate; gate logic; DEV harness refuses without a passing pilot for the same hashes and writes no output; decision rule |

## 7. Preregistered CPU feasibility pilot and gate (`t4_pilot.py`)

Run once after the freeze commit: `python -m ember_v1.teacher.t4_pilot`. One CPU process; wall limit 3600 s.

1. Microbenchmark: one decision per variant on `sim.bench.room` 32×32, N = 50, stochastic, step 3.
2. Synthetic cells (52; pilot-only seeds `t4-pilot@1/...`; IDs `P000…` so string order = numeric order):
   - base: 16×20 and 48×64 × {open room, corridor, multi room} × N = 1, 10, 50, 100, 200; stochastic p = 0.5; 3 exits;
   - deterministic: both grids × {open room, corridor} × N = 10, 100;
   - exits: 48×64 × {open room, multi room} × N = 50 with 1 and 6 exits;
   - high-ID risk: the person nearest the fire has the highest ID (16×20; 2 topologies × 2 fire models);
   - near entrapment: people within 6 cells of the ignition (corridor; N = 10, 50; 2 fire models);
   - initially unreachable: sealed rooms; half the people start without an exit path (both grids).
   16×20 cells run full episodes; 48×64 cells run the first 12 decisions. Every cell runs A0, A1 and A2.
3. DEV workload profile: frozen T1 replay on the 48 DEV scenarios (public workload and T1 time only). T4, A0 and A1 do **not** run on DEV.
4. Cost model per DEV decision: `W = area · (1 + E · (3 + [stochastic])) + 40 · active`; `est = α_v · W`, capped at budget + 500 ms. α from unbudgeted pilot decisions: median (central); p90 × 1.25 (upper). Evaluator metrics: `α_eval · W` for 4 policies. Total = T1 (measured) + A0 + A1 + A2 + evaluator.
5. Gate (all must hold):
   - G1 bounded: complete inside the wall limit; valid maps; 0 blocked moves; 0 prediction mismatches; max decision CPU ≤ 2500 ms; peak RSS ≤ 1024 MB;
   - G2 upper full-DEV comparison CPU ≤ 2 h;
   - G3 predicted DEV decisions at the cap ≤ 5 % per variant;
   - G4 fallback person-decisions ≤ 5 % of routable; budget-limited decisions ≤ 5 %, per variant;
   - G5 mechanism: A2 has ≥ 1 effective yield over the ID rule; A0 has 0 yields over the ID rule.

   `FULL_DEV_FEASIBLE` if all pass. `BLOCKED_BY_COMPUTE` if G1–G4 fail. `BLOCKED_BY_MECHANISM_CHECK` if only G5 fails.

Reported: CPU p50/p95/p99/max per variant, grid and N; CPU per episode; peak memory; fallback and budget-hit rates; blocked moves; contests and priority enforcement success; unnecessary waits; yields; diversions; waits by stratum.

## 8. Preregistered DEV analysis (`t4_eval.py`; NOT RUN)

- Gate: refuses unless the pilot decision is `FULL_DEV_FEASIBLE` for the same A0/A1/A2 config hashes. Wall limit `min(6 h, max(900 s, 1.5 × upper estimate))`.
- Data: the 48 DEV scenarios of plan `p1-02-v1` only (`t1_eval.load_dev`). T1, A0, A1 and A2 run each scenario with its own simulator seed.
- **Primary endpoint:** casualties per scenario, A2 − T1.
- Comparisons: A2−T1 (primary), A2−A0, A2−A1, A1−A0 (secondary).
- Other outcomes: evacuated fraction, unresolved, casualties + unresolved, evacuation success and casualties among initially at-risk people (`critical` + `high` strata from the step-0 public observation, same frozen risk module for every policy), steps, completion step, blocked moves per 100 attempts, collisions, reversals per 100 moves, wait rates for at-risk and low-risk people, unnecessary waits, priority enforcement rate, policy CPU; also fallback counts and per-stratum totals.
- Statistics: paired per scenario; family-cluster bootstrap (10 000 resamples, 95 %, fixed seeds per comparison); family sign test. **Decision rule:** "better" or "worse" only if the CI excludes 0 and the family sign test has p < 0.05 in the same direction; else `no_evidence_of_difference`. Secondary outcomes are exploratory and unadjusted.
- Negative findings are kept: the report lists every scenario where A2 is worse than T1 or A0.
- T2 and T3 do not run. Train, calibration, test, blind sets and EMBERBench are not read.

## 9. Pilot results

Filled after the single pilot run (`data/baselines/p1-08-t4-pilot.json`).

## 10. Limitations

- The resolver's lowest-ID rule still decides any contest that T4 did not prevent. Priority acts only through coordinated actions.
- The forecast is a public-rule estimate; the `l ≥ 2` stochastic bound is optimistic. Congestion is a FIFO model.
- Statelessness avoids hidden memory but allows order changes between steps.
- Urgency weights are reasoned, not fitted. A2 can be worse than A0 or T1 in specific scenarios (§5).
- The pilot uses synthetic layouts; DEV cost is a model estimate.
