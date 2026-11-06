# P1 teacher direction: five reference policies (T0–T4)

**Status: design note.** T0 is implemented (issue #7, [P1_T0_BASELINE.md](P1_T0_BASELINE.md)) with the shared `ember_v1.teacher.ReferencePolicy` interface and `grid-utils@1.0.0`. T1–T4 and the #25 coordinator are not implemented. This note extends HLD §4 D4.

## 1. The five policies

| ID | Policy | Issue | Output | Per-action values |
|---|---|---|---|---|
| T0 | Fire-aware shortest path: BFS to the nearest reachable exit; avoids cells that burn now (no fire forecast in T0 v1) | #7 | Full ID → action map | No |
| T1 | Congestion-aware coordination: prioritized cooperative planner (ascending ID), space-time reservation table, follow chains allowed, swaps forbidden, exit load balancing | #8 | Full ID → action map | No |
| T2 | Stochastic fire forecasting and rollout planning: for each person and action, Monte-Carlo returns over K teacher-sampled fire futures and H steps, with a fixed continuation policy | #9 | Full ID → action map | **Yes** (real Monte-Carlo estimates) |
| T3 | Dynamic ACO: ant-colony routing over a hazard- and congestion-weighted graph, replanned every step | #23 | Full ID → action map (not only paths) | No |
| T4 | Risk-based priority: per-person evacuation urgency sets planning order and route allocation | #24 | Full ID → action map (not only ranks) | No |

**T3 and T4 are standalone policies.** They are not renamed T1 variants, and they do not replace T0–T2. Each policy has its own name, version, configuration, code revision, seed and evaluation.

## 2. Inputs (all five)

Every policy reads only `Observation.to_dict()` (map, rules, people, burning, step). It never reads the simulator seed, the fire RNG state, the trapped set or a future burning set. To reason about fire, a policy samples its own futures from the public rules (`fire_model`, `spread_interval`, `spread_probability`, `exits_flammable`) with a teacher seed. The teacher seed comes from the episode lineage and the step, and the trace records it.

## 3. Shared planning utilities

Policies may share utility code. A shared utility is a pure function of the observation, the public rules and an explicit seed:

- fire-aware distance fields and reachable-exit sets;
- per-cell ignition probability from the public rule, `1 - (1 - p)^k` for `k` burning neighbours (1 if deterministic);
- teacher-sampled fire futures (a public-model API built from an observation and a teacher seed; it must not copy the evaluator's RNG);
- joint-action checking with the exact resolver, `ember_v1.sim.movement.resolve_moves`.

A shared utility has its own version, and every trace records it. Sharing code does not merge policies: each policy's decisions, scores, evaluation and provenance stay separate.

## 4. T3: dynamic ACO

- Graph: grid cells, 4-neighbour edges, walls removed.
- Edge cost at the planned arrival step: `1 + w_fire · P(cell burning by arrival) + w_cong · expected occupancy`. Cells that burn now have infinite cost.
- Ants per person and exit. Pheromone evaporates. Costs are recomputed from each new observation; pheromone may carry over as a warm start.
- Paths become a complete action map for all active IDs, checked with the exact resolver.
- Colony size, iterations, evaporation, replanning trigger and a hard CPU budget are fixed before evaluation. A fallback action is labelled as a fallback in the trace.
- Deterministic for a given teacher seed.

## 5. T4: risk-based priority

Per active person, from the observation and the policy's own uncertain forecasts only:

| Signal | Definition (public data only) |
|---|---|
| Immediate hazard exposure | Burning 4-neighbours now; steps until the next spread step (`interval - step % interval`); ignition probability of the current cell at that step. |
| Predicted entrapment risk | Over teacher-sampled futures: the fraction in which every reachable non-burning exit is cut off before the person's fire-aware arrival time. |
| Escape-time margin | Fire-aware shortest path length to each usable exit, plus the expected queue, compared with the estimated time before the route becomes unsafe. |

Urgency sets T4's own planning order, route allocation and reservations. **Urgency never overrides movement resolution:** the resolver decides every contest by lowest ID. T4 plans against the resolver's real result; an urgent high-ID person who contests a cell with a lower-ID person loses that cell. T4 records the per-person urgency components and the intended versus actual priority.

## 6. Comparison and combination (#25)

1. Evaluate **all five separately first**, on independent DEV scenarios with paired seeds. Measure casualties, evacuated and trapped people, congestion, latency, teacher CPU-seconds and failure rate. Agreement between teachers is not a quality measure.
2. **Do not assume that a more complex policy is better.** If T0 or T1 matches T2, T3 or T4, report that and prefer the simpler policy.
3. An optional candidate arbiter (#25) may choose a joint action set from the proposals, with a preregistered value function and observation-consistent sampled futures. It is compared against the best single policy at equal planning compute.
4. **Never blindly average incompatible scores.** Only T2 produces per-action values. T0, T1, T3 and T4 output actions only. A weighted mean of unlike scores is not an acceptable default. Per-action values are never invented for policies that do not compute them.
5. Mixing per-person actions from different policies can create an invalid or harmful joint plan. Every combined joint action is checked with the exact resolver.
6. The five-teacher gate (#25) must select and freeze one training teacher policy (one of T0–T4 or a validated arbiter) before bulk S1 generation. If a combination is not materially better, all five stay as baselines and only the strongest justified policy is distilled.

## 7. Trust boundary of replay verification

Full-episode replay (`verify_replay(partial=False)`, enforced by `sources.ingest_zip` for `teacher_verified`) proves only that the recorded actions, under the recorded scenario and seed, produce the recorded transitions in our simulator.

It does **not** authenticate:

- **teacher identity:** any program can produce a replay-consistent trace and write any teacher name;
- **label quality:** a bad or random policy produces replay-consistent traces too;
- **data provenance:** replay does not show that a trace came from the frozen teacher code, config and seed that it names, or that the scenario came from the approved generator split.

These protections are open work under **issue #10**: signed or hash-bound teacher code/config/seed provenance, re-generation spot checks, the generator and split manifest binding, and duplicate and leakage audits. **Bulk S1 generation is not authorized** by this PR or by replay verification alone. It needs #6, #7–#9, #23, #24, the #25 gate and #10.

## 8. Budget

Adding T3, T4 and the arbiter does **not** increase the research budget. The total Modal research budget stays about **$200** (HLD §5.3: $10 + $15 + $60 + $40 + $35 + $40 reserve). The T0–T4 and arbiter work is CPU work. It must fit inside the existing P1–P2 cap ($15), with per-episode compute recorded. Any extra workload is gated against that cap. The reserve is not spent without a written reason. No phase cap grows silently.

## 9. Guardrails

1. No actual future fire state, simulator seed or RNG position is a policy input. Test plan: each policy's output for one observation is identical under different simulator seeds.
2. Urgency ranking (T4) and pheromone routes (T3) never override the ID-based movement resolver.
3. Each policy is compared with paired seeds before any choice. There is no EMBERBench tuning, evaluation or hidden data.
4. No teacher data generation until the simulator gates pass and the #25 and #10 gates are met.
