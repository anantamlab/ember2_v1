# P1-13 controlled joint-action counterfactuals — method (fixture stage)

**Status:** fixture implementation `joint-counterfactual-fixture@0.2.0`. Synthetic scenarios only.
No TRAIN/DEV/selection data, no preference labels (`training_preference_allowed: false` always), no GPU.

## Question answered

At ONE replayed simulator state, what is the paired difference in outcomes between complete joint action A
and complete joint action B, when each is applied for one step and the SAME frozen continuation policy then
acts, under the SAME future fire? This is a *joint-action intervention effect under that continuation*.
It is not proof that one person's move caused a death, and not an effect over a population of states.

## Procedure (`counterfactual.shared_state_study`)

| Step | Implementation |
|---|---|
| 1 Common state | `verify_fixture` on the P1-12 report, then replay the first `step` records of one teacher's trajectory; every record must regenerate exactly. |
| 2 Five proposals | `multiteacher.probe_shared_state` on that one public observation (teachers are Markov; tested). |
| 3 Disagreement | Group identical joint maps; names are proposer sets (`T1+T4`). One group only → `no_disagreement`, no contrast. |
| 4 Legal alternatives | Each map is complete for every active person. A per-person composite (`composite_alternative`) is refused if the whole map has any blocked move. |
| 5 Intervention | One step from the identical state for every alternative. |
| 6 Matched fire | Replicate 0 = factual fire stream. Replicates 1..K-1 call `Simulator.fork_fire(seed)` with a seed derived from the scenario seed and prefix hash; all alternatives share it. Fire never sees people or actions, so fire is identical across alternatives within a replicate (checked). |
| 7 Continuation | A fresh, registry-pinned, retained teacher (default T1 1.0.0). |
| 8 Measures | casualties, evacuated, unresolved, casualties+unresolved, new casualties/evacuations, blocked moves, contested targets. |
| 9 Horizon, censoring, uncertainty | Branch ends at termination or horizon; non-terminal = censored. Per-replicate deltas, mean, exact sign test (descriptive). Uncertainty covers future fire only. |
| 10 Provenance | Prefix records and hash, fixture content hash, probe hash, proposal hashes, fork seeds, continuation identity, all in `report_hash`; `verify_counterfactual` re-executes every branch and re-derives every verdict. |

## Verdict rule (preregistered constant `VERDICT_RULE`)

- Any censored replicate → `inconclusive (censored)`.
- Equal primary (casualties+unresolved) and guard (casualties) in every replicate → `no_difference`.
- Stochastic fire with fewer than 3 replicates → `inconclusive (insufficient_replicates)`.
- Primary and guard not worse in every replicate and primary strictly better in at least one → `a_better`/`b_better`.
- Otherwise `inconclusive (mixed_sign)`.

`preference_candidate` marks `a_better`/`b_better`. It is an input to the P1-14 validator, which must apply its
own frozen thresholds (for example, a minimum number of strict improvements). Deterministic fire allows one
exact replicate.

## Known limits

- Effect depends on the continuation policy. A continuation-sensitivity check (T1 vs T4 continuation) is
  supported and should be in the TRAIN protocol.
- Weak dominance with many ties (for example 2 strict wins, 3 ties) is a weak preference. P1-14 must decide
  whether it is enough.
- The fork seed derives from the evaluator-only scenario seed. It never enters an observation.
- Cost per state: (alternatives ≤ 5) × replicates × horizon continuation decisions (T1 is cheap) plus one
  five-teacher probe (T2 dominates, about 2 CPU-s per decision in P1-09 terms).
