# T4 v1.2.0: node-budget repair and one confirmation pilot (preregistration)

**Status:** PREREGISTERED. This document, `src/ember_v1/teacher/t4_v12.py`, `src/ember_v1/teacher/t4_v12_pilot.py`, `tests/test_teacher_t4_v12.py` and the exploratory smoke report `data/exploratory/p1-08-t4-v12-smoke.json` are committed **before** any v1.2.0 policy runs on a `t4-pilot@3` cell. After that commit, the planner, the identities, the configuration, the seed namespace, the cells and the gate do not change. The v1 and v2 protocols, the v1.0.0 and v1.1.0 policies and hashes, their pilot reports and their blocked decisions (`BLOCKED_BY_MECHANISM_CHECK`, `T4_V2_BLOCKED`) stay as they are.

This is **one bounded attempt** that addresses only the G4 failure of v1.1.0 (node-budget exhaustion under transactional rollback). It does not change urgency weights, risk thresholds, strata, fire forecasts, priority orders, the A* cost, the reservation model or any frozen number.

## 1. The v1.1.0 failure

The v2 pilot (`data/baselines/p1-08-t4-v2-pilot.json`) failed only G4: A2 fallback fraction 0.185 and budget-limited fraction 0.071 (limits 0.05); A1 0.191 and 0.074; A0 passed (0.046, 0.017).

Diagnosis on an **exploratory** namespace `t4-explore@v12` (never a pilot namespace; 6 crowded 16×20 cells, N ≥ 100, 15 decisions each, A2 v1.1.0):

| Measure | Value |
|---|---|
| Decisions / limited decisions | 90 / 16 |
| Fallback person decisions (all `node_budget_fallback` in limited decisions, plus 34 `no_successor_least_hazard`) | 2 216 of 9 180 routable |
| A* nodes: vacate searches / root searches | 448 435 (81 %) / 105 412 |
| Vacate transactions that failed / succeeded | 14 559 / 3 001 |
| Identical repeated searches (full reservation state) | 0 |
| Most expensive root transaction | 15 921 of 20 000 nodes, 641 vacate requests to 36 people, push depth up to 27, **all** failed; the root then stays |

So the budget does not go to identical searches. It goes to a depth-first walk of push chains through a boxed crowd: every chain fails, and each failure is found only after full A* searches of every member. One early root can use most of the decision budget; every later person then gets the greedy fallback.

Two mechanisms were tried on the exploratory cells before this freeze:

1. **Exact reuse of failed vacates** (read-set validated memo with strict invalidation). It was exact (0 action differences without a limit) but reused only 3 076 failures; 16 limit hits remained, because a re-ask almost always reads a different requester path. Dropped.
2. **Exact infeasibility proof** (section 2). Adopted.

## 2. v1.2.0 change (`t4-priority-planner@1.2.0`)

`ProofPlanner` subclasses the frozen v1.1.0 `TransactionalPlanner` and adds one check at the entry of every vacate request (`plan_tx` with a requester). In v1.1.0 a vacate of `o` succeeds only through a push chain of distinct unplanned people `o = q0 → q1 → … → qk` in which each `q(i)` plans a τ = 1 move into the cell of `q(i+1)`, and `qk` moves into an empty cell or into the cell of a planned mover. Every move must pass `_blocked` when it is searched. The check uses only blocks that hold at frame entry and stay true for the whole frame (inside a frame only frame members add reservations and plans):

| | Block (all from v1.1.0) | Why it stays true in the frame |
|---|---|---|
| B1 | τ = 1 hazard ≥ hard; no finite distance | static |
| B2 | F2: a protected person's non-progress move | static; every frame member is a vacate |
| B3 | vertex key (n, 1) or the reverse edge key is held | held keys belong to people outside the frame; no member can release them |
| B4 | the planned occupant's τ = 1 walk reaches q's cell or o's cell | outside plans do not change; `o` is planned while any other member searches, so the v1.1.0 cycle check (T5) closes through the chain |
| B5 | the cell is o's cell for another member | same cycle as B4 |

A breadth-first search over unplanned people from `o` with these blocks only decides whether any chain can reach an empty or follow cell. If none can, the v1.1.0 vacate **must** fail, and a failed v1.1.0 vacate restores all reservations, plans, origins and forbid sets (T3). v1.2.0 then returns the same failure **without A* nodes**. Forbid sets, the full-plan requirement, later-τ reservations and the depth and node limits can only add failures, so the proof leaves them out.

Consequences:

- When v1.1.0 hits no limit, v1.2.0 decisions are **identical** to v1.1.0 (tested).
- When v1.1.0 hits a limit, v1.2.0 spends the saved nodes on later people.
- Node accounting is unchanged: `expansions` counts A* nodes actually expanded. `transaction.vacate_failed_infeasible` counts proved failures; they also count in `vacate_requests` and as refusals.

Identities (planner `t4-priority-planner@1.2.0`; config = v1.1.0 config with `planner` changed and the descriptive key `vacate_infeasibility_proof` added; every number unchanged):

| Name | Class | Priority | Config hash |
|---|---|---|---|
| T4 (A2) | `T4RiskPriorityV12` | risk | `sha256:79bd3322b9cf7ccd25928f3a70502089732496af5427f88f00ecd2ff6ba12504` |
| T4-A0 | `T4NoPriorityV12` | none (ID order) | `sha256:d95ac217d8dc61206c15fa508c6534ea89c2a259b5f6d964cb82ed69bcb047be` |
| T4-A1 | `T4DistancePriorityV12` | fire distance | `sha256:f177bc8fe197dde7c45d1586b4f3e988baa4dc334d447519af7c8b721b042638` |
| T1 | `T1CongestionAware` | – | as in `data/baselines/p1-04-t1-dev-v1.json` |

## 3. Tests (`tests/test_teacher_t4_v12.py`)

- Identity pins, the exact two-key config diff, v1.0.0 and v1.1.0 hashes unchanged.
- **Proof soundness:** for every proved failure, the frozen v1.1.0 `plan_tx` runs on the same state with an unlimited node budget; it must fail and restore the state exactly (randomized crowded grids and a crowded room).
- **Equivalence:** on every decision where v1.1.0 hits no limit, v1.2.0 actions and plan statuses equal v1.1.0, with no more A* nodes.
- v1.1.0 invariants for v1.2.0: I1 precedence audit, I2 no overwrite, I4 resolver agreement, I5 A0 integrity, budget fallbacks, push depth 0, determinism and order invariance.
- Pilot module: v2 scenarios unchanged by the parametrization (content hashes equal the committed v2 report), `t4-pilot@3` disjoint from v1, v2 and the exploratory set, decision label, refusal to overwrite.

## 4. Pre-freeze exploratory smoke (not evidence)

`python -m ember_v1.teacher.t4_v12_pilot --exploratory` ran the complete pilot procedure and every gate metric on the 74-cell design under namespace `t4-explore@v12` (seeds `derive_seed("t4-explore@v12/sim", ...)`). Report: `data/exploratory/p1-08-t4-v12-smoke.json` (decision rows omitted). It is exploratory: it checks that the gate is computable and that no metric is near a limit before the freeze. It is never a confirmation.

Result (74 cells, 0 rejected, 0 errors, wall 397 s, peak RSS 49 MB):

| Check | Smoke value | Limit | Pass |
|---|---|---|---|
| G1 | 0 predicted blocked, 0 blocked moves, 0 prediction mismatches, 0 verification mismatches; max decision CPU 218.8 ms | ≤ 2 500 ms; RSS ≤ 1 024 MB | yes |
| G2 | upper DEV comparison CPU 381 s | ≤ 7 200 s | yes |
| G3 | predicted cap fraction 0.0 (A0, A1, A2) | ≤ 0.05 | yes |
| G4 A0 | fallback 1 046 / 55 534 = 0.019; budget-limited 3 / 1 743 = 0.002 | ≤ 0.05 each | yes |
| G4 A1 | fallback 943 / 56 618 = 0.017; budget-limited 1 / 1 841 = 0.001 | ≤ 0.05 each | yes |
| G4 A2 | fallback 835 / 56 432 = 0.015; budget-limited 0 / 1 807 = 0.000 | ≤ 0.05 each | yes |
| G5a | 0 audit violations, 0 inversions | 0 | yes |
| G5b | 0 A0 ID inversions, 0 unexplained over-ID yields | 0 | yes |
| G5c | 388 verified yields; at-risk progress A2 10 166 ≥ A0 8 364 | ≥ 1; A2 ≥ A0 | yes |

For comparison, v1.1.0 on the v2 pilot had A2 fallback 0.185 and budget-limited 0.071. The smoke does not change any part of section 5 or 6.

## 5. Confirmation pilot (synthetic, independent)

- Namespace `t4-pilot@3/...`; scenario seeds `derive_seed("t4-pilot@3/sim", ...)`; scenario IDs `t4-pilot3-...`. Not generator plans, families or splits.
- Cells: the 74-cell v2 design (`t4_v2_pilot.pilot_cells_v2`, see `docs/P1_T4_RISK_PRIORITY_V2.md` section 4): 16×20 cells run full episodes, 48×64 cells the first 12 decisions.
- **Exclusion (fixed before the run).** A cell is rejected, recorded and not replaced if its content hash or seed equals any v1 pilot, v2 pilot or `t4-explore@v12` scenario, if its layout equals one of the two crowded v1 cells inspected in the v1 diagnosis, or if it duplicates another cell. The pre-run check (scenario construction only, no policy) rejected 0 of 74.
- Procedure: `t4_v2_pilot.build_report` with the v1.2.0 identities: A0, A1, A2 closed loop on every cell; neutral paired states (frozen T1 acts, A0 and A2 decide on its observations); the frozen-T1 DEV workload profile for the v1 cost model (public workload and T1 time only; **no T4 decision on DEV**).

## 6. Gate and run rules

All checks of `gate_v2` (`docs/P1_T4_RISK_PRIORITY_V2.md` section 5), **unchanged**: G1 bounded, G2 DEV CPU estimate, G3 cap hits, G4 planner health (fallback ≤ 0.05 of routable person decisions and budget-limited ≤ 0.05 of decisions, per variant), G5a precedence, G5b A0 integrity, G5c priority effect.

| Condition | Decision |
|---|---|
| Every check passes | `T4_READY_FOR_INDEPENDENT_SELECTION_EVALUATION` |
| Any check fails | `T4_V12_BLOCKED` (with the failed list) |

- `python -m ember_v1.teacher.t4_v12_pilot` runs **once**, one CPU process, no GPU, no network. Report: `data/baselines/p1-08-t4-v12-pilot.json`. The entry point refuses to overwrite it.
- A failed gate is a result. v1.2.0, the identities and this sample are not changed, tuned or re-run after the result. T4 then stays blocked.
- A pass does **not** select T4 and does not authorize a DEV comparison here. It only makes T4 v1.2.0 a candidate that the independent selection protocol (issue #25, `docs/P1_TEACHER_SELECTION_PROTOCOL_DRAFT.md`) may admit through its own frozen T4 entry rule and evaluate on its own prospective data.
- An interrupted run is not restarted silently; its state and the decision are recorded with the reason.

## 7. Limitations (stated in advance)

- v1.2.0 was designed after the v2 pilot failure, on exploratory cells of the same design. Those cells and the v1 and v2 cells are excluded from `t4-pilot@3`.
- The pilot has few independent cells; outcome totals (casualties, evacuations) are reported, not gated, and support no outcome claim.
- The proof is exact for the v1.1.0 rules as coded. The CPU-time limit depends on the wall clock in v1.1.0 already and is outside the equivalence claim.
- G5c is a mechanism check on single decisions; it does not show episode-level rescue.
