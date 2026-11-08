# P1 five-teacher selection: preparation (issue #25)

**Status:** protocol validator and gate implemented; protocol **not frozen**. **No teacher is selected.** No new DEV run, no confirmation run, no arbiter, no S1 data. Code: `src/ember_v1/teacher/{proposals,selection,arbiter,registry}.py`; tests: `tests/test_teacher_selection.py`, `tests/test_teacher_selection_gate.py`. Gate today: **`NO_SELECTION`** (no frozen protocol, no prospective evidence). All five candidates are eligible in the draft: T0–T3 v1.0.0 and T4 v1.2.0 (admitted by its independent pilot `t4-pilot@3`, PR #36). T4 v1.0.0 and v1.1.0 stay blocked records.

## 1. Common proposal record (`teacher-proposal@1.0.0`)

`proposals.proposal(decision)` turns any `PolicyDecision` into one record: teacher name, policy, version and config hash; observation digest and step; the **full** ID → action map; per-action values only for T2 (the only teacher that computes them); policy seed; CPU and wall time; `proposal_hash` (all fields except timing).

| Function | Purpose | Fails closed on |
|---|---|---|
| `validate_proposal` | check one proposal against its observation | missing/extra fields, hash mismatch, other observation, partial or unknown IDs, invalid action, values from an action-only teacher, values over non-allowed actions |
| `joint_check` | run a joint action map through the exact resolver | – (reports blocked people; `valid_joint_plan` is false if any move is blocked) |
| `combine` | build one joint map from per-person teacher choices | a missing person, an invalid source proposal. Records the source teacher of every action and the source proposal hashes, so an arbiter decision is never attributed to one teacher |
| `disagreement` | unanimous fraction and pairwise agreement | duplicate teachers. Agreement is descriptive and is **not** a quality measure |

There is no weighted average of unlike scores anywhere.

## 2. Five separated views

`selection.status()` keeps these apart; no function mixes them.

| View | Source | Today |
|---|---|---|
| Retained identities | `registry.inventory()` (shared with #10, `teacher-registry@1.1.0`) | T0–T3 v1.0.0 and T4 v1.2.0 retained (T4 with its admission record); T4 v1.0.0 and v1.1.0 kept as **blocked** records |
| Eligible teachers | `eligibility` of a frozen protocol, checked by `registry.validate_eligibility` | draft: T0, T1, T2, T3, T4 (v1.2.0) |
| Blocked / excluded | the same record plus `t4_entry_rule.blocked_identities`: exact blocked identities, verdicts, gates, report hashes | no candidate excluded; T4 v1.0.0 (G5 `BLOCKED_BY_MECHANISM_CHECK`) and v1.1.0 (G4 `T4_V2_BLOCKED`) are blocked identities of the eligible T4 |
| Historical exploratory evidence | `evidence_inventory()` of P1-03 … P1-06 (`evidence_class: historical_exploratory`, `counts_as_confirmation: false`) | 48 DEV scenarios, 16 families |
| Prospective confirmation evidence | one run after the protocol commit on a new family-disjoint set (`selection_gate`) | none |

## 3. Historical evidence inventory (committed DEV reports)

All reports use the same 48 DEV scenarios (16 families, plan `p1-02-v1`). Where a teacher appears in several reports, all final transition hashes agree (0 conflicts: T0 and T1 each appear in 3 reports).

| Teacher | Reports | Casualties | Evacuated | Unresolved | Policy CPU (s) | Committed verdict |
|---|---|---|---|---|---|---|
| T0 | P1-03, P1-04, P1-06 | 507 | 1 210 | 17 | not recorded in P1-03 | reference |
| T1 | P1-04, P1-05, P1-06 | 345 | 1 314 | 75 | 19.0 | better than T0 on casualties (P1-04) |
| T2 | P1-05 | 333 | 1 318 | 83 | 3 384 | no evidence of a casualty difference vs T1; not selected (P1-05) |
| T3 | P1-06 | 403 | 1 307 | 24 | 69.0 | fewer casualties than T0; no evidence vs T1 (P1-06) |
| T4 | – | – | – | – | – | v1.2.0 admitted without DEV evidence (pilot only, not outcome evidence); v1.0.0 G5, v1.1.0 G4 blocked (PR #32/#33) |

These totals come from **separate** comparisons on DEV, which was used for development. They are descriptive. They can never satisfy the prospective-confirmation requirement: the gate refuses any evidence with a `dev` field, a historical plan ID (`p1-02-v1`) or a non-prospective class. T1 is the **provisional operational baseline** on this historical basis; that is not a new confirmatory selection.

## 4. Protocol (`teacher-selection-protocol@1.1.0`)

`selection.draft_protocol()` is the P1-09 draft (docs/P1_TEACHER_SELECTION_PROTOCOL_DRAFT.md) as data, with `frozen: false`. `freeze_protocol` validates and adds `protocol_hash`; committing the frozen file to the trusted ref is the freeze. `validate_protocol` fails closed on:

- any missing or unknown field; candidates other than exactly T0–T4;
- eligibility invalid against the registry (blocked T4 eligible, T4 missing, exclusion without reason or without the exact blocked records, T4 excluded as `blocked_by_preregistered_gate` although v1.2.0 is retained);
- a T4 entry rule that does not close at protocol freeze, that does not list exactly the blocked T4 records, or (T4 eligible) whose `admitted` record is not the registry admission of the eligible T4 identity: verdict `T4_READY_FOR_INDEPENDENT_SELECTION_EVALUATION`, gate `gate_v2`, report `data/baselines/p1-08-t4-v12-pilot.json` with its hash, freeze commit `b7ff040`, namespace `t4-pilot@3`. An excluded T4 carries no admission;
- a baseline other than eligible T1 on a `historical_exploratory_provisional` basis;
- historical evidence used for anything other than `descriptive_only`;
- a confirmation design with a historical namespace, < 10 families, a different generator revision, an isolation list without all of `p1-02-v1/{train,dev,calibration,locked_test}`, `t4-pilot@1`, `t4-pilot@2`, `t4-explore@v12`, `t4-pilot@3` (T4 v1.2.0 smoke and pilot, PR #36), or more than one run per identity;
- a primary endpoint other than paired family-level casualties vs T1 (95 % cluster bootstrap with ≥ 1 000 seeded resamples, exact two-sided sign test at 0.05);
- missing safety guards (casualties + unresolved and evacuated, no net deterioration vs T1);
- compute caps missing or a USD cap above $15;
- a tie rule that does not order exactly the eligible candidates in the frozen simplicity order T0, T1, T3, T4, T2 (T0 one shortest path; T1 adds cooperative reservations; T3 adds a dynamic pheromone colony; T4 adds risk priority, transactional repair and an exact vacate proof; T2 adds stochastic forecast rollouts per action);
- an inline arbiter (an arbiter needs its own frozen protocol, referenced by `protocol_ref` and `protocol_hash`);
- excluded data without `locked_test`, `blind_test`, `calibration`, `emberbench`; missing stop rules.

The S1 entrypoint (`s1.verify_protocol`) applies this same validator, so the selection and provenance gates accept the same protocols only.

## 5. Selection gate

`selection_gate(protocol, evidence, *, protocol_ref, repo, historical, trusted_ref="origin/main")` returns `NO_SELECTION` with every reason, or a decision with a **receipt draft whose review is pending**. It needs:

1. a frozen, hash-consistent protocol equal to the file at `protocol_ref` in the trusted history;
2. prospective evidence (`teacher-selection-evidence@1.0.0`) under that protocol hash and commit, whose code commit descends from the protocol commit;
3. a confirmation manifest of the preregistered namespace and generator, exactly families × lineages;
4. an isolation audit against every preregistered historical set: no shared family, lineage, scenario hash or core geometry hash;
5. complete matched results for every eligible candidate and no result for an excluded one (a blocked T4 in the evidence stops the gate); identities equal to the protocol; no failed episode; CPU within caps.

Decision rule (frozen in code): a candidate replaces T1 only if the 95 % CI upper bound of the paired casualty difference is < 0, the family sign test has p < 0.05, casualties + unresolved and evacuated totals do not deteriorate vs T1, and its CPU is within `replacement_cpu_s_cap`. Several qualifiers: lower CPU, then the frozen simplicity order. None: `RETAIN_T1_NO_IMPROVEMENT`. The receipt draft has the `ember_v1.s1` receipt format; S1 refuses it until reviewers approve it and it is committed to the trusted ref.

## 6. Joint-map arbiter contract (`arbiter.py`)

Not run and not frozen. `validate_arbiter_protocol` needs observation-only inputs, independent teacher sampling, equal total planning CPU, whole maps only, a resolver-valid requirement, no score averaging, one predeclared `score_id`, eligible candidates and an eligible fallback. `select_joint_map` chooses among **whole** joint maps; a map with any blocked move is rejected even when every action is legal; scores with another `score_id` are refused; with no valid map it uses the fallback proposal, which must also pass the resolver. `accept_joint_map` refuses a `combine()` result with `valid_joint_plan: false`.

## 7. Dataset designs

### S0 — pipeline smoke data

- Source: `synthetic_fixture` traces only. Never train/dev/calibration/test data and never a reported metric.
- No teacher identity is claimed. No selection is needed.

### S1 — teacher-labelled training data (gated)

Only `ember_v1.s1.build_s1_training_set` can produce S1 records (issue #10). It needs a reviewed receipt from this gate, committed to the trusted ref, plus replay, provenance, regeneration and leakage checks. Generation compute must fit the P1–P2 CPU cap ($15).

## 8. Next steps (not done here)

- Review the protocol draft and commit a frozen protocol file (no outcome is evaluated before that commit).
- Generate the `p1-09-selection@1` confirmation manifest and its isolation audit.
- Run the one prospective comparison of T0–T4 (CPU only; T2 dominates cost).
- An arbiter, if wanted, needs its own frozen protocol before the confirmation run.
- T4 v1.2.0 (PR #36, merged) passed its preregistered pilot gate: `T4_READY_FOR_INDEPENDENT_SELECTION_EVALUATION`. The registry now retains it with its admission record and the draft makes it eligible (reviewer decision recorded 2026-11-08). It is **not selected**. Its pilot is not selection evidence. After the protocol freeze, no other T4 identity can enter this protocol version.
