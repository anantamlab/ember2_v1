# P1 five-teacher selection: preparation (issue #25)

**Status:** preparation only. **No teacher is selected.** No new DEV run, no arbiter, no S1 data. Code: `src/ember_v1/teacher/proposals.py`, `src/ember_v1/teacher/selection.py`; tests: `tests/test_teacher_selection.py`. Gate today: **`NO_SELECTION`** (no frozen protocol; T4 has no DEV evidence).

## 1. Common proposal record (`teacher-proposal@1.0.0`)

`proposals.proposal(decision)` turns any `PolicyDecision` into one record: teacher name, policy, version and config hash; observation digest and step; the **full** ID → action map; per-action values only for T2 (the only teacher that computes them); policy seed; CPU and wall time; `proposal_hash` (all fields except timing).

| Function | Purpose | Fails closed on |
|---|---|---|
| `validate_proposal` | check one proposal against its observation | missing/extra fields, hash mismatch, other observation, partial or unknown IDs, invalid action, values from an action-only teacher, values over non-allowed actions |
| `joint_check` | run a joint action map through the exact resolver | – (reports blocked people; `valid_joint_plan` is false if any move is blocked) |
| `combine` | build one joint map from per-person teacher choices | a missing person, an invalid source proposal. Records the source teacher of every action and the source proposal hashes, so an arbiter decision is never attributed to one teacher |
| `disagreement` | unanimous fraction and pairwise agreement | duplicate teachers. Agreement is descriptive and is **not** a quality measure |

There is no weighted average of unlike scores anywhere.

## 2. Evidence inventory (committed DEV reports)

`selection.evidence_inventory()` reads P1-03 … P1-06. All reports use the same 48 DEV scenarios (16 families, plan `p1-02-v1`). Where a teacher appears in several reports, all final transition hashes agree (0 conflicts: T0 and T1 each appear in 3 reports).

| Teacher | Reports | Casualties | Evacuated | Unresolved | Policy CPU (s) | Committed verdict |
|---|---|---|---|---|---|---|
| T0 | P1-03, P1-04, P1-06 | 507 | 1 210 | 17 | not recorded in P1-03 | reference |
| T1 | P1-04, P1-05, P1-06 | 345 | 1 314 | 75 | 19.0 | better than T0 on casualties (P1-04) |
| T2 | P1-05 | 333 | 1 318 | 83 | 3 384 | no evidence of a casualty difference vs T1; not selected (P1-05) |
| T3 | P1-06 | 403 | 1 307 | 24 | 69.0 | fewer casualties than T0; no evidence vs T1 (P1-06) |
| T4 | – | – | – | – | – | no DEV evidence: v1.0.0 `BLOCKED_BY_MECHANISM_CHECK`, v1.1.0 `T4_V2_BLOCKED` (PR #32/#33) |

These totals come from **separate** preregistered comparisons. They are not one five-way paired comparison, and they do not select a teacher. DEV was used for every comparison above; it is not a fresh confirmation set.

## 3. Dataset designs

### S0 — pipeline smoke data

- Source: `synthetic_fixture` traces only (hand-written or generated test fixtures).
- Use: tests of the ingest, encoding and training pipeline. Never train/dev/calibration/test data and never a reported metric (`sources.training_records` excludes fixtures unless `allow_fixtures=True`; issue #10 makes `require_provenance=True` refuse fixtures).
- No teacher identity is claimed. No selection is needed.

### S1 — teacher-labelled training data (gated)

S1 is generated only after **all** of these hold:

1. a frozen selection protocol (section 4) is committed before the evidence that it uses;
2. all five candidates have DEV evidence under that protocol (or the protocol states how a blocked candidate is handled, before any result);
3. the protocol's decision rule gives one training policy (one teacher or a validated arbiter), and that policy is frozen by identity hash;
4. the #10 provenance gate (`provenance.bulk_s1_gate`) gives `BULK_S1_AUTHORIZED`;
5. the generation compute fits the existing P1–P2 CPU cap ($15).

Each S1 record keeps the source teacher (or the arbiter and each source proposal hash), the selection protocol hash, replay evidence and provenance. Values are kept only when genuinely computed (T2).

## 4. Required fields of a selection protocol

`selection.freeze_protocol` refuses a protocol unless it has all of: `protocol_id`, `candidates` (exactly T0–T4, separately), `data`, `primary_endpoint`, `secondary_endpoints`, `comparisons`, `decision_rule`, `simplicity_rule` (prefer the simpler policy without evidence of a difference), `arbiter` (`null`, or equal compute to the best single teacher + no score averaging + joint resolver check), `compute_budget` (a positive USD cap ≤ $15), `stop_rules` (non-empty), `excluded_data` (must include `locked_test`, `blind_test`, `emberbench`), `frozen: true`. It adds `protocol_hash`.

## 5. Selection gate

`selection.selection_gate(protocol, inventory, protocol_commit=..., evidence_after_commit=...)` returns `NO_SELECTION` with every reason, or `READY_FOR_PREREGISTERED_COMPARISON`. It **never** returns a selected teacher (`selected_teacher` is always `null`). Reasons include: no or tampered protocol, no protocol commit, evidence older than the protocol, a candidate without evidence, more than one identity per candidate, failed episodes, cross-report conflicts, different DEV scenarios.

Today the reasons are: no frozen selection protocol; T4 has no DEV evidence.

## 6. Next steps (not done here)

- Write and commit the selection protocol (section 4) with a decision for blocked T4.
- Then run one paired five-way DEV comparison under it (CPU only; T2 dominates cost at about 3 400 CPU-s).
- An arbiter needs its own preregistered value function with observation-consistent sampled futures, at equal compute.
