# P1-14 multi-teacher outcome-aware dataset — protocol draft

**Status:** `DRAFT_NOT_FROZEN` (`data/protocols/p1-14-multiteacher-dataset.json`, version `0.3.0-draft`).
**Authorizes nothing.** `multiteacher_dataset.export` always raises `DATASET_GENERATION_BLOCKED`. Records can be
built only from `synthetic_fixture` reports. Issue #47. Depends on #45 (P1-12) and #46 (P1-13).

## 1. Evidence classes

| Class | Name | Built from | Use | Weight |
|---|---|---|---|---|
| A | successful demonstration | trajectory of a complete matched group whose casualties+unresolved AND casualties equal the group minimum (ties allowed) | SFT positive | scenario budget, equal split |
| B | unsuccessful demonstration | every other complete trajectory | contrastive only: analysis, and the states where P1-13 probes look for better actions | reported, not a positive target |
| C | counterfactual preference **candidate** | verified P1-13 contrast: `a_better`/`b_better`, uncensored, ≥3 replicates and ≥2 strict improvements (stochastic fire), or 1/1 under deterministic fire | preference pair only after **approval** (§2a) | 0 now (`preference_candidate_not_approved`); scenario budget over approved records later |
| D | inconclusive intervention | every other verified contrast (censored, mixed sign, no difference, below threshold) | excluded from every loss; kept and counted | 0 |
| E | student-visited state | reserved for P1-16 (#49) | refused by the validator | — |

Every record has a weight > 0 or weight 0 with its class's `exclusion_reason` (A non-selected:
`competing_class_A_joint_map_at_same_state`; B: `contrast_only_not_a_positive_target`; C unapproved:
`preference_candidate_not_approved`; D: `inconclusive_comparison`).

Rules that follow from the classes:

- A class comes from **matched outcome evidence**, never from the teacher's name. T0 can be class A in a
  scenario where shortest-path is optimal; T1 can be class B where it loses people.
- A trajectory-level label is weak per-decision evidence: a best trajectory can still hold poor moves, and a
  worse trajectory can hold good moves. Class C is the only per-state preference evidence.
- A failed teacher makes the group incomplete. Then no A/B record is built from that group.
- No action probability, confidence, value estimate or causal label is ever created
  (`forbidden_fields`). `teacher_values` holds only values T2 really computed, copied from its proposal.
- B records are **not** unlikelihood targets: a B action at a B state can be correct. B states feed the
  P1-13 probe queue.

## 2. Record schema `mt-dataset-record@0.3.0`

Common: `record_version, class, use, scope, scenario_hash, family_id, lineage_id, source, provenance,
observation, observation_digest, selection, weight, exclusion_reason, record_hash`.

| Class | Extra fields |
|---|---|
| A/B | `teachers` (all teachers that proposed this exact joint action at this exact state), `actions` (complete joint map), `proposal_hashes`, `teacher_values` (T2 only or null), `step`, `evidence` (group best, per-teacher trajectory outcomes incl. blocked moves, contested targets, peak contention, steps, termination), `replay` (per teacher: complete replay, steps verified, final transition hash), `step_diagnostics` (blocked moves, contention, exit use of this map at this state) |
| C | `chosen`, `rejected` (each: name, complete joint action, proposer set or composite sources, replicate-0 outcome), `evidence` (report hash, verdict, replicates, strict improvements, horizon, censoring, continuation identity, per-replicate deltas, causality limit), `approval` (§2a) |
| D | `alternatives` (both sides), `evidence` (as C, plus `below_threshold`) |

`provenance`: A/B — report content hash, simulator and collector module fingerprints, scenario source, and per
proposing teacher the exact identity, `code_fingerprint` and registry status; C/D — counterfactual report hash
and version, prefix hash and source, continuation identity, fire model. `selection`: the preregistered rule,
the decision and its reason (§4).

`validate_dataset` checks every field set, hash, public-observation leakage, completeness of every joint
map, the outcome rule for A/B, exact teacher identity + fingerprint + complete replay for every A/B proposer,
the thresholds and approval state for C, zero weight + reason for B/D/unapproved C/non-selected A, refusal of
E, duplicates (same class, state and joint action, also across family IDs), **one weighted whole map per
state** (contradictory demonstrations), **one identity and fingerprint per teacher** (source drift), and — once
≥20 scenarios exist — the family cap (§3). `check_targets_against_report` proves every A/B map is one teacher's
whole proposal at that state (a per-person mixture fails).

### 2a. Preference candidate ≠ approved preference

`approval.checks` (all required): `shared_initial_state`, `controlled_continuation` (every branch has the
report's continuation identity), `matched_exogenous_fire`, `uncensored`, `sufficient_replicates`,
`sufficient_strict_improvements`, `no_replicate_worse` (casualties+unresolved and casualties, every
replicate), `no_unresolved_increase` (every replicate). `status = approved` only when every check holds AND
the protocol is `FROZEN` with `preference_approval.approval_enabled = true`. Under this draft every C record is
`candidate_not_approved`, weight 0. A verdict compares two complete joint maps at one state under one
continuation; it never assigns blame to one person's action. `student_objective.admit_preference` refuses
anything else (class D, candidates, censored, shielded/corrected maps).

## 3. Sampling and weighting (auditable)

| Rule | Value | Reason |
|---|---|---|
| Unit | one budget of 1.0 per scenario for class A (and for approved C) | long episodes and large N do not dominate |
| Inside a scenario, class A | equal split over the tied-best **trajectories**, then over each trajectory's selected records | a short tied-best trajectory (a minority strategy) is not drowned by a long one |
| Uninformative scenario (all five teachers equal on primary and casualties) | budget × 0.25 | easy cases do not dominate |
| Teacher share of class A weight | > 0.5 → **review flag, never reweighted** | a cap would move weight from a strictly better teacher to a worse one |
| Family share of A+C weight | ≤ max(0.05, 2 / families) after ≥ 20 scenarios | a fixed 0.05 fails at 20 equal families by rounding; 2× uniform tolerates normal variation |
| Strata | class-A share per (active-people band, fire model) reported | rare large-N / stochastic scenarios stay visible |
| Epoch sampling | rows drawn ∝ weight to a fixed token budget, seed published | cost is bounded before training |

`audit` prints class counts, weight per class, teacher shares, family shares, strata shares, states with
competing A maps, the selection criterion that decided each, T2-value record count, exclusion counts and
candidate vs approved preferences.

Weighting audit (run 8):

| Concern | Finding | Change |
|---|---|---|
| Scenario budget | sound; equalizes hard and easy scenarios | kept |
| Teacher cap 0.5 | as a hard failure it either blocks a valid dataset or forces reweighting toward worse teachers | report + review only |
| Family cap 0.05 at ≥ 20 families | fails on any imbalance at exactly 20 families | max(0.05, 2/F) |
| Equally successful teachers | v0.1 split per record, so the longer trajectory took more weight; at shared states both maps were targets (mixture) | per-trajectory split; one target per state (§4) |
| Rare difficult scenarios | budget per scenario already protects them; no extra upweight (it would be an invented importance) | strata report |
| Sparse, expensive T2 | a T2 timeout makes the whole group incomplete, and timeouts concentrate in large-N scenarios → silent difficulty bias | P1-12 pilot measures T2 cost per stratum; incomplete-group rate per stratum is a pilot gate |
| Minority strategies | a teacher that wins only rare scenarios keeps its full scenario budget there | no cap removes it |

## 4. Coherent joint demonstrations (decided for Stage A; review before freeze)

Problem: when two class-A teachers propose different joint maps at the same state, per-person cross-entropy on
both trains a mixture of marginals; its argmax can be a joint map no teacher proposed, with blocked moves.

| | A. Whole-joint selection | B. Several joint alternatives | C. Shared strategy variable |
|---|---|---|---|
| Target is one complete coherent map | yes | yes (each alternative) | yes, given the strategy |
| Unintended per-person mixture | none (one weighted target per state; validator) | per-person CE: yes. Set likelihood −log Σ_m π(A_m): no at the optimum, but early gradients split, and the mode is chosen by ease of fit, not evidence | none if the strategy is a real conditioning input |
| Model change | none | none for set likelihood (scores whole maps) | extra strategy question/row; persons conditioned on it |
| Data complexity | low: rule over existing fields | low | needs a public strategy descriptor extractable from every map (e.g. exit allocation) |
| Inference latency | unchanged | unchanged | + one row (strategy) before person rows |
| N = 200 | unchanged; chunk rows teacher-forced on the selected map | each alternative needs its own reserved rows | strategy row also coordinates chunks |
| Beyond teachers | learns which teacher's whole plan wins in which condition; Stage B preferences can rank composite maps | same | can combine strategies with person moves; most expressive |

**Decision: A for Stage A SFT** (`select_joint_targets`, rule `p1-14-whole-joint-target@0.2.0`, protocol
`joint_target_rule.mode = "select_one"`): at each state exactly one complete class-A map is the target, ordered
by (1) approved same-state class C wins, (2) fewest blocked moves of the map at this state, (3) fewest MEAN
trajectory blocked moves over the proposing teachers, (4) fewest MEAN trajectory steps over the proposing
teachers, (5) action-map digest. No teacher name is a criterion or a model input; teacher identity stays in
provenance. B's set likelihood is implemented as an ablation (`student_objective.set_nll`). C is the
stage-A2 option if DEV shows mode averaging across *similar* states (exact contradictions are removed by A; near
duplicates are not), reported as the blocked-move rate of the student's raw assembled map.

### 4a. Conflicting valid maps (run 9)

When two successful teachers propose different valid maps at one state:

| Requirement | Implementation |
|---|---|
| Keep both, with provenance | every class-A map is a record with its proposing teachers, proposal hashes, replay and trajectory outcomes; the map that is not selected keeps weight 0 and reason `competing_class_A_joint_map_at_same_state` |
| Keep each map whole; no per-person mixing | `check_targets_against_report` refuses a map that is not one teacher's whole proposal; `coherence_problems` refuses two weighted maps at one state under `select_one` |
| Different whole maps as demonstrations, by a preregistered rule | protocol mode `sample_whole_maps`: every distinct class-A map at a state is selected; the state's mass is split EQUALLY over the distinct maps; the validator refuses unequal weights and a mode that differs from the protocol. **Preregistered, not active for Stage A** (see below) |
| No teacher-identity oracle | `model_example(record)` returns only `observation`, `actions`, `weight`; tests check that no teacher name or provenance field reaches it and that renaming teachers changes no selection or weight |
| Successful and unsuccessful policies separate | class B stays contrast-only with weight 0 |

Why `sample_whole_maps` is not active: the audited Clef head scores the people of one row conditionally
independently. Weighted SFT on two different maps at one state then learns per-person marginals, and the most
likely joint map can be a per-person mixture that no teacher proposed
(`test_sampled_whole_maps_need_person_level_conditioning_or_decode_to_a_mixture`). With one person per
reservation chunk, the SFT optimum is the exact mixture of whole maps and its mode is a teacher's map. Enabling
the mode therefore needs (1) person-level reservation conditioning (or another whole-map mechanism), (2) a
measured mixture-decode rate on held-out conflicting states, and (3) a protocol change in a reviewed commit
before any TRAIN data exists.

### 4b. Class A weighting audit (rare T2/T3/T4 strategies)

| Concern | v0.2 behaviour | v0.3 change |
|---|---|---|
| Selection criteria 3–4 took the MINIMUM over proposing teachers | a map that four teachers proposed had four draws; one lucky low-congestion trajectory beat a rare map that was better on average | MEAN over proposing teachers (exact fractions); the count of proposers is never a criterion (test: four proposers vs one, equal mean → name-free digest decides) |
| Teacher CPU cost | not a criterion | unchanged; stated in the protocol (`count_and_cost_neutrality`) |
| Per-teacher budget | every class-A trajectory gets an equal share; a teacher whose map lost at a shared state keeps its share on its own other selected records | unchanged (verified); a rare teacher's trajectory total equals a common one's |
| Merged record at a shared state | receives the share of every proposing teacher (visitation weighting) | unchanged under `select_one` (teacher totals stay equal); under `sample_whole_maps` the state mass is split equally per distinct map |
| Rare-strategy loss visibility | not reported | `audit.unselected_A_maps_by_teacher` (review only) |

## 5. Gates before real generation

1. P1-12 TRAIN collection protocol frozen and authorized; pilot passed (`docs/P1_12_TRAIN_COLLECTION_PROTOCOL.md`).
2. P1-13 TRAIN counterfactual budget frozen: states per scenario, replicates, horizon (to termination or ≥ the
   scenario's remaining steps), continuation(s).
3. This protocol frozen in its own commit with its hash; owner authorization in writing.
4. Leakage audit: no family, lineage or geometry core of DEV, calibration, selection, P1-15 test or the sealed
   index in any record.
