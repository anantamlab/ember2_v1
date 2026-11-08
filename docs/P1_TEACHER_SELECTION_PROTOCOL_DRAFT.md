# P1-09 — Teacher selection protocol (review draft v0.1)

**Status: DRAFT / NOT FROZEN / NO EXPERIMENT AUTHORIZATION.** Executable form: `ember_v1.teacher.selection.draft_protocol()` (`frozen: false`), validated by `validate_protocol` (see docs/P1_TEACHER_SELECTION_PREP.md sections 4–6). Freezing means committing a frozen protocol file to `main` after review.

Issue #25; implementation home: PR #35. Prepared after the completed exploratory DEV reports P1-03 through P1-06 and the blocked T4 v1.0.0/v1.1.0 pilots. This is a prospective protocol proposal, not an outcome report, selection receipt, or permission to create S1 labels.

## 1. Purpose and candidates

Retain five independently identifiable policy families:
- T0: fire-aware BFS shortest path
- T1: congestion-aware cooperative routing
- T2: stochastic fire-rollout policy with frozen T1 continuation
- T3: dynamic Ant Colony Optimization
- T4: endangered-person risk-priority routing

**Current eligibility (decided before any P1-09 confirmation evidence):**
- Eligible to be considered for prospective selection: frozen, previously validated T0, T1, T2, T3 v1.0.0 identities and T4 v1.2.0 (`sha256:79bd3322…2504`), subject to fresh preflight and compute gates.
- T4 v1.2.0 entered under the T4 entry rule below, before the protocol freeze: its preregistered independent pilot `t4-pilot@3` (freeze commit `b7ff040`, report `data/baselines/p1-08-t4-v12-pilot.json`, `sha256:b64b5589…2655`) returned `T4_READY_FOR_INDEPENDENT_SELECTION_EVALUATION` (PR #36). Admission is not selection, and the pilot is not selection evidence. The frozen protocol carries this admission record in `t4_entry_rule.admitted`.
- Ineligible: T4 v1.0.0 (mechanism G5 failed) and T4 v1.1.0 (v2 planner-health G4 failed on A1/A2). These are retained as failed reference-policy evidence but must not propose labels for the selection comparison, arbiter, or S1. The frozen protocol lists them in `t4_entry_rule.blocked_identities`.
- Tie-break simplicity order (after lower CPU), fixed now: T0, T1, T3, T4, T2.
- T4 is not silently discarded: every inventory, report, and selection receipt must include its ineligibility reason and its blocked experiment hashes.
- A subsequent T4 version can enter only after a new, preregistered, independent synthetic mechanism/compute pilot passes its full gate AND before this selection protocol is frozen. If not ready at the freeze deadline, T4 remains excluded throughout this version. Do not unlock it after seeing selection outcomes. A later validated T4 requires a new protocol, new confirmation data, and separate decision.

Issue #25's original all-five design remains the long-term goal. A blocked policy cannot be declared scientifically comparable merely to fill the fifth slot.

## 2. Evidence categories and data firewall

**Historical development evidence:** prior p1-02-v1 DEV comparisons (48 scenarios, 16 families) and historical T4 pilots may inform the written hypotheses and cost estimates but must be labeled exploratory for the present selection. They must never be represented as prospective independent confirmation.

**Prospective selection-confirmation set:** before any teacher is evaluated on it, freeze generator revision, scenario rules, inclusion criteria, family count, lineage count, seed namespace, split assignments, wall-clock/CPU caps, primary analyses, and the exact evaluator code. Initial design target: 24 new topology families times 3 independently generated lineages = 72 scenarios. If generator feasibility requires a different size, amend the protocol and freeze BEFORE generating any scored selection outcomes; never keep searching for a favorable split.

Use a new fixed namespace (e.g. p1-09-selection@1) and make a content/geometry/lineage/family isolation audit against ALL p1-02-v1 train, DEV, calibration, procedural locked-test manifests, T4 synthetic pilot families, and any previously used exploratory evaluation families. Reuse the independent scenario generator's public physics; do not sample from EMBERBench. Record attempted and failed generations, commitments, and exclusions. Do not tune eligibility, generator, policy or selection rule after scoring begins.

The set is **prospectively held out from policy development**, not cryptographically blind if its construction is public. Only one confirmatory run per frozen identity/protocol; no automatic reruns after a scientific negative result. The existing procedural locked test, any private blind set, calibration split, and EMBERBench remain excluded from teacher selection.

## 3. Observations, identity, replay, and proposal validity

Each proposal contains the exact registered policy name/version/config hash, source-code fingerprint, teacher seed, public observation digest, all active IDs with their legal immediate actions, optional actually computed action values, CPU and wall runtime, and a canonical proposal hash.

Only a teacher's validated public observation can influence decisions. Never provide hidden simulator seed, future fire, private trapped labels, scenario lineage or selection labels to a policy.

T2's five-action Monte Carlo values exist only for people genuinely scored; unscored individuals have missing values. T0, T1, and T3 are action-only. Do not fabricate value vectors, treat uncertain T2 scores as calibrated across policies, or average incompatible scores.

Validate full joint maps through the exact public resolver. An invalid, incomplete, blocked, over-budget, unregistered, or wrong-observation proposal is ineligible for selection and triggers a recorded failure/fallback. A mixed per-person proposal that produces a blocked map is not a viable selected action. Never mark a map safe just because each person's action is individually legal.

## 4. Candidate strategies

**S0 — frozen standalone:** Each eligible reference agent is run independently from the same initial scenario and evaluator fire seed. Keep their internal identities separate. T1 is the operational comparison benchmark based on historical development evidence; this is not a newly confirmed selection claim.

**S1 — optional joint-map arbiter:** Only test if a separately frozen implementation and compute gate are available before confirmation. It must select among whole, resolver-valid joint proposal maps under a predeclared observation-consistent score and independent teacher-side sampling. It must not inspect actual simulator future states or holdout outcomes. If mixing individual actions is proposed, require explicit joint validation and counterfactual evaluation of the resulting complete map, with a safe fixed fallback; no naive mixture of directions or value scores.

Compare arbiter against the strongest eligible standalone under **comparable total planning CPU**, counting every invoked candidate teacher plus the arbiter. If the arbiter fails feasibility, lacks a valid frozen protocol, or exceeds compute, skip it and select among standalone policies only.

## 5. Resource gates

Before confirmation, preregister one-worker CPU pilot bounds, a projected upper cost for all policies including T2 and any arbiter, per-decision resource caps, log limits, and a hard experiment stop policy. The total monetary allowance remains within the existing P1-P2 $15 cap and overall $200 project budget; local CPU preferred, no Modal or GPU without new owner approval.

Missing performance/identity metadata or a resource-gate failure blocks that candidate or the entire protocol according to predeclared dependency rules. Do not remove slow candidates on the basis of their confirmation outcomes.

## 6. Primary and safety outcomes

**Primary:** paired mean casualties per scenario relative to frozen T1, family-cluster bootstrap 95% confidence interval, and family-level exact two-sided sign test. Establish superiority only when the 95% interval excludes zero in the favorable direction AND the family sign test has p < 0.05. All 24 families form the resampling unit. Document family-level ties and the multiplicity of comparisons; non-primary inference is exploratory without preregistered adjustment. (Superseded for the candidate-vs-T1 tests by protocol 1.2.0: Holm-adjusted p < 0.05 over all eligible candidates; see `docs/P1_SELECTION_CONFIRMATION_SET.md` section 4.)

**Safety guardrails, set before confirmation:** also report casualties plus unresolved occupants at the step limit, evacuated fraction, people initially at high risk, and catastrophic tail episodes. A purported casualty gain that merely shifts comparable numbers into unresolved must not be described as a successful evacuation gain. Require no observed net deterioration in casualties-plus-unresolved totals and evacuated totals versus T1 before considering a replacement. This total-based guard is descriptive and conservative, not a statistical noninferiority proof; any formal noninferiority margin needs an additional preregistered protocol version.

**Secondary:** episodes completed, time to evacuation, blocked moves and collision counts, reversals/waiting, priority fairness by initial public risk strata (when an externally frozen risk assessor is used), latency p50/p95/p99, total CPU, decision failures, and teacher-specific fallbacks. Show every material regression, not only averages.

For candidates meeting all safety requirements, choose a replacement of T1 only if the primary superiority rule passes and its measured total compute remains within the protocol's predefined feasible envelope. Otherwise keep T1 as the provisional standalone training-source candidate, provided its provenance/quality gates pass. If multiple candidates qualify, prefer lower CPU cost and simpler verified behavior under frozen tie rules; do not optimize this rule after looking at the data.

**No teacher is selected by writing this document.** A final selection receipt must identify exactly one frozen policy or one independently validated arbiter, include identity hashes and completed confirmatory evidence, and be reviewed separately.

## 7. Stop conditions and fail-closed behavior

Stop without selecting when: freeze commit missing; scenario isolation invalid; unexpected evaluation errors; policy hash/source drift; any eligible policy reads forbidden private information; incomplete matched episodes; planned resource/wall cap exceeded; unverified proposal provenance; incomplete results; scientific rules changed after results; or a blocked T4 is included without a separate approved protocol revision.

Do not infer causality from a teacher disagreement count or a single counterfactual action score. Historical DEV is not prospective confirmation.

## 8. Distillation and provenance gate (separate work)

Keep bulk S1 generation and student training BLOCKED until:
1. protocol and implementation are frozen and source-revision pinned before new selection outcomes;
2. one complete prospective matched-family comparison passes its primary/safety/compute rules or explicitly preserves T1 under the frozen no-improvement rule;
3. a reviewed final selection receipt identifies one allowed teacher/arbiter and its exact code/config/seed hashes;
4. issue #10 verifies full-episode replay, observation-only provenance, manifest isolation, teacher-code regeneration and complete action/label integrity at the **mandatory** S1 entrypoint (no opt-out in training);
5. train-only generated labels remain disjoint from selection-confirmation, DEV, calibration, locked and blind families.

Hash binding and regeneration are acceptable for internal trusted code/data generation; hashes are not digital signatures. Treat externally supplied or adversarial records as untrusted unless an independent authenticated source mechanism is specified.

## 9. Required implementation changes in PRs #34 and #35

- Keep candidate inventory T0 through T4, but add independently frozen eligibility records (T4 blocked with reason). Do not require DEV evidence for a formally excluded candidate.
- Separate old descriptive DEV inventory from new after-protocol confirmation evidence. Never satisfy an after-freeze comparison requirement with old reports.
- Make the protocol and selection gate reject missing/ambiguous excluded-candidate reasons, unmatched family manifests, unfrozen identities or unverified results.
- Align issue #10 bulk-S1 dependencies with explicit blocked-T4 eligibility handling. Do not silently bypass a dependency.
- Consolidate the duplicate VALUE_TEACHERS registry in a shared utility.
- Require a resolver-valid final joint map; do not pass an invalid combined map to the simulator as selected teacher output.
- Run focused and full CPU tests and all GitHub CI jobs after rebasing onto main.

**Release criterion:** review and freeze this protocol and the matching executable gate before the first prospective selection outcome is evaluated. Until then: NO_SELECTION; BULK_S1_BLOCKED.
