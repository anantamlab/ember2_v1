# EMBER v1 — Multi-teacher, outcome-driven decision-model research direction

**Status:** proposed methodology; **NOT FROZEN**. **No bulk data, no student training and no new GPU spending authorized.**

**Supersedes for future training:** the T1-only student protocol in draft PR #42, T1-only S1 pilot in draft PR #43 and T1-only student harness objective in stacked draft PR #44. Their verified encoder, kernel, data-integrity and negative chunking results may be reused after code review. Existing P1-09 confirmation evidence and its immutable T1 selection receipt are NOT modified or repurposed as authorization for multi-teacher data.

**Goal:** Qwen3.5-2B should learn from the complementary policies of T0 shortest-path, T1 congestion-aware, T2 stochastic fire forecast, T3 dynamic ACO, and T4 v1.2 endangered-person priority, then find better joint decisions than any one policy. Lower both casualties and unresolved evacuees on **new unseen** family-disjoint scenarios; enforce latency, safety and budget limits. Do not claim real-world safety or tune on EMBERBench.

## What the five teachers provide

| Teacher | What it exposes | How used |
|---|---|---|
| T0 | Complete legal actions, shortest-path behavior and weak outcomes | Lower baseline, contrastive/negative evidence, not automatically a positive target |
| T1 | Complete congestion-coordinated actions | Strong practical baseline and common counterfactual continuation policy |
| T2 | Forecast-aware actions; some *actually computed* per-action values | Sparse high-cost reference and future-risk examples; never invent scores for unscored people |
| T3 | Dynamic ACO actions | Route adaptation, congestion/pheromone behavior and contrasts |
| T4 v1.2 | Risk-ordered actions, priority/crowding behavior | Urgent-person and equitable-evacuation examples |

All demonstrations must identify policy/version/config/source hash, policy seed, full pre-decision **public observation**, complete active-person joint action, legality/movement outcome, replay chain and full episode outcomes. The private simulator fire seed remains evaluator metadata only; policy observation MUST NOT expose it. Frozen failed T4 v1.0/v1.1 remain archival, not training candidates.

## Sequential research DAG (issues)

1. **#45 — Paired full trajectories (P1-12).** Independently generate TRAIN-only families and lineages, then each of five teachers runs the **same** scenario with the **same exogenous fire randomness** and a fresh policy instance. Compare closed-loop outcomes and conditions under which each policy is beneficial/harmful. Retain all, even poor episodes, as descriptive evidence, with separate positive-supervision eligibility.
2. **#46 — Shared-state counterfactuals (P1-13).** Do not compare decision actions from different rollout states merely because their step number matches. On a common replayed simulator state, collect all five proposals, choose fully specified valid joint-action interventions, replay each from that exact prefix, and continue under a preregistered *common* policy and *matched* future fire stream. Specify fixed horizon, terminal/censoring handling, repeated seed design and uncertainty. Only claim intervention effects for the whole joint action **under that continuation**, never personal blame for a death.
3. **#47 — SFT + outcome preference data (P1-14).** Separate evidence classes: (a) validated demonstrations (policy identity, full action map, outcomes) and (b) preference pairs from controlled interventions. Use context-aware outcome/quality weights and preserve policy diversity; do not average incompatible action-value scales. Weak T0 examples can be contrastive; teacher agreement is not itself correctness. Start with offline multi-teacher supervised initialization, then bounded preference learning only when causal evaluation and action-map likelihoods support it. Optional subsequent student-visited rollouts are a new phase, not covert online training.
4. **#48 — Independent evaluation (P1-15).** Before training outcomes, freeze family-disjoint student test, metrics, reference comparator and thresholds. Baseline ALL FIVE on the same unseen scenario seeds; define **strongest single deployable reference** without looking at test outcomes, and report a hindsight per-scenario best teacher separately as a nondeployable oracle. Primary: reduce casualties with no worsening unresolved and composite casualties+unresolved, as predeclared; paired family-level uncertainty, multiplicity and worst-case guards. Raw student and deterministic shield/coordinator must be separate arms. No reference teacher may execute at inference for the raw-student claim.
5. **#49 — Conditional improvement (P1-16).** Only after frozen offline evidence passes development gates, collect new STUDENT-visited TRAIN-only states; evaluate alternatives with controlled rollouts. Version every policy and dataset, use independent development families, hard resource caps and no tuning on the final evaluation.

## Gates, independence and resource constraints

- Independent simulator from public rules only. No EMBERBench code, fixtures, hidden or leaderboard feedback.
- Keep P1-09 five-teacher prospective results as immutable **baseline evidence only** (T1 435, T2 428, T3 460, T4 419, T0 608 casualties; P1-09 retained T1 under its original frozen thresholds). Never treat that choice as a selected multi-teacher dataset policy.
- New multi-teacher dataset approval requires NEW reviewed source schema, frozen TRAIN generator/lineage/split manifest, replay+regeneration/provenance gate, leakage audit and a separately authorized pilot. Existing pending single-teacher receipt does not unlock this.
- Before large data runs, measure T2: P1-09 used 7555 CPU seconds on 72 scenarios. Do not assume full-T2-on-every-TRAIN-scenario or extensive branches fit resource limits. Plan bounded stratified coverage and publish exclusions/selection probabilities. Every matched group included for the five-way comparison must contain all five; incomplete groups are invalid, not silently T1-only.
- No scenario from train, counterfactual or model-selection development may be in the prospective student test; avoid near-duplicates under geometry and family hashes. DEV has been inspected and is *development only*. The old sealed index and EMBERBench stay untouched.
- For N>100, full-state chunking must preserve every active person, token bounds **at every step**, an exact joint action map and cross-chunk safety. Existing non-coordinating chunk proxies worsened casualty outcomes; a valid resolver map alone is not proof of good coordination.
- Hardware: A10 inspect passed, but batching/padding parity, N=200 smoke, GPU adapter reload and measured training throughput have not. One GPU container, zero retries, hard timeouts; published projected dollar caps are **not enforced** by the Modal workspace. Approximately $200 total project ceiling, no unapproved spend. Separate written authorization for each GPU phase.

## First executable implementation and limits

The initial code in src/ember_v1/multiteacher.py collects **five matched FULL trajectories only on hand-built synthetic fixtures**, checks each decision against the common proposal contract, verifies full replay and matched fire on common steps, and produces identical-public-state disagreement probes. It refuses any real TRAIN/DEV/heldout scope, and every output is explicitly **training_eligible: false**. This is a correctness foundation, not a dataset and not a scientific comparison.

Schema `matched-multi-teacher@0.2.0` adds: a deterministic `content_hash` separate from host runtime cost; collector-measured CPU per decision; recorded teacher failures that make a group incomplete (never a silent four-teacher group); public congestion diagnostics; simulator, collector and teacher code fingerprints with registry pin status; `common_state_steps`; `verify_fixture` (regenerates every observation, transition, diagnostic and outcome without running teachers) and `regenerate` (re-runs all five teachers). A test on a hand-built divergent fixture shows that a fresh teacher at its own trajectory state reproduces its recorded proposal, so teachers are Markov in the public observation and a same-state probe is valid. The TRAIN collection protocol draft is `docs/P1_12_TRAIN_COLLECTION_PROTOCOL.md` (not frozen).

**Next implementation gate:** source-authenticated TRAIN scenario collection with a new protocol freeze, write-once data, exact five-policy completion, live per-decision and total CPU supervision, locked cross-split audit, independent source verification and one small prospective CPU pilot. No bulk collection, multi-teacher labels, student training or GPU deployment until those are reviewed and explicitly approved.

## Interpretation

Comparing two full policies on the same scenario answers which complete *policy process* achieved an outcome under matched initial/exogenous conditions. A controlled first-action intervention followed by the same continuation answers a narrower **joint-action counterfactual** question under that continuation. Neither is a general proof that one individual's choice killed another person. Record uncertainty and censored outcomes; do not turn retrospective temporal association into causal ground truth.
