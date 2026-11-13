# P1-14 multi-teacher student training method, chunk coordination and harness plan

**Status:** `DRAFT_NOT_FROZEN`. Supersedes the T1-only method of draft PR #42 for future training. #42/#43
stay as frozen-out history (their fail-closed guards stay). **No GPU, no Modal, no training is authorized.**

Model: pinned Qwen3.5-2B + Clef joint-schema head (pins, LoRA r16, bf16, `max_len` 16 384, A10) as in
`docs/P1_STUDENT_TRAINING_PROTOCOL.md` §3 of #42 (identities now copied into
`data/protocols/p1-14-student-train.json`; no #42 permission carries over). Batch size 1 until DeltaNet padding
parity is measured.
The student never calls a reference teacher at inference.

## 1. The student's probability model (audited in run 8)

Audit source: `unsloth/_vendor/clef/joint_schema_model.py` from the pinned wheel `unsloth-2026.10.11`
(re-downloaded; wheel sha256 `baff520e…0992e`, file sha256 `0e304cf7…dc3a3` = `p0b.AUDITED_CLEF_SHA256`).
Reference implementation and tests: `src/ember_v1/student_objective.py`, `tests/test_student_objective.py`.

Facts from the source:

1. **Normalization.** `JointSchemaHead.forward` returns, per question, raw logits, one per option of that
   question (`option_ids`, sorted). They are not normalized; upstream `systemone` applies `softmax` over the
   question's options. The trainer must use `log_softmax` over exactly those options. Our questions always have
   the 5 contract options, so the harness slice `logits[:, :5]` is exact; `_log_softmax` refuses any other count.
2. **Support.** The distribution is over the 5 **contract** actions for every active person. A move into a
   wall, the boundary or fire is a legal contract action that the simulator resolves as blocked. It is not a
   masked action; "allowed" means contract-allowed, not useful.
3. **Conditioning.** Fields cross-attend only to the token memory; no answer is an input. Given the row's token
   sequence x, the row's answers are conditionally independent. But each person's logits depend on all of x:
   the state, any reservation block, the other questions of the row and their order.

Derivation. For rows r_1..r_m (chunks C_1..C_m) and joint map A:

  P(A | s) = Π_j P(A_Cj | x_j) = Π_j Π_{i∈Cj} softmax(z_i(x_j))[a_i],   x_j = x_j(s, C_j, A_C<j)

The first equality is the chain rule over chunks (exact whatever x_j contains); the second is the head's
conditional independence inside a row. Without reservations x_j = x_j(s, C_j). With reservations (§4) x_j
contains the reserved cells of A_C<j, so x_j depends on the scored map itself. Test: the probabilities of all
5^N maps sum to 1 in both cases (enumeration, N = 3, two chunks).

Consequences:

- `joint_logprob` is the exact log P(A | s) **under the inference conditioning**, computed with one forward
  per row of THAT map (teacher forcing on the scored map, never on another map).
- Same-action cancellation in a pairwise loss is exact only when the row input is identical for both maps:
  always without reservations; with reservations, only for rows up to and including the first chunk where the
  two maps differ. A same-action person in a later row does NOT cancel (`cancelling_persons`; the test shows the
  "differing persons only" margin is wrong under reservations).
- π is the **raw** policy. Deployment may apply a deterministic shield/corrector g. The deployed distribution
  is P_g(B | s) = Σ_{A: g(A)=B} π(A | s). Nobody computes it at N ≥ 6; `induced_shield_probability` enumerates it
  for tiny N and otherwise refuses. Shielded outputs are never called exact probabilities and never enter a
  preference loss as raw actions.
- A factorized row cannot represent "either map 1 or map 2" without composites. Exact same-state conflicts are
  removed by whole-joint target selection (dataset §4); a strategy variable (option C) is the escalation.

## 2. Stage comparison

| Stage / objective | Data | Probability requirement | Verdict |
|---|---|---|---|
| A: weighted SFT (cross-entropy) | selected class-A whole maps, weights of P1-14 §3 | exact chain-rule likelihood of one coherent map | **Use first.** |
| A': SFT on all teachers equally | A+B | — | **Reject**: imitates losing trajectories. |
| A'': per-person CE on several A maps at one state | competing A | trains a mixture of marginals | **Reject** (dataset §4); set likelihood `set_nll` is the ablation |
| B: DPO (frozen Stage-A reference) | **approved** class C pairs only | exact joint log-probs under the same row conditioning; with reservations one forward per row per map (no cross-map cancellation after the first differing chunk) | **Use when ≥ preregistered number of approved pairs exists.** |
| B: ORPO (no reference model) | class A as SFT + class C as odds-ratio term | odds of the whole joint map: defined, but for many people the joint probability is tiny and the odds term saturates; per-person-normalized variant needed | Second choice; needs a numerical check first. |
| B: IPO | class C | same as DPO; squared loss resists over-fitting when pairs are near-deterministic | Variant to compare with DPO on DEV. |
| B: advantage-weighted regression | needs a scalar advantage per action | only ordinal, small-count, sometimes censored deltas exist; a scalar would be invented | **Reject** (no fabricated values). |
| B: KTO (unpaired good/bad) | A as good, B as bad | B actions are not known to be bad (§1 of P1-14) | **Reject**. |
| C: student-visited counterfactuals | class E (P1-16) | as Stage B | **Conditional**; only after a frozen offline student passes DEV (#49). |

DPO/IPO detail (revised run 8). h = [log π(w) − log π_ref(w)] − [log π(l) − log π_ref(l)], all four exact
joint log-probabilities from `joint_logprob` with identical chunking and the reservation function used at
inference. DPO: −log σ(β h), β preregistered (start 0.1). IPO: (h − 1/(2τ))², τ preregistered; compare on DEV.
Mixed with the Stage-A SFT loss (λ, start 0.5). Reference log-probs are precomputed once with the frozen
Stage-A adapter **row by row under each map's own reservations**; no second model in GPU memory. Cost per pair:
one forward per row of w and one per row of l (rows before the first differing chunk can be shared). Only
`admit_preference`-approved class C pairs enter; class D, unapproved candidates, censored contrasts and
shielded/corrected maps are refused. The v0.1 text ("loss on persons whose actions differ; others cancel
exactly") was valid only without reservations and is withdrawn.

What each teacher contributes (expected, to be measured — not assumed):

| Teacher | Expected useful signal | Evidence that decides |
|---|---|---|
| T0 | simple routing where shortest path is optimal; failure states for probes | class A only when tied best; B states → probes |
| T1 | congestion coordination | A/C where coordination changes outcomes |
| T2 | forecast-aware moves; computed per-action values | A/C; values kept as metadata only (no value loss in v1) |
| T3 | dynamic route adaptation | A/C |
| T4 v1.2 | endangered-person priority | A/C; P1-09 lowest casualties (419) |

## 3. Failure modes and controls

| Failure mode | Control |
|---|---|
| Imitating the average of incompatible teachers | class A by matched outcomes; one whole-map target per state (P1-14 §4); blocked-move metric of raw assembled maps |
| One teacher dominates | teacher share > 0.5 is a review flag (no reweighting toward worse teachers) |
| Easy scenarios dominate | uninformative factor 0.25; scenario budget |
| Preference over-fit / spurious pairs | thresholds (≥3 replicates, ≥2 strict wins), IPO variant, SFT mixing |
| Continuation dependence of C pairs | continuation-sensitivity sample (T1 vs T4) before freeze |
| Leakage from test or benchmark | family/core firewall; no P1-09, sealed suites or EMBERBench in training or tuning |
| Hidden teacher at inference | inference path imports no `ember_v1.teacher` module (test); raw and shielded arms separate |
| Token overflow | fail closed, never cut; compact encoding; chunk k from §4 |

## 4. N = 200 and cross-chunk coordination

Facts (#44 study `docs/P1_CHUNK_JOINT_MAP_STUDY.md`, sha `5f127545…`): N=200 needs chunks at 16 384 tokens;
k=100 compact rows reach 15 984 tokens (2.4 % margin); non-coordinating chunks raised casualties in 7 of 8
proxy cells (N=200: 552 → 593/597) and caused 106–19 890 cross-chunk blocked moves.

| Approach | Keeps every person | Token risk | Coordination | Verdict |
|---|---|---|---|---|
| Shared global state only (current) | yes | 2.4 % margin | none between chunk outputs | **insufficient** (F3, F5) |
| Post-hoc deterministic coordinator | yes | none | repairs conflicts after the model | allowed only as a separate, reported "shield" arm |
| **Sequential inter-chunk reservations** | yes | ≈ +850 tokens at N = 200, k = 80; k = 100 fails closed at N = 200 | chunk j sees the decided targets of chunks < j | **recommended representation** (run 8 study) |
| Bounded deterministic correction | yes | none | changes person-blocked movers | **rejected** as a solution: 1–25 changes per cell, sometimes worse; separate reported arm only |
| Greedy exit allocation chunks | yes | low | explicit exit groups | **rejected**: +83 to +250 C+U vs full T1 |
| Compact crowd representation (region counts) | no per-person identity in the state | low | implicit | future research; changes the encoding |
| Model global plan (exit/region assignment) + person decisions | yes | plan row + person rows | explicit | option C / stage A2 if DEV shows chunk coordination loss |

Run 8 CPU study (#44 `docs/P1_CHUNK_RESERVED_STUDY.md`, `p1-chunk-reserved@1.0.0`, sha `0b4990ec…24eb7c87`;
40 new synthetic scenarios, matched seeds, T1 per chunk as the proxy row):

- Reservations remove cross-chunk blocked moves (N = 200, k = 80: 2131 → 0) with 0 reservation conflicts.
- They do not reliably improve outcomes: vs static chunks −9/−11/−62 C+U at k = 50 but +4/+8/−8 at k = 80
  (N = 100/110/200), with severe regressions and +1 unresolved at N = 200. Collisions drop; outcomes do not follow.
- vs full T1: equal at N = 110, k = 100 (−1); at N = 200 the best feasible proxy (reserved, k = 80) is +125 C+U
  (+14.7 %), 10/10 scenarios worse.

Recommended design: order chunks by person ID (the tested order; a fire-proximity order is untested).
Row j holds the full public state plus a reservation block listing (person, target cell) for every person
decided in rows < j. k = 100 for N ≤ 110; k = 80 for N > 110. The joint likelihood stays exact (§1).
Training rows are teacher-forced on the selected whole map that the teacher computed on the FULL state (never
a chunk-local plan). Every row must pass the 16 384-token check at every step; a row over the bound fails
closed. Raw model maps and any corrections are logged separately; the claim arm has no corrector. N = 200 is
a research case: TRAIN max N is 120, and only a model-level DEV test can show whether the student closes the
proxy gap.

## 5. Training roadmap (revised run 8; DRAFT, nothing authorized)

### 5.1 Stages and entry gates

| Stage | Data | Objective | Entry gate (all required) | Exit / stop |
|---|---|---|---|---|
| S0 CPU | synthetic fixtures | mock head: `student_objective` tests, weighted sampler, reservation rows, token check every step | none (done in part: normalization, chain rule, DPO/IPO values, admission) | all CPU tests green on the frozen harness commit |
| GPU preflight | 1–3 synthetic rows | none (measurement) | owner authorization; S0 green | padding parity batch 1 vs 2; save→reload identical logits; warm forward ms per row at 1 k/4 k/8 k/16 k tokens; rows/step at the chosen k |
| A: weighted SFT | class A **selected** whole maps (`select_joint_targets`), weights P1-14 §3 | Σ_rows weight × CE over the 5 options per person (exact chain-rule NLL of the map, §1) | P1-12 pilot passed; P1-14 dataset frozen + authorized; leakage audit; preflight report | DEV gate (P1-15 rules on DEV, no test); stop rules §5.4 |
| B: controlled preference | `admit_preference`-approved class C pairs only | DPO −log σ(βh) or IPO (h − 1/(2τ))², exact joint log-probs per map, frozen Stage-A reference, mixed with SFT (λ 0.5) | Stage A passed DEV; ≥ preregistered approved pairs; approval enabled in a FROZEN protocol | DEV: B beats A on the primary without guard loss, else keep A |
| C: student-visited | class E (P1-16, #49) | as B | frozen offline student passed DEV **and** written #49 authorization | — (conditional) |

### 5.2 Fixed choices

| Item | Choice | Evidence / reason |
|---|---|---|
| Pins | Qwen3.5-2B revision of #42 §3; unsloth 2026.10.11 wheel `baff520e…0992e`; Clef file `0e304cf7…dc3a3`; tokenizer `87a7830d…` (state `a57b5c62…`) | `verify_runtime_pins` refuses drift |
| Adapter / head | LoRA r16 on attention + MLP projections; Clef JointSchemaHead trained in full; bf16 | as #42; head is the only output path |
| Batch | 1 until DeltaNet padding parity is measured | unmeasured parity = possible silent loss error |
| Sequence | `max_len` 16 384; a row over the bound **fails closed**, never truncated | token study |
| Joint representation | one question per person, 5 contract options; rows = chunks with sequential reservations (§4) | exact likelihood §1 |
| N range | TRAIN `p1-02-v1/train` has max N = 120 (pilot plan); N = 200 appears only in synthetic studies | N = 200 needs a new TRAIN generator protocol before any claim at N = 200 |
| Data balance | scenario budget; teacher share > 0.5 = review only; family cap max(0.05, 2/F); strata report | P1-14 §3 audit |
| Teacher identity | provenance only; never a model input; no teacher module in the inference path | test in #44 |

### 5.3 Loss and checkpoint validation (before any long run)

1. Initial loss check: with a near-uniform head the per-person CE is ≈ ln 5 = 1.609; a value far from it means
   wrong option slicing or masking.
2. Mock-head parity: the GPU loss on 3 fixed rows equals `student_objective.sft_nll` on the same logits (|Δ| < 1e-4).
3. Normalization: on N = 3 the trained head's 125 map probabilities sum to 1 (as the CPU test).
4. Save → reload → identical logits (max |Δ| ≤ 1e-3 in bf16) and identical decoded maps.
5. Preference terms: h for 4 fixed pairs equals the CPU reference computed from the same logits.

### 5.4 Stop rules (each one stops the phase and needs a new authorization to continue)

- Non-finite loss or gradient; a fail-closed row in training data; any pin drift.
- After the 50-step pilot: mean per-person CE not below 1.5 (no learning signal), or throughput below the
  measured preflight value / 2.
- Spend reaches the phase planning cap (§6), or the measured Modal spend + next phase cap would pass $200.
- DEV: casualties or unresolved guard fails, or the raw assembled maps have a higher blocked-move rate than the
  selected class-A maps on the same states (mode averaging) → option C (strategy variable) review, no Stage B.
- Stage B: DEV primary worse than Stage A → keep Stage A.

## 6. Harness rebuild

**Done (run 15, CPU only).** `src/ember_v1/student_train.py` is the #44 harness (f9696a4) ported to this path;
protocol `data/protocols/p1-14-student-train.json` (`DRAFT_NOT_FROZEN`, authorizes nothing).

- Kept from #44: model/tokenizer pinning, `verify_runtime_pins`, LoRA setup, checkpoint integrity and
  save/reload, memory logging, token-budget row checks, wall/cost guards, `gpu_train_plan`, `pilot_go_rule`,
  the `ClefLoraAdapter` (still NOT EXECUTED).
- Removed: the `s1` training set, the P1-11 `student_protocol` gate and T1-only hard labels. The P1-11
  protocol, its doc and its permissions stay on the #44 branch only.
- Added: `rows_from_epoch` (only `student_ingest.prepare_examples` + `sample_epoch` output becomes rows);
  per-draw loss scaling so one draw contributes exactly `student_objective.sft_nll` (tested against it);
  gradient accumulation counts draws; partial or interleaved draws are refused; `check_protocol` requires all
  five teachers, the ingest source, a matching hash and phase caps that cover their wall caps.
- A T1-only protocol stays refused in two layers (`check_protocol` id check, then `sole_teacher`).
- Not yet: reservation rows in training (`reservations: false` in the draft), DPO/IPO stage, strategy-mixing
  evaluation. Next: measured GPU preflight, then freeze with the measured token budget.

## 7. Cost plan (planning estimates, not billing caps)

Rate $1.545/h A10 (#42 §7). Throughput is **not measured**: the only A10 evidence (P0B attempt 7) is one
617-token call with 113.4 s client time including container start. 3 700 and 1 900 tokens/s are brackets for
planning only; the GPU preflight replaces them before Stage A is authorized.

| Phase | Work | Estimate | Planning cap |
|---|---|---|---|
| S0 | CPU only | $0 | $0 |
| GPU preflight | padding parity, reload, latency per row, chunk smoke | ≤ 1.5 h | $3 |
| Pilot | 50 steps | ≤ 1 h | $2 |
| Stage A | sampled epoch, 80 M tokens | 6.0 h at 3 700 t/s ($9.3); 11.7 h at 1 900 t/s ($18.1) | $25 |
| Stage B | ≈ 3 000 pairs; per pair one forward per row of w and l (reservations: no sharing after the first differing chunk) + reference pass | ≈ 30–40 M token-equivalents ($3.5–$7) | $10 |
| DEV evaluations | 2 runs | ≈ 2 h each | $8 |
| Final test (P1-15) | once | ≈ 3 h | $6 |
| Stage C (conditional) | reserved, not planned | — | $40 |
| **Total envelope** | | | **$94 plan**; ceiling ≈ $200 total; read the actual Modal spend before each authorization |

The $94 is a plan, not a cap. The Modal workspace has no enforced dollar cap. Each phase needs its own
written owner authorization. **No GPU task is authorized in run 8.**
