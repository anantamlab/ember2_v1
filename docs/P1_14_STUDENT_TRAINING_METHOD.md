# P1-14 multi-teacher student training method, chunk coordination and harness plan

**Status:** `DRAFT_NOT_FROZEN`. Supersedes the T1-only method of draft PR #42 for future training. #42/#43
stay as frozen-out history (their fail-closed guards stay). **No GPU, no Modal, no training is authorized.**

Model: pinned Qwen3.5-2B + Clef joint-schema head (pins, LoRA r16, bf16, `max_len` 16 384, A10) as in
`docs/P1_STUDENT_TRAINING_PROTOCOL.md` §3 of #42. Batch size 1 until DeltaNet padding parity is measured.
The student never calls a reference teacher at inference.

## 1. The student's probability model (why the objectives below are valid)

One decision row = full public state + one question per active person; the head gives logits over the 5
actions for each person. The model is therefore **factorized given the state**:

  log π(a | s) = Σ_i log softmax(z_i(s))[a_i]

- This is the exact log-likelihood of a complete joint action under the model. No sampling or
  approximation is needed. SFT cross-entropy is its negative (per-person mean).
- For two joint actions at the SAME state, one forward pass gives both log-likelihoods. People with the same
  action in both maps contribute the same term to both, so a pairwise objective depends only on the people
  whose actions differ.
- Limitation: a factorized model cannot represent a coordinated *distribution* over joint maps (for example
  "either A goes left and B right, or the reverse"). The argmax of per-person marginals can be a blocked
  composite. §4 addresses this for chunks; the same metric (blocked moves of the assembled argmax) is
  reported for single rows.
- With chunks (N > 100) and inter-chunk conditioning (§4), log π(a | s) = Σ_j log π(a_Cj | s, a_C<j): still
  exact by the chain rule.

## 2. Stage comparison

| Stage / objective | Data | Probability requirement | Verdict |
|---|---|---|---|
| A: weighted SFT (cross-entropy) | class A, weights of P1-14 §3 | per-person likelihood: exact | **Use first.** |
| A': SFT on all teachers equally | A+B | — | **Reject**: imitates losing trajectories. |
| B: DPO (frozen Stage-A reference) | class C pairs | log π(y_w|s) − log π(y_l|s): exact, one forward per pair; reference log-probs precomputed once | **Use when ≥ preregistered number of C pairs exists.** |
| B: ORPO (no reference model) | class A as SFT + class C as odds-ratio term | odds of the whole joint map: defined, but for many people the joint probability is tiny and the odds term saturates; per-person-normalized variant needed | Second choice; needs a numerical check first. |
| B: IPO | class C | same as DPO; squared loss resists over-fitting when pairs are near-deterministic | Variant to compare with DPO on DEV. |
| B: advantage-weighted regression | needs a scalar advantage per action | only ordinal, small-count, sometimes censored deltas exist; a scalar would be invented | **Reject** (no fabricated values). |
| B: KTO (unpaired good/bad) | A as good, B as bad | B actions are not known to be bad (§1 of P1-14) | **Reject**. |
| C: student-visited counterfactuals | class E (P1-16) | as Stage B | **Conditional**; only after a frozen offline student passes DEV (#49). |

DPO detail: β preregistered (start 0.1), loss on persons whose actions differ (others cancel exactly), mixed
with the Stage-A SFT loss (weight λ, start 0.5) to avoid drift on states without pairs. Reference log-probs
are computed once with the frozen Stage-A adapter; no second model in GPU memory.

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
| Imitating the average of incompatible teachers | class A by matched outcomes; §4 open decision of P1-14; blocked-move metric |
| One teacher dominates | teacher share cap 0.5 |
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
| **Sequential inter-chunk reservations** | yes | +≈2–3 tokens per already-decided person; needs k ≈ 50–80 at N=200 | chunk j sees the decided targets of chunks < j | **recommended** |
| Compact crowd representation (region counts) | no per-person identity in the state | low | implicit | future research; changes the encoding |
| Model global plan (exit/region assignment) + person decisions | yes | plan row + person rows | explicit | stage-2 option if reservations fail |

Recommended design: order chunks by a public deterministic rule (people nearest to fire first, ties by ID).
Row j holds the full public state plus a reservation block listing (person, target cell) for every person
decided in rows < j. The joint likelihood stays exact (§1). Training rows are built the same way from the
target joint map (teacher forcing). Every row must pass the 16 384-token check at every step; a row over the
bound fails closed. Raw model maps and any coordinator corrections are logged separately.

Next CPU experiment (no GPU): extend the #44 chunk study with a `t1_chunk_reserved` proxy (T1 per chunk,
earlier chunks' targets reserved) and measure casualties, cross-chunk blocked moves and max row tokens for
k ∈ {50, 80, 100} at N ∈ {110, 150, 200}.

## 5. Harness rebuild (Phase 7)

Reuse from #44 (`student_train.py`, after review): model/tokenizer pinning, `verify_runtime_pins`, LoRA setup,
checkpoint integrity and save/reload tests, GPU memory logging, token-budget checks, `gpu_train_plan` limits
and the `pilot_go_rule`. Remove from the NEW path: `teacher.sole_teacher`, T1-only row builders, T1-only
gold labels. Add: weighted row sampler (P1-14 §3), DPO loss on the factorized likelihood, reservation rows.
The old T1-only route stays refused in two layers (tested in #44).

S0 (CPU/mock, $0): weighted sampler determinism; DPO loss equals hand-computed value on a mock head; persons
with equal actions cancel; reservation rows round-trip; token check at every step; no teacher import in
inference.

GPU preflight (not authorized): A10 padding parity at batch 1 vs 2 (DeltaNet); N=200 chunk smoke; adapter
save → reload → identical logits; measured tokens/s for a 50-step pilot.

## 6. Cost plan (planning estimates, not billing caps)

Rate $1.545/h A10 (#42 §7). Throughput is **not measured**; 3 700 and 1 900 tokens/s bracket it.

| Phase | Work | Estimate | Planning cap |
|---|---|---|---|
| S0 | CPU only | $0 | $0 |
| GPU preflight | padding parity, reload, chunk smoke | ≤ 1.5 h | $3 |
| Pilot | 50 steps | ≤ 1 h | $2 |
| Stage A | sampled epoch, 80 M tokens | 6.0 h at 3 700 t/s ($9.3); 11.7 h at 1 900 t/s ($18.1) | $25 |
| Stage B | ≈3 000 pairs × 6 k tokens, one forward per pair + reference pass | ≈ 24 M token-equivalents ($2.8–$5.4) | $10 |
| DEV evaluations | 2 runs | ≈ 2 h each | $8 |
| Final test (P1-15) | once | ≈ 3 h | $6 |
| Stage C (conditional) | reserved, not planned | — | $40 |
| **Total envelope** | | | **$94** of ≈ $200; the actual spent amount must be read from Modal usage before each authorization |

The Modal workspace has no enforced dollar cap. Each phase needs its own written owner authorization.
