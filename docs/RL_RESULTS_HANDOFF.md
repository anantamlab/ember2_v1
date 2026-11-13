# EMBER V1 — RL results handoff

Status date: 2026-11-13. Audience: the next coding agent or researcher who works on the RL policy.
Read this after [CURRENT_STATE.md](CURRENT_STATE.md). It records the RL system as built and every RL metric
from Run 19 to R22. All numbers come from tracked evidence; each section names its source.

**Nothing in this file authorizes training.** A new run needs a frozen plan, a new run/attempt identity and owner
authorization (`AGENTS.md`). Final evaluation is sealed. V1.1 data (`p1-18-expand@1`, `p1-18-dataset-v2`) stays out.

## 1. Read order and sources

| Source | What it holds |
|---|---|
| [R22 report](R22_COUNTERFACTUAL_REPORT.md), [R22 plan](R22_COUNTERFACTUAL_PLAN.md) | Phase 3 results, CF method, checkpoint identities |
| [R21 report](R21_PPO_PHASE2_REPORT.md), [R21 plan](R21_PHASE2_PLAN.md) | Phase 2 results, selection curve, movement diagnostics |
| [R20 report](R20_PPO_PHASE1_REPORT.md) | Phase 1 method, numerical failures and their fixes |
| [Run 19 report](../handoff/r19/EMBER_R19_FINAL_REPORT.md), [base control](../handoff/r19/EMBER_R19_BASE_CONTROL.md) | V1 vs mask vs coordinator vs teachers on 148 cases |
| [handoff/HANDOFF.md](../handoff/HANDOFF.md) | Workstation archive, recovery notes |
| `data/rl/evidence/r2*/` | `summary.json`, `analysis.json`, `detailed_analysis.json`, `per_family.json`, `curves.html`, paired intervals |
| `handoff/operations/r20–r22/` | Exact launch, audit and archive scripts (historical; do not run unchanged) |

## 2. Checkpoints

| Item | State |
|---|---|
| V1 `p1-18-v1-train@1` | Private HF, rev `1fc405f7` (verified 2026-11-13). |
| RL checkpoints R20, R21 u96, R22 u16/u24 | On the RTX 5090 workstation only (`data/rl/checkpoint_registry.json`). Not on HF. |
| Base | `unsloth/Qwen3.5-2B@fffbc0d8711cf87df6d1f97938c5e50ef94aadc7`, frozen. Trainable: LoRA + float32 Clef head + critic. |
| GPU stack used | WSL Python 3.11, Torch 2.8.0+cu128, Transformers 5.2.0, Unsloth 2026.10.11, Triton 3.4.0, one RTX 5090 (32 GB). |

## 3. The RL system as built

| Part | Implementation | File |
|---|---|---|
| Environment | Wraps the independent CPU simulator. `reset()` returns the exact public observation; `step()` takes one simultaneous action map. Blocked moves are outcomes, not errors. Step cap = MDP terminal with unresolved people. | `src/ember_v1/rl_env.py` |
| Observation to the actor | Public contract only: `step, map{width,height,obstacles,exits}, rules, people, burning, allowed_actions`. V1 ASCII encoding, compact questions, sorted person IDs, chunk size 100. **No previous action or blocked flag.** | `contract.py`, `encoding.py` |
| Action space | Per person: `stay, north, south, east, west` (Clef five-option head). Joint action = one action per active person. | `contract.py`, `rl_ppo.py` |
| Actor | `V1Actor`: Clef/LoRA V1, sampled categorical at temperature 1, **no action mask, no coordinator**. Joint log-prob = sum of person log-probs. | `rl_ppo.py` |
| Critic | Fresh centralized MLP 1548→128→64→1 (tanh) on six 16×16 public channels + 12 scalars. | `rl_ppo.make_value_head` |
| Reward | Terminal `(E − 2C − 0.75U)/N`; potential shaping `eta·(γΦ' − Φ)`, Φ = negative public safe-exit BFS distance, η = 0.1. No move bonus, no stay penalty. | `rl_env.RewardConfig`, `potential` |
| PPO | One clipped ratio per **joint** transition (product over people). GAE on complete episodes. | `rl_ppo.optimize` |
| Counterfactual (R22) | Private snapshot at visited TRAIN states; matched-fire branches; candidates = student sample/argmax, T0–T4 proposals, two-person local variations; **fixed T1 continuation**; offline pair loss weight 0.05. | `rl_counterfactual.py`, `rl_cf_training.py`, `rl_cf_run.py` |
| Scenarios | Fresh independent generators; D4 topology-disjoint TRAIN / selection DEV / reporting DEV. | `rl_online_scenarios.py`, `rl_cf_scenarios.py` |
| Verification | Independent CPU replay; autograd likelihood audit; exact save/reload. | `rl_verify.py`, `rl_cf_verify.py`, `experiments/rl_*.py` |
| External rule arms (reporting only) | `v1-mask` = static legality mask; `v1-coordinate` = mask + joint conflict repair (falls back to stay). | `policy_arms.py`, `shield.py` |

Hyperparameters (frozen configs in `data/rl/`):

| Setting | R20 | R21 | R22 |
|---|---|---|---|
| Updates × episodes/update | 8 × 8 | 128 × 8 | 32 × 8 per arm (27 rounds used) |
| max_steps | 40 | 48 | 48 |
| γ / GAE λ | 0.99 / 0.95 | 1.0 / 0.97 | 1.0 / 0.97 |
| Shaping η | 0.1 | 0.1 | 0.1 |
| PPO epochs / clip / target joint KL | 2 / 0.2 / 0.04 | 2 / 0.2 / 0.04 | 2 / 0.2 / 0.04 |
| Entropy coef / value coef | 0.001 / 0.5 | 0.005 / 0.5 | 0.005 / 0.5 |
| LR LoRA / head / critic | 3e-6 / 1e-5 / 3e-4 | same | same |
| Accumulation groups / grad clip | 16 / 1.0 | 32 / 1.0 | 32 / 1.0 |
| CF coef / CF states per update | — | — | 0.05 / 4 |
| Max active people | 4–12 | 4–20 | up to 32 |

## 4. All metrics

### 4.1 Run 19 — V1 policy arms, 148 matched scenarios (48 DEV + 80 val core + 20 val stress)

| Metric | Raw V1 | Static mask | Coordinator | T0 | T1 | T2 | T3 | T4 |
|---|---|---|---|---|---|---|---|---|
| Evacuation | 0.1516 | 0.2346 | 0.2737 | 0.6000 | 0.6591 | 0.6602 | 0.6345 | 0.6550 |
| Casualties / person | 0.7474 | 0.6814 | 0.6472 | 0.3899 | 0.3147 | 0.3124 | 0.3515 | 0.3176 |
| Unresolved / person | 0.1010 | 0.0839 | 0.0791 | 0.0101 | 0.0262 | 0.0274 | 0.0140 | 0.0274 |
| Blocked / attempted | 0.8469 | 0.6770 | 0.0000 | 0.4557 | 0 | 0 | 0 | 0 |

- Raw V1 blocks: wall 106,481; boundary 243; fire 2,031; occupied 132,949; conflict 9,851; cycle 6,684.
- Raw action mix: north 26.0%, south 16.2%, west 11.6%, east 8.1%, stay 38.2% (directional prior). 37/148 raw episodes hit the step cap.
- Coordinator drives executed stay from 54% to 79%; its zero blocks come partly from standing still.
- Untrained base, raw: 2.54% evacuation (V1 SFT is a real gain over the base).
- Run 19 conclusion: movement repair buys 0.122 of 0.848 combined failure; the rest (0.726 vs T1 0.341) is decision quality.

### 4.2 R20 — Phase 1 pilot (16 fresh DEV families, 129 people)

| Actor | Evacuated | Casualties | Unresolved | Stay | Blocked / attempted |
|---|---|---|---|---|---|
| V1 greedy | 55.81% | 37.98% | 6.20% | 7.38% | 60.90% |
| RL greedy | 55.81% | 40.31% | 3.88% | 3.30% | 61.59% |
| V1 stochastic | 65.89% | 27.91% | 6.20% | 8.27% | 47.19% |
| RL stochastic | 63.57% | 32.56% | 3.88% | 6.65% | 50.54% |

RL − V1 greedy (2,000 bootstraps): evacuation [−2.13, +2.36] pp; casualties [0.00, +5.11] pp. 64 episodes, 1,946 transitions, 141 Adam steps, 522 s training. Parameter change: LoRA L2 0.158, head L2 0.571. **Stochastic V1 beat greedy V1 by 10 pp** here.

### 4.3 R21 — Phase 2 online PPO (reporting DEV: 32 families, 358 people)

| Policy | Evacuated | Casualties | Unresolved | Stay | Blocked / attempted |
|---|---|---|---|---|---|
| V1 | 174 (48.60%) | 130 (36.31%) | 54 (15.08%) | 10.68% | 68.33% |
| V1 + mask | 202 (56.42%) | 99 (27.65%) | 57 (15.92%) | 24.58% | 51.97% |
| V1 + coordinator | 218 (60.89%) | 90 (25.14%) | 50 (13.97%) | 49.55% | 0.00% |
| R20 pilot | 168 (46.93%) | 129 (36.03%) | 61 (17.04%) | 12.01% | 68.55% |
| R21 update 96 | 174 (48.60%) | 127 (35.47%) | 57 (15.92%) | 10.66% | 71.11% |

- PPO − V1: evacuated 0.00 [−2.89, +3.36] pp; casualties −0.84 [−3.76, +2.06]; unresolved +0.84 [−0.95, +2.56].
- Mask − V1: +7.82 [3.49, 12.46] pp. Coordinator − V1: +12.29 [7.58, 17.63] pp.
- Selection DEV utility by update (V1 −0.4577): 16 −0.4746, 32 −0.4831, 48 −0.4535, 64 −0.4451, 80 −0.4838, **96 −0.4176**, 112 −0.4662, 128 −0.5197. Non-monotone; last iterate worst.
- Movement, V1 → PPO: wall 1,932 → 2,149; boundary 102 → 161; occupied 1,572 → 1,483; conflict 199 → 177; fire 221 → 171; cycle 130 → 160; repeated ineffective 3,597 → 3,774.
- Policy shift on identical inputs: 37/358 argmax actions changed; entropy 0.648 → 0.721 nats; KL to V1 0.0666 nats/person; mass on admissible moves 89.39% → 89.26%; on distance-reducing admissible moves 64.33% → 62.94%.
- TRAIN (stochastic): 59.31% evacuated, 25.90% casualties; updates 113–128 at 63.01% vs 55.34% in 1–16 (different maps).
- Optimization: joint KL mean 0.0072, clip-outside 10.17%, 12 KL early stops, per-person entropy 0.609.
- Compute: 1,024 episodes, 38,950 transitions, 2,489 Adam steps, 18,803 s on one RTX 5090; 13.2 transitions/s collection; GPU mean 37.9%; peak 16.6 GB.

### 4.4 R22 — PPO-only vs PPO+CF from R21 u96 (reporting DEV: 32 families, 473 people)

| Policy | Evacuated | Casualties | Unresolved | Stay | Blocked / attempted | Wall/boundary | Repeated ineffective |
|---|---|---|---|---|---|---|---|
| V1 | 41.65% | 24.10% | 34.25% | 6.66% | 81.20% | 5,860 | 8,325 |
| V1 + mask | 54.12% | 16.91% | 28.96% | 19.98% | 56.18% | 0 | 4,204 |
| V1 + coordinator | 60.68% | 15.22% | 24.10% | 46.82% | 0.00% | 0 | 0 |
| R21 u96 | 39.53% | 23.47% | 37.00% | 5.24% | 83.06% | 6,323 | 8,870 |
| PPO-only u16 | 40.80% | 24.10% | 35.10% | 4.63% | 82.61% | 5,853 | 8,759 |
| PPO+CF u24 | 41.65% | 21.99% | 36.36% | 4.88% | 81.83% | 5,857 | 8,596 |
| T0 | 87.74% | 8.03% | 4.23% | 25.29% | 7.07% | 0 | 126 |
| T1 / T2 | 89.01% | 1.48% | 9.51% | 26.6 / 26.2% | 0% | 0 | 0 |
| T3 | 89.22% | 1.90% | 8.88% | 34.33% | 0% | 0 | 0 |
| T4 | 89.22% | 1.48% | 9.30% | 37.75% | 0% | 0 | 0 |

Movement counts (`data/rl/evidence/r22-counterfactual-ppo@1/detailed_analysis.json`):

| Arm | Attempted | Blocked | Wall | Boundary | Occupied | Conflict | Cycle | Stay |
|---|---|---|---|---|---|---|---|---|
| V1 | 11,170 | 9,070 | 5,731 | 129 | 2,414 | 209 | 296 | 797 |
| PPO u16 | 11,553 | 9,544 | 5,692 | 161 | 2,884 | 249 | 306 | 561 |
| CF u24 | 11,548 | 9,450 | 5,652 | 205 | 2,850 | 193 | 278 | 593 |
| T1 | 4,474 | 0 | 0 | 0 | 0 | 0 | 0 | 1,625 |

- **About half of all raw attempted moves hit a wall** (V1 51.3%, PPO 49.3%, CF 48.9%). Teachers attempt ~40% as many moves and never hit a wall.
- CF − PPO (5,000 paired bootstraps): evacuated +0.85 [−1.86, +3.48]; **casualties −2.11 [−3.96, −0.39]**; unresolved +1.27 [−1.27, +3.98]; repeated-ineffective rate −1.38 [−4.19, +1.39]; wall/boundary rate +0.06 [−2.90, +3.12].
- CF − V1: evacuated 0.00 [−4.68, +4.76]; casualties −2.11 [−5.90, +1.60].
- CF − R21 u96: wall/boundary rate −3.83 [−6.67, −1.00]; repeated-ineffective rate −2.09 [−4.06, −0.11].
- Selection: PPO-only u16 utility −0.087691 vs CF u24 −0.115468 → **PPO-only u16 frozen**.
- TRAIN stage diagnostics (PPO / CF): joint KL 0.0046/0.0045 (navigation) rising to 0.0166/0.0154 (large unseen); clip fraction 5.5% → 26.3% / 5.1% → 24.8%; person entropy 0.81 / 0.87 in the largest stage; KL early stops 1 / 3.
- CF studies: 108 states, 4,038 branch episodes, 43,201 branch transitions, **26 eligible pairs**. Teacher proposals helped in 24–28 of 108 states, tied in 80–83, harmed ≤ 1. 418 auxiliary steps reused the 26 pairs; the selected CF actor holds 372.
- Compute: 216 episodes per arm, 609/611 optimizer steps, 11,954 s total (incl. a failed 1,751 s attempt), GPU mean 63.7%, peak 25.8 GB.

### 4.5 Panels are not comparable across runs

Raw V1 evacuates 15.16% (Run 19), 55.81% (R20), 48.60% (R21) and 41.65% (R22) because the panels differ in size, population and fire. Compare arms only inside one table. Each RL run used one training seed.

## 5. Evaluation protocol used in R21 and R22

- Fresh TRAIN, selection-DEV and reporting-DEV families from the independent generators, D4 topology-disjoint from each other and from earlier sets.
- Primary utility `(E − 2C − 0.75U)/N`; casualty guard ≤ +2 pp vs raw V1; checkpoints selected on selection DEV only; reporting DEV used once after selection.
- Reported arms on one panel: V1, V1 + mask, V1 + coordinator, earlier learned arms and (R22) T0–T4; paired family bootstrap, 5,000 replicates.
- Movement table (attempted, blocked by cause, stay, repeated ineffective) reported for every arm.
- Greedy decoding is primary; R20 also reported stochastic decoding.
- Final held-out evaluation sealed; no run opened it.

## 6. Failures and fixes in R20–R22

| Problem | Fix |
|---|---|
| bf16 no-grad rollout vs autograd PPO forward differed by up to 0.0022 in joint log-prob (R20 attempt 4). | Collect with the same autograd-enabled forward as PPO; detach stored values. |
| `save_pretrained` silently casts the Clef head to bf16 and ignores `exact=True` (R20 attempt 5). | Save with `_save_clef_adapter(..., exact=True)`; check save/reload before training. |
| WDDM GPU memory saturation at 32 GB (R20 attempt 3). | Two environments per GPU batch; accumulate groups. |
| Compiled backward rejected a retained graph on the first CF pair (R22 attempt 1). | Separate single-use graphs for the auxiliary probe and update. |
| Fresh-load greedy replay failed under a different Torch thread count (R22 audit). | Set Torch CPU threads to 4 for replay, as in the trainer. |
| Geometry capacity error and D4 topology collision (R21 attempts 1–2). | Validate generator capacity and topology disjointness before GPU load. |
| Windows/WSL line endings (R20 attempt 1). | `core.autocrlf=true` checkout; hash committed LF blobs. |
| Derived audit files changing manifests. | Verify on scratch copies only. |
