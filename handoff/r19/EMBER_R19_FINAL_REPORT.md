# EMBER v1 — Run 19 final report (Milestone 1 → Milestone 2)

**Date:** 2026-11-12 · **Machine:** college workstation (Windows 11 host + WSL2 Ubuntu 24.04)
**Pinned code:** `b91654c1cb1bcb24f24aca01815e2ec5098cc625` (PR #53 head)
**Milestone 2 authorization commit:** `a3a9bf32763b98486a2ab95467ae4599ab787349` (created in the local mirror only, **not pushed**; patch handed back to the laptop)

---

## Milestone 1 — workstation migration

### Hardware and OS

| Item | Value |
|---|---|
| Host OS | Windows 11 Pro 10.0.26200 |
| Execution OS | WSL2 Ubuntu 24.04.4 LTS, kernel 6.6.87.2-microsoft-standard-WSL2 |
| CPU | Intel Core Ultra 9 285, 24 threads |
| RAM | 127.5 GB host / 62 GB in WSL |
| GPU | NVIDIA GeForce RTX 5090, 32 607 MiB, compute capability **12.0 (sm_120)** |
| Driver | 610.60 (KMD 610.60, CUDA UMD 13.3) — ≥ 570 required for CUDA 12.8 wheels |
| Disk | 948 GB free |
| Toolchain | gcc, make, curl, git present; no `sudo` needed, no Linux NVIDIA driver installed inside WSL |

### Environment (hash-locked)

- Python **3.11.13** (uv 0.8.13), venv built from `infra/p0b-requirements.lock.txt` with `--require-hashes --no-deps`; 107 packages.
- **torch 2.8.0+cu128**, CUDA 12.8, `cuda_available=True`, arch list `sm_70…sm_100, sm_120`, **triton 3.4.0**, transformers 5.2.0, unsloth 2026.10.11, xformers 0.0.32.post2.
- `p0b verify-install`: **PASS** — 0 version mismatches; audited upstream hashes (clef, decision, decision_from_lm, decision_common) all match.
- Lock sha256 (LF): `9613f929b53c0fed65105b1f808f64e8f842dab29e837b3dca3e503cbd60bc2c`.

### GPU verification (`r19-env-check@1`) — **PASS**

| Check | Result |
|---|---|
| CUDA available | True |
| RTX 5090 detection | NVIDIA GeForce RTX 5090, 31.84 GB |
| `sm_120` kernels | present in `torch.cuda.get_arch_list()` |
| bf16 matmul | 15.4 TFLOP/s, finite |
| SDPA | finite |
| Depthwise conv | finite |
| Triton JIT kernel | compiled, result exact (`triton 3.4.0`) |
| Pinned install audit | `verify-install` exit 0 |

### Traces (`r19-trace-replay@1`) — **PASS**

156/156 recorded V1 traces replay to their recorded outcomes (evacuated / casualties / unresolved / steps) with the original records manifest `283a753c7d4a1a3fc63e13f0c73c41c770810d19`.

### V1 checkpoint reproduction (`r19-v1-repro@1`) — **NOT_REPRODUCED (protocol verdict)**

Real V1 loaded on the 5090: base `unsloth/Qwen3.5-2B@fffbc0d8…` (16 files hash-verified), LoRA adapter + Clef joint head + pinned tokenizer.

| Quantity | Value |
|---|---|
| Steps replayed | 30 / 30 (5 episodes × 6 steps) |
| Step strings exactly equal | 23 / 30 |
| Person decisions exactly equal | **1528 / 1538 (99.35 %)** |
| Disagreements | 10, **all** top-1/top-2 flips (recorded action ranked 2nd) |
| Gaps of those flips | 0.0056, 0.0105, 0.0247, 0.0036, **0.0683**, 0.0441, 0.0218, 0.0075, 0.0367, 0.0154 |
| Median top-2 margin (all persons) | 1.214 |
| Peak GPU memory | 5.47 GB |

Nine of ten flips are below the frozen near-tie tolerance 0.05; **one (P83, 0.0683) exceeds it**, so the frozen rule returns `NOT_REPRODUCED`. The tolerance was not changed. The recorded action is the runner-up in every case and the gaps are 1–2 orders of magnitude below the median margin, which is the signature of bf16 kernel differences between A10 (Ampere) and Blackwell, not of a wrong checkpoint. `weights_digest` also differs from the training-time digest (`c31cae70…` vs `8c8fea05…`); the code documents that digest as informational after reload, and 99.35 % agreement rules out a mis-loaded checkpoint.

**Verdict:** Milestone 1 environment, code identity and data identity **PASS**. V1 reproduction is **99.35 % person-level**, but it does not meet the frozen 0.05 rule on one of 1538 decisions, and is reported as such.

### Milestone 1 artifact revisions (private `sankalphs/ember-v1-experiments`)

| Run | Verdict | Manifest commit | Verified commit |
|---|---|---|---|
| `student-rollouts/r19-env-check@1/attempt-1` | PASS | `cd04b1fdf3f277b3516b291b4e86f1de5c76eec9` | `86d90fd154b74d882ee407baa018bec788e2bc24` |
| `student-rollouts/r19-trace-replay@1/attempt-1` | PASS | `ec689e4a4c4194157b4771e9b71ca9643bf60974` | `6c679bbf252f8c99ffecc6b2cdbf192a0289d7f8` |
| `v1-evaluation/r19-v1-repro@1/attempt-1` | NOT_REPRODUCED | `0b76a36d332048fc26bdd067d052fce6637db2a7` | `6db58813907db8b9459152a312dc1f003c5bb783` |

Each was re-downloaded via `fetch-results` and the verdict independently rechecked.

---

## Milestone 2 — movement-aware inference

### Provenance

- Same checkpoint for all arms: `p1-18-v1-train@1` @ `1fc405f723718e308c8a22b137fee4a025086fde`
- Records: `p1-18-v1-records@1` @ `283a753c7d4a1a3fc63e13f0c73c41c770810d19`
- 148 frozen scenarios: 48 DEV + 80 expansion val core + 20 expansion val stress (identical seeds → identical fire)
- Arms interleaved in blocks of 4; **0 scenario failures, 0 token overflows, 0 missing pairs**
- Run identity `sha256:8bd2ee22f125134f53fbbffa86b3ae1f18ffbcde86542f7b7a607441ab039559`

### Results — 148 matched scenarios, pooled per person

| Metric | Raw V1 (A) | Static mask (B) | Joint coordinator (C) | T0 | T1 | T2 | T3 | T4 |
|---|---|---|---|---|---|---|---|---|
| Casualties / person | 0.7474 | 0.6814 | 0.6472 | 0.3899 | 0.3147 | 0.3124 | 0.3515 | 0.3176 |
| Unresolved / person | 0.1010 | 0.0839 | 0.0791 | 0.0101 | 0.0262 | 0.0274 | 0.0140 | 0.0274 |
| Combined failure / person | 0.8484 | 0.7654 | 0.7263 | 0.4000 | 0.3409 | 0.3398 | 0.3655 | 0.3450 |
| Blocked moves | 258 239 | 148 456 | 0 | 97 392 | 0 | 0 | 0 | 0 |
| Blocked / attempted move | 0.8469 | 0.6770 | 0.0000 | 0.4557 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Evacuation success | 0.1516 | 0.2346 | 0.2737 | 0.6000 | 0.6591 | 0.6602 | 0.6345 | 0.6550 |

Broken down by blocked-move type (A → B → C): wall 106 481 → 0 → 0; boundary 243 → 0 → 0; fire 2 031 → 0 → 0; occupied 132 949 → 115 288 → 0; conflict 9 851 → 18 332 → 0; cycle 6 684 → 14 836 → 0.

> Independent corroboration of the reported V1 failure profile: A's 106 481 wall conflicts and 132 949 occupied-cell conflicts match the reported ~106 000 / ~132 000, and A's 84.7 % blocked rate matches the reported ~84 %.

### Per set (combined failure / person, evacuation success)

| Set | A | B | C | T0 | T1 | T4 |
|---|---|---|---|---|---|---|
| DEV (48, 16 families) | 0.7705 / 0.230 | 0.6955 / 0.305 | 0.6067 / 0.393 | 0.3022 / 0.698 | 0.2422 / 0.758 | 0.2393 / 0.761 |
| val core (80, 20 families) | 0.8426 / 0.157 | 0.7422 / 0.258 | 0.7019 / 0.298 | 0.4117 / 0.588 | 0.3617 / 0.638 | 0.3636 / 0.636 |
| val stress (20, 5 families) | 0.8910 / 0.109 | 0.8195 / 0.181 | 0.8056 / 0.194 | 0.4369 / 0.563 | 0.3701 / 0.630 | 0.3796 / 0.620 |

Teacher baselines on this workstation reproduce the historical numbers exactly (DEV T0 0.3022, T1 0.2422, T2 0.2399, T3 0.2463, T4 0.2393 vs reported 0.302 / 0.242 / 0.240 / 0.246 / 0.239), which validates simulator and data identity end-to-end.

### Family-cluster bootstrap (n = 2000, seed 20261110), combined-failure difference [95 % CI]

| Contrast | DEV | val core | val stress |
|---|---|---|---|
| A − B | +0.0750 [+0.0370, +0.1167] | +0.1004 [+0.0450, +0.1490] | +0.0715 [+0.0100, +0.1607] |
| B − C | +0.0888 [+0.0395, +0.1431] | +0.0403 [+0.0207, +0.0639] | +0.0139 [−0.0027, +0.0365] |
| A − C | +0.1638 [+0.0846, +0.2489] | +0.1407 [+0.0768, +0.1978] | +0.0854 [+0.0128, +0.1977] |
| C − T0 | +0.3045 [+0.2017, +0.4161] | +0.2902 [+0.2354, +0.3441] | +0.3687 [+0.2492, +0.5244] |
| C − T4 | +0.3674 [+0.2666, +0.4608] | +0.3384 [+0.2860, +0.3858] | +0.4260 [+0.2894, +0.5890] |

Masking and coordination are both significant. The coordinator's advantage over masking is **not** significant on the stress set (5 families).

### What the post-processors actually changed

| | Person-actions changed | Steps changed | Repair iterations | Stay share (raw → executed) |
|---|---|---|---|---|
| A | 0 | 0 | 0 | 0.382 → 0.382 |
| B | 48 336 (10.8 % of its person-steps) | 8 295 / 11 120 | 0 | 0.450 → 0.509 |
| C | 45 243 mask + 89 390 coordinator = 134 633 (30.9 %) | 8 652 / 10 973 | 103 277 | 0.542 → **0.790** |

Raw action distribution (A): north 0.260, south 0.162, west 0.116, east 0.081, stay 0.382 — a strong north/south anisotropy on maps where direction should be roughly map-dependent.

### Cost

- Decision latency p50: A 119 ms, B 115 ms, C 114 ms; stress p50 ≈ 283–308 ms (200 people = 2 chunks).
- Peak model memory 5.47 GB; whole process observed at 10.0 GB VRAM, 99 % GPU utilisation, ~509 W.
- Wall time: 5 418 s for the resumed 132-scenario-segment (≈ 1.6 h total including the interrupted first segment).

### Artifact revisions (private `sankalphs/ember-v1-experiments`)

| Artifact | Contents | Manifest commit | Verified commit |
|---|---|---|---|
| `v1-policy-ablation/r19-v1-arms@1/attempt-1` | scenarios, per-arm raw + executed rollouts, metrics, logs, manifest | `bf0fdf5bd8eba13c112fb0084f869f172f55ed60` | `82c766354bf34efce0f01f8a7e46b092805fa91f` |
| `v1-policy-ablation/r19-v1-arms-compare@1` | `summary.json`, `arms_compare.json`, README | `c3b97c8aa419bff8fee91aaa9a07f923fccb0105` | `ce6a402536e88dc692cda2d88362b5c51975671d` |

---

## Research conclusions

**1. Invalid individual movements, joint coordination, or poor decisions?**
All three, in that order of size, but the dominant term is decision quality.
Invalid individual moves are large and cheap to remove: static legality masking cuts blocked moves from 84.7 % to 67.7 % of attempts and combined failure from 0.848 to 0.765 (Δ −0.083, CI excludes 0 in every set). Joint coordination removes the remaining 67.7 % of blocks and takes failure to 0.726 (Δ −0.039, significant on DEV/core, not on stress). Movement alone therefore accounts for 0.122 of the 0.848 — but the residual 0.726 against T1's 0.341 is decision quality, and it is 3× larger than everything movement repairs bought.

**2. How much does static legality masking improve evacuation?**
Evacuation success 0.1516 → 0.2346 (+55 % relative), combined failure 0.848 → 0.765. It removes **all** wall, boundary and fire moves (106 481 + 243 + 2 031 = 108 755 wasted or unsafe moves) and roughly doubles the share of people who leave the building, while leaving occupancy/conflict/cycle blocks untouched.

**3. How much does joint coordination add beyond masking?**
A further +0.039 evacuation (0.235 → 0.274) and −0.039 combined failure, at the cost of changing 30.9 % of all person-actions and driving "stay" from 54 % of raw decisions to **79 % of executed decisions**. It eliminates 100 % of blocked moves by construction, because the repair loop's fallback is "stay". The gain is real on DEV and core, statistically indistinguishable from zero on stress. **Zero blocked moves is therefore not evidence of good movement — it is partly an artifact of standing still.**

**4. Does coordinated V1 beat T0 or approach T1–T4?**
No. C's 0.726 combined failure is 0.326 worse than T0 (0.400) and 0.385 worse than T4 (0.345); every C-vs-teacher interval excludes zero by a wide margin. C also attempts only 91 458 moves against T4's 216 235 — it evacuates 27 % of people where T4 evacuates 66 %.

**5. What is justified for V2?**
The evidence points at representation and objective, not at more data:
- **Joint action representation.** V1 scores each person independently per chunk; it cannot represent "who yields to whom". C had to rewrite 8.1 person-actions per decision to make the raw preferences executable. A model that predicts a joint action map (the existing P1 chunk-joint map work) attacks the actual failure.
- **Directional bias.** north 26 % vs east 8 % of raw actions is not map-driven; the model has learned a prior, not a route. Exit-aware features/targets (distance-to-exit, potential-based progress) are the obvious fix.
- **Stay collapse.** 38 % raw stay and 79 % under repair means the policy under-moves. An explicit movement/progress auxiliary objective, plus DAgger-style correction on its own visited states, targets both the bias and the compounding error (37/148 A episodes hit the step cap).
- **Class-A teacher weighting** already exists and is the right lever for "correct outcome, not teacher imitation".

**6. Existing expanded dataset first, or revise representation/objective first?**
Revise representation and objective first. V1 was trained on 128.7 M tokens / 3 580 steps and still collapses to 0.726 under perfect movement execution, with a strong directional prior and a stay bias. More of the same per-person SFT targets will not fix a representational limitation. Use the existing `p1-18-dataset-v2` (7.6 GB) and `p1-18-expand@1` once the action representation is joint and the objective rewards progress, rather than collecting new data now.

---

## Critical limitations

1. **V1 is not bit-exact across machines.** 1 of 1538 decisions exceeds the 0.05 near-tie rule (0.068). A/B/C are internally consistent (all inferred on this 5090), but A's 0.770 DEV is not byte-identical to the historical 0.779.
2. **C's `blocked = 0` is definitional**, produced by a repair loop that can always fall back to "stay". Read it together with the 79 % stay share and the 91 k attempted moves.
3. **The stress set has 5 families**, so its intervals are wide and B−C is not significant there.
4. **No per-scenario oracle selection** was used; `best_T1-T4` exists in the comparison only as a non-deployable reference.
5. **A_raw has 25 % censored episodes** (37/148 hit `max_steps`); their people are counted as unresolved, not as evacuated.
6. **Cross-machine numerics** mean the difference between 0.770 and the reported 0.779 is not attributable to any single cause.
7. **The HF token used here is broader than least privilege** (global `repo.write`). It was used from `~/ember-ws/hf-home/token` only and is now logged out locally, but **it must be revoked and replaced by the owner** — it cannot be revoked by an agent.

## Handoff actions for the laptop

1. Apply `ws-tmp/authorize-r19-v1-arms.patch` (or recreate the same single-block edit) and push it, so the executed commit `a3a9bf32…` exists on the branch. Content is identical to what was executed.
2. Add the five artifacts above to `data/hf/registry.json` (`manifest_commit`, `manifest_sha256`, `verified_commit`) so they become part of the tracked registry.
3. Review `v1-policy-ablation/r19-v1-arms-compare@1` (`summary.json`) — it contains both raw and executed action traces separated, as required.
