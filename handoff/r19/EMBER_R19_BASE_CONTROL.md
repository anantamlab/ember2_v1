# Untrained base control vs V1 — 148 matched scenarios

**Date:** 2026-11-12 · **Machine:** RTX 5090 workstation (WSL2) · **Code:** `a3a9bf32763b98486a2ab95467ae4599ab787349`
**Scenarios:** the exact 148 frozen scenarios of `r19-v1-arms@1` (48 DEV + 80 core + 20 stress), same seeds, same fire
**Evaluation code:** `ember_v1.sim` + `teacher.evaluate.run_episode` + `ember_v1.policy_arms` (unchanged objects), same as V1's arms

## What "plain Qwen3.5-2B" is, exactly

The base LM has no action outputs, so it must run through V1's decision interface. The control is **V1 at optimizer step 0**:

| Component | State |
|---|---|
| Encoder | `unsloth/Qwen3.5-2B@fffbc0d8711cf87df6d1f97938c5e50ef94aadc7` — pinned public weights, untouched |
| LoRA r16 | created fresh; **all 186 `lora_B` tensors are exactly zero** → the adapter is a mathematical no-op, so the encoder is the plain base model |
| Head | created fresh with protocol seed `20261121`; sha256 `0f31b5311331e30daef25d4d6c61dccc636fa17c2b0e134c338b39a997467630` = **bit-identical** to `head_init_sha256_a10` in the V1 identity |
| Training steps | 0 |

This is verifiable, not assumed: the head digest matches V1's recorded initialisation exactly.

## Results — all 148 matched scenarios, pooled per person

| Policy | Casualties/person | Unresolved/person | Combined/person | Evacuation success | Blocked moves | Blocked/attempt |
|---|---|---|---|---|---|---|
| V1 raw | 0.7474 | 0.1010 | **0.8484** | 0.1516 | 258 239 | 0.8469 |
| **Base raw** | 0.8509 | 0.1237 | **0.9746** | 0.0254 | 419 691 | 0.7963 |
| V1 static mask | 0.6814 | 0.0839 | **0.7654** | 0.2346 | 148 456 | 0.6770 |
| **Base static mask** | 0.8257 | 0.1261 | **0.9518** | 0.0482 | 308 463 | 0.5799 |
| V1 joint coord | 0.6472 | 0.0791 | **0.7263** | 0.2737 | 0 | 0.0000 |
| **Base joint coord** | 0.7911 | 0.1166 | **0.9077** | 0.0923 | 0 | 0.0000 |
| T0 | 0.3899 | 0.0101 | 0.4000 | 0.6000 | 97 392 | 0.4557 |
| T1 | 0.3147 | 0.0262 | 0.3409 | 0.6591 | 0 | 0 |
| T2 | 0.3124 | 0.0274 | 0.3398 | 0.6602 | 0 | 0 |
| T3 | 0.3515 | 0.0140 | 0.3655 | 0.6345 | 0 | 0 |
| T4 | 0.3176 | 0.0274 | 0.3450 | 0.6550 | 0 | 0 |

### Per set (combined failure / person)

| Set | V1 raw | Base raw | V1 mask | Base mask | V1 coord | Base coord | T0 | T4 |
|---|---|---|---|---|---|---|---|---|
| DEV (48) | 0.7705 | 0.9700 | 0.6955 | 0.9446 | 0.6067 | 0.9129 | 0.3022 | 0.2393 |
| core (80) | 0.8426 | 0.9720 | 0.7422 | 0.9408 | 0.7019 | 0.9075 | 0.4117 | 0.3636 |
| stress (20) | 0.8910 | 0.9791 | 0.8195 | 0.9650 | 0.8056 | 0.9055 | 0.4369 | 0.3796 |

### V1 − base, combined failure, family-cluster bootstrap [95 % CI]

| Contrast | DEV | core | stress |
|---|---|---|---|
| raw − raw | −0.1995 [−0.2586, −0.1234] | −0.1294 [−0.2028, −0.0665] | −0.0882 [−0.1187, −0.0511] |
| mask − mask | −0.2491 [−0.3211, −0.1683] | −0.1986 [−0.3073, −0.0866] | −0.1454 [−0.2495, −0.0498] |
| coord − coord | −0.3062 [−0.4261, −0.1751] | −0.2055 [−0.3264, −0.0838] | −0.0998 [−0.2252, +0.0456] (ns) |

Base raw vs T0: +0.6678 / +0.5603 / +0.5423 (all exclude zero) — the untrained model is far below even the weakest teacher.

### Why the untrained model fails

| | north | south | east | west | stay |
|---|---|---|---|---|---|
| V1 raw actions | 0.26 | 0.16 | 0.08 | 0.12 | 0.38 |
| Base raw actions | 0.34 | **0.57** | 0.03 | 0.06 | **0.00** |

The untrained head has a strong north/south prior (91 % of actions) and **never** chooses "stay". It moves on every step, into fire and into occupied cells, which is why its blocked rate (0.796) is *lower* than V1's (0.847) while its outcomes are far worse. V1 learned both a stay policy and weaker directional bias — that is what training bought.

## Failure record (not dropped)

One scenario, `g-71671a9a15f17520` (stress, 200 people), failed in all three arms with
`policy_error:TrainError: truncation_detected {"total":100,"skipped":0,"truncated":1}` — the ASCII state for a
100-person chunk exceeds `max_seq_length = 16384`, and the framework refuses to truncate silently.
It is kept in the results with its partial outcome (200 unresolved for the base arms; V1 completes the same scenario normally).

Sensitivity: excluding it changes combined failure by ≤ 0.008 for every policy (e.g. base coord 0.9077 → 0.9055, V1 coord 0.7263 → 0.7358). Conclusions are unaffected.

## Interpretation

1. **Training is real and significant.** V1 beats the untrained model at every arm: −0.126 (raw), −0.187 (mask), −0.182 (coordinated) combined failure, and 3–6× the evacuation rate (15.2 % vs 2.5 % raw; 27.4 % vs 9.2 % coordinated). Every interval excludes zero except stress-coordinated.
2. **But training covered only a small part of the gap.** The distance from untrained to V1 (−0.13 to −0.31) is much smaller than the distance from V1 to T0 (+0.29 to +0.37) and T4 (+0.34 to +0.43). The model learned a stay policy and some direction, not evacuation planning.
3. **Post-processing amplifies what the scores know.** V1's advantage over the untrained model grows from raw (−0.126) to coordinated (−0.182): the coordinator exposes score quality that the raw argmax wastes.
4. **Blocked-move rate is still not the objective.** The untrained policy has fewer blocked moves than V1 but 6× worse evacuation.
5. For V2: a model trained for 3 580 steps from this initialisation still lands at 0.726 with perfect movement. The remaining deficit is representational (per-person independent actions, no joint coupling, no exit-aware target), not a shortage of training on the current objective.

## Artifacts (private `sankalphs/ember-v1-experiments`)

| Artifact | Manifest commit | Verified commit |
|---|---|---|
| `v1-policy-ablation/r19-base-control@1` | `0271c937b5156a821db12e231392eeefcd147de3` | `667a28dfae7bc2f5af429a9aaa554b2833f7ba99` |
| `v1-policy-ablation/r19-base-control-compare@1` | `315ebe5c79da4a06cd0344117e2b7db5d3e7bfb9` | `134e76879f8ee20fae64d756a111d76c748318be` |

Both published with `ARTIFACT.json` + `VERIFIED.json` and remote-verified by `hfstore` (0 problems).
