# P1-11 R7 — chunked inference correctness study (CPU, synthetic)

Status: **mechanics PASS on synthetic rooms; joint decision quality NOT preserved by non-coordinating
chunk proxies (negative result, kept).** This study does not measure the student.

| Item | Value |
|---|---|
| Code | `src/ember_v1/chunk_joint.py` (`p1-chunk-joint-map@1.0.0`), `experiments/p1_chunk_joint_map.py` |
| Result | `data/training/p1-11-student@1/chunk_study.json`, sha256 `5f127545abfbda8ecff60051d6c5d703923a270088bb5199af0e3e9e34ce6f31` (LF bytes) |
| Tokenizer | pinned Qwen3.5-2B `tokenizer.json`, sha256 `87a7830d…db72de4`; state sha256 `a57b5c62…77c6ca` (equals the A10 runtime metadata) |
| Bound | 16 384 tokens (`token_budget.SERVE_MAX_LEN`), one spare token required |
| Scenarios | `chunk_joint.synthetic_scenario` only: crowded rooms with a partition wall, 2 gaps, 3 exits; 20×30 grid (24×36 for N > 120). No TRAIN, DEV, calibration, held-out, locked or blind scenario is read; a test checks that the hashes are not in any split manifest. |
| Grid | N ∈ {50, 100, 110, 200}; k ∈ {25, 50} / {50, 100} / {100, 110} / {100}; fire deterministic and stochastic; seeds 0, 1, 2; 90 episodes |
| Host | CPU, Python 3.14.7, Windows 11; wall time 456 s; no GPU, no model, no label |

## 1. What is checked at every observation

For every observation of every episode (not only the first step):

1. `token_budget.chunk_rows` builds the rows. Every active person is in exactly one chunk, in ID order.
2. Every row carries the same full public global state.
3. Every row encodes with the real tokenizer within the bound (`clef_counts` status `OK`). A row over the
   bound stops the episode and is recorded as an error. No row is cut.
4. The per-chunk predictions are assembled into one joint action map. A missing, extra, duplicate or
   invalid action fails closed (`ChunkError`).
5. The complete map goes to the public resolver (`sim.movement.resolve_moves`, `lowest_id_wins`).
6. Blocked moves, cross-chunk blocked moves (blocked by a person of another chunk) and per-person
   differences from the unchunked T1 decision on the same observation are counted. No fallback action
   exists: an error ends the episode.

## 2. Predictors

| Predictor | Meaning |
|---|---|
| `t1_joint` | The unchunked T1 joint decision, split into chunks and assembled again. Plumbing check: assembly and resolution must not change it. |
| `t1_chunk_static` | T1 runs per chunk. People of other chunks are static obstacles: the chunk sees them, but not their moves. |
| `t1_chunk_blind` | T1 runs per chunk. People of other chunks are removed from the chunk's view. |

The two proxies are bounds on two information patterns. They are not the student. The student sees the
full state (with every person) in every chunk row, but not the other chunks' outputs.

## 3. Results (sums over 6 episodes per row: 2 fire models × 3 seeds)

| N | k | predictor | runs complete / errors | evacuated | casualties | unresolved | decisions (multi-chunk) | blocked moves (cross-chunk) | person actions differing from unchunked T1 | max row tokens | max unchunked tokens | unchunked single-row status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 25 | `t1_chunk_blind` | 6/6 / 0 | 269 | 31 | 0 | 362 (229) | 2099 (1480) | 1707 / 11527 (14.8%) | 4190 | 7452 | OK 362 |
| 50 | 25 | `t1_chunk_static` | 6/6 / 0 | 252 | 46 | 2 | 443 (260) | 394 (201) | 4276 / 13427 (31.8%) | 4200 | 7463 | OK 443 |
| 50 | 25 | `t1_joint` | 6/6 / 0 | 283 | 17 | 0 | 310 (201) | 0 (0) | 0 / 10388 (0.0%) | 4205 | 7469 | OK 310 |
| 50 | 50 | `t1_joint` | 6/6 / 0 | 283 | 17 | 0 | 310 (0) | 0 (0) | 0 / 10388 (0.0%) | 7469 | 7469 | OK 310 |
| 100 | 50 | `t1_chunk_blind` | 6/6 / 0 | 476 | 124 | 0 | 448 (267) | 8259 (5824) | 6055 / 26764 (22.6%) | 8049 | 14545 | OK 448 |
| 100 | 50 | `t1_chunk_static` | 6/6 / 0 | 479 | 121 | 0 | 489 (302) | 720 (401) | 11243 / 29648 (37.9%) | 8123 | 14642 | OK 489 |
| 100 | 50 | `t1_joint` | 6/6 / 0 | 529 | 71 | 0 | 430 (258) | 0 (0) | 0 / 26434 (0.0%) | 8120 | 14644 | OK 430 |
| 100 | 100 | `t1_joint` | 6/6 / 0 | 529 | 71 | 0 | 430 (0) | 0 (0) | 0 / 26434 (0.0%) | 14644 | 14644 | OK 430 |
| 110 | 100 | `t1_chunk_blind` | 6/6 / 0 | 563 | 95 | 2 | 486 (139) | 2591 (1499) | 1180 / 30058 (3.9%) | 14705 | 16014 | OK 486 |
| 110 | 100 | `t1_chunk_static` | 6/6 / 0 | 572 | 86 | 2 | 498 (149) | 132 (106) | 2955 / 30871 (9.6%) | 14767 | 16081 | OK 498 |
| 110 | 100 | `t1_joint` | 6/6 / 0 | 568 | 90 | 2 | 487 (148) | 0 (0) | 0 / 30575 (0.0%) | 14765 | 16080 | OK 487 |
| 110 | 110 | `t1_joint` | 6/6 / 0 | 568 | 90 | 2 | 487 (0) | 0 (0) | 0 / 30575 (0.0%) | 16080 | 16080 | OK 487 |
| 200 | 100 | `t1_chunk_blind` | 6/6 / 0 | 603 | 597 | 0 | 647 (349) | 28065 (19890) | 20694 / 69968 (29.6%) | 15886 | 28986 | OK 326, SCHEMA_TOO_LONG 284, STATE_CUT 37 |
| 200 | 100 | `t1_chunk_static` | 6/6 / 0 | 607 | 593 | 0 | 664 (390) | 2179 (846) | 29358 / 77898 (37.7%) | 15984 | 29095 | OK 300, SCHEMA_TOO_LONG 336, STATE_CUT 28 |
| 200 | 100 | `t1_joint` | 6/6 / 0 | 648 | 552 | 0 | 650 (353) | 0 (0) | 0 / 71398 (0.0%) | 15976 | 29125 | OK 324, SCHEMA_TOO_LONG 297, STATE_CUT 29 |

"Unchunked single-row status" counts the observations where one row with every person would fit
(`OK`) or not. "Differing" compares each chunked person action with the unchunked T1 action on the same
observation. Episodes differ in length, so the denominators differ.

## 4. Findings

| # | Finding | Kind |
|---|---|---|
| F1 | 90 of 90 episodes complete, 0 errors. Every observation passed checks 1–5. | mechanics, positive |
| F2 | `t1_joint` chunked equals `t1_joint` unchunked in every count (0 differences, same outcomes). Assembly and resolution are lossless when each chunk returns the joint decision. | plumbing, positive |
| F3 | Non-coordinating per-chunk decisions are worse than the joint decision in 7 of 8 proxy cells. Casualties: N=50 17 → 46 / 31; N=100 71 → 121 / 124; N=200 552 → 593 / 597. | quality, **negative** |
| F4 | One proxy cell is better: N=110, k=100, static: 86 casualties vs 90 (only 149 of 498 decisions had 2 chunks). This is one small cell of 6 episodes; it is not evidence that chunking helps. | quality, mixed |
| F5 | Chunking causes cross-chunk conflicts: blind 1 480–19 890 cross-chunk blocked moves per cell; static 106–846. The unchunked decision has 0 blocked moves. | coordination, **negative** |
| F6 | 3.9–37.9 % of person actions differ from the unchunked decision. | action difference, **negative** |
| F7 | Token margin is small: max chunk row 15 984 tokens at N=200, k=100 (400 tokens, 2.4 %, under the bound). A larger map or more fire cells can exceed the bound at k=100. Such a row fails closed; it is never cut. | budget, risk |
| F8 | Single unchunked rows fit up to N=110 here (max 16 080). At N=200, 49–55 % of observations need chunks. | budget |
| F9 | The verbose question style does not fit at k=100 with the N=200 smoke state: 17 441 and 17 255 tokens (`STATE_CUT`); k=50 is needed (`docs/P1_CHUNK_SMOKE.md`). The protocol trains and evaluates compact only. | budget, **negative** |

## 5. What this does and does not show

- It shows that the chunk pipeline keeps every person once, keeps the full state, stays within the real
  tokenizer bound or fails closed, and gives the resolver a complete joint map at every observation.
- It does **not** show that chunking preserves joint decision quality. "No person was dropped" is not
  that claim. F3–F6 show a loss for decision makers that do not coordinate across chunks.
- It does not measure the student. The student's chunk behaviour is unknown until it is evaluated.
  The protocol evaluates the student with chunks for N > 100 (`chunk_k` 100), so the DEV and holdout
  results include any chunk effect. This study does not change the protocol.
- 3 seeds × 2 fire models per cell, synthetic rooms only. The counts are descriptive. No interval and no
  significance claim is made.

## 6. R7 status

| R7 part | Status |
|---|---|
| every observation within 16 384 tokens | PASS (synthetic, compact) |
| every active person once | PASS |
| assembled joint map passes the public resolver | PASS |
| chunked vs unchunked on overlapping sizes, negative results kept | DONE (§3, §4) |
| chunking preserves joint decision quality | **NOT SHOWN**; proxies show a loss |
| the same checks on the GPU encoder | PENDING: `chunk_smoke` (not run, needs written authorization) |

Reproduce (CPU, about 8 minutes):

```
PYTHONPATH=src python experiments/p1_chunk_joint_map.py <tokenizer.json> data/training/p1-11-student@1/chunk_study.json
```
