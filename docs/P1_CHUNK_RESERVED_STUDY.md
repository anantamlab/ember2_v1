# P1 reserved-chunk coordination study (`p1-chunk-reserved@1.0.0`)

**CPU only.** No model, no GPU, no label, no TRAIN/DEV/P1-09/test/EMBERBench scenario. Coordination research
for the multi-teacher student (P1-14 training method §4), not a return to T1-only learning. T1 on one chunk's
sub-observation stands in for one student row. The 1.0.0 study (`docs/P1_CHUNK_JOINT_MAP_STUDY.md`,
`data/training/p1-11-student@1/chunk_study.json`) and its negative result are kept unchanged.

Output: `data/research/p1-chunk-reserved/study.json` (sha256 `0b4990ec…24eb7c87`, 400 runs, 654.8 s wall,
8 workers). Script `experiments/p1_chunk_reserved.py` (sha256 `11cfcd60…dd28da4c`). Command:

```
PYTHONPATH=src uv run --isolated --no-project --with tokenizers==0.22.2 --python <venv python> \
  python experiments/p1_chunk_reserved.py <pinned tokenizer.json> data/research/p1-chunk-reserved/study.json 8
```

The script refuses a tokenizer whose sha256 is not `87a7830d…` or whose state hash is not the A10
`metadata.tokenizer_sha256` `a57b5c62…`.

## 1. Design

- Scenarios: `chunk_joint.synthetic_scenario` with tag `p1-chunk-reserved@1.0.0` (ids `chunk-rsv-…`), new seeds
  10–14, deterministic and stochastic fire; N ∈ {50, 100, 110, 200}; 40 scenarios. Every method runs on the
  same scenario (matched seeds; pairs by scenario).
- k ∈ {50, 80, 100}; limit 16 384 tokens per row, checked at **every step** with the pinned tokenizer; a row
  over the limit fails the episode closed (never cut).
- Methods:

| Method | What a row sees | Coordination |
|---|---|---|
| `t1_joint` | full state; T1 on all people (reference; rows checked at k = N, or k = 100 for N = 200) | full |
| `t1_chunk_static` | full state; T1 on the chunk only (uncoordinated proxy, as 1.0.0) | none between chunks |
| `t1_chunk_reserved` | full state + reservation block of earlier chunks' targets | sequential reservations |
| `t1_chunk_reserved_fix` | as reserved, then `correct_conflicts`: bounded deterministic change of person-blocked movers, reverted if blocked moves rise | reported as a separate corrected arm; raw map logged |
| `t1_chunk_exitalloc` | chunks = greedy exit groups (capacity ⌈n / reachable exits⌉); each chunk routes only to its exit | explicit exit allocation |

- No chunked method calls the full planner (test: spy, ≤ 5 people per call). Raw and corrected maps are
  logged separately (`corrections_log`, `raw_blocked`, `raw_cross_chunk_blocked`).

## 2. Results (sums over 10 scenarios per cell; Δ = method − reference; C+U = casualties + unresolved)

Severe regression: one scenario worse by ≥ max(3, ⌈0.05 N⌉) on C+U.

### Against full T1 (`t1_joint`)

| N | k | static ΔC+U (sev) | reserved ΔC+U (sev) | reserved+fix ΔC+U | exit alloc ΔC+U (sev) | cross-chunk blocked static → reserved |
|---|---|---|---|---|---|---|
| 50 | ≥ N | — | — | — | +83 (5) | — |
| 100 | 50 | +132 (10) | +123 (9) | +119 | +250 (9) | 702 → 0 |
| 100 | 80 | +5 (0) | +9 (1) | +9 | +250 (9) | 322 → 0 |
| 110 | 50 | +66 (6) | +55 (4) | +55 | +190 (9) | 1105 → 0 |
| 110 | 80 | +21 (0) | +29 (1) | +29 | +190 (9) | 467 → 0 |
| 110 | 100 | 0 (0) | −1 (0) | −1 | +190 (9) | 125 → 0 |
| 200 | 50 | +320 (10) | +258 (9) | +261 | +209 (8) | 3560 → 5 |
| 200 | 80 | +133 (7) | +125 (5) | +137 | +147 (5) | 2131 → 0 |
| 200 | 100 | +64 (3) | **fail closed 10/10** | fail closed | +147 (5) | 1475 → — |

Reference totals: N = 50: 38 cas; N = 100: 248; N = 110: 263; N = 200: 853. Unresolved rose by 1 (reserved)
and 3 (exit allocation) at N = 200; 4–7 for exit allocation at N = 50/110.

### Reserved against static (same k)

| N | k | ΔC+U | worse / better / equal | severe |
|---|---|---|---|---|
| 100 | 50 | −9 | 3 / 4 / 3 | 2 |
| 100 | 80 | +4 | 3 / 1 / 6 | 0 |
| 110 | 50 | −11 | 1 / 7 / 2 | 1 |
| 110 | 80 | +8 | 3 / 4 / 3 | 1 |
| 110 | 100 | −1 | 1 / 2 / 7 | 0 |
| 200 | 50 | −62 | 2 / 8 / 0 | 0 |
| 200 | 80 | −8 (unresolved +1) | 3 / 6 / 1 | 2 |

### Tokens and compute

- Max row tokens: N = 200, k = 80: static 13 378, reserved 14 224 (+846, 13 % margin); N = 110, k = 100:
  reserved 14 760; N = 200, k = 100 reserved: **16 900 at step 0 → fail closed** (all 10 scenarios).
- Proxy decision wall per step (shared CPU, 8 parallel workers; not student or GPU latency), N = 200:
  joint p50/p95/p99 57/125/157 ms; reserved k = 80 67/120/145 ms. CPU sums are equal within 10 %.
- Joint-map validity: every completed step produced a complete valid joint map; 0 reservation conflicts.

## 3. Findings

1. **Reservations remove cross-chunk collisions** (e.g. 2131 → 0 at N = 200, k = 80) and keep the exact
   chain-rule likelihood (P1-14 §1).
2. **They do not reliably improve outcomes.** Against static they help at k = 50 and are flat or worse at
   k = 80 (N = 100: +4, N = 110: +8, N = 200: −8 with 2 severe regressions and +1 unresolved). This is the
   failure case the study was asked to report: collisions drop, outcomes do not follow.
3. **No chunked proxy matches full T1 at N = 200.** Best feasible: reserved k = 80, +125 C+U (+14.7 %), 10/10
   scenarios worse, 5 severe. At N ≤ 110 with k = 100 (one or two rows) reserved equals full T1 (−1).
4. **Reserved k = 100 does not fit at N = 200.** The reservation block costs ≈ 850 tokens at k = 80 and pushes
   the second row of k = 100 over 16 384.
5. **Bounded correction is a weak, risky shield.** 1–25 corrections per cell; −4 at N = 100, k = 50; +12 worse
   at N = 200, k = 80. Not a coordination solution; keep only as a reported separate arm.
6. **Greedy exit allocation fails** (+83 to +250 C+U vs full T1; up to +7 unresolved). It beats static only at
   N = 200, k = 50. Rejected in this form.

## 4. Recommended design (for the student, P1-14 §4)

- Representation: sequential reserved rows; k = 100 for N ≤ 110 (≤ 2 rows; equal to full T1 in the proxy);
  **k = 80 for N > 110** (k = 100 fails closed). Chunk order: person ID (tested); other orders are untested.
- Targets: the selected whole teacher map (computed by the teacher on the FULL state) teacher-forced into the
  reserved rows. The proxy shows that chunk-local planning loses people; the student must learn the global
  map, so its rows see the full state and the earlier reservations, and its target is never a chunk-local plan.
- No deterministic corrector in the claim arm; no exit-allocation chunking.
- N = 200 stays a research case: TRAIN `p1-02-v1/train` has max N = 120, and the proxy gap at N = 200 is
  +14.7 %. A model-level DEV test decides it, not this proxy.

## 5. Limits

The proxy is T1 per chunk, not a learned model; a student that learns the global map may coordinate better or
worse than this proxy. Ten scenarios per cell; no significance test is claimed. Wall-clock numbers are proxy
compute on a shared CPU.
