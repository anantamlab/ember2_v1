# Full-view report

On this report panel, the new raw full-view model is better than original V1 on utility.
PPO did not run because its entry gate failed, so no PPO gain over DAgger is established.

The raw utility difference is +0.323427 (95% interval [0.193501, 0.453877]). Evacuation increased
from 49.16% to 60.35%, and casualties fell from 33.99% to 23.78%. The masked comparison does not
establish an advantage: utility difference +0.021678, interval [-0.095381, 0.148340].
T1 still evacuated 90.49% with 3.57% casualties, so the new model remains far below the teacher.

## One-shot report

All seven arms ran once on the same 128 families and 1,430 people, with a 48-step cap. Namespace:
`ember2-full-view-full-view-20261114t133215z-37025f-reporting`. Candidate selection used SELECT only; this report panel is spent.

| Arm | Evacuated | Casualties | Unresolved | Utility | Blocked / attempted | Wall / boundary |
|---|---:|---:|---:|---:|---:|---:|
| New candidate, raw | 863/1,430 | 340 | 227 | 0.008916 | 39.03% | 0 / 0 |
| New candidate + mask | 863/1,430 | 340 | 227 | 0.008916 | 39.03% | 0 / 0 |
| DAgger before PPO, raw | 863/1,430 | 340 | 227 | 0.008916 | 39.03% | 0 / 0 |
| T1 teacher | 1,294/1,430 | 51 | 85 | 0.788986 | 0.00% | 0 / 0 |
| Random legal moves | 255/1,430 | 718 | 457 | -1.065559 | 11.01% | 0 / 0 |
| Original V1, raw | 703/1,430 | 486 | 241 | -0.314510 | 71.80% | 9,028 / 419 |
| Original V1 + mask | 871/1,430 | 376 | 183 | -0.012762 | 45.07% | 0 / 0 |

Utility is `(evacuated - 2*casualties - .75*unresolved) / people`. Wall and boundary blocks fell from
9,028 and 419 in original raw V1 to zero in the new raw model. Remaining new-model blocks were
5,114 occupied-cell blocks, 1,638 conflicts and 1,490 cycles. Static legality does not solve crowd coordination.

The 95% percentile intervals below resample whole matched families 5,000 times, seed 0, and recompute
population-weighted rates. Rate differences are percentage points (pp); utility differences are unscaled.
Full precision is retained in the compact JSON summary. The candidate and DAgger are the same S2 checkpoint;
their identical results and zero intervals are not a PPO experiment.

| Paired comparison | Metric | Difference | 95% interval |
|---|---|---:|---:|
| New − original, raw | evacuated_rate | +11.188811 pp | [+6.355485, +16.184140] pp |
| New − original, raw | casualty_rate | -10.209790 pp | [-14.605633, -5.941882] pp |
| New − original, raw | unresolved_rate | -0.979021 pp | [-4.795473, +2.585695] pp |
| New − original, raw | utility | +0.323427 | [+0.193501, +0.453877] |
| New − original, masked | evacuated_rate | -0.559441 pp | [-4.804427, +3.852153] pp |
| New − original, masked | casualty_rate | -2.517483 pp | [-6.876165, +1.596421] pp |
| New − original, masked | unresolved_rate | +3.076923 pp | [+0.000000, +6.169597] pp |
| New − original, masked | utility | +0.021678 | [-0.095381, +0.148340] |
| Candidate − DAgger, raw | evacuated_rate | +0.000000 pp | [+0.000000, +0.000000] pp |
| Candidate − DAgger, raw | casualty_rate | +0.000000 pp | [+0.000000, +0.000000] pp |
| Candidate − DAgger, raw | unresolved_rate | +0.000000 pp | [+0.000000, +0.000000] pp |
| Candidate − DAgger, raw | utility | +0.000000 | [+0.000000, +0.000000] |

## Training and selection

Each active person gets one row containing the whole public map, every public person/status, fire,
exits and rules, with that person marked. The row includes their last action/outcome and four neighbour
statuses. No hidden state or route hint enters the encoder. Clef action order: east, north, south, stay, west.

T1 was chosen on eight TRAIN families (51 people). T1, T2 and T4 tied; the frozen name tie-break chose T1.
Token preflight checked 5,684 complete rows on 256 TRAIN geometries, including dense-fire variants:
maximum 628 tokens against a 16,384-token context. No input was truncated. GPU checks passed before fitting.

| Round | TRAIN families | Rows | Beta | Student-only agreement | Rows SHA256 |
|---|---:|---:|---:|---:|---|
| d0 | 256 | 27,979 | 1.0 | N/A (teacher only) | `39f566261d08ad3d90aa4b685e128424f2e0879b41ecd9fcd19a476bc26f53fa` |
| d1 | 256 | 37,677 | 0.5 | 67.73% | `f68d9f413603c526d278f997879c49938f31bc753322df847335a98dfbd8b7fa` |
| d2 | 256 | 42,005 | 0.25 | 61.40% | `b7ee6c9995fa0091f77e2b377a026b1c0d361035fc24632c419677d014eb7337` |
| d3 | 256 | 53,639 | 0.0 | 41.60% | `d783975988fd58a42ae81dce6ec5f439d060eeae7f491c9869edfcbd20c51c0f` |

Fits used one epoch of each deduplicated union, accumulation 8, LoRA LR 0.0002, head LR 0.0001 and
warmup ratio 0.03. The last union contained 161,300 rows. The 99% student-state agreement stop was not reached.
Selection retained the highest raw utility, with earlier checkpoints winning ties.

| Checkpoint, raw SELECT | Evacuated | Casualties | Unresolved | Utility | Wall / boundary |
|---|---:|---:|---:|---:|---:|
| s0 | 124/360 | 96 | 140 | -0.480556 | 0 / 0 |
| s1 | 225/360 | 57 | 78 | 0.145833 | 0 / 0 |
| s2 | 230/360 | 44 | 86 | 0.215278 | 0 / 0 |
| s3 | 185/360 | 55 | 120 | -0.041667 | 0 / 0 |

S2 was retained. Its raw SELECT utility was 0.215278 against T1’s 0.893056, a gap of 0.677778.
The frozen pre-PPO gate requires zero wall/boundary blocks and a utility gap at most 0.05. It failed,
so both small-network PPO and the three LLM PPO seeds were skipped. No PPO training ran.
Implemented PPO controls have CPU coverage: per-person ratios, legal masks in sampling and loss,
imitation decay 0.90, blocked reward -0.05, KL stop, every-group replay audit, and the +2 pp casualty guard
against the masked DAgger parent. Actor, critic, optimizer and RNG state support exact CPU next-update resume;
GPU continuation was not exercised.

## Preserved stop and recovery

Original round-3 fitting exited 0 but its completion verifier failed: first shuffled 8-row group loss
0.7414049749495462 versus an unrelated final 4-row group loss of 1.771141804754734. All 20,163 groups were finite.
Its FAIL record, log, weights and loss curve remain unchanged. The owner then instructed “Then finish ember2”.

A separately frozen, optimizer-free audit compared saved S2 and S3 on the same deterministically selected
4,096 existing TRAIN rows. Dropout was off; all five actions were scored without an external mask.
The strict finite-loss decrease passed: 0.940203726719648 to 0.936353487933297.
This small TRAIN diagnostic decrease is not a generalization result. It repairs the mismatched-group diagnostic;
it does not relabel the historical failed fit. No fit, collection or spent report arm was repeated. S3 was
subsequently evaluated on SELECT and did not replace S2.

Three original code blockers were repaired: evidence/attempt-root binding, frozen checkpoint and original
manifest-at-load binding, and the masked-parent selection rule. One fresh recovery review required an additional
controller/recovery-root equality check, mock audit entrypoint and real completion-gate tests, and accurate
original-checkpoint metadata. Important findings were fixed in one RED→GREEN pass; 41 focused tests passed.
The symptom, cause, repair and test for every recorded failure are in [FULL_VIEW_FAILURES](FULL_VIEW_FAILURES.md)
and the compact `failures.json`. No inherited test was weakened or changed.

## Identity and storage

Candidate and DAgger checkpoint: `/home/darshith/ember-ws/ember2-evidence/2026-11-14/full-view-20261114T133215Z-37025f/s2`.
Weights digest: `sha256:c55a773cb1af8151fb5c051de911d5e3fd2ee604288971f594c34dd593701adc`.
File-table SHA256: `706620c21a39820df95281264914d046667b11f36275641cd0f92f003e59e017`.
Base: `unsloth/Qwen3.5-2B@fffbc0d8711cf87df6d1f97938c5e50ef94aadc7`.
Controller source: `15b1e885089a39e2ae39ea83e5a9461c9f3db872`. Training sources and each producer result are in the registry.

Original plan SHA256: `389ff09bd8950d7fe9e0cfb77fd5b2f16f2482de8a3e2cfeb2c20b705749c8d6`.
Recovery plan SHA256: `3a7f5bfe3bcc584c77f4483f5e16b1d7f9e2bc18d1352649d5358d9268fe0483`.
Candidate-freeze SHA256: `d93ff510f54ae85b27e68d67993b6d59e28962f4ff49a45a19f6ef9e5322ebfe`.
Paired-intervals SHA256: `13107294f35b96273f7e136bc67bae8499ab92858517c4300ef1d2baf4bae5fa`.
The checkpoint registry and compact summary record all seven report hashes at full precision.

Upload pending: HF_TOKEN is absent from the environment. The package remains local; no full-view HF repository, revision or private-visibility check is claimed. The intended private destination is the HF_TOKEN account’s ember2-v1-models. The chat credential and saved logins were not used.

Local package `/home/darshith/ember-ws/ember2-evidence/2026-11-14/full-view-20261114T133215Z-37025f/package-recovery-01` contains 909 payload files and
3,007,520,712 bytes; manifest SHA256 `9fc70001cf9627215d149269f3c35a87ac6203bea2b40e0736a1f860321daa63`.
The previous stopped package is immutable. Final documentation/test receipts are indexed separately in the final ledger.

## Budget and limits

Experiments used 21.729982 GPU hours and 1.560027 conservatively charged
CPU hours through run completion. The original limits are 60 GPU hours, 16 CPU hours, 150 GiB evidence,
one GPU job at a time and $0 cloud spend. Initial recovery CPU included the old final ledger plus one
1,200-second preparation reservation. Later postprocessing, final tests and manual audit costs are added
once in the final ledger. Evidence root: `/home/darshith/ember-ws/ember2-evidence/2026-11-14/full-view-20261114T133215Z-37025f`.

There is one DAgger initialization seed. Teacher selection used only eight TRAIN families. The single
128-family report panel is spent; its intervals do not estimate training-seed variability. The first direct
original-V1 comparison measures the combined representation, history and DAgger changes, not their individual
causal contributions. The model still has high casualties and crowd blocks; masked superiority is unproven.
No sealed final evaluation, EMBERBench, hidden benchmark or reserved V1.1 data was used. No cloud compute,
public repository/model, tag, force push or development push to main occurred. Protected simulator,
generator, teacher, handoff and inherited test files stayed unchanged.

Human-view PR60 is merged at `4ef795a7f3efb2a720d3b2e63ad723dc8fa1cb02`, with its full history inherited here.
Human training, evidence and private `sankalphs/ember-v1-models` remain separate. Exact final-head local
verification, visibility, merge and shutdown receipts are recorded on PR1 and in the external final ledger.
