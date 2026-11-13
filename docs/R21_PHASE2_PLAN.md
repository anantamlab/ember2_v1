# Phase 2 online PPO plan — frozen before new rollouts

Owner authorization: the Phase 2 request in this session explicitly authorizes substantial
online PPO, bounded diagnosis/retries, checkpoint selection and the five-arm comparison.
Continue the verified `r20-ppo-pilot@1/attempt-6` actor, critic and Adam state; preserve both
that parent and the original `p1-18-v1-train@1`. No SFT or expansion artifacts are inputs.

Budget: at most 21,600 seconds of experiment process wall time across attempts, at most
1,024 new complete on-policy TRAIN episodes (128 updates of eight), four attempts including
resumes. Reserve the final 1,800 seconds for checkpoint verification and comparisons.
Stop at an update boundary if the remaining budget cannot support another update.

Phase 1 evidence: likelihood replay and exact saving passed after documented fixes.
Evacuation did not improve; casualties increased. Mean per-person entropy was 0.349 nats,
blocked movement exceeded 54%, and only 64 small-room episodes were trained. These support
testing broader experience and exploration, not claiming a discovered numerical instability.
Use gamma=1 for the finite-horizon outcome objective: discounting a negative terminal
outcome otherwise rewards deferring it. GAE lambda=.97, entropy=.005, unchanged joint
clipping and conservative V1 learning rates; no reference KL anchor to weak SFT behavior.
Terminal reward remains `(E-2C-.75U)/N`, public potential shaping eta=.1, terminal Phi=0.
No movement bonus or universal stay penalty. One prespecified configuration; only change
it if measured instability requires a bounded, separately recorded correction.

Independent procedural generator: eight structural strata, rotated exits, sparse/dense
populations 4–20, rooms/partitions/doors/corridors/pillars/irregular geometry, inaccessible
regions, 1–4 exits, deterministic/stochastic fire and 1–3 ignitions. Geometry seeds are
split-specific. Verify D4 topology hashes across all new TRAIN, selection DEV and reporting
DEV, and against Phase 1 maps. No benchmark or historical dataset reads.

Fix 32 selection DEV and 32 reporting DEV families before training. Evaluate selection DEV
every 16 updates. Select the trained checkpoint maximizing `(E-2C-.75U)/N`, subject to
casualty rate no more than two percentage points above original raw V1 on selection DEV;
if none qualify save the highest-utility trained checkpoint and report the safety failure.
Reporting DEV never influences updates, hyperparameters or selection. Compare raw V1,
V1 mask, V1 coordinator, Phase 1 raw, selected PPO raw on reporting DEV. All primary arms
use argmax; learned PPO has no masking/repair. Bootstrap paired independent families for
uncertainty; explicitly limit conclusions to this generated distribution.

Implementation tasks: (1) test generator coverage/isolation and selection, (2) add resumable
continuation with exact actor/critic/optimizer/RNG checkpoints, immutable provenance and
five-arm evaluation, (3) execute GPU training, (4) replay environment and advantage evidence,
verify save/reload, summarize curves/latency/failure modes, full CPU tests then update #53.
Keep a durable execution ledger here and external evidence. Stop before Phase 3.

Ledger: baseline branch/head verified at 9f12f1b1ea1a0190c3a59caf302e88ac9e9466d7; PR #53
and latest comments read. Existing integrated research branch retained. No Phase 2 rollouts
have started. Final whole-branch fresh-context review follows execution.

Attempt 1: stopped before GPU load/rollouts when a rare random exit landed in a tiny
isolated component. Full 1,024-family regression reproduced it. Corrected population
capacity by placing one exit in the largest boundary-connected public component only
when fewer than seven non-exit cells are reachable. No outcome rejection or reward change.
Attempt 2 uses its own fresh TRAIN namespace and the same fixed DEV geometry generator.

Attempt 2: also stopped before GPU load/rollouts. D4 isolation caught two repeated pillar
geometries, including a TRAIN/selection collision. Added independently sampled interior
obstacles to every structural stratum and a full namespace/cross-split regression for all
four authorized attempts. No collided scenario was used for training or evaluation.
Fresh-context review found a soft-only deadline: added a process alarm and refusal to
resume attempts with missing durable end records. External launcher also enforces remaining
aggregate time. Both findings fixed before actual TRAIN collection.

CPU verification during training exposed an existing supervisor race: a worker with a
rejected code fingerprint could flush its entire trajectory before the parent drained
messages. The report then failed replay instead of recording the code-drift failure.
A deterministic fixture regression failed before the fix and passed after rejecting that
worker's untrusted decisions. Original test unchanged; no real teacher collection ran.
This deliberately changes the supervisor and its importing code fingerprints. The frozen
P1-17 plan remains byte-for-byte unchanged. Its rebuild test now pins both exact historical
and corrected fingerprints, requires the live plan hash to differ, and compares every
other plan field after substituting only those three explicit revisions. This preserves
historical provenance and detects future unexpected drift rather than authorizing reuse
of an old plan with changed code. The failed full-suite logs remain available.
