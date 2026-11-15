# Full-view design and execution runbook

This is `anantamlab/ember2_v1`, a separate private project. The Python package remains `ember_v1`.
Human-view reports are inherited history; no human-view experiment runs here. Before collecting, freeze
`ember2-v1-full-view@1` with owner-prompt authorization, panels, arms, configurations and hashes.

## Observation and objectives

Each active person gets one row containing the entire strictly validated public observation: complete
map, every person's ID/position/status, fire, exits, rules, step and allowed actions. That person is marked.
Append only their last action/outcome and the four neighbour-cell statuses derived from that observation.
No simulator-private information, future state or teacher route hint enters the actor. Serialization is
lossless: map and fire are separate layers so burning exits and people do not hide terrain.

| Measured failure | Design response |
|---|---|
| F1: 81% blocked, about half wall/boundary | Explicit north/south/east/west cell types and last outcome |
| F2: few samples, crowd joint ratio | Per-person PPO ratio, fresh TRAIN families and three LLM seeds |
| F3: teacher states only | DAgger on student states, teacher labels for every active person |
| F4: no blocked signal | Per-person blocked reward -0.05, alongside terminal and potential rewards |
| F5: no action mask | Identical legal mask in sampling and PPO loss; report raw and masked separately |

The full-map T1–T4 retained registry teachers are realizable from this input. On a TRAIN-only reference
panel, choose highest utility, then lower casualty rate, then teacher name; freeze its exact registry
identity before collection. That reference panel is excluded from all subsequent splits. Do not modify
teachers, generators, simulator or hashed handoff files.

## Preconditions and phase order

1. CPU mock tests for every new module; all inherited human-view tests retained. Check full-row token
   lengths on TRAIN geometry with the pinned tokenizer before freezing. No truncation. Use compact
   lossless layers if necessary; record encoding and maximum tokens in the plan.
2. Freeze the write-once plan under the external evidence root, hash it and post summary/hash to PR #1.
3. CPU selection references: chosen teacher and random legal moves, same frozen selection families.
4. Optimizer-free LLM GPU check: actual Clef items, option order east/north/south/stay/west, finite normalized
   scores and nonzero LoRA/head gradients; exact actor save/reload and preserved-group replay checks.
   Require PASS before fitting.
5. DAgger round 0: 256 new TRAIN families, beta 1, fresh adapter from the pinned base. One epoch, accumulation
   8, LoRA LR 0.0002, head LR 0.0001, warmup 0.03. Up to three further 256-family rounds at beta .5/.25/0,
   training on the union. Stop when agreement on student-executed states reaches .99; teacher-executed
   states do not count toward that gate. Keep the best selection checkpoint if a fit regresses.
6. Before RL: raw wall and boundary blocks must be zero and raw utility within .05 of the teacher on
   selection. On failure, preserve the SFT result, skip both PPO arms and proceed to candidate freeze/report.
7. Before any PPO: tested checkpoints save actor, critic, optimizer, config, update, selection history,
   sampling RNG, Python RNG, NumPy RNG and Torch CPU/CUDA RNG. Restore these before the next update; require
   exact CPU next-update equivalence. A started attempt is never restarted; resume support is tested but
   not used to repeat an authorized attempt.
8. Small-network PPO and three LLM PPO seeds, all using fresh TRAIN, per-person GAE/ratios, legal mask,
   decaying imitation anchor, KL early stop and every-group replay audit before optimizer steps. Freeze
   sample counts and seeds in the plan. Select checkpoints per seed using selection only; casualty rate
   cannot exceed the immutable DAgger parent by .02 and utility must improve. Earlier checkpoints win
   ties; across seeds, highest guarded utility then smaller seed. Keep DAgger if no seed qualifies.
9. Freeze checkpoint file hashes before report. Exactly one invocation per arm on a new shared 128-family
   report panel: candidate raw, candidate masked, DAgger raw, chosen teacher, random legal moves, original
   V1 raw and original V1 masked. Distinct arms remain separate invocations even if candidate equals DAgger.
   Original V1 uses its original encoding and pinned registry checkpoint; do not reinterpret it as a new row.
10. Compute paired family bootstrap 95% percentile intervals, 5,000 resamples, for candidate minus original
    V1 and candidate minus DAgger: evacuation, casualties, unresolved and utility. No report reselection.

## Evidence, budgets and stops

Evidence root: `/home/darshith/ember-ws/ember2-evidence/<utc-date>/full-view-<utc>-<id>/`, outside every
checkout and human-view root. Create attempt markers before launch using exclusive creation. Rows,
weights and raw logs stay there, with SHA256 manifests. Commit compact summaries only.

All panels are symmetry-invariant topology-disjoint. Preflight the entire declared collection/PPO schedule
against selection/report/reference geometry before training; recheck datasets and hashes when read. No
reserved V1.1 data, sealed FINAL, EMBERBench or hidden benchmark access. No Modal or cloud compute/spend.

One GPU process at a time. Limits: 60 GPU hours, 16 CPU hours, 150 GiB evidence, $0 cloud. Persistent
supervisor writes immutable command/start/result records, enforces remaining wall budgets and verifies
completion before advancing. Stop a phase on nonfinite loss, replay above the unchanged tolerance,
topology overlap, hash mismatch or budget exhaustion. Preserve failures; a corrected new attempt needs
a CPU regression where possible and a clean verified push before launch. Never rerun an existing attempt.

Only clean locally verified heads may be pushed (`EMBER_VERIFY_REQUIRE=3.11`); post every pushed-head
receipt to PR #1. Final full tier plus torch PPO tests must pass with zero failures/skips in PPO tests.
Manifest-package evidence, create only a private HF `ember2-v1-models` in the `HF_TOKEN` account, verify
privacy, upload then download/hash every file. Read credentials only from `HF_TOKEN`. If absent or unable
to write, keep the package local and state "upload pending". Never use a pasted or stored login token.

Write report, model card, checkpoint registry, CURRENT_STATE and deferred version-2 work to match exact
evidence identities. Merge PR #1 with a merge commit only after every owner gate passes, with clean tree
and no running process. No direct main development push, force push, release tag or public announcement.

## Pre-report runtime repair

The original-V1 control bridge must use `ArmPolicy.decide(observation).actions`. A control-only repair
before any candidate freeze or report may use a frozen, append-only runtime source supplement to this
same owner-authorized plan. The original plan and all learning code, panels, arms, seeds, configurations
and budgets stay unchanged. The supplement cannot authorize a report retry.

An independent, CPU-tested boundary monitor may hold only the old controller while its existing child
finishes under the original source and deadline. It checks the real Linux exit status and the original
completion gates, records the completed phase and then retires the held controller. Never stop or resume
that child to repeat work. A failed administrative attachment stays preserved under its own name.

After normal completion, verify and push the corrected clean head. Freeze the source supplement with
the unchanged parent-plan hash, exact owner authorization, old/new source IDs, archive hash, controlled
stop hash, every completed result/log/output hash, extra CPU charges and a new continuation name. Post
its hash before starting another phase. Only controller, bookkeeping, tests and the exact original-policy
API correction may differ; actor, observer, reward, teacher, generator and simulator code cannot change.

The new controller validates that supplement, confirms the old controller is terminal and reconstructs
selection state by reading declared PASS records. It re-verifies outputs without launching their commands
or charging their phase time again. An undeclared started attempt, changed hash, failed gate or changed
supplement stops the continuation. Every future phase keeps its actual source ID; the final evidence
retains both archives and the controlled stop. Final full-tier and torch merge gates still apply.
