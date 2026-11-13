# Counterfactual RL implementation and experiment plan

The owner's Phase 3 request in this session explicitly authorizes implementation and real
counterfactual-assisted PPO experiments. It supersedes earlier stop-before-Phase-3 text.
Final evaluation remains sealed. No new SFT, V1.1, 1,200-scenario expansion,
`p1-18-dataset-v2`, hidden benchmark scenarios or internal benchmark simulator are inputs.

Dependency: Phase 2 is still running when this plan is written. Finish its frozen budget,
audits and independent reporting. Phase 3 GPU training must wait for its completed,
hash-verified DEV-selected checkpoint; never substitute a provisional latest iterate.
Keep original V1, Phase 1 and selected Phase 2 immutable.

## Bounded experiment

Use at most four additional hours of Phase 3 experiment process wall time, including
collection, counterfactual CPU work, GPU updates and comparisons; reserve 1,800 seconds
for final verification/evaluation. Maximum 256 newly collected on-policy episodes per
continuation arm, 32 updates of eight, three guarded attempts across the experiment.
Train two arms sequentially from the identical verified Phase 2 actor/critic/Adam state:
PPO-only and PPO+counterfactual. Give both the same prespecified episode/update budget;
if time requires an earlier stop, compare matched completed update counts. Report actual
interactions, counterfactual branches and optimizer steps separately. Never label branch
or preference data as fresh on-policy experience.

PPO settings initially inherit Phase 2: exact joint categorical probabilities, shared
outcome reward `(E-2C-.75U)/N`, gamma=1, GAE=.97, clipping=.2, joint KL target=.04,
entropy=.005, unchanged conservative LoRA/head learning rates and centralized critic.
Only the CF arm adds a bounded outcome-weighted pairwise auxiliary loss (coefficient .05)
at actual PPO update boundaries. The reference margin is the behavior policy's joint
log-probability difference at collection, not teacher likelihood or a teacher score.
All counterfactual examples retain an explicit offline label and collection-policy identity.

## Exact interventions and evidence

- Save scenario, evaluator seed, complete private state including RNG, current chain hash
  and public-observation digest. Restore every field exactly, independently replay the
  factual prefix, and test next-step record equality. Snapshot data never enters policies.
- Select at most four states per CF update by deterministic episode/step indexing, from
  states the current on-policy student actually visited. Do not select using outcomes.
- Candidates: factual sampled student action; student greedy joint action; T0–T4 decisions
  on deep copies of the same public observation; four seeded changes affecting at most
  two people. Deduplicate identical full maps; at most 12 candidates. Do not repair the
  online student action or reward movement. Record teacher failures/budget limits.
- Continue every intervention to the original episode terminal under the same frozen
  default T1 policy (version/config/source fingerprint recorded). T1 chooses continuation
  actions; actual independent-simulator E/C/U outcomes determine value. These are conditional
  action effects under T1 continuation, not individual causal blame or universal optimality.
- Deterministic fire: one exact branch. Stochastic fire: eight controlled replicates, one
  factual future stream and seven reproducible forks. Within each replicate all candidates
  start with identical private state and future RNG; compare every common-step ignition and
  draw count. Save full branch traces for independent restoration/replay.
- Store per-replicate E/C/U, terminal and restricted completion time, blocked/conflict counts,
  paired outcome utility advantage versus the factual student, standard error and bootstrap
  uncertainty. Record unresolved separately; completion at max_steps is right-censored.
  No time bonus is added to reward. CF records never use DEV states.
- Auxiliary training uses at most one conservative positive-versus-factual pair per state:
  positive mean measured utility, empirical paired lower interval above zero, no measured
  casualty regression. Record all alternatives, including harmed/tied/inconclusive results.
  Small-replicate intervals and candidate selection are exploratory; only independent
  reporting DEV can establish an overall statistically credible benefit.

## Curriculum and evaluation

Independent fresh namespace and topology isolation from all Phase 1/2 maps. Five public
curriculum stages: small navigation; doors/crowds; dynamic fire; competing exits; larger
populations and irregular geometry. Advance using a trailing TRAIN-only evacuation rate
threshold .55 and casualty rate at most .30, after at least 16 episodes per stage; force
coverage of the next stage after 32 TRAIN episodes so a weak stage cannot consume the run.
Both arms use the same rule, report any resulting curriculum difference. Keep at most
20 active people for memory-safe untruncated inference; larger than the simple stage.

Fix 32 selection DEV and 32 further reporting DEV families before training. Include all
stages and structural strata, with additional unseen geometry families. Selection every
64 episodes uses outcome utility and Phase 2's +2pp casualty tolerance. Reporting DEV
never influences training/curriculum/selection. Evaluate original V1, frozen selected
Phase 2, matched-budget PPO-only continuation, PPO+CF, and T0–T4. Teacher variants and
resource limits must be named exactly; budget-limited variants cannot be called full-budget
reference teachers. Use bounded teacher candidate generation where necessary, and evaluate
the frozen default teachers on reporting DEV where practical within the CPU budget.

Report paired family bootstrap 95% intervals for CF-minus-PPO evacuation/casualty/unresolved,
coordination, stays, repeated ineffective moves, latency, per-stage and per-family outcomes.
Do not claim statistically credible benefit unless the independent reporting evidence
supports it without unacceptable casualty regression. Freeze the best DEV-selected candidate
and its exact identity for a later authorized final evaluation; do not run that evaluation.

## Implementation tasks and tests

- [ ] Add `rl_counterfactual.py`: explicit snapshot/restore, public-only candidates,
  matched-fire full-horizon branches, paired estimates, replay verifier and TRAIN curriculum.
  Tests must first fail for missing implementation and then prove exact restoration,
  private-state isolation, same-state/future-fire pairing, tamper rejection and uncertainty.
- [ ] Add a separately logged optional auxiliary hook to `rl_ppo.optimize`, default off.
  Explicit GPU checks verify signed pairwise joint gradients and finite actual updates.
- [ ] Add a guarded resumable two-arm runner from verified Phase 2, fixed DEV and artifact
  provenance; CPU teachers receive public observations only. Freeze exact configuration
  before any new TRAIN teacher calls or Phase 3 GPU collection.
- [ ] Execute both arms and teacher comparisons, audit branch/online traces and actual
  optimizer states, save exact checkpoints and report benefits or failure honestly.
- [ ] Complete native Python 3.11 verification on the final clean committed head before
  pushing the single integrated draft PR #53; retain failures and full receipts.

Ledger: author intent and plan frozen before Phase 3 teacher TRAIN calls/rollouts. Phase 2
continues independently from source 90d8e67. Read its external resume ledger for live state.
Snapshot restoration and matched-fire branch tests now pass; legacy simulator and teacher
sources remain unchanged. The optional optimizer hook defaults off, preserving Phase 2.
Run the two arms in matched update rounds by swapping only trainable tensors/critic/Adam/RNG
on the single GPU; shared frozen base weights do not need repeated loading. This ensures
equal update counts if the wall budget ends early.
