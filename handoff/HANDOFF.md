# EMBER V1 research handoff: November 12–13, 2026

This private repository preserves the EMBER V1 research branch through [draft PR #53](https://github.com/anantamlab/ember_v1/pull/53), commit `1bccff43aa25569d32d319fab3a12517fd4bb5d4`, with a separate handoff commit. The last 36 hours of work covered the RTX 5090 workstation validation, Run 19 V1 policy comparisons, the first online PPO pilot, a longer PPO continuation, and a matched counterfactual-assisted experiment. The source history through the PR head is intact. Nothing in this handoff changes the original PR, starts training, or opens the sealed final evaluation.

Start with the [Phase 3 report](../docs/R22_COUNTERFACTUAL_REPORT.md) for the result and checkpoint decision. The [Phase 2 report](../docs/R21_PPO_PHASE2_REPORT.md) explains why longer PPO alone did not improve evacuation, and the [Phase 1 report](../docs/R20_PPO_PHASE1_REPORT.md) documents the first verified PPO update from the actual trained V1 model. The [workstation report](r19/EMBER_R19_FINAL_REPORT.md) and [untrained base control](r19/EMBER_R19_BASE_CONTROL.md) preserve Run 19 context. Read Run 19's prospective training suggestions as historical; the 1,200-scenario expansion, `p1-18-dataset-v2`, and V1.1 remained excluded from all three RL phases.

| Work | Measured result |
|---|---|
| Run 19 workstation and controls | Reproduced trained V1 on the RTX 5090 with one recorded cross-machine near-tie exception. On 148 matched cases, V1 evacuated 15.16% raw, 23.46% with the mask, and 27.37% with coordination. The untrained raw control evacuated 2.54%. |
| R20 / Phase 1 | 64 on-policy episodes, 1,946 transitions, 141 Adam steps. Verified real LoRA and Clef updates; no DEV evacuation gain. |
| R21 / Phase 2 | 1,024 episodes, 38,950 transitions, 2,489 Adam steps. Update 96 tied original V1 at 174/358 evacuated. Persistent wall/boundary blocks and repeated ineffective moves limited the raw policy. |
| R22 / Phase 3 | 216 committed episodes per arm. PPO+CF evacuated 197/473 versus 193/473 for PPO-only and 197/473 for original V1. Casualties were 104 versus 114 for PPO-only, a paired difference of −2.11 pp [−3.96, −0.39]. Evacuation difference was +0.85 pp [−1.86, +3.48]. The separate DEV selector froze PPO-only update 16. |

These panels differ across runs; compare arms within a row. Mask and coordinator gains belong to external execution rules, not learned RL. T0–T4 evacuated roughly 88–89% on R22's reporting DEV. The learned policies remain far below those references. R22 used 108 verified counterfactual studies, of which 26 yielded eligible pairs under fixed T1 continuation. It applied 418 auxiliary optimizer steps; the selected CF actor retains 372.

## What is in this repository

- `src/ember_v1/rl_env.py`, `rl_ppo.py`, and `rl_run.py`: public-observation simulator wrapper, simultaneous actions, differentiable Clef policy, centralized critic, joint likelihood ratios and complete-episode advantages.
- `src/ember_v1/rl_online.py` and `rl_online_scenarios.py`: bounded online training, fresh topology families, checkpointing, and isolated DEV selection and reporting.
- `src/ember_v1/rl_counterfactual.py` and `rl_cf_*.py`: exact private snapshots, matched fire branches, public action proposals, conservative offline pair loss alongside PPO, shared TRAIN curriculum, resume ancestry, and independent replay.
- `data/rl/`, `experiments/rl_*.py`, and `docs/R2*.md`: frozen configurations, verification and analysis code, compact evidence, curves and reports. The [handoff file inventory](FILES.sha256) records the extra files preserved here.
- [Run 19](r19/): local summaries, workstation/control reports, small analysis scripts, and the two local-only authorization patches. `f8bc7c873406cacdfa9fa97376d1657807b731bd` and `a3a9bf32763b98486a2ab95467ae4599ab787349` were sibling commits based on `b91654c1cb1bcb24f24aca01815e2ec5098cc625`; the latter was the executed Run 19 baseline. The patches are historical records, not new authorization or changes applied to the current experiment configuration.
- [Operations archive](operations/): selected workstation launch, failure diagnosis, audit, analysis and copy verification scripts and resume notes. They contain original machine paths and some stale process IDs. Use them to reconstruct how the runs were executed; do not run them unchanged. [Verification receipts and logs](verification/) record the tested R20, R21 and R22 report heads.

The T3 review covered [Train V1 with PPO in Evacuation Simulator](t3-thread://v2/4b966427-8587-45b7-b6d4-3d12035f1788/bec9b47b-790f-4af8-8f24-61dd5f08fb1d) and [Run EMBER V1 RTX 5090 Experiments](t3-thread://v2/4b966427-8587-45b7-b6d4-3d12035f1788/9f21da93-bee1-4c8e-b564-80db124f3c12). This handoff carries the implementation and run decisions, not raw conversation exports.

## Exact checkpoints and artifact access

Canonical WSL evidence root: `/home/darshith/ember-ws/rl-evidence`. Verified Windows copies: `C:/Users/Admin/Documents/ember-evidence`. Paths in the following table are relative to the canonical root; reporting manifests are tracked under `data/rl/evidence`.

| Checkpoint | Canonical directory | Actor digest (SHA-256) |
|---|---|---|
| R20 pilot | `r20-ppo-pilot@1/attempt-6/checkpoint` | `f6b56137acc791bd1ed0862547a1dd3ca2ef81f336af6c969947840b46c66374` |
| R21 update 96 | `r21-online-ppo@1/attempt-3/checkpoints/update-0096` | `901e22e8be0af3f4ff7002b9041d16e4a0dacd8c602a79b66121429507275120` |
| R22 frozen PPO-only update 16 | `r22-counterfactual-ppo@1/attempt-2/ppo/checkpoints/update-0016` | `df5afa5a6d09031f5375503a8816eeeec26ee4faeaf4ee0ed040df821524154d` |
| R22 selected PPO+CF update 24 | `r22-counterfactual-ppo@1/attempt-2/cf/checkpoints/update-0024` | `05d665f4f7acedee57e2fa2acd9faa390df3f65b11de2719058cfb4ab8a00c72` |

R20's trainer is a sibling of `checkpoint/`. For R21/R22, each checkpoint directory contains `actor/` and `trainer.pt`; the Windows mirror uses `selected_checkpoint/` under the run or arm directory. The original trained V1 is `p1-18-v1-train@1` at model artifact revision `1fc405f723718e308c8a22b137fee4a025086fde`, with the pinned `unsloth/Qwen3.5-2B@fffbc0d8711cf87df6d1f97938c5e50ef94aadc7` base. The Run 19 reports record the private artifact repository and verified revisions for workstation and control outputs.

Git carries source and compact evidence. Model weights, trainer binaries, raw trajectories, simulator-private snapshots, full run logs, virtual environments, credential files and machine caches remain outside Git. The GitHub repo alone cannot run the GPU checkpoint replays: the next operator needs access to the canonical evidence and pinned model cache. The [verified counterfactual dataset index](../data/rl/evidence/r22-counterfactual-ppo@1/verified_counterfactual_dataset.json) points to study files across both R22 attempts. R22's [manifest](../data/rl/evidence/r22-counterfactual-ppo@1/EVIDENCE_MANIFEST.json) has SHA-256 `c88ff645362a96366ad5f386d7e48211b5f32cc8df37c476424c73b0b07d7e17`.

## How to pick up the work

1. Read [AGENTS.md](../AGENTS.md). Use native Windows Python 3.11 for the full CPU gate. Install the test extras with `python -m pip install -e ".[test]"`, then run `python scripts/verify_local.py run --python <path-to-python-3.11>` on a clean committed tree. The original final R22 head had 1,620 passed, 2 skipped and 10 expected failures.
2. Use the pinned WSL Python 3.11 GPU environment for model replays. The runs used Torch 2.8.0+cu128, Transformers 5.2.0 and Unsloth 2026.10.11. Set Torch's CPU thread count to four for R22 fresh-load reporting replay, as in the trainer. Read each run's execution record and keep the original actor/optimizer ancestry intact.
3. Verify existing evidence before training anything. CPU replay entry points are `python -m ember_v1.rl_verify <run-copy>` for R20/R21 and `python -m ember_v1.rl_cf_verify <run-copy>` for R22. Work on scratch evidence copies so derived audit files cannot change the canonical manifest. Run the exact-checkpoint GPU audit only with the appropriate verified artifacts and resource budget.
4. Preserve failed attempts. R22 attempt 1 hit a donated-buffer backward error on the first eligible pair; the fix uses separate single-use backward graphs, and attempt 2 resumed four verified rounds. The first completion audit also failed its greedy replay under a different thread setting; the corrected audit reproduced 2,759 reporting decisions exactly. Both failures and their proofs are documented.
5. The completed experiment's scheduler was deleted. **Final evaluation is sealed.** No training, benchmark access, merge, or final evaluation is pending from these runs. If research continues, investigate the wall and boundary errors and the sparse, continuation-dependent CF feedback before committing to more of the same PPO. Any new TRAIN or final evaluation needs its own authorized plan.
