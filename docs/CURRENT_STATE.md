# EMBER V1 — current state (authoritative)

Status date: 2026-11-13. This file is the single entry point for new developers and coding agents.
If another file disagrees with this one about the current state, this file wins. Other files keep their historical value.

## 1. Canonical repository

- **`https://github.com/anantamlab/ember_v1` is the only development repository.** All code, experiments, research PRs, documentation and registry updates start here.
- `anantamlab/ember-v1-research-handoff` is a read-only historical backup. Its only commit, `41821c917b63956a4760a6842dcd35bbc604500f`, is part of this repository's history. It was a fast-forward from PR #53 head `1bccff43aa25569d32d319fab3a12517fd4bb5d4`. Do not push to it.
- Do not use old workstation scratch directories as a source of code. Large weights and raw evidence stay in external storage (§5).

## 2. What EMBER V1 is

EMBER V1 is a standalone fire-evacuation decision policy. One model decides a move for every active person at each simulator step.

- **Student model:** frozen `unsloth/Qwen3.5-2B` backbone with a LoRA adapter and a float32 Clef joint decision head (five actions per person).
- **Supervised V1:** trained by SFT on traces from five rule-based teachers (T0–T4), collected in an independent CPU simulator. Artifact `p1-18-v1-train@1`.
- **RL phases:** online PPO (R20, R21) and counterfactual-assisted PPO (R22). All start from V1.
- **Independence:** the simulator and scenario generators are independent of EMBERBench. No EMBERBench private scenario or hidden simulator code is in this repository or in any experiment input.

## 3. Where the code is

| Area | Location |
|---|---|
| Independent simulator (fire, movement, scenarios) | `src/ember_v1/sim/`, scenario generators in `src/ember_v1/gen/`, `rl_online_scenarios.py`, `rl_cf_scenarios.py` |
| Teachers T0–T4, evaluation, selection | `src/ember_v1/teacher/` (`t0.py` … `t4_v12.py`, `*_eval.py`, `selection.py`) |
| Five-teacher collection, multi-teacher dataset, counterfactual labels | `train_collect.py`, `multiteacher*.py`, `train_dataset.py`, `counterfactual.py`, `train_counterfactual.py` |
| Observation contract, encoding, chunked joint decisions | `contract.py`, `encoding.py`, `chunk_joint.py`, `sft_prep.py`, `token_budget.py` |
| Student SFT, inference, evaluation | `student_train.py`, `student_run.py`, `student_objective.py`, `student_eval_protocol.py`, `agent.py` |
| Action masking (shield) and coordination arms | `shield.py`, `policy_arms.py` |
| RL environment and rollouts | `rl_env.py` |
| PPO actor, centralized critic, advantages, optimizer | `rl_ppo.py`, `rl_run.py` (Phase 1) |
| Online PPO with checkpoint resume and DEV selection | `rl_online.py` (Phase 2) |
| Counterfactual snapshots, matched-fire branches, auxiliary loss | `rl_counterfactual.py`, `rl_cf_training.py`, `rl_cf_run.py`, `rl_cf_evidence.py` (Phase 3) |
| Independent replay and verification | `rl_verify.py`, `rl_cf_verify.py`, `experiments/rl_*.py` |
| Artifact store (private Hugging Face) | `hfstore.py`, `data/hf/registry.json` |
| Workstation packaging | `workstation.py`, `ops/workstation/`, `docs/RUN19_WORKSTATION.md` |
| Local test gate | `scripts/verify_local.py`, `AGENTS.md` |
| Checkpoint registry and verifier | `data/rl/checkpoint_registry.json`, `scripts/verify_checkpoint.py` |

Frozen RL configurations: `data/rl/r20-ppo-pilot@1.json`, `r21-online-ppo@1.json`, `r22-counterfactual-ppo@1.json`. Compact evidence (manifests, identities, analyses, curves): `data/rl/evidence/`.

## 4. Completed experiments and verified findings

| Run | What ran | Result |
|---|---|---|
| SFT V1 (`p1-18-v1-train@1`) | Five-teacher SFT on Modal | Trained checkpoint, verified on private HF |
| Run 19 | RTX 5090 workstation reproduction and policy-arm comparison | V1 reproduced, with 1 of 1,538 decisions over its near-tie threshold across GPUs. On 148 matched cases: raw 15.16%, mask 23.46%, coordination 27.37% evacuated. Untrained base raw: 2.54%. |
| R20 Phase 1 | 64 episodes, 1,946 transitions, 141 Adam steps | **Real PPO weight updates verified** (finite nonzero LoRA/head gradients, exact reload). No DEV evacuation gain: 55.81% vs 55.81% on 16 families. |
| R21 Phase 2 | 1,024 episodes, 38,950 transitions, 2,489 Adam steps | **Update 96 tied raw V1**: 174/358 (48.60%) each. PPO − V1 evacuated 0.00 pp [−2.89, +3.36]; casualties −0.84 pp [−3.76, +2.06]. |
| R22 Phase 3 | 216 episodes per arm, PPO-only vs PPO+CF, from R21 update 96 | **PPO+CF reduced casualties vs PPO-only**: 104 vs 114 of 473; −2.11 pp [−3.96, −0.39]. **No established evacuation gain**: 197 vs 193; +0.85 pp [−1.86, +3.48]. Original V1 also had 197. |

Further verified facts:

- **R22 selection:** the predefined selection-DEV rule froze **PPO-only update 16** (utility −0.087691 vs −0.115468 for CF update 24). Reporting DEV did not change that choice.
- **Rule-based execution still wins over raw learned policies.** R21 reporting DEV: V1 + mask 56.42%, V1 + coordinator 60.89%, raw V1 48.60%. R22 reporting DEV: V1 + mask 54.12%, V1 + coordinator 60.68%, best raw learned arm 41.65%. These gains come from external rules, not from RL.
- **Teachers are far stronger.** T0–T4 evacuated 87.74–89.22% on R22's reporting DEV.
- **Main failure mode:** more than 81% of attempted moves stay blocked in both R22 learned arms; about half hit a wall or boundary. Counterfactual feedback was sparse: 26 of 108 verified studies gave a usable pair.
- **Evaluation panels differ between runs.** Compare arms only inside one run's table. Each run used one training seed. Intervals are paired family bootstraps (5,000 replicates for R21/R22, 2,000 for R20) and do not cover training-seed variance.

Full detail: [R20 report](R20_PPO_PHASE1_REPORT.md), [R21 report](R21_PPO_PHASE2_REPORT.md), [R22 report](R22_COUNTERFACTUAL_REPORT.md), [Run 19 report](../handoff/r19/EMBER_R19_FINAL_REPORT.md), [handoff](../handoff/HANDOFF.md).

## 5. Checkpoints and external artifacts

Git holds code and compact evidence only. Weights, trainer states, raw trajectories, simulator-private snapshots and full logs are external. The machine-readable registry is [`data/rl/checkpoint_registry.json`](../data/rl/checkpoint_registry.json).

| ID | Checkpoint | Parent | Identity | Storage | Status (2026-11-13) |
|---|---|---|---|---|---|
| `v1` | `p1-18-v1-train@1` | — | HF revision `1fc405f723718e308c8a22b137fee4a025086fde` | private HF `sankalphs/ember-v1-models`, path `v1/p1-18-v1-train@1` | **Available.** `hfstore verify` PASS, 14 files, 0 problems |
| `r20` | R20 attempt 6 | `v1` | `f6b56137acc791bd1ed0862547a1dd3ca2ef81f336af6c969947840b46c66374` | workstation `r20-ppo-pilot@1/attempt-6/checkpoint` | Workstation only |
| `r21-u96` | R21 update 96 | `r20` | `901e22e8be0af3f4ff7002b9041d16e4a0dacd8c602a79b66121429507275120` | workstation `r21-online-ppo@1/attempt-3/checkpoints/update-0096` | Workstation only |
| `r22-ppo-u16` | R22 PPO-only update 16 (frozen candidate) | `r21-u96` | `df5afa5a6d09031f5375503a8816eeeec26ee4faeaf4ee0ed040df821524154d` | workstation `r22-counterfactual-ppo@1/attempt-2/ppo/checkpoints/update-0016` | Workstation only |
| `r22-cf-u24` | R22 PPO+CF update 24 | `r21-u96` | `05d665f4f7acedee57e2fa2acd9faa390df3f65b11de2719058cfb4ab8a00c72` | workstation `r22-counterfactual-ppo@1/attempt-2/cf/checkpoints/update-0024` | Workstation only |

- Workstation paths are relative to `/home/darshith/ember-ws/rl-evidence` (WSL). The verified Windows mirror is `C:/Users/Admin/Documents/ember-evidence`, with each R21/R22 selection under `selected_checkpoint/`.
- **The R20–R22 checkpoints are not on Hugging Face.** They are reachable only through the RTX 5090 workstation or its Windows mirror. A fresh clone cannot replay them on GPU without that access. Uploading them to `sankalphs/ember-v1-models` (with `hfstore publish`) is an open task that needs workstation access.
- Base model: `unsloth/Qwen3.5-2B@fffbc0d8711cf87df6d1f97938c5e50ef94aadc7` (public). Tokenizer files ship in each actor directory.
- Access: V1 needs read access to the private `sankalphs` HF repos (`hf auth login`). Run HF commands with `env -u HF_TOKEN` when a stale `HF_TOKEN` is set.

Commands:

```bash
python scripts/verify_checkpoint.py --list                                      # registry summary
env -u HF_TOKEN PYTHONPATH=src python -m ember_v1.hfstore verify p1-18-v1-train@1   # V1 remote identity
env -u HF_TOKEN PYTHONPATH=src python -m ember_v1.hfstore fetch p1-18-v1-train@1 --dest <empty-dir>
python scripts/verify_checkpoint.py r22-ppo-u16 <copy>/r22-counterfactual-ppo@1/attempt-2/ppo/checkpoints/update-0016
```

`verify_checkpoint.py` hashes a local copy against the tracked identity record. To load an actor, pass the actor directory to `ember_v1.rl_ppo.V1Actor(cfg, checkpoint=...)` in the pinned GPU environment. To resume, use each runner's `--resume` option (see each report's reproduction section). Parameter-digest checks need the GPU replay environment.

## 6. Install, test, reproduce

```bash
python -m pip install -e ".[test]"                               # Python 3.11
python scripts/verify_local.py run --python <path-to-python3.11>              # default tier, clean committed tree
python scripts/verify_local.py run --with-slow --python <path-to-python3.11>  # full tier (see AGENTS.md)
python scripts/verify_local.py status
```

- Test tiers: seven slow end-to-end files are skipped by default and run with `--with-slow` (`AGENTS.md`, "Test tiers").
- Reference result (complete suite, before tiers): R22 report head `1bccff4` had **1,620 passed, 2 skipped, 10 xfailed** on Python 3.11 ([receipt](../handoff/verification/20261113T153655Z-1bccff43aa25-py3.11.json)).
- GPU work used WSL Python 3.11 with Torch 2.8.0+cu128, Transformers 5.2.0, Unsloth 2026.10.11 and Triton 3.4.0. Set Torch CPU threads to 4 for R22 fresh-load replay.
- CPU replay of existing evidence: `python -m ember_v1.rl_verify <run-copy>` (R20/R21), `python -m ember_v1.rl_cf_verify <run-copy>` (R22). Use scratch copies so derived files cannot change archived manifests.
- Historical workstation scripts in `handoff/operations/` and `handoff/r19/` contain old paths and process IDs. **They are evidence only. Do not run them unchanged.** The two Run 19 authorization patches do not authorize any new experiment.

## 7. Boundaries that stay in force

- **Final held-out evaluation is sealed.** No run has opened it. Opening it needs a separate owner-authorized plan.
- **V1.1 data is reserved.** The 1,200-scenario expansion (`p1-18-expand@1`) and `p1-18-dataset-v2` (with `p1-18-sft-v2*` and `p1-18-tokens-v2f@2`) are for V1.1 only. No RL phase used them. Do not use them for more V1 training.
- No PPO, SFT, counterfactual run or final evaluation is authorized now. Each new run needs a frozen plan and owner authorization (see `AGENTS.md`).
- EMBERBench private scenarios and its hidden simulator stay out of this repository.

## 8. Historical records

These files describe earlier phases. Do not read them as the current state.

- `PHASE_CHECKPOINT.json`: P0A-era status ("trained_checkpoints": "none", P0B next). It is superseded by this file.
- `README.md` sections on P0B on Modal and padding gates: P0A/P0B design notes.
- `docs/P1_*`, `docs/P0B_*`, `docs/HLD.md`: phase plans, protocols and results in time order.
- `handoff/`: the hashed archive (`handoff/FILES.sha256`, 65 files). Keep its bytes unchanged.

## 9. Recommended next milestone

For RL work, read **[RL_RESULTS_HANDOFF.md](RL_RESULTS_HANDOFF.md)**: the RL system as built and every R19–R22 metric.

Diagnose the raw policy's movement failures before more PPO of the same kind. Target the wall/boundary blocks and repeated ineffective moves (R21/R22 diagnostics). Test whether a representation or objective change fixes them on selection DEV, under a new frozen plan. Before that, upload the R20–R22 selected checkpoints to private HF so they no longer depend on one workstation.
