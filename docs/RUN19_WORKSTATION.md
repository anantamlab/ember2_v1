# Run 19: laptop → RTX 5090 workstation

The laptop is the only development machine. The workstation (RTX 5090, i9) only executes. It never edits code:

1. It checks out an exact pushed commit (40-hex, never a branch).
2. It downloads exact private Hugging Face revisions.
3. It runs one AUTHORIZED run config.
4. It uploads the result to an immutable path.

The laptop then fetches and reviews the result.

## 1. Storage (private Hugging Face repos, namespace `sankalphs`)

| Repo | Type | Contents |
|---|---|---|
| `sankalphs/ember-v1-datasets` | dataset | teacher traces, SFT/token sets, calibration, counterfactuals, ops evidence |
| `sankalphs/ember-v1-models` | model | V1 checkpoint `v1/p1-18-v1-train@1` (LoRA r16 + Clef head + tokenizer + identity) |
| `sankalphs/ember-v1-experiments` | dataset | V1 eval records, V2 staging, workstation results |

Rules (`src/ember_v1/hfstore.py`, `ember-hfstore@1.0.0`):

- Every artifact is a versioned folder (`…/<name>@<n>`) with `ARTIFACT.json`. This file holds per-file size, sha256 and git sha1, the source commit, provenance, checks and status.
- `VERIFIED.json` exists only after a remote verification passes. The remote verification has three parts:
  - The tree listing must match: file set, sizes and LFS sha256 / blob ids.
  - All critical files and a seeded sample are downloaded and content-hashed.
  - The model checkpoint gets a full download.
- A path is immutable. A publish with a different manifest is refused. A publish with the same manifest is an idempotent no-op.
- Uploads resume: run the same `publish-plan` command again.
- A file that looks like a credential (`hf_…`, `ghp_…`, `sk-…`, Modal keys) stops the publish.
- `data/hf/registry.json` records every artifact: manifest commit, verified commit and manifest sha256. The workstation always downloads at the **manifest commit** and re-checks `manifest_sha256`.

Laptop commands (tools venv with `huggingface_hub[hf_xet]`, `unset HF_TOKEN` first so the `hf auth login` token is used):

```bash
python -m ember_v1.hfstore status
python -m ember_v1.hfstore publish-plan --plan data/hf/publish_plan.json --evidence-root D:/Coding/ember-evidence
python -m ember_v1.hfstore verify --repo models --path v1/p1-18-v1-train@1 --revision <manifest commit> --full
```

### Not uploaded (by decision)

| Item | Reason |
|---|---|
| `p1-17-backup` | byte-identical copy of `p1-17-train-full` (810/810 files) |
| `p1-18-tokens-v2@1.partial`, `p1-18-tokens-v2f@1.partial` | superseded partial builds; the complete `@2` / `@1` sets are uploaded |
| periodic training checkpoints | only on the Modal volume `ember-v1-checkpoints`; `final/` is the V1 checkpoint |
| base model `unsloth/Qwen3.5-2B@fffbc0d8…` | public, pinned. `identity.json` records every file hash, and the workstation verifies them after download. Residual risk: upstream deletion. |

## 2. Execution package

- `ops/workstation/ws.sh`: Linux or WSL2. GPU runs go through this script.
- `ops/workstation/ws.ps1`: Windows host. It runs `ws.sh` inside WSL2 and also gives a CPU-only native route.
- `src/ember_v1/workstation.py` (`ember-workstation@1.0.0`) has the shared CLI: `prepare-run`, `execute-run`, `status`, `publish-results`, `fetch-results`.
- `infra/p0b-requirements.lock.txt` is the hash-locked GPU stack: Python 3.11, torch 2.8.0 + CUDA 12.8 (sm_120 / Blackwell), triton 3.4.0, unsloth 2026.10.11, transformers 5.2.0. It is Linux only.
- `infra/ws-cpu-requirements.lock.txt` is the hash-locked HF client for the native-Windows CPU route.

Run kinds:

| Kind | Device | Purpose |
|---|---|---|
| `env-check` | GPU | lock drift, CUDA, sm_120 arch, bf16 matmul, SDPA, triton JIT, `p0b verify-install` |
| `trace-replay` | CPU | replay the V1 rollout traces in the simulator and match the recorded outcomes |
| `v1-reproduce` | GPU | reload V1, replay recorded steps, compare argmax per step (near-tie 0.05) |
| `v1-policy-ablation` | GPU | arms A/B/C (§5). The config is **NOT_AUTHORIZED**. |

Safety behaviour:

- `prepare-run` writes `execution_manifest.json`, which records code, config sha, revisions, input manifests, environment and an identity hash. It refuses these cases:
  - a config with a status other than `AUTHORIZED`
  - a dirty checkout or a checkout not at the given commit
  - a config outside the checkout
  - an attempt already prepared with other inputs
- `execute-run`:
  - refuses a COMPLETE run
  - refuses INTERRUPTED / FAILED / stale RUNNING without `--resume`
  - holds a lock file
  - re-checks the environment and the input hashes
  - writes to `partial/` and renames to `final/` only on success
  - marks INTERRUPTED on SIGTERM / SIGHUP
- `publish-results` publishes a COMPLETE run to `<result_path>/attempt-<n>`. An incomplete run needs `--allow-incomplete` and goes to a separate suffixed path.
- `ws.sh run` starts with `nohup setsid`, so closing the terminal does not stop the run. `ws.ps1 run` starts a hidden `wsl.exe` that lives as long as the run.

## 3. One-time workstation setup (user actions)

1. **GitHub read access.** On the workstation, make an SSH key: `ssh-keygen -t ed25519`. Add the public key to `anantamlab/ember_v1` as a **read-only deploy key** (Settings → Deploy keys). Do not use a personal token with write access.
2. **Hugging Face.** Make a **fine-grained read-only** token for the three `sankalphs/ember-v1-*` repos. Give write access only for the experiments repo, which needs it for `publish`. Enter the token with `ws.sh login` (prompted, not echoed). Remove it with `ws.sh logout`.
3. **Windows host only.**
   - In an admin PowerShell, run `wsl --install -d Ubuntu-24.04` and reboot.
   - In Ubuntu, run `sudo apt-get install -y build-essential git curl`.
   - Use the Windows NVIDIA driver ≥ 570. Never install a Linux driver inside WSL.
4. Native Linux: driver ≥ 570, plus `build-essential git curl`.

## 4. Workstation commands

`C` is the exact pushed commit of PR #53 that contains this file. Revisions come from `data/hf/registry.json` at that commit (`artifacts.<id>.manifest_commit`).

Windows host (PowerShell, in a copy of `ops/workstation/ws.ps1` from commit `C`):

```powershell
.\ws.ps1 doctor
.\ws.ps1 bootstrap C
.\ws.ps1 wsl setup C
.\ws.ps1 wsl login
.\ws.ps1 wsl prepare C data/runs/r19-env-check@1.json
.\ws.ps1 run r19-env-check@1
.\ws.ps1 wsl status r19-env-check@1
.\ws.ps1 wsl publish r19-env-check@1
```

Linux / inside WSL2:

```bash
ws.sh doctor
ws.sh setup C
ws.sh login
ws.sh prepare C data/runs/r19-env-check@1.json
ws.sh run r19-env-check@1            # detached
ws.sh status r19-env-check@1
ws.sh publish r19-env-check@1
# then:
ws.sh prepare C data/runs/r19-trace-replay@1.json --experiments-revision <p1-18-v1-records@1 manifest commit>
ws.sh prepare C data/runs/r19-v1-repro@1.json --model-revision <p1-18-v1-train@1 manifest commit> \
      --experiments-revision <p1-18-v1-records@1 manifest commit>
ws.sh run r19-v1-repro@1
ws.sh publish r19-v1-repro@1
```

If a run stops (reboot, kill), `ws.sh status` shows INTERRUPTED. Continue with `ws.sh resume <run_id>`, or use `--attempt 2` for a fresh attempt.

Laptop review of a published result:

```bash
python -m ember_v1.workstation fetch-results --path v1-evaluation/r19-v1-repro@1/attempt-1 \
  --revision <manifest commit printed by publish> --dest D:/Coding/ember-evidence/r19-v1-repro@1 \
  --records D:/Coding/ember-evidence/p1-18-v1/eval-out/rollouts.jsonl
```

## 5. Transfer test status (2026-11-11)

| Step | Where | Result |
|---|---|---|
| Laptop-simulated workstation: clean detached checkout of `020dfeaf`, fresh Python 3.11 venv from `infra/ws-cpu-requirements.lock.txt` (hub 1.33.0), empty `HF_HOME` | laptop | done |
| `prepare-run r19-trace-replay@1 --attempt 900` (records @ `283a753c`) | laptop | PASS: 36 files fetched and hash-checked |
| `execute-run` | laptop | COMPLETE: 156/156 traces replay to the recorded outcome |
| `execute-run` again | laptop | REFUSED (COMPLETE is never rerun) |
| `publish-results` → experiments `student-rollouts/r19-trace-replay@1/attempt-900` | laptop | PASS: manifest commit `f50a6216`, verified `d3456fb7` |
| `publish-results` again | laptop | idempotent no-op |
| `fetch-results` with hub 2.4.0 | laptop | PASS: verdict rechecked, 156/156 |
| `env-check` (GPU, sm_120) | workstation | **NOT EXECUTED**: no workstation access |
| `v1-reproduce` | workstation | **NOT EXECUTED** |

Attempt 900 is reserved for this laptop simulation. Its manifest records host and platform. Workstation runs use attempts 1, 2, ….

## 6. Prepared, not authorized: V1 policy arms (`data/runs/r19-v1-arms@1.json`)

All arms use one checkpoint (`p1-18-v1-train@1`), the same scenarios and the same fire. Fire depends only on the scenario seed. The arms are interleaved in blocks of 4 scenarios, with one rollouts file per arm.

- **A_raw**: the neural argmax.
- **B_static_mask**: per person, the best-scored admissible action. It drops moves off-grid, into walls or into fire.
- **C_joint_coord**: B, then a joint repair with the exact resolver `sim.movement.resolve_moves`, until no move is blocked.

Every record keeps `raw_trace` (model argmax) apart from `trace` (executed actions), with change counters. B and C are post-processing, not model outputs. To run the arms, the owner must set `status: AUTHORIZED` and an authorization note in the config on the laptop, and push.
