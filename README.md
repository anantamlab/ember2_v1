# ember_v1 — Human-view Evacuation Decision Model (Anantam Labs)

Package `ember_v1`, CLI `ember-v1`, Modal app `ember-v1`. Formerly "evac-decision" (HLD v2).

A standalone evacuation decision policy using Qwen3.5 and Unsloth FastDecisionModel, with an independent CPU simulator. It contains no official benchmark or hidden evaluation data.

**This repository is the human-view (partial-view) project.** The full-view project lives in
[`anantamlab/ember2_v1`](https://github.com/anantamlab/ember2_v1), in a separate checkout.
Human-view evidence: `/home/darshith/ember-ws/rl-evidence/human-pov/`; private HF: `sankalphs/ember-v1-models`.
Full-view evidence: `/home/darshith/ember-ws/ember2-evidence/`; private HF: the HF_TOKEN account
repository `ember2-v1-models`. Each project owns its evidence and HF repository. `anantamlab/ember-v1-research-handoff` is a read-only backup; its commit is part of this history.

## Start here

Read **[docs/CURRENT_STATE.md](docs/CURRENT_STATE.md)**. It is the authoritative current state: code map, completed SFT and RL runs, verified findings, checkpoint locations, how to verify and what remains. The [research handoff](handoff/HANDOFF.md) keeps the Run 19 to Phase 3 archive with exact checkpoints and recovery notes. For RL work, read the [RL results handoff](docs/RL_RESULTS_HANDOFF.md): the RL system as built and all R19–R22 metrics.

Current human-view candidate: frozen LLM PPO update 20 of `human-pov-ws-full-20261113t214243z`.
Its 32-family report panel is spent. Read the [model card](docs/HUMAN_POV_MODEL_CARD.md),
[report](docs/HUMAN_POV_REPORT.md) and [deferred work](docs/NEXT_STEPS.md). No new human-view run is authorized.

Historical full-map results (2026-11-13):

- **V1 (SFT, `p1-18-v1-train@1`)** is trained and stored on private Hugging Face at revision `1fc405f7`.
- **[R20 Phase 1](docs/R20_PPO_PHASE1_REPORT.md)** verified real PPO weight updates on V1, with no DEV evacuation gain.
- **[R21 Phase 2](docs/R21_PPO_PHASE2_REPORT.md)** online PPO update 96 tied raw V1 at 48.60% evacuation on its reporting panel.
- **[R22 Phase 3](docs/R22_COUNTERFACTUAL_REPORT.md)** counterfactual PPO reduced casualties relative to PPO-only on reporting DEV (−2.11 pp [−3.96, −0.39]) but did not establish an evacuation gain (+0.85 pp [−1.86, +3.48]). The predefined DEV selector froze PPO-only update 16.
- Rule-based masking and coordination still outperform the raw learned policies on the compared panels. The T0–T4 teachers are far stronger.
- **Final held-out evaluation is sealed.** The 1,200-scenario expansion and `p1-18-dataset-v2` are reserved for V1.1.
- Checkpoints: [data/rl/checkpoint_registry.json](data/rl/checkpoint_registry.json); verify with `python scripts/verify_checkpoint.py --list`. The R20–R22 weights are on the workstation only, not on Hugging Face.

`PHASE_CHECKPOINT.json` and the P0B sections below are historical P0A/P0B records. They are not the current state.

## Run locally

There is no GitHub CI. Before every push, run the complete CPU suite locally on Python 3.11 with
`python scripts/verify_local.py run` — see [AGENTS.md](AGENTS.md).

```bash
python -m pip install -e '.[test]'
python scripts/verify_local.py run --python <path-to-python3.11>   # default tier; add --with-slow for the full tier
python -m pytest -q
python -m ember_v1.cli encode examples/observation.json --mode ascii --local-neighbors
python -m ember_v1.cli encode examples/observation.json --mode coords
python -m ember_v1.cli prepare examples/teacher_trace.json --mode ascii
python -m ember_v1.cli profile examples/observation.json
```

`profile` without a Qwen tokenizer reports characters/bytes only; **these are not token counts**. For a token-count proxy, install `'.[probe]'` and supply a local/pinned tokenizer; the real P0B gate still must measure Unsloth's complete encoded record and actual GPU latency.

```bash
python -m pip install -e '.[probe]'
python -m ember_v1.cli profile examples/observation.json --tokenizer /path/to/pinned/qwen-tokenizer
```

The `prepare` subcommand emits Unsloth-compatible `state`, `questions`, `gold` records while preserving stable person IDs. Sample records are **synthetic fixtures only**. The real teacher must be instrumented in Phase 1; no teacher score distributions have been inferred or fabricated.

## Strict boundaries

- Only fields from the public observation contract cross into the student model. Unknown keys, including `seed`, `scenario_id`, `rng_state`, `trapped`, and future data, cause errors.
- Every active person is retained exactly once; inactive persons receive no decision. Actions must exactly cover active IDs or fail closed.
- ASCII and coordinate encodings are ablations. The model is not trained to decide from these examples.
- Raw conflict counts in `diagnostics` are **not** the benchmark's full movement resolution; do not use them as a solver.
- Core training framework: [Unsloth FastDecisionModel](https://github.com/unslothai/notebooks/blob/main/nb/Qwen3_5_(4B)-Decision.ipynb). [Bandits Jev](https://github.com/bandr-ai/bandits/tree/main/recipes/jev) may be a comparison arm, but its trace judge is not our ground truth.

## Data sources

Supervised training data enters through `ember-v1 ingest` and is registered in `data/sources.json` with a quality tier and SHA-256 (HLD v3.1 §7.1). Only `teacher_verified` data is eligible for supervised training. PPO uses independently generated TRAIN scenarios and simulator outcomes recorded in its own evidence archive.

```bash
ember-v1 ingest data/raw/evac-decision-p0a-padding-gate.zip \
  --source-id p0a-scaffold-fixtures --quality synthetic_fixture \
  --out data/ingested/p0a-scaffold-fixtures.jsonl
```

S0 (`p0a-scaffold-fixtures`) holds 1 hand-written trace (2 decisions). It is for the pipeline smoke and regression tests only, never for reported metrics. Put raw zips in `data/raw/` (git-ignored).

## Repository destination

`ember_v1` is a standalone project, not part of `anantamlab/EMBERBench`. EMBERBench stays the evaluation instrument.

## Historical: P0B on Modal (P0A-era notes)

These P0B notes predate the trained V1. The P0B harness itself was not run as written here. V1 was later trained on Modal (see [docs/CURRENT_STATE.md](docs/CURRENT_STATE.md)).

Design: [HLD v3 §5](docs/HLD.md#5-modal-deployment). Logic lives in `src/ember_v1/p0b.py`; `infra/modal_app.py` runs it on Modal GPUs. The harness is preflight-hardened on CPU only (P0B-01). The upstream audit, open GPU-only assumptions, the full run checklist and the budget are in [docs/P0B_PREFLIGHT.md](docs/P0B_PREFLIGHT.md).

```bash
python -m pip install -e '.[test,modal]'
modal setup
modal run infra/modal_app.py --task inspect                          # check Unsloth internals first
modal run infra/modal_app.py --task smoke --question-style compact --repeats 100
modal run infra/modal_app.py --task padding --ns 1,10,50,200        # writes gate inputs + verdicts
modal run infra/modal_app.py --task throughput --gpu L40S --repeats 100
```

Results go to `results/` (git-ignored). Exit code 2 means `FAIL` or `UNVERIFIED`. A row passes only with positive evidence that the whole state and every question were encoded; unknown means fail.

Serving for rollouts: `modal deploy infra/modal_app.py`, then
`DecisionAgent(modal_predictor(), question_style="compact")`. Modal batches concurrent
`decide` calls from many simulator workers into one GPU batch. The padded single-forward path
(`batched_forward=True`) stays off. It starts only with `padding_evidence_path` set to a full `padding`
result that passes the re-run gate on the same stack.

A local CUDA host can run the same smoke: `python experiments/phase0b_unsloth_smoke.py`.

## Shield (ablation arm, off by default)

`DecisionAgent(..., shield=True)` drops only off-grid, wall and burning-cell moves per person and counts overrides in `agent.stats`. It never plans routes. Report raw and shielded results apart. Verify assumption A4 (HLD §2.2) against EMBERBench `movement.py` first.

## P0B handoff: padding-equivalence gate

Read [the DeltaNet/Clef padding-equivalence test spec](docs/P0B_PADDING_EQUIVALENCE.md) before writing the Modal batching harness. The Qwen3.5-2B backbone combines gated DeltaNet, causal convolution and attention; masking semantics of the entire encoder **and** Clef decision head must be verified in real GPU batches. The current upstream Clef collator right-pads; packed `cu_seqlens` support is unverified.

The harness emits actual singleton and right-padded probabilities for unequal-length environments. Left padding is out of scope (HLD D3) and recorded as such; packed varlen is unverified. To re-check a gate input, run:

```bash
python -m ember_v1.padding_gate results/modal-padding-results.json --output results/padding-verdict.json
```

The checker fails on absent IDs, truncated inputs, large score differences and high-confidence decision flips. CPU tests cover the checker logic only. **No GPU inference or packed-varlen capability is claimed.**

**Backbone lock:** Qwen3.5-2B only. The older HLD originally explored 0.8B/4B, but backbone selection is now frozen for this project; future changes require an explicit separate decision.

**GitHub:** [`anantamlab/ember_v1`](https://github.com/anantamlab/ember_v1), a dedicated repository separate from EMBERBench.
