# ember_v1 — Evacuation Decision Model (Anantam Labs)

Package `ember_v1`, CLI `ember-v1`, Modal app `ember-v1`. Formerly "evac-decision" (HLD v2).

A standalone evacuation decision policy using Qwen3.5 and Unsloth FastDecisionModel, with an independent CPU simulator. It contains no official benchmark or hidden evaluation data.

Start with the [workstation and research handoff](handoff/HANDOFF.md) for Run 19 through Phase 3, exact checkpoints, and reproduction notes.

The [Phase 2 online PPO experiment](docs/R21_PPO_PHASE2_REPORT.md) completed 1,024 on-policy episodes from the verified Phase 1 checkpoint. On fresh reporting DEV, the selected PPO actor tied raw V1 at 48.60% evacuation. The report records the casualty uncertainty, persistent movement failures, exact checkpoint identity and reproduction commands. The [Phase 1 pilot report](docs/R20_PPO_PHASE1_REPORT.md) and [High-Level Design](docs/HLD.md) retain the project's research history.

The [Phase 3 experiment](docs/R22_COUNTERFACTUAL_REPORT.md) compared PPO-only with counterfactual-assisted PPO from the verified Phase 2 checkpoint. Counterfactual assistance reduced casualties on reporting DEV, but the evacuation difference remained inconclusive. The selection split froze PPO-only update 16. The report links both trained checkpoint identities and the independently replayed counterfactual dataset index. Final evaluation remains sealed.

## Run locally

There is no GitHub CI. Before every push, run the complete CPU suite locally on Python 3.11 with
`python scripts/verify_local.py run` — see [AGENTS.md](AGENTS.md).

```bash
python -m pip install -e '.[test]'
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

## P0B on Modal (**NOT RUN**)

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
