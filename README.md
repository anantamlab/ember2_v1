# ember_v1 — Evacuation Decision Model (Anantam Labs, Phase 0A)

Package `ember_v1`, CLI `ember-v1`, Modal app `ember-v1`. Formerly "evac-decision" (HLD v2).

A **standalone, training-boundary-safe** starter for adapting **pretrained Qwen3.5** into a Jev-style non-generative decision policy using **Unsloth FastDecisionModel**. It does not fork or edit EMBERBench, does not run GPU training, and contains **no official benchmark or hidden evaluation data**.

Read the phase-by-phase [High-Level Design](docs/HLD.md). This initial project implements data-contract and profiling groundwork plus a **CPU-only padding-parity verdict checker** for upcoming GPU evidence; no real padding equivalence has been measured yet. The GPU `FastDecisionModel` compatibility test is the next gate.

## Run locally

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

All training data enters through `ember-v1 ingest` and is registered in `data/sources.json` with a quality tier and SHA-256 (HLD v3.1 §7.1). Only `teacher_verified` data is training-eligible.

```bash
ember-v1 ingest data/raw/evac-decision-p0a-padding-gate.zip \
  --source-id p0a-scaffold-fixtures --quality synthetic_fixture \
  --out data/ingested/p0a-scaffold-fixtures.jsonl
```

S0 (`p0a-scaffold-fixtures`) holds 1 hand-written trace (2 decisions). It is for the pipeline smoke and regression tests only, never for reported metrics. Put raw zips in `data/raw/` (git-ignored).

## Repository destination

`ember_v1` is a standalone project, not part of `anantamlab/EMBERBench`. EMBERBench stays the evaluation instrument.

## P0B on Modal (**not executed yet**)

Design: [HLD v3 §5](docs/HLD.md#5-modal-deployment). Logic lives in `src/ember_v1/p0b.py`; `infra/modal_app.py` runs it on Modal GPUs.

```bash
python -m pip install -e '.[test,modal]'
modal setup
modal run infra/modal_app.py --task inspect                          # check Unsloth internals first
modal run infra/modal_app.py --task smoke --question-style verbose
modal run infra/modal_app.py --task smoke --question-style compact
modal run infra/modal_app.py --task padding --ns 1,10,50,200        # writes gate inputs + verdicts
modal run infra/modal_app.py --task throughput --gpu L40S
EMBER_FAST_KERNELS=1 modal run infra/modal_app.py --task inspect     # opt-in fla/causal-conv1d layer
```

Results go to `results/` (git-ignored). Exit code 2 means a gate FAIL.

Serving for rollouts: `modal deploy infra/modal_app.py`, then
`DecisionAgent(modal_predictor(), question_style="compact")`. Modal batches concurrent
`decide` calls from many simulator workers into one GPU batch. The padded single-forward path
(`batched_forward=True`) stays off until the padding gate passes.

A local CUDA host can run the same smoke: `python experiments/phase0b_unsloth_smoke.py`.

## Shield (ablation arm, off by default)

`DecisionAgent(..., shield=True)` drops only off-grid, wall and burning-cell moves per person and counts overrides in `agent.stats`. It never plans routes. Report raw and shielded results apart. Verify assumption A4 (HLD §2.2) against EMBERBench `movement.py` first.

## P0B handoff: padding-equivalence gate

Read [the DeltaNet/Clef padding-equivalence test spec](docs/P0B_PADDING_EQUIVALENCE.md) before writing the Modal batching harness. The Qwen3.5-2B backbone combines gated DeltaNet, causal convolution and attention; masking semantics of the entire encoder **and** Clef decision head must be verified in real GPU batches. The current upstream Clef collator right-pads; packed `cu_seqlens` support is unverified.

The harness should emit actual singleton, right-padded, left-padded and (if proven supported) packed probabilities for unequal-length environments. Then run:

```bash
python -m ember_v1.padding_gate results/modal-padding-results.json --output results/padding-verdict.json
```

The checker fails on absent IDs, truncated inputs, large score differences and high-confidence decision flips. CPU tests cover the checker logic only. **No GPU inference or packed-varlen capability is claimed.**

**Backbone lock:** Qwen3.5-2B only. The older HLD originally explored 0.8B/4B, but backbone selection is now frozen for this project; future changes require an explicit separate decision.

**GitHub status:** not published. Push to a dedicated repository (e.g., `anantamlab/ember-v1`), not EMBERBench.
