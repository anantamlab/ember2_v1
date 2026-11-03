# ember_v1 — Evacuation Decision Model (Anantam Labs) — HLD v3.1

**Project:** `ember_v1` (repo `ember_v1`, Python package `ember_v1`, CLI `ember-v1`, Modal app `ember-v1`).
**Status:** P0A done (CPU). P0B harness ready for Modal, **not executed**. No T0–T2 teacher yet. Data sources: one registered fixture source S0 (§7.1). No trained model.
**Supersedes:** HLD v2 (project "evac-decision"). The v2 phase gates stay; v3 adds a concrete teacher, a Modal plan, a token budget and four design decisions from verified Unsloth facts. v3.1 renames the project and adds the data-source registry.
**Boundary:** separate repo. Never write training code or assets into EMBERBench.

## 0. What changed from v2

| Area | v2 | v3 | Why |
|---|---|---|---|
| Research claim | "Competitive with strongest planner" | **Amortize** an expensive observation-only planner into a policy that meets the deadline | A student trained on teacher labels rarely beats that teacher. The useful claim is "near-teacher safety at a fraction of teacher compute". |
| Teacher | "Strong reference teacher" (undefined) | T0/T1/T2 stack (§4.4). T2 gives real per-action scores | P1 can not start without a concrete teacher. Real scores make KL distillation legitimate. |
| Question text | One long sentence + 5 long option texts per person | `compact` style (ID, position, grid delta) as the P0B candidate | Questions are 84–94% of the payload at N=200 (D2). The Clef encoder **cuts the state** when a row is too long. |
| Left padding | Required, else gate fails | **Out of scope (D3)**; gate accepts an explicit, recorded exclusion | The Clef head takes `attention_mask.sum()` as the length: right padding only. |
| Safety shield | Not present | Disclosed single-person shield as an **ablation arm** (off by default) | Removes an error class (wall/boundary/fire moves) with no route planning. |
| GPU platform | "Modal harness" (not designed) | `infra/modal_app.py`: `P0BHarness` + `DecisionPolicy` (`@modal.batched`) | Real cross-environment GPU batching with no custom server. |
| Smoke truncation check | `report["truncated"]` | Per-row check: report flag, record flag **and** `input_tokens >= state_tokens` | `build_dataset` reports `skipped`/`total`; a truncation key is not confirmed. The old check could pass truncated rows. |

## 1. Problem and research question

**Task.** Each step, for every active person, choose one of `stay|north|south|east|west`. The input is only the public `Observation.to_dict()`. Movement: lowest ID wins a contested cell, following chains are legal, swaps and rotations block, fire entry blocks or kills (`enter_fire`).

**Question.** Can a pretrained Qwen3.5-2B with a typed non-generative decision head amortize an expensive, observation-only reference planner (T2) so that, on unseen topology families:

1. safety (avoidable casualties) is not worse than the best deterministic planner (T1) by more than a preregistered margin,
2. it reaches a preregistered fraction of T2 quality, and
3. its p99 step latency meets the documented deadline, where T2 does not?

If T1 (cheap) already matches T2, the amortization claim fails. Report that as a negative result.

## 2. Facts and assumptions

### 2.1 Verified (sources in v2 §1; Unsloth read from `main`, 2026-11, re-check at pin time)

- EMBERBench `Observation.to_dict()` keys: `observation_version, step, map, rules, people, burning, allowed_actions`. No seed, RNG, future fire or trapped flag.
- `Agent.act(obs) -> {active_id: action}`, fresh agent per episode, fail-closed. The core has no timeout.
- `FastDecisionModel.from_pretrained(model_name, max_seq_length, load_in_4bit)` returns `(model, tokenizer)`. The model is a `ClefDecisionModel` with `.encoder` (backbone) and `.head` (fp32 head).
- `build_dataset(rows, tokenizer, model) -> (items, report)`. `report` has `total`, `skipped`. Over-length rows: **the state is cut, questions are kept**; rows that still do not fit are skipped.
- `predict(model, tokenizer, state, questions_dict)` gives per question `answer` and per-option `probabilities` (temperature-scaled). One state per call.
- `DecisionTrainer` has `head_learning_rate` (default constant `HEAD_LEARNING_RATE = 1e-4`), `kl_weight`, `label_smoothing`, `brier_weight`, `permute_fields`. `evaluate(..., batch_size=)`, `calibrate(...)`, `split_holdout(...)` exist.
- Vendored Clef `collate_records(records, pad_token_id, device)` right-pads. The head uses `attention_mask.sum()` as length. No `cu_seqlens`, no `position_ids` handling.
- Notebook pins: `torch==2.8.0`, `transformers==5.2.0`, `trl==0.22.2`, `xformers==0.0.32.post2`; `unsloth` unpinned.

### 2.2 Assumptions to verify (owner phase in brackets)

| # | Assumption | Check |
|---|---|---|
| A1 | `model(collate_records(...))` returns per-record, per-question option logits | `--task inspect`, then `padding` cross-checks argmax vs `predict()` [P0B] |
| A2 | `EncodedRecord.questions[i]` has `name`/`option_ids`, or keeps dict order | `inspect` prints attributes [P0B] |
| A3 | `build_dataset` accepts `questions` as a dict (notebook data uses a JSON string) | `smoke` [P0B] |
| A4 | Off-grid and wall moves resolve as "stay", not as an error | Read EMBERBench `movement.py` before the shield is used [P0A+] |
| A5 | fla / causal-conv1d kernels actually run for Qwen3.5 DeltaNet layers | Profiler kernel names in `inspect` [P0B] |
| A6 | A LoRA checkpoint dir loads through `from_pretrained(model_name=dir)` | First P3 checkpoint [P3] |
| A7 | The real evaluation runner permits a remote (Modal) call inside `act()` | Read the runner/adapter; else measure a co-located GPU [P2] |

## 3. Architecture

```text
                    ONLINE DECISION PLANE (per simulator step)
 Simulator --obs--> DecisionAgent.act()
                     | validate_observation (whitelist, canonical order)
                     | encode_observation (ascii | coords [+neighbors])
                     | decision_questions (one per active ID, verbose|compact)
                     v
                   predictor -------------------------------> Modal DecisionPolicy (GPU)
                     |   local: unsloth_predictor              @modal.batched(max 32, 10 ms)
                     |   remote: modal_predictor               path A: loop predict()   (default)
                     |                                         path B: one right-padded forward
                     |                                                 (only after padding gate PASS)
                     v
                   per-ID {answer, probabilities[5]}
                     | [optional, disclosed] shield: drop off-grid/wall/fire moves
                     | validate_actions (exact active ID set, no repair)
                     v
                 Simulator.step  --> closed-loop metrics + diagnostics

                    OFFLINE DATA / TRAINING PLANE
 S0 p0a zip (fixture) --ember-v1 ingest--> validated traces [pipeline_smoke only]
 Independent generator (topology families, private seeds)
   -> Teacher T0/T1/T2 (observation-only) -> trace records (+ T2 action scores)
   -> simulator replay verification -> ember-v1 ingest (quality tier, sha256)
   -> data/sources.json registry -> family/lineage split
   -> build_decision_row -> Modal training job (DecisionTrainer, LoRA + head)
   -> calibrate (calibration split) -> checkpoint on Modal Volume
   -> closed-loop dev eval (DecisionPolicy) -> DAgger relabel (T2) if needed
```

### 3.1 Components

| Component | Module | Status |
|---|---|---|
| Observation whitelist, action validator | `contract.py` | Done, tested |
| Encodings (ascii, coords, neighbors) | `encoding.py` | Done, tested |
| Row builder, question styles | `dataset.py` | Done, tested (`compact` new) |
| Agent adapter (fail-closed, shield, stats) | `agent.py` | Done, tested with fakes |
| Shield (ablation arm) | `shield.py` | Done, tested; A4 open |
| Raw intent diagnostics (not a solver) | `diagnostics.py` | Done, tested |
| Padding verdict | `padding_gate.py` | Done, tested (token-count and left-scope checks new) |
| GPU measurements | `p0b.py` | Written, schema-tested on CPU, **not run on GPU** |
| Modal app | `infra/modal_app.py` | Definition loads with `modal==1.6.3`, **not run** |
| Data-source registry + zip ingestion | `sources.py`, `ember-v1 ingest`, `data/sources.json` | Done, tested; S0 registered |
| Independent generator + simulator | `generator/`, `sim/` | P1 |
| Teacher T0/T1/T2 + trace logger | `teacher/` | P1 |
| Training job, eval harness, reports | `training/`, `eval/`, `reports/` | P2–P3 |

## 4. Design decisions

### D1 — Backbone: Qwen3.5-2B only (unchanged)
Pin the model commit and all runtime versions in every result file (`runtime_metadata`). No backbone tournament.

### D2 — One shared state, one typed question per active ID
Keep real IDs (they set conflict priority). Each question carries the ID and position, so the head can bind ID to location.

**Token budget (measured characters, synthetic 24x24, ascii):**

| N | state | state+neighbors | questions verbose (JSON) | questions compact (JSON) |
|---|---|---|---|---|
| 50 | 1,747 | 3,756 | 16,157 | 8,307 |
| 200 | 4,483 | 12,516 | 64,892 | 33,492 |

Instruction + option text (no JSON syntax) at N=200: 40,202 -> 8,802 chars (4.6x) with `compact`. Exact token counts come only from `build_dataset` on the real tokenizer (P0B `smoke`). Rule: **a truncated row is a failed row**. Never train on a row whose state was cut.

If compact still does not fit at N=200: (1) raise `max_seq`, measure latency; (2) drop `ADJ=` neighbors for large N; (3) chunk questions into groups that share the full state prefix. Choose by measurement, not by guess.

### D3 — Right padding only; left padding out of scope
The head computes the length from `attention_mask.sum()` and uses absolute spans. Left padding needs a custom head, which v2 forbids without a demonstrated need. The gate now accepts `metadata.left_unsupported_reason` and then requires only `single` + `right`. The verdict reports `left_tested: false`. Left results together with that flag fail. Packed varlen stays `PACKED_UNSUPPORTED`.

The harness runs three batch orders (short first, short last, short in the middle) and needs a PASS on each. It also requires 100% argmax agreement between the batched adapter (T=1) and `predict()` before the batched path is trusted.

### D4 — Teacher: observation-only, three levels
The teacher sees only what the student sees (plus the known rules). So the imitation target is realizable. A privileged teacher (seed or future fire) is an upper-bound report only, never a label source.

| Level | Method | Cost | Role |
|---|---|---|---|
| T0 | Fire-aware BFS to nearest reachable exit, avoid burning and "soon-burning" cells | ms | Baseline; DAgger fallback check |
| T1 | Prioritized cooperative planner: process people in **ascending ID** (matches lowest-ID-wins), space-time reservation table, follow-chains allowed, swaps forbidden, exit load balancing by estimated arrival time | 10–100 ms | Strongest deterministic baseline; P2 reference |
| T2 | Rollout-improved T1: for each person, evaluate each of the 5 actions with others fixed to T1, continue all with T1 for H steps over K sampled fire futures (stochastic rules) | seconds | **Label source**; gives real per-action value estimates |

T2 per-action values are real Monte-Carlo estimates. They allow `kl_weight > 0` soft targets in P3 (as an ablation vs hard labels). Record K, H and the continuation policy version in every trace. Do not call these values causal: other agents are held fixed.

Movement-rule parity: the independent simulator must pass a shared test table (contested cells, chains, swaps, rotations, fire entry) that matches EMBERBench `movement.py` behavior before any teacher data is used.

### D5 — Training recipe (P3 start point)
- `DecisionTrainer`, LoRA r=16, alpha=16 (notebook), `head_learning_rate=1e-4` (library default), LoRA LR 2e-4, cosine, 1–2 epochs.
- `permute_fields=True` as the order-robustness ablation. ID strings never change.
- `calibrate()` on the calibration split only. Fit temperature once, freeze it in P7.
- Hard labels first. Soft T2 labels (`kl_weight`) as one ablation arm.
- bf16 LoRA first (no quantization noise in parity tests); 4-bit only after the bf16 baseline exists.

### D6 — Shield: disclosed ablation arm
`shield.py` removes, per person, only moves that can not beat `stay`: off-grid, wall, burning cell. It keeps moves into occupied cells. It never plans. `DecisionAgent(shield=True)` picks the highest-probability admissible action and counts overrides. Always report raw and shielded results apart. High override rates are a model-quality finding, not a success.

### D7 — Coordination (P4, unchanged intent)
Arms: (a) raw argmax, (b) auxiliary exit-assignment question, (c) joint re-ranking with **exact** movement-resolver semantics. Attribute any solver gain to the solver, not to the model.

## 5. Modal deployment

### 5.1 Topology

| Object | Type | GPU | Purpose |
|---|---|---|---|
| `P0BHarness` | `app.cls` | L4 (override with `--gpu L40S`) | `inspect`, `smoke`, `padding`, `throughput` |
| `DecisionPolicy` | `app.cls` + `@modal.batched(max_batch_size=32, wait_ms=10)` | L4 / L40S | Rollout and eval inference |
| `RolloutWorker` (P1+) | `app.function`, CPU | none | Runs simulator episodes; calls `DecisionPolicy.decide.remote()`; fan out with `.map(seeds)` |
| `Trainer` (P3) | `app.function` | L40S | `DecisionTrainer`; writes checkpoints to Volume |
| `ember-v1-hf-cache` | Volume | — | HF weights cache (`HF_HOME=/cache/hf`) |
| `ember-v1-checkpoints` | Volume | — | LoRA + head checkpoints, calibration temperatures |

Many CPU workers call `decide.remote()` at the same time. Modal groups those calls into one GPU batch. This is the cross-environment batching v2 asks for, without a custom server.

### 5.2 Rules
- **Path B (one padded forward) stays off** (`batched_forward=False`) until the padding gate PASSes on the pinned stack and GPU type.
- Every response carries `model_revision`, `path`, `batch_size`, `server_s`. Eval reports use them.
- Two latencies: **GPU latency** (`server_s` / batch) and **end-to-end** (client wall clock, includes network and `wait_ms`). The deadline claim uses the latency of the real deployment (A7).
- Fail-closed: truncation, missing IDs or a network error raise. No fallback action.
- Image pins follow the notebook. Pin `unsloth`/`unsloth_zoo` to exact versions after the first PASSing `inspect` + `smoke`. Do not change pins inside a phase.
- Fast DeltaNet kernels are an opt-in image layer (`EMBER_FAST_KERNELS=1`). Run P0B with and without it. Report which kernels ran (profiler), not which packages imported.

### 5.3 Budget (proposal; replace with measured values after P0B)

| Phase | Cap | Note |
|---|---|---|
| P0B | $10 | Short runs on L4 and L40S |
| P1–P2 | $15 | Mostly CPU (teacher T2 is CPU-heavy) |
| P3 | $60 | Pilot training + representation A/B |
| P4 | $40 | Coordination arms |
| P5–P6 | $35 | DAgger / AWR only if a gate needs them |
| Reserve | $40 | Not spent without a written reason |

Track GPU-seconds from `server_s`, simulator CPU time separately. Never convert dollars to episodes before measurement.

## 6. Phases and gates

The v2 DAG, pass criteria and stop-and-review protocol stay. Additions:

| Phase | Added deliverable / gate |
|---|---|
| P0A | Done. +compact questions, shield, gate checks (token count, left scope), data-source registry, S0 ingested. |
| P0B | Run order: `inspect` -> `smoke` (both question styles) -> `padding` (3 orders) -> `throughput` (L4, L40S; with/without fast kernels). Gate: no truncation at N<=200 for the chosen style, adapter agreement 100%, padding PASS, p50/p95/p99 and VRAM recorded. |
| P1 | Simulator movement parity table; T0/T1/T2; trace schema (§7); T2 traces ingested as S1 (`teacher_verified` after replay); T2 cost per step measured. Gate: T2 beats T1 on dev by a measurable margin, else rethink the amortization claim before training. |
| P2 | Preregister: primary endpoint, margins, paired-seed CI method, N episodes from pilot variance. |
| P3 | Arms: ascii vs coords vs ascii+neighbors; hard vs soft (T2) labels. Raw and shielded reported apart. |
| P5 | DAgger states labeled by T2, with lineage. |

### 6.1 Start here (first working week)

1. `pip install -e '.[test,modal]'`, `modal setup`, run the CPU suite.
2. `modal run infra/modal_app.py --task inspect`. Resolve A1, A2, A5. If the batched adapter fails, fix only `p0b.batched_probabilities` / `_question_names`.
3. `--task smoke --question-style verbose` and `--question-style compact`. Pick the style that has no truncation at N=200 and lower latency.
4. `--task padding --ns 1,10,50,200`. Record the verdict files.
5. `--task throughput` on L4 and L40S. Choose the GPU by env-steps per dollar.
6. Pipeline smoke with S0 (§7.1): a few `DecisionTrainer` steps on the 2 fixture decisions, then `predict()`. Pass = the run completes and the model overfits them. (A Modal `train` task is the first P3 code item.)
7. In parallel (CPU): build the independent simulator + movement parity table, then T0 and T1.
8. Update `PHASE_CHECKPOINT.json` with the P0B verdict and the exact pins.

## 7. Data contract (P1)

One trace record per (episode, step):

```json
{
  "episode_id": "fam-A/gen-v1/seed-000123", "family": "fam-A", "lineage": "gen-v1/seed-000123",
  "step": 8, "observation": {"...": "Observation.to_dict() only"},
  "teacher": {"level": "T2", "version": "t2@<git-sha>", "K": 16, "H": 20, "continuation": "t1@<git-sha>"},
  "actions": {"P10": "north", "P2": "south"},
  "action_values": {"P10": {"stay": -3.1, "north": -1.2, "south": -4.0, "east": -2.2, "west": -9.9}},
  "ties": {"P10": ["north"]},
  "next_observation_hash": "sha256:...", "outcome": {"evacuated": 7, "casualties": 0, "steps": 41}
}
```

`action_values` and `ties` are present only when T2 produced them. Splits by `family` (topology generator) and `lineage`; never by timestep. The locked test manifest lives outside the training storage.

### 7.1 Data sources and quality tiers

All data enters through `ember-v1 ingest <zip> --source-id ID --quality TIER --out data/ingested/ID.jsonl`. Ingest reads zip members in memory (JSON/JSONL only, never executes code). It validates every trace with the observation whitelist and the exact active-ID action check, and fails closed. It prefixes `group_id` with the source ID, so lineage stays unique across sources. It records the zip SHA-256 in `data/sources.json` (tracked). Raw zips (`data/raw/`) and ingested JSONL (`data/ingested/`) are git-ignored and are regenerated from the registry.

| Tier | Meaning | Training-eligible |
|---|---|---|
| `teacher_verified` | T0/T1/T2 output, replayed in the simulator | Yes |
| `teacher_unverified` | Teacher output, not yet replayed | No (replay first) |
| `synthetic_fixture` | Hand-written example labels | No; `pipeline_smoke` split only |

`sources.training_records()` returns only eligible records. `allow_fixtures=True` is for pipeline smoke runs only, and those runs never feed a reported metric.

**Registered sources**

| ID | File | SHA-256 | Content | Tier | Use |
|---|---|---|---|---|---|
| S0 `p0a-scaffold-fixtures` | `evac-decision-p0a-padding-gate.zip` | `bb924684…3157ded` | P0A code snapshot; 1 trace (2 decisions, teacher `example-teacher-v0`), 2 non-trace JSON | `synthetic_fixture` | See below |
| S1 (P1) | generator + T2 traces | per shard | Main training data | `teacher_verified` | Train / dev / calibration |

**How S0 is useful.** Its labels are hand-written, so they are not evidence of good behavior, and one trace can not teach a policy. S0 is useful as:

1. **Code base.** It is the P0A source this repo starts from.
2. **End-to-end pipeline smoke.** ingest -> `build_decision_row` -> `FastDecisionModel.build_dataset` -> a few `DecisionTrainer` steps on Modal -> `predict()`. Expected: the run completes and the model can overfit the 2 decisions. This tests wiring, not quality.
3. **Regression fixture.** Its trace and observation are canonical inputs for contract, encoding and ingest tests.

S0 never enters the train, dev, calibration or locked-test splits, and never counts in a reported metric.

## 8. Metrics

Primary (preregister one): avoidable casualties per episode vs T1 on unseen families. Secondary: evacuated fraction, steps to evacuate, fire exposure, blocked moves, contested proposals, swaps, oscillation, exit load balance, shield override rate, p50/p95/p99 GPU and end-to-end latency, GPU-seconds per episode, fail-closed error count.

## 9. Risks

| Risk | Signal | Response |
|---|---|---|
| Rows too long at large N | `smoke` truncation / skip | D2 fallbacks; cap supported N and say so |
| Batched path differs from singleton | Padding gate FAIL | Stay on path A; report throughput of path A |
| DeltaNet fast kernels missing | Profiler shows torch fallback | Measure both; pick by measured latency |
| T2 not better than T1 | P1 gate | Re-scope claim before any training spend |
| Student imitates but fails closed-loop | P3 closed-loop gap | P5 DAgger with T2 labels |
| ID binding errors at large N | Wrong-ID / missing-ID counts | Fail gate; test position-in-question variants |
| Remote call not allowed in real runner | A7 | Measure co-located GPU latency for the claim |
| Budget overrun | Phase cap reached | Stop, report, use reserve only with a written reason |

## 10. Ownership

The `ember_v1` repo owns `contract`, `encoding`, `dataset`, `agent`, `shield`, `diagnostics`, `padding_gate`, `p0b`, `sources`, `infra`, `data/sources.json`, and (later) `generator`, `sim`, `teacher`, `training`, `eval`, `reports`. EMBERBench owns its simulator and evaluation data. The trained policy touches only `Observation.to_dict()` and `Agent.act()`.
