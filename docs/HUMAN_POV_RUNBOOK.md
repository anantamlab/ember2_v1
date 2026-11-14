# EMBER V1 — human point of view: design and runbook

Status date: 2026-11-13. Audience: the person or agent who trains on the GPU workstation.
Read [CURRENT_STATE.md](CURRENT_STATE.md) and [RL_RESULTS_HANDOFF.md](RL_RESULTS_HANDOFF.md) first.

The authorized workstation run completed DAgger, CPU PPO, LLM PPO and the single report evaluation on 2026-11-14. See [HUMAN_POV_REPORT.md](HUMAN_POV_REPORT.md) for results and storage status. The instructions below describe how to execute a newly authorized run; do not repeat the completed report. The tracked draft PPO config remains unchanged.

## 1. What changes and why

| Before (R19–R22) | Now (human point of view) |
|---|---|
| One model reads the full map and answers for every person. | Each person decides from their own view. |
| Full map, all people, all fire, absolute coordinates. | Cells in sight, own memory, own last move. No coordinates, no map size. |
| The model must read a large ASCII map to find walls. Half of all moves hit a wall. | The four neighbour cells are named in the view. A person sees a wall before they walk into it. |
| No previous action, no blocked flag. Loops repeat. | The view holds the last action and its result. |
| Labels from full-map teachers on teacher states only. | Labels from a teacher that uses the same view, on the states the student visits (DAgger). |
| PPO: one probability ratio for the whole crowd, one shared reward. | PPO: one ratio and one reward per person, legal-move mask, imitation anchor. |
| Human feedback: none. | Human feedback: none. The belief teacher is automatic. |

## 2. What a person knows

`src/ember_v1/human_view.py`, `ViewConfig`:

| Setting | Default | Meaning |
|---|---|---|
| `radius` | 4 | Sight distance in cells. The window is `(2r+1) × (2r+1)`, the person at the centre. |
| `line_of_sight` | true | A wall hides the cells behind it. The eight neighbour cells are always seen. |
| `memory` | true | Cells seen earlier stay known. People are never remembered. |
| `exit_knowledge` | `seen` | `seen`: only exits the person saw. `one`: one exit position is known from the start. `all`: all exit positions are known. |
| `hints` | true | The view holds the person's own route estimate (§3). |
| `fire_detour` | 6 | Extra steps the person accepts to stay away from cells next to fire. |

A view contains: `step`, `rules`, the window (`@ # . E F P ?`), the kind of each neighbour cell, the last action
and its outcome, known safe exits as offsets, and the route estimate. The grid edge looks like a wall.

A view never contains: absolute coordinates, the map size, cells that were never in sight, other people's
identities, the simulator seed or any private state. Tests prove this: a view does not change when a hidden cell
changes, and the same surroundings in a larger building give the same view.

## 3. The belief teacher (replaces human feedback)

`src/ember_v1/human_teacher.py`. It decides from the person's own knowledge only.

1. `plan()` makes a route estimate on the person's memory. Goal `exit`: shortest way to a known safe exit; unseen
   cells count as free. Goal `explore`: go to the nearest known cell that borders unseen space. Goal `flee`:
   move away from known fire. Goal `none`: stay.
2. `belief_action()` takes the action with the fewest steps left. It moves only for a real gain. On a tie it keeps
   the last direction. After a block by another person it takes an equally good free cell. After a head-on block
   it steps aside.

Every label is a function of the view. Thus the student can learn every label. Full-map teachers T0–T4 are
available as an ablation only (`--label-mode privileged|consistent`). Published work shows that labels from a
teacher with more information can make the student worse.

Measured on 32 fresh TRAIN families (`human-probe`, a code check, not an experiment result):

| Policy | Evacuated | Casualties | Unresolved | Wall/boundary blocks |
|---|---|---|---|---|
| Belief teacher, radius 4, exits seen | 74.9% | 7.0% | 18.0% | 0 |
| Belief teacher, radius 2 | 71.5% | 11.8% | 16.6% | 0 |
| Belief teacher, radius 7 | 80.3% | 7.3% | 12.4% | 0 |
| Belief teacher, one exit known | 80.3% | 7.0% | 12.7% | 0 |
| Belief teacher, all exits known | 84.2% | 5.4% | 10.4% | 0 |
| Belief teacher, no memory | 61.1% | 8.5% | 30.4% | 0 |
| Random legal moves | 16.1% | 40.8% | 43.1% | 0 |
| T1 with the full map (reference) | 87.0% | 3.4% | 9.6% | 0 |

The gap between 74.9% and 87.0% is the price of the human view. It is not a defect of the student.

## 4. Files

| File | Content | Tested here |
|---|---|---|
| `src/ember_v1/human_view.py` | View, line of sight, memory, route estimate, legal-move mask | Yes |
| `src/ember_v1/human_encoding.py` | View → text row for the LLM; view → numbers for a small network | Yes |
| `src/ember_v1/human_teacher.py` | Belief teacher, random baseline, label modes | Yes |
| `src/ember_v1/human_env.py` | Per-person environment and rewards, episode statistics | Yes |
| `src/ember_v1/human_dagger.py` | DAgger collection, datasets, training loop, evaluation, command line | Yes, with the mock student |
| `src/ember_v1/human_ppo.py` | Per-person PPO, small actor, training loop | Yes, small actor, torch on CPU |
| `human_ppo.HumanLLMActor`, `human_dagger.load_adapter("clef")` | The LLM on human views | Verified on the workstation; see the report. |
| `experiments/human_gpu_checks.py` | Workstation check of the LLM path | Verified on the workstation; see the report. |
| `data/rl/human-pov-ppo.draft.json` | Draft PPO settings | Loaded by the tests |
| `src/ember_v1/student_train.py` | One added method: `MockDecisionAdapter.scores` | Yes |

The simulator, the scenario generators, the teachers T0–T4 and the old RL code are unchanged.

## 5. Forked environments: decision

No external environment was forked. Reasons:

- The project rules need an independent simulator with exact EMBERBench movement and fire rules. No public
  environment has these rules.
- The closest public environments are POGEMA (multi-agent grid, partial view) and MiniGrid (egocentric view).
  Their good ideas are small and are now in `human_view.py`: an egocentric window, a sight radius, and cells that
  stay known after the agent moves away.
- A fork would add a dependency and a second simulator to verify. It would not add a rule that we need.

## 6. Steps on the workstation

Use Python 3.11 and the pinned GPU stack (`CURRENT_STATE.md` §6). Write all outputs **outside the checkout**
(for example `/home/darshith/ember-ws/rl-evidence/human-pov/`). Every output path is write-once.

### Step 0 — Get the code and test it (no authorization needed)

```bash
git fetch && git checkout feat/human-pov
python -m pip install -e ".[test]"
python scripts/verify_local.py run --python <python3.11>
python -m pytest tests/test_human_ppo.py        # runs only where torch is installed
```

Expected: the CPU suite passes. `tests/test_human_ppo.py` is skipped where torch is absent and runs 5 tests where
torch is present. Do not continue if a test fails.

### Step 1 — Freeze a plan and get authorization

Write a plan that names: the view settings, the namespaces, the number of rounds and scenarios, the fit settings,
the PPO config and the budget. Copy `data/rl/human-pov-ppo.draft.json` to a new file, set `"status": "authorized"`
only after the owner authorizes that file. Steps 2 to 9 need this authorization.

Edge cases:
- Use namespaces that no earlier run used. A namespace decides the maps.
- Do not use `p1-18-expand@1` or `p1-18-dataset-v2`. They are reserved for V1.1.
- One view setting for the whole run. `fit` refuses datasets with different view settings.

### Step 2 — CPU reference numbers (no GPU)

```bash
python -m ember_v1.human_dagger eval --namespace <ns> --split select --count 32 --kind mock --out <ev>/ref-select.json
```

This gives the belief teacher and the random baseline on selection DEV. These two numbers are the ceiling and
the floor for the student in this view setting.

### Step 3 — GPU checks of the LLM path (no training)

```bash
python experiments/human_gpu_checks.py --out <ev>/gpu-checks.json
```

It must print `"status": "PASS"`. It checks: a human-view row builds one Clef item with no truncation; the score
order is the action order `east, north, south, stay, west`; scores are log-probabilities; one backward pass gives
finite, nonzero LoRA and head gradients. It makes no optimizer step and saves nothing.

Edge cases:
- `FAIL option order`: stop. The action index mapping is wrong; PPO and masking would train the wrong action.
- A truncation error: lower `radius` or raise `max_len`. Never truncate a view.
- Out of memory: the base model alone must fit. Close other GPU processes.

### Step 4 — DAgger round 0 (teacher only)

```bash
python -m ember_v1.human_dagger collect --namespace <ns>-r0 --count 256 --out <ev>/d0 --kind mock
python -m ember_v1.human_dagger fit --data <ev>/d0 --out <ev>/s0 --kind clef --epochs 1
python -m ember_v1.human_dagger eval --namespace <ns> --split select --count 32 --out <ev>/s0-select.json \
       --student <ev>/s0 --kind clef --train-data <ev>/d0
```

- `collect` with no `--student` runs on CPU. It writes `rows.jsonl` and `manifest.json` (row hash, label counts,
  map families).
- `fit` starts from the frozen base with a fresh LoRA and a fresh head. It does **not** start from V1, because the
  input format is new. It writes the checkpoint and `human_fit_report.json`.
- Read `human_fit_report.json`: `train_metrics.legal_rate` is the share of raw choices that are visible legal
  moves. This is the direct check of the old wall-hit failure. Expect it to be near 1.0 after one epoch.
- `eval` reports four arms. `student_raw` is the model alone. `student_masked` adds the legal-move mask; report
  it as a rule-assisted arm, as before.

Edge cases:
- `last_loss` is not below `first_loss`: do not continue. Check the learning rate and the row text.
- `legal_rate` below 0.95: the model does not read the `NEXT` line. Train one more epoch before round 1.
- Label balance: `manifest.stats.labels` shows the action counts. A `stay` share above 50% means many jammed
  episodes; raise `--max-steps` or lower crowd size before more collection.
- `eval` stops with "shares its topology": an evaluation map is a training family. Change the namespace.

### Step 5 — DAgger rounds 1 to 3 (student states, teacher labels)

For round `k` = 1, 2, 3 use beta 0.5, 0.25, 0.0 and a new namespace per round:

```bash
python -m ember_v1.human_dagger collect --namespace <ns>-r<k> --count 256 --out <ev>/d<k> \
       --student <ev>/s<k-1> --beta <beta> --round <k> --kind clef
python -m ember_v1.human_dagger fit --data <ev>/d0 ... <ev>/d<k> --out <ev>/s<k> --kind clef --resume <ev>/s<k-1>
python -m ember_v1.human_dagger eval --namespace <ns> --split select --count 32 --out <ev>/s<k>-select.json \
       --student <ev>/s<k> --kind clef --train-data <ev>/d0 ... <ev>/d<k>
```

- Beta is the probability that the teacher's action is executed. The teacher labels every view in every case.
- `fit` always trains on the union of all rounds. Exact duplicates are dropped. A view with two different labels
  is kept twice and counted (`aggregate.conflicting_states`).
- `--unmasked` lets the student execute illegal moves in collection. Keep the default (masked) unless the plan
  asks for the raw policy.

Edge cases:
- `--resume` needs `optimizer.pt` in the earlier checkpoint. If it is absent, omit `--resume` and train from the
  base on the union (slower, same result class).
- Selection utility goes down between rounds: keep the earlier checkpoint. Do not delete a round. Record it.
- A started round is never rerun in place. The output directory must not exist. Use a new directory and say why.
- The student and teacher agree on almost all rows (`label_equals_executed` near `rows`): more rounds add little.
  Stop DAgger.

### Step 6 — Gate before RL

Continue to PPO only when, on selection DEV:
- `student_raw` wall and boundary blocks are near zero, and
- `student_raw` utility is within about 0.05 of `belief_teacher`.

If the student is far below the teacher, PPO will not repair it. Do more DAgger data or epochs first.
R20–R22 showed this: PPO on a policy that could not move gave no gain.

### Step 7 — Small-network PPO (upper bound, CPU or GPU)

```bash
python -m ember_v1.human_ppo --config <authorized-config.json> --out <ev>/ppo-small --actor small --device cpu
```

This answers one question cheaply: can RL beat the belief teacher from a human view in this simulator?
`summary.json` gives the baseline, every check and the selected update.

- If the small actor beats the belief teacher: its policy is a better teacher. A later plan can use it for labels.
- If it does not: the teacher is near the ceiling for this view. Spend effort on the view (radius, exit knowledge),
  not on more RL.

### Step 8 — LLM PPO from the DAgger checkpoint

```bash
python experiments/human_gpu_checks.py --out <ev>/gpu-checks-actor.json --checkpoint <ev>/s<k>
python -m ember_v1.human_ppo --config <authorized-llm-config.json> --out <ev>/ppo-llm --actor llm --checkpoint <ev>/s<k>
```

Use the LLM overrides in the draft config (`minibatch 16`, `accumulation 16`, `collect_with_grad true`,
`replay_tolerance 0.001`, the `llm` learning rates).

Edge cases:
- `on-policy probability replay differs`: the sampling forward and the training forward disagree. Set
  `collect_with_grad` to true (the R20 fix). Do not raise the tolerance to hide it.
- `kl_early_stop` in most updates: lower the learning rates.
- Known issue from R20: the public `save_pretrained` casts the Clef head to bf16. `HumanLLMActor.save` uses the
  exact saver. The DAgger `fit` checkpoint uses the public saver; check `actor_vs_adapter` in the GPU check.
- The imitation term starts at `imitation_coef` and decays by `imitation_decay` per update. It keeps the DAgger
  policy while the critic is still poor. Set `imitation_coef` to 0 only for an ablation.

### Step 9 — Report and store

- Run `eval --split report` **once**, for the selected checkpoint only, with `--train-data` for every dataset.
- Report all arms in one table. Name `student_masked` and `belief_teacher` as rule-based arms.
- The numbers are not comparable with R19–R22: the task changed (partial view).
- Store weights and raw evidence outside Git with SHA256 manifests. Commit compact summaries only.
- Upload selected checkpoints to private HF so that they do not depend on one workstation.
- The sealed final evaluation stays sealed.

## 7. Rewards and PPO settings

Per-person reward (`HumanRewardConfig`): `+1` evacuated, `−2` casualty, `−0.75` unresolved at the step cap, given
at the step where the person is resolved. Potential shaping on the true safe distance (`eta`). Optional cost per
blocked move (`blocked`). `team` mixes in the mean reward of the group. With `eta = 0` and `team = 0`, the sum of
all person rewards divided by the population equals the old team reward (tested).

The reward and the critic may use the full public state. They are training signals. An actor never receives them.

## 8. Limits and open ideas

- The LLM path passed workstation GPU checks; each new environment still needs steps3 and8.
- The belief teacher is a heuristic. It loses to T1 by about 12 points because of the view, and it queues poorly
  in dense crowds (about one third of its attempted moves are blocked by other people).
- Not modelled yet: smoke that shortens sight, sound or speech between people, panic, different walking speeds.
  Each is a `ViewConfig` or reward extension.
- One training seed gives no variance estimate. Plan three seeds for any claim of improvement.
