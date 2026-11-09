# P1-12 matched five-teacher TRAIN collection — protocol draft

**Status:** `DRAFT_NOT_FROZEN`. Version `p1-12-matched-train@0.3.0-draft` (run 9: pilot runner and freeze candidate, not executed).
**Authorizes nothing.** No TRAIN collection, no labels, no training, no GPU. `multiteacher.collect_fixture`
refuses every scope except `synthetic_fixture`; only `train_pilot.execute` writes scope `p1-12-train-pilot@1`,
and only after every gate of §4 passed. This draft must be reviewed, frozen in a separate commit and
authorized in writing by the owner before any TRAIN scenario is run. The pending T1 selection receipt
(P1-09) does not unlock this protocol.

## 1. What a collection unit is

One **matched group** = one independent TRAIN scenario × all five retained registry identities
(T0 1.0.0, T1 1.0.0, T2 1.0.0, T3 1.0.0, T4 1.2.0), each from a fresh `Simulator(scenario)` with the scenario's own fire
seed, a fresh policy instance and a deep copy of `observe()` only. The group record is the
`matched-multi-teacher@0.2.0` report of `multiteacher.py`:

- full per-step public observation, proposal (complete joint action), transition record, public diagnostics
  (movement outcomes, blocked moves, contested targets, fire adjacency, exit use);
- outcome per teacher: casualties, evacuated, unresolved, casualties+unresolved, causes, steps, termination;
- runtime per decision (collector thread CPU, wall, teacher self-report) — kept out of `content_hash`;
- provenance: simulator and collector module fingerprints, scenario content hash and source, each teacher's
  identity, `code_fingerprint` and registry pin status;
- recorded teacher failures (`teacher_exception`, `invalid_proposal`, `cpu_cap`). A group with any failure is
  `group_complete: false`, is kept, is reported, and is **never** used in a five-way comparison or silently
  turned into a four-teacher or T1-only sample.

Same step index in two trajectories is **not** a common state. `common_state_steps` lists the only shared
states (normally step 0). Same-state disagreement and counterfactual work belongs to P1-13 (#46).

## 2. Source and leakage gate (all must pass before the first teacher call)

| Gate | Rule |
|---|---|
| Source | `data/splits/p1-02-v1/train.json`, manifest `sha256:96a56481…b442d`, spec `sha256:992f8be3…3644`, generator `gen-1.0.0`; regenerate each scenario from the manifest and compare its content hash. |
| Split | TRAIN only. Refuse any scenario, family, lineage or geometry core hash that appears in DEV, calibration, P1-07/P1-09 selection, any pilot namespace (T2/T3/T4/s1-pilot@1), the sealed index or the blind firewall (reuse `s1_pipeline` held-out sets). |
| Identity | `registry.check_pins() == []`; all five identities retained; T4 v1.0/v1.1 blocked identities refused. |
| Code | Clean worktree at a pushed commit; collector, simulator and teacher fingerprints recorded. |
| Output | Write-once directory; every group verified by `verify_fixture`; the manifest of groups hashed. |

## 3. Cost model (planning estimate, not a hard limit)

P1-09 measured CPU over 72 scenarios (`data/selection/p1-09-selection@1/outcome/selection_gate.json`):
T0 7.1 s, T2 **7555.2 s**, T3 134.0 s, T4 68.8 s (T1 is the reference and is cheap, about T0 scale).
T2 dominates: about **105 CPU-s per scenario** on average.

| Unit | Scenarios | T2 CPU (×2 safety) | Others | Wall at 1 process |
|---|---|---|---|---|
| Pilot (§4.4, N-stratified) | 8 | ≈ 0.4 h central, ≤ 1.0 h upper | < 1 min | ≈ 25 min |
| Full TRAIN (`p1-02-v1/train`) | 189 (64 families) | ≈ 11 h | ≈ 10 min | ≈ 11 h, or ≈ 3 h with 4 processes |

This is CPU only, $0 Modal. 189 scenarios in 64 families is a small source; a larger TRAIN generator run is a
separate, later protocol (new lineages, same split firewall).

## 4. Pilot (runner implemented, NOT executed; separate authorization)

Code: `src/ember_v1/train_pilot.py` (`p1-12-train-pilot@0.2.0-draft`), `src/ember_v1/pilot_supervisor.py`
(`p1-12-pilot-supervisor@1.0.0`); tests `tests/test_train_pilot.py`, fakes `tests/pilot_fakes.py`.
Proposed plan: `data/collection/p1-12-train-pilot@1/plan_PROPOSED.json`, status `PROPOSED_NOT_FROZEN`.
Freeze candidate, plan hash and reviewer checklist: `docs/P1_12_PILOT_FREEZE_CANDIDATE.md`.
Authorization template (status `PENDING`): `data/collection/p1-12-train-pilot@1/authorization_TEMPLATE.json`.

### 4.1 Why the plan hash changed (pre-freeze, documented)

The run-8 plan `sha256:07ea910e…c25a5` was never frozen. It is superseded before freeze because:

1. the runner did not exist; the plan now binds the simulator, generator and collector/supervisor/runner
   module fingerprints, so any code change after review changes the hash;
2. teacher policy seeds were missing (T2 20261107, T3 20261123); a run needs them;
3. the T2 cost estimate ignored N (see 4.4); the acceptance limit changed accordingly;
4. a geometry near-duplicate audit and an approval policy were added;
5. `status` is now outside the hash, so freezing does not change the reviewed hash;
6. the stale `commit_at_proposal` field was removed (a commit cannot name itself; the freeze commit binds);
7. CI (Linux) showed that `s1_pipeline.module_fingerprint` gave different digests on Windows: on a
   case-insensitive file system `from .sim import Scenario` added a phantom module `ember_v1.sim.Scenario`
   (resolved to `sim/scenario.py`). Module paths are now case-exact, so every platform computes the same
   identities. The first run-9 hash `sha256:4a188432…c19c` (Windows-only identities) is superseded.
   Teacher code fingerprints were not affected (no phantom module in their closures).
8. CI (Linux, push run of #53) showed one more defect: the five workers shared one `multiprocessing.Queue`.
   Killing one worker (here for `code_drift`) while its feeder thread held the queue's write lock made three
   healthy teachers fail too. The supervisor now gives each worker its own queue, never reads a killed
   worker's queue again (a cut message could block the reader), and reads at most 256 messages per worker
   per poll (a flooding worker cannot delay the cap checks). Regression fake: `flood_then_hang_t2`. Hash
   `sha256:d006a31f…a4e1` is superseded by `sha256:eb8dbe29…b664`.

Scenario IDs, seeds, salt and selection rule are unchanged: the same eight scenarios.

### 4.2 Gates (`run_preconditions`; every reason is reported, any reason refuses)

| # | Gate |
|---|---|
| 1 | Plan read with `git show <freeze commit>:data/collection/p1-12-train-pilot@1/plan.json`; the commit is in `origin/main`; status `FROZEN`; `plan_hash` equals the digest of its body |
| 2 | Live rebuild (`live_plan`) equals the frozen hash: TRAIN manifest, eight scenario hashes, disjointness and geometry audits, five teacher fingerprints and seeds, simulator, generator, collector, supervisor and runner fingerprints, limits |
| 3 | `registry.check_pins() == []` and `check_registry() == []` |
| 4 | Authorization read from git at a commit in `origin/main`; kind `p1-12-multiteacher-train-pilot-authorization` 1.0.0; status `APPROVED`; exact plan hash, freeze commit, eight `{scenario_id, scenario_hash}`, attempt `1` (int), five `{version, config_hash, code_fingerprint}`, limits, 40-hex execution commit; P1-09 records refused |
| 5 | GitHub approval (`verify_approval`): the PR named in the record is merged into `main`, its merge commit is in `origin/main`, the latest non-comment review by an allowlisted login (not the PR author, account type `User`) is `APPROVED`, and that review's commit holds byte-identical authorization content. Free-text names are never evidence. The API failing means refusal |
| 6 | `HEAD` equals the authorized execution commit, is in `origin/main`, descends from the freeze commit; clean checkout (no modified, untracked or unexpected ignored file) |
| 7 | No `attempt-*.json` and no `attempt-*/` exist; attempt must be 1 |

Then `execute` writes `attempt-1.json` with `open("x")` + `fsync` before any worker starts.

### 4.3 Runtime supervision (`pilot_supervisor.run_group`)

| Item | Value |
|---|---|
| Process model | one spawned process and one message queue per teacher per group (five live at most); groups in plan order; a killed worker's queue is not read again |
| Per-decision CPU | OS process CPU since `decide()` began, polled every 0.1 s: T0/T1/T3/T4 5 s, T2 30 s; breach kills that worker |
| Group wall | 7 200 s from group start; breach kills every live worker of the group |
| Pilot wall | 14 400 s absolute deadline; breach kills live workers; no new group starts (`not_started`) |
| Retries | 0; no substitute scenario; a failed teacher is never re-run in that group |
| Failure kinds | `teacher_exception`, `invalid_proposal`, `decision_cpu_cap`, `group_wall_cap`, `pilot_wall_cap`, `worker_crashed`, `worker_unresponsive`, `missing_policy_response`, `code_drift` (worker fingerprint ≠ plan), `identity_drift`, `interrupted`, `supervisor_error` |
| Effect | group `group_complete: false`; every streamed decision kept and replay-verified; never training-eligible |
| Source of truth | the supervisor's own simulator replays each streamed proposal; a stream that does not replay is `integrity_error` (collector fault), not a teacher result |
| Regeneration | the two smallest-N groups (`g-0821b9cfcfbdc9a9`, `g-b9f6d81f3b5bfce7`) run again after the eight, same caps; content hash must match |
| Interrupt | evidence written, marker kept, `KeyboardInterrupt` re-raised; the marker blocks every later run |

### 4.4 T2 cost, re-estimated by people count

P1-09 per-scenario T2 CPU (72 scenarios) in the same N bin as each pilot scenario:

| Bin (N) | Pilot scenarios | P1-09 n | mean s | max s |
|---|---|---|---|---|
| 0–11 | N 6, 8 | 22 | 5.8 | 33.5 |
| 12–39 | N 24, 33 | 23 | 57.3 | 223.4 |
| 40–99 | N 46 | 17 | 113.1 | 667.5 |
| ≥ 100 | N 114, 115, 120 | 10 | 418.8 | 847.8 |

Central **1 496 CPU-s**, upper (acceptance limit) **3 725 CPU-s**, others ≈ 43 CPU-s, expected wall ≈ 25 min
(T2 dominates; teachers run in parallel). The run-8 figure 1 679 s (global mean × 8 × 2) was close in total by
chance; its per-scenario gate (209.9 s mean) would likely have failed on the three N ≥ 100 scenarios, which
are about 4× the global mean. Worst case (every decision at the 20 s T2 budget for `max_steps`) is
24 500 s > 14 400 s pilot wall: then the caps end the pilot and the gate fails. In P1-09, 3 of 10 N ≥ 100
scenarios had T2 budget-limited decisions; such a decision depends on host speed, so that group may not
regenerate. The preregistered regeneration groups are the two smallest, which had none.

### 4.5 Commands

```
PYTHONPATH=src python -m ember_v1.train_pilot plan --out data/collection/p1-12-train-pilot@1/plan_PROPOSED.json
PYTHONPATH=src python -m ember_v1.train_pilot verify-plan --plan data/collection/p1-12-train-pilot@1/plan_PROPOSED.json
PYTHONPATH=src python -m ember_v1.train_pilot template --plan data/collection/p1-12-train-pilot@1/plan_PROPOSED.json --out data/collection/p1-12-train-pilot@1/authorization_TEMPLATE.json
PYTHONPATH=src python -m ember_v1.train_pilot run --plan-ref <freeze>:data/collection/p1-12-train-pilot@1/plan.json --auth-ref <main commit>:data/collection/p1-12-train-pilot@1/authorization.json
```

The first three run no teacher. `run` refuses today (no frozen plan, no authorization).

Acceptance gate (`pilot_acceptance`): exactly the 8 planned scenarios; 8/8 verified complete groups;
incomplete rate 0 in every stratum; leakage check per group; both preregistered regenerations identical; T2
CPU total ≤ 3 725 s, else re-plan before full TRAIN; group and pilot wall caps held. Failed groups are kept
and reported. No substitution, no retry. The pilot never extends past eight scenarios. Eight cases support
no population-level teacher claim.

### 4.6 Trust boundary

The approval check binds the record to a GitHub account action (an `APPROVED` review on a merged PR by an
allowlisted account that is not the author). It cannot tell the owner from any process that holds that
account's token on this machine. The agent that prepared this plan holds the `sankalphs` token and must not
create, approve or merge the authorization PR. For stronger binding, require a hardware-backed signed commit
or a reviewer account whose token is never on the execution host.

## 5. What this data may be used for

Descriptive five-teacher comparison and the source for P1-13 common-state prefixes. It does **not** by itself
create supervised targets: eligibility of a demonstration is decided by the frozen P1-14 dataset protocol
(#47, `docs/P1_14_MULTITEACHER_DATASET.md`), never by this collector (`supervision_eligibility: not_assessed`).

## 6. Open review questions

1. One fire seed per scenario is accepted for this feasibility pilot. A second seed per scenario belongs to the
   full-TRAIN protocol (fire-outcome variance), not to this pilot.
2. Answered (run 9): 105 CPU-s/scenario was not representative. The plan uses a people-stratified estimate
   (§4.4: central 1 496 s, upper 3 725 s).
3. Answered (run 9): every teacher runs in its own killable process. Internal budgets (T2 20 s, T3 3 s) are
   not a substitute: T0/T1/T4 have none, and no internal budget stops a hang or an infinite loop.
4. Owner decision: the approver allowlist (`sankalphs`, `Sathvikar01`, the two repository admins on
   2026-11-09). GitHub forbids self-approval, so the record needs a PR by one account approved by the other.
