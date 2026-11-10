# P1-12 eight-scenario five-teacher TRAIN pilot — freeze candidate

**Status: FREEZE CANDIDATE FOR OWNER REVIEW. NOT FROZEN. NOT AUTHORIZED. NOT EXECUTED.**
Prepared in run 9 (2026-11-09) on PR #50 (issue #45). Protocol: `docs/P1_12_TRAIN_COLLECTION_PROTOCOL.md` §4.

Proposed plan: `data/collection/p1-12-train-pilot@1/plan_PROPOSED.json`

**Proposed plan hash: `sha256:cb93c6d17e5b6c74755ca3a569b43955916302805983e39b562e834f6a9a605a`** (plan 0.3.0, run 15; supersedes `sha256:eb8dbe293db7f1c1e1b4f342c07f21853031e9e6ab5419eda2833c51f761b664` of plan 0.2.0 — only the approval policy and the runner fingerprint changed, see `docs/P1_12_PILOT_FREEZE_RECORD.md` §Re-freeze 0.3.0)

`status` is outside the hash: the FROZEN copy keeps this hash. `tests/test_train_pilot.py` rebuilds the plan from
the current code and data and fails if the committed plan differs (no stale hash). Run-8 hash
`sha256:07ea910e…c25a5` and the run-9 hashes `sha256:4a188432…c19c` and `sha256:d006a31f…a4e1` are superseded before freeze; reasons in protocol §4.1. Same eight scenarios.

## 1. Scenarios (exactly eight, TRAIN `p1-02-v1/train`)

| scenario_id | scenario_hash | family | label | fire | N | max_steps |
|---|---|---|---|---|---|---|
| g-a2993d188ad45093 | sha256:0e85ca6079f2002b7706c2cc43796fc47ae42098af324d0d5334ffa03c641108 | fam-a30f0e46d2991ba4 | feasible | stochastic | 120 | 300 |
| g-6dfef26f38a144d0 | sha256:99fa0d0409db18edb77b86fe6c0a31bea365380169be04b933d9ca316f3fc8ff | fam-b31994dc6f5d7f16 | difficult | deterministic | 33 | 141 |
| g-b9f6d81f3b5bfce7 | sha256:8f871dec96187985b941894b131f23281edaf0f755c3fc352e89ca32ee8918c3 | fam-ace58bfa33b4b50c | feasible | deterministic | 8 | 186 |
| g-35cca9f1d42320be | sha256:6333c75294fbaa6c19e904ae1feb8c1cc64c177f8cd4e344206a63d7ef0d8f6d | fam-f388fe3b7e57ed9b | unreachable | deterministic | 24 | 169 |
| g-0821b9cfcfbdc9a9 | sha256:c43b142da86e55a9cd8d9e120ba885ed3044a6b7eb0af6a1f7b712b947e5bb84 | fam-bf7d52a86541e7be | difficult | stochastic | 6 | 57 |
| g-0ee55267388889cc | sha256:45e5807ee9a0eaf9f5f38fbdc5e82b3463f35d65251fb953a05f618c34172294 | fam-349e5a71b9f50060 | feasible | deterministic | 114 | 129 |
| g-1a8b6230b6107b0e | sha256:07635cf78bd747464e6e97be8b6260240bb051ea05625df084c9044c44eabf75 | fam-514ba4d9f9c920d9 | feasible | deterministic | 46 | 109 |
| g-5082bdcde677aea4 | sha256:54f05807b587a5e7557f1e90b9b488f688d63c1c338def7cf8fd051ec1497c7f | fam-59e9dcf5b2a3a27a | difficult | stochastic | 115 | 134 |

Design (recomputed in `plan.design`, pinned by a test): 8 families; 4 feasible, 3 difficult, 1 unreachable;
5 deterministic, 3 stochastic; N = 6, 8, 24, 33, 46, 114, 115, 120; one group and one fire seed per scenario.
Selection inputs: `scenario_hash`, `label`, `family_id`, `fire_model`, `people` only. No teacher outcome on
any TRAIN scenario exists or was read (a test scrambles every other field and gets the same eight).
This is a pipeline feasibility pilot. Eight cases support no population-level teacher claim.

## 2. Generator and simulator identities

| Item | Value |
|---|---|
| TRAIN manifest | `sha256:96a564819bdc6910708cb2891b043ae3fa6a3d9066d5927ba721aa61d44b442d` |
| Spec / split plan | `sha256:992f8be3…3644` / `sha256:94ef33df…afaa` |
| Generator | revision `gen-1.0.0`; `gen.generator` `sha256:58335ff4…7a0a`; `gen.splits` `sha256:d1ed4077…2e45` |
| Simulator | observation `1.0`; `sim.simulator` `sha256:4c961d54…6d2d` (includes the evaluator-only `fork_fire` of P1-13, moved here so the identity is final for the whole stack) |
| Collector | `matched-multi-teacher@0.2.0`; `multiteacher` `sha256:061eb525…6d16` |
| Supervisor | `p1-12-pilot-supervisor@1.0.0`; `sha256:5920da88…212a` |
| Runner | `train_pilot` `sha256:d9839da8…5d79` |

Full values: `plan.identities`. Fingerprints are digests of the module and its transitive `ember_v1` imports
(LF-normalized).

## 3. Five teacher identities

| Teacher | Version | config_hash | code_fingerprint | seed |
|---|---|---|---|---|
| T0 | 1.0.0 | sha256:bdc4d3580bcf3889348c563aaf622334d135a535d9cb2aaeedc3c0528f10208a | sha256:8ddf9370263284107ae5249c9adafe7323ed8deab2d45077fe9894979e6c84aa | — |
| T1 | 1.0.0 | sha256:5047e942db3326706b6184c9f29e0b992078920f03c8a53adc8cc24258345361 | sha256:52dc61f4ff9599e55bfd29f47068bf301895321ce85f6c9aaa9feec1dc568eac | — |
| T2 | 1.0.0 | sha256:302139e18ba05fdcdc5561b7390b95f725600ec8791e336d065f77cea0c507e5 | sha256:afe987d862caaa32701777050f19fb3126c03f9edd6ab8ac6ed375a95504db74 | 20261107 |
| T3 | 1.0.0 | sha256:5a8d4985b1f65d5de18a06c86d8dd522e2ca95e2accd583165b7e7d7ea1d02e0 | sha256:82294a3f0d7aa3caf38598505dbffd367c5cb14ade1f725fd9ee23e73c2c14f1 | 20261123 |
| T4 | 1.2.0 | sha256:79bd3322b9cf7ccd25928f3a70502089732496af5427f88f00ecd2ff6ba12504 | sha256:fa66a0f354dc82f992d1f0a1896851fc90dc9e19dea244f815a76857361ce75e | — |

`registry.check_pins() == []`. Blocked T4 1.0.0 and 1.1.0 are refused by the registry. Every worker reports its
own code fingerprint; a mismatch is a `code_drift` teacher failure.

## 4. Leakage audit

| Check | Result |
|---|---|
| ID disjointness (family, lineage, scenario, core hash; whole TRAIN manifest: 64 families, 189 scenarios) vs DEV, calibration, locked_test (regenerated), P1-09 selection, t2-pilot@1, t3-pilot@1, t4-pilot@1/2/3, t4-explore@v12 | 0 overlaps |
| Geometry: the 8 maps vs 420 held-out maps (56 p1-02 held-out families, 24 P1-09 families, 340 pilot maps): topology (8 symmetries + exit renaming), near-duplicate, wall near-duplicate and both scale-free variants | 0 hits |
| Geometry within the pilot (28 pairs) | 0 hits |
| Nearest held-out maps (not duplicates under the frozen rules) | g-0ee55267388889cc vs t4-pilot@1 open_room 48×64: wall 0.019, exits 0.114 (exit rule 0.10); g-5082bdcde677aea4 vs t4-pilot@2 open_room 48×64: wall 0.050 (rule 0.03), exits 0.056. Sparse open rooms; the wall-only rule does not apply below 5 % walls by design |
| Policy input | exactly the public observation keys; no private key anywhere (checked per stored observation) |
| Disclosed TRAIN-internal overlap | g-0821b9cfcfbdc9a9 is also one of five scenarios of `s1-pilot@1` (draft PR #43, `FROZEN_SELECTION_NOT_RUN`, T1-only, obsolete after the pivot). Both are TRAIN, no outcome exists, so it is not leakage. **Owner decision (run 10, 2026-11-09): accepted** only while (a) `s1-pilot@1` has no outcome evidence (checked: every remote branch holds only its selection `pilot.json`, status `FROZEN_SELECTION_NOT_RUN`), (b) `s1-pilot@1` stays permanently blocked (never authorized, never run; PR #43 is superseded), (c) neither pilot's data enters any validation or test set, (d) the overlap stays documented, (e) it is never counted as two independent sources. Scenario identities are not changed for this overlap |

## 5. Runtime caps and failure policy

Per-decision CPU 5 s (T0, T1, T3, T4), 30 s (T2); group wall 7 200 s; pilot wall 14 400 s; one killable
process per teacher; 0 retries; no substitutes. Failure kinds and effects: protocol §4.3 and
`plan.failure_policy`. A failed group is kept, replay-verified as partial evidence and never training-eligible.

## 6. Attempt marker and cancellation

`data/collection/p1-12-train-pilot@1/attempt-1.json` is created with `open("x")` + `fsync` after every gate and
before any worker starts. It holds the plan hash, the authorization commit, path and sha256, the execution
commit, limits and platform. Ctrl-C: workers are killed, the group and evidence are written, the marker stays,
`KeyboardInterrupt` is re-raised. Any marker or `attempt-*/` directory blocks every later run. Attempt 2
needs a new plan version and a new authorization.

## 7. Regeneration procedure

1. Plan: `PYTHONPATH=src python -m ember_v1.train_pilot verify-plan --plan data/collection/p1-12-train-pilot@1/plan_PROPOSED.json`
   (rebuilds from code and data; prints `same: true`).
2. Scenarios: `train_pilot.pilot_scenarios` regenerates the eight from the TRAIN manifest and checks scenario,
   lineage and family hashes.
3. Groups: `multiteacher.verify_fixture(report, scenario)` re-executes every recorded joint action without
   running a teacher and re-hashes the report.
4. Teachers: the two preregistered groups (`g-0821b9cfcfbdc9a9`, `g-b9f6d81f3b5bfce7`) are re-run inside the
   attempt; `content_hash` must match. Any other group can be re-run later with `multiteacher.regenerate`
   (diagnostic only; a T2 budget-limited group may differ by design).

## 8. Expected output schema

`attempt-1/groups/<scenario_id>.json`: `matched-multi-teacher@0.2.0`, scope `p1-12-train-pilot@1`,
`training_eligible: false`, `supervision_eligibility: not_assessed`, five policies in T0–T4 order with every
observation, proposal, transition, diagnostics, outcome or failure; `pilot {plan_hash, scenario_id, attempt}`;
`runtime {run, supervision}` (excluded from `content_hash`). `attempt-1/regen/<id>.json`, `attempt-1/log.jsonl`,
`attempt-1/evidence.json` (`p1-12-train-pilot-evidence@1.0.0`: group rows, regeneration, acceptance).

## 9. Estimated CPU cost

T2 central 1 496 CPU-s, upper 3 725 CPU-s (acceptance limit), worst case 24 500 s (caps act first); others
≈ 43 CPU-s; expected wall ≈ 25 min; $0 Modal. Basis and caveats: protocol §4.4.

## 10. Plan hash

`sha256:eb8dbe293db7f1c1e1b4f342c07f21853031e9e6ab5419eda2833c51f761b664`

## 11. Independent reviewer checklist

- [ ] `verify-plan` prints `same: true` for this hash on a clean checkout of the freeze candidate commit.
- [ ] The eight IDs and hashes above equal `plan.scenarios`; design counts match §1.
- [ ] No teacher outcome on these TRAIN scenarios exists in the repository history.
- [ ] Leakage audit (§4) reviewed, including the two nearest open-room maps and the s1-pilot@1 overlap.
- [ ] Teacher fingerprints equal `registry` pins; T4 is 1.2.0; blocked identities refused.
- [ ] Caps, failure policy, no-retry and no-substitute rules acceptable.
- [ ] T2 estimate and acceptance limit acceptable; worst case understood.
- [ ] Approver allowlist (`sankalphs`, `Sathvikar01`) correct; trust boundary (protocol §4.6) accepted.
- [ ] `tests/test_train_pilot.py` passes, including crash, hang, silent exit, invalid action, code drift,
      forged stream, interrupt, tampered plan, missing approval and existing-marker cases.
- [ ] Complete local suite PASS on Python 3.11 and 3.12 on the exact head (`scripts/verify_local.py run`,
      receipt in the PR; `AGENTS.md`). GitHub CI was removed in run 10 and is not evidence.

## 12. Steps after review (owner only; the agent does none of them)

1. Freeze: copy `plan_PROPOSED.json` to `data/collection/p1-12-train-pilot@1/plan.json` with
   `"status": "FROZEN"` only, commit, merge to `main` through a reviewed PR. That commit is the freeze commit.
2. Authorization: fill `authorization_TEMPLATE.json` as `data/collection/p1-12-train-pilot@1/authorization.json`
   (plan freeze commit, execution commit, approver login, PR number, status `APPROVED`); open a PR from one
   admin account; the other allowlisted account approves it on GitHub; merge it.
3. Run on a clean checkout of the execution commit (CPU only):

```
git checkout <execution commit>
PYTHONPATH=src python -m ember_v1.train_pilot run --plan-ref <freeze commit>:data/collection/p1-12-train-pilot@1/plan.json --auth-ref <authorization merge commit>:data/collection/p1-12-train-pilot@1/authorization.json
```
