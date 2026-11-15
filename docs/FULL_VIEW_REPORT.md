# Full-view run stopped at the fit gate

The run stopped on `dagger-r3-fit-01` at 2026-11-15T10:23:03Z. The trainer exited0 and saved its weights,
but the unchanged completion verifier rejected the last-group loss. PR #1 stays draft and open; no merge
is authorized with this failed gate. No final candidate was frozen.

The stopped run does not establish that the new full-view model is better than original V1; no direct report comparison ran. PPO did not run, so no PPO gain over DAgger is established.

| Frozen report arm | Outcome | Paired interval |
|---|---|---|
| New model, raw | NOT RUN: stopped before candidate freeze | Unavailable |
| New model + legal mask | NOT RUN: stopped before candidate freeze | Unavailable |
| DAgger before PPO, raw | NOT RUN: stopped before candidate freeze | Unavailable |
| T1 teacher | NOT RUN: stopped before candidate freeze | Unavailable |
| Random legal moves | NOT RUN: stopped before candidate freeze | Unavailable |
| Original V1, raw | NOT RUN: stopped before candidate freeze | Unavailable |
| Original V1 + legal mask | NOT RUN: stopped before candidate freeze | Unavailable |

The planned 128-family report panel remains unspent. No report invocation, paired bootstrap interval,
small-network PPO or LLM PPO ran. The pre-PPO gate was not reached. The planned 5,000 paired family
resamples cannot be reported without matched report outcomes. None of these missing values is zero.

## What completed

The model uses one row per person: the complete public map, all people, fire and exits, that person
marked, their last action/outcome and the four neighbour-cell statuses. It contains no hidden state or
route hint. Action order is east, north, south, stay, west. T1 was chosen on eight TRAIN reference families
by the frozen utility/casualty/name rule; T1/T2/T4 tied. Token preflight checked 5,684 complete rows on 256
TRAIN geometries, maximum 628 against context 16,384, without truncation. Optimizer-free GPU checks passed
before fitting. Simulator, generators, T0–T4 teachers, hashed handoff files and inherited tests stayed unchanged.

| Round | New TRAIN families | Rows | Beta | Student-only agreement | Rows SHA256 |
|---|---:|---:|---:|---:|---|
| d0 | 256 | 27979 | 1.0 | n/a (teacher only) | `39f566261d08ad3d90aa4b685e128424f2e0879b41ecd9fcd19a476bc26f53fa` |
| d1 | 256 | 37677 | 0.5 | 0.6773336168403147 | `f68d9f413603c526d278f997879c49938f31bc753322df847335a98dfbd8b7fa` |
| d2 | 256 | 42005 | 0.25 | 0.6139811852649584 | `b7ee6c9995fa0091f77e2b377a026b1c0d361035fc24632c419677d014eb7337` |
| d3 | 256 | 53639 | 0.0 | 0.41604056749752977 | `d783975988fd58a42ae81dce6ec5f439d060eeae7f491c9869edfcbd20c51c0f` |

Each round used 256 new TRAIN families; topology hashes are disjoint from all held-out panels. The .99
DAgger stopping rule counts only student-executed decisions; it was not attained. Every fit used one
epoch on the deduplicated union, shuffle seed 0, accumulation 8, LoRA LR .0002, head LR .0001, warmup .03.

| Fit | Union rows | Optimizer steps | First group loss | Last group loss | Completion gate |
|---|---:|---:|---:|---:|---|
| s0 | 27979 | 3498 | 1.603625089 | 0.794675581 | PASS |
| s1 | 65656 | 8207 | 0.233472679 | 0.213736302 | PASS |
| s2 | 107661 | 13458 | 1.211377739 | 0.255948972 | PASS |
| s3 | 161300 | 20163 | 0.741404975 | 1.771141805 | FAIL |

The failed s3 union has 161,300 rows and 20,163 optimizer steps. All loss groups were finite. The first
group averages 8 shuffled examples and the final partial group averages 4 different examples. Their
comparison fails the configured completion condition; it does not by itself establish that the full
training objective worsened. A separate TRAIN-only2,000-row probe had 48.25% agreement and 100% static
legality. That probe is not a selection or report score. No code/hash/config fault was identified that
justifies another training attempt. Changing the gate, another epoch, a seed retry or evaluating an older
snapshot to bypass the stop would violate the frozen rules, so none was done.

| Raw SELECT snapshot | Evacuated | Casualties | Unresolved | Utility | Blocked / attempt | Wall / boundary |
|---|---:|---:|---:|---:|---:|---:|
| s0 | 124/360 | 96 | 140 | -0.480556 | 72.15% | 0 / 0 |
| s1 | 225/360 | 57 | 78 | 0.145833 | 41.56% | 0 / 0 |
| s2 | 230/360 | 44 | 86 | 0.215278 | 46.77% | 0 / 0 |
| s3 (rejected fit) | NOT RUN | — | — | — | — | — |

These are the already-completed32-family SELECT results. T1 evacuated 338/360,
with 0 casualties and utility 0.893056. Best completed SELECT snapshot:
`s2`; this was never promoted to a frozen final candidate. Static legality does not
resolve crowd occupancy, conflict and cycle blocks. There is no direct comparison with original V1.

## Failure ledger and audit

- first-head verification setup: Receipt creation failed; first log also reported one untracked change. Pytest itself returned 1458 passed,206 skipped,10 xfailed. Cause: External verification-state symlink target was not created; external log directory symlink was not excluded in local Git metadata. Repair: Created a fresh state target and a new verification directory; added only /.local-test-logs to .git/info/exclude. No source/test input changed. Check: New named first-head-02 clean DEFAULT verification PASS; receipt and log hashes in first-push.json.
- new uncommitted full-view test-fixture development: 8 failed,8 passed in first focused run; errors occurred at test fixture construction and teacher construction. Cause: New fixtures incorrectly treated Simulator.observe() as an object with .to_dict(); the new teacher wrapper incorrectly passed seed to deterministic T1, whose retained constructor takes no seed . Repair: Corrected only newly authored, uncommitted fixture calls to the established dict API; preserved their assertions. Production teacher factory now inspects the retained constructor and supplies seed only when supported. No inherited test changed. Check: Full-view and inherited human PPO focused check:16 passed; with evidence/supervisor and human DAgger checks:35 passed. All new tests committed at32a10f42.
- Linux CPU handoff prototype: Pinned Python exposes neither os.pidfd_open nor signal.pidfd_send_signal. Cause: Those optional wrappers were not compiled into this interpreter; runtimeglibc2.39 has both functions. Repair: Use ctypes bindings to the same libc pidfd functions, preserving process-identity protection. Check: ActualLinux CPU lifecyclecheckPASS: parentheld/childnormalexit/noadvance/reaped.
- WMI boundary launcher setup: First Create rejected ProcessStartupInformation with Type mismatch; no process was created. Cause: The generic CIM startup instance used an untyped ShowWindow property. Repair: Build from the actual CIM class and pass uint16 ShowWindow0; ReturnValue0, PID43424. Check: Existing-launcher check confirmednone before the corrected invocation.
- Boundary handoff01 attachment: Startupguard rejected because collection had completed normally before attachment. Cause: Administrativepreparation overlapped phasecompletion; oldsupervisor legitimately advancedtofit. Repair: Preserve setupfailure/log and attach a new handoff02 to the currentfit after expanding the same verified monitor for fitcompletion. Check: No parent-held record; parent4320remainedrunning; d1collectPASS and fitPID22218live. No experimentrelaunched.
- Handoff02 launcher bookkeeping: Exclusive metadata creation failed after monitor spawn because generated launcher02 retained launcher01 unsuffixed filename. Cause: Mechanical replacement did not rename boundary-handoff-launch.json. Repair: Recovered actual monitor23876 from its own start record, verified /proc command identity, parentheld T and originalfit alive, preserved launcher source and recovery receipt, did NOT relaunch. Future launchers must reserve UNIQUEmetadata BEFORE spawning. Check: Actualmonitor command verifiedlive; parent-held record, heartbeat, fitPID22218live andGPUactive. Completionresults remainproduced by the independentmonitor.
- Original V1 control bridge, caught before reporting: Two CPU cases raised AttributeError; the wrapper called ArmPolicy.act, which does not exist. Cause: The new wrapper used the wrong API for the inherited original V1 policy. Repair: Call decide(obs).actions. Preserve the live original fit to normal completion via a pidfd boundary handoff, then activate a verified control-only runtime supplement before any report. Check: Real original encoder and ArmPolicy with CPU mock adapter: two RED cases, then five GREEN checks including full DAgger; final complete control/protocol checks49passed. No real report invocation was consumed.
- Boundary cleanup race: The new cleanup regression raised FileNotFoundError when the child was already reaped. Cause: Cleanup assumed the child proc entry survived parent retirement. Repair: Treat a missing already-reaped child as terminal while still retiring the controller and closing pidfds. The active archived monitor was not hotpatched. Check: test_reaped_child_does_not_interrupt_controller_retirement_or_descriptor_cleanup observed RED then GREEN; final49checks passed. Actual archived handoff02 independently completed PASS.
- Strict continuation CPU fixtures: New RED cases exposed missing source-amendment validation, completed-record reconstruction and live amendment-hash enforcement. Cause: The original supervisor intentionally supported only its initial write-once run, so no safe continuation protocol existed. Repair: Validate a frozen control-only supplement, source/archive/plan/auth/command/result/log/output hashes and the terminal original controller; read eight completed phases without launching them or charging their time twice. Check: amendment/resume/live-hash RED and GREEN logs retained; complete CPU control/protocol49passed; exact clean Windows and WSL DEFAULT1519passed206slow-skipped10xfail,exit0.
- Controller source formatting gate: A local source formatting check failed before push due to CRLF normalization in new controller code. Cause: Windows source creation preserved CRLF in files checked by the formatting audit. Repair: Normalize the new controller files to LF in commit8fb33d6. Preserve the preceding local commit1e608bf; no failed-gate push occurred. Check: Clean exact-head Windows and WSL DEFAULT receipts on 8fb33d6 PASS; all source-control diffs remained control-only.
- Unexecuted package snapshot regression after adding credential audit: The unchanged snapshot harness selected the newly inserted credential loop and raised NameError before exercising copying. Cause: Its AST selector identifies the top-level for-p-in-files payload-copy block; the added audit loop initially used the same iterator spelling. Repair: Use the descriptive source_file variable in the credential audit, leaving the original copy block and test unchanged. Check: The unchanged snapshot regression passed again; package helper compiles.
- Unexecuted publication helper start-marker audit: CPU fixture rejected a completed report because report-candidate_raw.result.json does not exist. Cause: An overly broad start-marker glob confused the child one-shot arm guard with the parent supervisor phase. Only the parent has a result, with an -01 suffix. Repair: Validate the arm guard schema, panel and frozen-candidate hash, then require the existing report-arm-01 PASS/exit0 result. Never manufacture a new result. Status helpers use only actual phase markers for live-process checks. Check: Actual packaging guard extracted via AST: RED1failed assertion, then GREEN1passed. Independent package-copy snapshot check also PASS. No actual publication or report invocation ran.
- Unexecuted publication helper snapshot: RED: actual payload-copy block hardlinked source evidence, so a later durable-note write changed packaged bytes Cause: Hardlinked inodes made the package mutable through its source paths Repair: Copy each payload file to an independent inode before sealing Check: check_package_snapshot.py executes actual AST copy block: observed RED assertion, then GREEN unchanged packaged bytes and distinct inode
- Isolated CPU dependency preparation: python -m pip check: no pip module Cause: UV-managed CPU environment does not include pip Repair: Use installed uv pip check --python, which passed Check: Corrected focused invocation: 10 passed, 0 failed, 0 skipped; uv pip check: 18 compatible packages.
- Isolated CPU dependency preparation: Direct focused pytest collection:3 import errors for ember_v1 Cause: Noneditable isolated clone requires its src in PYTHONPATH, which verify_local supplies Repair: Set explicit PYTHONPATH to that clone src;10 passed, no source/test assertion changes Check: Corrected focused invocation: 10 passed, 0 failed, 0 skipped; uv pip check: 18 compatible packages.
- Windows CPU completion-verifier import: Two new regression cases failed on the real Windows CPU environment before the fix. Cause: The Linux-only fcntl GPU-lock module was imported unconditionally, preventing CPU-only module use on Windows. Repair: Move fcntl imports into GPU lock/unlock branches. The fix was developed and verified in an isolated clone while the active fit source stayed unchanged, then integrated after that fit and the controlled handoff finished. Check: Existing CPU supervisor tests plus new regressions: 4 passed in 2.66s; clean DEFAULT gate: 1474 passed, exit0. RED/GREEN logs and patch bundle retained.
- Completion callback transport: The filtered probe received HTTP403 before any completion trigger was used . Cause: The relay did not permit the external request. Repair: Disabled the webhook and enabled an internal same-thread continuation; no experiment phase was restarted. Check: Probe status and scheduler task state retained in supervisor/callback-probe-01.json and scheduler-state.json.
- Round3 fit completion: Child exit0, but last group loss 1.771141804754734 exceeded first group loss 0.7414049749495462; the controller stopped with exit1. Cause: The unchanged completion criterion compares the first and last shuffled minibatch-group loss. The last group is larger, so it fails. These are different examples (8 versus4 rows), not a matched before/after objective; this alone cannot establish overall convergence or lack of it. No code/hash/config fault was identified that justifies another training attempt. Repair: None under current frozen gates. A revised diagnostic/optimization experiment requires a new frozen plan and authorization. Check: The exact unchanged CPU completion verifier reproduces AssertionError on the retained full-fit.json. All20,163 loss groups and all saved floating tensors are finite; dataset/config/log hashes match. No retry or alternate gate was used .
- Read-only status helper after experiment stop: The helper asserted on the retained FAIL phase. Cause: The helper was prepared for active successful phase records and asserted every result PASS. Repair: Used actual stopped/result/launcher records and an independent read-only terminal audit. No experiment source or attempt marker changed. Check: stopped-run-audit-01.json reproduces the gate and verifies no related live processes and zero GPU compute processes.

The original controller was retired at a verified boundary after its existing r1 fit completed normally.
The new control-only runtime read and reverified eight PASS records, then started only unstarted phases.
No experiment attempt or report arm was restarted. Both controller stop records remain preserved;
the historical controlled transition is separate from this genuine failed completion gate.

Independent audit reproduced the unchanged CPU verifier failure, verified phase logs/config/dataset
hashes and topology isolation, and checked all saved checkpoint/optimizer tensors for finite values.
The rejected s3 weights remain in evidence and the registry, clearly marked FAIL. Controller/launcher
and phase processes are terminal; GPU compute processes0. No sealed FINAL, hidden benchmark, EMBERBench,
reserved V1.1, cloud computation/spend, public resource, release tag or announcement was used .

## Unresolved implementation review blockers

The one fresh read-only review found three important defects in paths that did not run. They remain
unresolved and independently block merging or reusing this implementation for another experiment:

- The entry points do not enforce the frozen evidence-root and attempt-marker-root declarations. An
  unchanged plan copied elsewhere could establish a fresh marker namespace. Bind resolved paths and
  test relocation rejection before any further collection or reporting.
- Reporting verifies frozen file tables but can load an unrestricted `--student` path; original V1
  files are not rehashed immediately before loading. Bind each arm to its frozen checkpoint and test
  wrong-path/tampered-byte rejection before consuming an invocation.
- The cross-seed selector adds raw-parent utility/casualty conditions after the declared masked-parent
  guard. Reconcile this against an explicitly frozen selection rule and test differing raw/masked
  baselines before another PPO run. No guard or tolerance was relaxed during this closeout.

The actual run used the declared root, reverified eight completed records without repeating them, and
stopped before all three affected paths. These findings do not invalidate the retained phase hashes or
make the failed fit pass. They are retained as draft-PR blockers rather than launching unrelated latent-path
development after the experiment stop. Future reuse without correction risks duplicate invocations,
wrong report weights or a selection rule that differs from the plan. The review also deferred minor
spacing cleanup in generated historical ledger prose. Full disposition: external
`verification/final-branch-review-01.json`. No second review or new experiment was run.

## Identity, storage and budgets

Original plan SHA256 `389ff09bd8950d7fe9e0cfb77fd5b2f16f2482de8a3e2cfeb2c20b705749c8d6`. Control-only runtime supplement SHA256 `fbf97cd181b2412d89c0e7494e3f63df76a2a6f42a2bd8024ef1a96a4d15398c`. Controller source `8fb33d67738fc018ef8e3861ec4adf5b6b4f3320`. Per-phase producing sources and result/log hashes are in the compact summary and checkpoint registry.
Stop audit SHA256 `5037bb8226870a351c73e9ab582e7cca0be7e01ba7cfebf61a8a5a224b514b3b`.
Failed fit JSON SHA256 `79575f1732f1014919b71eab8fd3e2eaa1b5bfcc574aee7ee1169bfccd8d82b2`.
Failed phase result SHA256 `defab742179f92bc21d498f13d3554d91474e95a94651ee8382801d249c20175`.
External evidence: `/home/darshith/ember-ws/ember2-evidence/2026-11-14/full-view-20261114T133215Z-37025f`. Compact summary: `data/full-view/ember2-full-view-full-view-20261114t133215z-37025f/summary.json`.

Upload pending: HF_TOKEN is absent from the environment. No new HF repository was created and no HF revision or private-visibility verification is claimed for ember2-v1-models. The intended destination is a private model repository under the HF_TOKEN account. The credential pasted in chat was not used .

The local stopped-core package contains 637 files and 2903030884 bytes;
manifest SHA256 `940ffed4063b6eab7cfd5ddc3d7974f4957d3044fd0fcdde79f8d2e7a0da456c`. Its independently copied snapshot preserves all produced
checkpoints, rows, raw logs, source archives and core diagnostic records. Final Git documents and exact-head
test receipts are indexed in the external final ledger, without modifying the sealed core manifest.

Runtime cost: 19.667993 GPU hours and 0.671587 charged
CPU hours through stop, against 60/16 hours. Runtime CPU already includes the749.916331-second source-transition
ledger and all earlier manual portability/preparation costs; they are not added twice. Package audit used
27.579646 additional CPU-category wall seconds. Final documentation, local tests and audit
costs are reconciled separately in the external final ledger. Evidence is below150GiB; cloud spend$0.

## Limits and next authorization

This is an incomplete research run, not a validated candidate. One initialization seed; three completed
SELECT snapshots; no final s3 SELECT, original-V1 report comparison, PPO gain or report interval. Exact PPO
checkpoint restoration passed CPU tests, but no full-view PPO training or GPU continuation was exercised.
Any revised convergence diagnostic, optimization study, report or larger dataset needs a new frozen plan
and owner authorization. See NEXT_STEPS.md. Human-view PR60 merged independently and stays frozen.
