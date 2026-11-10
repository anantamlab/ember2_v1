# P1-12 eight-scenario five-teacher TRAIN pilot — freeze record

**Status: FREEZE PROPOSED FOR OWNER REVIEW. Not authorized. Not executed.** Prepared in run 10 (2026-11-09).
The freeze takes effect only when the owners merge this PR (and its base #50) into `main`; the runner reads the
plan only from a commit in the history of `origin/main`.

| Item | Value |
|---|---|
| Frozen plan | `data/collection/p1-12-train-pilot@1/plan.json` |
| Reviewed source | `data/collection/p1-12-train-pilot@1/plan_PROPOSED.json` (unchanged) |
| **Plan hash** | **`sha256:eb8dbe293db7f1c1e1b4f342c07f21853031e9e6ab5419eda2833c51f761b664`** (status is outside the hash) |
| Proposed blob sha256 (LF) | `372d83d9d6178577214c3ed1ec128f43e3dc90b9aef52aa22e3daea151041cc5` |
| Frozen blob sha256 (LF) | `2094a3471025d1922adfc17c58e77a0fd907058d3bbab1c76000923135368c48` |
| Only difference | `"status": "PROPOSED_NOT_FROZEN"` → `"status": "FROZEN"` (byte-checked by `test_frozen_plan_if_present_is_the_reviewed_plan_with_only_status_changed`) |
| Reviewed base commit | `153c949` on `research/p1-12-five-teacher-foundation` (PR #50); full SHA in the PR body |
| Freeze commit | the commit of this PR that adds `plan.json`; after merge, any `main` commit that holds this exact file |
| Scope | 8 TRAIN scenarios, attempt 1, five teachers T0–T4, CPU only; pipeline validation, no teacher-superiority claim |

## Code identities bound by the plan (`plan.identities`, `plan.teacher_pins`)

| Component | Identity |
|---|---|
| Generator | `gen-1.0.0`; module `sha256:58335ff478528a6d0e526e3ffc79ddb11ad1f3f33812a5b9e501ecdc844d7a0a`; splits `sha256:d1ed4077c6df51cd278a34e0c3e62f74b3418042c342769f3561c204ea872e45` |
| Simulator | observation `1.0`; module `sha256:4c961d54c3fc001f27aa0d1045d29090b8889716c0db720d7cb474e7a0e86d2d` |
| Collector | `matched-multi-teacher@0.2.0`; `sha256:061eb525303c1689e7b77040b1055e9323222acff62309e6fc6e03b2a3d56d16` |
| Supervisor | `p1-12-pilot-supervisor@1.0.0`; `sha256:5920da88408c5a0336ea926907463a8e758531c5dd3fd1e7dfc6dd71dd65212a` |
| Runner | `train_pilot` `sha256:d9839da83b343e3599ff745b7606e5cfde26ba95c34d062a4d278dbd624f5d79` |
| T0–T4 | config hashes and code fingerprints as in `docs/P1_12_PILOT_FREEZE_CANDIDATE.md` §3 (T4 = 1.2.0) |

Any change to these modules after the freeze changes a fingerprint: the runner then refuses the plan
(`verify-plan` differs; workers report `code_drift`). A changed plan needs a new plan version, not an edit.

## Run-10 gate recheck (independent of run 9)

| Gate | Result | How checked |
|---|---|---|
| Exactly 8 independent TRAIN scenarios, 8 families, all in the TRAIN manifest with the same hash | PASS | gate script on the frozen file |
| Design 4/3/1 labels, 5 deterministic / 3 stochastic, one group and one fire seed per scenario | PASS | `design_check` |
| Five retained teachers T0–T4; `registry.check_pins() == []`; `check_registry() == []`; T4 1.0.0 and 1.1.0 blocked | PASS | registry |
| Teacher pins and code identities equal a live rebuild | PASS | `teacher_pins()`, `code_identities()` |
| Plan rebuild | `same: true`, `differs: []` | `train_pilot verify-plan --plan …/plan.json` (Python 3.11) |
| Complete scenario regeneration (8 of 8, hashes checked) | PASS | `pilot_scenarios` |
| Historical and future evaluation isolation (DEV, calibration, locked test, P1-09, all pilot namespaces; geometry) | PASS (0 overlaps, 0 geometry hits) | `disjointness_audit`, `geometry_audit`; P1-15 `check_test_isolation` (#53) refuses TRAIN families in test sets |
| Matched initial conditions and fire randomness | PASS | one fire seed per scenario shared by all five teachers; supervisor replays every stream (tests) |
| Killable workers, per-decision CPU caps (5 s; T2 30 s), group 7 200 s, pilot 14 400 s | PASS | `limits`; supervisor tests (hang, crash, silent exit, cap) |
| Crash and timeout handling, zero retries, no substitution | PASS | `failure_policy`; tests |
| Atomic attempt marker (`open("x")` + fsync), interrupt keeps marker | PASS | tests |
| Full replay verification, forged stream = integrity error | PASS | tests |
| Source authentication (trusted `origin/main` history) and authorization / reviewer identity checks | PASS | tests (`verified_github_review`, `unverifiable_or_wrong`, `approval_of_other_bytes`) |
| Authorization template equals `authorization_template(plan)` and is `PENDING` | PASS | gate script; tests |

## T2 time budget and host load

- T2 stops its search on **thread CPU time** (20 s per decision), not wall time; the supervisor cap is 30 s
  process CPU. Other processes on the host slow wall time, not thread CPU time, so host load alone does not
  change T2 decisions. CPU frequency, cache and a different machine can still change how far the search gets.
- A decision that reaches the 20 s budget is marked `budget_limited`. Such a group may not regenerate
  identically on another host or run. The plan regenerates only the two smallest groups (N = 6 and 8), whose
  P1-09 people bin had a 0 % budget-limited share and a maximum of 33.5 CPU-s per scenario, far below the budget.
- The largest scenario (N = 120) had a 30 % budget-limited share in its P1-09 bin. Its content is therefore
  host-dependent by design. That is acceptable for a pipeline pilot; it must not be read as a T2 quality result.
- Wall caps (group 7 200 s, pilot 14 400 s) can end the pilot under extreme host load. That outcome is recorded
  and fails the acceptance gate; it is not retried. Run on an otherwise idle machine.

## Disclosed overlap (owner decision, run 10)

`g-0821b9cfcfbdc9a9` is also in the obsolete T1-only `s1-pilot@1` (draft PR #43). Accepted under the
conditions in protocol §6 item 5. Checked in run 10: every remote branch holds only the selection file
`data/s1/s1-pilot@1/pilot.json` (`FROZEN_SELECTION_NOT_RUN`), so no outcome evidence exists. The block of
`s1-pilot@1` is procedural: its runner exists only on unmerged #43 and needs an owner authorization that will
not be issued. Recommended owner action: close #43 unmerged as superseded.

## Local test evidence

Complete suite on this head with `scripts/verify_local.py run` (Python 3.11 and 3.12): see the PR body.
