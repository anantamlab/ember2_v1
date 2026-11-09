# P1-12 pilot runbook — attempt 1 of `p1-12-train-pilot@1`

**Status: NOT EXECUTED. Authorization: `PENDING_AUTHORIZATION`.** Prepared in run 12 (2026-11-10). This
runbook approves nothing. Only an owner runs it, and only after the authorization PR
([P1_12_AUTHORIZATION_HANDOFF.md](P1_12_AUTHORIZATION_HANDOFF.md)) is merged with an independent approval.

| Item | Value |
|---|---|
| Plan | `data/collection/p1-12-train-pilot@1/plan.json`, status `FROZEN` |
| Plan hash | `sha256:eb8dbe293db7f1c1e1b4f342c07f21853031e9e6ab5419eda2833c51f761b664` |
| Freeze commit | `827e88fa79d9fae9667f2fda590b4b23852988fc` |
| Execution commit | `1a0128ca5711557fba9db837f0ed251233bf6359` (tree equal to the freeze commit tree) |
| Scenarios | 8 TRAIN (4 feasible, 3 difficult, 1 unreachable); regeneration checks: `g-0821b9cfcfbdc9a9`, `g-b9f6d81f3b5bfce7` |
| Teachers | T0, T1, T2, T3, T4 (pinned fingerprints in the plan); one spawned, killable process per teacher per group |
| Limits | decision CPU s: T0 5, T1 5, T2 30, T3 5, T4 5; group wall 7200 s; pilot wall 14400 s; retries 0; substitutes none |
| Expected cost | wall about 1496 s; T2 CPU central 1496 s, upper limit 3725 s; Modal USD 0; no GPU |

Tools used below (read-only, run no teacher): `scripts/pilot_preflight.py`, `scripts/pilot_postrun.py`.
Run them from a SEPARATE tools checkout that contains them (`main` after this PR merges), never from inside the
execution checkout: an extra file there fails the clean-tree gate.

## 0. Rules

- One attempt. Any `attempt-*.json` or `attempt-*/` blocks every later run. Never delete one to "retry".
- No retry, no substitution, no edit of plan, code or data. Any failure is evidence: keep it and report it.
- No Modal, no GPU, no student training, no EMBERBench, no full TRAIN expansion. Full TRAIN needs a new frozen
  plan and a new authorization.
- Pilot output is `scope p1-12-train-pilot@1`, `training_eligible: false`. It is not an approved student training
  label set and must not be copied into any dataset export.
- Run on an otherwise idle machine: close browsers and other heavy programs. The preflight needs at least 6
  logical CPUs, 4 GiB available RAM and 1 GiB free disk (preflight requirements, not frozen plan fields).

## 1. Verify the authorization (owner)

```bash
N=<authorization PR number>
gh pr view $N --repo anantamlab/ember_v1 --json state,mergedAt,mergeCommit,author,baseRefName,headRefOid
gh api repos/anantamlab/ember_v1/pulls/$N/reviews --jq '.[] | [.user.login, .state, .commit_id] | @tsv'
gh api repos/anantamlab/ember_v1/pulls/$N/files --jq '.[].filename'
```

Required: `state` MERGED into `main` with a merge commit (not squash or rebase); author is one allowlisted owner;
the latest review of the OTHER owner is `APPROVED` on the PR's final head commit; the only file is
`data/collection/p1-12-train-pilot@1/authorization.json`. Set `AUTH=<mergeCommit oid, 40 hex>`.

## 2. Clean execution checkout (fresh clone; never run tests or installs in it)

```bash
git clone https://github.com/anantamlab/ember_v1.git ember_v1-pilot
cd ember_v1-pilot
git checkout --detach 1a0128ca5711557fba9db837f0ed251233bf6359
git fetch origin pull/$N/head          # the approved review commit must exist locally
git status --porcelain=v1 --untracked-files=all --ignored   # must print nothing
python --version                       # 3.11.x or 3.12.x; no package install is needed (no runtime deps)
cd ..
```

Do not use the agent worktrees `ember_v1-ex` / `ember_v1-ex2` (test logs, `.local-test-logs/` is not on the
clean-tree allowlist; the gate stays strict). The agent prepared `D:/Coding/ember_v1-pilot` (pristine,
detached at the execution commit, zero ignored files) on its own machine; an owner may use it on that machine.

## 3. Tools checkout and preflight (read-only)

```bash
git clone https://github.com/anantamlab/ember_v1.git ember_v1-tools   # main, contains scripts/pilot_*.py
python -I ember_v1-tools/scripts/pilot_preflight.py --repo ember_v1-pilot --auth-commit $AUTH \
    --out pilot-evidence/preflight_pre.json
```

Required: status `READY_AUTHORIZED`, exit 0. `PENDING_AUTHORIZATION` (exit 2) or `BLOCKED` (exit 1) means stop.
The preflight checks plan hash; live rebuild (scenario, teacher, simulator, generator, collector identities);
regeneration of the eight scenarios; teacher pins; HEAD, ancestry and tree equality with the freeze; clean tree;
no attempt artifact in this checkout, any worktree of it, or `origin/main`; machine capacity; then the runner's
own gate `run_preconditions` including the GitHub approval.

## 4. Run attempt 1 (the only teacher calls; owner action)

Bash:

```bash
cd ember_v1-pilot
PYTHONPATH=src python -m ember_v1.train_pilot run \
  --plan-ref 827e88fa79d9fae9667f2fda590b4b23852988fc:data/collection/p1-12-train-pilot@1/plan.json \
  --auth-ref $AUTH:data/collection/p1-12-train-pilot@1/authorization.json \
  > ../pilot-evidence/run_stdout.json 2> ../pilot-evidence/run_stderr.txt
echo "exit=$?" | tee ../pilot-evidence/run_exit.txt
cd ..
```

PowerShell:

```powershell
cd ember_v1-pilot
$env:PYTHONPATH = "src"
python -m ember_v1.train_pilot run `
  --plan-ref 827e88fa79d9fae9667f2fda590b4b23852988fc:data/collection/p1-12-train-pilot@1/plan.json `
  --auth-ref "${AUTH}:data/collection/p1-12-train-pilot@1/authorization.json" `
  > ..\pilot-evidence\run_stdout.json 2> ..\pilot-evidence\run_stderr.txt
"exit=$LASTEXITCODE" | Tee-Object ..\pilot-evidence\run_exit.txt
cd ..
```

What the runner does, in order: every gate again; atomic write-once marker `attempt-1.json` (open `x` + fsync)
BEFORE the first teacher call; the eight groups in plan order, each with five spawned teacher workers under
`pilot_supervisor` (per-decision CPU caps, group and pilot wall deadlines, kill on breach); the two regenerations
under the same caps; acceptance; write-once `attempt-1/evidence.json`. Exit 0 = accepted, 1 = not accepted.

Supervision by the operator:

- Progress: `attempt-1/log.jsonl` (one fsynced line per event). Read it; do not edit it.
- Do not interrupt. An interrupt is recorded (`interrupted`), starts no new group, and consumes the attempt.
- External hard stop only if the process is still alive after 15000 s (pilot wall 14400 s + 600 s grace):
  kill the whole process tree. Record the time and the reason. The attempt is consumed.

## 5. Post-run verification (read-only)

```bash
python -I ember_v1-tools/scripts/pilot_postrun.py --repo ember_v1-pilot --out pilot-evidence/postrun.json
python -I ember_v1-tools/scripts/pilot_preflight.py --repo ember_v1-pilot --auth-commit $AUTH \
    --out pilot-evidence/preflight_post.json      # must now be BLOCKED: attempt_1_unused false
```

`pilot_postrun.py` replays every stored group report in a fresh simulator (`multiteacher.verify_fixture`),
checks both regenerations (`g-0821b9cfcfbdc9a9`, `g-b9f6d81f3b5bfce7`) equal the primary `content_hash` and replay,
re-hashes the marker and `evidence.json`, and lists sha256 (raw and LF) of every attempt file.
`integrity_verified` must be true. `accepted` reports the acceptance verdict; false is a valid, reportable result.

## 6. Evidence packet (owner opens a results PR; no retry)

Commit, in a new PR into `main`: `attempt-1.json`, `attempt-1/` (log, groups, regen, evidence) and, under
`data/collection/p1-12-train-pilot@1/attempt-1-review/`: `preflight_pre.json`, `preflight_post.json`,
`postrun.json`, `run_stdout.json`, `run_stderr.txt`, `run_exit.txt`. The PR body lists the `postrun.json` file
hashes, the acceptance verdict and problems, the wall and T2 CPU totals, and states: pilot evidence only,
eight cases, no population-level teacher claim, `training_eligible: false`, no full TRAIN expansion.

## 7. Failure handling

| Event | Action |
|---|---|
| Preflight `PENDING_AUTHORIZATION` or `BLOCKED` before the run | Stop. Fix the cause (authorization, idle machine, fresh clone). No marker was written; attempt 1 is still unused. |
| Runner refuses at a gate (`PilotBlocked`) | Stop. No marker was written. Report the reason. Do not edit code or data to pass. |
| Teacher failure, cap breach, crash, integrity error, interrupt | Runner records it; group incomplete; acceptance fails. Keep everything. No retry, no substitute. |
| Power loss or kill mid-run | Marker exists, evidence may be missing. Run step 5; report as incomplete. Attempt consumed. |
| `pilot_postrun.py` integrity problem | Report it as a collector fault. Do not repair files. |
| Any outcome | Next step is a review and, if needed, a new plan version + new authorization for attempt 2. Never automatic. |
