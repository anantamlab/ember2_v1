# P1-12 pilot authorization handoff (two owners)

**Status: `PENDING_AUTHORIZATION`. This document approves nothing.** Prepared in run 12 (2026-11-10) by an
agent. The agent did not open, review, approve or merge any authorization PR, and holds no approval role.
Procedure details: [P1_12_AUTHORIZATION_PR_TEMPLATE.md](P1_12_AUTHORIZATION_PR_TEMPLATE.md). Execution after
approval: [P1_12_PILOT_RUNBOOK.md](P1_12_PILOT_RUNBOOK.md).

## What the authorization covers (exactly)

| Field | Value |
|---|---|
| Pilot | `p1-12-train-pilot@1`, attempt **1** only |
| Scenarios | exactly the **8** frozen TRAIN scenarios in the plan (IDs and hashes already in the template) |
| Teachers | **T0, T1, T2, T3, T4**, fingerprints exactly as in the plan (already in the template) |
| Plan hash | `sha256:eb8dbe293db7f1c1e1b4f342c07f21853031e9e6ab5419eda2833c51f761b664` |
| `plan_freeze.commit` | `827e88fa79d9fae9667f2fda590b4b23852988fc` |
| `execution_commit` | `1a0128ca5711557fba9db837f0ed251233bf6359` |
| Limits | unchanged from the plan: decision CPU s T0 5 / T1 5 / T2 30 / T3 5 / T4 5; group wall 7200 s; pilot wall 14400 s; 1 process per teacher |
| Retries / substitutions | **0 / none** |
| Spend | no Modal, no GPU, no training, no benchmark (Modal USD 0) |

Allowed owners: `sankalphs`, `Sathvikar01`. **Owner A** opens the PR. **Owner B** (the other one) reviews and
approves its final commit from their own GitHub session. GitHub forbids self-approval; the runner also checks
author ≠ approver and that the approved commit holds byte-identical record content.

Any session that holds Owner A's token (including an agent session) must never act as Owner B.

## Checklist — Owner A (author)

```text
[ ] 1. Own session, own account. git fetch origin; origin/main contains 1a0128ca5711557fba9db837f0ed251233bf6359.
[ ] 2. git switch -c auth/p1-12-train-pilot-1 origin/main
[ ] 3. Copy data/collection/p1-12-train-pilot@1/authorization_TEMPLATE.json to .../authorization.json.
       Change ONLY: status -> "APPROVED"; plan_freeze.commit -> 827e88fa79d9fae9667f2fda590b4b23852988fc;
       execution_commit -> 1a0128ca5711557fba9db837f0ed251233bf6359; reviewer.github_login -> <Owner B login>.
       Do not touch plan_hash, scenarios, teacher_fingerprints, limits, statement, kind, version.
[ ] 4. Commit only that file. Push. Open a PR into main (body: template "PR body" section).
[ ] 5. Set approval.pull_request -> <this PR number, integer>. Commit, push. This is the FINAL commit.
[ ] 6. Request review from Owner B. Do not approve, merge or push after Owner B approves.
[ ] 7. After Owner B's APPROVED review of the final commit: merge with "Create a merge commit"
       (not squash, not rebase). Record the merge commit oid = AUTH.
```

## Checklist — Owner B (independent approver)

```text
[ ] 1. Own session, own account (not Owner A's token, not an agent token).
[ ] 2. PR changes exactly one file: data/collection/p1-12-train-pilot@1/authorization.json.
[ ] 3. Diff vs authorization_TEMPLATE.json changes ONLY: status APPROVED; plan_freeze.commit
       827e88fa79d9fae9667f2fda590b4b23852988fc; execution_commit 1a0128ca5711557fba9db837f0ed251233bf6359;
       reviewer.github_login = YOUR login; approval.pull_request = this PR's number.
[ ] 4. plan_hash = sha256:eb8dbe293db7f1c1e1b4f342c07f21853031e9e6ab5419eda2833c51f761b664; attempt 1;
       8 scenarios; T0-T4; limits unchanged; no retries, no substitutes.
[ ] 5. Optional check: python -I scripts/pilot_preflight.py --repo <clean checkout at 1a0128c> must say
       PENDING_AUTHORIZATION before merge (all non-authorization checks pass).
[ ] 6. Submit an "Approve" review on the PR's CURRENT head commit. If anything is pushed after it, re-review.
```

## Content check an owner can run before opening the PR (read-only)

```bash
PYTHONPATH=src python - <<'EOF'
import json; from ember_v1 import train_pilot as tp
plan = json.load(open(tp.FROZEN_PLAN_FILE)); auth = json.load(open(tp.AUTH_FILE))
print(tp.check_authorization(auth, plan) or "content OK (GitHub approval is checked separately at run time)")
EOF
```

PowerShell (from the repository root of the authorization branch):

```powershell
$env:PYTHONPATH = "src"
python -c "import json; from ember_v1 import train_pilot as tp; plan = json.load(open(tp.FROZEN_PLAN_FILE, encoding='utf-8')); auth = json.load(open(tp.AUTH_FILE, encoding='utf-8')); print(tp.check_authorization(auth, plan) or 'content OK (GitHub approval is checked separately at run time)')"
Remove-Item Env:PYTHONPATH
```

## Agent preflight result in run 12 (no teacher call)

Checkout `D:/Coding/ember_v1-pilot`: fresh worktree, detached at `1a0128ca…`, zero modified, untracked or
ignored paths. `scripts/pilot_preflight.py` result:

| Check | Result |
|---|---|
| git identity / ancestry (freeze ⊂ HEAD ⊂ origin/main) | pass |
| tree of HEAD = tree of freeze commit; no path changed since freeze | pass |
| plan hash, FROZEN, plan blob unchanged | pass |
| live rebuild = frozen plan (`differs: []`, 49 s) | pass |
| scenario fingerprints; 8 scenarios regenerate with recorded hashes | pass |
| teacher fingerprints and registry pins | pass |
| simulator / generator / collector identities | pass |
| attempt 1 unused (checkout, all worktrees, origin/main) | pass |
| clean tree (strict; `.local-test-logs/` stays NOT allowlisted) | pass |
| machine: 24 logical CPUs, 23.1 GiB RAM, 54.9 GiB free disk | **fail at test time: 1.22 GiB RAM available** (other programs open) |
| runner gate reasons | only: `authorization.json does not exist`, `no multi-teacher TRAIN pilot authorization` |

Overall `BLOCKED` at that moment only because of available RAM; with the machine idle the same checks give
`PENDING_AUTHORIZATION`. Re-run the preflight immediately before the authorized run.

Run 13 (2026-11-10), same checkout, still detached at `1a0128ca…` with zero modified, untracked or ignored
paths before and after the preflight: every identity, rebuild (`differs: []`, 36 s), scenario, teacher,
attempt and clean-tree check passes. `machine` fails again: 24 logical CPUs, 23.12 GiB total, **3.14 GiB
available** (< 4.0; browsers open), 54.9 GiB free disk. Runner gate reasons: authorization only. Overall
`BLOCKED`. The 4 GiB rule stays unchanged.
