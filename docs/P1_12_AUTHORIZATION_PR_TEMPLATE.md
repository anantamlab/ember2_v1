# Authorization PR template — `p1-12-multiteacher-train-pilot-authorization`

**Status of this template: PENDING. It approves nothing.** Prepared in run 10 (2026-11-09) for the owners.
The agent does not open, approve or merge the authorization PR and does not run the pilot.

## Preconditions (all before the authorization PR is opened)

1. PR #50 and the freeze PR are merged into `main` (merge commits recommended). `git fetch origin` shows
   `data/collection/p1-12-train-pilot@1/plan.json` with status `FROZEN` in `origin/main`.
2. `PYTHONPATH=src python -m ember_v1.train_pilot verify-plan --plan data/collection/p1-12-train-pilot@1/plan.json`
   prints `same: true` and plan hash `sha256:eb8dbe293db7f1c1e1b4f342c07f21853031e9e6ab5419eda2833c51f761b664`
   on a clean checkout of the execution commit.
3. The execution commit is a reviewed commit on `main` that contains the frozen plan, and the complete local suite
   passed on it on Python 3.11 and 3.12 (`scripts/verify_local.py run`; `AGENTS.md`).

## Record to commit

Copy `data/collection/p1-12-train-pilot@1/authorization_TEMPLATE.json` to
`data/collection/p1-12-train-pilot@1/authorization.json` and change only these fields:

| Field | Value |
|---|---|
| `status` | `APPROVED` |
| `plan_hash` | `sha256:eb8dbe293db7f1c1e1b4f342c07f21853031e9e6ab5419eda2833c51f761b664` (already set; do not edit) |
| `plan_freeze.commit` | 40-hex commit in `origin/main` that holds the FROZEN `plan.json` |
| `plan_freeze.path` | `data/collection/p1-12-train-pilot@1/plan.json` (already set) |
| `attempt` | `1` (already set) |
| `scenarios` | the eight IDs and hashes (already set; must equal the plan exactly) |
| `teacher_fingerprints` | T0–T4 (already set; must equal the plan exactly) |
| `execution_commit` | 40-hex reviewed commit on `main` to check out and run |
| `reviewer.github_login` | the approving admin: `sankalphs` or `Sathvikar01`, **not** the PR author |
| `approval.pull_request` | the number of the authorization PR itself (integer) |

Do not edit `limits`, `statement`, `kind` or `authorization_version`. `check_authorization` refuses any other
difference.

## GitHub procedure (independent review is mandatory)

1. Admin A (`sankalphs` or `Sathvikar01`) opens a PR into `main` that adds only `authorization.json`.
2. Admin B (the other allowlisted account) reviews the exact bytes and submits an **APPROVED** review on GitHub
   from their own session. A self-review, a comment, or an approval made with Admin A's token is not valid.
   An agent holding the `sankalphs` token must never act as Admin B.
3. The record names Admin B in `reviewer.github_login`. Merge only after the approval of the final commit.
4. The runner checks: PR merged into `main`; merge commit in `origin/main`; the latest review by Admin B is
   `APPROVED`; Admin B is not the author; the approved commit holds byte-identical record content.

## PR body (copy)

```
Authorizes attempt 1 of p1-12-train-pilot@1 (eight TRAIN scenarios, five teachers T0–T4, CPU only).

- Plan hash: sha256:eb8dbe293db7f1c1e1b4f342c07f21853031e9e6ab5419eda2833c51f761b664
- Frozen plan commit: <40-hex>  (data/collection/p1-12-train-pilot@1/plan.json, status FROZEN)
- Execution commit: <40-hex>
- Scenarios: exactly 8 (listed in the record); attempt: 1; retries: 0; substitutes: none
- Teacher fingerprints: T0, T1, T2, T3, T4 (1.2.0) as in the frozen plan
- Author: <Admin A login>; required independent approver: <Admin B login>
- Local full suite on the execution commit: Python 3.11 <counts, exit 0>; Python 3.12 <counts, exit 0>
- No Modal, GPU, training, benchmark or evaluation data.
```

## Run (after the authorization PR is merged; owner action)

```
git fetch origin && git checkout <execution commit>
PYTHONPATH=src python -m ember_v1.train_pilot run \
  --plan-ref <freeze commit>:data/collection/p1-12-train-pilot@1/plan.json \
  --auth-ref <authorization merge commit>:data/collection/p1-12-train-pilot@1/authorization.json
```

Run on an otherwise idle machine. One attempt only: any marker or `attempt-*/` directory blocks every later run.
