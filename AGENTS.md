# AGENTS.md — rules for the full-view ember2_v1 project

These rules are mandatory for every human or AI agent that changes this repository. They replace the
GitHub Actions workflow `P0A contract tests` (`.github/workflows/p0a-tests.yml`), which was removed in run 10
(2026-11-09). **GitHub CI no longer validates anything. Local evidence is the only test evidence.**

Permanent rule: **Implement → full local test → verify PASS → push. Never push untested code.**

## Canonical repository and current state

`anantamlab/ember2_v1` is the full-view project. `anantamlab/ember_v1` is the separate human-view project.
Human-view and R19–R22 results in docs/ are inherited history, not results of this project. Keep the Python
package name `ember_v1`. The same local test, provenance and owner authorization rules below apply.
Full-view evidence belongs outside every checkout at `/home/darshith/ember-ws/ember2-evidence/`.
Full-view private HF destination is the HF_TOKEN account's `ember2-v1-models`; never upload new evidence
to human-view `sankalphs/ember-v1-models` or its `/home/darshith/ember-ws/rl-evidence/human-pov/` root.
The read-only `anantamlab/ember-v1-research-handoff` backup and files in `handoff/FILES.sha256` stay unchanged.
Read `docs/CURRENT_STATE.md` first. All owner NOT AUTHORIZED rules remain in force.

## Mandatory rules

1. Read this file, the PR you continue and its latest resume notes before you change anything. Check the
   actual branch heads (`git fetch`, `gh pr list`) before you edit a stacked branch.
2. Work in Python 3.11 (owner decision, runs 13-17: 3.11 only) with the test dependencies installed with the test dependencies installed:
   `python -m pip install -e ".[test]"`.
3. Run focused tests during development (one file, `-k`, one test).
4. Before EVERY push, execute the CPU test suite locally through `scripts/verify_local.py run` on Python 3.11
   (push with `EMBER_VERIFY_REQUIRE=3.11`). 3.12 runs only if the owner asks. See "Test tiers" for when the full
   tier (`run --with-slow`) is required.
5. Never skip, deselect, weaken or silently mark failing tests as expected failures. Do not edit a test only to
   get a green result. A test may change only when the behaviour it pins changes on purpose; say why in the
   commit message. The only sanctioned skip is the owner-approved `slow` tier below; add a file to it only
   with owner approval.

## Test tiers (owner decision 2026-11-13)

Seven end-to-end files are marked `pytestmark = pytest.mark.slow`: `test_s1_pipeline`, `test_selection_confirmation`,
`test_train_collect`, `test_train_pilot`, `test_pilot_preflight`, `test_s1_gate`, `test_expand`. They took 518 s of
a 912 s suite. `tests/conftest.py` skips them (reported as skipped) unless pytest gets `--run-slow`.

- **Default tier** (`verify_local.py run`, ~6.5 min): enough for a push whose change does not touch the code below.
- **Full tier** (`verify_local.py run --with-slow`): required when a change touches `s1.py`, `s1_pipeline.py`,
  `train_collect.py`, `train_pilot.py`, `pilot_supervisor.py`, `expand*.py`, `multiteacher.py`, `gen/`, `sim/`,
  `teacher/` or `scripts/pilot_preflight.py`, and before any release tag. The receipt records `"tier"`.
- Report the tier with the counts in the PR. Never call a default-tier run "the complete suite".
6. Preserve the full test output and record the Python version, Git SHA, test counts (passed, failed, errors,
   skipped, xfailed, xpassed) and exit code. `scripts/verify_local.py` does this for you.
7. Never push after a failed full-suite run.
8. Fix failures and rerun the complete suite before pushing.
9. Maintain the strict simulator independence, provenance and train/evaluation isolation rules: no evaluation,
   locked-test, DEV or benchmark data in training; no teacher outcomes read before a freeze; no real TRAIN labels,
   GPU training or Modal runs without a frozen plan and a verified owner authorization.
10. Keep reproducible checkpoints and durable resume notes (what was done, exact heads, what is pending).
11. Record the local test evidence (receipt summary: SHA, Python versions, counts, exit code) in the PR
    description or a PR comment for every pushed head.
12. If a supported Python version is not available, say so in the PR. Never claim coverage that did not run.

## Local verification

```bash
# once per interpreter (3.11 and 3.12); any checkout path works
python -m pip install -e ".[test]"

# complete suite, every interpreter, on a CLEAN tree (commit first)
python scripts/verify_local.py run --python /path/to/py3.11/python --python /path/to/py3.12/python
# or: EMBER_VERIFY_PYTHONS="/path/py311:/path/py312" python scripts/verify_local.py run   (';' on Windows)

python scripts/verify_local.py status          # receipts for HEAD
```

What `run` does for each interpreter:

- runs exactly `python -m pytest -q` (plus `--run-slow` with `--with-slow`) from the repository root; it accepts
  no other selection arguments and removes
  `PYTEST_ADDOPTS` and `PYTEST_PLUGINS` from the test environment;
- puts this checkout's `src/` first on `PYTHONPATH` and refuses PASS if `ember_v1` is imported from anywhere
  else (an editable install can point at another worktree);
- streams the full output to `.local-test-logs/<utc>-<sha12>-py<X.Y>.log` (git-ignored) and preserves pytest's
  exit code;
- reports PASS only for exit code 0, a parsed summary line, 0 failed, 0 errors, 0 deselected and at least one
  passed test; skipped, xfailed and xpassed counts are reported;
- digests the tree (HEAD, tracked diff, untracked files) before and after the run and reports
  `SOURCE_CHANGED` (exit 3) if anything changed;
- writes a receipt to `<git common dir>/ember-local-verify/receipts/` and next to the log.

Exit code: pytest's code if it failed; 3 source changed; 4 environment failed; 5 verdict failed with exit 0.

## Pre-push hook (optional, recommended)

```bash
python scripts/verify_local.py install-hook     # writes <git common dir>/hooks/pre-push (all worktrees)
```

The hook (`scripts/git-hooks/pre-push`) refuses a push unless every pushed commit has a PASS receipt from a clean
tree for each required Python (`EMBER_VERIFY_REQUIRE`, default `3.11,3.12`) and the receipt's log still exists
with the recorded hash. It does not run the suite itself.

Limitations: `git push --no-verify` bypasses it; it is per clone (not versioned in `.git`); receipts are local
files, so the hook checks discipline, not cryptographic proof; a branch without `scripts/verify_local.py`
is refused until it contains this policy. **This file and the hook do not enforce anything on GitHub.**

## GitHub settings

- Branch protection and rulesets are not available on this private repository plan (`gh api
  repos/anantamlab/ember_v1/branches/main/protection` returns HTTP 403 "Upgrade to GitHub Pro"). No required
  status check exists, so removing the workflow blocks no merge. If protection is enabled later, do not add the
  old `contract (3.11)` / `contract (3.12)` checks; keep any review requirements.
- Old CI results on earlier PR heads stay as history; they are not evidence for new heads.

## Trust and approval boundaries

- Never manufacture owner approval. Never use an available admin token to approve, merge or impersonate an
  independent human review. Authorization records stay `PENDING` until the owners act on GitHub.
- Do not merge research PRs, run TRAIN collection, GPU training or Modal jobs unless the owner explicitly
  authorizes that exact step.

## Research process rules (owner, run 17)

- One-owner authorization is sufficient for bounded CPU research experiments (`single_owner_merged_pr`).
- GitHub Actions CI/CD stays disabled. Do not add workflows.
- Prefer one integrated research PR (#53). A small immutable results/evidence or authorization PR is acceptable.
  Do not open stacked PRs per phase.
- Do not ask for confirmation between ordinary implementation stages. Ask once at a real authorization point.
- Run long experiments as persistent background processes (WMI-launched, hidden window) with a supervisor and a
  completion-triggered verifier. Do not send repeated progress messages.
- Preserve negative results, interruptions and failures. Never rerun a started group or attempt without a new
  authorization that names it.
- **Never bypass an attempt guard with a fresh clone.** Collection markers live in an evidence root outside every
  checkout, bound by the plan (P1-17 onward).
- No EMBERBench evaluation, no hidden benchmark access; the final evaluation set stays sealed.
- Large full-view evidence stays out of Git under `/home/darshith/ember-ws/ember2-evidence/`,
  with SHA256 manifests. Commit only compact summaries; historical Windows paths are archives.
- Owner prompt authorization for `ember-v1-closeout@1` permits documentation, required CPU tests and a
  merge commit after all stated gates pass. It authorizes no new human-view experiment.
  Deferred phases in `docs/NEXT_STEPS.md` need a new frozen plan and owner authorization.

## Full-view owner authorization

The owner prompt authorizes `ember2-v1-full-view@1`: private repository creation, bounded local TRAIN
collection/DAgger/small and LLM PPO, one frozen report invocation per arm, private HF upload, and a
gated merge commit. The frozen external plan copies that authorization. No cloud compute or spend,
sealed evaluation, reserved V1.1, simulator/generator/teacher changes, public resources or force push.
Limits: 60 GPU hours, 16 CPU hours, 150 GiB evidence, one GPU job. Never restart an existing attempt.
