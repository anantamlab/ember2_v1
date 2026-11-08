# Resume instructions

1. Read this file and `docs/overnight/STATUS.json`.
2. `git fetch --all` and `gh pr list --state all --limit 10`. Compare heads with STATUS.json.
3. Find the first phase that is not `DONE`, `BLOCKED` or `SKIPPED` and whose dependencies are met.
4. Continue from that checkpoint. Do not repeat a `DONE` phase. Do not rerun a frozen experiment (the v1 pilot, the v2 pilot, any DEV run).

## Current checkpoint

- Branch `p1/t4-v11-repair`, based on PR #32 head `45d8c2b`. Stacked draft PR #33.
- **All phases P0–P10 are DONE or SKIPPED.** Read `docs/overnight/FINAL_REPORT.md`. Nothing remains to run; the next step is owner review of the draft PRs #32–#35.
- P0–P6 done. The v2 pilot ran once: `T4_V2_BLOCKED` (G4). P5 (DEV) is skipped by protocol. Never rerun `t4_v2_pilot` or run `t4_v2_eval`.
- P7 on `p1/teacher-selection-prep` (worktree `D:/Coding/ember_v1-p7`), P8 on `p1/teacher-provenance` (worktree `D:/Coding/ember_v1-p8`). Both from main `b68535f`. Test a worktree with `PYTHONPATH=src` (the venv is an editable install of the main checkout).

## Test commands

- T4 focused: `.venv/Scripts/python.exe -m pytest tests/test_teacher_t4.py tests/test_teacher_t4_pilot.py`
- Full suite: `python ~/.claude/bin/full_suite.py -- .venv/Scripts/python.exe -m pytest`
