# Resume instructions

1. Read this file and `docs/overnight/STATUS.json`.
2. `git fetch --all` and `gh pr list --state all --limit 10`. Compare heads with STATUS.json.
3. Find the first phase that is not `DONE`, `BLOCKED` or `SKIPPED` and whose dependencies are met.
4. Continue from that checkpoint. Do not repeat a `DONE` phase. Do not rerun a frozen experiment (the v1 pilot, the v2 pilot, any DEV run).

## Current checkpoint

- Branch `p1/t4-v11-repair`, based on PR #32 head `45d8c2b`.
- Phase P0 done. Next: P1 (failure analysis and defect fixtures).

## Test commands

- T4 focused: `.venv/Scripts/python.exe -m pytest tests/test_teacher_t4.py tests/test_teacher_t4_pilot.py`
- Full suite: `python ~/.claude/bin/full_suite.py -- .venv/Scripts/python.exe -m pytest`
