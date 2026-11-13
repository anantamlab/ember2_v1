# Run 8 resume ledger (2026-11-09)

Durable copy of the run state. Read this first after a session restart, then `git status --short` in each
worktree. Rules: no merge, no Modal/GPU, no real TRAIN collection, no real labels, no training, no
EMBERBench, no FROZEN status, P1-09 evidence untouched.

Start heads (verified, CI 4/4 green each): #44 `63e5a11`, #50 `aa161d3`, #51 `98fa855`, #52 `d20b3cb`,
#53 `0035897`. Stack ancestry main → #50 → #51 → #52 → #53 verified.

Worktrees: #44 `D:/Coding/ember_v1-cj`, #50 `ember_v1-mt`, #51 `ember_v1-cf`, #52 `ember_v1-ds`, #53 `ember_v1-ev`.
Test runner: `PYTHONPATH=src D:/Coding/ember_v1/.venv/Scripts/python.exe -m pytest` in the worktree.
Pinned tokenizer copy (sha256 `87a7830d…`): scratchpad `tokenizer.json` of this session; the original is in
an earlier session scratchpad `…/9a954f2e-…/scratchpad/tok/tokenizer.json`.

## Steps

- [x] P0 baseline full suite at #53 `0035897`: 1298 passed, 10 xfailed (strict, T4 v1.0.0 defect), 0 skipped,
      0 failed, 639 s. CPU-cap fix `aa161d3` is test-only (cap 5→100 ms, burn 30→300 ms); production default
      30 000 ms unchanged.
- [x] Clef source re-fetched from PyPI wheel (sha256 `baff520e…`), `joint_schema_model.py` sha256 `0e304cf7…`
      = audited. Head: per-question logits over that question's options; no answer-to-answer conditioning.
- [x] P3 `t1_chunk_reserved` code in #44 (`chunk_joint.py`, `token_budget.py`, `experiments/p1_chunk_reserved.py`,
      `tests/test_chunk_reserved.py`); study output `data/research/p1-chunk-reserved/study.json`; doc
      `docs/P1_CHUNK_RESERVED_STUDY.md`. Command uses `uv run --isolated --no-project --with tokenizers==0.22.2`.
- [x] P1/P2 `student_objective.py` + tests in #52; dataset §4 decision A; training-method §1 derivation.
- [x] P4 dataset hardening in #52 (`mt-dataset-record@0.2.0`, protocol `0.2.0-draft`, approval checks, validators).
- [x] P5 pilot prep in #50: `train_pilot.py`, tests, `data/collection/p1-12-train-pilot@1/plan_PROPOSED.json`
      (plan_hash `sha256:07ea910e…c25a5`, PROPOSED_NOT_FROZEN). `run` always refuses.
- [x] P6 eval gates in #53 (`0.2.0-draft`): unresolved/casualty bounds 0 + point estimates, per-outcome severe
      regression + Clopper-Pearson, privacy INVALID, deployable comparator, claims table, latency rule (null).
- [x] P7 roadmap in #52 training-method §5–7.
- [ ] P8 commits in order #50 → merge into #51 → #52 → #53; #44 separate; push; CI; PR comments; full suite on
      the final #53 head and on #44; memory ledger; final report.
