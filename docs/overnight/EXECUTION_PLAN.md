# Overnight execution plan (2026-11-08)

Operator: autonomous agent (Claude Code). Owner review: next morning. All PRs stay draft.

## Verified starting state

| Item | SHA |
|---|---|
| `main` (after PR #31) | `b68535ff6d282462663fcd073a57a2d3a92012f9` |
| PR #32 head (`p1/t4-risk-priority`) | `45d8c2b9ae9d91f99d1ec642f017cd3611974a9f` |
| T4 v1.0.0 frozen protocol commit | `99a65b8a947ef857f79514bac9e639ebaddb5b3a` |

Checked with `git fetch` and `gh pr list` at the start of the session. No newer commits existed.

## Phases and branches

| Phase | Work | Branch | Base | Depends on |
|---|---|---|---|---|
| P0 | Ledger | `p1/t4-v11-repair` | PR #32 head | — |
| P1 | T4 v1 failure analysis, defect fixtures | `p1/t4-v11-repair` | PR #32 head | P0 |
| P2 | T4 v1.1 shared repair correction | `p1/t4-v11-repair` | PR #32 head | P1 |
| P3 | V2 preregistration | `p1/t4-v11-repair` | — | P2 |
| P4 | V2 synthetic confirmation pilot (once) | `p1/t4-v11-repair` | — | P2, P3 |
| P5 | Single frozen DEV comparison | `p1/t4-v11-repair` | — | P4 = `T4_V2_FULL_DEV_FEASIBLE` |
| P6 | T4 verdict | `p1/t4-v11-repair` | — | P5 or P4 block |
| P7 | Issue #25 five-teacher selection preparation | `p1/teacher-selection-prep` | `main` | none (T4 marked by status) |
| P8 | Issue #10 provenance and trust gates | `p1/teacher-provenance` | `main` | none |
| P9 | Cross-review of every new PR | all | — | P2–P8 |
| P10 | Final handoff | `p1/t4-v11-repair` | — | all |

One stacked draft PR carries P0–P6 (base `p1/t4-risk-priority`). P7 and P8 each get an independent draft PR from `main`.

## Hard limits

- Local CPU only. No Modal, GPU, training, bulk teacher labels, EMBERBench, hidden tests or private salts.
- One worker process for experiments (16 GB RAM machine).
- v1.0.0 code, hashes, pilot report and G5 failure stay unchanged.
- DEV is not a fresh confirmation set. No tuning to DEV or to the v1 pilot.
- The v2 pilot runs once. A failed gate is a result, not a reason to retry.
