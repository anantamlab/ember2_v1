# P1-09 selection receipt: review packet

Status: **PENDING REVIEW.** `review.status` is `pending` and `reviewers` is empty. No person approved this receipt. Bulk S1 stays `BULK_S1_BLOCKED`.

This packet adds no new experiment. It does not change the protocol, the manifest, the isolation audit, the attempt marker, the evidence, the gate output or the draft receipt. The receipt `data/selection/p1-09-selection@1/receipt.json` is a byte copy of `outcome/receipt_DRAFT_PENDING_REVIEW.json`.

## 1. What the receipt says

| Field | Value |
|---|---|
| `receipt_version` | `teacher-selection-receipt@1.0.0` |
| `decision` | `RETAIN_T1_NO_IMPROVEMENT` |
| `selected.name`, `policy`, `version` | `T1`, `congestion_aware_cooperative_planner`, `1.0.0` |
| `selected.config_hash` | `sha256:5047e942db3326706b6184c9f29e0b992078920f03c8a53adc8cc24258345361` (frozen before DEV, issue #8) |
| `selected.code_fingerprint` | `sha256:52dc61f4ff9599e55bfd29f47068bf301895321ce85f6c9aaa9feec1dc568eac` (preflight record before the run; equal to current code) |
| `protocol_id`, `protocol_hash` | `p1-09-teacher-selection`, `sha256:ae72c2cf1a0dae4f46b3a6d158f426af0853bc4100cd15706c74f203a6765c21` |
| `protocol_ref` | git `64b2d9873953ba750fd6c08fb84761f1bcd8e0bb`, `data/selection/p1-09-selection@1/protocol.json` |
| `confirmation` | namespace `p1-09-selection@1`, manifest `sha256:4e9283b2…`, evidence `sha256:524e17c2…`, 24 families, 72 lineages, 72 scenario hashes |
| `receipt_hash` (pending) | `sha256:d6e94bd4a43631a5138cc110b3b04d75f94a150ce15fe07b89094443918d4022` |

## 2. Machine verification (read-only)

Command: `python -m ember_v1.receipt_review verify --repo .` (needs the git history of `origin/main`). Report: `data/selection/p1-09-selection@1/review/receipt_verification.json`. Result: **`VERIFIED_PENDING_REVIEW`, 18 of 18 checks pass.**

| Check | What it proves |
|---|---|
| `protocol_trusted_and_hash`, `protocol_worktree_equals_trusted` | the protocol loads from commit `64b2d98` in `origin/main`; its hash is `ae72c2cf…`; the working-tree file is the same bytes |
| `manifest_hash`, `isolation_audit` | manifest hash matches its content and the protocol preregistration; audit `pass: true`, hash `802b8fa5…` |
| `attempt_marker`, `evidence_hash_and_complete` | marker `f8321266…` and evidence `524e17c2…` match their content; evidence `complete: true`, no errors |
| `committed_gate_decision`, `gate_reproduces_identically` | the frozen `selection.selection_gate`, re-evaluated on the committed evidence (no teacher runs), gives the committed `selection_gate.json` byte-for-byte in canonical form |
| `draft_equals_committed_gate_draft`, `receipt_draft_hash`, `receipt_hash`, `receipt_equals_draft_except_review` | the receipt is the gate's own draft, unedited |
| `selects_exactly_T1_v1_0_0`, `config_equals_registry_pin` | decision and identity are exactly T1 1.0.0 with the registry config hash |
| `fingerprint_equals_current_code`, `fingerprint_equals_preflight` | the code fingerprint is the one recorded before the run and equals the current `main` code |
| `confirmation_lists_equal_manifest` | family, lineage and scenario lists equal the confirmation manifest (S1 uses them as a hold-out) |
| `s1_refuses_only_for_review` | `s1.verify_receipt` fails only on its last check, "receipt is not reviewed". Every earlier S1 check passes |

`git diff 46c445e a39c3f3 -- src/` is empty: the code that ran the confirmation is the code on `main`.

## 3. Scientific rationale

1. **The protocol was frozen before any confirmation result.** Freeze commit `64b2d98`; one attempt; the marker was written before any teacher ran; the runner refuses a second attempt.
2. **The confirmation set is new and isolated.** 24 families x 3 lineages, disjoint from `p1-02-v1` (all four splits) and all pilot namespaces (isolation audit pass).
3. **No candidate met the replacement rule.** A candidate replaces T1 only if the 95 % family-cluster CI upper bound of (candidate − T1) casualties is < 0, Holm-adjusted family sign-test p < 0.05, the safety rule holds, and policy CPU ≤ 7200 s.

| | T0 | T2 | T3 | T4 v1.2.0 |
|---|---|---|---|---|
| Mean casualty difference per scenario | +2.403 | −0.097 | +0.347 | −0.222 |
| 95 % CI | [+1.083, +4.194] | [−0.208, −0.014] | [−0.222, +1.056] | [−0.722, +0.097] |
| Holm p | 0.00161 (harmful) | 0.375 | 1.0 | 1.0 |
| Safety | FAIL | pass | pass | pass |
| Policy CPU ≤ 7200 s | pass | FAIL (7555.2 s) | pass | pass |

4. **The decision follows the frozen no-improvement rule.** T1 is retained because no alternative qualified, not because T1 is proven best.
5. **T1 is cheap enough for bulk labels.** 43.3 s policy CPU for 72 scenarios (mean 13.1 ms per decision, max 78.1 ms). T2 needs 175 times more CPU for 7 fewer casualties in 2835 people.

## 4. Limitations (a reviewer must accept these explicitly)

| # | Limitation | Consequence for S1 and the student |
|---|---|---|
| L1 | T1 is retained, not shown superior or optimal. On this set T1 had 435 casualties and 71 unresolved people of 2835 | The student's ceiling is T1 behaviour. Student-vs-T1 parity is the target; any claim beyond T1 needs separate evidence |
| L2 | The sign test had low power: T2 had 4 informative families, all favourable; p = 0.125 is the minimum possible with 4 | A small, consistent T2 gain is not excluded. A new test needs a new, separately frozen protocol and namespace |
| L3 | One run per identity, one host (Windows 11, Python 3.14.7) | CPU times are this host's. Determinism of T1 is covered by full replay and regeneration checks, not by repeated runs |
| L4 | The simulator is the independent P1 simulator, not EMBERBench. No EMBERBench equivalence and no physical fidelity is claimed | All teacher quality evidence is in-simulator. Cross-environment results are open |
| L5 | T1 is action-only: it computes no per-action values (`registry.VALUE_TEACHERS` = {T2}) | S1 labels are hard actions only. No soft labels, scores or probabilities may be invented (issue #10) |
| L6 | Hashes and git ancestry are bindings, not signatures. `review.reviewers` is a recorded claim, not a verified identity | The protection is against accidental and inconsistent edits by trusted contributors, not against a party that controls the repository |
| L7 | The confirmation set is now spent as evidence | It is a hold-out forever: `s1.build_s1_training_set` refuses any training family, lineage or scenario in the receipt's lists |
| L8 | T4 v1.2.0 and T3 tie with T1 within noise; T0 is harmful | T0 is a lower-bound reference only. No multi-teacher mixture is authorized (an arbiter receipt is refused by S1) |

## 5. How a reviewer approves (human only)

Only an authorized named reviewer does these steps. No automated agent may run step 3 or invent a name.

1. Read this packet, `docs/P1_09_CONFIRMATION_RESULT.md` and the limitations above.
2. Run `python -m ember_v1.receipt_review verify --repo .` on a clean checkout with `origin/main` fetched. Require `VERIFIED_PENDING_REVIEW`.
3. Run `python -m ember_v1.receipt_review approve --receipt data/selection/p1-09-selection@1/receipt.json --reviewer "<your full name>" --out data/selection/p1-09-selection@1/receipt.json`. The tool refuses placeholder names and non-pending receipts, and rehashes the receipt.
4. Run step 2 again. Require `VERIFIED_APPROVED`. Regenerate `review/receipt_verification.json` with `--out`.
5. Commit, get pull-request review, and merge to `main`. S1 then reads the receipt only from that commit (`receipt_ref` = {git, merge commit, `data/selection/p1-09-selection@1/receipt.json`}).

After approval, bulk S1 is still blocked until every other issue #10 gate passes (`docs/P1_S1_PIPELINE.md`).

## 6. Review status

| Item | State |
|---|---|
| Machine verification | PASS (18/18), `VERIFIED_PENDING_REVIEW` |
| Named reviewer approval | **NOT GIVEN** |
| Receipt on `main` with approved status | no |
| Bulk S1 | `BULK_S1_BLOCKED` |
