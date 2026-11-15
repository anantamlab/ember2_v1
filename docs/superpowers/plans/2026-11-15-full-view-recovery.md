# Full-view completion recovery implementation plan

> Use superpowers:executing-plans for native implementation and one final whole-branch review.

Goal: finish only unstarted phases under the original budgets and owner STOP rules.
Authorization: original owner prompt plus "Then finish ember2". Owner opted out of questions.
Architecture: preserve every historical failure and checkpoint. Repair root/checkpoint/selector
bindings. A separately frozen, named, optimizer-free TRAIN audit compares the saved S2 parent
and S3 child on the same deterministic 4096-row panel from the existing deduplicated TRAIN union.
Select rows by lowest SHA256 of canonical row JSON, before model loading, with dataset/row hashes.
Use full five-option unmasked cross entropy, dropout off, float64 accumulation, strict child < parent
and all finite. No optimizer, fitting, collection, SELECT or REPORT in the audit. Failure stops recovery.
This repairs the mismatched stochastic group comparison without increasing any tolerance. The old
first/last-loss gate and failed attempt remain unchanged and FAILED. A new audit is independent evidence.

Alternatives: retraining is prohibited; ignoring the fit gate is prohibited; comparing identical TRAIN
examples isolates the convergence diagnostic defect and is the chosen implementation judgment.

Global constraints: Python3.11; protected and existing tests untouched; 60 GPUh/16 CPUh/150GiB/$0;
one GPU job; all 7 report arms still unspent and remain one-shot; no hidden/reserved data.

Review focus: relocated plan; wrong CLI checkpoint; mutated original manifest; raw/masked baseline
disagreement; changed or failed reuse records. Each requires a new CPU regression.

- [ ] New CPU RED tests for root/checkpoint/selector bindings; implement minimal helpers; GREEN.
- [ ] CPU tests for deterministic audit rows/strict finite loss and immutable recovery reuse.
- [ ] New frozen recovery JSON binds parent plan/source/archive, failed result, all completed outputs,
  fixed TRAIN panel, global budget ledger and original arms/panels. Post hash before GPU work.
- [ ] Clean source commit, verify_local PASS, push with Python3.11 receipt, immediate PR1 link.
- [ ] Persistent named audit/controller, completion verifier and budget accounting. Reuse completed phases
  by read-only hash verification; do not spawn them. Only audit-PASS permits unstarted S3 SELECT onward.
- [ ] Additive evidence package, truthful report/card/registry/state and private HF upload or explicit pending.
- [ ] Exact final clean-head full tier + torch PPO, review, verified push/receipt, guarded merge.
  Update existing final comments on both PRs; shutdown processes and disable fallback.
