# Standing authorization for bounded CPU research — PROPOSAL (not in force)

**Status: DRAFT PROPOSAL. It authorizes nothing.** It takes effect only when one owner approves it in a
written, signed record (see §4). It never applies to the frozen eight-scenario pilot `train-pilot@1`
(plan `sha256:eb8dbe29…b664`, execution commit `1a0128ca…`). That pilot keeps its two-owner authorization
PR procedure (`docs/P1_12_AUTHORIZATION_HANDOFF.md`) unchanged.

## 1. Purpose

Low-cost CPU experiments (collection, replay, counterfactual branching, dataset builds, CPU training checks)
should not each need a new two-owner PR. One owner approves one bounded **research plan**. Every experiment
inside that plan runs without a new approval, but keeps immutable evidence.

## 2. What a research plan must state

| Field | Content |
|---|---|
| `plan_id`, `plan_hash` | stable id; `sha256` of the canonical plan JSON |
| Dataset scope | exact split namespaces and scenario/family lists (or generator spec + seeds); held-out sets named as excluded |
| Experiments | list of experiment kinds allowed (e.g. `teacher_collection`, `replay_verify`, `counterfactual_branch`, `dataset_build`, `cpu_train_check`) |
| CPU cap | total CPU-seconds and wall-hours for the whole plan; per-experiment caps |
| Paid cap | `$0` for a CPU plan. Any Modal spend needs a separate approval |
| Data isolation | no DEV, calibration, locked-test, selection or EMBERBench data; leakage gate before the first call |
| Reproducibility | pinned code commit per experiment; source hashes; seeds; replay requirement |
| Failure rule | no automatic retry or substitution; failures and censored runs are kept and reported |
| Evidence | write-once output dirs, run logs with exit codes, manifest with hashes, one summary per experiment |
| Expiry | an end date or a total-experiment count |

## 3. When a new approval is still required

A new approval (two owners, as now) is required if an experiment would:

1. change restricted data access (new split, any held-out or sealed data, EMBERBench material);
2. exceed the plan's CPU caps, or spend money;
3. materially change the research protocol (teacher set, outcome definitions, class rules, preference evidence
   rules, evaluation protocol), or change a frozen plan;
4. use collected data for real model training (needs the frozen training protocol and its own approval).

## 4. Approval record

`data/authorizations/<plan_id>.json`: `plan_hash`, `approved_by` (GitHub login of a human owner),
`approved_at`, caps, expiry. The owner commits it from their own session, or approves a PR that adds it.
Agents never create, approve or edit an approval record. Runners verify the record hash and the caps before
each experiment and append one line per experiment to `<plan_id>/ledger.jsonl` (id, commit, start/end, CPU-s,
exit code, output hash).

## 5. First intended plan (after the pilot verdict)

`full-train@1`: `p1-02-v1/train`, 189 scenarios (64 families) × T0–T4 = 945 teacher episodes, attempt 1, CPU
only. Caps are set from the pilot's MEASURED numbers (T2 CPU per people band, group completion, replay rate),
not from the planning estimate (≈ 11 h serial, ≈ 3 h with 4 processes,
`docs/P1_12_TRAIN_COLLECTION_PROTOCOL.md` §3). It also needs a measured RAM budget per process on this machine.
A later plan adds independently generated, family-disjoint N = 150–200 TRAIN scenarios outside every sealed
family.
