# P1-12 matched five-teacher TRAIN collection — protocol draft

**Status:** `DRAFT_NOT_FROZEN`. Version `p1-12-matched-train@0.1.0-draft`.
**Authorizes nothing.** No TRAIN collection, no labels, no training, no GPU. `multiteacher.collect_fixture`
refuses every scope except `synthetic_fixture`. This draft must be reviewed, frozen in a separate commit and
authorized in writing by the owner before any TRAIN scenario is run. The pending T1 selection receipt
(P1-09) does not unlock this protocol.

## 1. What a collection unit is

One **matched group** = one independent TRAIN scenario × all five retained registry identities
(T0 1.0.0, T1 1.0.0, T2 1.0.0, T3 1.0.0, T4 1.2.0), each from a fresh `Simulator(scenario)` with the scenario's own fire
seed, a fresh policy instance and a deep copy of `observe()` only. The group record is the
`matched-multi-teacher@0.2.0` report of `multiteacher.py`:

- full per-step public observation, proposal (complete joint action), transition record, public diagnostics
  (movement outcomes, blocked moves, contested targets, fire adjacency, exit use);
- outcome per teacher: casualties, evacuated, unresolved, casualties+unresolved, causes, steps, termination;
- runtime per decision (collector thread CPU, wall, teacher self-report) — kept out of `content_hash`;
- provenance: simulator and collector module fingerprints, scenario content hash and source, each teacher's
  identity, `code_fingerprint` and registry pin status;
- recorded teacher failures (`teacher_exception`, `invalid_proposal`, `cpu_cap`). A group with any failure is
  `group_complete: false`, is kept, is reported, and is **never** used in a five-way comparison or silently
  turned into a four-teacher or T1-only sample.

Same step index in two trajectories is **not** a common state. `common_state_steps` lists the only shared
states (normally step 0). Same-state disagreement and counterfactual work belongs to P1-13 (#46).

## 2. Source and leakage gate (all must pass before the first teacher call)

| Gate | Rule |
|---|---|
| Source | `data/splits/p1-02-v1/train.json`, manifest `sha256:96a56481…b442d`, spec `sha256:992f8be3…3644`, generator `gen-1.0.0`; regenerate each scenario from the manifest and compare its content hash. |
| Split | TRAIN only. Refuse any scenario, family, lineage or geometry core hash that appears in DEV, calibration, P1-07/P1-09 selection, any pilot namespace (T2/T3/T4/s1-pilot@1), the sealed index or the blind firewall (reuse `s1_pipeline` held-out sets). |
| Identity | `registry.check_pins() == []`; all five identities retained; T4 v1.0/v1.1 blocked identities refused. |
| Code | Clean worktree at a pushed commit; collector, simulator and teacher fingerprints recorded. |
| Output | Write-once directory; every group verified by `verify_fixture`; the manifest of groups hashed. |

## 3. Cost model (planning estimate, not a hard limit)

P1-09 measured CPU over 72 scenarios (`data/selection/p1-09-selection@1/outcome/selection_gate.json`):
T0 7.1 s, T2 **7555.2 s**, T3 134.0 s, T4 68.8 s (T1 is the reference and is cheap, about T0 scale).
T2 dominates: about **105 CPU-s per scenario** on average.

| Unit | Scenarios | T2 CPU (×2 safety) | Others | Wall at 1 process |
|---|---|---|---|---|
| Pilot (§4) | 8 | ≈ 0.5 h | < 1 min | < 1 h |
| Full TRAIN (`p1-02-v1/train`) | 189 (64 families) | ≈ 11 h | ≈ 10 min | ≈ 11 h, or ≈ 3 h with 4 processes |

This is CPU only, $0 Modal. 189 scenarios in 64 families is a small source; a larger TRAIN generator run is a
separate, later protocol (new lineages, same split firewall).

## 4. Pilot (separate authorization)

- 8 TRAIN scenarios, 8 distinct families, stratified by generator label (feasible 4, difficult 3,
  unreachable 1), largest N included (up to 200), selected by a hash rank published before the run.
- Caps: 1 process per teacher in a killable subprocess; per-decision CPU cap 30 s (T2), 5 s (others);
  per-group wall cap 2 h; total pilot wall cap 4 h; zero retries.
- Pass criteria: 8/8 groups verified by `verify_fixture`; 2 groups regenerated with identical
  `content_hash` (except a documented time-budget fallback); no leakage-gate failure; measured T2 CPU per
  scenario within 2× of §3, else re-plan before full TRAIN.
- Failure handling: a failed group is kept and reported. No substitution, no silent retry.

## 5. What this data may be used for

Descriptive five-teacher comparison and the source for P1-13 common-state prefixes. It does **not** by itself
create supervised targets: eligibility of a demonstration is decided by the frozen P1-14 dataset protocol
(#47, `docs/P1_14_MULTITEACHER_DATASET.md`), never by this collector (`supervision_eligibility: not_assessed`).

## 6. Open review questions

1. Should the pilot include a second seed per scenario to measure fire-outcome variance?
2. Is 105 CPU-s/scenario representative of TRAIN, whose N distribution differs from P1-09?
3. Is a hard subprocess kill needed for all teachers, or only T2/T3 (which have internal budgets)?
