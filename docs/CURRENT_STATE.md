# ember2_v1 current state

The full-view scientific run is complete. All seven one-shot arms ran on 128 shared report families
(1,430 people); the panel is spent. The controller exited 0, and no training, collection or evaluation remains active.

The final candidate is S2 DAgger: raw utility 0.008916 versus original raw V1 -0.314510, difference
+0.323427 (95% interval [0.193501, 0.453877]). The masked comparison is inconclusive. PPO was skipped
because the SELECT utility gap to T1 exceeded 0.05, so no PPO gain is established.

The old S3 fit FAIL and stopped closeout remain preserved. A new frozen, optimizer-free matched-TRAIN
audit passed a strict small loss decrease, then the unstarted SELECT/report phases ran once.
S3 did not replace S2. Review blockers were repaired with CPU regressions; no completed phase was repeated.

Candidate `/home/darshith/ember-ws/ember2-evidence/2026-11-14/full-view-20261114T133215Z-37025f/s2`. Weights digest `sha256:c55a773cb1af8151fb5c051de911d5e3fd2ee604288971f594c34dd593701adc`.
Original plan `389ff09bd8950d7fe9e0cfb77fd5b2f16f2482de8a3e2cfeb2c20b705749c8d6`; recovery `3a7f5bfe3bcc584c77f4483f5e16b1d7f9e2bc18d1352649d5358d9268fe0483`.
Evidence `/home/darshith/ember-ws/ember2-evidence/2026-11-14/full-view-20261114T133215Z-37025f`. Package manifest `9fc70001cf9627215d149269f3c35a87ac6203bea2b40e0736a1f860321daa63`.
Upload pending: HF_TOKEN is absent from the environment. The package remains local; no full-view HF repository, revision or private-visibility check is claimed. The intended private destination is the HF_TOKEN account’s ember2-v1-models. The chat credential and saved logins were not used.

Private full-view project: anantamlab/ember2_v1. Integration and exact final-head verification are recorded
on [PR1](https://github.com/anantamlab/ember2_v1/pull/1) and in the external final ledger.
Private human-view anantamlab/ember_v1 is separate; PR60 merged at
`4ef795a7f3efb2a720d3b2e63ad723dc8fa1cb02`, with full history inherited here. Human evidence and private
`sankalphs/ember-v1-models` are unchanged. No new human experiment ran.

See [report](FULL_VIEW_REPORT.md), [model card](FULL_VIEW_MODEL_CARD.md),
[failure ledger](FULL_VIEW_FAILURES.md), [registry](../data/rl/full-view-checkpoint-registry.json)
and [deferred version2 work](NEXT_STEPS.md). All checkpoint identities, raw report hashes, limits and
storage status match the evidence. New work requires a new frozen plan and owner authorization.
