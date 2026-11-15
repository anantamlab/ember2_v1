# ember2_v1 current state

**STOPPED: round3 fit completion gate failed . No final candidate, report results or PPO run . PR1 stays
draft and open; no failed-gate merge.**

The stopped run does not establish that the new full-view model is better than original V1; no direct report comparison ran. PPO did not run, so no PPO gain over DAgger is established.

Private full-view repository: `anantamlab/ember2_v1`. Separate private human-view project:
`anantamlab/ember_v1`, PR60 merged at`4ef795a7f3efb2a720d3b2e63ad723dc8fa1cb02`. Human-view results in
inherited history are not full-view results. Its private HF location stays`sankalphs/ember-v1-models`;
human evidence stays under`/home/darshith/ember-ws/rl-evidence/human-pov/`. No new human experiment ran.

Full-view evidence:`/home/darshith/ember-ws/ember2-evidence/2026-11-14/full-view-20261114T133215Z-37025f`. Original plan SHA256 `389ff09bd8950d7fe9e0cfb77fd5b2f16f2482de8a3e2cfeb2c20b705749c8d6`. Control-only runtime supplement SHA256 `fbf97cd181b2412d89c0e7494e3f63df76a2a6f42a2bd8024ef1a96a4d15398c`. Controller source `8fb33d67738fc018ef8e3861ec4adf5b6b4f3320`. Per-phase producing sources and result/log hashes are in the compact summary and checkpoint registry.
Manifest SHA256:`940ffed4063b6eab7cfd5ddc3d7974f4957d3044fd0fcdde79f8d2e7a0da456c`. Upload pending: HF_TOKEN is absent from the environment. No new HF repository was created and no HF revision or private-visibility verification is claimed for ember2-v1-models. The intended destination is a private model repository under the HF_TOKEN account. The credential pasted in chat was not used .

Three unexecuted-path review blockers also remain unresolved: evidence-root binding, report checkpoint/hash binding and a PPO selection-rule discrepancy. They independently prevent merge or further use.

See [report](FULL_VIEW_REPORT.md), [model card](FULL_VIEW_MODEL_CARD.md),
[registry](../data/rl/full-view-checkpoint-registry.json) and [deferred work](NEXT_STEPS.md).
All attempts and the failed s3 weights remain preserved. The128-family report panel remains unspent.
Exact final-head local receipts, push identity, final reviewer comments and shutdown/scheduler checks
are recorded in the PR and external final ledger. No further experiment is authorized by these documents.
