# Full-view research model card: no final candidate

Run `ember2-full-view-full-view-20261114t133215z-37025f` stopped at the unchanged round3 fit gate. No final candidate was frozen or approved.
Snapshots s0–s2 completed fit/SELECT; s3 is rejected and retained only for audit. The best completed
SELECT snapshot is `s2`, with raw SELECT utility 0.215278;
this is not a report score. Registry: `data/rl/full-view-checkpoint-registry.json`, with producing
source/result hashes, complete file tables, optimizer hashes and the rejected checkpoint status.

Base `unsloth/Qwen3.5-2B@fffbc0d8711cf87df6d1f97938c5e50ef94aadc7`; frozen backbone, LoRA r16,
float32 Clef five-option head. Action order: east, north, south, stay, west. Input is one complete public
full-view observation per active person, self marker, own last action/outcome and four neighbour-cell
statuses. Separate map layers are preserved. No hidden state, route hint or input truncation.

| Frozen report arm | Outcome | Paired interval |
|---|---|---|
| New model, raw | NOT RUN: stopped before candidate freeze | Unavailable |
| New model + legal mask | NOT RUN: stopped before candidate freeze | Unavailable |
| DAgger before PPO, raw | NOT RUN: stopped before candidate freeze | Unavailable |
| T1 teacher | NOT RUN: stopped before candidate freeze | Unavailable |
| Random legal moves | NOT RUN: stopped before candidate freeze | Unavailable |
| Original V1, raw | NOT RUN: stopped before candidate freeze | Unavailable |
| Original V1 + legal mask | NOT RUN: stopped before candidate freeze | Unavailable |

The stopped run does not establish that the new full-view model is better than original V1; no direct report comparison ran. PPO did not run, so no PPO gain over DAgger is established.

Original plan SHA256 `389ff09bd8950d7fe9e0cfb77fd5b2f16f2482de8a3e2cfeb2c20b705749c8d6`. Control-only runtime supplement SHA256 `fbf97cd181b2412d89c0e7494e3f63df76a2a6f42a2bd8024ef1a96a4d15398c`. Controller source `8fb33d67738fc018ef8e3861ec4adf5b6b4f3320`. Per-phase producing sources and result/log hashes are in the compact summary and checkpoint registry.
Package manifest SHA256 `940ffed4063b6eab7cfd5ddc3d7974f4957d3044fd0fcdde79f8d2e7a0da456c`. Upload pending: HF_TOKEN is absent from the environment. No new HF repository was created and no HF revision or private-visibility verification is claimed for ember2-v1-models. The intended destination is a private model repository under the HF_TOKEN account. The credential pasted in chat was not used .

Do not claim a final model, passing convergence gate, report score, gain over original V1, PPO gain,
training-seed robustness, perfect planning, comparison with R19–R22, sealed/hidden-set or EMBERBench score.
The static action mask is an external rule. CPU checkpoint continuation tests do not establish exact GPU
continuation. See FULL_VIEW_REPORT.md for retained negative results, diagnosis and budgets.
