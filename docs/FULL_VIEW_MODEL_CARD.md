# Full-view model card

The frozen candidate is the S2 DAgger checkpoint, `/home/darshith/ember-ws/ember2-evidence/2026-11-14/full-view-20261114T133215Z-37025f/s2`. PPO was skipped by the frozen
utility gate. Producing source: `8fb33d67738fc018ef8e3861ec4adf5b6b4f3320`; original fit result: PASS.
The separate rejected S3 fit remains FAIL in the registry, with its later TRAIN diagnostic acceptance
recorded separately. It is not the final candidate.

Base `unsloth/Qwen3.5-2B@fffbc0d8711cf87df6d1f97938c5e50ef94aadc7`, frozen backbone, LoRA rank 16,
float32 Clef five-option head. Action order: east, north, south, stay, west. Input is one complete public
full-map observation per active person, marked self, all public people/status, fire, exits and rules,
their last action/outcome and four neighbour statuses. No hidden state or route hint is supplied.
T1 labels and all DAgger fitting used TRAIN data; candidate selection used SELECT only.

| Arm | Evacuated | Casualties | Unresolved | Utility | Blocked / attempted | Wall / boundary |
|---|---:|---:|---:|---:|---:|---:|
| New candidate, raw | 863/1,430 | 340 | 227 | 0.008916 | 39.03% | 0 / 0 |
| New candidate + mask | 863/1,430 | 340 | 227 | 0.008916 | 39.03% | 0 / 0 |
| DAgger before PPO, raw | 863/1,430 | 340 | 227 | 0.008916 | 39.03% | 0 / 0 |
| T1 teacher | 1,294/1,430 | 51 | 85 | 0.788986 | 0.00% | 0 / 0 |
| Random legal moves | 255/1,430 | 718 | 457 | -1.065559 | 11.01% | 0 / 0 |
| Original V1, raw | 703/1,430 | 486 | 241 | -0.314510 | 71.80% | 9,028 / 419 |
| Original V1 + mask | 871/1,430 | 376 | 183 | -0.012762 | 45.07% | 0 / 0 |

The new raw model is better than original raw V1 on this panel: utility difference +0.323427,
95% paired-family interval [0.193501, 0.453877]. A masked advantage is not established: +0.021678,
interval [-0.095381, 0.148340]. PPO did not run, so no PPO gain over DAgger is established.
The report panel contains 128 families and 1,430 people and is spent. Full intervals and failures are
in [FULL_VIEW_REPORT](FULL_VIEW_REPORT.md) and [FULL_VIEW_FAILURES](FULL_VIEW_FAILURES.md).

Weights digest `sha256:c55a773cb1af8151fb5c051de911d5e3fd2ee604288971f594c34dd593701adc`; file-table SHA256 `706620c21a39820df95281264914d046667b11f36275641cd0f92f003e59e017`.
Candidate-freeze SHA256 `d93ff510f54ae85b27e68d67993b6d59e28962f4ff49a45a19f6ef9e5322ebfe`. All file hashes are in the checkpoint registry.
Original plan `389ff09bd8950d7fe9e0cfb77fd5b2f16f2482de8a3e2cfeb2c20b705749c8d6`; recovery plan `3a7f5bfe3bcc584c77f4483f5e16b1d7f9e2bc18d1352649d5358d9268fe0483`.
Local evidence `/home/darshith/ember-ws/ember2-evidence/2026-11-14/full-view-20261114T133215Z-37025f`; package manifest `9fc70001cf9627215d149269f3c35a87ac6203bea2b40e0736a1f860321daa63`.

Upload pending: HF_TOKEN is absent from the environment. The package remains local; no full-view HF repository, revision or private-visibility check is claimed. The intended private destination is the HF_TOKEN account’s ember2-v1-models. The chat credential and saved logins were not used.

Do not claim multiple independent SFT seeds, a PPO benefit, superiority over masked original V1,
teacher-level performance, component-specific causal gains, exact GPU trainer continuation, or a
sealed/hidden/EMBERBench score. No reserved V1.1 data was used. Raw legality does not solve crowd
coordination: the new model still had 340 casualties and 8,242 crowd blocks on this panel.
