# Deferred human-view work

Status: the V1 human-view candidate is frozen LLM PPO update 20 of run
`human-pov-ws-full-20261113t214243z`. Its report panel is spent. Nothing in this document runs now.

## Phase A — Does PPO add anything?

Compare s0 against update 20 on a new 256-family report panel, with the belief teacher and random legal
moves. Report paired bootstrap intervals over families, with 5,000 resamples.

## Phase B — Where does the model break?

Compare update 20 against the teacher at radius 2, radius 7, no memory, no line of sight,
exit knowledge `one` and `all`, and no hints. A crash or truncation is a result.

## Phase C — No-hint ablation

Train one model with `--no-hints`, the same hyperparameters as s0 and the same DAgger stop rule;
compare with the hinted model. Question: does the model plan or copy the route hint?

## Phase D — PPO with freedom to pass the teacher

First save critic and optimizer state in PPO checkpoints. Then run three seeds from s0 with
`imitation_decay 0.90` and `reward.blocked -0.05`; select per seed; one report.

Each phase needs a new frozen plan and a new owner authorization. Estimated total cost: about 15 GPU hours.

Known limits: one training seed, 32 report families, and the route hint is part of the input.
The selection improvement includes three fewer casualties among 353 people; evacuation increased by one
person (311 to 312), not three. RL gain is not proven: s0 was not evaluated on the spent report panel.
PPO checkpoints cannot resume exactly because critic and optimizer state were not saved.
