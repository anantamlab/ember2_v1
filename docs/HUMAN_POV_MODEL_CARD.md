# Human-view model card

This is a per-person fire-evacuation decision policy using the frozen
`unsloth/Qwen3.5-2B@fffbc0d8711cf87df6d1f97938c5e50ef94aadc7` backbone, a LoRA adapter and a five-option
Clef head. Action order: east, north, south, stay, west. DAgger s0 is the supervised parent.

Input: radius-4 partial view, line of sight and memory enabled, exits learned by sight, the person's last
action and outcome, four neighbour-cell statuses, and a route hint. It sees no hidden simulator state.
The route hint is a rule-computed input; the raw model still receives it.

Frozen checkpoint: `human-pov-ppo-llm-u20`, LLM PPO update 20 of
`human-pov-ws-full-20261113t214243z`, directory `ppo-llm-repair-01/update-0020`.
Source: `df4b431892a7b1190b44df0353651daeeffbb001`.
File-table identity: `sha256:fa49b0d765ea545fde6ea72d63885fbbdf4639a1db5f0d5bd6e6dd8a7ba6d04f`.
Candidate-freeze SHA256: `561f28f47d1b79a34aac3174cf7cf25209bcd322b13444d71ce7c861d2fbe711`.

| Checkpoint file | SHA256 |
|---|---|
| `README.md` | `4839c3850d0e52cafa33d37fa7a11983f5bbf5a4a689df90a3aa57a39f6cffe1` |
| `adapter_config.json` | `0db1fd12dcab6a7ac615622dca7c20a2bfab2a71315a6c58945f448d112e88a0` |
| `adapter_model.safetensors` | `12bdcf5af9161169b180f2cf2dc82015481ad4680553f3c38825e4ce17a2e0ba` |
| `chat_template.jinja` | `ea210e5b4997b7b1d3af5ee3ad34e7677ffe38df7469e8d88089789b5a27fae5` |
| `joint_head.safetensors` | `29c6c48b3912997a389e8d46cab87da6596b7f55d1435a30cc103b0fcab410c5` |
| `joint_head_config.json` | `9ec32207deca79a5ec809cb8306c8222d435163da3800e6fdb42e8789a69b8f8` |
| `processor_config.json` | `14932921ca485d458a04dafd8069fbb0a4505622a48208d19ed247115801385b` |
| `tokenizer.json` | `87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4` |
| `tokenizer_config.json` | `1a71fa1e006db5fddc1903926f8d645a608a598504e630c1d82fb7057ee30978` |
| `unsloth_decision_config.json` | `38a82e265362e2e863d5108e78cfc9927f744b4befd3ceffab5be35c738fd31f` |

One spent report panel: 32 families, 352 people, 48 steps, namespace
`human-pov-ws-full-20261113t214243z`. Utility is `(E - 2*C - 0.75*U) / N`.

| Arm | Evacuated | Casualties | Unresolved | Utility | Wall / boundary blocks |
|---|---:|---:|---:|---:|---:|
| Selected model raw | 303/352 (86.08%) | 27 (7.67%) | 22 (6.25%) | 0.660511 | 0 / 0 |
| Selected model + legal mask | 303/352 (86.08%) | 27 (7.67%) | 22 (6.25%) | 0.660511 | 0 / 0 |
| Belief teacher | 303/352 (86.08%) | 28 (7.95%) | 21 (5.97%) | 0.656960 | 0 / 0 |
| Random legal moves | 99/352 (28.12%) | 132 (37.50%) | 121 (34.38%) | -0.726562 | 0 / 0 |

The mask, random legal moves and belief teacher use rules. Report SHA256:
`b31f6e36c6fb603b3fa8798dabe583ce530546a1bdee154269553fb2b85eb0df`.

Private HF location: `sankalphs/ember-v1-models`, path
`human-pov/human-pov-ws-full-20261113t214243z/ppo-llm-repair-01/update-0020`.
Manifest revision: `8243ec7242e2475f764cecb9debd14990afe84bb`;
verified revision: `310f357782359eb07e27ac8e2d3064d2669bac68`.
Package manifest SHA256: `994e02f3bc68457bfc3aedabdf091ac6f7bd728d12d3a44cc869e86a0a55dd98`.
All 269 payload hashes were downloaded and verified, including an independent repeat; the registry and
`storage-verification.json` preserve the receipts. Actor checkpoints support inference; PPO critic and
optimizer state are absent, so exact training continuation is unavailable.

## Do not claim

- Proven RL gain over s0: s0 was not evaluated on this report panel.
- A direct comparison with R19–R22: those full-view panels and tasks differ.
- A sealed-set, EMBERBench or hidden-benchmark score: none was run.
- Training-seed robustness: there is only one seed.
- Planning without a hint: the route hint is part of the input.

See [HUMAN_POV_REPORT.md](HUMAN_POV_REPORT.md) and [NEXT_STEPS.md](NEXT_STEPS.md).
