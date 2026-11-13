#!/usr/bin/env bash
set -euo pipefail
ROOT=$HOME/ember-ws
C=$ROOT/checkouts/a3a9bf32763b98486a2ab95467ae4599ab787349
PY=$ROOT/venvs/9613f929b53c0fed/bin/python
V1=$ROOT/runs/r19-v1-arms@1/attempt-1/final
B=$ROOT/runs/r19-base-control@1/run
export PYTHONPATH=$C/src
cd "$C"
"$PY" -m ember_v1.p1_18_compare \
  --list "$V1/scenarios.jsonl" \
  --student V1_A_raw="$V1/rollouts-A_raw.jsonl" \
  --student V1_B_mask="$V1/rollouts-B_static_mask.jsonl" \
  --student V1_C_coord="$V1/rollouts-C_joint_coord.jsonl" \
  --student BASE_A_raw="$B/rollouts-A_raw.jsonl" \
  --student BASE_B_mask="$B/rollouts-B_static_mask.jsonl" \
  --student BASE_C_coord="$B/rollouts-C_joint_coord.jsonl" \
  --teachers "$ROOT/compare/dev-teachers/run18/dev-teachers/p1-18-dev-teachers@1" \
  --teachers "$ROOT/compare/expand/run18/expansion/p1-18-expand@1" \
  --out /mnt/c/Users/Admin/Documents/Test/ws-tmp/base_compare.json
echo DONE
