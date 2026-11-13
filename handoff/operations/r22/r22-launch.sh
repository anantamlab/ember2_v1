#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Admin/Documents/Test/ember_v1
export PYTHONPATH="$PWD/src"
attempt=${1:-1}
wall_seconds=${2:-14400}
resume=()
if [ -n "${3:-}" ]; then resume=(--resume "$3"); fi
while true; do
  set +e
  /home/darshith/ember-ws/cpu-tests/bin/python /mnt/c/Users/Admin/Documents/Test/ws-tmp/r22-source-gate.py
  gate=$?
  set -e
  if [ "$gate" -eq 0 ]; then break; fi
  if [ "$gate" -ne 75 ]; then exit "$gate"; fi
  sleep 10
done
export HF_HOME=/home/darshith/ember-ws/hf-home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
unset HF_TOKEN
exec timeout --signal=TERM "$wall_seconds" /home/darshith/ember-ws/venvs/9613f929b53c0fed/bin/python -m ember_v1.rl_cf_run --config data/rl/r22-counterfactual-ppo@1.json --model-root /home/darshith/ember-ws/artifacts/models/1fc405f723718e308c8a22b137fee4a025086fde/v1/p1-18-v1-train@1 --phase2 /home/darshith/ember-ws/rl-evidence/r21-online-ppo@1/attempt-3 --output-root /home/darshith/ember-ws/rl-evidence --attempt "$attempt" "${resume[@]}"
