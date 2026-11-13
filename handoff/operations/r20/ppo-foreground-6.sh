#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Admin/Documents/Test/ember_v1
export HF_HOME=/home/darshith/ember-ws/hf-home
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD/src
unset HF_TOKEN || true
mkdir -p /home/darshith/ember-ws/rl-evidence
exec timeout --signal=TERM 9100 /home/darshith/ember-ws/venvs/9613f929b53c0fed/bin/python -m ember_v1.rl_run --config data/rl/r20-ppo-pilot@1.json --model-root /home/darshith/ember-ws/artifacts/models/1fc405f723718e308c8a22b137fee4a025086fde/v1/p1-18-v1-train@1 --output-root /home/darshith/ember-ws/rl-evidence --attempt 6 >/home/darshith/ember-ws/rl-evidence/attempt-6-console.log 2>&1 </dev/null
