#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Admin/Documents/Test/ember_v1
export HF_HOME=/home/darshith/ember-ws/hf-home
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src"
unset HF_TOKEN
exec timeout --signal=TERM 21600 /home/darshith/ember-ws/venvs/9613f929b53c0fed/bin/python -m ember_v1.rl_online --config data/rl/r21-online-ppo@1.json --model-root /home/darshith/ember-ws/artifacts/models/1fc405f723718e308c8a22b137fee4a025086fde/v1/p1-18-v1-train@1 --phase1 /home/darshith/ember-ws/rl-evidence/r20-ppo-pilot@1/attempt-6 --output-root /home/darshith/ember-ws/rl-evidence --attempt 3 > /home/darshith/ember-ws/rl-evidence/r21-attempt-3.console.log 2>&1