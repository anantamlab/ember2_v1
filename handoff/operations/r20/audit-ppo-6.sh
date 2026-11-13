#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Admin/Documents/Test/ember_v1
export HF_HOME=/home/darshith/ember-ws/hf-home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$PWD/src PYTHONUNBUFFERED=1
unset HF_TOKEN || true
EVIDENCE=/home/darshith/ember-ws/rl-evidence/r20-ppo-pilot@1/attempt-6
MODEL=/home/darshith/ember-ws/artifacts/models/1fc405f723718e308c8a22b137fee4a025086fde/v1/p1-18-v1-train@1
exec > /home/darshith/ember-ws/rl-evidence/attempt-6-audit-console.log 2>&1
test -f "$EVIDENCE/summary.json"
/home/darshith/ember-ws/cpu-tests/bin/python -m ember_v1.rl_verify "$EVIDENCE"
/home/darshith/ember-ws/venvs/9613f929b53c0fed/bin/python experiments/rl_gpu_checks.py --model-root "$MODEL" --evidence "$EVIDENCE"
date -u +%FT%TZ > "$EVIDENCE/AUDITS_COMPLETE"