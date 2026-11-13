#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Admin/Documents/Test/ember_v1
export HF_HOME=/home/darshith/ember-ws/hf-home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src"
unset HF_TOKEN
/home/darshith/ember-ws/venvs/9613f929b53c0fed/bin/python /mnt/c/Users/Admin/Documents/Test/ws-tmp/r21-autograd.py
/home/darshith/ember-ws/venvs/9613f929b53c0fed/bin/python -m pytest experiments/rl_gpu_checks.py -q