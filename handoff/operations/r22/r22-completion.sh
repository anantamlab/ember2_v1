#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Admin/Documents/Test/ember_v1
export PYTHONPATH="$PWD/src" HF_HOME=/home/darshith/ember-ws/hf-home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
unset HF_TOKEN
started=$(date +%s)
ops=/mnt/c/Users/Admin/Documents/Test/ws-tmp
export R22_ATTEMPT=${1:-1}
root=/home/darshith/ember-ws/rl-evidence/r22-counterfactual-ppo@1/attempt-$R22_ATTEMPT
set +e
/home/darshith/ember-ws/cpu-tests/bin/python -m ember_v1.rl_cf_verify "$root" > /home/darshith/ember-ws/rl-evidence/r22-cpu-audit.console.log 2>&1
cpu_exit=$?
printf '%s\n' "$cpu_exit" > "$ops/r22-cpu-audit.exit"
set -e
if [ "$cpu_exit" -ne 0 ]; then exit "$cpu_exit"; fi
cpu_end=$(date +%s)
set +e
/home/darshith/ember-ws/venvs/9613f929b53c0fed/bin/python "$ops/r22-gpu-audit.py" > /home/darshith/ember-ws/rl-evidence/r22-gpu-audit.console.log 2>&1
gpu_exit=$?
printf '%s\n' "$gpu_exit" > "$ops/r22-gpu-audit.exit"
set -e
if [ "$gpu_exit" -ne 0 ]; then exit "$gpu_exit"; fi
gpu_end=$(date +%s)
/home/darshith/ember-ws/cpu-tests/bin/python -c 'import json,sys;from pathlib import Path;(Path(sys.argv[1])/"audit_compute.json").write_text(json.dumps({"cpu_audit_seconds":int(sys.argv[2]),"gpu_audit_seconds":int(sys.argv[3])},indent=2))' "$root" "$((cpu_end-started))" "$((gpu_end-cpu_end))"
/home/darshith/ember-ws/cpu-tests/bin/python /mnt/c/Users/Admin/Documents/Test/ws-tmp/r22-archive.py
