#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/Admin/Documents/Test/ember_v1
export PYTHONPATH="$PWD/src" HF_HOME=/home/darshith/ember-ws/hf-home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 R22_ATTEMPT=2
unset HF_TOKEN
ops=/mnt/c/Users/Admin/Documents/Test/ws-tmp
evidence=/home/darshith/ember-ws/rl-evidence
root=$evidence/r22-counterfactual-ppo@1/attempt-2
test "$(tr -d '\r\n' < "$ops/r22-train.exit")" = 0
test "$(tr -d '\r\n' < "$ops/r22-cpu-audit.exit")" = 0
test -f "$ops/r22-attempt-2-first-audit/r22-gpu-audit.exit"
test ! -e "$root/EVIDENCE_MANIFEST.json"
started=$(date +%s)
set +e
/home/darshith/ember-ws/venvs/9613f929b53c0fed/bin/python "$ops/r22-gpu-audit.py" > "$evidence/r22-gpu-audit-retry.console.log" 2>&1
gpu_exit=$?
printf '%s\n' "$gpu_exit" > "$ops/r22-gpu-audit.exit"
set -e
if [ "$gpu_exit" -ne 0 ]; then exit "$gpu_exit"; fi
gpu_end=$(date +%s)
/home/darshith/ember-ws/cpu-tests/bin/python -c 'import json,sys;from pathlib import Path;(Path(sys.argv[1])/"audit_compute.json").write_text(json.dumps({"gpu_retry_seconds":int(sys.argv[2]),"prior_retry_verification_reserve_seconds":40,"first_completion_and_reload_diagnostic_budgeted_seconds":360,"retry_process_cap_seconds":3200},indent=2))' "$root" "$((gpu_end-started))"
/home/darshith/ember-ws/cpu-tests/bin/python "$ops/r22-archive.py"
