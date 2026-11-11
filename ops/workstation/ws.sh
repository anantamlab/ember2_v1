#!/usr/bin/env bash
# EMBER v1 workstation wrapper (Linux or WSL2). Execution only: never edit code here.
#
#   ws.sh doctor                         OS, CPU, RAM, disk, GPU, driver, compiler (no installs)
#   ws.sh setup <commit>                 checkout <commit> + Python 3.11 venv from the hash-locked requirements
#   ws.sh login                          HF token for this machine (prompted, never echoed, stored in $HF_HOME)
#   ws.sh prepare <commit> <config> [--model-revision R] [--dataset-revision R] [--experiments-revision R] [--attempt N]
#   ws.sh run <run_id> [attempt]         start detached (survives closing the terminal); log under the run dir
#   ws.sh fg <run_id> <attempt> [--resume]   foreground execution (Windows launcher)
#   ws.sh resume <run_id> [attempt]      continue an INTERRUPTED / FAILED run from its partial output
#   ws.sh status <run_id> [attempt]
#   ws.sh logs <run_id> [attempt]        follow the newest console log
#   ws.sh publish <run_id> [attempt] [--allow-incomplete]
#   ws.sh logout                         remove the HF token from this machine
#
# Environment: EMBER_WS_ROOT (default ~/ember-ws), EMBER_REPO_URL (default git@github.com:anantamlab/ember_v1.git)
set -euo pipefail

ROOT="${EMBER_WS_ROOT:-$HOME/ember-ws}"
REPO_URL="${EMBER_REPO_URL:-git@github.com:anantamlab/ember_v1.git}"
UV_VERSION="0.8.13"
export HF_HOME="$ROOT/hf-home"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
unset HF_TOKEN || true                       # only the token stored by `ws.sh login` is used

die() { echo "ws.sh: $*" >&2; exit 2; }
need_commit() { [[ "${1:-}" =~ ^[0-9a-f]{40}$ ]] || die "need a full 40-hex git commit (never a branch name), got '${1:-}'"; }

checkout_dir() { echo "$ROOT/checkouts/$1"; }

ensure_checkout() {
  local c="$1" d
  d="$(checkout_dir "$c")"
  mkdir -p "$ROOT/checkouts"
  if [ ! -d "$ROOT/repo" ]; then git clone --no-checkout "$REPO_URL" "$ROOT/repo"; fi
  if [ ! -d "$d" ]; then
    git -C "$ROOT/repo" fetch --quiet origin
    git -C "$ROOT/repo" cat-file -e "$c^{commit}" 2>/dev/null || die "commit $c is not on GitHub (push it from the laptop)"
    git -C "$ROOT/repo" worktree add --detach "$d" "$c" >/dev/null
  fi
  [ "$(git -C "$d" rev-parse HEAD)" = "$c" ] || die "$d is not at $c"
  [ -z "$(git -C "$d" status --porcelain)" ] || die "$d has local changes; execution checkouts are never edited"
}

lock_key() { sha256sum "$(checkout_dir "$1")/infra/p0b-requirements.lock.txt" | cut -c1-16; }
venv_dir() { echo "$ROOT/venvs/$(lock_key "$1")"; }

ensure_uv() {
  if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
    curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | sh
  fi
  export PATH="$HOME/.local/bin:$PATH"
}

py() {  # py <commit> args...  (the venv of that commit's lock, the code of that commit)
  local c="$1"; shift
  local v; v="$(venv_dir "$c")"
  [ -f "$v/.ember-ok" ] || die "no complete environment for $c: run 'ws.sh setup $c' first"
  PYTHONPATH="$(checkout_dir "$c")/src" EMBER_WS_ROOT="$ROOT" "$v/bin/python" "$@"
}

run_commit() {  # commit of a prepared run
  local m="$ROOT/runs/$1/attempt-${2:-1}/execution_manifest.json"
  [ -f "$m" ] || die "run $1 attempt ${2:-1} is not prepared ($m)"
  python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["code"]["commit"])' "$m"
}

cmd="${1:-}"; shift || true
case "$cmd" in
  doctor)
    echo "== OS";      (cat /etc/os-release 2>/dev/null | grep -E '^(NAME|VERSION)=') || true; uname -sr
    grep -qi microsoft /proc/version 2>/dev/null && echo "WSL2: yes" || echo "WSL2: no"
    echo "== CPU";     (lscpu 2>/dev/null | grep -E 'Model name|^CPU\(s\)') || nproc
    echo "== RAM";     free -g | head -2
    echo "== Disk ($ROOT)"; mkdir -p "$ROOT"; df -h "$ROOT" | tail -1
    echo "== GPU";     nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv || echo "nvidia-smi FAILED"
    drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)"
    if [ -n "$drv" ] && [ "$drv" -ge 570 ]; then echo "driver >= 570 (CUDA 12.8 wheels): OK"; else echo "driver must be >= 570 for the CUDA 12.8 PyTorch 2.8.0 wheels: NOT OK"; fi
    echo "== Tools";   for t in git gcc curl python3; do printf '%-8s %s\n' "$t" "$(command -v $t || echo MISSING)"; done
    echo "(gcc is needed by triton's JIT; Ubuntu: sudo apt-get install -y build-essential git curl)"
    ;;
  setup)
    need_commit "${1:-}"; c="$1"
    ensure_checkout "$c"; ensure_uv
    v="$(venv_dir "$c")"
    if [ ! -f "$v/.ember-ok" ]; then          # an unfinished venv is rebuilt from scratch
      rm -rf "$v"
      uv python install 3.11
      uv venv --python 3.11 "$v"
      VIRTUAL_ENV="$v" uv pip install --require-hashes --no-deps -r "$(checkout_dir "$c")/infra/p0b-requirements.lock.txt"
      date -u +%FT%TZ > "$v/.ember-ok"
    fi
    py "$c" -m ember_v1.p0b verify-install
    py "$c" -c 'import torch;print("torch",torch.__version__,"cuda",torch.version.cuda,"available",torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",torch.cuda.get_arch_list() if torch.cuda.is_available() else "")'
    echo "setup OK: checkout $(checkout_dir "$c"), venv $v"
    ;;
  login)
    mkdir -p "$HF_HOME"
    any="$(dirname "$(ls "$ROOT"/venvs/*/.ember-ok 2>/dev/null | head -1)")"; [ "$any" != "." ] || die "run 'ws.sh setup <commit>' first"
    "$any/bin/hf" auth login && "$any/bin/hf" auth whoami
    ;;
  prepare)
    need_commit "${1:-}"; c="$1"; cfg="${2:-}"; shift 2 || die "usage: ws.sh prepare <commit> <config> [...]"
    ensure_checkout "$c"
    py "$c" -m ember_v1.workstation prepare-run --commit "$c" --config "$cfg" "$@"
    ;;
  fg)  # foreground execution (used by ws.ps1 inside a hidden wsl.exe that lives as long as the run)
    id="${1:-}"; att="${2:-1}"; shift 2 || die "usage: ws.sh fg <run_id> <attempt> [--resume]"
    c="$(run_commit "$id" "$att")"; ensure_checkout "$c"
    d="$ROOT/runs/$id/attempt-$att"; mkdir -p "$d/logs"
    py "$c" -m ember_v1.workstation execute-run --run "$id" --attempt "$att" "$@"       >"$d/logs/console-$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1
    ;;
  run|resume)
    id="${1:-}"; att="${2:-1}"; [ -n "$id" ] || die "usage: ws.sh $cmd <run_id> [attempt]"
    c="$(run_commit "$id" "$att")"; ensure_checkout "$c"
    d="$ROOT/runs/$id/attempt-$att"; mkdir -p "$d/logs"
    log="$d/logs/console-$(date -u +%Y%m%dT%H%M%SZ).log"
    extra=(); [ "$cmd" = resume ] && extra=(--resume)
    v="$(venv_dir "$c")"
    PYTHONPATH="$(checkout_dir "$c")/src" EMBER_WS_ROOT="$ROOT" nohup setsid "$v/bin/python" -m ember_v1.workstation \
      execute-run --run "$id" --attempt "$att" "${extra[@]}" >"$log" 2>&1 < /dev/null &
    echo "started pid $! ; console log: $log"
    echo "check: ws.sh status $id $att   follow: ws.sh logs $id $att"
    ;;
  status)
    id="${1:-}"; att="${2:-1}"; c="$(run_commit "$id" "$att")"
    py "$c" -m ember_v1.workstation status --run "$id" --attempt "$att"
    ;;
  logs)
    id="${1:-}"; att="${2:-1}"
    tail -n 50 -f "$(ls -t "$ROOT/runs/$id/attempt-$att"/logs/console-*.log | head -1)"
    ;;
  publish)
    id="${1:-}"; att="${2:-1}"; shift 2 || shift || true
    c="$(run_commit "$id" "$att")"
    py "$c" -m ember_v1.workstation publish-results --run "$id" --attempt "$att" "$@"
    ;;
  logout)
    any="$(dirname "$(ls "$ROOT"/venvs/*/.ember-ok 2>/dev/null | head -1)")"
    [ "$any" != "." ] && "$any/bin/hf" auth logout || true
    rm -f "$HF_HOME/token" "$HF_HOME/stored_tokens"
    echo "HF token removed from $HF_HOME. Also revoke it on huggingface.co if the machine is shared."
    ;;
  *) sed -n '2,/^set -euo/p' "$0" | sed '$d'; exit 2 ;;
esac
