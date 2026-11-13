<#
EMBER v1 workstation launcher for a WINDOWS host. Execution only: never edit code on the workstation.

GPU runs go through WSL2 (the pinned unsloth / triton / CUDA 12.8 stack is Linux-only; native Windows GPU
execution is not supported). This script checks the host, bootstraps the WSL side, and starts runs inside a
hidden wsl.exe that lives as long as the run (closing your terminal does not stop it).
The native-Windows route covers CPU work only (trace-replay, fetch/verify, publish).

  .\ws.ps1 doctor                              host OS, RAM, GPU, driver, WSL state
  .\ws.ps1 bootstrap <commit>                  WSL: clone the repo, install ws.sh of <commit> to ~/ember-ws/bin
  .\ws.ps1 run <run_id> [attempt]              start a prepared run detached (hidden wsl.exe)
  .\ws.ps1 resume <run_id> [attempt]           same, continuing from partial output
  .\ws.ps1 wsl <ws.sh args...>                 any ws.sh command inside WSL (setup, login, prepare, status, publish, logout)
  .\ws.ps1 cpu-setup <commit>                  native Windows: checkout + Python 3.11 venv (hash-locked HF client only)
  .\ws.ps1 cpu <commit> <workstation args...>  native Windows: python -m ember_v1.workstation ... (CPU kinds only)

Env: EMBER_WSL_DISTRO (default Ubuntu-24.04), EMBER_WS_ROOT_WIN (default $HOME\ember-ws),
     EMBER_REPO_URL (default git@github.com:anantamlab/ember_v1.git)
#>
param([Parameter(Position = 0)][string]$Cmd = "help", [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest)
$ErrorActionPreference = "Stop"
$Distro = if ($env:EMBER_WSL_DISTRO) { $env:EMBER_WSL_DISTRO } else { "Ubuntu-24.04" }
$RootWin = if ($env:EMBER_WS_ROOT_WIN) { $env:EMBER_WS_ROOT_WIN } else { Join-Path $HOME "ember-ws" }
$RepoUrl = if ($env:EMBER_REPO_URL) { $env:EMBER_REPO_URL } else { "git@github.com:anantamlab/ember_v1.git" }

function Need-Commit([string]$c) { if ($c -notmatch '^[0-9a-f]{40}$') { throw "need a full 40-hex git commit, got '$c'" } }
function In-Wsl([string]$line) { & wsl.exe -d $Distro -- bash -lc $line; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } }

switch ($Cmd) {
  "doctor" {
    $os = Get-CimInstance Win32_OperatingSystem
    "OS: $($os.Caption) $($os.Version)"
    "RAM: {0:N1} GB" -f ($os.TotalVisibleMemorySize / 1MB)
    "CPU: $((Get-CimInstance Win32_Processor | Select-Object -First 1).Name)"
    "Disk free ($((Split-Path $RootWin -Qualifier))): {0:N0} GB" -f ((Get-PSDrive ((Split-Path $RootWin -Qualifier).TrimEnd(':'))).Free / 1GB)
    try { & nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv } catch { "nvidia-smi FAILED: $_" }
    "--- WSL"
    try { & wsl.exe --status; & wsl.exe -l -v } catch { "WSL not installed. Admin PowerShell: wsl --install -d $Distro  (then reboot)" }
    "GPU in WSL2 needs only the Windows NVIDIA driver (>= 570 for CUDA 12.8); never install a Linux driver inside WSL."
  }
  "bootstrap" {
    $c = $Rest[0]; Need-Commit $c
    In-Wsl "set -e; mkdir -p ~/ember-ws/bin; command -v git >/dev/null || { echo 'install git in WSL: sudo apt-get install -y git build-essential curl'; exit 2; }; [ -d ~/ember-ws/repo ] || git clone --no-checkout '$RepoUrl' ~/ember-ws/repo; git -C ~/ember-ws/repo fetch --quiet origin; git -C ~/ember-ws/repo show '${c}:ops/workstation/ws.sh' > ~/ember-ws/bin/ws.sh; chmod +x ~/ember-ws/bin/ws.sh; ~/ember-ws/bin/ws.sh doctor"
  }
  { $_ -in "run", "resume" } {
    $id = $Rest[0]; $att = if ($Rest.Count -gt 1) { $Rest[1] } else { "1" }
    if (-not $id) { throw "usage: ws.ps1 $Cmd <run_id> [attempt]" }
    $flag = if ($Cmd -eq "resume") { " --resume" } else { "" }
    $p = Start-Process -FilePath "wsl.exe" -WindowStyle Hidden -PassThru `
      -ArgumentList @("-d", $Distro, "--", "bash", "-lc", "~/ember-ws/bin/ws.sh fg '$id' '$att'$flag")
    "started hidden wsl.exe pid $($p.Id) (stays alive for the whole run). Check: .\ws.ps1 wsl status $id $att"
  }
  "wsl" {
    $q = ($Rest | ForEach-Object { "'" + ($_ -replace "'", "'\''") + "'" }) -join " "
    In-Wsl "~/ember-ws/bin/ws.sh $q"
  }
  "cpu-setup" {
    $c = $Rest[0]; Need-Commit $c
    New-Item -ItemType Directory -Force -Path "$RootWin\checkouts" | Out-Null
    if (-not (Test-Path "$RootWin\repo")) { git clone --no-checkout $RepoUrl "$RootWin\repo" }
    git -C "$RootWin\repo" fetch --quiet origin
    $d = "$RootWin\checkouts\$c"
    if (-not (Test-Path $d)) { git -C "$RootWin\repo" worktree add --detach $d $c }
    $v = "$RootWin\venv-cpu"
    if (-not (Test-Path "$v\.ember-ok")) {
      if (Test-Path $v) { Remove-Item -Recurse -Force $v }
      & py -3.11 -m venv $v
      & "$v\Scripts\python.exe" -m pip install --require-hashes --no-deps -r "$d\infra\ws-cpu-requirements.lock.txt"
      Get-Date -Format o | Out-File "$v\.ember-ok"
    }
    "CPU route ready: $d with $v"
  }
  "cpu" {
    $v = "$RootWin\venv-cpu"
    if (-not (Test-Path "$v\.ember-ok")) { throw "run .\ws.ps1 cpu-setup <commit> first" }
    $commit = $Rest[0]; Need-Commit $commit
    $wargs = @($Rest | Select-Object -Skip 1)
    if (-not (Test-Path "$RootWin\checkouts\$commit")) { throw "run .\ws.ps1 cpu-setup $commit first" }
    $env:PYTHONPATH = "$RootWin\checkouts\$commit\src"; $env:EMBER_WS_ROOT = $RootWin; $env:HF_HOME = "$RootWin\hf-home"
    Remove-Item Env:HF_TOKEN -ErrorAction SilentlyContinue
    & "$v\Scripts\python.exe" -m ember_v1.workstation @wargs
    exit $LASTEXITCODE
  }
  default { Get-Content $PSCommandPath | Select-Object -Skip 1 -First 18 }
}
