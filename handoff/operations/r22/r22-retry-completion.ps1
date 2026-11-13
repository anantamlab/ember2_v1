$ops='C:/Users/Admin/Documents/Test/ws-tmp'
& wsl.exe -e timeout --signal=TERM 3200 bash /mnt/c/Users/Admin/Documents/Test/ws-tmp/r22-retry-completion.sh *> "$ops/r22-completion-retry.console.log"
$LASTEXITCODE | Set-Content "$ops/r22-completion.exit"
