$trainId = [int](Get-Content C:/Users/Admin/Documents/Test/ws-tmp/r21-train-3.pid)
Wait-Process -Id $trainId -ErrorAction SilentlyContinue
& wsl.exe -e bash -lc 'cd /mnt/c/Users/Admin/Documents/Test/ember_v1 && export PYTHONPATH=$PWD/src && /home/darshith/ember-ws/cpu-tests/bin/python -m ember_v1.rl_verify /home/darshith/ember-ws/rl-evidence/r21-online-ppo@1/attempt-3 > /home/darshith/ember-ws/rl-evidence/r21-audit-3.console.log 2>&1'
$LASTEXITCODE | Set-Content C:/Users/Admin/Documents/Test/ws-tmp/r21-audit-3.exit