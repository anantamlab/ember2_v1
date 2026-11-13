$trainingPid = [int](Get-Content C:/Users/Admin/Documents/Test/ws-tmp/r21-train-3.pid)
Wait-Process -Id $trainingPid -ErrorAction SilentlyContinue
# Wait for the actual audit processes, not a timed guess.
$auditorPid = 40120
$replayPid = 15128
Wait-Process -Id $auditorPid,$replayPid -ErrorAction SilentlyContinue
& wsl.exe -e bash -lc 'cd /mnt/c/Users/Admin/Documents/Test/ember_v1 && export PYTHONPATH=$PWD/src && /home/darshith/ember-ws/cpu-tests/bin/python /mnt/c/Users/Admin/Documents/Test/ws-tmp/r21-archive.py' *> C:/Users/Admin/Documents/Test/ws-tmp/r21-archive.console.log
$LASTEXITCODE | Set-Content C:/Users/Admin/Documents/Test/ws-tmp/r21-archive.exit