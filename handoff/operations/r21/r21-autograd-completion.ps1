$trainId = [int](Get-Content C:/Users/Admin/Documents/Test/ws-tmp/r21-train-3.pid)
Wait-Process -Id $trainId -ErrorAction SilentlyContinue
& wsl.exe -e bash /mnt/c/Users/Admin/Documents/Test/ws-tmp/r21-autograd.sh *> C:/Users/Admin/Documents/Test/ws-tmp/r21-autograd.console.log
$LASTEXITCODE | Set-Content C:/Users/Admin/Documents/Test/ws-tmp/r21-autograd.exit