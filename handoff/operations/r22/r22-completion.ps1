param([Parameter(Mandatory=$true)][int]$TrainingPid, [int]$Attempt = 1, [int]$WallSeconds = 3600)
Wait-Process -Id $TrainingPid -ErrorAction SilentlyContinue
if (!(Test-Path C:/Users/Admin/Documents/Test/ws-tmp/r22-train.exit) -or
    [int](Get-Content C:/Users/Admin/Documents/Test/ws-tmp/r22-train.exit) -ne 0) {
    'Training did not exit successfully; completion audits were not launched.' |
        Set-Content C:/Users/Admin/Documents/Test/ws-tmp/r22-completion.console.log
    1 | Set-Content C:/Users/Admin/Documents/Test/ws-tmp/r22-completion.exit
    exit 1
}
& wsl.exe -e timeout --signal=TERM $WallSeconds bash /mnt/c/Users/Admin/Documents/Test/ws-tmp/r22-completion.sh $Attempt *> C:/Users/Admin/Documents/Test/ws-tmp/r22-completion.console.log
$LASTEXITCODE | Set-Content C:/Users/Admin/Documents/Test/ws-tmp/r22-completion.exit
