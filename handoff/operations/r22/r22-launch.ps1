param([int]$Attempt = 1, [int]$WallSeconds = 14400, [string]$Resume = '')
$ErrorActionPreference='Stop'
try {
 [IO.File]::WriteAllText('C:/Users/Admin/Documents/Test/ws-tmp/r22-launch-status.json','{"status":"WAITING_FOR_VERIFIED_PHASE2"}')
 foreach ($name in 'r21-audit-3.exit','r21-autograd.exit','r21-archive.exit') {
   $path = Join-Path 'C:/Users/Admin/Documents/Test/ws-tmp' $name
   if (!(Test-Path -LiteralPath $path) -or [int](Get-Content -LiteralPath $path) -ne 0) {
     throw "Phase 2 completion gate failed: $name"
   }
 }
 $launchArgs = @($Attempt, $WallSeconds)
 if ($Resume) { $launchArgs += $Resume }
 & wsl.exe -e bash /mnt/c/Users/Admin/Documents/Test/ws-tmp/r22-launch.sh @launchArgs *> C:/Users/Admin/Documents/Test/ws-tmp/r22-train.console.log
 $runExit=$LASTEXITCODE
 $runExit | Set-Content C:/Users/Admin/Documents/Test/ws-tmp/r22-train.exit
 @{status='EXITED';exit_code=$runExit} | ConvertTo-Json | Set-Content C:/Users/Admin/Documents/Test/ws-tmp/r22-launch-status.json
} catch {
 @{status='DEPENDENCY_OR_LAUNCH_FAILURE';message=$_.Exception.Message} | ConvertTo-Json | Set-Content C:/Users/Admin/Documents/Test/ws-tmp/r22-launch-status.json
 exit 1
}
