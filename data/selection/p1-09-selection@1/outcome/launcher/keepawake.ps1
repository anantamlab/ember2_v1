# Keep the system awake (no sleep) until the confirmation launcher writes run_exit.txt, or 5 h pass.
$out = 'C:\Users\sanka\AppData\Local\Temp\claude\1817b1d6ffc1cf67\45b40714-8984-4bd7-b28d-8bebfcaf89ff\scratchpad\run'
Add-Type -Namespace W -Name P -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
$deadline = (Get-Date).AddHours(5)
while (-not (Test-Path "$out\run_exit.txt") -and (Get-Date) -lt $deadline) {
    [void][W.P]::SetThreadExecutionState([uint32]2147483649)   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    Start-Sleep -Seconds 30
}
[void][W.P]::SetThreadExecutionState([uint32]2147483648)       # ES_CONTINUOUS: release
"keepawake end $(Get-Date -Format o)" | Out-File "$out\keepawake_end.txt"
