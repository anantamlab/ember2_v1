param([int]$VerificationPid = 0)
$ErrorActionPreference='Stop'
try {
 if ($VerificationPid) { Wait-Process -Id $VerificationPid -ErrorAction SilentlyContinue }
 $repo='C:/Users/Admin/Documents/Test/ember_v1'
 $sourceHead=(& git -C $repo rev-parse HEAD).Trim()
 if ((& git -C $repo status --porcelain)) {throw 'Source tree is not clean'}
 $receiptFile=Get-ChildItem -LiteralPath "$repo/.git/ember-local-verify/receipts" -Filter "$sourceHead-py3.11-*.json" | Sort-Object Name -Descending | Select-Object -First 1
 if (!$receiptFile) {throw 'No full-suite receipt for current source'}
 $receipt=Get-Content -LiteralPath $receiptFile.FullName -Raw | ConvertFrom-Json
 if ($receipt.status -ne 'PASS' -or !$receipt.tree_clean -or !$receipt.tree_unchanged) {throw 'Full-suite verification did not pass cleanly'}
 $files=@(& git -C $repo ls-files -- src/ember_v1) | Where-Object {$_ -like '*.py'}
 $files+=@('data/rl/r22-counterfactual-ppo@1.json','experiments/rl_cf_report.py','experiments/rl_cf_gpu_checks.py','experiments/rl_gpu_checks.py','experiments/rl_online_report.py')
 $fingerprints=[ordered]@{}
 foreach ($relative in $files) {$fingerprints[$relative]=(Get-FileHash -LiteralPath "$repo/$relative" -Algorithm SHA256).Hash.ToLowerInvariant()}
 $testLog='.local-test-logs/'+[IO.Path]::GetFileName($receipt.log)
 $ready=[ordered]@{status='READY';verified_source_head=$sourceHead;full_test_receipt=$receipt;local_test_log=$testLog;runtime_file_hashes=$fingerprints;phase2_dependency='r21-online-ppo@1/attempt-3 COMPLETE plus all final audit exits';phase3_authorization='Owner Phase3 request in current thread';final_evaluation_authorized=$false}
 $ready | ConvertTo-Json -Depth 25 | Set-Content -LiteralPath C:/Users/Admin/Documents/Test/ws-tmp/r22-launch-ready.json -Encoding utf8
 0 | Set-Content C:/Users/Admin/Documents/Test/ws-tmp/r22-ready-maker.exit
} catch {
 $_.Exception.Message | Set-Content C:/Users/Admin/Documents/Test/ws-tmp/r22-ready-maker.failure
 1 | Set-Content C:/Users/Admin/Documents/Test/ws-tmp/r22-ready-maker.exit
 exit 1
}
