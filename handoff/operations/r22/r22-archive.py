from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time
run_id='r22-counterfactual-ppo@1'
attempt=f"attempt-{os.environ.get('R22_ATTEMPT','1')}"
root=Path('/home/darshith/ember-ws/rl-evidence')/run_id/attempt
assert json.loads((root/'summary.json').read_text())['status']=='COMPLETE'
assert json.loads((root/'independent_counterfactual_audit.json').read_text())['status']=='PASS'
assert json.loads((root/'optimizer_reload_audit.json').read_text())['status']=='PASS'
launcher_ops=Path('/mnt/c/Users/Admin/Documents/Test/ws-tmp')
exit_codes={name:int((launcher_ops/name).read_text().strip())
            for name in ('r22-train.exit','r22-cpu-audit.exit','r22-gpu-audit.exit')}
assert all(code==0 for code in exit_codes.values()),exit_codes
(root/'FINAL_AUDIT_EXIT_CODES.json').write_text(json.dumps({'status':'PASS','exit_codes':exit_codes},indent=2)+'\n')
subprocess.run([sys.executable,'experiments/rl_cf_report.py',str(root)],check=True)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
ops=root/'reproduction_ops';ops.mkdir(exist_ok=True)
for name in ('r22-source-gate.py','r22-launch.sh','r22-launch.ps1','r22-gpu-audit.py','r22-archive.py',
             'r22-completion.sh','r22-completion.ps1','r22-family-isolation.py','r22-family-isolation.json',
             'r22-launch-ready.json','r22-launch-observed.json',*exit_codes):
 shutil.copy2(launcher_ops/name,ops/name)
if attempt == 'attempt-2':
 for name in ('r22-verify-retry-cpu.py','r22-verify-retry-gpu.py','r22-family-isolation-attempt2.json'):
  shutil.copy2(launcher_ops/name,ops/name)
 prior=root.parent/'attempt-1'
 for name,source in (
  ('attempt-1-failure.json',prior/'failure.json'),
  ('attempt-1-train.console.log',launcher_ops/'r22-attempt-1-ops'/'r22-train.console.log'),
  ('attempt-1-train.exit',launcher_ops/'r22-attempt-1-ops'/'r22-train.exit'),
  ('attempt-1-cpu-retry-verification.json',root.parent/'attempt-1-round-0004-verification'/'result.json'),
  ('attempt-1-real-pair-probe.json',root.parent/'attempt-1-round-0004-real-pair-probe.json')):
  shutil.copy2(source,ops/name)
 prior_files=[{'path':str(p.relative_to(prior)),'size':p.stat().st_size,'sha256':digest(p)}
              for p in sorted(prior.rglob('*')) if p.is_file()]
 (root/'ANCESTRY_MANIFEST.json').write_text(json.dumps({'canonical_root':str(prior),'files':prior_files},indent=2)+'\n')
 first_audit=launcher_ops/'r22-attempt-2-first-audit'
 if first_audit.is_dir():
  shutil.copytree(first_audit,ops/'first-completion-audit')
 for name in ('r22-retry-completion.sh','r22-retry-completion.ps1','r22-reload-diagnostic.py'):
  if (launcher_ops/name).exists():
   shutil.copy2(launcher_ops/name,ops/name)
 audit_retry=root.parent.parent/'r22-gpu-audit-retry.console.log'
 if audit_retry.is_file():
  shutil.copy2(audit_retry,ops/audit_retry.name)
files=[{'path':str(p.relative_to(root)),'size':p.stat().st_size,'sha256':digest(p)} for p in sorted(root.rglob('*')) if p.is_file() and p.name!='EVIDENCE_MANIFEST.json']
(root/'EVIDENCE_MANIFEST.json').write_text(json.dumps({'artifact':f'{run_id}/{attempt}','files':files,'bytes':sum(f['size'] for f in files)},indent=2)+'\n')
dest=Path('/mnt/c/Users/Admin/Documents/ember-evidence')/run_id/attempt
dest.mkdir(parents=True,exist_ok=False)
for p in root.iterdir():
 if p.is_file():shutil.copy2(p,dest/p.name)
for name in ('counterfactual','reproduction_ops','audit_inputs'):
 if (root/name).exists():shutil.copytree(root/name,dest/name)
best=json.loads((root/'best_checkpoint.json').read_text())
for arm,item in best['trained_arms'].items():
 d=dest/arm;d.mkdir()
 for p in (root/arm).iterdir():
  if p.is_file():shutil.copy2(p,d/p.name)
 shutil.copytree(Path(item['checkpoint']),d/'selected_checkpoint')
checks=[]
for p in sorted(dest.rglob('*')):
 if not p.is_file():continue
 rel=p.relative_to(dest);parts=rel.parts
 source=Path(best['trained_arms'][parts[0]]['checkpoint'])/Path(*parts[2:]) if len(parts)>1 and parts[0] in ('ppo','cf') and parts[1]=='selected_checkpoint' else root/rel
 assert p.stat().st_size==source.stat().st_size and digest(p)==digest(source)
 checks.append({'path':str(rel),'size':p.stat().st_size,'sha256':digest(p)})
(dest/'COPY_VERIFICATION.json').write_text(json.dumps({'status':'PASS','canonical_evidence':str(root),'files':checks},indent=2))
print(json.dumps({'status':'PASS','copy':str(dest),'files':len(checks),'bytes':sum(f['size'] for f in checks),'manifest_sha256':digest(root/'EVIDENCE_MANIFEST.json')}))
