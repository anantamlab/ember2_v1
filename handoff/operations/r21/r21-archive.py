from pathlib import Path
import hashlib,json,shutil,subprocess,sys
root=Path('/home/darshith/ember-ws/rl-evidence/r21-online-ppo@1/attempt-3')
assert json.loads((root/'summary.json').read_text())['status']=='COMPLETE'
assert json.loads((root/'independent_replay.json').read_text())['status']=='PASS'
assert json.loads((root/'optimizer_resume_audit.json').read_text())['status']=='PASS'
exit_codes={name:int((Path('/mnt/c/Users/Admin/Documents/Test/ws-tmp')/name).read_text().strip()) for name in ('r21-audit-3.exit','r21-autograd.exit')}
assert all(code==0 for code in exit_codes.values()),exit_codes
(root/'FINAL_AUDIT_EXIT_CODES.json').write_text(json.dumps({'status':'PASS','exit_codes':exit_codes},indent=2)+'\n')
subprocess.run([sys.executable,'experiments/rl_online_report.py',str(root)],check=True)
subprocess.run([sys.executable,'/mnt/c/Users/Admin/Documents/Test/ws-tmp/r21-probe-analysis.py'],check=True)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
ops=root/'reproduction_ops'
ops.mkdir(exist_ok=True)
for name in ('r21-train-3.sh','r21-autograd.py','r21-autograd.sh','r21-archive.py','r21-probe-analysis.py','r21-midrun-audit.py'):
 shutil.copy2(Path('/mnt/c/Users/Admin/Documents/Test/ws-tmp')/name,ops/name)
files=[]
for p in sorted(root.rglob('*')):
 if p.is_file() and p.name!='EVIDENCE_MANIFEST.json':files.append({'path':str(p.relative_to(root)),'size':p.stat().st_size,'sha256':digest(p)})
manifest={'artifact':'r21-online-ppo@1/attempt-3','files':files,'bytes':sum(r['size'] for r in files)}
(root/'EVIDENCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
dest=Path('/mnt/c/Users/Admin/Documents/ember-evidence/r21-online-ppo@1/attempt-3')
dest.mkdir(parents=True,exist_ok=False)
best=json.loads((root/'best_checkpoint.json').read_text())
selected=Path(best['actor']).parent
for p in root.iterdir():
 if p.is_file():shutil.copy2(p,dest/p.name)
shutil.copytree(selected,dest/'selected_checkpoint')
shutil.copytree(root/'reproduction_ops',dest/'reproduction_ops')
checks=[]
for p in sorted(dest.rglob('*')):
 if p.is_file():
  rel=p.relative_to(dest)
  original=(selected/Path(*rel.parts[1:])) if rel.parts[0]=='selected_checkpoint' else root/rel
  assert p.stat().st_size==original.stat().st_size and digest(p)==digest(original)
  checks.append({'path':str(rel),'size':p.stat().st_size,'sha256':digest(p)})
(dest/'COPY_VERIFICATION.json').write_text(json.dumps({'status':'PASS','canonical_evidence':str(root),'files':checks},indent=2))
print(json.dumps({'status':'PASS','files':len(checks),'bytes':sum(r['size'] for r in checks),'manifest_sha256':digest(root/'EVIDENCE_MANIFEST.json'),'copy':str(dest)}))