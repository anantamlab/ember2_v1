import hashlib
import json
import shutil
from pathlib import Path

base=Path('/home/darshith/ember-ws/rl-evidence')
root=base/'r20-ppo-pilot@1/attempt-6'
repo=Path('/mnt/c/Users/Admin/Documents/Test/ember_v1')
for source, target in [('attempt-6-console.log','console.log'),('attempt-6-audit-console.log','audit_console.log')]:
    shutil.copyfile(base/source,root/target)
failures=[]
for attempt in range(1,6):
    p=root.parent/f'attempt-{attempt}'
    failure=json.loads((p/'failure.json').read_text())
    failures.append({'attempt':attempt,**failure})
(root/'negative_attempts.json').write_text(json.dumps(failures,indent=2)+'\n')
shutil.copyfile(repo/'docs/R20_PPO_PHASE1_REPORT.md',root/'REPORT.md')
table=[]
for p in sorted(root.rglob('*')):
    if not p.is_file() or p.name in {'EVIDENCE_MANIFEST.json','SHA256SUMS'}:
        continue
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    table.append({'path':p.relative_to(root).as_posix(),'size':p.stat().st_size,'sha256':h.hexdigest()})
manifest={'artifact':'r20-ppo-pilot@1/attempt-6','status':'COMPLETE','training_code_commit':json.load(open(root/'execution.json'))['code_commit'],
          'files':table,'bytes':sum(f['size'] for f in table)}
(root/'EVIDENCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
(root/'SHA256SUMS').write_text(''.join(f"{f['sha256']}  {f['path']}\n" for f in table))
shutil.copyfile(root/'EVIDENCE_MANIFEST.json',repo/'data/rl/evidence/r20-ppo-pilot@1/EVIDENCE_MANIFEST.json')
print({'files':len(table),'bytes':manifest['bytes'],'manifest_sha256':hashlib.sha256((root/'EVIDENCE_MANIFEST.json').read_bytes()).hexdigest()})
