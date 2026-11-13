from pathlib import Path
import hashlib,json,math,time
from ember_v1.rl_verify import verify
from ember_v1.rl_cf_verify import verify_study_path
from ember_v1.rl_cf_training import prioritize_visits
p=Path('/home/darshith/ember-ws/rl-evidence/r22-counterfactual-ppo@1/attempt-1')
started=time.monotonic()
q=p/'resume/round-0004'
identity=json.loads((q/'identity.json').read_text())
with (q/'state.pt').open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==identity['sha256']
out=p.parent/'attempt-1-round-0004-verification'
out.mkdir(exist_ok=False)
audits={}
for a in ('ppo','cf'):
 d=out/a; d.mkdir()
 for n in ('execution.json','scenarios.json'):
  (d/n).symlink_to(p/a/n)
 for n in ('episodes.jsonl','transitions.jsonl','ppo_targets.jsonl'):
  with (p/a/n).open() as stream,(d/n).open('w') as target:
   for l in stream:
    if not l.endswith('\n'):continue
    row=json.loads(l)
    if row['label'] in ('train-0','train-1','train-2','train-3'):target.write(l)
 audits[a]=verify(d)
updates=[json.loads(l) for l in (p/'cf'/'updates.jsonl').open()]
n=0
for u in updates[:4]:
 for e in u['counterfactual_studies']:
  h,proof=verify_study_path(e['path']);assert h==e['study_hash'] and proof['visited_source_verified'] and proof['continuation_recomputed']
  n+=1
r={'status':'PASS','resume':identity,'online':audits,'committed_studies_replayed':n,'seconds':time.monotonic()-started}
(out/'result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
