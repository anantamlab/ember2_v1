from pathlib import Path
import json,hashlib,sys
p=Path('/home/darshith/ember-ws/rl-evidence/r22-counterfactual-ppo@1/attempt-2')
w=Path('/mnt/c/Users/Admin/Documents/ember-evidence/r22-counterfactual-ppo@1/attempt-2')
ops=Path('/mnt/c/Users/Admin/Documents/Test/ws-tmp')
def j(d,n):return json.loads((d/n).read_text())
def sha(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
exits={n:int((ops/n).read_text()) for n in ('r22-train.exit','r22-cpu-audit.exit','r22-gpu-audit.exit','r22-completion.exit')}
assert set(exits.values())=={0},exits
bound=j(p,'FINAL_AUDIT_EXIT_CODES.json')
assert bound=={'status':'PASS','exit_codes':{k:v for k,v in exits.items() if k!='r22-completion.exit'}}
s=j(p,'summary.json')
assert s['status']=='COMPLETE' and not s['final_evaluation_run']
a=j(p,'independent_counterfactual_audit.json')
g=j(p,'optimizer_reload_audit.json')
assert a['status']==g['status']=='PASS'
assert g['torch_num_threads']==4
assert all(d['saved_actor_max_probability_error']<=1e-5 and d['reporting_decisions_reproduced'] for d in g['arms'].values())
best=j(p,'best_checkpoint.json')
manifest=j(p,'EVIDENCE_MANIFEST.json')
entries={f['path']:f for f in manifest['files']}
actual_files={str(f.relative_to(p)) for f in p.rglob('*') if f.is_file() and f.name!='EVIDENCE_MANIFEST.json'}
assert actual_files==set(entries) and len(entries)==len(manifest['files'])
manifest_sha=sha(p/'EVIDENCE_MANIFEST.json')
copy=j(w,'COPY_VERIFICATION.json')
assert copy['status']=='PASS'
assert {f['path'] for f in copy['files']}=={str(f.relative_to(w)) for f in w.rglob('*') if f.is_file() and f.name!='COPY_VERIFICATION.json'}
for f in copy['files']:
 path=f['path']; parts=Path(path).parts
 canonical=path
 if len(parts)>2 and parts[0] in ('ppo','cf') and parts[1]=='selected_checkpoint':
  canonical=str(Path(best['trained_arms'][parts[0]]['checkpoint']).relative_to(p).joinpath(*parts[2:]))
 assert (w/path).stat().st_size==f['size'] and sha(w/path)==f['sha256'],path
 expected=manifest_sha if path=='EVIDENCE_MANIFEST.json' else entries[canonical]['sha256']
 assert expected==f['sha256'],path
for arm,item in best['trained_arms'].items():
 rel=Path(item['checkpoint']).relative_to(p)
 assert g['arms'][arm]['loaded_digest']==item['identity']['actor_digest']
 assert j(w/arm/'selected_checkpoint','identity.json')==item['identity']
 for f in item['identity']['files']:
  assert entries[str(rel/f['path'])]['sha256']==f['sha256']
assert best['frozen_candidate']['name']=='ppo' and best['frozen_candidate']['identity']==best['trained_arms']['ppo']['identity']
old=j(p,'ANCESTRY_MANIFEST.json')
assert Path(old['canonical_root'])==p.parent/'attempt-1'
for f in old['files']:
 path=p.parent/'attempt-1'/f['path']
 assert path.stat().st_size==f['size'] and sha(path)==f['sha256'],f['path']
out={'status':'PASS','actual_exit_codes':exits,'manifest_sha256':manifest_sha,'manifest_files':len(entries),
 'manifest_bytes':manifest['bytes'],'ancestry_manifest_sha256':sha(p/'ANCESTRY_MANIFEST.json'),
 'ancestry_files_rehashed':len(old['files']),'copy_files_rehashed':len(copy['files']),
 'copy_bytes_rehashed':sum(f['size'] for f in copy['files']),'copy_verification_sha256':sha(w/'COPY_VERIFICATION.json'),
 'selected_identity_sha256':{k:sha(w/k/'selected_checkpoint'/'identity.json') for k in ('ppo','cf')}}
(ops/'r22-publication-verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
