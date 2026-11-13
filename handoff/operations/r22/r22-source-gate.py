from pathlib import Path
import hashlib,json,subprocess,sys,time
repo=Path('/mnt/c/Users/Admin/Documents/Test/ember_v1')
ready=Path('/mnt/c/Users/Admin/Documents/Test/ws-tmp/r22-launch-ready.json')
if not ready.exists():sys.exit(75)
r=json.loads(ready.read_text())
if r.get('status')!='READY':sys.exit(76)
if subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip():sys.exit(75)
family=json.loads(Path('/mnt/c/Users/Admin/Documents/Test/ws-tmp/r22-family-isolation.json').read_text())
if family.get('status')!='PASS' or any(family['overlap'].values()) or family['config_sha256']!=r['runtime_file_hashes']['data/rl/r22-counterfactual-ppo@1.json']:sys.exit(76)
receipt=r['full_test_receipt']
if receipt['status']!='PASS' or not receipt['tree_clean'] or not receipt['tree_unchanged']:sys.exit(76)
for relative,want in r['runtime_file_hashes'].items():
 p=repo/relative
 if repo.resolve() not in p.resolve().parents or hashlib.sha256(p.read_bytes()).hexdigest()!=want:sys.exit(76)
if hashlib.sha256((repo/r['local_test_log']).read_bytes()).hexdigest()!=receipt['log_sha256'].split(':')[-1]:sys.exit(76)
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
out={'status':'READY','actual_launch_head':head,'verified_source_head':r['verified_source_head'],'time_unix':time.time(),'config_sha256':r['runtime_file_hashes']['data/rl/r22-counterfactual-ppo@1.json']}
if '--check-only' in sys.argv:
 print(json.dumps({'source_ready':True,'verified_source_head':r['verified_source_head'],'actual_head':head}));sys.exit(0)
Path('/mnt/c/Users/Admin/Documents/Test/ws-tmp/r22-launch-observed.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out))