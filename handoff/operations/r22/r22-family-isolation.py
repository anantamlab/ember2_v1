from pathlib import Path
import json,time,hashlib
from ember_v1.rl_cf_scenarios import scenarios
from ember_v1.gen.analysis import topology_hash
from ember_v1.sim import Scenario
repo=Path('/mnt/c/Users/Admin/Documents/Test/ember_v1')
cfg=json.loads((repo/'data/rl/r22-counterfactual-ppo@1.json').read_text())
def h(s):return topology_hash(s.height,s.width,s.obstacles,[p for _,p in s.exits])
new=[s for stage in range(5) for s,_ in scenarios(cfg['scenario_namespace'],'train',cfg['updates']*cfg['episodes_per_update'],stage=stage,max_steps=cfg['max_steps'])]
new += [s for split in ('select','report') for s,_ in scenarios(cfg['scenario_namespace'],split,cfg['dev_episodes'],max_steps=cfg['max_steps'])]
newhash={h(s) for s in new};assert len(newhash)==len(new)
counts={}
for label,path in (('phase1','/home/darshith/ember-ws/rl-evidence/r20-ppo-pilot@1/attempt-6/scenarios.json'),('phase2','/home/darshith/ember-ws/rl-evidence/r21-online-ppo@1/attempt-3/scenarios.json')):
 old=json.loads(Path(path).read_text());oldhash={h(Scenario.from_dict(s)) for split in ('train','dev','select','report') for s in old.get(split,[])}
 counts[label]=len(oldhash&newhash);assert counts[label]==0
result={'status':'PASS','new_topology_families':len(newhash),'overlap':counts,'namespace':cfg['scenario_namespace'],'config_sha256':hashlib.sha256((repo/'data/rl/r22-counterfactual-ppo@1.json').read_bytes()).hexdigest(),'time_unix':time.time()}
Path('/mnt/c/Users/Admin/Documents/Test/ws-tmp/r22-family-isolation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))