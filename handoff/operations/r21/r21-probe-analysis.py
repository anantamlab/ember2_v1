from pathlib import Path
from collections import defaultdict,Counter
import json,math
from ember_v1.rl_env import safe_distances
from ember_v1.shield import admissible_actions
from ember_v1.sim.movement import DELTAS
from ember_v1.student_train import ACTION_ORDER
root=Path('/home/darshith/ember-ws/rl-evidence/r21-online-ppo@1/attempt-3')
rows=defaultdict(dict)
for line in (root/'transitions.jsonl').open():
 r=json.loads(line)
 if r['label'] in ('v1','phase1','ppo') and r['observation']['step']==0:rows[r['label']][r['scenario_id']]=r
assert set(rows['v1'])==set(rows['phase1'])==set(rows['ppo'])
result={}
for arm,states in rows.items():
 totals=Counter();count=0;entropy=stay=toward=legal=kl=0.
 for sid,r in states.items():
  obs=r['observation'];ids=sorted(r['sampled_actions']);positions={p['id']:tuple(p['position']) for p in obs['people']};dist=safe_distances(obs);adm=admissible_actions(obs)
  reference=rows['v1'][sid]
  assert obs==reference['observation']
  for pid,lp,old in zip(ids,r['person_logprobs'],reference['person_logprobs']):
   count+=1;probs=[math.exp(x) for x in lp];entropy-=sum(p*x for p,x in zip(probs,lp));stay+=probs[ACTION_ORDER.index('stay')]
   totals['argmax_'+r['sampled_actions'][pid]]+=1
   totals['argmax_changed_from_sft']+=r['sampled_actions'][pid]!=reference['sampled_actions'][pid]
   kl+=sum(p*(x-y) for p,x,y in zip(probs,lp,old))
   pos=positions[pid];current=dist.get(pos,obs['map']['height']*obs['map']['width'])
   for action,p in zip(ACTION_ORDER,probs):
    legal+=p*(action in adm[pid]);dr,dc=DELTAS[action];target=(pos[0]+dr,pos[1]+dc)
    toward+=p*(action in adm[pid] and dist.get(target,current)>=0 and dist.get(target,current)<current)
 result[arm]={'states':len(states),'people':count,'person_entropy_nats':entropy/count,'stay_probability':stay/count,'static_legal_probability':legal/count,'distance_decrease_probability':toward/count,'kl_to_sft_per_person':kl/count,**dict(totals)}
(root/'initial_state_policy_audit.json').write_text(json.dumps({'matched_initial_public_states':True,'metrics':result,'note':'Public-distance and legality measurements are diagnostics only, never training rewards.'},indent=2)+'\n')
print(json.dumps(result))