from pathlib import Path
import json,time,sys
import unsloth
import torch
from ember_v1.rl_ppo import V1Actor
from ember_v1.rl_cf_training import CounterfactualObjective
from ember_v1.rl_run import REPO
from ember_v1.student_train import config_from_protocol,load_protocol
started=time.monotonic()
p=Path('/home/darshith/ember-ws/rl-evidence/r22-counterfactual-ppo@1/attempt-1')
q=p/'resume/round-0004'
binding=json.loads((p/'parent_binding.json').read_text())
saved=torch.load(q/'state.pt',map_location='cpu',weights_only=False)
assert saved['completed_rounds']==4 and saved['parent_binding']==binding
assert saved['config_sha256']==json.loads((q/'identity.json').read_text())['config_sha256']
for arm,s in saved['arms'].items():
 logs=list(map(json.loads,(p/arm/'updates.jsonl').open()))
 ep=list(map(json.loads,(p/arm/'episodes.jsonl').open()))
 assert s['updates']==logs[:4] and s['episodes']==ep[:32]
 assert all(torch.isfinite(t).all() for t in s['critic'].values())
 assert sorted({int(i['step']) for i in s['optimizer']['state'].values()})==[1996+sum(u['optimizer_steps'] for u in logs[:4])]
original=Path('/home/darshith/ember-ws/artifacts/models/1fc405f723718e308c8a22b137fee4a025086fde/v1/p1-18-v1-train@1')
cfg=config_from_protocol(load_protocol(REPO/json.loads((original/'identity.json').read_text())['protocol']['path']))
torch.set_num_threads(4)
actor=V1Actor(cfg,Path(binding['checkpoint'])/'actor')
assert actor.ad.weights_digest()==binding['actor_digest']
for arm,s in saved['arms'].items():
 with torch.no_grad():
  for name,param in actor.model.named_parameters():
   if name in s['trainable']:param.copy_(s['trainable'][name])
 assert actor.ad.weights_digest()==s['actor_digest']
assert actor.ad.weights_digest()==json.loads((p/'counterfactual/update-0005-state-00.json').read_text())['source']['policy_digest']
actor.set_trainable(True)
# Diagnostic only: reproduce the first failed pair on the exact saved behavior
# actor. No new scenarios/teacher calls, no optimizer, no parameter update.
obj=CounterfactualObjective([json.loads((p/'counterfactual/update-0005-state-00.json').read_text())])
assert obj.pairs
with torch.enable_grad():
 loss,e=obj(actor)
 (.05*loss).backward()
norms={}
for group,check in [('lora',lambda n:'lora_' in n),('head',lambda n:n.startswith('head.'))]:
 grads=[t.grad for n,t in actor.model.named_parameters() if check(n) and t.grad is not None]
 assert grads and all(torch.isfinite(t).all() for t in grads)
 norms[group]=sum(float(t.float().square().sum()) for t in grads)**.5
 assert norms[group]>0
assert actor.ad.weights_digest()==saved['arms']['cf']['actor_digest']
r={'status':'PASS','checkpoint_round':4,'replayed_archived_pair':e,'gradient_norms':norms,'optimizer_steps':0,'seconds':time.monotonic()-started}
(p.parent/'attempt-1-round-0004-real-pair-probe.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
