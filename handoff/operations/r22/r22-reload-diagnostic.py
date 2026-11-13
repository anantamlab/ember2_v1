from pathlib import Path
import json,gc
import unsloth,torch
from ember_v1.rl_ppo import V1Actor
from ember_v1.student_train import config_from_protocol,load_protocol
from ember_v1.rl_run import REPO
r=Path('/home/darshith/ember-ws/rl-evidence/r22-counterfactual-ppo@1/attempt-2')
model=Path('/home/darshith/ember-ws/artifacts/models/1fc405f723718e308c8a22b137fee4a025086fde/v1/p1-18-v1-train@1')
cfg=config_from_protocol(load_protocol(REPO/json.loads((model/'identity.json').read_text())['protocol']['path']))
torch.set_num_threads(4)
for arm,i in json.loads((r/'best_checkpoint.json').read_text())['trained_arms'].items():
 a=V1Actor(cfg,Path(i['actor']))
 print('ARM',arm,i['update'],a.ad.weights_digest(),flush=True)
 for mode in (False,True):
  a.set_trainable(mode)
  errs=[];mismatches=[];pop=[]
  for x in map(json.loads,(r/arm/'transitions.jsonl').open()):
   if x['label']!=arm:continue
   with torch.set_grad_enabled(mode):
    maps,_,lp,_,full=a.sample(a.prepare([x['observation']]),greedy=True)
   err=float((full[0].cpu()-torch.tensor(x['person_logprobs'])).abs().max())
   errs.append(err);pop.append(len(maps[0]))
   if maps[0]!=x['sampled_actions'] and len(mismatches)<4:
    mismatches.append((len(errs),x['scenario_id'],x['observation']['step'],len(maps[0]),err))
   if len(errs)>=384 and mismatches:break
  print('AUDIT_DIAGNOSTIC',arm,mode,len(errs),max(errs),sum(t>1e-5 for t in errs),mismatches,flush=True)
 del a
 gc.collect();torch.cuda.empty_cache()
