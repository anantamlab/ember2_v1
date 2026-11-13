from pathlib import Path
import gc,json,os,runpy,time
import unsloth
import torch
import ember_v1.rl_ppo as ppo
from ember_v1.student_train import config_from_protocol,load_protocol
from ember_v1.rl_run import REPO
from safetensors.torch import load_file
root=Path('/home/darshith/ember-ws/rl-evidence/r22-counterfactual-ppo@1')/f"attempt-{os.environ.get('R22_ATTEMPT','1')}"
summary=json.loads((root/'summary.json').read_text());assert summary['status']=='COMPLETE'
binding=json.loads((root/'parent_binding.json').read_text())
parent=Path(binding['checkpoint'])
original_root=Path('/home/darshith/ember-ws/artifacts/models/1fc405f723718e308c8a22b137fee4a025086fde/v1/p1-18-v1-train@1')
original_identity=json.loads((original_root/'identity.json').read_text())
cfg=config_from_protocol(load_protocol(REPO/original_identity['protocol']['path']))
torch.set_num_threads(4)  # Match the trainer before creating any actor or compiling inference.
real_actor=ppo.V1Actor
started=time.monotonic()
def parent_actor(config,checkpoint):
 actor=real_actor(config,parent/'actor')
 assert actor.ad.weights_digest()==binding['actor_digest']
 return actor
ppo.V1Actor=parent_actor
from ember_v1.rl_cf_evidence import lineage_roots
audit=runpy.run_path('experiments/rl_gpu_checks.py')['audit_checkpoint']
first=next(p for p in reversed(lineage_roots(root)) if any(
    json.loads(line)['label']=='train-0' for line in (p/'cf'/'episodes.jsonl').open()))
for arm in ('ppo','cf'):
 if first == root:
  audit(original_root,root/arm)
 else:
  view=root/'audit_inputs'/'gpu-first-rollout'/arm
  view.mkdir(parents=True,exist_ok=True)
  (view/'execution.json').write_text((first/arm/'execution.json').read_text())
  with (view/'transitions.jsonl').open('w') as dst:
   for line in (first/arm/'transitions.jsonl').open():
    if line.endswith('\n') and json.loads(line)['label']=='train-0':dst.write(line)
  audit(original_root,view)
 gc.collect();torch.cuda.empty_cache()
ppo.V1Actor=real_actor
best=json.loads((root/'best_checkpoint.json').read_text())
parent_state=torch.load(parent/'trainer.pt',map_location='cpu',weights_only=False)
parent_steps=sorted({int(s['step']) for s in parent_state['optimizer']['state'].values()})
proof={}
for arm,item in best['trained_arms'].items():
 trainer=torch.load(Path(item['checkpoint'])/'trainer.pt',map_location='cpu',weights_only=False)
 steps=sorted({int(s['step']) for s in trainer['optimizer']['state'].values()})
 assert min(steps)>min(parent_steps)
 aux=sum(u['auxiliary_optimizer_steps'] for u in trainer['updates'])
 assert aux==item['auxiliary_optimizer_steps_in_checkpoint']
 if arm=='cf':assert aux>0
 changes={}
 for name in ('adapter_model.safetensors','joint_head.safetensors'):
  old=load_file(str(parent/'actor'/name));new=load_file(str(Path(item['actor'])/name))
  assert old.keys()==new.keys()
  delta=sum(float((new[k].float()-old[k].float()).square().sum()) for k in old)**.5
  assert delta>0;changes[name]=delta
 actor=real_actor(cfg,Path(item['actor']))
 assert actor.ad.weights_digest()==item['identity']['actor_digest']
 replayed=0;max_error=0.
 for line in (root/arm/'transitions.jsonl').open():
  r=json.loads(line)
  if r['label']!=arm:continue
  with torch.no_grad():
   maps,_,lp,_,full=actor.sample(actor.prepare([r['observation']]),greedy=True)
  if maps[0]!=r['sampled_actions'] or r['sampled_actions']!=r['executed_actions']:
   difference=float((full[0].cpu()-torch.tensor(r['person_logprobs'])).abs().max())
   raise AssertionError(f"{arm} {r['scenario_id']} step {r['observation']['step']}: "
                        f"greedy replay differs; max person logprob delta={difference}; threads={torch.get_num_threads()}")
  max_error=max(max_error,abs(float(lp[0])-r['old_joint_logprob']),float((full[0].cpu()-torch.tensor(r['person_logprobs'])).abs().max()))
  replayed+=1
 assert replayed>0 and max_error<=1e-5
 proof[arm]={'optimizer_state_steps':steps,'auxiliary_steps_in_selected_checkpoint':aux,'selected_tensor_l2_from_phase2':changes,'reporting_decisions_reproduced':replayed,'saved_actor_max_probability_error':max_error,'loaded_digest':actor.ad.weights_digest()}
 del actor,trainer;gc.collect();torch.cuda.empty_cache()
result={'status':'PASS','parent_optimizer_state_steps':parent_steps,'first_rollout_evidence':str(first),
        'torch_num_threads':torch.get_num_threads(),
        'arms':proof,'seconds':time.monotonic()-started}
(root/'optimizer_reload_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
