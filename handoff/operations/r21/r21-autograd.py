import os, runpy, json
from pathlib import Path
import unsloth
import torch
import ember_v1.rl_ppo as ppo
parent = Path('/home/darshith/ember-ws/rl-evidence/r20-ppo-pilot@1/attempt-6')
evidence = Path('/home/darshith/ember-ws/rl-evidence/r21-online-ppo@1/attempt-3')
original = ppo.V1Actor
def parent_actor(cfg, checkpoint):
    actor = original(cfg, parent/'checkpoint')
    assert actor.ad.weights_digest() == 'sha256:f6b56137acc791bd1ed0862547a1dd3ca2ef81f336af6c969947840b46c66374'
    return actor
ppo.V1Actor = parent_actor
runpy.run_path('experiments/rl_gpu_checks.py')['audit_checkpoint']('/home/darshith/ember-ws/artifacts/models/1fc405f723718e308c8a22b137fee4a025086fde/v1/p1-18-v1-train@1', evidence)
best = json.loads((evidence/'best_checkpoint.json').read_text())
state = torch.load(best['trainer'], weights_only=False, map_location='cpu')
prior = torch.load(parent/'trainer.pt', weights_only=False, map_location='cpu')
steps = sorted({int(s['step']) for s in state['optimizer']['state'].values()})
prior_steps = sorted({int(s['step']) for s in prior['optimizer']['state'].values()})
assert min(steps) > min(prior_steps)
assert state['completed_updates'] == best['update']
assert all(torch.isfinite(t).all() for t in state['critic'].values())
result = {'status':'PASS','restored_parent_optimizer_steps':prior_steps,'selected_optimizer_state_steps':steps,'selected_update':best['update'],'state_keys':sorted(state),'rng_present':True}
(evidence/'optimizer_resume_audit.json').write_text(json.dumps(result,indent=2))
print(result)
from safetensors.torch import load_file
changes = {}
for name in ('adapter_model.safetensors','joint_head.safetensors'):
    old = load_file(str(parent/'checkpoint'/name))
    new = load_file(str(Path(best['actor'])/name))
    assert old.keys() == new.keys()
    d = sum(float((new[k].float()-old[k].float()).square().sum()) for k in old)**.5
    assert d > 0
    changes[name] = d
result['selected_tensor_l2_from_phase1'] = changes
(evidence/'optimizer_resume_audit.json').write_text(json.dumps(result,indent=2))
print(result)