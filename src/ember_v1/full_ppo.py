"""Full-view actors and resumable PPO; shared per-person optimizer stays in human_ppo."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from .full_dagger import run_episode, teacher_policy
from .full_encoding import feature_size, view_features, view_row
from .full_view import FullEvacuationEnv
from .human_dagger import topology
from .human_env import HumanRewardConfig, summarize
from .human_ppo import (HumanLLMActor, actor_policy, check_config, collect, make_critic, optimize)
from .rl_online_scenarios import build_scenarios
from .student_train import ACTION_ORDER


class FullSmallActor(torch.nn.Module):
    def __init__(self, hidden=128):
        super().__init__();self.hidden=hidden
        self.net=torch.nn.Sequential(torch.nn.Linear(feature_size(),hidden),torch.nn.Tanh(),
                                     torch.nn.Linear(hidden,hidden),torch.nn.Tanh(),torch.nn.Linear(hidden,5))

    @property
    def device(self):return next(self.parameters()).device

    def logits(self,views):
        return self.net(torch.tensor([view_features(v) for v in views],dtype=torch.float32,device=self.device))

    def parameter_groups(self,cfg):
        return [{'params':list(self.parameters()),'lr':cfg['actor_lr'],'name':'actor'}]

    def save(self,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=False)
        torch.save(self.state_dict(),path/'actor.pt')
        (path/'actor.json').write_text(json.dumps({'kind':'full-small','hidden':self.hidden}))

    @classmethod
    def load(cls,path,device='cpu'):
        path=Path(path);actor=cls(json.loads((path/'actor.json').read_text())['hidden']).to(device)
        actor.load_state_dict(torch.load(path/'actor.pt',map_location=device,weights_only=True))
        return actor


class FullLLMActor(HumanLLMActor):
    def logits(self,views):
        items=[self.inner.ad.item(view_row(v,ACTION_ORDER[0])) for v in views]
        return torch.cat(self.inner.logits({'items':items,'lengths':[1]*len(items)}))


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def save_checkpoint(path,actor,critic,optimizer,generator,cfg,update,baseline,checks):
    path=Path(path);path.mkdir(parents=True,exist_ok=False);actor.save(path/'actor')
    state={'format':'ember2-full-ppo@1','critic':critic.state_dict(),'optimizer':optimizer.state_dict(),
           'generator':generator.get_state(),'torch_cpu_rng':torch.get_rng_state(),
           'torch_cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
           'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'config':dict(cfg),
           'config_sha256':config_hash(cfg),'update':update,'baseline':baseline,'checks':checks}
    torch.save(state,path/'trainer.pt')
    table={str(p.relative_to(path)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(path.rglob('*')) if p.is_file()}
    (path/'identity.json').write_text(json.dumps({'format':state['format'],'config_sha256':state['config_sha256'],
                                               'update':update,'files':table},sort_keys=True,indent=2)+'\n')


def restore_checkpoint(path,critic,optimizer,generator,cfg):
    """Load actor separately, then restore trainer/RNG *after* all module construction."""
    path=Path(path);identity=json.loads((path/'identity.json').read_text())
    for name,digest in identity['files'].items():
        p=(path/name).resolve()
        if not p.is_relative_to(path.resolve()) or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
            raise ValueError('PPO checkpoint hash mismatch')
    state=torch.load(path/'trainer.pt',map_location='cpu',weights_only=False)
    if state['format']!='ember2-full-ppo@1' or state['config_sha256']!=config_hash(cfg):
        raise ValueError('PPO checkpoint configuration mismatch')
    critic.load_state_dict(state['critic']);optimizer.load_state_dict(state['optimizer'])
    generator.set_state(state['generator']);random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_cpu_rng'])
    if state['torch_cuda_rng']:
        torch.cuda.set_rng_state_all(state['torch_cuda_rng'])
    return state


def collect_full(scenarios,actor,critic,cfg,generator):
    return collect(scenarios,actor,critic,cfg,None,HumanRewardConfig(**cfg['reward']),generator,
                   env_factory=FullEvacuationEnv,teacher_fn=teacher_policy(cfg['teacher'],cfg['teacher_seed']))


def make_optimizer(actor,critic,cfg):
    return torch.optim.Adam(actor.parameter_groups(cfg)+[{'params':list(critic.parameters()),'lr':cfg['critic_lr'],'name':'critic'}])


def train(actor,critic,cfg,out,select,held_topologies,*,resume=None):
    check_config(cfg);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    generator=torch.Generator().manual_seed(cfg['seed']);optimizer=make_optimizer(actor,critic,cfg)
    def selection(update):
        stats=summarize([run_episode(s,actor_policy(actor,masked=True)) for s in select])
        return {'update':update,**stats}
    checks=[];baseline=selection(0);first=1
    if resume is not None:
        state=restore_checkpoint(resume,critic,optimizer,generator,cfg)
        baseline,checks,first=state['baseline'],state['checks'],state['update']+1
    save_checkpoint(out/'update-0000',actor,critic,optimizer,generator,cfg,first-1,baseline,checks)
    for update in range(first,cfg['updates']+1):
        scenarios=[s for s,_ in build_scenarios(f"{cfg['namespace']}/u{update}",'train',cfg['episodes_per_update'],cfg['max_steps'])]
        if set(held_topologies)&{topology(s) for s in scenarios}:
            raise RuntimeError('PPO TRAIN topology overlap')
        batch=collect_full(scenarios,actor,critic,cfg,generator)
        coef=cfg['imitation_coef']*cfg['imitation_decay']**(update-1)
        metrics=optimize(actor,critic,optimizer,batch,cfg,coef,generator)
        entry={'update':update,'imitation_coef':coef,'train':batch['episodes'],**metrics}
        if update%cfg['eval_every']==0 or update==cfg['updates']:
            check=selection(update);checks.append(check);entry['selection']=check
            save_checkpoint(out/f'update-{update:04d}',actor,critic,optimizer,generator,cfg,update,baseline,checks)
        with (out/'log.jsonl').open('a') as f:f.write(json.dumps(entry,sort_keys=True)+'\n')
    # The immutable DAgger baseline is the guard. No failed guard fallback is allowed.
    eligible=[r for r in checks if r['casualty_rate']<=baseline['casualty_rate']+.02 and r['utility']>baseline['utility']]
    best=max(eligible,key=lambda r:(r['utility'],-r['update'])) if eligible else baseline
    result={'config':cfg,'baseline':baseline,'checks':checks,'selected_update':best['update'],
            'selected_utility':best['utility'],'guard_passed':bool(eligible),'selected':best}
    with (out/'summary.json').open('x') as f:json.dump(result,f,indent=2)
    return result
