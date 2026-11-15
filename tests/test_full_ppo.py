import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from ember_v1 import full_ppo as fp
from ember_v1.full_view import FullObserver
from ember_v1.human_ppo import actor_policy, optimize
from ember_v1.human_view import admissible
from ember_v1.sim import Simulator
from sim_helpers import scenario

torch.set_num_threads(4)


def fixture():return scenario(['...E','....','F...'],{'A':(0,0),'B':(1,1)},interval=6,max_steps=5,enter_fire='blocked')


def config():
    cfg=json.loads((Path(__file__).parents[1]/'data/rl/human-pov-ppo.draft.json').read_text())
    cfg.update(smoke_test=True,teacher='T1',teacher_seed=0,hidden=16,minibatch=8,accumulation=2,ppo_epochs=1,
               updates=2,eval_every=1,select_count=1,episodes_per_update=1,max_steps=5)
    return cfg


def test_small_actor_full_view_mask_save_and_reload(tmp_path):
    torch.manual_seed(1);actor=fp.FullSmallActor(16)
    views=FullObserver().observe(Simulator(fixture()).observe())
    actions=actor_policy(actor)(views)
    assert all(actions[p] in admissible(v) for p,v in views.items())
    actor.save(tmp_path/'actor');again=fp.FullSmallActor.load(tmp_path/'actor')
    assert torch.equal(actor.logits(list(views.values())),again.logits(list(views.values())))


def test_checkpoint_restores_exact_next_update_actor_critic_adam_and_rng(tmp_path):
    torch.manual_seed(2);random.seed(3);np.random.seed(4)
    cfg=config();actor=fp.FullSmallActor(16);critic=fp.make_critic();gen=torch.Generator().manual_seed(5)
    optimizer=fp.make_optimizer(actor,critic,cfg)
    batch=fp.collect_full([fixture()],actor,critic,cfg,gen)
    optimize(actor,critic,optimizer,batch,cfg,.5,gen)
    fp.save_checkpoint(tmp_path/'checkpoint',actor,critic,optimizer,gen,cfg,1,{'update':0},[{'update':1}])
    next_batch=fp.collect_full([fixture()],actor,critic,cfg,gen)
    metrics=optimize(actor,critic,optimizer,next_batch,cfg,.4,gen)
    expected_random=(random.random(),np.random.random(),torch.rand(3))
    restored=fp.FullSmallActor.load(tmp_path/'checkpoint/actor');critic2=fp.make_critic();gen2=torch.Generator()
    optimizer2=fp.make_optimizer(restored,critic2,cfg)
    state=fp.restore_checkpoint(tmp_path/'checkpoint',critic2,optimizer2,gen2,cfg)
    replay=fp.collect_full([fixture()],restored,critic2,cfg,gen2)
    assert replay['actions']==next_batch['actions'] and replay['old']==next_batch['old']
    metrics2=optimize(restored,critic2,optimizer2,replay,cfg,.4,gen2)
    assert metrics==metrics2 and state['update']==1
    assert all(torch.equal(a,b) for a,b in zip(actor.parameters(),restored.parameters()))
    assert all(torch.equal(a,b) for a,b in zip(critic.parameters(),critic2.parameters()))
    for a,b in zip(optimizer.state.values(),optimizer2.state.values()):
        assert a.keys()==b.keys()
        assert all(torch.equal(a[k],b[k]) if torch.is_tensor(a[k]) else a[k]==b[k] for k in a)
    assert random.random()==expected_random[0] and np.random.random()==expected_random[1]
    assert torch.equal(torch.rand(3),expected_random[2])
    with pytest.raises(ValueError,match='configuration'):fp.restore_checkpoint(tmp_path/'checkpoint',critic2,optimizer2,gen2,{**cfg,'seed':99})
    p=tmp_path/'checkpoint/actor/actor.pt';p.write_bytes(p.read_bytes()+b'x')
    with pytest.raises(ValueError,match='hash mismatch'):fp.restore_checkpoint(tmp_path/'checkpoint',critic2,optimizer2,gen2,cfg)


def test_train_smoke_writes_complete_trainer_states(tmp_path):
    torch.manual_seed(1);cfg=config();actor=fp.FullSmallActor(16);critic=fp.make_critic()
    result=fp.train(actor,critic,cfg,tmp_path/'run',[fixture()],set())
    assert len(result['checks'])==2
    state=torch.load(tmp_path/'run/update-0002/trainer.pt',weights_only=False)
    assert state['critic'] and state['optimizer']['state'] and state['update']==2
