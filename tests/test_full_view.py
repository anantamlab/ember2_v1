import copy

import pytest

from ember_v1.full_encoding import encode_view, feature_size, view_features, view_row
from ember_v1.full_view import FullEvacuationEnv, FullObserver, validate_view
from ember_v1.human_env import HumanRewardConfig
from ember_v1.sim import Simulator
from sim_helpers import scenario


def fixture():
    return scenario(['..E.','.#..','...F'],{'A':(0,0),'B':(1,2)},interval=5,max_steps=6,enter_fire='blocked')


def test_full_view_retains_complete_public_state_and_marks_each_person():
    obs=Simulator(fixture()).observe();views=FullObserver().observe(obs)
    assert set(views)=={'A','B'}
    for pid,v in views.items():
        validate_view(v)
        assert v['observation']==obs and v['person_id']==pid
        assert f'SELF={pid}' in encode_view(v)
        assert len(view_features(v))==feature_size()
        assert view_row(v,'stay')['gold']=={'you':'stay'}
    changed=copy.deepcopy(obs);changed['map']['obstacles'].append([2,0])
    assert encode_view(FullObserver().observe(changed)['A'])!=encode_view(views['A'])
    changed['seed']=42
    with pytest.raises(ValueError):FullObserver().observe(changed)


def test_neighbours_last_action_outcome_and_blocked_reward():
    env=FullEvacuationEnv(fixture(),HumanRewardConfig(eta=0,blocked=-.05))
    views,_=env.reset();assert views['A']['moves']['north']=='wall'
    after,rewards,done,_=env.step({'A':'north','B':'stay'})
    assert not done and rewards['A']==-.05
    assert after['A']['last_action']=='north' and after['A']['last_outcome']=='blocked_boundary'
    assert encode_view(after['A'])!=encode_view(views['A'])
    corrupted=copy.deepcopy(after['A']);corrupted['moves']['north']='floor'
    with pytest.raises(ValueError):validate_view(corrupted)
    corrupted=copy.deepcopy(after['A']);corrupted['last_action']=None
    with pytest.raises(ValueError):validate_view(corrupted)


def test_encoding_keeps_burning_exit_and_inactive_people():
    s=scenario(['B..','...'],{'A':(1,1)},max_steps=3)
    view=FullObserver().observe(Simulator(s).observe())['A'];text=encode_view(view)
    assert 'F..' in text and 'E1=0,0' in text
    v=copy.deepcopy(view);v['observation']['people'].append({'id':'C','position':[0,2],'status':'casualty'})
    assert 'C=0,2:casualty' in encode_view(v)
    assert view_features(v)!=view_features(view)
