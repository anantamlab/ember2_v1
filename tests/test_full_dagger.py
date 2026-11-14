import copy
import json

import pytest

from ember_v1.full_dagger import (adapter_policy,aggregate,collect,evaluate,random_policy,read_dataset,
                                  teacher_policy,train_dataset,write_dataset)
from ember_v1.full_view import FullObserver
from ember_v1.human_view import admissible
from ember_v1.sim import Simulator
from ember_v1.student_train import MockDecisionAdapter
from sim_helpers import scenario


def fixture():return scenario(['...E','....','F...'],{'A':(0,0),'B':(1,1)},interval=6,max_steps=5,enter_fire='blocked')


def test_teacher_is_a_function_of_full_public_view_and_random_is_legal():
    obs=Simulator(fixture()).observe();views=FullObserver().observe(obs)
    from ember_v1.teacher.t1 import T1CongestionAware
    assert teacher_policy('T1',0)(views)==T1CongestionAware().act(obs)
    for seed in range(5):
        moves=random_policy(seed)(views)
        assert all(moves[p] in admissible(v) for p,v in views.items())


def test_student_state_agreement_excludes_teacher_executed_rows():
    rows,stats=collect([fixture()],student=random_policy(1),beta=.5)
    acted=[r for r in rows if r['actor']=='student']
    assert stats['student_rows']==len(acted)>0
    assert stats['student_agreement']==sum(r['label']==r['executed'] for r in acted)/len(acted)
    _,only_teacher=collect([fixture()]);assert only_teacher['student_agreement'] is None
    with pytest.raises(ValueError):collect([fixture()],beta=.5)


def test_mock_fit_aggregate_write_once_and_hash_boundary(tmp_path):
    rows,stats=collect([fixture()]);data=tmp_path/'data'
    write_dataset(data,rows,{'split':'train','view_version':'full-view@1','teacher':'T1','round':0,'stats':stats})
    loaded,meta=read_dataset(data);assert loaded==rows
    union,info=aggregate([loaded,loaded]);assert info['duplicates_dropped']==len(rows)
    cfg={'epochs':3,'grad_accum_draws':1,'lr':.5,'head_lr':.5,'warmup_ratio':0.}
    result=train_dataset([data],tmp_path/'model','mock',cfg)
    assert result['report']['last_loss']<result['report']['first_loss']
    model=MockDecisionAdapter.load(tmp_path/'model')
    evaluated=evaluate([fixture()],adapter_policy(model,True))
    assert evaluated['summary']['episodes']==1
    assert evaluated['summary']['movements']['blocked_wall']==0
    with pytest.raises(FileExistsError):write_dataset(data,rows,meta)
    (data/'rows.jsonl').write_text((data/'rows.jsonl').read_text()+'\n')
    with pytest.raises(ValueError,match='hash mismatch'):read_dataset(data)
