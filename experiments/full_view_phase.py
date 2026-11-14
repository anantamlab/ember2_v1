"""Single write-once full-view phase, launched by the external budget supervisor."""
from __future__ import annotations

import argparse
import copy
import inspect
import json
from pathlib import Path

from ember_v1 import full_dagger as fd
from ember_v1.full_encoding import view_row
from ember_v1.full_evidence import assert_disjoint,file_table,sha256,verify_table,write_once
from ember_v1.full_view import FullObserver
from ember_v1.rl_online_scenarios import build_scenarios
from ember_v1.sim import Simulator
from ember_v1.student_train import config_from_protocol,load_protocol
from ember_v1.teacher.registry import teacher_class


def read(path):return json.loads(Path(path).read_text())
def scenarios(namespace,split,count,steps):return [s for s,_ in build_scenarios(namespace,split,count,steps)]


def check_plan(path):
    path=Path(path);plan=read(path)
    assert plan['status']=='frozen' and plan['id']=='ember2-v1-full-view@1'
    assert plan['authorization']['source']=='owner prompt'
    assert sha256(path)==path.with_suffix('.sha256').read_text().split()[0]
    for name,digest in plan['config_hashes'].items():assert sha256(path.parent/name)==digest
    return plan


def preflight(a):
    from ember_v1.student_run import hf_tokenizer
    from ember_v1.token_budget import clef_counts
    cfg=config_from_protocol(load_protocol());tokenizer=Path(a.original)/'final/tokenizer.json'
    assert sha256(tokenizer)==cfg['tokenizer_json_sha256']
    tok=hf_tokenizer(tokenizer);lengths=[];n=0
    for s in scenarios(a.namespace,'train',256,48):
        obs=Simulator(s).observe()
        # Dense public-fire and late-step variants cover larger rows without consulting outcomes.
        variants=[obs,copy.deepcopy(obs)]
        walls={tuple(p) for p in obs['map']['obstacles']}
        variants[1]['burning']=[[r,c] for r in range(s.height) for c in range(s.width) if (r,c) not in walls]
        variants[1]['step']=47
        for variant in variants:
            for view in FullObserver().observe(variant).values():
                row=view_row(view,'stay');count=clef_counts(tok,row['state'],row['questions'],cfg['max_len'])
                if count['status']!='OK':raise ValueError('full-view token preflight requires a compact encoding before freeze')
                lengths.append(count['total']);n+=1
    manifest=read(Path(a.original)/'ARTIFACT.json')
    assert sha256(Path(a.original)/'ARTIFACT.json')=='fb7573a93f9630b8486eeaa6d3e2bfce58ad1e2e8e7db74ff3f39651949d0952'
    verify_table(a.original,manifest['files'])
    result={'status':'PASS','namespace':a.namespace,'split':'train','families':256,'rows_checked':n,
            'max_tokens':max(lengths),'context_limit':cfg['max_len'],'encoding':'full-view@1 lossless layered ASCII',
            'variants':['initial public state','dense public fire, step47'],'truncated':0,
            'original_v1':{'root':str(a.original),'revision':'1fc405f723718e308c8a22b137fee4a025086fde',
                           'manifest_sha256':sha256(Path(a.original)/'ARTIFACT.json'),'files_checked':len(manifest['files'])}}
    write_once(a.out,result)


def teacher_reference(a):
    panel=scenarios(a.namespace,'train',8,24);arms={}
    for name in ('T1','T2','T3','T4'):
        arms[name]=fd.evaluate(panel,fd.teacher_policy(name,0))
        print(json.dumps({'teacher':name,'summary':arms[name]['summary']}),flush=True)
    chosen=min(arms,key=lambda k:(-arms[k]['summary']['utility'],arms[k]['summary']['casualty_rate'],k))
    cls=teacher_class(chosen);instance=cls(seed=0) if 'seed' in inspect.signature(cls).parameters else cls()
    write_once(a.out,{'namespace':a.namespace,'split':'train','count':8,'max_steps':24,'arms':arms,
                      'chosen_teacher':chosen,'identity':instance.identity.to_dict(),'seed':instance.seed,
                      'rule':'highest utility, lower casualty rate, lexicographic teacher name',
                      'topologies':[fd.topology(s) for s in panel]})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['preflight','teacher-reference','reference','collect','fit','eval','ppo'])
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--namespace');ap.add_argument('--plan',type=Path)
    ap.add_argument('--original',type=Path);ap.add_argument('--round',type=int,default=0);ap.add_argument('--student',type=Path)
    ap.add_argument('--data',type=Path,nargs='*',default=[]);ap.add_argument('--split',choices=['select','report'],default='select')
    ap.add_argument('--arm');ap.add_argument('--config',type=Path);ap.add_argument('--actor',choices=['small','llm'])
    a=ap.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    if a.phase=='preflight':preflight(a);return
    if a.phase=='teacher-reference':teacher_reference(a);return
    plan=check_plan(a.plan);ev=a.plan.parent.parent;panels=read(a.plan.parent/'family-panels.json')
    teacher=plan['teacher']['name'];seed=plan['teacher']['seed'] or 0
    if a.phase=='reference':
        select=plan['panels']['select'];ss=scenarios(select['namespace'],'select',select['count'],plan['max_steps'])
        result={name:fd.evaluate(ss,policy) for name,policy in [('teacher',fd.teacher_policy(teacher,seed)),('random_legal',fd.random_policy(0))]}
        write_once(a.out,result)
    elif a.phase=='collect':
        cfg=plan['dagger']['rounds'][a.round];ss=scenarios(cfg['namespace'],'train',cfg['count'],plan['max_steps'])
        held=set(panels['select'])|set(panels['report'])|set(panels['teacher-reference'])|set(panels['token-preflight'])
        assert not held&{fd.topology(s) for s in ss}
        adapter=fd.load_adapter('clef',a.student) if a.student else None
        rows,stats=fd.collect(ss,teacher=teacher,student=fd.adapter_policy(adapter,True) if adapter else None,
                              beta=cfg['beta'],seed=0,round_index=a.round)
        manifest=fd.write_dataset(a.out,rows,{'split':'train','view_version':'full-view@1','teacher':teacher,
            'teacher_identity':plan['teacher'],'namespace':cfg['namespace'],'count':cfg['count'],'round':a.round,
            'beta':cfg['beta'],'stats':stats,'topology_hashes':sorted({fd.topology(s) for s in ss})})
        print(json.dumps({'rows':manifest['rows'],'rows_sha256':manifest['rows_sha256'],'stats':stats}))
    elif a.phase=='fit':
        print(json.dumps(fd.train_dataset(a.data,a.out,'clef',plan['dagger']['fit'],0,a.student),default=str))
    elif a.phase=='eval':
        panel=plan['panels'][a.split];ss=scenarios(panel['namespace'],a.split,panel['count'],plan['max_steps'])
        assert not {fd.topology(s) for s in ss}&{h for d in a.data for h in fd.read_dataset(d)[1]['topology_hashes']}
        if a.split=='report':
            assert a.arm in plan['report_arms'];freeze=read(ev/'candidate-freeze.json')
            verify_table(freeze['checkpoint'],freeze['files']);verify_table(freeze['dagger_checkpoint'],freeze['dagger_files'])
            marker=ev/'attempts'/('report-'+a.arm+'.started.json')
            write_once(marker,{'arm':a.arm,'panel':panel,'candidate_freeze_sha256':sha256(ev/'candidate-freeze.json')})
        if a.arm=='teacher':policy=fd.teacher_policy(teacher,seed)
        elif a.arm=='random_legal':policy=fd.random_policy(0)
        elif a.arm in ('original_v1_raw','original_v1_masked'):
            from ember_v1.policy_arms import ArmPolicy
            from ember_v1.student_train import ClefLoraAdapter
            cfg=config_from_protocol(load_protocol());adapter=ClefLoraAdapter(cfg,checkpoint=Path(plan['original_v1']['root'])/'final')
            original=ArmPolicy(adapter,cfg,a.arm,'A_raw' if a.arm.endswith('_raw') else 'B_static_mask')
            policy=lambda views:original.act(next(iter(views.values()))['observation'])
        else:
            policy=fd.adapter_policy(fd.load_adapter('clef',a.student),masked=a.arm=='candidate_masked')
        result=fd.evaluate(ss,policy);result.update(arm=a.arm,split=a.split,panel=panel,student=str(a.student or ''))
        write_once(a.out,result);print(json.dumps(result['summary']))
    else:
        import torch
        from ember_v1.full_ppo import FullLLMActor,FullSmallActor,make_critic,train
        cfg=read(a.config);assert cfg['namespace'].startswith('ember2-full-view-')
        torch.manual_seed(cfg['seed']);torch.set_num_threads(4)
        actor=(FullSmallActor(cfg['hidden']) if a.actor=='small' else FullLLMActor({**config_from_protocol(load_protocol()),**cfg['llm'],'seed':cfg['seed']},a.student))
        select=plan['panels']['select'];ss=scenarios(select['namespace'],'select',select['count'],plan['max_steps'])
        held=set(panels['select'])|set(panels['report'])|set(panels['teacher-reference'])|set(panels['token-preflight'])
        print(json.dumps(train(actor,make_critic(actor.device),cfg,a.out,ss,held)))

if __name__=='__main__':main()
