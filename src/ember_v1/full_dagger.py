"""Full-public-view DAgger; reuse the tested adapter and cross-entropy optimizer."""
from __future__ import annotations

import hashlib
import json
import math
import inspect
from collections import Counter
from pathlib import Path

from .contract import ACTIONS
from .full_encoding import QUESTION_ID, view_row
from .full_view import FullEvacuationEnv, validate_view
from .human_dagger import _uniform, _write_json, fit, load_adapter, topology, write_dataset
from .human_env import EpisodeStats, HumanRewardConfig, summarize
from .human_view import admissible
from .teacher.registry import teacher_class


def teacher_policy(name, seed=0):
    cls = teacher_class(name)
    teacher = cls(seed=seed) if 'seed' in inspect.signature(cls).parameters else cls()
    def act(views, routes=None):
        if not views:
            return {}
        observation = next(iter(views.values()))['observation']
        return teacher.act(observation)
    return act


def random_policy(seed=0):
    def act(views, routes=None):
        return {p: sorted(admissible(v))[int(_uniform(seed,p,json.dumps(v,sort_keys=True))*len(admissible(v)))]
                for p,v in sorted(views.items())}
    return act


def adapter_policy(adapter, masked=True):
    def act(views, routes=None):
        result = {}
        for p,v in sorted(views.items()):
            scores = adapter.scores(view_row(v,ACTIONS[0]))[QUESTION_ID]
            if set(scores) != set(ACTIONS) or not all(math.isfinite(x) for x in scores.values()):
                raise ValueError('full-view student must score five finite actions')
            result[p] = min(admissible(v) if masked else ACTIONS,key=lambda a:(-scores[a],a))
        return result
    return act


def run_episode(scenario, policy, *, seed=None):
    env=FullEvacuationEnv(scenario,HumanRewardConfig(eta=0))
    views,info=env.reset(seed=seed);stats=EpisodeStats(scenario.scenario_id);done=False
    while not done:
        views,_,done,info=env.step(policy(views))
        stats.add(info['record'])
    return stats.result(info)


def evaluate(scenarios, policy):
    episodes=[run_episode(s,policy) for s in scenarios]
    return {'summary':summarize(episodes),'episodes':episodes}


def collect(scenarios, *, teacher='T1', student=None, beta=1., seed=0, round_index=0):
    if not 0 <= beta <= 1 or beta < 1 and student is None:
        raise ValueError('invalid beta/student')
    rows=[];episodes=[]
    for scenario in scenarios:
        env=FullEvacuationEnv(scenario,HumanRewardConfig(eta=0));views,info=env.reset()
        teacher_act=teacher_policy(teacher,seed);stats=EpisodeStats(scenario.scenario_id);done=False
        while not done:
            gold=teacher_act(views);student_actions=student(views) if beta<1 else {};actions={}
            for p,v in sorted(views.items()):
                by_teacher=beta>=1 or _uniform(seed,scenario.scenario_id,v['observation']['step'],p)<beta
                actions[p]=gold[p] if by_teacher else student_actions[p]
                rows.append({'scenario_id':scenario.scenario_id,'round':round_index,'view':v,'label':gold[p],
                             'executed':actions[p],'actor':'teacher' if by_teacher else 'student'})
            views,_,done,info=env.step(actions);stats.add(info['record'])
        episodes.append(stats.result(info))
    student_rows=[r for r in rows if r['actor']=='student']
    return rows,{'rows':len(rows),'labels':dict(Counter(r['label'] for r in rows)),
                 'student_rows':len(student_rows),'student_matches':sum(r['label']==r['executed'] for r in student_rows),
                 'student_agreement':sum(r['label']==r['executed'] for r in student_rows)/len(student_rows) if student_rows else None,
                 'episodes':summarize(episodes)}


def read_dataset(path):
    path=Path(path);meta=json.loads((path/'manifest.json').read_text())
    if meta.get('split')!='train' or meta.get('view_version')!='full-view@1':
        raise ValueError('not a full-view TRAIN dataset')
    if hashlib.sha256((path/'rows.jsonl').read_bytes()).hexdigest()!=meta['rows_sha256']:
        raise ValueError('dataset hash mismatch')
    rows=[json.loads(line) for line in (path/'rows.jsonl').read_text().splitlines()]
    if len(rows)!=meta['rows']:
        raise ValueError('dataset row-count mismatch')
    for r in rows:
        validate_view(r['view'])
        if r['label'] not in ACTIONS or r['executed'] not in ACTIONS:
            raise ValueError('unsupported dataset action')
    return rows,meta


def aggregate(datasets):
    known={};result=[];total=0
    for rows in datasets:
        for r in rows:
            total+=1;row=view_row(r['view'],r['label']);labels=known.setdefault(row['state'],set())
            if r['label'] not in labels:
                labels.add(r['label']);result.append(row)
    return result,{'input_rows':total,'rows':len(result),'duplicates_dropped':total-len(result),
                  'conflicting_states':sum(len(v)>1 for v in known.values())}


def train_dataset(paths,out,kind,cfg,seed=0,parent=None):
    loaded=[read_dataset(p) for p in paths]
    teacher_ids={m['teacher'] for _,m in loaded}
    if len(teacher_ids)!=1:
        raise ValueError('mixed full-view teachers')
    rows,agg=aggregate([r for r,_ in loaded]);adapter=load_adapter(kind,parent,train=True)
    def progress(record):
        if record['optimizer_steps'] == 1 or record['optimizer_steps'] % 100 == 0:
            with Path(str(out)+'.progress.jsonl').open('a') as f:
                f.write(json.dumps(record,allow_nan=False)+'\n')
    report=fit(adapter,rows,cfg,seed=seed,progress=progress)
    out=Path(out)
    if kind=='clef':
        from unsloth.models.decision import _save_clef_adapter
        out.mkdir(parents=True,exist_ok=False)
        _save_clef_adapter(adapter.model,str(out),adapter.tok,exact=True)
        adapter.torch.save(adapter.opt.state_dict(),out/'optimizer.pt')
    else:
        adapter.save(out)
    # All rows remain in the datasets. This bounded diagnostic is a TRAIN probe only.
    probe=loaded[-1][0][:2000];policy=adapter_policy(adapter,False)
    choices=[policy({'p':r['view']})['p'] for r in probe]
    result={'report':report,'aggregate':agg,'datasets':[{'path':str(p),'rows_sha256':m['rows_sha256']} for p,(_,m) in zip(paths,loaded)],
            'train_probe':{'rows':len(probe),'agreement':sum(a==r['label'] for a,r in zip(choices,probe))/len(probe),
                           'legal_rate':sum(a in admissible(r['view']) for a,r in zip(choices,probe))/len(probe)},
            'weights_digest':adapter.weights_digest(),'config':cfg,'seed':seed}
    _write_json(out/'full-fit.json',result)
    return result
