from pathlib import Path
from collections import Counter
import json,math,statistics,sys
from ember_v1.rl_cf_evidence import paired_statistics
p=Path(sys.argv[1])
def load(n): return json.loads((p/n).read_text())
def rows(n): return [json.loads(l) for l in (p/n).open()]
h=load('training_history.json')
s=load('summary.json')
a=load('independent_counterfactual_audit.json')
best=load('best_checkpoint.json')
panel={x:[r for r in rows('episodes.jsonl') if r['label']==x] for x in ('v1','v1-mask','v1-coordinate','phase2')}
panel.update({x:[r for r in rows(x+'/episodes.jsonl') if r['label']==x] for x in ('ppo','cf')})
teachers=rows('teacher_episodes.jsonl')
panel.update({x:[r for r in teachers if r['label']==x and r['status']=='COMPLETE'] for x in ('T0','T1','T2','T3','T4')})
train={}
for arm,d in h.items():
 us=d['updates']
 applied=[m for u in us for m in u['minibatches'] if 'gradient_norms' in m]
 assert len(applied)==s['optimizer_steps'][arm]
 assert all(set(m['gradient_norms'])=={'lora','head','critic'} and all(math.isfinite(g) and g>0 for g in m['gradient_norms'].values()) for m in applied)
 assert max(u['max_probability_replay_error'] for u in us)==0
 assert sum('counterfactual' in m for m in applied)==(s['auxiliary_optimizer_steps'] if arm=='cf' else 0)
 stage={}
 for st in sorted({u['stage'] for u in us}):
  ulist=[u for u in us if u['stage']==st]
  mm=[m for u in ulist for m in u['minibatches']]
  stage[st]={'rounds':len(ulist),'kl_early_stops':sum(u['kl_early_stop'] for u in ulist),
    **{k:statistics.mean(m[k] for m in mm) for k in ('joint_kl','clip_fraction','person_entropy')}}
 train[arm]={'stages':stage,'gradient_checks':len(applied),'likelihood_replay_max':0,
 'collection_steps_per_second':s['training'][arm]['steps']/sum(u['collect_seconds'] for u in us),
 'counterfactual_seconds':sum(u['counterfactual_seconds'] for u in us),
 'selected_optimizer_steps':sum(u['optimizer_steps'] for u in us if u['update']<=best['trained_arms'][arm]['update']),
 'aux_by_proposal':dict(Counter(m['counterfactual']['chosen'] for m in applied if 'counterfactual' in m))}
 # Match verified full state grouping: no repeated-movement field exists in teacher rows.
def repeat_teacher(r):
 previous={}
 total=0
 for d,rec in zip(r['decisions'],r['records']):
  active={q['id']:tuple(q['position']) for q in d['observation']['people'] if q['status']=='active'}
  current={k:(active[k],d['actions'][k]) for k,m in rec['movements'].items() if m['outcome'].startswith('blocked_')}
  # A casualty in the same step is excluded, as in the neural episode counter.
  current={k:v for k,v in current.items() if k not in rec.get('casualties',[])}
  total+=sum(previous.get(k)==v for k,v in current.items());previous=current
 return total
movement={k:{metric:sum(r.get(metric,0) for r in v) for metric in
    ('attempted','blocked','blocked_wall','blocked_boundary','blocked_conflict','blocked_occupied','blocked_cycle','stay','person_steps','repeated_ineffective_moves','postprocess_changes')} for k,v in panel.items() if v}
for k in ('T0','T1','T2','T3','T4'):
 if panel[k]:movement[k]['repeated_ineffective_moves']=sum(map(repeat_teacher,panel[k]))
prior=p.parent/'attempt-1'
gpu={}
for d in (prior,p):
 samples=[]
 for line in (d/'gpu.jsonl').open():
  try:samples.append(tuple(map(float,json.loads(line)['sample'].split(',')[1:])))
  except ValueError:pass
 gpu[d.name]={'samples':len(samples),'mean_util_percent':statistics.mean(v[0] for v in samples),'max_util_percent':max(v[0] for v in samples),'peak_memory_mib':max(v[1] for v in samples)}
def comparisons(a,b):
 ids={r['scenario_id'] for r in panel[a]}&{r['scenario_id'] for r in panel[b]}
 return paired_statistics([r for r in panel[a] if r['scenario_id'] in ids],[r for r in panel[b] if r['scenario_id'] in ids])
studies=[json.loads(Path(x['path']).read_text()) for x in load('verified_counterfactual_dataset.json')['studies']]
assert len(studies)==a['counterfactual']['studies']
failures={k:Counter(x['teacher_diagnostics'][k]['status'] for x in studies) for k in ('T0','T1','T2','T3','T4')}
out={'train':train,'gpu':gpu,'movement':movement,'cf_minus_raw_v1':comparisons('v1','cf'),
 'cf_minus_phase2':comparisons('phase2','cf'),'teacher_proposal_status':failures,
 'total_experiment_seconds':s['elapsed_seconds']+s['prior_wall_seconds']}
Path(sys.argv[2]).write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
