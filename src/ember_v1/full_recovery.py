"""Immutable saved-weight diagnostics and completed-phase reuse after the fit stop."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import tarfile

from .contract import ACTIONS
from .full_evidence import sha256, verify_table, validate_plan_roots, file_table


def canonical(row):
    return json.dumps(row,sort_keys=True,separators=(',',':'),ensure_ascii=True)


def audit_rows(rows,count):
    if not 0 < count <= len(rows):
        raise ValueError('audit panel count outside TRAIN union')
    return sorted(rows,key=lambda row:(hashlib.sha256(canonical(row).encode()).hexdigest(),canonical(row)))[:count]


def mean_cross_entropy(adapter,rows,progress=None):
    if not rows:raise ValueError('empty matched TRAIN panel')
    losses=[]
    for i,row in enumerate(rows):
        scores=adapter.scores(row)
        if set(scores)!=set(row['gold']):raise ValueError('audit question mismatch')
        per_question=[]
        for q,gold in row['gold'].items():
            values=scores[q]
            if set(values)!=set(ACTIONS) or not all(math.isfinite(v) for v in values.values()):
                raise ValueError('audit requires five finite unmasked scores')
            largest=max(values.values())
            normalizer=largest+math.log(math.fsum(math.exp(v-largest) for v in values.values()))
            loss=normalizer-values[gold]
            if not math.isfinite(loss):raise ValueError('non-finite audit loss')
            per_question.append(loss)
        losses.append(math.fsum(per_question)/len(per_question))
        if progress and ((i+1)%128==0 or i+1==len(rows)):progress(i+1,losses[-1])
    return math.fsum(losses)/len(losses)


def compare_losses(parent,child):
    if not all(math.isfinite(v) for v in (parent,child)) or not child < parent:
        raise ValueError('strict matched TRAIN loss decrease failed')
    return {'status':'PASS','parent_mean_ce':parent,'child_mean_ce':child,'delta':child-parent,
            'criterion':'finite child_mean_ce < parent_mean_ce; identical frozen TRAIN rows; no tolerance'}


def validate_reused(ev,name,command,category,plan_hash,declaration,*,failed_fit=None):
    ev=Path(ev);result=ev/'attempts'/f'{name}.result.json';start=ev/'attempts'/f'{name}.started.json'
    if sha256(result)!=declaration['result_sha256'] or sha256(start)!=declaration['start_sha256']:
        raise ValueError('recovery start/result hash mismatch')
    record=json.loads(result.read_text());begin=json.loads(start.read_text())
    expected={'command':list(map(str,command)),'category':category,'plan_sha256':plan_hash,
              'source_head':declaration['source_head'],'exit_code':0,'status':declaration['status']}
    if any(record.get(k)!=v for k,v in expected.items()):raise ValueError('recovery phase binding mismatch')
    if any(begin.get(k)!=record.get(k) for k in ('command','category','source_head','plan_sha256')):
        raise ValueError('recovery start binding mismatch')
    if record['status']!='PASS' and not (name==failed_fit and record['status']=='FAIL'):
        raise ValueError('failed phase cannot be reused without its separate diagnostic repair')
    if sha256(ev/'logs'/f'{name}.log')!=record['log_sha256']:raise ValueError('recovery log hash mismatch')
    output=Path(declaration['output']).resolve()
    if not output.is_relative_to(ev.resolve()):raise ValueError('recovery output outside evidence')
    if output.is_dir() and file_table(output)!=declaration['output_files']:
        raise ValueError('recovery complete output hash table mismatch')
    verify_table(output if output.is_dir() else output.parent,declaration['output_files'])
    return record


def load_recovery(path,head):
    path=Path(path).resolve();ev=path.parent.parent
    if sha256(path)!=path.with_suffix('.sha256').read_text().split()[0]:raise ValueError('recovery hash mismatch')
    value=json.loads(path.read_text());plan_path=ev/'config/plan.json';plan=json.loads(plan_path.read_text())
    validate_plan_roots(plan_path,plan)
    if (value['kind']!='ember2-saved-fit-diagnostic-recovery@1' or value['status']!='frozen'
            or value['source_head']!=head or value['parent_plan_sha256']!=sha256(plan_path)
            or value['authorization']['original']!=plan['authorization']
            or value['authorization']['source']!='owner prompt'
            or value['authorization']['continuation']!='Then finish ember2'):
        raise ValueError('recovery source/plan/authorization mismatch')
    if value['run_name']!='run-recovery-01' or value['failed_phase']!='dagger-r3-fit-01':
        raise ValueError('unknown recovery scope')
    for ref in (value['stop'],value['budget_ledger'],value['source_archive'],value['audit']['panel']):
        p=(ev/ref['path']).resolve()
        if not p.is_relative_to(ev) or sha256(p)!=ref['sha256']:raise ValueError('recovery reference hash mismatch')
    with tarfile.open(ev/value['source_archive']['path']) as tar:
        if tar.pax_headers.get('comment')!=head:raise ValueError('recovery archive commit mismatch')
    if list((ev/'attempts').glob('report-*.started.json')) or (ev/'candidate-freeze.json').exists():
        raise ValueError('recovery cannot consume or repeat a spent report')
    actual={p.name.removesuffix('.result.json') for p in (ev/'attempts').glob('*.result.json')}
    declared=set(value['completed_phases'])
    # This loader runs before the new audit or continuation starts.
    if actual!=declared:raise ValueError('recovery phase declarations incomplete')
    starts={p.name.removesuffix('.started.json') for p in (ev/'attempts').glob('*.started.json')}
    if starts!=actual:raise ValueError('unfinished attempt prevents recovery')
    for name,dec in value['completed_phases'].items():
        record=json.loads((ev/'attempts'/f'{name}.result.json').read_text())
        validate_reused(ev,name,record['command'],record['category'],record['plan_sha256'],dec,
                        failed_fit=value['failed_phase'])
    for key in ('parent','child'):
        ref=value['audit'][key];verify_table(ev/ref['path'],ref['files'])
    return value
