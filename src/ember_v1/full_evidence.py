"""Write-once full-view records, topology isolation and paired family inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def validate_plan_roots(path, plan):
    """Bind real frozen plans to their durable evidence and attempt-marker roots."""
    actual = Path(path).resolve().parent.parent
    # Small CPU protocol fixtures predate source/root declarations. Production plans have source_head.
    required = 'source_head' in plan or 'evidence_root' in plan or 'attempt_markers_root' in plan
    if required and (not plan.get('evidence_root') or not plan.get('attempt_markers_root')
            or Path(plan['evidence_root']).resolve() != actual
            or Path(plan['attempt_markers_root']).resolve() != actual / 'attempts'):
        raise ValueError('frozen evidence/attempt root mismatch')
    return actual


def report_checkpoint(arm, freeze, requested):
    key = 'dagger_checkpoint' if arm == 'dagger_raw' else 'checkpoint'
    expected = Path(freeze[key]).resolve()
    if requested is not None and Path(requested).resolve() != expected:
        raise ValueError('report checkpoint differs from frozen arm')
    return expected


def verify_original(pinned):
    """Validate the pinned original artifact again at use, before model loading."""
    if 'manifest_sha256' not in pinned:
        # Legacy mock-only fixtures have no pinned artifact manifest.
        return
    root = Path(pinned['root']); manifest = root / 'ARTIFACT.json'
    if sha256(manifest) != pinned['manifest_sha256']:
        raise ValueError('original manifest hash mismatch')
    table = json.loads(manifest.read_text())['files']
    if len(table) != pinned['files_checked']:
        raise ValueError('original manifest count mismatch')
    verify_table(root, table)


def select_ppo_seed(seeds, masked_parent):
    eligible = [s for s in seeds if s['guard_passed']
                and s['selected_utility'] > masked_parent['utility']
                and s['selected']['casualty_rate'] <= masked_parent['casualty_rate'] + .02]
    return max(eligible, key=lambda s: (s['selected_utility'], -s['config']['seed'])) if eligible else None


def sha256(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write_once(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')


def file_table(root):
    root=Path(root)
    return [{'path':str(p.relative_to(root)),'size':p.stat().st_size,'sha256':sha256(p)}
            for p in sorted(root.rglob('*')) if p.is_file()]


def verify_table(root,table):
    root=Path(root).resolve()
    for row in table:
        p=(root/row['path']).resolve()
        if not p.is_relative_to(root) or not p.is_file() or p.stat().st_size!=row['size'] or sha256(p)!=row['sha256']:
            raise ValueError('evidence hash mismatch: '+row['path'])


def assert_disjoint(panels):
    names=list(panels)
    for i,a in enumerate(names):
        if len(panels[a])!=len(set(panels[a])):
            raise ValueError('duplicate topology within panel '+a)
        for b in names[i+1:]:
            if set(panels[a]) & set(panels[b]):
                raise ValueError('topology overlap: '+a+' / '+b)
    return {'status':'PASS','overlap':0,'panels':{k:len(v) for k,v in panels.items()}}


def paired_intervals(left,right,*,resamples=5000,seed=0):
    """Resample whole matched families, then recompute population-weighted rates."""
    a={r['scenario_id']:r for r in left};b={r['scenario_id']:r for r in right}
    if set(a)!=set(b) or len(a)!=len(left) or len(b)!=len(right) or not a:
        raise ValueError('paired report must contain unique identical families')
    ids=sorted(a)
    if any(a[k]['people']!=b[k]['people'] for k in ids):
        raise ValueError('paired population mismatch')
    keys=('people','evacuated','casualties','unresolved')
    arr_a=np.array([[a[k][v] for v in keys] for k in ids],dtype=np.float64)
    arr_b=np.array([[b[k][v] for v in keys] for k in ids],dtype=np.float64)
    def metrics(t):
        n=t[...,0]
        return np.stack([t[...,1]/n,t[...,2]/n,t[...,3]/n,(t[...,1]-2*t[...,2]-.75*t[...,3])/n],axis=-1)
    indices=np.random.default_rng(seed).integers(0,len(ids),size=(resamples,len(ids)))
    differences=metrics(arr_a[indices].sum(1))-metrics(arr_b[indices].sum(1))
    point=metrics(arr_a.sum(0))-metrics(arr_b.sum(0));interval=np.quantile(differences,[.025,.975],axis=0)
    return {'families':len(ids),'resamples':resamples,'seed':seed,'method':'paired-family percentile, weighted totals',
            'metrics':{key:{'delta':float(point[i]),'low':float(interval[0,i]),'high':float(interval[1,i])}
                       for i,key in enumerate(('evacuated_rate','casualty_rate','unresolved_rate','utility'))}}
