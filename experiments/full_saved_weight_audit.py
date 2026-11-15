"""Optimizer-free, one-shot matched TRAIN loss audit of already saved weights."""
import argparse
import gc
import json
from pathlib import Path

from ember_v1 import full_dagger as fd
from ember_v1.full_evidence import sha256,verify_table,write_once
from ember_v1.full_recovery import compare_losses,mean_cross_entropy


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--recovery',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    recovery=json.loads(a.recovery.read_text());ev=a.recovery.parent.parent;audit=recovery['audit']
    assert sha256(a.recovery)==a.recovery.with_suffix('.sha256').read_text().split()[0]
    panel=ev/audit['panel']['path'];assert sha256(panel)==audit['panel']['sha256']
    rows=[json.loads(line) for line in panel.read_text().splitlines()];assert len(rows)==4096==audit['rows']
    results={}
    for key in ('parent','child'):
        ref=audit[key];checkpoint=ev/ref['path'];verify_table(checkpoint,ref['files'])
        adapter=fd.load_adapter('clef',checkpoint);adapter.model.eval();before=adapter.weights_digest()
        value=mean_cross_entropy(adapter,rows,lambda n,loss:print(json.dumps({'checkpoint':key,'rows':n,'last_ce':loss}),flush=True))
        assert adapter.weights_digest()==before;verify_table(checkpoint,ref['files'])
        results[key]=value
        write_once(a.out.with_name(a.out.stem+'-'+key+'.json'),{'status':'FINITE','mean_ce':value,
            'rows':len(rows),'weights_digest':before,'panel_sha256':sha256(panel),'no_optimizer':True})
        del adapter;gc.collect()
        import torch
        torch.cuda.empty_cache()
    try:result=compare_losses(results['parent'],results['child'])
    except ValueError as error:result={'status':'FAIL','error':str(error),'mean_ce':results}
    write_once(a.out,{**result,'rows':len(rows),'panel_sha256':sha256(panel),
        'recovery_sha256':sha256(a.recovery),'historical_fit_status':'FAIL_PRESERVED',
        'training_steps':0,'collection_episodes':0,'select_episodes':0,'report_episodes':0})


if __name__=='__main__':main()
