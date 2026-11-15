"""Fresh-review regressions: real loader, mocked audit entrypoint, real completion gate."""
import importlib.util
import json
import math
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import pytest
from ember_v1.full_evidence import file_table,sha256,write_once
from ember_v1.full_recovery import load_recovery


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).parents[1]/path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def fixture(tmp_path):
    ev=tmp_path/'evidence';head='a'*40
    for cp in ('s2','s3'):
        (ev/cp).mkdir(parents=True);(ev/cp/'weights').write_text(cp)
    write_once(ev/'s3/full-fit.json',{'historical':'failed'})
    panel=ev/'config/panel.jsonl';panel.parent.mkdir()
    row={'state':'whole CPU fixture','gold':{'p':'east'}}
    panel.write_text((json.dumps(row)+'\n')*4096)
    plan={'source_head':head,'evidence_root':str(ev),'attempt_markers_root':str(ev/'attempts'),
          'authorization':{'source':'owner prompt','text':'owner-authorized fixture'}}
    write_once(ev/'config/plan.json',plan)
    for name in ('stop','budget'):write_once(ev/f'{name}.json',{'fixture':True})
    archive=ev/'source.tar'
    with tarfile.open(archive,'w',format=tarfile.PAX_FORMAT,pax_headers={'comment':head}) as tar:
        tar.add(ev/'stop.json',arcname='fixture.json')
    (ev/'logs').mkdir();(ev/'logs/dagger-r3-fit-01.log').write_text('preserved')
    command=['python','fit','--out',str(ev/'s3')]
    record={'status':'FAIL','exit_code':0,'command':command,'source_head':'old','category':'gpu',
            'plan_sha256':sha256(ev/'config/plan.json'),'log_sha256':sha256(ev/'logs/dagger-r3-fit-01.log')}
    write_once(ev/'attempts/dagger-r3-fit-01.started.json',record)
    write_once(ev/'attempts/dagger-r3-fit-01.result.json',record)
    def ref(p):return {'path':str(p.relative_to(ev)),'sha256':sha256(p)}
    value={'kind':'ember2-saved-fit-diagnostic-recovery@1','status':'frozen','source_head':head,
        'parent_plan_sha256':sha256(ev/'config/plan.json'),'authorization':{'source':'owner prompt',
            'original':plan['authorization'],'continuation':'Then finish ember2'},'run_name':'run-recovery-01',
        'failed_phase':'dagger-r3-fit-01','stop':ref(ev/'stop.json'),'budget_ledger':ref(ev/'budget.json'),
        'source_archive':ref(archive),'audit':{'panel':ref(panel),'rows':4096,
            'parent':{'path':'s2','files':file_table(ev/'s2')},'child':{'path':'s3','files':file_table(ev/'s3')}},
        'completed_phases':{'dagger-r3-fit-01':{'source_head':'old','status':'FAIL','output':str(ev/'s3'),
            'result_sha256':sha256(ev/'attempts/dagger-r3-fit-01.result.json'),
            'start_sha256':sha256(ev/'attempts/dagger-r3-fit-01.started.json'),'output_files':file_table(ev/'s3')}}}
    path=ev/'config/recovery.json';write_once(path,value);path.with_suffix('.sha256').write_text(sha256(path)+'\n')
    return ev,path,head


def test_recovery_loader_rejects_other_controller_root_before_work(tmp_path):
    ev,path,head=fixture(tmp_path)
    assert load_recovery(path,head,evidence=ev)['status']=='frozen'
    with pytest.raises(ValueError,match='controller.*root'):load_recovery(path,head,evidence=tmp_path/'other')
    assert not (ev/'supervisor').exists()
    (ev/'s2/weights').write_text('tampered')
    with pytest.raises(ValueError,match='hash'):load_recovery(path,head,evidence=ev)


@pytest.mark.parametrize('mode',('pass','fail','changed_weights','changed_panel','changed_checkpoint'))
def test_saved_weight_entrypoint_one_shot_with_mock_adapter(tmp_path,monkeypatch,mode):
    ev,path,head=fixture(tmp_path);entry=module('audit_entry','experiments/full_saved_weight_audit.py')
    calls=[]
    class Mock:
        def __init__(self,checkpoint):self.checkpoint=checkpoint;self.model=SimpleNamespace(eval=lambda:None);self.digests=0
        def scores(self,row):
            loss=1. if self.checkpoint.name=='s2' else (2. if mode=='fail' else .5)
            prob=math.exp(-loss)
            return {'p':{'east':math.log(prob),**{a:math.log((1-prob)/4) for a in ('north','south','stay','west')}}}
        def weights_digest(self):
            self.digests+=1
            return str(self.digests) if mode=='changed_weights' else self.checkpoint.name
    def load(kind,checkpoint):
        calls.append(checkpoint.name)
        if mode=='changed_checkpoint':(checkpoint/'weights').write_text('tamper')
        return Mock(checkpoint)
    monkeypatch.setattr(entry.fd,'load_adapter',load)
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda:None)))
    out=ev/'audit.json';monkeypatch.setattr(sys,'argv',['audit','--recovery',str(path),'--out',str(out)])
    if mode=='changed_panel':(ev/'config/panel.jsonl').write_text('{}\n')
    if mode.startswith('changed_'):
        with pytest.raises((AssertionError,ValueError)):entry.main()
        assert not out.exists()
    else:
        entry.main();result=json.loads(out.read_text())
        assert result['status']==('PASS' if mode=='pass' else 'FAIL')
        assert result['training_steps']==result['select_episodes']==result['report_episodes']==0
        assert calls==['s2','s3']
        with pytest.raises(FileExistsError):entry.main()
    assert json.loads((ev/'attempts/dagger-r3-fit-01.result.json').read_text())['status']=='FAIL'


def test_real_repaired_fit_rejects_failure_and_wrong_bindings(tmp_path,monkeypatch):
    ev,path,head=fixture(tmp_path)
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    mod=module('real_repaired','scripts/full_view_recovery.py');obj=mod.RecoverySupervisor.__new__(mod.RecoverySupervisor)
    obj.ev=ev;obj.head=head;obj.recovery_hash=sha256(path);obj.recovery=json.loads(path.read_text())
    result={'status':'PASS','historical_fit_status':'FAIL_PRESERVED','recovery_sha256':obj.recovery_hash,
        'panel_sha256':obj.recovery['audit']['panel']['sha256'],'rows':4096,'training_steps':0,
        'collection_episodes':0,'select_episodes':0,'report_episodes':0,'parent_mean_ce':1.,'child_mean_ce':.5}
    audit=ev/'saved-s3-train-audit-01.json';write_once(audit,result)
    (ev/'logs/saved-s3-train-audit-01.log').write_text('CPU mock')
    write_once(ev/'attempts/saved-s3-train-audit-01.result.json',{'status':'PASS','exit_code':0,'source_head':head,
        'log_sha256':sha256(ev/'logs/saved-s3-train-audit-01.log')})
    assert obj.repaired_fit()=={'historical':'failed'}
    for changed in ({'status':'FAIL'},{'recovery_sha256':'wrong'},{'panel_sha256':'wrong'},{'child_mean_ce':2.}):
        audit.write_text(json.dumps({**result,**changed}))
        with pytest.raises((AssertionError,ValueError)):obj.repaired_fit()
    audit.write_text(json.dumps({**result,'status':'FAIL'}))
    monkeypatch.setattr(obj,'phase',lambda *args,**kwargs:None)
    monkeypatch.setattr(obj,'run',lambda:pytest.fail('advanced after failed audit'))
    obj.recovery_path=path
    with pytest.raises(AssertionError):obj.recover()


def test_original_report_metadata_names_actual_pinned_checkpoint(tmp_path,monkeypatch):
    from test_full_report_protocol import setup
    _,run=setup(tmp_path,monkeypatch);out=tmp_path/'report.json';run('original_v1_raw',out)
    assert json.loads(out.read_text())['student']==str(tmp_path/'original/final')
