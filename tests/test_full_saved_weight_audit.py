"""The completion diagnostic uses identical TRAIN examples and cannot retry a fit."""
import math
import pytest
from ember_v1.full_recovery import audit_rows, mean_cross_entropy, compare_losses, validate_reused
from ember_v1.full_evidence import sha256, write_once, file_table


def test_audit_panel_is_order_independent_and_not_an_input_truncation():
    rows=[{'state':str(i),'gold':{'self':'east'},'questions':{'self':{}}} for i in range(20)]
    assert audit_rows(rows,8)==audit_rows(list(reversed(rows)),8)
    assert len(audit_rows(rows,8))==8
    assert all(row in rows for row in audit_rows(rows,8))
    with pytest.raises(ValueError):audit_rows(rows,21)


def test_same_panel_improves_even_when_different_group_diagnostic_increases():
    rows=[{'gold':{'p':'east'},'state':s} for s in ('easy','hard')]
    class Mock:
        def __init__(self,losses):self.losses=losses
        def scores(self,row):
            p=math.exp(-self.losses[row['state']])
            return {'p':{'east':math.log(p),'north':math.log((1-p)/4),'south':math.log((1-p)/4),
                         'stay':math.log((1-p)/4),'west':math.log((1-p)/4)}}
    parent=Mock({'easy':.2,'hard':2.});child=Mock({'easy':.1,'hard':1.})
    assert 1.>.2  # mismatched last/first groups would reject this genuine matched decrease
    assert compare_losses(mean_cross_entropy(parent,rows),mean_cross_entropy(child,rows))['status']=='PASS'
    with pytest.raises(ValueError):compare_losses(1.,1.)
    with pytest.raises(ValueError):compare_losses(1.,float('nan'))
    class Bad:
        def scores(self,row):return {'p':{'east':float('inf')}}
    with pytest.raises(ValueError):mean_cross_entropy(Bad(),rows)


def test_reuse_binds_failed_fit_without_relabelling_or_calling_verifier(tmp_path):
    (tmp_path/'logs').mkdir();(tmp_path/'logs/fit.log').write_text('saved')
    output=tmp_path/'s3';output.mkdir();write_once(output/'full-fit.json',{'failed':True})
    record={'status':'FAIL','exit_code':0,'category':'gpu','source_head':'old','plan_sha256':'plan',
            'command':['python','fit','--out',str(output)],'log_sha256':sha256(tmp_path/'logs/fit.log')}
    write_once(tmp_path/'attempts/fit.started.json',{k:record[k] for k in ('command','category','source_head','plan_sha256')})
    write_once(tmp_path/'attempts/fit.result.json',record)
    declaration={'result_sha256':sha256(tmp_path/'attempts/fit.result.json'),
                 'start_sha256':sha256(tmp_path/'attempts/fit.started.json'),'output_files':file_table(output),
                 'output':str(output),'source_head':'old','status':'FAIL'}
    args=(tmp_path,'fit',record['command'],'gpu','plan',declaration)
    with pytest.raises(ValueError,match='failed'):validate_reused(*args)
    assert validate_reused(*args,failed_fit='fit')==record
    assert sha256(tmp_path/'attempts/fit.result.json')==declaration['result_sha256']
    (output/'full-fit.json').write_text('{}')
    with pytest.raises(ValueError,match='hash'):validate_reused(*args,failed_fit='fit')


def test_recovery_controller_never_spawns_declared_or_started_phase(tmp_path,monkeypatch):
    import importlib.util
    import sys
    scripts=__import__('pathlib').Path(__file__).parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location('recovery_cpu',scripts/'full_view_recovery.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    obj=module.RecoverySupervisor.__new__(module.RecoverySupervisor)
    obj.ev=tmp_path;obj.py=sys.executable;obj.recovery_path=tmp_path/'recovery.json'
    write_once(obj.recovery_path,{'immutable':True});obj.recovery_hash=sha256(obj.recovery_path)
    obj.recovery={'completed_phases':{'fit':{'result_sha256':'hash'}},'failed_phase':'fit','run_name':'recovery'}
    write_once(tmp_path/'attempts/fit.result.json',{'plan_sha256':'parent','status':'FAIL'})
    calls=[]
    monkeypatch.setattr(module,'validate_reused',lambda *a,**kw:calls.append('hash-check'))
    monkeypatch.setattr(obj,'repaired_fit',lambda:{'saved':'weights'})
    write_once(tmp_path/'saved-s3-train-audit-01.json',{'status':'PASS'})
    monkeypatch.setattr(module.Supervisor,'phase',lambda *a,**kw:pytest.fail('restarted phase'))
    assert obj.phase('fit',['fit'],'gpu',lambda:pytest.fail('old failed verifier repeated'))=={'saved':'weights'}
    assert calls==['hash-check']
    assert module.read(tmp_path/'attempts/fit.result.json')['status']=='FAIL'
    write_once(tmp_path/'attempts/other.started.json',{'spent':True})
    with pytest.raises(FileExistsError):obj.phase('other',[],'cpu',lambda:None)
