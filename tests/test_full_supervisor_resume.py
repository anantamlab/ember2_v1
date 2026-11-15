import importlib.util
import json
import os
from pathlib import Path
import pytest

from ember_v1.full_evidence import sha256, write_once


def test_continuation_phase_reads_verified_output_without_new_process_or_budget_charge(tmp_path, monkeypatch):
    path=Path(__file__).parents[1]/'scripts/full_view_supervisor.py'
    spec=importlib.util.spec_from_file_location('supervisor_resume',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    obj=module.Supervisor.__new__(module.Supervisor)
    obj.repo=tmp_path/'repo';obj.repo.mkdir();obj.ev=tmp_path/'evidence';obj.ev.mkdir()
    obj.py=module.sys.executable;obj.head='new';obj.cpu=9.;obj.gpu=3.;obj.base_env=os.environ.copy()
    write_once(obj.ev/'config/plan.json',{'frozen':True})
    (obj.ev/'logs').mkdir();(obj.ev/'logs/done.log').write_text('complete')
    write_once(obj.ev/'output.json',{'answer':7})
    command=[obj.py,'-c','print("must not execute")']
    record={'status':'PASS','exit_code':0,'source_head':'old','category':'cpu','command':command,
        'seconds':3.,'plan_sha256':sha256(obj.ev/'config/plan.json'),'log_sha256':sha256(obj.ev/'logs/done.log')}
    write_once(obj.ev/'attempts/done.started.json',record)
    write_once(obj.ev/'attempts/done.result.json',record)
    obj.amendment={'run_name':'run-continuation-01','completed_phases':{'done':{
        'source_head':'old','result_sha256':sha256(obj.ev/'attempts/done.result.json')}}}
    monkeypatch.setattr(module.subprocess,'check_output',lambda args,**kw:obj.head+'\n' if 'rev-parse' in args else '')
    def launch_forbidden(*args,**kwargs):
        raise AssertionError('completed attempt relaunched')
    monkeypatch.setattr(module.subprocess,'Popen',launch_forbidden)
    value=obj.phase('done',command[1:],'cpu',lambda:json.loads((obj.ev/'output.json').read_text()))
    assert value=={'answer':7}
    assert obj.cpu==9. and obj.gpu==3.


def test_changed_runtime_supplement_stops_before_reuse_or_launch(tmp_path, monkeypatch):
    path=Path(__file__).parents[1]/'scripts/full_view_supervisor.py'
    spec=importlib.util.spec_from_file_location('supervisor_changed_amendment',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    obj=module.Supervisor.__new__(module.Supervisor)
    obj.repo=tmp_path;obj.ev=tmp_path;obj.head='head'
    (tmp_path/'logs').mkdir()
    obj.py=module.sys.executable;obj.cpu=obj.gpu=0.;obj.base_env=os.environ.copy()
    write_once(tmp_path/'config/plan.json',{'frozen':True})
    obj.amendment_path=tmp_path/'amendment.json'
    obj.amendment_path.write_text('{}');obj.amendment_sha256=sha256(obj.amendment_path)
    obj.amendment={'completed_phases':{},'run_name':'run-continuation-01'}
    obj.amendment_path.write_text('{"changed":true}')
    monkeypatch.setattr(module.subprocess,'check_output',lambda args,**kw:obj.head+'\n' if 'rev-parse' in args else '')
    with pytest.raises(ValueError,match='amendment hash'):
        obj.phase('fresh',['-c','pass'],'cpu',lambda:None)
