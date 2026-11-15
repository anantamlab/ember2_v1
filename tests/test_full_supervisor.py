import importlib.util
import os
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('full_supervisor',Path(__file__).parents[1]/'scripts/full_view_supervisor.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def supervisor(tmp_path,monkeypatch):
    obj=module.Supervisor.__new__(module.Supervisor)
    obj.repo=tmp_path/'repo';obj.repo.mkdir();obj.ev=tmp_path/'evidence';obj.ev.mkdir()
    for name in ('attempts','logs'):(obj.ev/name).mkdir()
    obj.py=module.sys.executable;obj.head='a'*40;obj.cpu=obj.gpu=0;obj.base_env=os.environ.copy()
    monkeypatch.setattr(module.subprocess,'check_output',lambda args,**kw:obj.head+'\n' if 'rev-parse' in args else '')
    return obj


def test_phase_verifies_after_completion_and_preserves_failed_attempt(tmp_path,monkeypatch):
    obj=supervisor(tmp_path,monkeypatch)
    called=[]
    def verify():
        assert 'done' in (obj.ev/'logs/good.log').read_text();called.append(True);return {'status':'PASS'}
    assert obj.phase('good',['-c','print("done")'],'cpu',verify)['status']=='PASS'
    assert called==[True] and obj.cpu>0
    with pytest.raises(RuntimeError,match='exited 3'):
        obj.phase('bad',['-c','raise SystemExit(3)'],'cpu',lambda:pytest.fail('verifier ran after failure'))
    assert module.read(obj.ev/'attempts/bad.result.json')['status']=='FAIL'
    with pytest.raises(FileExistsError):obj.phase('bad',['-c','print("retry")'],'cpu',lambda:None)
    assert module.read(obj.ev/'attempts/bad.result.json')['exit_code']==3


def test_budget_stops_before_start_and_completion_verifier_can_fail(tmp_path,monkeypatch):
    obj=supervisor(tmp_path,monkeypatch);obj.cpu=16*3600
    with pytest.raises(RuntimeError,match='budget exhausted'):obj.phase('over',['-c','pass'],'cpu',lambda:None)
    assert not (obj.ev/'attempts/over.started.json').exists()
    obj.cpu=0
    def reject():raise ValueError('hash mismatch')
    with pytest.raises(ValueError,match='hash mismatch'):obj.phase('verify-fail',['-c','pass'],'cpu',reject)
    assert module.read(obj.ev/'attempts/verify-fail.result.json')['status']=='FAIL'
