"""Mock-model fixtures exercise one-shot reporting without using an experiment panel."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from ember_v1 import student_train
from ember_v1.full_evidence import file_table, sha256, write_once
from sim_helpers import scenario

ARMS=('candidate_raw','candidate_masked','dagger_raw','teacher','random_legal',
      'original_v1_raw','original_v1_masked')


class CPUScores(student_train.MockDecisionAdapter):
    def scores(self,row):
        return {p:{'east':10.,'north':4.,'south':3.,'stay':5.,'west':2.} for p in row['questions']}


def setup(tmp_path,monkeypatch):
    path=Path(__file__).parents[1]/'experiments/full_view_phase.py'
    spec=importlib.util.spec_from_file_location('full_cpu_report_protocol',path)
    phase=importlib.util.module_from_spec(spec);spec.loader.exec_module(phase)
    monkeypatch.setattr(phase,'scenarios',lambda *args:[scenario(['.#E','...'],{'A':(0,0)},max_steps=2)])
    monkeypatch.setattr(phase.fd,'load_adapter',lambda *args:CPUScores())
    monkeypatch.setattr(student_train,'ClefLoraAdapter',lambda cfg,checkpoint:CPUScores())
    checkpoint=tmp_path/'checkpoint';checkpoint.mkdir();(checkpoint/'weights.bin').write_bytes(b'mock-actor')
    table=file_table(checkpoint)
    write_once(tmp_path/'candidate-freeze.json',{'checkpoint':str(checkpoint),'files':table,
        'dagger_checkpoint':str(checkpoint),'dagger_files':table})
    write_once(tmp_path/'config/family-panels.json',{'report':['mock-report'],'select':['mock-select']})
    plan=tmp_path/'config/plan.json'
    write_once(plan,{'status':'frozen','id':'ember2-v1-full-view@1','authorization':{'source':'owner prompt'},
        'config_hashes':{'family-panels.json':sha256(tmp_path/'config/family-panels.json')},
        'teacher':{'name':'T1','seed':0},'max_steps':2,'report_arms':list(ARMS),
        'panels':{'report':{'namespace':'ember2-full-view-cpu-report-fixture','count':1}},
        'original_v1':{'root':str(tmp_path/'original')}})
    plan.with_suffix('.sha256').write_text(sha256(plan)+'\n')
    def run(arm,out):
        monkeypatch.setattr(sys,'argv',[str(path),'eval','--out',str(out),'--plan',str(plan),
                                      '--split','report','--arm',arm,'--student',str(checkpoint)])
        phase.main()
    return phase,run


@pytest.mark.parametrize('arm',ARMS)
def test_each_report_bridge_runs_once_and_refuses_a_second_output(tmp_path,monkeypatch,arm):
    phase,run=setup(tmp_path,monkeypatch)
    output=tmp_path/'result.json';run(arm,output)
    result=json.loads(output.read_text())
    assert result['arm']==arm and result['split']=='report'
    assert result['summary']['episodes']==1 and result['summary']['people']==1
    marker=tmp_path/'attempts'/('report-'+arm+'.started.json');digest=sha256(marker)
    monkeypatch.setattr(phase.fd,'evaluate',lambda *args:pytest.fail('spent arm evaluated again'))
    with pytest.raises(FileExistsError):run(arm,tmp_path/'second.json')
    assert sha256(marker)==digest and not (tmp_path/'second.json').exists()


def test_report_model_load_failure_is_preserved_and_cannot_be_retried(tmp_path,monkeypatch):
    phase,run=setup(tmp_path,monkeypatch)
    def fail(*args):raise RuntimeError('model load failure')
    monkeypatch.setattr(phase.fd,'load_adapter',fail)
    with pytest.raises(RuntimeError,match='model load failure'):run('candidate_raw',tmp_path/'failed.json')
    marker=tmp_path/'attempts/report-candidate_raw.started.json'
    assert marker.exists() and not (tmp_path/'failed.json').exists()
    monkeypatch.setattr(phase.fd,'load_adapter',lambda *args:CPUScores())
    with pytest.raises(FileExistsError):run('candidate_raw',tmp_path/'retry.json')
    assert not (tmp_path/'retry.json').exists()
