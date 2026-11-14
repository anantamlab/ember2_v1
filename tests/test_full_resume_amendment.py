import importlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from ember_v1.full_evidence import file_table, sha256

OLD, NEW = 'a'*40, 'b'*40


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    plan = root/'config/plan.json'
    authorization = {'source':'owner prompt','text':'bounded private local work'}
    dump(plan, {'id':'ember2-v1-full-view@1','status':'frozen','source_head':OLD,
                'authorization':authorization})
    output = root/'s0'
    output.mkdir()
    (output/'weights.bin').write_bytes(b'frozen-actor')
    (root/'logs').mkdir()
    (root/'logs/done.log').write_text('done')
    record = {'status':'PASS','exit_code':0,'phase':'done','source_head':OLD,
              'category':'gpu','seconds':3.,'plan_sha256':sha256(plan),
              'command':['python','phase.py','--out',str(output)],
              'log_sha256':sha256(root/'logs/done.log')}
    dump(root/'attempts/done.started.json', record)
    dump(root/'attempts/done.result.json', record)
    result_hash = sha256(root/'attempts/done.result.json')
    dump(root/'supervisor/run.stopped.json', {'status':'STOPPED','type':'CONTROLLED_SOURCE_TRANSITION',
        'source_head':OLD,'phase':'done','handoff':'boundary-handoff-02'})
    dump(root/'supervisor/boundary-handoff-02/result.json', {'status':'PASS','phase_result_sha256':result_hash})
    archive = root/'source-runtime-01.tar'
    with tarfile.open(archive, 'w', format=tarfile.PAX_FORMAT, pax_headers={'comment':NEW}) as tar:
        content=b'code'; info=tarfile.TarInfo('source.py'); info.size=len(content)
        tar.addfile(info, io.BytesIO(content))
    outputs=[{**item,'path':'s0/'+item['path']} for item in file_table(output)]
    amendment = {'kind':'ember2-runtime-source-amendment@1','status':'frozen',
        'plan_id':'ember2-v1-full-view@1','amendment_id':'runtime-source-01',
        'parent_plan_sha256':sha256(plan),'old_source_head':OLD,'new_source_head':NEW,
        'authorization':authorization,'run_name':'run-continuation-01','additional_cpu_seconds':17.,
        'controller_stop':{'path':'supervisor/run.stopped.json','sha256':sha256(root/'supervisor/run.stopped.json')},
        'source_archive':{'path':'source-runtime-01.tar','sha256':sha256(archive)},
        'completed_phases':{'done':{'source_head':OLD,'result_sha256':result_hash,'output_files':outputs}}}
    return amendment


def load(root, amendment, changes=('scripts/full_view_supervisor.py',)):
    path=root/'config/runtime-source-01.json'
    dump(path, amendment)
    path.with_suffix('.sha256').write_text(sha256(path)+'\n')
    module=importlib.import_module('ember_v1.full_resume')
    return module.load_amendment(path, root, NEW, changes)


def test_frozen_source_supplement_binds_all_prior_outputs_and_budget(tmp_path):
    amendment=fixture(tmp_path)
    result=load(tmp_path, amendment)
    assert result['run_name']=='run-continuation-01'
    assert result['additional_cpu_seconds']==17.
    assert result['completed_phases']['done']['source_head']==OLD


@pytest.mark.parametrize('change', ['plan','source','authorization','name','negative_cpu','nonfinite_cpu',
                                   'panel_override','failed_phase','unfinished_phase','changed_output','stop_type','start_binding'])
def test_supplement_cannot_erase_failures_override_plan_or_replay_started_work(tmp_path, change):
    amendment=fixture(tmp_path)
    if change=='plan':amendment['parent_plan_sha256']='bad'
    elif change=='source':amendment['new_source_head']=OLD
    elif change=='authorization':amendment['authorization']={'source':'other'}
    elif change=='name':amendment['run_name']='../run'
    elif change=='negative_cpu':amendment['additional_cpu_seconds']=-1
    elif change=='nonfinite_cpu':amendment['additional_cpu_seconds']=float('nan')
    elif change=='panel_override':amendment['panels']={'report':'different'}
    elif change=='failed_phase':
        path=tmp_path/'attempts/done.result.json'; record=json.loads(path.read_text())
        record['status']='FAIL'; dump(path,record)
        amendment['completed_phases']['done']['result_sha256']=sha256(path)
    elif change=='unfinished_phase':dump(tmp_path/'attempts/unfinished.started.json',{})
    elif change=='changed_output':(tmp_path/'s0/weights.bin').write_bytes(b'tampered')
    elif change=='stop_type':
        path=tmp_path/'supervisor/run.stopped.json'; record=json.loads(path.read_text())
        record['type']='ACTUAL_PHASE_FAILURE'; dump(path,record)
        amendment['controller_stop']['sha256']=sha256(path)
    elif change=='start_binding':
        path=tmp_path/'attempts/done.started.json';record=json.loads(path.read_text())
        record['command']=['python','different.py'];dump(path,record)
    with pytest.raises(ValueError):load(tmp_path,amendment)


def test_source_supplement_cannot_change_actor_or_teacher_code(tmp_path):
    with pytest.raises(ValueError):
        load(tmp_path,fixture(tmp_path),('src/ember_v1/full_ppo.py',))


def test_reuse_also_checks_frozen_checkpoint_bytes(tmp_path):
    amendment=fixture(tmp_path)
    (tmp_path/'s0/weights.bin').write_bytes(b'tampered')
    module=importlib.import_module('ember_v1.full_resume')
    with pytest.raises(ValueError):
        module.reuse_completed(tmp_path,'done',['python','phase.py','--out',str(tmp_path/'s0')],
            'gpu',amendment['parent_plan_sha256'],amendment,lambda:pytest.fail('changed actor accepted'))


def test_entrypoint_change_is_limited_to_original_policy_api(tmp_path):
    amendment=fixture(tmp_path)
    path=tmp_path/'config/runtime-source-01.json'
    dump(path,amendment);path.with_suffix('.sha256').write_text(sha256(path)+'\n')
    module=importlib.import_module('ember_v1.full_resume')
    before="policy=lambda views:original.act(next(iter(views.values()))['observation'])\n"
    after="policy=lambda views:original.decide(next(iter(views.values()))['observation']).actions\n"
    result=module.load_amendment(path,tmp_path,NEW,['experiments/full_view_phase.py'],phase_texts=(before,after))
    assert result['new_source_head']==NEW
    with pytest.raises(ValueError):
        module.load_amendment(path,tmp_path,NEW,['experiments/full_view_phase.py'],
                              phase_texts=(before,after+'learning_rate=99\n'))
