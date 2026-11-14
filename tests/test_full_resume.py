import importlib
import json
from pathlib import Path

import pytest

from ember_v1.full_evidence import sha256, write_once


def setup_phase(tmp_path, **changes):
    (tmp_path / 'attempts').mkdir()
    (tmp_path / 'logs').mkdir()
    (tmp_path / 'logs/done.log').write_text('completed\n')
    (tmp_path / 'output.json').write_text('{"answer": 7}')
    record = {'status': 'PASS', 'exit_code': 0, 'source_head': 'old', 'category': 'gpu',
              'command': ['python', 'phase.py'], 'plan_sha256': 'plan',
              'log_sha256': sha256(tmp_path / 'logs/done.log'), 'seconds': 3.}
    record.update(changes)
    write_once(tmp_path / 'attempts/done.started.json', {})
    write_once(tmp_path / 'attempts/done.result.json', record)
    return {'run_name': 'run-continuation-01', 'completed_phases': {
        'done': {'result_sha256': sha256(tmp_path / 'attempts/done.result.json'), 'source_head': 'old'}}}


def reuse(tmp_path, amendment, verify):
    module = importlib.import_module('ember_v1.full_resume')
    return module.reuse_completed(tmp_path, 'done', ['python', 'phase.py'], 'gpu', 'plan', amendment, verify)


def test_resume_reads_completed_output_without_relaunching_or_charging_it(tmp_path):
    amendment = setup_phase(tmp_path)
    reused, value = reuse(tmp_path, amendment, lambda: json.loads((tmp_path / 'output.json').read_text()))
    assert reused and value == {'answer': 7}
    assert json.loads((tmp_path / 'attempts/done.result.json').read_text())['seconds'] == 3.
    receipt = tmp_path / 'supervisor/run-continuation-01/reuse/done.json'
    assert json.loads(receipt.read_text())['original_result_sha256'] == amendment['completed_phases']['done']['result_sha256']


@pytest.mark.parametrize('change', [
    {'status': 'FAIL'}, {'exit_code': 1}, {'source_head': 'other'},
    {'command': ['python', 'different.py']}, {'plan_sha256': 'different'}, {'category': 'cpu'},
])
def test_resume_rejects_failed_or_misbound_attempt_before_output_verification(tmp_path, change):
    amendment = setup_phase(tmp_path, **change)
    with pytest.raises(ValueError):
        reuse(tmp_path, amendment, lambda: pytest.fail('unsafe result accepted'))


def test_resume_rejects_tampered_result_or_log(tmp_path):
    amendment = setup_phase(tmp_path)
    (tmp_path / 'logs/done.log').write_text('tampered')
    with pytest.raises(ValueError, match='hash'):
        reuse(tmp_path, amendment, lambda: pytest.fail('tampered log accepted'))


def test_undeclared_started_attempt_is_never_relaunched(tmp_path):
    module = importlib.import_module('ember_v1.full_resume')
    setup_phase(tmp_path)
    with pytest.raises(FileExistsError):
        reuse(tmp_path, {'run_name': 'run-continuation-01', 'completed_phases': {}}, lambda: None)
    assert module.reuse_completed(tmp_path, 'fresh', ['python'], 'gpu', 'plan',
                                 {'completed_phases': {}}, lambda: None) == (False, None)


@pytest.mark.parametrize('status,want', [(0, 0), (768, 3), (9, -9)])
def test_linux_process_snapshot_retains_real_exit_status_and_identity(tmp_path, status, want):
    module = importlib.import_module('ember_v1.full_resume')
    proc = tmp_path / '42'
    proc.mkdir()
    fields = ['0'] * 50
    fields[0], fields[1], fields[2], fields[19], fields[49] = 'Z', '10', '42', '12345', str(status)
    (proc / 'stat').write_text('42 (python with spaces) ' + ' '.join(fields))
    (proc / 'cmdline').write_bytes(b'python\x00phase.py\x00')
    result = module.process_snapshot(42, proc_root=tmp_path)
    assert result == {'pid': 42, 'state': 'Z', 'parent_pid': 10, 'group_pid': 42,
                      'start_ticks': 12345, 'exit_code': want, 'command': ['python', 'phase.py']}
