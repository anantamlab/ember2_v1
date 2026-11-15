"""Strict completed-phase reuse and Linux process identity for a controlled handoff."""
from __future__ import annotations

import json
import ctypes
import math
import os
from pathlib import Path
import select
import signal
import re
import tarfile
import time

from .full_evidence import sha256, verify_table, write_once


CONTROL_CHANGES = {
    'pyproject.toml', 'docs/FULL_VIEW_RUNBOOK.md', 'scripts/full_view_supervisor.py',
    'scripts/full_view_boundary_handoff.py', 'experiments/full_view_phase.py',
    'experiments/full_handoff_checks.py', 'src/ember_v1/full_resume.py',
    'tests/test_full_supervisor_cpu_import.py', 'tests/test_full_original_control.py',
    'tests/test_full_resume.py', 'tests/test_full_handoff.py', 'tests/test_full_handoff_fit.py',
    'tests/test_full_resume_amendment.py', 'tests/test_full_supervisor_resume.py',
    'tests/test_full_report_protocol.py',
    'tests/test_full_resume_cleanup.py',
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def load_amendment(path, evidence, head, changed_files, *, phase_texts=None):
    """Only the declared pre-report control repair may supplement the unchanged frozen plan."""
    path, evidence = Path(path).resolve(), Path(evidence).resolve()
    _require(path.is_relative_to(evidence), 'amendment outside evidence')
    _require(sha256(path)==path.with_suffix('.sha256').read_text().split()[0], 'amendment hash mismatch')
    value=json.loads(path.read_text());plan_path=evidence/'config/plan.json'
    plan=json.loads(plan_path.read_text());plan_hash=sha256(plan_path)
    keys={'kind','status','plan_id','amendment_id','parent_plan_sha256','old_source_head',
          'new_source_head','authorization','run_name','additional_cpu_seconds','controller_stop',
          'source_archive','completed_phases'}
    _require(set(value)==keys, 'amendment cannot override plan fields')
    _require(value['kind']=='ember2-runtime-source-amendment@1' and value['status']=='frozen', 'amendment not frozen')
    _require(plan['status']=='frozen' and value['plan_id']==plan['id']=='ember2-v1-full-view@1', 'plan identity mismatch')
    _require(value['amendment_id']=='runtime-source-01', 'unknown source amendment')
    _require(value['parent_plan_sha256']==plan_hash, 'parent plan hash mismatch')
    _require(value['old_source_head']==plan['source_head'] and value['new_source_head']==head
             and head!=plan['source_head'] and re.fullmatch('[0-9a-f]{40}',head), 'source identity mismatch')
    _require(value['authorization']==plan['authorization'] and value['authorization']['source']=='owner prompt',
             'owner authorization mismatch')
    _require(re.fullmatch(r'run-continuation-[0-9]+',value['run_name']), 'unsafe continuation name')
    cost=value['additional_cpu_seconds']
    _require(isinstance(cost,(int,float)) and not isinstance(cost,bool) and math.isfinite(cost)
             and 0<=cost<16*3600, 'invalid additional CPU budget')
    _require(set(changed_files)<=CONTROL_CHANGES, 'learning/protected source change is not authorized by this supplement')
    if 'experiments/full_view_phase.py' in changed_files:
        _require(phase_texts is not None, 'entrypoint source comparison required')
        old,new=phase_texts
        before="original.act(next(iter(views.values()))['observation'])"
        after="original.decide(next(iter(views.values()))['observation']).actions"
        _require(old.count(before)==1 and old.replace(before,after)==new, 'entrypoint learning behavior changed')
    _require(not list((evidence/'attempts').glob('report-*.started.json'))
             and not (evidence/'candidate-freeze.json').exists(), 'source repair must precede report freeze')
    results={p.name.removesuffix('.result.json'):p for p in (evidence/'attempts').glob('*.result.json')}
    starts={p.name.removesuffix('.started.json') for p in (evidence/'attempts').glob('*.started.json')}
    _require(starts==set(results), 'unfinished started attempt')
    frozen={}
    for name, record_path in results.items():
        record=json.loads(record_path.read_text())
        begin=json.loads((evidence/'attempts'/(name+'.started.json')).read_text())
        _require(all(begin.get(key)==record.get(key) for key in
                     ('command','category','plan_sha256','source_head')), 'phase start/result binding mismatch')
        _require(record.get('status')=='PASS' and record.get('exit_code')==0, 'failed phase cannot be erased')
        seconds=record['seconds']
        _require(isinstance(seconds,(int,float)) and math.isfinite(seconds) and seconds>=0
                 and record['category'] in ('gpu','cpu'), 'invalid prior budget record')
        _require(sha256(evidence/'logs'/(name+'.log'))==record['log_sha256'], 'prior log hash mismatch')
        if record['plan_sha256']==plan_hash:
            frozen[name]=record
        else:
            _require(record['plan_sha256'] is None and name in ('token-preflight-01','teacher-reference-01'),
                     'unrecognized prior plan')
    _require(set(value['completed_phases'])==set(frozen), 'completed phase declaration mismatch')
    for name, declaration in value['completed_phases'].items():
        record=frozen[name]
        _require(declaration['source_head']==record['source_head']==plan['source_head']
                 and declaration['result_sha256']==sha256(results[name]), 'completed source/result mismatch')
        output=Path(record['command'][record['command'].index('--out')+1]).resolve()
        _require(output.is_relative_to(evidence), 'phase output outside evidence')
        actual=({p.relative_to(evidence).as_posix() for p in output.rglob('*') if p.is_file()}
                if output.is_dir() else {output.relative_to(evidence).as_posix()})
        table=declaration['output_files']
        _require(bool(table) and {r['path'].replace('\\','/') for r in table}==actual,
                 'incomplete completed output declaration')
        verify_table(evidence,table)
    stop_ref=value['controller_stop']
    _require(stop_ref['path']=='supervisor/run.stopped.json', 'wrong original controller stop')
    _require(sha256(evidence/stop_ref['path'])==stop_ref['sha256'], 'controller stop hash mismatch')
    stop=json.loads((evidence/stop_ref['path']).read_text())
    _require(stop['status']=='STOPPED' and stop['type']=='CONTROLLED_SOURCE_TRANSITION'
             and stop['source_head']==plan['source_head'] and stop['phase'] in frozen, 'not a controlled source transition')
    _require(re.fullmatch(r'boundary-handoff-[0-9]+',stop['handoff']), 'invalid handoff identity')
    handoff=json.loads((evidence/'supervisor'/stop['handoff']/'result.json').read_text())
    _require(handoff['status']=='PASS' and handoff['phase_result_sha256']==sha256(results[stop['phase']]),
             'handoff did not verify original completion')
    archive_ref=value['source_archive'];archive=(evidence/archive_ref['path']).resolve()
    _require(archive.is_relative_to(evidence) and sha256(archive)==archive_ref['sha256'], 'source archive hash mismatch')
    with tarfile.open(archive) as tar:
        _require(tar.pax_headers.get('comment')==head, 'source archive commit mismatch')
    return value


def _pidfd_open(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid)
    libc = ctypes.CDLL(None, use_errno=True)
    call = libc.pidfd_open
    call.argtypes, call.restype = [ctypes.c_int, ctypes.c_uint], ctypes.c_int
    result = call(pid, 0)
    if result < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return result


def _pidfd_signal(handle, value):
    if hasattr(signal, 'pidfd_send_signal'):
        return signal.pidfd_send_signal(handle, value)
    libc = ctypes.CDLL(None, use_errno=True)
    call = libc.pidfd_send_signal
    call.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
    call.restype = ctypes.c_int
    if call(handle, value, None, 0) < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


def reuse_completed(evidence, name, command, category, plan_hash, amendment, verify):
    """Read a declared completed result; never turn a started attempt into a launch."""
    evidence = Path(evidence)
    result = evidence / 'attempts' / (name + '.result.json')
    started = evidence / 'attempts' / (name + '.started.json')
    declared = amendment['completed_phases'].get(name)
    if declared is None:
        if result.exists() or started.exists():
            raise FileExistsError('undeclared existing attempt: ' + name)
        return False, None
    if not result.is_file() or sha256(result) != declared['result_sha256']:
        raise ValueError('completed result hash mismatch: ' + name)
    record = json.loads(result.read_text())
    expected = {'status': 'PASS', 'exit_code': 0, 'source_head': declared['source_head'],
                'category': category, 'command': list(map(str, command)), 'plan_sha256': plan_hash}
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError('completed attempt binding/status mismatch: ' + name)
    if sha256(evidence / 'logs' / (name + '.log')) != record['log_sha256']:
        raise ValueError('completed log hash mismatch: ' + name)
    if 'output_files' in declared:
        verify_table(evidence, declared['output_files'])
    value = verify()
    write_once(evidence / 'supervisor' / amendment['run_name'] / 'reuse' / (name + '.json'), {
        'status': 'PASS_REUSED_WITHOUT_LAUNCH', 'original_result_sha256': declared['result_sha256'],
        'original_source_head': record['source_head'], 'new_charge_seconds': 0,
    })
    return True, value


def process_snapshot(pid, *, proc_root=Path('/proc')):
    """Read Linux identity and unreaped child exit status without reaping another parent's child."""
    root = Path(proc_root) / str(pid)
    fields = (root / 'stat').read_text().rsplit(') ', 1)[1].split()
    state = fields[0]
    raw = int(fields[49]) if state == 'Z' else None
    exit_code = (-(raw & 127) if raw & 127 else raw >> 8) if raw is not None else None
    command = [s.decode() for s in (root / 'cmdline').read_bytes().split(b'\0') if s]
    return {'pid': pid, 'state': state, 'parent_pid': int(fields[1]), 'group_pid': int(fields[2]),
            'start_ticks': int(fields[19]), 'exit_code': exit_code, 'command': command}


class HeldChild:
    """Hold a Linux supervisor while its existing session finishes; never relaunch the child."""
    def __init__(self, parent_pid, child_pid, parent_command, child_command):
        self.parent_pid, self.child_pid = parent_pid, child_pid
        self.parent_command, self.child_command = parent_command, child_command
        self.parent_fd = self.child_fd = None

    def __enter__(self):
        parent, child = process_snapshot(self.parent_pid), process_snapshot(self.child_pid)
        if (parent['command'] != self.parent_command or child['command'] != self.child_command
                or child['parent_pid'] != self.parent_pid or child['group_pid'] != self.child_pid
                or parent['state'] == 'Z' or child['state'] == 'Z'):
            raise ValueError('handoff process identity/session mismatch')
        self.parent_start, self.child_start = parent['start_ticks'], child['start_ticks']
        try:
            self.parent_fd = _pidfd_open(self.parent_pid)
            self.child_fd = _pidfd_open(self.child_pid)
            _pidfd_signal(self.parent_fd, signal.SIGSTOP)
            deadline = time.monotonic() + 2
            while process_snapshot(self.parent_pid)['state'] not in ('T', 't'):
                if time.monotonic() >= deadline:
                    raise TimeoutError('supervisor did not stop')
                time.sleep(.01)
            return self
        except BaseException:
            self.__exit__(*(__import__('sys').exc_info()))
            raise

    def wait(self, timeout, guard):
        deadline = time.monotonic() + timeout
        poller = select.poll()
        poller.register(self.child_fd, select.POLLIN)
        while True:
            parent, child = process_snapshot(self.parent_pid), process_snapshot(self.child_pid)
            if (parent['start_ticks'] != self.parent_start or child['start_ticks'] != self.child_start
                    or parent['state'] not in ('T', 't')):
                raise ValueError('handoff process identity/hold changed')
            guard()
            if child['state'] == 'Z':
                return child['exit_code']
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('original phase deadline reached during handoff')
            poller.poll(max(1, min(1000, int(remaining * 1000))))

    def kill_parent(self):
        if self.parent_fd is not None:
            try:
                _pidfd_signal(self.parent_fd, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def __exit__(self, error_type, error, traceback):
        try:
            if self.child_fd is not None:
                try:
                    child = process_snapshot(self.child_pid)
                    if child['state'] != 'Z':
                        os.killpg(self.child_pid, signal.SIGTERM)
                        end = time.monotonic() + 5
                        while process_snapshot(self.child_pid)['state'] != 'Z' and time.monotonic() < end:
                            time.sleep(.05)
                        if process_snapshot(self.child_pid)['state'] != 'Z':
                            os.killpg(self.child_pid, signal.SIGKILL)
                except (ProcessLookupError,FileNotFoundError):
                    pass
            self.kill_parent()
        finally:
            for handle in (self.child_fd, self.parent_fd):
                if handle is not None:
                    os.close(handle)
