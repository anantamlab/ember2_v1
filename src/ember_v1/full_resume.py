"""Strict completed-phase reuse and Linux process identity for a controlled handoff."""
from __future__ import annotations

import json
import ctypes
import os
from pathlib import Path
import select
import signal
import time

from .full_evidence import sha256, write_once


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
                except ProcessLookupError:
                    pass
            self.kill_parent()
        finally:
            for handle in (self.child_fd, self.parent_fd):
                if handle is not None:
                    os.close(handle)
