import os

import pytest

from ember_v1 import full_resume


def test_reaped_child_does_not_interrupt_controller_retirement_or_descriptor_cleanup(monkeypatch):
    # The child may disappear from /proc immediately after its verified parent is killed.
    reader,writer=os.pipe()
    held=full_resume.HeldChild.__new__(full_resume.HeldChild)
    held.child_pid=123;held.parent_fd=reader;held.child_fd=writer
    retired=[]
    monkeypatch.setattr(held,'kill_parent',lambda:retired.append(True))
    def disappeared(*args,**kwargs):raise FileNotFoundError('child already reaped')
    monkeypatch.setattr(full_resume,'process_snapshot',disappeared)
    try:
        held.__exit__(None,None,None)
        assert retired==[True]
        for descriptor in (reader,writer):
            with pytest.raises(OSError):os.fstat(descriptor)
    finally:
        for descriptor in (reader,writer):
            try:os.close(descriptor)
            except OSError:pass
