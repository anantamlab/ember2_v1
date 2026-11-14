"""CPU-only Linux lifecycle check; no scenario generation, model or GPU use."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def main():
    from ember_v1.full_resume import HeldChild, process_snapshot
    with tempfile.TemporaryDirectory(prefix='ember2-handoff-cpu-') as folder:
        root = Path(folder)
        parent_code = (
            'import subprocess,sys,pathlib; '
            'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(1)"],start_new_session=True); '
            'pathlib.Path(sys.argv[1]).write_text(str(p.pid)); p.wait(); '
            'pathlib.Path(sys.argv[2]).write_text("advanced")'
        )
        command = [sys.executable, '-c', parent_code, str(root/'pid'), str(root/'advanced')]
        parent = subprocess.Popen(command)
        try:
            deadline = time.monotonic() + 5
            while not (root/'pid').exists():
                assert time.monotonic() < deadline
                time.sleep(.01)
            child = int((root/'pid').read_text())
            expected = [sys.executable, '-c', 'import time; time.sleep(1)']
            with HeldChild(parent.pid, child, command, expected) as held:
                assert held.wait(5, lambda: None) == 0
                assert not (root/'advanced').exists()
                assert process_snapshot(parent.pid)['state'] in ('T', 't')
                held.kill_parent()
            assert parent.wait(timeout=5) == -9
            assert not (root/'advanced').exists()
            deadline = time.monotonic() + 5
            while (Path('/proc')/str(child)).exists():
                assert time.monotonic() < deadline, 'finished child was not reaped'
                time.sleep(.01)
        finally:
            if parent.poll() is None:
                parent.kill()
                parent.wait()
    print(json.dumps({'status':'PASS','parent_held':True,'child_completed_normally':True,
                      'no_next_phase':True,'temporary_processes_reaped':True,'gpu_jobs':0}))


if __name__ == '__main__':
    main()
