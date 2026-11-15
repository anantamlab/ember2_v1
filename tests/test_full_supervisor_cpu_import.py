"""CPU completion verification must not require the Linux-only GPU lock dependency."""
import json
from pathlib import Path
import runpy

import pytest


@pytest.mark.parametrize("status,accepted", [("PASS", True), ("FAIL", False)])
def test_cpu_completion_verifier_works_without_loading_a_gpu(status, accepted, tmp_path):
    namespace = runpy.run_path(str(Path(__file__).parents[1] / "scripts/full_view_supervisor.py"),
                               run_name="ember2_supervisor_cpu_import")
    record = tmp_path / "completion.json"
    record.write_text(json.dumps({"status": status, "n": 7}))
    verify = namespace["Supervisor"].require_pass
    if accepted:
        assert verify(record) == {"status": "PASS", "n": 7}
    else:
        with pytest.raises(AssertionError):
            verify(record)
