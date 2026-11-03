"""Structured local reports for failed Modal runs (CPU only, simulated remote calls)."""
import json

import pytest

from ember_v1 import p0b, run_report
from ember_v1.p0b import result_exit_code

CLEAN = "e" * 40


def _plan():
    return p0b.gpu_run_plan("inspect", "L4", p0b.PINNED_MODEL, "bf16", "inspect", CLEAN)


class _Clock:
    def __init__(self, *ticks):
        self.ticks = list(ticks)

    def __call__(self):
        return self.ticks.pop(0)


def load_model(*args):  # same name as the remote load frame
    raise OSError("unsloth/Qwen3.5-2B does not appear to have a file named model.safetensors")


def task_inspect(*args):  # same name as the remote task frame
    raise RuntimeError("CUDA error: device-side assert triggered")


class FunctionTimeoutError(Exception):  # name of Modal's call-timeout exception
    pass


def _run(tmp_path, call):
    path = tmp_path / "results" / "inspect-L4-bf16-compact.json"
    result, elapsed = run_report.call_remote_reported(call, plan=_plan(), path=path, clock=_Clock(10.0, 52.34))
    return result, elapsed, path


@pytest.mark.parametrize("call, stage, status", [
    (lambda: load_model(), "model_load", "FAIL"),
    (lambda: task_inspect(), "remote_task", "FAIL"),
    (lambda: (_ for _ in ()).throw(FunctionTimeoutError("timed out after 600s")), "timeout", "UNVERIFIED"),
    (lambda: (_ for _ in ()).throw(KeyboardInterrupt()), "interrupted", "UNVERIFIED"),
    (lambda: (_ for _ in ()).throw(ConnectionError("lost")), "unknown", "UNVERIFIED"),
])
def test_remote_failure_writes_a_structured_report(tmp_path, call, stage, status):
    result, elapsed, path = _run(tmp_path, call)
    assert result is None and elapsed == pytest.approx(42.34)
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["schema"] == run_report.SCHEMA
    assert (report["task"], report["gpu"], report["code_commit"]) == ("inspect", "L4", CLEAN)
    assert report["model_revision"] == f"{p0b.PINNED_MODEL}@{p0b.PINNED_MODEL_COMMIT}"
    assert (report["stage"], report["status"]) == (stage, status)
    assert report["error"]["type"].endswith(("OSError", "RuntimeError", "FunctionTimeoutError",
                                             "KeyboardInterrupt", "ConnectionError"))
    assert report["local_elapsed_s"] == 42.3
    assert report["cost_estimate"]["kind"] == "ESTIMATE_NOT_A_BILLING_CAP"
    assert report["compatibility_pass_established"] is False and "No model compatibility PASS" in report["note"]
    assert result_exit_code(report) == 2
    assert list(path.parent.iterdir()) == [path]  # no temp file left behind


def test_success_returns_the_result_and_writes_nothing(tmp_path):
    result, elapsed, path = _run(tmp_path, lambda: {"status": "PASS"})
    assert result == {"status": "PASS"} and elapsed == pytest.approx(42.34) and not path.exists()


def test_failure_replaces_a_stale_pass(tmp_path):
    path = tmp_path / "results" / "inspect-L4-bf16-compact.json"
    run_report.write_json_atomic(path, {"status": "PASS"})
    run_report.call_remote_reported(lambda: load_model(), plan=_plan(), path=path, clock=_Clock(0.0, 1.0))
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "FAIL"


def test_system_exit_is_not_swallowed(tmp_path):
    with pytest.raises(SystemExit):
        _run(tmp_path, lambda: (_ for _ in ()).throw(SystemExit(3)))


def test_report_redacts_tokens_credentials_and_home_paths(tmp_path):
    secret = ("401 for https://user:pa55@huggingface.co with token hf_abcdefghijklmnopqrstuv, HF_TOKEN=xyz123, "
              "Authorization: Bearer eyJhbGciOi.abc, modal ak-1234567890abcd / as-abcdefghijkl, "
              "api_key: k-999 read /home/alice/.cache/huggingface/token and C:\\Users\\bob\\.modal.toml "
              "and /root/.modal.toml")

    def call():
        raise PermissionError(secret)

    _, _, path = _run(tmp_path, call)
    text = path.read_text(encoding="utf-8")
    for leaked in ("pa55", "hf_abcdefghijklmnopqrstuv", "xyz123", "eyJhbGciOi", "ak-1234567890abcd",
                   "as-abcdefghijkl", "k-999", "alice", "bob", ".modal.toml", "/root/"):
        assert leaked not in text, leaked
    assert "<redacted" in text and "<home-path>" in text


def test_report_has_no_environment_or_file_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "as-shouldnotappear123")
    _, _, path = _run(tmp_path, lambda: load_model())
    report = json.loads(path.read_text(encoding="utf-8"))
    assert "as-shouldnotappear123" not in json.dumps(report)
    assert set(report) == {"schema", "task", "gpu", "code_commit", "model_revision", "status", "stage", "error",
                           "remote_frames", "local_elapsed_s", "cost_estimate", "compatibility_pass_established",
                           "note"}
    assert all("/" not in f and "\\" not in f for f in report["remote_frames"])  # function names only


def test_message_is_bounded():
    assert len(run_report.sanitize("x" * 5000)) == run_report.MAX_MESSAGE


def test_atomic_write_keeps_the_old_file_on_error(tmp_path):
    path = tmp_path / "r.json"
    run_report.write_json_atomic(path, {"status": "FAIL"})
    with pytest.raises(TypeError):
        run_report.write_json_atomic(path, {1j: 1})  # a complex key is not JSON
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "FAIL"}
    assert list(tmp_path.iterdir()) == [path]
