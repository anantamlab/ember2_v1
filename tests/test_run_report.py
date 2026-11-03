"""Bounded-wait watchdog and structured local reports for Modal runs (CPU only, fake call handles)."""
import json
from collections import namedtuple
from pathlib import Path

import pytest

from ember_v1 import p0b, run_report
from ember_v1.p0b import result_exit_code

CLEAN = "e" * 40
CALL_ID = "fc-01TESTCALL"
APP_ID = "ap-TESTAPP"
Stats = namedtuple("Stats", "backlog num_total_runners num_running_inputs input_headroom")
QUEUED, STARTING, RUNNING, EMPTY = Stats(1, 0, 0, 0), Stats(0, 1, 0, 0), Stats(0, 1, 1, 0), Stats(0, 0, 0, 0)
DEADLINE = run_report.QUEUE_WAIT_LIMIT_S + 900 + 600 + run_report.CLIENT_GRACE_S


def _plan():
    return p0b.gpu_run_plan("inspect", "L4", p0b.PINNED_MODEL, "bf16", "inspect", CLEAN)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def load_model(*args):  # same name as the remote load frame
    raise OSError("unsloth/Qwen3.5-2B does not appear to have a file named model.safetensors")


def task_inspect(*args):  # same name as the remote task frame
    raise RuntimeError("CUDA error: device-side assert triggered")


class FunctionTimeoutError(Exception):  # name of Modal's execution-timeout exception
    pass


class RemoteError(Exception):  # Modal error without remote frames (e.g. app stopped)
    pass


class InputCancellation(Exception):
    pass


class ConnectionError(Exception):  # noqa: A001 - shaped like modal.exception.ConnectionError
    pass


ConnectionError.__module__ = "modal.exception"


class FakeCall:
    """FunctionCall stand-in. `script` items are consumed per get(): "wait" (no output within the
    timeout), an exception instance or callable to raise, or ("return", value). Then it waits forever."""

    def __init__(self, clock, script=(), after_cancel="terminal", cancel_error=None):
        self.clock, self.script, self.after_cancel, self.cancel_error = clock, list(script), after_cancel, cancel_error
        self.object_id = CALL_ID
        self.gets, self.cancels, self.cancelled = [], [], False

    def _wait(self, timeout):
        self.clock.t += timeout
        raise TimeoutError()

    def get(self, timeout=None):
        assert timeout is not None and 0 < timeout <= max(run_report.POLL_INTERVAL_S, run_report.CANCEL_VERIFY_S)
        self.gets.append(timeout)
        if self.cancelled:
            if self.after_cancel == "terminal":
                raise InputCancellation("Input cancelled")
            if isinstance(self.after_cancel, tuple):
                return self.after_cancel[1]
            self._wait(timeout)
        step = self.script.pop(0) if self.script else "wait"
        if step == "wait":
            self._wait(timeout)
        if isinstance(step, tuple):
            return step[1]
        if isinstance(step, BaseException):
            raise step
        step()

    def cancel(self, terminate_containers=False):
        self.cancels.append(terminate_containers)
        if self.cancel_error is not None:
            raise self.cancel_error
        self.cancelled = True


class Submitter:
    def __init__(self, call=None, error=None):
        self.call, self.error, self.count = call, error, 0

    def __call__(self):
        self.count += 1
        if self.error is not None:
            raise self.error
        return self.call


def stats_seq(*items):
    """get_current_stats() stand-in: returns items in order, then repeats the last one."""
    items = list(items)

    def current():
        item = items.pop(0) if len(items) > 1 else items[0]
        if isinstance(item, BaseException):
            raise item
        return item
    return current


def _watch(tmp_path, call, stats=None, submit=None, clock=None):
    path = tmp_path / "results" / "inspect-L4-bf16-compact.json"
    submit = submit or Submitter(call)
    result, watch = run_report.watch_remote_call(submit, plan=_plan(), path=path, current_stats=stats,
                                                 app_id=APP_ID, clock=clock or call.clock)
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    return result, watch, report, submit, path


def _assert_not_pass(report, stage, path):
    assert report["schema"] == run_report.SCHEMA and report["stage"] == stage
    assert report["status"] == "UNVERIFIED" and report["compatibility_pass_established"] is False
    assert result_exit_code(report) == 2
    assert (report["task"], report["gpu"], report["code_commit"]) == ("inspect", "L4", CLEAN)
    assert report["model_revision"] == f"{p0b.PINNED_MODEL}@{p0b.PINNED_MODEL_COMMIT}"
    assert report["cost_estimate"]["kind"] == "ESTIMATE_NOT_A_BILLING_CAP"
    assert list(path.parent.iterdir()) == [path]  # no temp file left behind


# --- success -----------------------------------------------------------------------------------

def test_success_returns_the_result_after_one_submission(tmp_path):
    clock = Clock()
    call = FakeCall(clock, ["wait", "wait", ("return", {"status": "PASS"})])
    result, watch, report, submit, _ = _watch(tmp_path, call, stats_seq(QUEUED, RUNNING))
    assert result == {"status": "PASS"} and submit.count == 1 and call.cancels == []
    assert watch["function_call_id"] == CALL_ID and watch["execution_state"] == "completed"
    assert watch["elapsed_s"] == 60.0 and watch["first_start_evidence_s"] == 60.0
    # The provisional record stays until the entrypoint writes the result; it is never a PASS.
    assert report["stage"] == "in_progress" and report["status"] == "UNVERIFIED"
    assert report["outstanding_function_call_id"] == CALL_ID


def test_a_transient_poll_error_does_not_resubmit(tmp_path):
    call = FakeCall(Clock(), [ConnectionError("reset"), "wait", ("return", {"status": "PASS"})])
    result, watch, _, submit, _ = _watch(tmp_path, call, stats_seq(QUEUED))
    assert result == {"status": "PASS"} and submit.count == 1 and watch["poll_errors"] == 1


def test_provisional_record_replaces_a_stale_pass_before_waiting(tmp_path):
    path = tmp_path / "results" / "inspect-L4-bf16-compact.json"
    run_report.write_json_atomic(path, {"status": "PASS"})
    seen = []

    class Peek(FakeCall):
        def get(self, timeout=None):
            seen.append(json.loads(path.read_text(encoding="utf-8")))
            return ("ok", 1)

    run_report.watch_remote_call(Submitter(Peek(Clock())), plan=_plan(), path=path, clock=Clock())
    assert seen[0]["stage"] == "in_progress" and seen[0]["function_call_id"] == CALL_ID
    assert seen[0]["status"] == "UNVERIFIED" and seen[0]["manual_cleanup"]


# --- queue and deadline ------------------------------------------------------------------------

def test_confirmed_queue_wait_is_cancelled_at_the_queue_limit(tmp_path):
    call = FakeCall(Clock())
    result, watch, report, submit, path = _watch(tmp_path, call, stats_seq(QUEUED))
    assert result is None and submit.count == 1
    _assert_not_pass(report, "queue_timeout", path)
    assert report["local_elapsed_s"] == run_report.QUEUE_WAIT_LIMIT_S
    assert call.cancels == [True]  # explicit cancel, containers terminated
    # A generic terminal error after the cancel is not proof of cancellation: keep the ID and cleanup.
    assert report["watchdog"]["cancellation"]["outcome"] == "terminal_error_observed"
    assert report["watchdog"]["execution_state"] == "queued"
    assert report["watchdog"]["queued_snapshots"] == run_report.QUEUE_WAIT_LIMIT_S // run_report.POLL_INTERVAL_S
    assert report["outstanding_function_call_id"] == CALL_ID and report["manual_cleanup"]
    assert report["function_call_id"] == CALL_ID and report["app_id"] == APP_ID


@pytest.mark.parametrize("stats", [
    None,                                            # no stats source
    stats_seq(ConnectionError("stats down"), QUEUED),  # one failed reading
    stats_seq(EMPTY),                                # inconsistent: no input, no container
])
def test_unknown_queue_state_waits_for_the_client_deadline(tmp_path, stats):
    call = FakeCall(Clock())
    result, watch, report, submit, path = _watch(tmp_path, call, stats)
    assert result is None and submit.count == 1
    _assert_not_pass(report, "client_deadline", path)
    assert report["local_elapsed_s"] == DEADLINE == run_report.client_deadline_s(_plan())
    assert report["watchdog"]["execution_state"] == "unknown"  # never claimed "queued"
    assert call.cancels == [True]


def test_deadline_after_the_run_may_have_started(tmp_path):
    call = FakeCall(Clock())
    result, _, report, submit, path = _watch(tmp_path, call, stats_seq(QUEUED, QUEUED, STARTING, RUNNING))
    assert result is None and submit.count == 1
    _assert_not_pass(report, "client_deadline", path)
    assert report["watchdog"]["execution_state"] == "started_observed"
    assert report["watchdog"]["first_start_evidence_s"] == 90.0
    assert report["local_elapsed_s"] == DEADLINE and call.cancels == [True]


def test_short_queue_then_start_is_not_cancelled_at_the_queue_limit(tmp_path):
    script = ["wait"] * 15 + [("return", {"status": "FAIL"})]
    call = FakeCall(Clock(), script)
    result, watch, _, _, _ = _watch(tmp_path, call, stats_seq(QUEUED, QUEUED, RUNNING))
    assert result == {"status": "FAIL"} and call.cancels == [] and watch["elapsed_s"] == 450.0


# --- cancellation ------------------------------------------------------------------------------

def test_cancellation_failure_keeps_the_call_id_and_cleanup_steps(tmp_path):
    call = FakeCall(Clock(), cancel_error=ConnectionError("cancel rpc failed"))
    result, _, report, submit, path = _watch(tmp_path, call, stats_seq(QUEUED))
    assert result is None and submit.count == 1
    _assert_not_pass(report, "queue_timeout", path)
    cancel = report["watchdog"]["cancellation"]
    assert cancel["outcome"] == "failed" and cancel["error"]["message"] == "cancel rpc failed"
    assert report["outstanding_function_call_id"] == CALL_ID
    assert any(CALL_ID in step and "terminate_containers=True" in step for step in report["manual_cleanup"])
    assert f"modal app stop --yes {APP_ID}" in report["manual_cleanup"]


def test_unanswered_cancel_is_reported_as_unverified(tmp_path):
    call = FakeCall(Clock(), after_cancel="silent")
    _, _, report, _, path = _watch(tmp_path, call, stats_seq(QUEUED))
    _assert_not_pass(report, "queue_timeout", path)
    assert report["watchdog"]["cancellation"]["outcome"] == "unverified"
    assert report["outstanding_function_call_id"] == CALL_ID


def test_result_arriving_with_the_cancel_is_kept(tmp_path):
    call = FakeCall(Clock(), after_cancel=("return", {"status": "PASS"}))
    result, watch, _, submit, _ = _watch(tmp_path, call, stats_seq(QUEUED))
    assert result == {"status": "PASS"} and submit.count == 1
    assert watch["cancellation"]["outcome"] == "completed_before_cancel"


# --- polling failure, interruption, submission -------------------------------------------------

def test_repeated_poll_errors_stop_and_cancel(tmp_path):
    call = FakeCall(Clock(), [ConnectionError("down")] * run_report.MAX_CONSECUTIVE_POLL_ERRORS)
    result, _, report, submit, path = _watch(tmp_path, call, stats_seq(QUEUED))
    assert result is None and submit.count == 1
    _assert_not_pass(report, "poll_failed", path)
    assert report["error"]["message"] == "down" and call.cancels == [True]


def test_user_interrupt_cancels_and_reports(tmp_path):
    call = FakeCall(Clock(), ["wait", KeyboardInterrupt()])
    result, _, report, submit, path = _watch(tmp_path, call, stats_seq(QUEUED))
    assert result is None and submit.count == 1
    _assert_not_pass(report, "interrupted", path)
    assert call.cancels == [True] and report["watchdog"]["cancellation"]["outcome"] == "terminal_error_observed"


def test_submit_failure_is_unverified_without_a_call_id(tmp_path):
    submit = Submitter(error=RuntimeError("Please add a payment method to use L4 GPU functions."))
    result, watch, report, _, path = _watch(tmp_path, None, submit=submit, clock=Clock())
    assert result is None and submit.count == 1
    _assert_not_pass(report, "submit_failed", path)
    assert report["function_call_id"] is None and watch["cancellation"] is None


def test_system_exit_is_re_raised_after_a_cancel(tmp_path, capsys):
    call = FakeCall(Clock(), [SystemExit(3)])
    submit = Submitter(call)
    with pytest.raises(SystemExit):
        _watch(tmp_path, call, submit=submit)
    assert submit.count == 1 and call.cancels == [True] and CALL_ID in capsys.readouterr().err


# --- the interval between spawn() and the first poll ---------------------------------------------

def _stderr_record(capsys):
    lines = [json.loads(line) for line in capsys.readouterr().err.splitlines() if line.startswith("{")]
    assert len(lines) == 1
    return lines[0]


def test_provisional_write_failure_cancels_and_reports_on_stderr(tmp_path, monkeypatch, capsys):
    path = tmp_path / "results" / "inspect-L4-bf16-compact.json"
    run_report.write_json_atomic(path, {"status": "PASS"})  # stale result from an older run

    def no_disk(*args, **kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(run_report, "write_json_atomic", no_disk)
    call = FakeCall(Clock())
    submit = Submitter(call)
    with pytest.raises(OSError, match="No space left"):  # the original error, not a later one
        run_report.watch_remote_call(submit, plan=_plan(), path=path, app_id=APP_ID, clock=call.clock)
    assert submit.count == 1 and call.cancels == [True]
    record = _stderr_record(capsys)
    assert record["function_call_id"] == CALL_ID and record["ember_v1_watchdog"] == "post_submit_error"
    assert record["cancellation"] == "terminal_error_observed"
    assert any(CALL_ID in step for step in record["manual_cleanup"])
    assert not path.exists()  # the stale PASS is removed when no report can be written


def test_log_failure_after_submit_cancels_and_writes_a_report(tmp_path, capsys):
    path = tmp_path / "results" / "inspect-L4-bf16-compact.json"
    call = FakeCall(Clock())
    submit = Submitter(call)

    def log(line):
        raise BrokenPipeError("stdout closed")

    with pytest.raises(BrokenPipeError):
        run_report.watch_remote_call(submit, plan=_plan(), path=path, app_id=APP_ID, clock=call.clock, log=log)
    assert submit.count == 1 and call.cancels == [True] and call.gets == [run_report.CANCEL_VERIFY_S]
    report = json.loads(path.read_text(encoding="utf-8"))
    _assert_not_pass(report, "post_submit_error", path)
    assert report["outstanding_function_call_id"] == CALL_ID
    assert report["error"]["type"].endswith("BrokenPipeError")
    assert _stderr_record(capsys)["function_call_id"] == CALL_ID


def test_interrupt_after_submit_cancels_and_re_raises(tmp_path):
    path = tmp_path / "results" / "inspect-L4-bf16-compact.json"
    call = FakeCall(Clock())
    submit = Submitter(call)

    def log(line):
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        run_report.watch_remote_call(submit, plan=_plan(), path=path, clock=call.clock, log=log)
    assert submit.count == 1 and call.cancels == [True]
    _assert_not_pass(json.loads(path.read_text(encoding="utf-8")), "interrupted", path)


def test_cancel_failure_after_a_write_failure_keeps_the_original_error(tmp_path, monkeypatch, capsys):
    def no_disk(*args, **kwargs):
        raise PermissionError("read-only file system")

    monkeypatch.setattr(run_report, "write_json_atomic", no_disk)
    call = FakeCall(Clock(), cancel_error=ConnectionError("cancel rpc failed"))
    submit = Submitter(call)
    with pytest.raises(PermissionError, match="read-only"):
        run_report.watch_remote_call(submit, plan=_plan(), path=tmp_path / "r.json", app_id=APP_ID,
                                     clock=call.clock)
    record = _stderr_record(capsys)
    assert submit.count == 1 and call.cancels == [True] and call.gets == []
    assert record["cancellation"] == "failed" and f"modal app stop --yes {APP_ID}" in record["manual_cleanup"]


def test_broken_stderr_does_not_hide_the_original_error(tmp_path, monkeypatch):
    class Broken:
        def write(self, text):
            raise OSError("stderr closed")

        def flush(self):
            raise OSError("stderr closed")

    def no_disk(*args, **kwargs):
        raise OSError("disk gone")

    monkeypatch.setattr(run_report, "write_json_atomic", no_disk)
    monkeypatch.setattr(run_report.sys, "stderr", Broken())
    call = FakeCall(Clock())
    with pytest.raises(OSError, match="disk gone"):
        run_report.watch_remote_call(Submitter(call), plan=_plan(), path=tmp_path / "r.json", clock=call.clock)
    assert call.cancels == [True]


def test_final_report_write_failure_still_records_the_cancel_outcome(tmp_path, monkeypatch, capsys):
    real = run_report.write_json_atomic
    writes = []

    def second_write_fails(path, doc):
        writes.append(doc.get("stage"))
        if len(writes) > 1:
            raise OSError("disk full")
        return real(path, doc)

    monkeypatch.setattr(run_report, "write_json_atomic", second_write_fails)
    call = FakeCall(Clock())
    with pytest.raises(OSError, match="disk full"):
        run_report.watch_remote_call(Submitter(call), plan=_plan(), path=tmp_path / "r.json",
                                     current_stats=stats_seq(QUEUED), clock=call.clock)
    record = _stderr_record(capsys)
    assert call.cancels == [True] and record["ember_v1_watchdog"] == "queue_timeout"
    assert record["cancellation"] == "terminal_error_observed" and record["function_call_id"] == CALL_ID


# --- cancellation evidence -----------------------------------------------------------------------

@pytest.mark.parametrize("after_cancel_error", [
    RemoteError("Function call was cancelled"),  # wording alone is not cancellation-specific evidence
    RuntimeError("CUDA error: an illegal memory access was encountered"),
    InputCancellation("Input was cancelled by user"),
])
def test_no_error_after_cancel_counts_as_confirmed(tmp_path, after_cancel_error):
    class Call(FakeCall):
        def get(self, timeout=None):
            if self.cancelled:
                raise after_cancel_error
            return super().get(timeout)

    call = Call(Clock())
    _, _, report, submit, path = _watch(tmp_path, call, stats_seq(QUEUED))
    _assert_not_pass(report, "queue_timeout", path)
    cancel = report["watchdog"]["cancellation"]
    assert submit.count == 1 and cancel["outcome"] == "terminal_error_observed"
    assert cancel["error"]["type"].endswith(type(after_cancel_error).__name__)
    assert report["outstanding_function_call_id"] == CALL_ID and report["manual_cleanup"]


def test_cancel_outcomes_never_include_confirmed():
    src = Path(run_report.__file__).read_text(encoding="utf-8")
    assert 'outcome="confirmed"' not in src
    assert set(run_report.UNCONFIRMED_CANCEL) == {"failed", "unverified", "terminal_error_observed"}


# --- remote outcomes ---------------------------------------------------------------------------

@pytest.mark.parametrize("step, stage, status", [
    (lambda: load_model(), "model_load", "FAIL"),
    (lambda: task_inspect(), "remote_task", "FAIL"),
    (FunctionTimeoutError("timed out after 600s"), "remote_timeout", "UNVERIFIED"),
    (RemoteError(""), "unknown", "UNVERIFIED"),
])
def test_remote_outcomes_are_terminal_and_not_cancelled(tmp_path, step, stage, status):
    call = FakeCall(Clock(), ["wait", step])
    result, watch, report, submit, path = _watch(tmp_path, call, stats_seq(RUNNING))
    assert result is None and submit.count == 1 and call.cancels == []
    assert (report["stage"], report["status"]) == (stage, status) and result_exit_code(report) == 2
    assert report["compatibility_pass_established"] is False and report["manual_cleanup"] == []
    assert list(path.parent.iterdir()) == [path]


def test_never_fail_when_the_model_was_not_reached(tmp_path):
    # Every client-side ending is UNVERIFIED; FAIL needs remote load/task frames.
    assert set(run_report.UNVERIFIED_STAGES).isdisjoint(run_report.FAIL_STAGES)
    for stage in run_report.UNVERIFIED_STAGES:
        assert run_report.failure_report(_plan(), None, 1.0, stage=stage)["status"] == "UNVERIFIED"


def test_remote_connection_error_inside_load_is_a_load_failure_not_a_poll_error(tmp_path):
    def load_model():  # noqa: F811 - remote frame name
        raise ConnectionError("huggingface.co unreachable")

    call = FakeCall(Clock(), [lambda: load_model()])
    _, watch, report, _, _ = _watch(tmp_path, call, stats_seq(RUNNING))
    assert report["stage"] == "model_load" and watch["poll_errors"] == 0


# --- report hygiene ----------------------------------------------------------------------------

def test_report_redacts_tokens_credentials_and_home_paths(tmp_path):
    secret = ("401 for https://user:pa55@huggingface.co with token hf_abcdefghijklmnopqrstuv, HF_TOKEN=xyz123, "
              "Authorization: Bearer eyJhbGciOi.abc, modal ak-1234567890abcd / as-abcdefghijkl, "
              "api_key: k-999 read /home/alice/.cache/huggingface/token and C:\\Users\\bob\\.modal.toml "
              "and /root/.modal.toml")
    call = FakeCall(Clock(), [PermissionError(secret)])
    _, _, report, _, path = _watch(tmp_path, call, stats_seq(RUNNING))
    text = path.read_text(encoding="utf-8")
    for leaked in ("pa55", "hf_abcdefghijklmnopqrstuv", "xyz123", "eyJhbGciOi", "ak-1234567890abcd",
                   "as-abcdefghijkl", "k-999", "alice", "bob", ".modal.toml", "/root/"):
        assert leaked not in text, leaked
    assert "<redacted" in text and "<home-path>" in text


def test_report_has_no_environment_or_file_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "as-shouldnotappear123")
    call = FakeCall(Clock(), [lambda: load_model()])
    _, _, report, _, _ = _watch(tmp_path, call, stats_seq(RUNNING))
    assert "as-shouldnotappear123" not in json.dumps(report)
    assert set(report) == {"schema", "task", "gpu", "code_commit", "model_revision", "status", "stage", "error",
                           "remote_frames", "local_elapsed_s", "function_call_id", "app_id", "watchdog",
                           "outstanding_function_call_id", "manual_cleanup", "cost_estimate",
                           "compatibility_pass_established", "note"}
    assert all("/" not in f and "\\" not in f for f in report["remote_frames"])  # function names only


def test_message_is_bounded():
    assert len(run_report.sanitize("x" * 5000)) == run_report.MAX_MESSAGE


def test_snapshots_are_bounded(tmp_path):
    _, _, report, _, _ = _watch(tmp_path, FakeCall(Clock()), None)
    assert len(report["watchdog"]["snapshots"]) == run_report.MAX_SNAPSHOTS


def test_atomic_write_keeps_the_old_file_on_error(tmp_path):
    path = tmp_path / "r.json"
    run_report.write_json_atomic(path, {"status": "FAIL"})
    with pytest.raises(TypeError):
        run_report.write_json_atomic(path, {1j: 1})  # a complex key is not JSON
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "FAIL"}
    assert list(tmp_path.iterdir()) == [path]


# --- installed Modal SDK contract (no live app) ------------------------------------------------

def test_watchdog_uses_only_public_modal_1_6_3_call_apis():
    modal = pytest.importorskip("modal")  # CI installs no modal; checked where the client runs
    import dataclasses
    import inspect

    import modal._functions as functions
    import modal.types as types

    assert modal.__version__ == "1.6.3"
    assert "timeout" in inspect.signature(modal.FunctionCall.get).parameters
    assert "terminate_containers" in inspect.signature(modal.FunctionCall.cancel).parameters
    assert hasattr(modal.Function, "spawn") and hasattr(modal.Function, "get_current_stats")
    assert {"backlog", "num_total_runners", "num_running_inputs"} <= {
        f.name for f in dataclasses.fields(types.FunctionCurrentStats)}
    # get(timeout=) without an output raises the builtin TimeoutError (no modal TimeoutError imported).
    assert "TimeoutError" not in vars(functions)
    assert "raise TimeoutError()" in inspect.getsource(functions._Invocation.poll_function)
