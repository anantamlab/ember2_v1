"""Local reports and a bounded-wait watchdog for Modal GPU runs (P0B-02). Pure Python: CPU tests
drive it with fake call handles, without Modal.

`watch_remote_call` submits the remote call exactly once (`spawn`), then waits with bounded
`FunctionCall.get(timeout=...)` polls. It never resubmits. After each empty poll it reads a
`Function.get_current_stats()` snapshot (public Modal API: backlog, runners, running inputs).

- Confirmed queue timeout: every snapshot since submission says "queued, no container"
  (backlog >= 1, 0 runners, 0 running inputs), at least MIN_QUEUED_SNAPSHOTS of them, no stats
  error, and QUEUE_WAIT_LIMIT_S has passed. Then the call is cancelled.
- Client deadline: when the queue state is not confirmed (stats failed, inconsistent, or a container
  was seen), the call is cancelled at `client_deadline_s(plan)` = queue limit + Modal startup
  timeout + Modal execution timeout + grace. The report then says the execution state is unknown
  (or that a start was observed); it never claims the call stayed queued.
- A timed-out `get()` does not cancel the remote call. Cancellation is an explicit
  `FunctionCall.cancel(terminate_containers=True)`, then one more bounded `get()` checks it:
  a terminal error means "confirmed", no answer means "unverified", an error from cancel() means
  "failed". If not confirmed, the report keeps the FunctionCall ID and the manual cleanup steps.

Every non-success ends in a FAIL/UNVERIFIED report written atomically at `path`; a provisional
UNVERIFIED record with the FunctionCall ID is written right after submission, so a stale PASS never
survives. FAIL needs evidence from the remote code (load or task frames); a call that never reached
the model is UNVERIFIED.

Limits: if the local process is killed abruptly (SIGKILL, power loss), only the provisional record
remains, and the call is not cancelled by this code. If `modal run` fails while it creates the app
(image build, GPU access refused), the entrypoint never runs and no local report exists. Modal logs
(`modal app logs <app-id>`, `modal function calls <function-id>`, the dashboard) are then the evidence.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
import traceback
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

SCHEMA = "ember_v1.p0b.run_failure/2"
MAX_MESSAGE = 500
# Function names of the remote code path, used to place a failure in a stage. Remote
# exceptions re-raised by Modal can carry these frames; if none is present the stage stays unknown.
LOAD_FRAMES = frozenset({"load", "load_model", "from_pretrained", "load_lm_as_decision_model", "_load_backbone",
                         "runtime_metadata"})
TASK_FRAMES = frozenset({"inspect", "smoke", "padding", "throughput", "task_inspect", "task_smoke", "task_padding",
                         "task_throughput"})

# Watchdog limits (seconds). Modal's startup (900 s) and execution (600 s for inspect) timeouts
# start only once a container exists; queue time is bounded only here, on the client.
QUEUE_WAIT_LIMIT_S = 300
POLL_INTERVAL_S = 30
MIN_QUEUED_SNAPSHOTS = 3
CLIENT_GRACE_S = 60
CANCEL_VERIFY_S = 15
MAX_CONSECUTIVE_POLL_ERRORS = 3
MAX_SNAPSHOTS = 20

UNVERIFIED_STAGES = ("in_progress", "submit_failed", "queue_timeout", "client_deadline", "interrupted",
                     "poll_failed", "remote_timeout", "unknown")
FAIL_STAGES = ("model_load", "remote_task")

_REDACTIONS = (
    (re.compile(r"hf_[A-Za-z0-9]{8,}"), "<redacted-hf-token>"),
    (re.compile(r"\b(?:ak|as)-[A-Za-z0-9]{8,}"), "<redacted-modal-token>"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"), "Bearer <redacted>"),
    (re.compile(r"(?i)\b([\w-]*(?:token|secret|password|passwd|api[_-]?key|credential)s?)\b\s*[=:]\s*\S+"),
     r"\1=<redacted>"),
    (re.compile(r"(://)[^/\s:@]+:[^/\s@]+@"), r"\1<redacted>@"),
    # Home directories and anything below them (user names, token files, local layout).
    (re.compile(r"(?i)\b[A-Z]:[\\/]+Users[\\/]+[^\s'\"]+"), "<home-path>"),
    (re.compile(r"/(?:home|Users)/[^\s'\"]+"), "<home-path>"),
    (re.compile(r"/root/[^\s'\"]+"), "<root-path>"),
)


def sanitize(text: Any, limit: int = MAX_MESSAGE) -> str:
    """Error text without tokens, credentials or home paths, cut to `limit` characters."""
    out = str(text)
    for pattern, repl in _REDACTIONS:
        out = pattern.sub(repl, out)
    return out[:limit]


def write_json_atomic(path: Path | str, doc: Mapping[str, Any]) -> Path:
    """Write JSON to a temp file in the same directory, fsync, then rename over `path`.

    A reader sees either the old file or the complete new one, never a partial file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def _frames(exc: BaseException) -> list[str]:
    """Function names only (no file paths) of the traceback, innermost last."""
    return [frame.name for frame in traceback.extract_tb(exc.__traceback__)][-20:]


def _error(exc: BaseException | None) -> dict[str, str] | None:
    if exc is None:
        return None
    return {"type": f"{type(exc).__module__}.{type(exc).__qualname__}", "message": sanitize(exc)}


def failure_stage(exc: BaseException) -> tuple[str, str]:
    """(stage, status) of an exception that ended the remote call.

    Timeouts and interrupts leave the model unjudged (UNVERIFIED). An exception raised by
    the remote code is a FAIL of that stage. Anything else is UNVERIFIED.
    """
    name = type(exc).__name__
    if isinstance(exc, KeyboardInterrupt):
        return "interrupted", "UNVERIFIED"
    if "Timeout" in name or isinstance(exc, TimeoutError):
        return "remote_timeout", "UNVERIFIED"
    frames = set(_frames(exc))
    if frames & LOAD_FRAMES:
        return "model_load", "FAIL"
    if frames & TASK_FRAMES:
        return "remote_task", "FAIL"
    return "unknown", "UNVERIFIED"


def _is_poll_error(exc: BaseException) -> bool:
    """A client/transport error while polling (not a result of the remote code)."""
    if set(_frames(exc)) & (LOAD_FRAMES | TASK_FRAMES):
        return False
    return isinstance(exc, ConnectionError) or (
        type(exc).__name__ == "ConnectionError" and type(exc).__module__.startswith("modal"))


def client_deadline_s(plan: Mapping[str, Any]) -> int:
    """Overall client wait: queue limit + Modal startup timeout + Modal execution timeout + grace."""
    return QUEUE_WAIT_LIMIT_S + int(plan["startup_timeout_s"]) + int(plan["timeout_s"]) + CLIENT_GRACE_S


def manual_cleanup(app_id: str | None, call_id: str | None) -> list[str]:
    steps = []
    if call_id:
        steps.append("python -c \"import modal; modal.FunctionCall.from_id('%s').cancel(terminate_containers=True)\""
                     % call_id)
    if app_id:
        steps.append(f"modal app stop --yes {app_id}")
    steps.append("Check that nothing runs: modal app list; modal container list")
    return steps


def failure_report(plan: Mapping[str, Any], exc: BaseException | None, elapsed_s: float, *,
                   stage: str | None = None, watch: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Structured non-PASS report. `stage` overrides the stage taken from `exc`."""
    if stage is None:
        stage, status = failure_stage(exc) if exc is not None else ("unknown", "UNVERIFIED")
    else:
        status = "FAIL" if stage in FAIL_STAGES else "UNVERIFIED"
    watch = dict(watch or {})
    call_id = watch.get("function_call_id")
    cancel = watch.get("cancellation") or {}
    outstanding = stage == "in_progress" or cancel.get("outcome") in ("failed", "unverified")
    return {
        "schema": SCHEMA,
        "task": plan.get("task"),
        "gpu": plan.get("gpu"),
        "code_commit": plan.get("code_commit"),
        "model_revision": f"{plan.get('model')}@{plan.get('revision')}",
        "status": status,
        "stage": stage,
        "error": _error(exc),
        "remote_frames": _frames(exc) if exc is not None else [],
        "local_elapsed_s": round(elapsed_s, 1),
        "function_call_id": call_id,
        "app_id": watch.get("app_id"),
        "watchdog": watch,
        "outstanding_function_call_id": call_id if outstanding else None,
        "manual_cleanup": manual_cleanup(watch.get("app_id"), call_id) if outstanding else [],
        "cost_estimate": dict(plan.get("cost_estimate") or {}),
        "compatibility_pass_established": False,
        "note": ("No model compatibility PASS was established. cost_estimate is an ESTIMATE, not a bill; "
                 "read actual charges on the Modal usage page. Modal logs hold the remote evidence."),
    }


def _snapshot(current_stats: Callable[[], Any] | None, at: float) -> dict[str, Any]:
    """One `get_current_stats()` reading, classified. Never raises."""
    if current_stats is None:
        return {"t_s": round(at, 1), "state": "unknown", "error": "no stats source"}
    try:
        stats = current_stats()
        snap = {"backlog": int(stats.backlog), "runners": int(stats.num_total_runners),
                "running_inputs": int(stats.num_running_inputs)}
    except Exception as exc:  # noqa: BLE001 - a failed reading only makes the queue state unknown
        return {"t_s": round(at, 1), "state": "unknown", "error": _error(exc)}
    if snap["running_inputs"] > 0:
        state = "running"
    elif snap["runners"] > 0:
        state = "container"
    elif snap["backlog"] >= 1:
        state = "queued"
    else:
        state = "unknown"  # no input seen and no container: not evidence of queueing
    return {"t_s": round(at, 1), "state": state, **snap}


def _cancel(call: Any) -> tuple[dict[str, Any], Any, bool]:
    """Explicit cancel, then one bounded get() to check it. Returns (record, late_result, has_late_result)."""
    rec: dict[str, Any] = {"requested": True, "terminate_containers": True, "outcome": None, "error": None,
                           "verify": None}
    try:
        call.cancel(terminate_containers=True)
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001
        rec.update(outcome="failed", error=_error(exc))
        return rec, None, False
    try:
        late = call.get(timeout=CANCEL_VERIFY_S)
    except TimeoutError:
        rec.update(outcome="unverified", verify=f"no terminal state within {CANCEL_VERIFY_S} s after cancel")
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001
        if isinstance(exc, KeyboardInterrupt) or _is_poll_error(exc):
            rec.update(outcome="unverified", verify="check interrupted or failed", error=_error(exc))
        else:
            rec.update(outcome="confirmed", verify=f"get() after cancel ended with {type(exc).__name__}")
    else:
        rec.update(outcome="completed_before_cancel", verify="the call returned its result before the cancel")
        return rec, late, True
    return rec, None, False


def watch_remote_call(submit: Callable[[], Any], *, plan: Mapping[str, Any], path: Path | str,
                      current_stats: Callable[[], Any] | None = None, app_id: str | None = None,
                      clock: Callable[[], float] = time.monotonic,
                      log: Callable[[str], None] | None = None) -> tuple[Any, dict[str, Any]]:
    """Submit once and wait within bounds. Return (result, watch); result is None after a written report.

    `submit()` must return a FunctionCall-like object (`object_id`, `get(timeout=)`, `cancel(terminate_containers=)`).
    SystemExit is not caught.
    """
    started = clock()

    def elapsed() -> float:
        return clock() - started

    deadline = client_deadline_s(plan)
    watch: dict[str, Any] = {
        "app_id": app_id, "function_call_id": None, "submissions": 0,
        "queue_wait_limit_s": QUEUE_WAIT_LIMIT_S, "client_deadline_s": deadline, "poll_interval_s": POLL_INTERVAL_S,
        "polls": 0, "poll_errors": 0, "stats_errors": 0, "queued_snapshots": 0, "snapshots": [],
        "first_start_evidence_s": None, "execution_state": "not_submitted", "cancellation": None,
    }

    def finish(stage: str, exc: BaseException | None) -> tuple[None, dict[str, Any]]:
        watch["elapsed_s"] = round(elapsed(), 1)
        write_json_atomic(path, failure_report(plan, exc, elapsed(), stage=stage, watch=watch))
        if log:
            log(json.dumps({"stage": stage, "function_call_id": watch["function_call_id"],
                            "cancellation": (watch["cancellation"] or {}).get("outcome")}))
        return None, watch

    try:
        watch["submissions"] += 1
        call = submit()
    except KeyboardInterrupt as exc:
        return finish("interrupted", exc)
    except Exception as exc:  # noqa: BLE001 - the input may or may not exist; nothing to cancel by ID
        watch["execution_state"] = "unknown"
        return finish("submit_failed", exc)
    watch["function_call_id"] = getattr(call, "object_id", None)
    watch["execution_state"] = "submitted"
    # Provisional record: replaces any older result and keeps the call ID if this process dies.
    write_json_atomic(path, failure_report(plan, None, elapsed(), stage="in_progress", watch=watch))
    if log:
        log(json.dumps({"submitted": watch["function_call_id"], "app_id": app_id, "client_deadline_s": deadline}))

    trigger, cause, errors_in_row = None, None, 0
    try:
        while True:
            remaining = deadline - elapsed()
            if remaining <= 0:
                trigger = "client_deadline"
                break
            try:
                result = call.get(timeout=min(POLL_INTERVAL_S, remaining))
            except TimeoutError:
                watch["polls"] += 1
                errors_in_row = 0
            except Exception as exc:  # noqa: BLE001
                watch["polls"] += 1
                if _is_poll_error(exc):
                    watch["poll_errors"] += 1
                    errors_in_row += 1
                    if errors_in_row >= MAX_CONSECUTIVE_POLL_ERRORS:
                        trigger, cause = "poll_failed", exc
                        break
                    continue
                watch["execution_state"] = "ended_with_error"  # terminal: nothing left to cancel
                return finish(failure_stage(exc)[0], exc)
            else:
                watch["polls"] += 1
                watch["execution_state"] = "completed"
                watch["elapsed_s"] = round(elapsed(), 1)
                return result, watch
            snap = _snapshot(current_stats, elapsed())
            watch["stats_errors"] += "error" in snap
            watch["queued_snapshots"] += snap["state"] == "queued"
            watch["snapshots"] = (watch["snapshots"] + [snap])[-MAX_SNAPSHOTS:]
            if snap["state"] in ("running", "container") and watch["first_start_evidence_s"] is None:
                watch["first_start_evidence_s"] = snap["t_s"]
            # Confirmed only if every reading so far was "queued" and none failed.
            readings = watch["polls"] - watch["poll_errors"]
            confirmed_queued = (watch["first_start_evidence_s"] is None and watch["stats_errors"] == 0
                                and watch["queued_snapshots"] == readings
                                and watch["queued_snapshots"] >= MIN_QUEUED_SNAPSHOTS)
            watch["execution_state"] = ("started_observed" if watch["first_start_evidence_s"] is not None
                                        else "queued" if confirmed_queued else "unknown")
            if confirmed_queued and elapsed() >= QUEUE_WAIT_LIMIT_S:
                trigger = "queue_timeout"
                break
    except KeyboardInterrupt as exc:
        trigger, cause = "interrupted", exc

    rec, late, has_late = _cancel(call)
    watch["cancellation"] = rec
    if has_late:  # the call finished between the last poll and the cancel: keep its result
        watch["execution_state"] = "completed"
        watch["elapsed_s"] = round(elapsed(), 1)
        return late, watch
    return finish(trigger, cause)
