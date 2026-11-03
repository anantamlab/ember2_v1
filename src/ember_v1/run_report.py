"""Local reports for Modal GPU runs (P0B-02). Pure Python: CPU tests drive it without Modal.

`call_remote_reported` runs the remote call. If the call raises (model load error, remote
exception, timeout, interrupt), it writes a structured FAIL/UNVERIFIED report atomically
and returns None. The report never establishes model compatibility.

Limits: if the local process is killed abruptly (SIGKILL, power loss), or `modal run`
fails while it builds the image (before the entrypoint runs), no local report can be
written. Modal logs (`modal app logs <app-id>`, the web dashboard) are then the evidence.
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

SCHEMA = "ember_v1.p0b.run_failure/1"
MAX_MESSAGE = 500
# Function names of the remote code path, used to place a failure in a stage. Remote
# exceptions re-raised by Modal can carry these frames; if none is present the stage stays unknown.
LOAD_FRAMES = frozenset({"load", "load_model", "from_pretrained", "load_lm_as_decision_model", "_load_backbone",
                         "runtime_metadata"})
TASK_FRAMES = frozenset({"inspect", "smoke", "padding", "throughput", "task_inspect", "task_smoke", "task_padding",
                         "task_throughput"})

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


def failure_stage(exc: BaseException) -> tuple[str, str]:
    """(stage, status) of a failed remote call.

    Timeouts and interrupts leave the model unjudged (UNVERIFIED). An exception raised by
    the remote code is a FAIL of that stage. Anything else is UNVERIFIED.
    """
    name = type(exc).__name__
    if isinstance(exc, KeyboardInterrupt):
        return "interrupted", "UNVERIFIED"
    if "Timeout" in name or isinstance(exc, TimeoutError):
        return "timeout", "UNVERIFIED"
    frames = set(_frames(exc))
    if frames & LOAD_FRAMES:
        return "model_load", "FAIL"
    if frames & TASK_FRAMES:
        return "remote_task", "FAIL"
    return "unknown", "UNVERIFIED"


def failure_report(plan: Mapping[str, Any], exc: BaseException, elapsed_s: float) -> dict[str, Any]:
    stage, status = failure_stage(exc)
    return {
        "schema": SCHEMA,
        "task": plan.get("task"),
        "gpu": plan.get("gpu"),
        "code_commit": plan.get("code_commit"),
        "model_revision": f"{plan.get('model')}@{plan.get('revision')}",
        "status": status,
        "stage": stage,
        "error": {"type": f"{type(exc).__module__}.{type(exc).__qualname__}", "message": sanitize(exc)},
        "remote_frames": _frames(exc),
        "local_elapsed_s": round(elapsed_s, 1),
        "cost_estimate": dict(plan.get("cost_estimate") or {}),
        "compatibility_pass_established": False,
        "note": ("No model compatibility PASS was established. cost_estimate is an ESTIMATE, not a bill; "
                 "read actual charges on the Modal usage page. Modal logs hold the remote evidence."),
    }


def call_remote_reported(call: Callable[[], Any], *, plan: Mapping[str, Any], path: Path | str,
                         clock: Callable[[], float] = time.monotonic) -> tuple[Any, float]:
    """Run `call()`. Return (result, elapsed_s); on failure write the report and return (None, elapsed_s).

    SystemExit is not caught. KeyboardInterrupt is reported (UNVERIFIED) when the process survives it.
    """
    started = clock()
    try:
        return call(), clock() - started
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - every failure gets a report
        elapsed = clock() - started
        write_json_atomic(path, failure_report(plan, exc, elapsed))
        return None, elapsed
