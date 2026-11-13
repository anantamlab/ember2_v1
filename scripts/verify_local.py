#!/usr/bin/env python3
"""Local full-suite verification for EMBER v1 (replaces GitHub Actions; see AGENTS.md).

Commands
  run           Run the COMPLETE pytest suite (`python -m pytest -q`) with each interpreter, keep the full output in
                an ignored log, write a receipt, and print PASS only when pytest passed and the tree did not change.
  check-push    Pre-push gate: read git's pre-push lines on stdin and require a PASS receipt for every pushed
                commit and every required Python minor version (default 3.11 and 3.12).
  install-hook  Install the repository pre-push hook into the shared git hooks directory.
  status        Print the receipts for a commit (default HEAD).

Stdlib only. Selection arguments are not accepted: the run is always the complete suite. PYTEST_ADDOPTS and
PYTEST_PLUGINS are removed from the test environment so that nothing can deselect tests silently.

Tiers (owner decision 2026-11-13): tests marked `slow` are skipped (reported as skipped, never deselected) unless
pytest gets `--run-slow`. `run` is the default tier; `run --with-slow` is the full tier. The receipt records the
tier and the exact command.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

RECEIPT_KIND = "ember-local-verify@1"
LOG_DIR_NAME = ".local-test-logs"
DEFAULT_REQUIRED = ("3.11", "3.12")
PYTEST_CMD = ("-m", "pytest", "-q")
SLOW_FLAG = "--run-slow"
STRIPPED_ENV = ("PYTEST_ADDOPTS", "PYTEST_PLUGINS")
ZERO_SHA = "0" * 40
EXIT_SOURCE_CHANGED = 3
EXIT_ENV_FAILED = 4
EXIT_VERDICT_FAILED = 5  # pytest exited 0 but the verdict failed (no summary, deselection, wrong import path)

_SUMMARY = re.compile(r"^=*\s*(?P<body>(?:\d+ \w+(?:, )?)+)(?: in [\d.]+s.*)?\s*=*$")
_COUNT = re.compile(r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed|warnings?|deselected)")


# ----- pure helpers (unit tested) ------------------------------------------------------------------------------
def parse_summary(lines: Sequence[str]) -> dict[str, int] | None:
    """Counts from pytest's final summary line (`-q` form), or None when no summary line exists."""
    for line in reversed(lines):
        s = line.strip()
        if not s or " in " not in s and not s.startswith("="):
            continue
        m = _SUMMARY.match(s)
        if not m:
            continue
        counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0, "xfailed": 0, "xpassed": 0,
                  "warnings": 0, "deselected": 0}
        found = False
        for n, kind in _COUNT.findall(m.group("body")):
            key = {"error": "errors", "warning": "warnings"}.get(kind, kind)
            counts[key] = int(n)
            found = True
        if found:
            return counts
    return None


def verdict(exit_code: int, counts: Mapping[str, int] | None, unchanged: bool, import_ok: bool) -> tuple[str, list[str]]:
    """PASS only for a clean pytest exit with a parsed summary, zero failures/errors, no deselection and no
    source change. Returns (status, reasons)."""
    reasons: list[str] = []
    if not import_ok:
        reasons.append("ember_v1 was not imported from this checkout's src/")
    if exit_code != 0:
        reasons.append(f"pytest exit code {exit_code}")
    if counts is None:
        reasons.append("no pytest summary line found")
    else:
        if counts["failed"] or counts["errors"]:
            reasons.append(f"{counts['failed']} failed, {counts['errors']} errors")
        if counts["deselected"]:
            reasons.append(f"{counts['deselected']} tests deselected")
        if counts["passed"] == 0:
            reasons.append("no test passed")
    if not unchanged:
        reasons.append("the source tree changed during the test run")
    if not unchanged:
        return "SOURCE_CHANGED", reasons
    return ("PASS", []) if not reasons else ("FAIL", reasons)


def missing_for_push(push_lines: Iterable[str], receipts: Sequence[Mapping[str, Any]],
                     required: Sequence[str], resolve=lambda sha: sha) -> list[str]:
    """Problems that block a push (empty = allowed). `push_lines` are git pre-push stdin lines."""
    problems = []
    for line in push_lines:
        parts = line.split()
        if len(parts) != 4:
            continue
        local_ref, local_sha, _remote_ref, _remote_sha = parts
        if local_sha == ZERO_SHA:
            continue  # branch deletion pushes no code
        sha = resolve(local_sha)
        ok = {r.get("python_minor") for r in receipts
              if r.get("kind") == RECEIPT_KIND and r.get("status") == "PASS" and r.get("head") == sha
              and r.get("tree_clean") is True and r.get("log_verified", True) is True}
        miss = [v for v in required if v not in ok]
        if miss:
            problems.append(f"{local_ref} {sha[:12]}: no PASS receipt for Python {', '.join(miss)}")
    return problems


# ----- git and environment -------------------------------------------------------------------------------------
def _git(root: Path, *args: str, binary: bool = False) -> Any:
    r = subprocess.run(["git", "-C", str(root), *args], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.decode('utf-8', 'replace').strip()}")
    return r.stdout if binary else r.stdout.decode("utf-8", "replace").strip()


def repo_root(start: Path | None = None) -> Path:
    return Path(_git(start or Path.cwd(), "rev-parse", "--show-toplevel"))


def receipts_dir(root: Path) -> Path:
    common = Path(_git(root, "rev-parse", "--git-common-dir"))
    if not common.is_absolute():
        common = (root / common).resolve()
    return common / "ember-local-verify" / "receipts"


def tree_state(root: Path) -> dict[str, Any]:
    """HEAD plus a digest of every tracked change and untracked (not ignored) file."""
    head = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain=v1", "-uall")
    h = hashlib.sha256()
    h.update(head.encode())
    h.update(status.encode())
    h.update(_git(root, "diff", "HEAD", "--binary", binary=True))
    for line in status.splitlines():
        if line.startswith("?? "):
            p = root / line[3:].strip('"')
            if p.is_file():
                h.update(p.read_bytes())
    return {"head": head, "clean": status == "", "digest": "sha256:" + h.hexdigest(),
            "status_lines": len(status.splitlines()) if status else 0}


def _env_for(root: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in STRIPPED_ENV}
    src = str(root / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    return env


_PROBE = ("import json, platform, sys, pytest, ember_v1; print(json.dumps({'version': platform.python_version(), "
          "'implementation': platform.python_implementation(), 'executable': sys.executable, "
          "'pytest': pytest.__version__, 'ember_v1_file': ember_v1.__file__}))")


def probe(python: str, root: Path) -> dict[str, Any]:
    r = subprocess.run([python, "-c", _PROBE], capture_output=True, env=_env_for(root), cwd=root)
    if r.returncode != 0:
        raise RuntimeError(f"{python}: cannot import pytest/ember_v1 "
                           f"(install with `python -m pip install -e \".[test]\"`): "
                           f"{r.stderr.decode('utf-8', 'replace').strip()[-400:]}")
    info = json.loads(r.stdout.decode("utf-8").strip().splitlines()[-1])
    src = os.path.normcase(str((root / "src").resolve()))
    info["import_ok"] = os.path.normcase(str(Path(info["ember_v1_file"]).resolve())).startswith(src + os.sep)
    return info


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def pytest_args(with_slow: bool = False) -> list[str]:
    return [*PYTEST_CMD, *([SLOW_FLAG] if with_slow else [])]


def run_one(python: str, root: Path, with_slow: bool = False) -> dict[str, Any]:
    before = tree_state(root)
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        info = probe(python, root)
    except RuntimeError as e:
        print(f"[verify] ENV FAILED: {e}", file=sys.stderr)
        return {"status": "ENV_FAILED", "python": python, "reasons": [str(e)], "exit_code": EXIT_ENV_FAILED}
    minor = ".".join(info["version"].split(".")[:2])
    log_dir = root / LOG_DIR_NAME
    log_dir.mkdir(exist_ok=True)
    log = log_dir / f"{stamp}-{before['head'][:12]}-py{minor}.log"
    cmd = [python, *pytest_args(with_slow)]
    header = [f"# ember-local-verify run {stamp}", f"# command: {' '.join(cmd)}", f"# cwd: {root}",
              f"# head: {before['head']}  tree_clean: {before['clean']}  tree_digest: {before['digest']}",
              f"# python: {info['version']} {info['implementation']} {info['executable']}",
              f"# pytest: {info['pytest']}  ember_v1: {info['ember_v1_file']}",
              f"# platform: {platform.platform()}", ""]
    print(f"[verify] Python {info['version']} -> {log}", flush=True)
    started = _dt.datetime.now(_dt.timezone.utc)
    lines: list[str] = []
    with log.open("w", encoding="utf-8", newline="\n", buffering=1) as out:  # line-buffered: a crash keeps output
        out.write("\n".join(header) + "\n")
        proc = subprocess.Popen(cmd, cwd=root, env=_env_for(root), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        assert proc.stdout is not None
        for raw in proc.stdout:
            text = raw.decode("utf-8", "replace").rstrip("\r\n")
            lines.append(text)
            out.write(text + "\n")
            print(text, flush=True)
        code = proc.wait()
        out.write(f"\n# exit_code: {code}\n")
    seconds = round((_dt.datetime.now(_dt.timezone.utc) - started).total_seconds(), 1)
    after = tree_state(root)
    unchanged = after["digest"] == before["digest"]
    counts = parse_summary(lines)
    status, reasons = verdict(code, counts, unchanged, info["import_ok"])
    receipt = {"kind": RECEIPT_KIND, "status": status, "reasons": reasons, "head": before["head"],
               "tree_clean": before["clean"] and after["clean"], "tree_digest": before["digest"],
               "tree_unchanged": unchanged, "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
               "python_minor": minor, "python": info, "platform": platform.platform(),
               "tier": "full" if with_slow else "default",
               "command": " ".join(["python", *pytest_args(with_slow)]), "stripped_env": [k for k in STRIPPED_ENV if k in os.environ],
               "exit_code": code, "counts": counts, "seconds": seconds, "started_utc": stamp,
               "log": str(log), "log_sha256": _sha256_file(log)}
    rdir = receipts_dir(root)
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / f"{before['head']}-py{minor}-{stamp}.json").write_text(json.dumps(receipt, indent=2) + "\n",
                                                                  encoding="utf-8")
    (log_dir / f"{stamp}-{before['head'][:12]}-py{minor}.json").write_text(json.dumps(receipt, indent=2) + "\n",
                                                                         encoding="utf-8")
    # Preserve pytest's own exit code; a zero exit that still failed the verdict gets a distinct nonzero code.
    receipt["exit_code_out"] = code or (0 if status == "PASS" else
                                        EXIT_SOURCE_CHANGED if status == "SOURCE_CHANGED" else EXIT_VERDICT_FAILED)
    return receipt


def _fmt_counts(c: Mapping[str, int] | None) -> str:
    if not c:
        return "no summary"
    return ", ".join(f"{c[k]} {k}" for k in ("passed", "failed", "errors", "skipped", "xfailed", "xpassed",
                                              "deselected", "warnings") if c.get(k))


def load_receipts(root: Path) -> list[dict[str, Any]]:
    out = []
    d = receipts_dir(root)
    if d.is_dir():
        for p in sorted(d.glob("*.json")):
            try:
                r = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            log = Path(r.get("log", ""))
            r["log_verified"] = log.is_file() and _sha256_file(log) == r.get("log_sha256")
            out.append(r)
    return out


# ----- commands ------------------------------------------------------------------------------------------------
def cmd_run(args: argparse.Namespace) -> int:
    root = repo_root()
    pythons = args.python or [p for p in os.environ.get("EMBER_VERIFY_PYTHONS", "").split(os.pathsep) if p] \
        or [sys.executable]
    state = tree_state(root)
    if not state["clean"]:
        print(f"[verify] WARNING: the working tree has {state['status_lines']} uncommitted change(s). "
              "A receipt for a dirty tree never allows a push.", file=sys.stderr)
    results = [run_one(py, root, with_slow=args.with_slow) for py in pythons]
    print("\n[verify] ===== summary =====")
    print(f"[verify] head {state['head']}  clean={state['clean']}")
    worst = 0
    for r in results:
        py = r.get("python", {})
        ver = py.get("version", "?") if isinstance(py, dict) else py
        print(f"[verify] Python {ver}: {r['status']}  exit={r.get('exit_code')}  "
              f"{_fmt_counts(r.get('counts'))}  {r.get('seconds', '')}s  log={r.get('log', '-')}")
        for why in r.get("reasons", []):
            print(f"[verify]   reason: {why}")
        worst = worst or r.get("exit_code_out", r.get("exit_code", 1))
    covered = sorted({r.get("python_minor") for r in results if r["status"] == "PASS"})
    missing = [v for v in DEFAULT_REQUIRED if v not in covered]
    if missing:
        print(f"[verify] NOT COVERED: Python {', '.join(missing)} (no PASS in this run)")
    print(f"[verify] OVERALL: {'PASS' if worst == 0 else 'FAIL'}")
    return worst


def cmd_check_push(args: argparse.Namespace) -> int:
    root = repo_root()
    required = [v for v in os.environ.get("EMBER_VERIFY_REQUIRE", ",".join(DEFAULT_REQUIRED)).split(",") if v]
    lines = sys.stdin.read().splitlines()

    def resolve(sha: str) -> str:
        return _git(root, "rev-parse", f"{sha}^{{commit}}")

    problems = missing_for_push(lines, load_receipts(root), required, resolve)
    if problems:
        print("pre-push: REFUSED. Full local suite evidence is missing (AGENTS.md):", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print("Run on a clean tree:  python scripts/verify_local.py run --python <py3.11> --python <py3.12>",
              file=sys.stderr)
        return 1
    return 0


HOOK = """#!/bin/sh
# Installed by scripts/verify_local.py install-hook (EMBER v1 local-testing policy, AGENTS.md).
root=$(git rev-parse --show-toplevel) || exit 1
script="$root/scripts/verify_local.py"
if [ ! -f "$script" ]; then
  echo "pre-push: $script is missing on this branch; merge the local-testing policy first. Push refused." >&2
  exit 1
fi
PY="${EMBER_VERIFY_PYTHON:-python}"
command -v "$PY" >/dev/null 2>&1 || PY=python3
exec "$PY" "$script" check-push "$@"
"""


def cmd_install_hook(args: argparse.Namespace) -> int:
    root = repo_root()
    try:
        custom = _git(root, "config", "--get", "core.hooksPath")
    except RuntimeError:
        custom = ""
    if custom:
        print(f"core.hooksPath is set ({custom}); copy scripts/git-hooks/pre-push there yourself.", file=sys.stderr)
        return 1
    common = Path(_git(root, "rev-parse", "--git-common-dir"))
    hook = (common if common.is_absolute() else root / common) / "hooks" / "pre-push"
    if hook.exists() and hook.read_text(encoding="utf-8", errors="replace") != HOOK and not args.force:
        print(f"{hook} exists with other content; rerun with --force to replace it.", file=sys.stderr)
        return 1
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text(HOOK, encoding="utf-8", newline="\n")
    hook.chmod(0o755)
    print(f"installed {hook}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    root = repo_root()
    sha = _git(root, "rev-parse", f"{args.rev}^{{commit}}")
    rows = [r for r in load_receipts(root) if r.get("head") == sha]
    if not rows:
        print(f"no receipts for {sha}")
        return 1
    for r in rows:
        print(f"{r['started_utc']}  py{r['python_minor']}  {r['status']}  clean={r['tree_clean']}  "
              f"log_ok={r['log_verified']}  {_fmt_counts(r.get('counts'))}  {r['log']}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    rn = sub.add_parser("run", help="run the complete suite and write receipts")
    rn.add_argument("--python", action="append", help="interpreter to test with (repeatable); default: "
                    "$EMBER_VERIFY_PYTHONS (os.pathsep-separated) or the current interpreter")
    rn.add_argument("--with-slow", action="store_true", help="full tier: also run tests marked slow")
    cp = sub.add_parser("check-push", help="pre-push gate (reads git pre-push lines on stdin)")
    cp.add_argument("hook_args", nargs="*", help="remote name and URL passed by git (unused)")
    ih = sub.add_parser("install-hook", help="install the pre-push hook")
    ih.add_argument("--force", action="store_true")
    st = sub.add_parser("status", help="list receipts for a commit")
    st.add_argument("rev", nargs="?", default="HEAD")
    a = ap.parse_args(argv)
    return {"run": cmd_run, "check-push": cmd_check_push, "install-hook": cmd_install_hook,
            "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
