"""Trusted-source helpers shared by the S1 gate (#10) and teacher selection (#25).

The only trust root kind is `git`: a document is trusted when it is read from
a commit (`git show <commit>:<path>`) that is an ancestor of the trusted ref
(default `origin/main`, changed only through reviewed pull requests). Any
other kind is unknown and fails closed.

This is a hash-and-history binding, not authentication. A party that can
write to the trusted ref or to the local clone can create a trusted-looking
document. See docs/P1_TEACHER_PROVENANCE.md, "Trust boundary", for when
signatures become necessary.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from .gen.spec import GENERATOR_REVISION
from .gen.splits import MANIFEST_VERSION
from .sim.canonical import digest

TRUST_ROOT_KINDS = ("git",)
_HEX40 = re.compile(r"^[0-9a-f]{40}$")


def sha256_lf(raw: bytes) -> str:
    """sha256 of LF-normalized bytes (stable across Windows and Linux checkouts)."""
    return "sha256:" + hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


def _git(repo: str | Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, timeout=60)


def is_ancestor(repo: str | Path, older: str, newer: str) -> bool:
    """True if commit `older` is `newer` or an ancestor of it (full 40-hex ids or refs)."""
    if not isinstance(older, str) or not isinstance(newer, str) or not older or not newer:
        return False
    try:
        return _git(repo, "merge-base", "--is-ancestor", older, newer).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def load_trusted_json(repo: str | Path, ref: Any, *, trusted_ref: str = "origin/main") -> tuple[Any, dict[str, Any]]:
    """JSON document at {"kind": "git", "commit", "path"} in `repo`, if the commit is in `trusted_ref`."""
    if not isinstance(ref, Mapping) or set(ref) != {"kind", "commit", "path"}:
        raise ValueError("trust ref must be exactly {kind, commit, path}")
    if ref["kind"] not in TRUST_ROOT_KINDS:
        raise ValueError(f"unknown trust root kind {ref['kind']!r}")
    commit, path = ref["commit"], ref["path"]
    if not isinstance(commit, str) or not _HEX40.match(commit):
        raise ValueError("trust ref commit must be a full 40-hex commit id")
    pp = PurePosixPath(path) if isinstance(path, str) else None
    if pp is None or not path or pp.is_absolute() or ".." in pp.parts or "\\" in path or ":" in path:
        raise ValueError("trust ref path must be a relative repository path")
    if not is_ancestor(repo, commit, trusted_ref):
        raise ValueError(f"commit {commit[:12]} is not in the history of trusted ref {trusted_ref!r}")
    shown = _git(repo, "show", f"{commit}:{path}")
    if shown.returncode != 0:
        raise ValueError(f"{path} does not exist at commit {commit[:12]}")
    return json.loads(shown.stdout.decode("utf-8")), {
        "kind": "git", "commit": commit, "path": path, "trusted_ref": trusted_ref, "sha256": sha256_lf(shown.stdout)}


def check_manifest(manifest: Any) -> None:
    """A split manifest of the current generator whose hash matches its content. Raise ValueError."""
    if not isinstance(manifest, Mapping) or manifest.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError("unsupported manifest version")
    body = {k: v for k, v in manifest.items() if k != "manifest_hash"}
    if digest(body) != manifest.get("manifest_hash"):
        raise ValueError("manifest hash does not match its content")
    if manifest.get("generator_revision") != GENERATOR_REVISION:
        raise ValueError("manifest generator revision differs from the current generator")
