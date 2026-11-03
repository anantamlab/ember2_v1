"""Training-data source registry and zip ingestion (no extraction to disk).

Every source has a `quality` tier. Only `teacher_verified` traces (simulator-
replayed teacher output, HLD v3 section 7) are training-eligible by default.
`synthetic_fixture` sources (hand-written example labels) feed the pipeline
smoke split only, so made-up labels never count as teacher evidence.

Zip members are read in memory and parsed as JSON only; nothing is executed or
written outside the requested output file.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .contract import validate_actions, validate_observation

QUALITIES = ("teacher_verified", "teacher_unverified", "synthetic_fixture")
TRAINING_QUALITIES = frozenset({"teacher_verified"})
_TRACE_KEYS = {"observation", "actions", "group_id", "source"}
_MAX_MEMBER_BYTES = 50 * 2**20  # guard against oversized / zip-bomb members


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _traces(data: Any) -> Iterable[Mapping[str, Any]]:
    """A member holds one trace object, a list of them, or {"traces": [...]}."""
    if isinstance(data, Mapping) and isinstance(data.get("traces"), list):
        data = data["traces"]
    items = data if isinstance(data, list) else [data]
    for item in items:
        if isinstance(item, Mapping) and set(item) >= _TRACE_KEYS:
            yield item


def ingest_zip(zip_path: Path, *, source_id: str, quality: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate every teacher trace in a zip. Return (records, manifest entry).

    Fail closed: an invalid trace raises; it is never repaired or skipped.
    JSON members without trace keys (e.g. a bare observation) are counted only.
    """
    if quality not in QUALITIES:
        raise ValueError(f"quality must be one of {QUALITIES}")
    if not source_id or "/" in source_id:
        raise ValueError("source_id must be nonempty and contain no '/'")
    records: list[dict[str, Any]] = []
    other_json = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in sorted(zf.infolist(), key=lambda i: i.filename):
            if info.is_dir() or not info.filename.lower().endswith((".json", ".jsonl")):
                continue
            if info.file_size > _MAX_MEMBER_BYTES:
                raise ValueError(f"member too large: {info.filename}")
            raw = zf.read(info)
            text = raw.decode("utf-8")
            docs = ([json.loads(line) for line in text.splitlines() if line.strip()]
                    if info.filename.lower().endswith(".jsonl") else [json.loads(text)])
            found = 0
            for doc in docs:
                for i, trace in enumerate(_traces(doc)):
                    obs = validate_observation(trace["observation"])
                    actions = validate_actions(obs, trace["actions"])
                    if not isinstance(trace["group_id"], str) or not trace["group_id"]:
                        raise ValueError(f"{info.filename}: nonempty group_id required")
                    if not isinstance(trace["source"], str) or not trace["source"]:
                        raise ValueError(f"{info.filename}: teacher source/version required")
                    records.append({
                        "source_id": source_id, "quality": quality,
                        "member": info.filename, "member_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
                        "index": i,
                        # Prefix keeps lineage unique across sources for group splits.
                        "group_id": f"{source_id}/{trace['group_id']}",
                        "teacher": trace["source"], "observation": obs, "actions": actions,
                    })
                    found += 1
            other_json += found == 0
    manifest = {
        "source_id": source_id, "quality": quality, "path": str(zip_path.name),
        "sha256": sha256_file(zip_path), "traces": len(records),
        "decisions": sum(len(r["actions"]) for r in records),
        "groups": sorted({r["group_id"] for r in records}),
        "non_trace_json_members": other_json,
        "training_eligible": quality in TRAINING_QUALITIES,
    }
    return records, manifest


def training_records(records: Iterable[Mapping[str, Any]], *, allow_fixtures: bool = False) -> list[Mapping[str, Any]]:
    """Filter to training-eligible records. `allow_fixtures` is for pipeline smoke only."""
    allowed = TRAINING_QUALITIES | ({"synthetic_fixture"} if allow_fixtures else set())
    return [r for r in records if r["quality"] in allowed]
