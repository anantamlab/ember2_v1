"""Canonical JSON and SHA-256 digests for states, records and transitions."""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    """Sorted keys, no whitespace, ASCII only, no NaN or infinity."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()
