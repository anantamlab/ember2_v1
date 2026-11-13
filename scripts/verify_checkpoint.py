#!/usr/bin/env python3
"""Hash a local checkpoint copy against its tracked identity record (CPU only, read only).

    python scripts/verify_checkpoint.py --list
    python scripts/verify_checkpoint.py <checkpoint-id> <local-checkpoint-dir>

The registry is data/rl/checkpoint_registry.json. Each workstation checkpoint names a tracked
identity record and the JSON keys of its digest and file list. The script checks that every
listed file exists with the recorded size and SHA-256. It never writes into the checkpoint.
It does not load the model; the actor parameter digest needs the GPU replay environment.
Hugging Face checkpoints use `python -m ember_v1.hfstore verify|fetch` instead.
Exit code: 0 PASS, 1 FAIL, 2 usage or registry error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "data" / "rl" / "checkpoint_registry.json"


def load_registry(path: Path = REGISTRY) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def entry(registry: dict[str, Any], checkpoint_id: str) -> dict[str, Any]:
    for item in registry["checkpoints"]:
        if item["id"] == checkpoint_id:
            return item
    raise KeyError(f"unknown checkpoint id {checkpoint_id!r}")


def dig(obj: Any, keys: list[str]) -> Any:
    for key in keys:
        obj = obj[key]
    return obj


def identity(item: dict[str, Any], repo: Path = REPO) -> tuple[str, list[dict[str, Any]]]:
    """The recorded actor digest and file list from the tracked identity record."""
    rec = item["identity_record"]
    data = json.loads((repo / rec["file"]).read_text(encoding="utf-8"))
    return dig(data, rec["digest_key"]), dig(data, rec["files_key"])


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(item: dict[str, Any], root: Path, repo: Path = REPO) -> dict[str, Any]:
    if item["storage"]["kind"] != "workstation":
        raise ValueError(f"{item['id']} is stored on {item['storage']['kind']}; use: {item['verify']}")
    digest, files = identity(item, repo)
    if digest != item["actor_digest"]:
        raise ValueError(f"registry digest {item['actor_digest']} != identity record {digest}")
    problems = []
    for f in files:
        p = root / f["path"]
        if not p.is_file():
            problems.append(f"missing {f['path']}")
        elif p.stat().st_size != f["size"]:
            problems.append(f"size {f['path']}: {p.stat().st_size} != {f['size']}")
        elif sha256_file(p) != f["sha256"]:
            problems.append(f"sha256 {f['path']}")
    return {"id": item["id"], "actor_digest": digest, "root": str(root), "files": len(files),
            "problems": problems, "verdict": "FAIL" if problems else "PASS"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint_id", nargs="?")
    ap.add_argument("root", nargs="?", type=Path)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--registry", type=Path, default=REGISTRY)
    a = ap.parse_args(argv)
    reg = load_registry(a.registry)
    if a.list:
        for item in reg["checkpoints"]:
            ident = item.get("actor_digest") or item.get("parameter_digest")
            print(f"{item['id']:<12} parent={item['parent'] or '-':<10} {item['availability']['status']:<34} {ident}")
        return 0
    if not a.checkpoint_id or not a.root:
        ap.error("give <checkpoint-id> <local-checkpoint-dir>, or --list")
    try:
        result = verify(entry(reg, a.checkpoint_id), a.root)
    except (KeyError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=1))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
