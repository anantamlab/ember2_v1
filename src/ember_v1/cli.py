"""Phase-0 offline CLI; no GPU, no model calls and no benchmark execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .dataset import QUESTION_STYLES, build_decision_row
from .encoding import encode_observation
from .contract import validate_observation
from .sources import QUALITIES, ingest_zip


def _read(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(prog="ember-v1")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("encode", help="encode model-visible Observation.to_dict() JSON")
    p.add_argument("observation")
    p.add_argument("--mode", choices=("ascii", "coords"), default="ascii")
    p.add_argument("--local-neighbors", action="store_true")
    p = sub.add_parser("prepare", help="create typed-decision training row from a teacher trace")
    p.add_argument("trace", help="JSON with observation, actions, group_id, source")
    p.add_argument("--mode", choices=("ascii", "coords"), default="ascii")
    p.add_argument("--local-neighbors", action="store_true")
    p.add_argument("--question-style", choices=QUESTION_STYLES, default="verbose")
    p = sub.add_parser("profile", help="measure proxy payload and exact tokenizer counts where available")
    p.add_argument("observation")
    p.add_argument("--tokenizer", help="installed/local Qwen tokenizer path or permitted Hugging Face ID")
    p.add_argument("--max-sequence", type=int, default=2048)
    p = sub.add_parser("ingest", help="validate teacher traces in a zip and register the source")
    p.add_argument("zip")
    p.add_argument("--source-id", required=True)
    p.add_argument("--quality", choices=QUALITIES, required=True)
    p.add_argument("--out", required=True, help="output JSONL of validated trace records")
    p.add_argument("--registry", default="data/sources.json")
    p = sub.add_parser("gen-splits", help="build, audit and write immutable scenario split manifests (CPU only)")
    p.add_argument("--plan", help="plan JSON (default: the built-in p1-02-v1 plan)")
    p.add_argument("--out", required=True, help="public split directory (index, train, dev, calibration)")
    p.add_argument("--sealed-dir", help="directory for the locked test manifest; must be outside --out")
    p = sub.add_parser("verify-splits", help="rebuild a split index and compare every file with the rebuild")
    p.add_argument("--dir", required=True, help="public split directory with index.json")
    p.add_argument("--sealed-dir", help="also verify the locked test manifest in this directory")
    args = parser.parse_args()
    if args.cmd in ("gen-splits", "verify-splits"):
        from .gen.splits import SplitPlan, verify_split_files, write_split_files
        if args.cmd == "gen-splits":
            plan = SplitPlan.from_dict(_read(args.plan)) if args.plan else SplitPlan.default()
            idx = write_split_files(plan, args.out, args.sealed_dir)
            print(json.dumps({"plan_id": idx["plan"]["plan_id"], "index_hash": idx["index_hash"],
                              "splits": idx["splits"], "audit_passed": idx["audit"]["passed"]}, indent=2, sort_keys=True))
        else:
            print(json.dumps(verify_split_files(args.dir, args.sealed_dir), indent=2, sort_keys=True))
        return
    if args.cmd == "ingest":
        records, entry = ingest_zip(Path(args.zip), source_id=args.source_id, quality=args.quality)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records), encoding="utf-8")
        reg_path = Path(args.registry)
        registry = json.loads(reg_path.read_text(encoding="utf-8")) if reg_path.exists() else {"sources": []}
        registry["sources"] = [s for s in registry["sources"] if s["source_id"] != entry["source_id"]] + [entry]
        reg_path.parent.mkdir(parents=True, exist_ok=True)
        reg_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(entry, sort_keys=True))
    elif args.cmd == "encode":
        print(encode_observation(_read(args.observation), mode=args.mode, local_neighbors=args.local_neighbors))
    elif args.cmd == "prepare":
        trace = _read(args.trace)
        row = build_decision_row(trace["observation"], trace["actions"], group_id=trace["group_id"], source=trace["source"], encoding=args.mode, local_neighbors=args.local_neighbors, question_style=args.question_style)
        print(json.dumps(row, sort_keys=True))
    elif args.cmd == "profile":
        obs = validate_observation(_read(args.observation))
        profiles: dict[str, Any] = {"active": sum(p["status"] == "active" for p in obs["people"]), "max_sequence": args.max_sequence, "warning": "Counts measure a serialized payload proxy, not Unsloth's exact encoded record; gate requires FastDecisionModel.build_dataset and actual encoder lengths."}
        if args.tokenizer:
            try:
                from transformers import AutoTokenizer  # type: ignore
            except ImportError as exc:
                raise SystemExit("Install the optional probe dependency: pip install -e '.[probe]'") from exc
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        else:
            tokenizer = None
        for mode, local, style in ((m, n, q) for m in ("ascii", "coords") for n in (False, True) for q in QUESTION_STYLES):
            row = build_decision_row(obs, {p["id"]: "stay" for p in obs["people"] if p["status"] == "active"}, group_id="profile-only", source="synthetic-profile", encoding=mode, local_neighbors=local, question_style=style)
            payload = json.dumps({"state": row["state"], "questions": row["questions"]}, separators=(",", ":"))
            entry = {"characters": len(payload), "utf8_bytes": len(payload.encode("utf-8"))}
            if tokenizer is not None:
                entry["proxy_tokens"] = len(tokenizer.encode(payload, add_special_tokens=False))
                entry["proxy_exceeds_context"] = entry["proxy_tokens"] > args.max_sequence
            profiles[f"{mode}{'+neighbors' if local else ''}/{style}"] = entry
        print(json.dumps(profiles, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
