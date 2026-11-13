"""P1-17 Phase 8: SFT example files for the Qwen3.5-2B student from the VALID p1-17 dataset (no training).

Input: a `train_dataset` output directory (status VALID). Checks, in order (the output directory appears only
when every check passes; a failure during streaming leaves `<out>.partial` for inspection, never reused):
  1. `dataset.json` hash recomputes; status VALID; protocol 1.0.0; every record file sha256 equals the manifest.
  2. Held-out isolation: no record family, lineage, scenario or geometry core of the DEV or calibration
     manifests (only their ID fields are read; no DEV scenario is generated or run).
Then, streaming one record file at a time:
  - SFT examples = selected, weighted class-A records through `multiteacher_dataset.model_example` only
    (observation, whole joint map, weight). Teacher names, provenance and outcomes never pass.
  - Family-isolated internal validation: the `val_families` TRAIN families with the lowest
    sha256(salt|family_id) are the validation split; a family is never in both splits.
  - Rows: `student_ingest.id_chunks(observation, k)` + `student_objective.rows_for` (no reservations,
    k = 100 per the student-train draft): every active person is the target of exactly one row (the loss mask);
    rows are re-checked, and the rendered state length is measured. Token counts need the pinned tokenizer,
    which is not on this machine: they are measured on the training host before any epoch is fixed.
Output (write-once, outside every checkout): `train.jsonl`, `val.jsonl` (model view + example_id),
`examples_index.jsonl` (example_id -> split, family, scenario, people; audit only, never a model input),
`sft_manifest.json`. Status `PREPARED_NOT_AUTHORIZED_FOR_TRAINING`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import multiteacher_dataset as ds
from . import student_ingest as si
from . import student_objective as so
from .sim.canonical import digest

VERSION = "p1-17-sft-prep@1.0.0"
REPO = Path(__file__).resolve().parents[2]
HELDOUT_MANIFESTS = ("data/splits/p1-02-v1/dev.json", "data/splits/p1-02-v1/calibration.json")
PLAN_FILE = "data/collection/p1-17-train-full@1/plan.json"
SALT = "p1-17-sft-val@1"
STATUS = "PREPARED_NOT_AUTHORIZED_FOR_TRAINING"


class PrepBlocked(RuntimeError):
    """A check failed; nothing was written."""


def _read(p: Path) -> Any:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _sha256(p: Path) -> str:
    return "sha256:" + hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _inside(child: Path, parent: Path) -> bool:
    try:
        Path(os.path.abspath(child)).relative_to(Path(os.path.abspath(parent)))
        return True
    except ValueError:
        return False


def heldout_ids(repo: Path = REPO) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {"family_id": set(), "lineage_id": set(), "scenario_hash": set(), "core_hash": set()}
    for rel in HELDOUT_MANIFESTS:
        for e in _read(repo / rel)["scenarios"]:
            for k in out:
                out[k].add(e[k])
    return out


def val_families(families: Sequence[str], n: int, salt: str = SALT) -> list[str]:
    rank = sorted(families, key=lambda f: hashlib.sha256(f"{salt}|{f}".encode()).hexdigest())
    return sorted(rank[:n])


def check_dataset(ddir: Path) -> dict[str, Any]:
    d = _read(ddir / "dataset.json")
    problems = []
    if digest({k: v for k, v in d.items() if k != "dataset_hash"}) != d.get("dataset_hash"):
        problems.append("dataset.json hash does not recompute")
    if d.get("status") != "VALID" or d.get("n_problems") != 0:
        problems.append(f"dataset status {d.get('status')!r} is not VALID")
    if d.get("protocol_version") != ds.load_protocol()["version"]:
        problems.append("dataset protocol version is not the current frozen protocol")
    for sid, h in sorted((d.get("record_files") or {}).items()):
        p = ddir / "records" / f"{sid}.jsonl"
        if not p.exists() or _sha256(p) != h:
            problems.append(f"{sid}: record file missing or changed")
    if problems:
        raise PrepBlocked("; ".join(problems[:10]))
    return d


def _row_stats(ex: Mapping[str, Any], k: int) -> dict[str, int]:
    obs = ex["observation"]
    chunks = si.id_chunks(obs, k)
    rows = so.rows_for(obs, ex["actions"], chunks)
    active = sorted(p["id"] for p in obs["people"] if p["status"] == "active")
    targeted = sorted(i for r in rows for i in r["targets"])
    if targeted != active or any(set(r["targets"]) != set(r["ids"]) for r in rows):
        raise PrepBlocked(f"{ex['example_id'][:19]}: loss mask does not cover every active person exactly once")
    return {"people": len(active), "rows": len(rows), "state_chars": max(len(r["state"]) for r in rows)}


def prepare(*, dataset_dir: Path, out: Path, n_val_families: int = 6, k: int = 100, repo: Path = REPO,
            plan_path: Path | None = None) -> dict[str, Any]:
    if _inside(out, repo) or _inside(out, dataset_dir):
        raise PrepBlocked(f"output {out} must be outside the checkout and the dataset")
    if out.exists():
        raise PrepBlocked(f"output {out} exists: write-once")
    d = check_dataset(dataset_dir)
    plan = _read(plan_path or repo / PLAN_FILE)
    core_of = {s["scenario_hash"]: s.get("core_hash") for s in plan["scenarios"]}
    if not all(core_of.values()):
        raise PrepBlocked("every plan scenario needs its geometry core hash for the held-out check")
    held = heldout_ids(repo)
    fams = sorted(d["family_scenarios_with_A"])
    val = set(val_families(fams, n_val_families))
    files = sorted((dataset_dir / "records").glob("*.jsonl"))
    leaks: Counter = Counter()
    stats = {"train": Counter(), "val": Counter()}
    rows_hist = {"train": Counter(), "val": Counter()}
    max_chars = {"train": 0, "val": 0}
    weight = {"train": 0.0, "val": 0.0}
    split_fams: dict[str, set] = {"train": set(), "val": set()}
    out_tmp = out.with_name(out.name + ".partial")
    if out_tmp.exists():
        raise PrepBlocked(f"{out_tmp} exists from an earlier interrupted run; inspect it, never overwrite")
    out_tmp.mkdir(parents=True)
    fh = {s: open(out_tmp / f"{s}.jsonl", "x", encoding="utf-8", newline="\n") for s in ("train", "val")}
    idx = open(out_tmp / "examples_index.jsonl", "x", encoding="utf-8", newline="\n")
    try:
        for p in files:
            for line in p.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                for key, val_ in (("family_id", r["family_id"]), ("lineage_id", r["lineage_id"]),
                                  ("scenario_hash", r["scenario_hash"]),
                                  ("core_hash", core_of[r["scenario_hash"]])):
                    if val_ in held[key]:
                        leaks[key] += 1
                if not (r["class"] == "A" and r["selection"]["selected"] and r["weight"]):
                    continue
                ex = ds.model_example(r)
                ex = {"example_id": si.example_id(ex), **ex}
                split = "val" if r["family_id"] in val else "train"
                st = _row_stats(ex, k)
                fh[split].write(json.dumps(ex, sort_keys=True, separators=(",", ":")) + "\n")
                idx.write(json.dumps({"example_id": ex["example_id"], "split": split, "family_id": r["family_id"],
                                      "scenario_hash": r["scenario_hash"], "step": r["step"], **st},
                                     sort_keys=True) + "\n")
                stats[split]["examples"] += 1
                stats[split]["rows"] += st["rows"]
                rows_hist[split][st["rows"]] += 1
                max_chars[split] = max(max_chars[split], st["state_chars"])
                weight[split] += ex["weight"]
                split_fams[split].add(r["family_id"])
    finally:
        for f in (*fh.values(), idx):
            f.flush()
            os.fsync(f.fileno())
            f.close()
    if leaks:
        raise PrepBlocked(f"held-out identifiers in the dataset: {dict(leaks)} (partial output kept at {out_tmp})")
    if split_fams["train"] & split_fams["val"]:
        raise PrepBlocked("a family is in both splits")
    body = {"version": VERSION, "status": STATUS, "dataset_hash": d["dataset_hash"],
            "protocol_version": d["protocol_version"], "salt": SALT, "k": k, "reservations": False,
            "val_families": sorted(val), "train_families": len(split_fams["train"]),
            "heldout_manifests": {rel: _read(repo / rel)["manifest_hash"] for rel in HELDOUT_MANIFESTS},
            "heldout_overlap": {k_: 0 for k_ in held},
            "splits": {s: {"examples": stats[s]["examples"], "rows": stats[s]["rows"],
                           "rows_per_example": dict(sorted(rows_hist[s].items())),
                           "max_state_chars": max_chars[s], "weight_total": round(weight[s], 9),
                           "families": len(split_fams[s])} for s in ("train", "val")},
            "files": {n: _sha256(out_tmp / n) for n in ("train.jsonl", "val.jsonl", "examples_index.jsonl")},
            "model_view": "observation, actions (whole joint map), weight; example_id for bookkeeping",
            "weight_use": "student_ingest.sample_epoch draws by weight; every row has loss_weight 1 (no double use)",
            "token_budget": "NOT MEASURED HERE: the pinned Qwen3.5-2B tokenizer is not on this machine; count tokens "
                            "per row on the training host (token_budget.max_chunk) and fail closed on any row over "
                            "the 16,384 bound before fixing token_budget_per_epoch",
            "training": "not authorized: needs the frozen student-train protocol and an owner authorization"}
    (out_tmp / "sft_manifest.json").write_text(json.dumps({**body, "manifest_hash": digest(body)}, indent=1,
                                                          sort_keys=True), encoding="utf-8")
    os.replace(out_tmp, out)
    return {**body, "manifest_hash": digest(body)}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ember_v1.sft_prep", description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-families", type=int, default=6)
    ap.add_argument("--k", type=int, default=100)
    a = ap.parse_args(argv)
    try:
        res = prepare(dataset_dir=Path(a.dataset), out=Path(a.out), n_val_families=a.val_families, k=a.k)
    except PrepBlocked as e:
        print(json.dumps({"status": "BLOCKED", "reason": str(e)}))
        return 2
    print(json.dumps({"status": res["status"], "splits": res["splits"], "manifest_hash": res["manifest_hash"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
