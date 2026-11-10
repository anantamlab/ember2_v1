"""P1-18 Stage 4: dataset v2 = P1-17 SFT data + whole joint demonstrations from the p1-18-expand@1 TRAIN partition.

Records (`build`): the P1-17 builder's two passes (`train_dataset`) on every complete, verified expansion TRAIN
group, under protocol `p1-18-multiteacher-dataset` 1.1.0 (= P1-14 1.0.0 with the expansion scope allowed):
class A = complete joint maps of the group-best teachers (outcome), weight per family unit, the same selection,
coherence and audit checks. Counterfactual classes C/D are not built here (Stage 6 evaluates them separately).
Incomplete or unverified groups give no record and are listed.

SFT v2 (`sft`): train = every P1-17 SFT train example (unchanged bytes) + every selected, weighted expansion class-A
example; val = the P1-17 SFT val split (same 6 families, so V1 and V2 imitation numbers compare). Expansion
val/diagnostic partitions never enter any SFT file. Weights are per family (protocol unit `family`), so the two
sources combine without re-normalization: each family carries one scenario budget per scenario-with-A / count.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import expand as E
from . import expand_collect as X
from . import multiteacher_dataset as ds
from . import student_ingest as si
from . import train_dataset as td
from .sim.canonical import digest

VERSION = "p1-18-expand-dataset@1.0.0"
PROTOCOL = "data/protocols/p1-18-multiteacher-dataset.json"
P17_SFT = "D:/Coding/ember-evidence/p1-17-sft@1"


class BuildBlocked(RuntimeError):
    pass


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return "sha256:" + h.hexdigest()


def _pass1(job: Mapping[str, Any]) -> dict[str, Any]:
    X.register_scope()
    sid = job["scenario_id"]
    try:
        protocol = ds.load_protocol(job["protocol_path"])
        g = json.loads(Path(job["group_path"]).read_text(encoding="utf-8"))
        if g.get("scope") != X.SCOPE or g.get("scenario_hash") != job["scenario_hash"]:
            raise BuildBlocked("group scope or scenario hash is not the manifest's")
        recs = ds.demonstrations(g, job["scenario"], protocol, family_id=job["family_id"], teacher_dev_similar=False)
        recs = ds.select_joint_targets(recs, protocol)
        problems = ds.check_targets_against_report(recs, g)
        Path(job["out"]).parent.mkdir(parents=True, exist_ok=True)
        Path(job["out"]).write_text(td._jsonl(recs), encoding="utf-8", newline="\n")
        sel = any(r["class"] == "A" and r["selection"]["selected"] for r in recs)
        outs = {n: p["outcome"] for n, p in g["policies"].items()}
        return {"scenario_id": sid, "ok": not problems, "problems": problems, "records": len(recs),
                "family_id": job["family_id"], "scenario_hash": job["scenario_hash"], "selected_A": sel,
                "outcomes": {n: {k: o[k] for k in ("casualties", "unresolved", "casualties_plus_unresolved")}
                             for n, o in outs.items()}}
    except Exception as e:  # noqa: BLE001 - reported, never hidden
        return {"scenario_id": sid, "ok": False, "problems": [f"{sid}: {type(e).__name__}: {e}"], "records": 0,
                "family_id": job["family_id"], "scenario_hash": job["scenario_hash"], "selected_A": False}


def _pass2(job: Mapping[str, Any]) -> dict[str, Any]:
    X.register_scope()
    return td._pass2(job)


def build(*, repo: Path, root: Path, out: Path, workers: int = 4,
          manifest: Mapping[str, Any] | None = None, scenarios: Sequence[tuple[Mapping[str, Any], Any]] | None = None
          ) -> dict[str, Any]:
    """Records of every complete, verified TRAIN-partition group under `root` (write-once `out`)."""
    t0 = time.time()
    if out.exists():
        raise BuildBlocked(f"{out} exists (write-once)")
    if td._inside(out, repo) or td._inside(out, root):
        raise BuildBlocked(f"output {out} must be outside the checkout and the collection root")
    protocol_path = repo / PROTOCOL
    protocol = ds.load_protocol(protocol_path)
    if X.SCOPE not in protocol["scope"]["allowed_now"] or protocol.get("status") != "FROZEN":
        raise BuildBlocked("dataset protocol does not allow the expansion scope")
    manifest = manifest or json.loads((repo / E.OUT_DIR / "manifest.json").read_text(encoding="utf-8"))
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    if plan["manifest_hash"] != manifest["manifest_hash"]:
        raise BuildBlocked("collection root is not this manifest's")
    train = [s for s in plan["scenarios"] if s["partition"] == "train"]
    if scenarios is None:
        scenarios = E.materialize(manifest, partitions=("train",))
    by_id = {e["scenario_id"]: s for e, s in scenarios}
    use, skipped = [], {}
    for s in train:
        sid = s["scenario_id"]
        rp, gp = root / "rows" / f"{sid}.json", root / "groups" / f"{sid}.json"
        if not rp.exists():
            skipped[sid] = "not_collected"
            continue
        row = json.loads(rp.read_text(encoding="utf-8"))
        if not (row.get("group_complete") and row.get("verified")):
            skipped[sid] = "incomplete_or_unverified"
            continue
        if _sha(gp) != row.get("file_sha256"):
            raise BuildBlocked(f"{sid}: group file differs from its row")
        use.append(s)
    staging = out / "staging"
    jobs1 = [{"scenario_id": s["scenario_id"], "group_path": str(root / "groups" / f"{s['scenario_id']}.json"),
              "scenario": by_id[s["scenario_id"]], "scenario_hash": s["scenario_hash"], "family_id": s["family_id"],
              "protocol_path": str(protocol_path), "out": str(staging / "unweighted" / f"{s['scenario_id']}.jsonl")}
             for s in use]
    out.mkdir(parents=True)
    r1 = sorted(td._map(_pass1, jobs1, workers), key=lambda x: x["scenario_id"])
    problems = [p for x in r1 for p in x["problems"]]
    fam_counts = dict(sorted(Counter(x["family_id"] for x in r1 if x["selected_A"]).items()))
    r2: list[dict[str, Any]] = []
    if not problems:
        (staging / "records").mkdir(parents=True, exist_ok=True)
        jobs2 = [{"scenario_id": x["scenario_id"], "protocol_path": str(protocol_path),
                  "src": str(staging / "unweighted" / f"{x['scenario_id']}.jsonl"),
                  "out": str(staging / "records" / f"{x['scenario_id']}.jsonl"), "family_scenarios": fam_counts}
                 for x in r1]
        r2 = sorted(td._map(_pass2, jobs2, workers), key=lambda x: x["scenario_id"])
        problems += [p for x in r2 for p in x["problems"]]
    summary = td.finish(protocol, r1, r2, problems, fam_counts)
    status = "VALID" if not summary["problems"] else "INVALID"
    if status == "VALID":
        os.replace(staging / "records", out / "records")
    best = Counter(t for x in r1 if x.get("outcomes") for t in td._best_teachers(x["outcomes"]))
    body = {"version": VERSION, "status": status, "scope": X.SCOPE, "protocol_id": protocol["protocol_id"],
            "protocol_version": protocol["version"], "protocol_sha256": _sha(protocol_path),
            "manifest_hash": manifest["manifest_hash"], "collection_plan_hash": plan["plan_hash"],
            "groups_used": len(r1), "groups_skipped": skipped, "family_scenarios_with_A": fam_counts,
            "best_teacher_counts": dict(sorted(best.items())),
            "record_files": {x["scenario_id"]: x["file_sha256"] for x in r2}, "wall_s": round(time.time() - t0, 1),
            **summary}
    doc = {**body, "dataset_hash": digest(body)}
    (out / "dataset.json").write_text(json.dumps(doc, indent=1, sort_keys=True), encoding="utf-8", newline="\n")
    return doc


def sft(*, dataset_dir: Path, out: Path, p17_sft: Path = Path(P17_SFT), k: int = 100,
        manifest: Mapping[str, Any] | None = None, repo: Path = Path(".")) -> dict[str, Any]:
    """SFT v2 directory: P1-17 train + val unchanged, plus expansion TRAIN class-A examples in train."""
    from . import sft_prep as sp
    from . import student_run as sr
    if out.exists():
        raise BuildBlocked(f"{out} exists (write-once)")
    d = json.loads((dataset_dir / "dataset.json").read_text(encoding="utf-8"))
    if d.get("status") != "VALID" or digest({k_: v for k_, v in d.items() if k_ != "dataset_hash"}) \
            != d.get("dataset_hash"):
        raise BuildBlocked("expansion dataset is not VALID or its hash does not recompute")
    old = sr.check_sft_dir(p17_sft)
    manifest = manifest or json.loads((repo / E.OUT_DIR / "manifest.json").read_text(encoding="utf-8"))
    part = {e["scenario_hash"]: e["partition"] for e in manifest["scenarios"]}
    val_fams = set(json.loads((p17_sft / "sft_manifest.json").read_text(encoding="utf-8"))["val_families"])
    tmp = out.with_name(out.name + ".partial")
    tmp.mkdir(parents=True)
    shutil.copyfile(p17_sft / "val.jsonl", tmp / "val.jsonl")
    shutil.copyfile(p17_sft / "train.jsonl", tmp / "train.jsonl")
    n_old = sum(1 for _ in si_lines(tmp / "train.jsonl"))
    stats, rows_hist, fams, weight, max_chars = Counter(), Counter(), set(), 0.0, 0
    with open(tmp / "train.jsonl", "a", encoding="utf-8", newline="\n") as fh:
        for sid, h in sorted(d["record_files"].items()):
            p = dataset_dir / "records" / f"{sid}.jsonl"
            if _sha(p) != h:
                raise BuildBlocked(f"{sid}: record file changed")
            for line in si_lines(p):
                r = json.loads(line)
                if part.get(r["scenario_hash"]) != "train":
                    raise BuildBlocked(f"{sid}: a record is not from the TRAIN partition")
                if r["family_id"] in val_fams:
                    raise BuildBlocked("an expansion family equals a P1-17 val family")
                if not (r["class"] == "A" and r["selection"]["selected"] and r["weight"]):
                    continue
                ex = ds.model_example(r)
                ex = {"example_id": si.example_id(ex), **ex}
                st = sp._row_stats(ex, k)
                fh.write(json.dumps(ex, sort_keys=True, separators=(",", ":")) + "\n")
                stats["examples"] += 1
                stats["rows"] += st["rows"]
                rows_hist[st["rows"]] += 1
                max_chars = max(max_chars, st["state_chars"])
                weight += ex["weight"]
                fams.add(r["family_id"])
    body = {"version": VERSION, "k": k, "reservations": False, "dataset_hash": d["dataset_hash"],
            "p17_sft_manifest_hash": old["manifest_hash"], "val_families": sorted(val_fams),
            "splits": {"train": {"examples_p17": n_old, "examples_expansion": stats["examples"],
                                 "examples": n_old + stats["examples"], "expansion_rows": stats["rows"],
                                 "expansion_rows_per_example": dict(sorted(rows_hist.items())),
                                 "expansion_families": len(fams), "expansion_weight_total": round(weight, 9),
                                 "expansion_max_state_chars": max_chars},
                       "val": "p1-17-sft@1 val.jsonl (unchanged)"},
            "files": {n: _sha(tmp / n) for n in ("train.jsonl", "val.jsonl")},
            "heldout": "expansion val/diagnostic partitions, DEV, calibration and locked test are absent "
                       "(TRAIN partition check per record; isolation audit of p1-18-expand@1)"}
    (tmp / "sft_manifest.json").write_text(json.dumps({**body, "manifest_hash": digest(body)}, indent=1,
                                                      sort_keys=True), encoding="utf-8", newline="\n")
    os.replace(tmp, out)
    return {**body, "manifest_hash": digest(body)}


def si_lines(p: Path):
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield line


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ember_v1.expand_dataset")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--repo", type=Path, default=Path("."))
    b.add_argument("--root", type=Path, default=Path(X.STORAGE_ROOT))
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--workers", type=int, default=4)
    s = sub.add_parser("sft")
    s.add_argument("--dataset", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        r = build(repo=a.repo, root=a.root, out=a.out, workers=a.workers)
        print(json.dumps({k: r[k] for k in ("status", "groups_used", "n_problems", "records_total", "sft_examples",
                                            "best_teacher_counts", "dataset_hash")}, indent=1))
        return 0 if r["status"] == "VALID" else 1
    r = sft(dataset_dir=a.dataset, out=a.out)
    print(json.dumps(r, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
