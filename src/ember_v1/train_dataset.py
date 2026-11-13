"""P1-17 Phase 6: multi-teacher dataset records (classes A-D) from the verified collection p1-17-train-full@1.

Protocol: `data/protocols/p1-14-multiteacher-dataset.json` 1.0.0 (FROZEN). Code: `multiteacher_dataset`.

Gates, in order (nothing is written before every gate passes):
  1. `multiteacher_dataset.export`: FROZEN protocol, owner authorization, collection gate (verification report
     hash recomputes and equals the protocol's pin; 189/189 verified_complete; plan hash).
  2. Every group file sha256 equals its collection row; every scenario regenerates from the committed manifest.
Build (parallel, one process per group):
  pass 1  replay the group (`verify_fixture` inside `demonstrations`), class A/B records, whole-map selection,
          whole-proposal check against the group (`check_targets_against_report`)  -> staging/unweighted
  pass 2  family-unit weights with the global family counts, per-record and per-state checks  -> staging/records
  global  audit, family-share cap, duplicate hashes, teacher source drift, selection-bias audit
Only when every check passes is `staging/` renamed to `records/` and `dataset.json` written with status VALID.
Otherwise `dataset.json` has status INVALID with every problem and the staging files stay as evidence.
The output directory must be outside every checkout; it is write-once.

Class C/D (counterfactual preferences) are not built here: no real-scope counterfactual exists yet and
preference approval is disabled, so every C would have weight 0. Class E stays reserved (P1-16).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import multiteacher_dataset as ds
from .sim.canonical import digest

VERSION = "p1-17-train-dataset@1.0.0"
SCOPE = "p1-17-train-full@1"
REPO = Path(__file__).resolve().parents[2]
PLAN_FILE = "data/collection/p1-17-train-full@1/plan.json"
TRAIN_FILE = "data/splits/p1-02-v1/train.json"


class BuildBlocked(RuntimeError):
    """A gate failed; nothing was written."""


def _read(p: Path) -> Any:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _sha256(p: Path) -> str:
    return "sha256:" + hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _write_once(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "x", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


def _jsonl(recs: Sequence[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in recs)


def _load_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x]


def _inside(child: Path, parent: Path) -> bool:
    try:
        Path(os.path.abspath(child)).relative_to(Path(os.path.abspath(parent)))
        return True
    except ValueError:
        return False


def light(r: Mapping[str, Any]) -> dict[str, Any]:
    """The fields `audit` and `selection_bias_audit` read; the observation is reduced to people status and
    fire model. Used for the whole-dataset checks without holding every observation in memory."""
    obs = r["observation"]
    return {"class": r["class"], "teachers": r.get("teachers", []), "weight": r["weight"],
            "family_id": r["family_id"], "scenario_hash": r["scenario_hash"],
            "observation_digest": r["observation_digest"], "selection": r.get("selection"),
            "exclusion_reason": r.get("exclusion_reason"), "approval": r.get("approval"),
            "evidence": r["evidence"], "step": r.get("step"), "teacher_values": bool(r.get("teacher_values")),
            "observation": {"people": [{"status": p["status"]} for p in obs["people"]],
                            "rules": {"fire_model": obs["rules"]["fire_model"]}}}


# ----- workers ------------------------------------------------------------------------------------------------
def _pass1(job: Mapping[str, Any]) -> dict[str, Any]:
    sid = job["scenario_id"]
    try:
        protocol = ds.load_protocol(job["protocol_path"])
        g = _read(Path(job["group_path"]))
        recs = ds.demonstrations(g, job["scenario"], protocol, family_id=job["family_id"],
                                 teacher_dev_similar=job["teacher_dev_similar"])
        recs = ds.select_joint_targets(recs, protocol)
        problems = ds.check_targets_against_report(recs, g)
        if g.get("scope") != SCOPE or g.get("scenario_hash") != job["scenario_hash"]:
            problems.append(f"{sid}: group scope or scenario hash is not the plan's")
        _write_once(Path(job["out"]), _jsonl(recs))
        sel = any(r["class"] == "A" and r["selection"]["selected"] for r in recs)
        outs = {n: p["outcome"] for n, p in g["policies"].items()}
        return {"scenario_id": sid, "ok": not problems, "problems": problems, "records": len(recs),
                "family_id": job["family_id"], "scenario_hash": job["scenario_hash"], "selected_A": sel,
                "outcomes": {n: {k: o[k] for k in ("casualties", "unresolved", "casualties_plus_unresolved")}
                             for n, o in outs.items()}}
    except Exception as e:  # noqa: BLE001 - every failure is reported, never hidden
        return {"scenario_id": sid, "ok": False, "problems": [f"{sid}: {type(e).__name__}: {e}"], "records": 0,
                "family_id": job["family_id"], "scenario_hash": job["scenario_hash"], "selected_A": False}


def _pass2(job: Mapping[str, Any]) -> dict[str, Any]:
    sid = job["scenario_id"]
    try:
        protocol = ds.load_protocol(job["protocol_path"])
        recs = ds.assign_weights(_load_jsonl(Path(job["src"])), protocol, family_scenarios=job["family_scenarios"])
        problems = [f"{sid}[{i}]: {e}" for i, r in enumerate(recs) for e in ds.record_problems(r, protocol)]
        problems += [f"{sid}: {e}" for e in ds.coherence_problems(recs)]
        keys = Counter((r["class"], r["observation_digest"], digest(r["actions"])) for r in recs)
        problems += [f"{sid}: duplicate record {k[0]} {k[1][:19]}" for k, n in keys.items() if n > 1]
        sft = [ds.model_example(r) for r in recs if r["class"] == "A" and r["selection"]["selected"] and r["weight"]]
        _write_once(Path(job["out"]), _jsonl(recs))
        teachers = sorted({(t, digest(p.get("identity")), p.get("code_fingerprint")) for r in recs
                           for t, p in r["provenance"]["teachers"].items()})
        return {"scenario_id": sid, "ok": not problems, "problems": problems[:50], "n_problems": len(problems),
                "light": [light(r) for r in recs], "hashes": [r["record_hash"] for r in recs],
                "teachers": teachers, "sft_examples": len(sft),
                "file_sha256": _sha256(Path(job["out"]))}
    except Exception as e:  # noqa: BLE001
        return {"scenario_id": sid, "ok": False, "problems": [f"{sid}: {type(e).__name__}: {e}"], "n_problems": 1,
                "light": [], "hashes": [], "teachers": [], "sft_examples": 0, "file_sha256": None}


def _map(fn, jobs: Sequence[Mapping[str, Any]], workers: int) -> list[dict[str, Any]]:
    if workers <= 1:
        return [fn(j) for j in jobs]
    with mp.get_context("spawn").Pool(workers) as pool:
        return list(pool.imap_unordered(fn, jobs, chunksize=1))


# ----- gates and build ----------------------------------------------------------------------------------------
def gates(plan: Mapping[str, Any], root: Path, verification: Mapping[str, Any], protocol: Mapping[str, Any],
          authorization: str, out: Path, repo: Path) -> dict[str, Any]:
    gate = ds.export(protocol=protocol, authorization=authorization, verification=verification)
    problems = []
    if plan.get("plan_hash") != verification.get("plan_hash") or plan.get("namespace") != SCOPE:
        problems.append("plan is not the verified collection's plan")
    if _inside(out, repo) or _inside(out, root):
        problems.append(f"output {out} must be outside the checkout and the collection root")
    if out.exists():
        problems.append(f"output {out} exists: the build is write-once")
    for spec in plan["scenarios"]:
        sid = spec["scenario_id"]
        gp, rp = root / "groups" / f"{sid}.json", root / "rows" / f"{sid}.json"
        if not gp.exists() or not rp.exists():
            problems.append(f"{sid}: group or row missing")
            continue
        if _sha256(gp) != _read(rp).get("file_sha256"):
            problems.append(f"{sid}: group file sha256 differs from its row")
    if problems:
        raise BuildBlocked("; ".join(problems[:20]))
    return gate


def build(*, repo: Path, root: Path, verification_path: Path, out: Path, authorization: str,
          workers: int = 8, plan_path: Path | None = None, protocol_path: Path | None = None,
          scenarios: Sequence[tuple[Mapping[str, Any], Any]] | None = None) -> dict[str, Any]:
    t0 = time.time()
    plan = _read(plan_path or repo / PLAN_FILE)
    protocol_path = Path(protocol_path or ds.PROTOCOL_PATH)
    protocol = ds.load_protocol(protocol_path)
    verification = _read(verification_path)
    gate = gates(plan, root, verification, protocol, authorization, out, repo)
    if scenarios is None:
        from .train_collect import collection_scenarios
        scenarios = collection_scenarios(_read(repo / TRAIN_FILE), plan)
    by_id = {e["scenario_id"]: s for e, s in scenarios}
    flagged = set((plan.get("geometry_audit") or {}).get("teacher_dev_similar_families") or [])
    staging = out / "staging"
    jobs1 = [{"scenario_id": s["scenario_id"], "group_path": str(root / "groups" / f"{s['scenario_id']}.json"),
              "scenario": by_id[s["scenario_id"]], "scenario_hash": s["scenario_hash"], "family_id": s["family_id"],
              "teacher_dev_similar": s["family_id"] in flagged, "protocol_path": str(protocol_path),
              "out": str(staging / "unweighted" / f"{s['scenario_id']}.jsonl")} for s in plan["scenarios"]]
    out.mkdir(parents=True)
    _write_once(out / "build_start.json", json.dumps({"version": VERSION, "gate": gate, "workers": workers,
                                                      "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                                                     indent=1, sort_keys=True))
    r1 = sorted(_map(_pass1, jobs1, workers), key=lambda x: x["scenario_id"])
    problems = [p for x in r1 for p in x["problems"]]
    fam_counts = dict(sorted(Counter(x["family_id"] for x in r1 if x["selected_A"]).items()))
    r2: list[dict[str, Any]] = []
    if not problems:
        jobs2 = [{"scenario_id": x["scenario_id"], "protocol_path": str(protocol_path),
                  "src": str(staging / "unweighted" / f"{x['scenario_id']}.jsonl"),
                  "out": str(staging / "records" / f"{x['scenario_id']}.jsonl"), "family_scenarios": fam_counts}
                 for x in r1]
        r2 = sorted(_map(_pass2, jobs2, workers), key=lambda x: x["scenario_id"])
        problems += [p for x in r2 for p in x["problems"]]
    summary = finish(protocol, r1, r2, problems, fam_counts)
    status = "VALID" if not summary["problems"] else "INVALID"
    if status == "VALID":
        os.replace(staging / "records", out / "records")
    files = {x["scenario_id"]: x["file_sha256"] for x in r2}
    body = {"version": VERSION, "status": status, "scope": SCOPE, "protocol_id": protocol["protocol_id"],
            "protocol_version": protocol["version"], "protocol_sha256": _sha256(protocol_path),
            "record_version": ds.RECORD_VERSION, "gate": gate, "plan_hash": plan["plan_hash"],
            "verification_hash": verification["verification_hash"], "authorization": authorization,
            "groups": len(r1), "family_scenarios_with_A": fam_counts, "record_files": files,
            "wall_s": round(time.time() - t0, 1), "training_eligible": False,
            "training_eligible_note": "records are dataset evidence; training needs the P1-14 training protocol, "
                                      "an owner authorization and the SFT split (Phase 8)", **summary}
    _write_once(out / "dataset.json", json.dumps({**body, "dataset_hash": digest(body)}, indent=1, sort_keys=True))
    return {**body, "dataset_hash": digest(body)}


def finish(protocol: Mapping[str, Any], r1: Sequence[Mapping[str, Any]], r2: Sequence[Mapping[str, Any]],
           problems: list[str], fam_counts: Mapping[str, int]) -> dict[str, Any]:
    """Whole-dataset checks over the light records of every scenario."""
    problems = list(problems)
    lights = [x for r in r2 for x in r["light"]]
    hashes = Counter(h for r in r2 for h in r["hashes"])
    problems += [f"duplicate record hash {h}" for h, n in hashes.items() if n > 1]
    seen: dict[str, set] = defaultdict(set)
    for r in r2:
        for t, ident, fp in r["teachers"]:
            seen[t].add((ident, fp))
    problems += [f"teacher {t} source drift: {len(v)} identities or fingerprints" for t, v in sorted(seen.items())
                 if len(v) > 1]
    a = ds.audit(lights) if lights else {}
    review: list[str] = []
    if lights:
        w = protocol["weighting"]
        if a["scenarios"] >= w["min_scenarios_for_share_caps"]:
            review += [f"teacher {t} share {s} > {w['teacher_share_A_review']} (review; never auto-reweighted)"
                       for t, s in a["teacher_share_A"].items() if s > w["teacher_share_A_review"]]
            cap = max(w["max_family_share"]["absolute"], w["max_family_share"]["relative_to_uniform"] / a["families"])
            problems += [f"family {f} share {s} > {round(cap, 9)}" for f, s in a["family_share"].items() if s > cap]
    bias = ds.selection_bias_audit(lights) if lights else {}
    no_a = sorted(x["scenario_id"] for x in r1 if not x["selected_A"])
    t0_only = sorted(x["scenario_id"] for x in r1 if x.get("outcomes") and not x["selected_A"] and
                     _best_teachers(x["outcomes"]) == {"T0"})
    return {"problems": problems[:200], "n_problems": len(problems), "review": review, "audit": a,
            "selection_bias": bias, "records_total": sum(x["records"] for x in r1),
            "sft_examples": sum(x["sft_examples"] for x in r2),
            "scenarios_without_class_A": no_a, "scenarios_only_T0_best": t0_only,
            "families_with_A": len(fam_counts)}


def _best_teachers(outs: Mapping[str, Mapping[str, int]]) -> set[str]:
    bp = min(o["casualties_plus_unresolved"] for o in outs.values())
    bc = min(o["casualties"] for o in outs.values())
    return {n for n, o in outs.items() if o["casualties_plus_unresolved"] == bp and o["casualties"] == bc}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ember_v1.train_dataset", description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, help="collection evidence root (plan storage.root)")
    ap.add_argument("--verification", required=True, help="train_collect verify report")
    ap.add_argument("--out", required=True, help="new output directory outside every checkout")
    ap.add_argument("--authorization", required=True, help="owner authorization reference")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    try:
        res = build(repo=REPO, root=Path(a.root), verification_path=Path(a.verification), out=Path(a.out),
                    authorization=a.authorization, workers=a.workers)
    except (BuildBlocked, ds.DatasetError) as e:
        print(json.dumps({"status": "BLOCKED", "reason": str(e)}))
        return 2
    print(json.dumps({k: res[k] for k in ("status", "groups", "records_total", "sft_examples", "n_problems",
                                          "dataset_hash")}))
    return 0 if res["status"] == "VALID" else 1


if __name__ == "__main__":
    sys.exit(main())
