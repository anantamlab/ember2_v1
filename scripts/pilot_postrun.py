"""Read-only post-run verification of attempt 1 of p1-12-train-pilot@1. Runs NO teacher.

    python scripts/pilot_postrun.py --repo <execution checkout> --out <path outside it>/postrun.json

From the execution checkout's own code (`<repo>/src`, bytecode writing off) it:
1. loads the FROZEN plan from the freeze commit and regenerates the eight TRAIN scenarios (hash-checked);
2. replays every stored group report in a fresh simulator (`multiteacher.verify_fixture`): every joint action,
   observation, record and outcome is re-derived and every hash recomputed;
3. checks the two planned regenerations: stored regeneration == primary `content_hash`, and both replay;
4. re-hashes the attempt marker and `evidence.json` and checks they name the frozen plan hash;
5. lists sha256 (raw bytes and LF-normalized) and size of every attempt file, for the evidence packet.

`integrity_verified` is about the stored bytes; `accepted` is the runner's acceptance verdict, copied from
`evidence.json` and re-checked against the replay. A failed attempt is still verified and reported: no retry.
Exit 0 = integrity verified (whatever the acceptance verdict); 1 = an integrity problem.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

FREEZE_COMMIT = "827e88fa79d9fae9667f2fda590b4b23852988fc"
PLAN_HASH = "sha256:eb8dbe293db7f1c1e1b4f342c07f21853031e9e6ab5419eda2833c51f761b664"
OUT_DIR = "data/collection/p1-12-train-pilot@1"
POSTRUN_VERSION = "p1-12-pilot-postrun@1.0.0"


def _load(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def file_hashes(root: Path, paths: list[Path]) -> list[dict[str, Any]]:
    out = []
    for p in sorted(paths):
        b = p.read_bytes()
        out.append({"path": p.relative_to(root).as_posix(), "bytes": len(b),
                    "sha256": "sha256:" + hashlib.sha256(b).hexdigest(),
                    "sha256_lf": "sha256:" + hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()})
    return out


def verify_attempt(out_dir: Path, plan: dict[str, Any], scenarios: dict[str, Any], *,
                   plan_hash: str | None = None, attempt: int = 1) -> dict[str, Any]:
    """Integrity of one stored attempt. `scenarios` maps scenario_id -> regenerated Scenario."""
    from ember_v1 import multiteacher as mt
    from ember_v1 import train_pilot as tp
    from ember_v1.sim.canonical import digest
    paths = tp.attempt_paths(out_dir, attempt)
    problems: list[str] = []
    want_hash = plan_hash or plan["plan_hash"]
    marker = evidence = None
    if not paths["marker"].is_file():
        problems.append("attempt marker missing")
    else:
        marker = _load(paths["marker"])
        body = {k: v for k, v in marker.items() if k != "attempt_hash"}
        if digest(body) != marker.get("attempt_hash"):
            problems.append("attempt marker hash does not match its content")
        if marker.get("plan_hash") != want_hash or marker.get("attempt") != attempt:
            problems.append("attempt marker names another plan or attempt")
    if not paths["evidence"].is_file():
        problems.append("evidence.json missing (attempt ended before evidence; report as incomplete)")
    else:
        evidence = _load(paths["evidence"])
        body = {k: v for k, v in evidence.items() if k != "evidence_hash"}
        if digest(body) != evidence.get("evidence_hash"):
            problems.append("evidence hash does not match its content")
        if evidence.get("plan_hash") != want_hash:
            problems.append("evidence names another plan hash")
        if marker is not None and evidence.get("attempt") != marker:
            problems.append("evidence does not embed the attempt marker")
    groups: dict[str, dict[str, Any]] = {}
    for s in plan["scenarios"]:
        sid = s["scenario_id"]
        p = paths["groups"] / f"{sid}.json"
        if not p.is_file():
            groups[sid] = {"present": False, "replayed": False}
            continue
        rep = _load(p)
        row: dict[str, Any] = {"present": True, "group_complete": rep.get("group_complete"),
                               "training_eligible": rep.get("training_eligible"),
                               "content_hash": rep.get("content_hash"), "integrity_error": rep.get("integrity_error")}
        if rep.get("training_eligible") is not False:
            problems.append(f"{sid}: training_eligible is not false")
        if "integrity_error" in rep:
            row["replayed"] = False
        else:
            try:
                row["replayed"] = bool(mt.verify_fixture(rep, scenarios[sid])["verified"])
            except Exception as e:  # noqa: BLE001 - any replay fault is an integrity problem, reported
                row["replayed"], row["replay_error"] = False, f"{type(e).__name__}: {e}"
                problems.append(f"{sid}: stored report does not replay ({row['replay_error'][:120]})")
        groups[sid] = row
    regen: dict[str, dict[str, Any]] = {}
    for sid in plan["regeneration"]["scenario_ids"]:
        p = paths["regen"] / f"{sid}.json"
        prim = groups.get(sid, {})
        if not p.is_file():
            regen[sid] = {"present": False, "identical": False}
            continue
        again = _load(p)
        primary_hash = prim.get("content_hash")
        same = ("integrity_error" not in again and primary_hash is not None
                and again.get("content_hash") == primary_hash)
        row = {"present": True, "identical": same, "content_hash": again.get("content_hash")}
        if same:
            try:
                row["replayed"] = bool(mt.verify_fixture(again, scenarios[sid])["verified"])
            except Exception as e:  # noqa: BLE001
                row["replayed"] = False
                problems.append(f"{sid}: regeneration does not replay ({type(e).__name__})")
        regen[sid] = row
    if evidence is not None:
        claimed = {k: bool(v.get("regenerated_identical")) for k, v in (evidence.get("regeneration") or {}).items()}
        if claimed != {k: v["identical"] for k, v in regen.items()}:
            problems.append("evidence regeneration verdicts differ from the stored files")
    files = [paths["marker"]] if paths["marker"].is_file() else []
    if paths["dir"].is_dir():
        files += [p for p in paths["dir"].rglob("*") if p.is_file()]
    complete = sum(1 for g in groups.values() if g.get("group_complete") and g.get("replayed"))
    acc = (evidence or {}).get("acceptance") or {}
    return {"postrun_version": POSTRUN_VERSION, "integrity_verified": not problems, "problems": problems,
            "accepted": bool(acc.get("pass")) and not problems and complete == len(plan["scenarios"]),
            "runner_acceptance": acc, "complete_replayed_groups": complete, "groups": groups,
            "regeneration": regen, "files": file_hashes(out_dir, files), "teacher_calls": 0,
            "training_labels": "none: pilot output is scope p1-12-train-pilot@1, training_eligible false"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pilot_postrun.py", description=__doc__.splitlines()[0])
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--trusted-ref", default="origin/main")
    ap.add_argument("--out", required=True, type=Path, help="report path OUTSIDE the execution checkout")
    a = ap.parse_args(argv)
    repo = a.repo.resolve()
    if repo in [a.out.resolve(), *a.out.resolve().parents]:
        raise SystemExit("--out must be outside the execution checkout")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(repo / "src"))
    import ember_v1
    if Path(ember_v1.__file__).resolve().parent != (repo / "src" / "ember_v1").resolve():
        raise SystemExit(f"ember_v1 imported from {ember_v1.__file__}, not from {repo / 'src'}")
    from ember_v1 import train_pilot as tp
    from ember_v1 import trust
    plan, _ = trust.load_trusted_json(repo, {"kind": "git", "commit": FREEZE_COMMIT, "path": tp.FROZEN_PLAN_FILE},
                                      trusted_ref=a.trusted_ref)
    if plan.get("plan_hash") != PLAN_HASH or tp.check_plan(plan):
        raise SystemExit(f"frozen plan problem: {plan.get('plan_hash')} {tp.check_plan(plan)}")
    train = _load(repo / "data/splits/p1-02-v1/train.json")
    scenarios = {e["scenario_id"]: s for e, s in tp.pilot_scenarios(train, plan)}
    rep = verify_attempt(repo / OUT_DIR, plan, scenarios, plan_hash=PLAN_HASH)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    sys.stdout.write(json.dumps({k: rep[k] for k in ("integrity_verified", "accepted", "complete_replayed_groups",
                                                     "problems")}, indent=1) + "\n")
    return 0 if rep["integrity_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
