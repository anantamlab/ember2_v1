"""P1-18 Stage 6: DAgger labels on student-visited states (local CPU).

Input: student rollout records with per-step traces (`student_run.rollouts`) on TRAIN-partition scenarios of
p1-18-expand@1 only, and the verified T0-T4 groups of the same scenarios.

Per episode:
1. replay the trace (`student_run.replay_trace`); the outcome must equal the rollout record (else the episode is
   refused, never repaired);
2. pick up to `STATES` steps by a preregistered salted-hash rank (not by outcome, not by student error);
3. ask the scenario's group-best teachers (T1-T4 with the lowest casualties + unresolved, then casualties, on the
   matched teacher episodes; T0 is diagnostic only) for a decision on that exact public observation (teachers
   are Markov in the observation: a fresh pinned instance gives the decision it would make there);
4. one best teacher, or several that agree on the whole joint map -> a whole-joint-map label; best teachers that
   disagree -> INCONCLUSIVE (kept and counted, no label, no per-person blame); a decision over the P1-17 CPU
   cap -> no label (`teacher_cap`).

SFT view (`sft_examples`): one weight budget per family (the dataset unit), spread evenly over that family's
labelled states. Expansion val/diagnostic, DEV and the locked test never enter (TRAIN check per episode).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .sim.canonical import digest

VERSION = "p1-18-dagger@1.0.0"
SALT = "p1-18-dagger@1/state"
STATES = 8
TEACHERS = ("T1", "T2", "T3", "T4")


class DaggerError(RuntimeError):
    pass


def pick_steps(n_steps: int, scenario_hash: str, k: int = STATES) -> list[int]:
    rank = sorted(range(n_steps), key=lambda t: hashlib.sha256(f"{SALT}|{scenario_hash}|{t}".encode()).hexdigest())
    return sorted(rank[:k])


def best_teachers(group: Mapping[str, Any]) -> list[str]:
    o = {t: group["policies"][t]["outcome"] for t in TEACHERS}
    bp = min(v["casualties_plus_unresolved"] for v in o.values())
    bc = min(v["casualties"] for v in o.values() if v["casualties_plus_unresolved"] == bp)
    return [t for t in TEACHERS if o[t]["casualties_plus_unresolved"] == bp and o[t]["casualties"] == bc]


def _teacher(name: str, plan: Mapping[str, Any]) -> Any:
    from .teacher import registry as reg
    p = plan["teacher_pins"]["teachers"][name]
    cls = reg.teacher_class(name, p["version"])
    return cls() if p["seed"] is None else cls(seed=p["seed"])


def label_episode(job: Mapping[str, Any]) -> dict[str, Any]:
    """Worker: labels of one student episode (write-once JSONL)."""
    from . import student_run as sr
    from .sim import Scenario
    from .teacher import proposals
    sid = job["scenario_id"]
    try:
        out = Path(job["out"])
        if out.exists():
            raise DaggerError(f"{sid}: label file exists (write-once)")
        rec, plan = job["rollout"], job["plan"]
        if job["partition"] != "train":
            raise DaggerError(f"{sid}: not a TRAIN-partition scenario")
        scn = Scenario.from_dict(job["scenario"])
        if scn.content_hash() != rec["scenario_hash"]:
            raise DaggerError(f"{sid}: scenario hash differs from the rollout")
        group = json.loads(Path(job["group_path"]).read_text(encoding="utf-8"))
        if group.get("scenario_hash") != rec["scenario_hash"] or not group.get("group_complete"):
            raise DaggerError(f"{sid}: no complete matched teacher group")
        best = best_teachers(group)
        steps = set(pick_steps(len(rec["trace"]), rec["scenario_hash"], job.get("states", STATES)))
        seen: dict[int, tuple[dict[str, Any], dict[str, str]]] = {}

        def visit(t: int, obs: Mapping[str, Any], acts: dict[str, str]) -> None:
            if t in steps:
                seen[t] = (copy.deepcopy(dict(obs)), dict(acts))
        rp = sr.replay_trace(scn, rec["trace"], {"question_style": job["question_style"]}, visit=visit)
        if any(rp[k] != rec[k] for k in ("evacuated", "casualties", "unresolved")):
            raise DaggerError(f"{sid}: trace replay outcome differs from the rollout record")
        caps = plan["limits"]["decision_cpu_s"]
        recs, cpu = [], Counter()
        for t in sorted(seen):
            obs, s_act = seen[t]
            props, status = {}, None
            for name in best:
                pol = _teacher(name, plan)
                c0 = time.process_time()
                d = pol.decide(copy.deepcopy(obs))
                used = time.process_time() - c0
                cpu[name] += used
                if used > caps[name]:
                    status = "teacher_cap"
                    break
                prop = proposals.proposal(d)
                proposals.validate_proposal(obs, prop)
                props[name] = prop["actions"]
            if status is None:
                status = "label" if len({digest(a) for a in props.values()}) == 1 else "inconclusive"
            label = next(iter(props.values())) if status == "label" else None
            agree = (sum(s_act.get(p) == a for p, a in label.items()) / max(len(label), 1)) if label else None
            recs.append({"version": VERSION, "scenario_id": sid, "scenario_hash": rec["scenario_hash"],
                         "family_id": rec["family_id"], "student": rec["label"], "step": t, "status": status,
                         "best_teachers": best, "distinct_maps": len({digest(a) for a in props.values()}),
                         "observation_digest": digest(obs), "student_agreement": agree,
                         "observation": obs if status == "label" else None, "actions": label})
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in recs),
                       encoding="utf-8", newline="\n")
        return {"scenario_id": sid, "ok": True, "states": len(recs), "status": dict(Counter(r["status"] for r in recs)),
                "teacher_cpu_s": {k: round(v, 2) for k, v in cpu.items()}, "best": best,
                "student_agreement": [r["student_agreement"] for r in recs if r["student_agreement"] is not None]}
    except Exception as e:  # noqa: BLE001 - reported, never hidden
        return {"scenario_id": sid, "ok": False, "error": f"{type(e).__name__}: {e}"}


def run(*, rollouts: Path, root: Path, out: Path, repo: Path = Path("."), workers: int = 4,
        states: int = STATES, manifest: Mapping[str, Any] | None = None,
        scenarios: Sequence[tuple[Mapping[str, Any], Any]] | None = None) -> dict[str, Any]:
    """Label every TRAIN-partition rollout in `rollouts` whose label file does not exist yet (resumable)."""
    from . import expand as E
    from . import expand_collect as X
    from . import student_train as st
    X.register_scope()
    t0 = time.time()
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    manifest = manifest or json.loads((repo / E.OUT_DIR / "manifest.json").read_text(encoding="utf-8"))
    if plan["manifest_hash"] != manifest["manifest_hash"]:
        raise DaggerError("teacher root is not this manifest's")
    if X.check_teachers(plan):
        raise DaggerError("teacher code differs from the collection pins")
    part = {e["scenario_hash"]: e["partition"] for e in manifest["scenarios"]}
    if scenarios is None:
        scenarios = E.materialize(manifest, partitions=("train",))
    by_hash = {s.content_hash(): (e, s) for e, s in scenarios}
    style = st.config_from_protocol(st.load_protocol())["question_style"]
    rows = [json.loads(x) for x in rollouts.read_text(encoding="utf-8").splitlines() if x]
    jobs = []
    for r in rows:
        if not (out / "labels" / f"{r['scenario_id']}.jsonl").exists():
            e, s = by_hash.get(r["scenario_hash"], (None, None))
            if e is None or part.get(r["scenario_hash"]) != "train":
                raise DaggerError(f"{r['scenario_id']}: not a TRAIN-partition scenario of the manifest")
            jobs.append({"scenario_id": r["scenario_id"], "rollout": r, "plan": plan, "partition": "train",
                         "scenario": s.to_dict(), "group_path": str(root / "groups" / f"{e['scenario_id']}.json"),
                         "out": str(out / "labels" / f"{r['scenario_id']}.jsonl"), "states": states,
                         "question_style": style})
    out.mkdir(parents=True, exist_ok=True)
    if workers <= 1:
        res = [label_episode(j) for j in jobs]
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            res = list(pool.imap_unordered(label_episode, jobs, chunksize=1))
    with open(out / "runs.jsonl", "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"version": VERSION, "rollouts": str(rollouts), "jobs": len(jobs), "states": states,
                            "wall_s": round(time.time() - t0, 1),
                            "errors": [r for r in res if not r["ok"]]}, sort_keys=True) + "\n")
    return summarize(out)


def summarize(out: Path) -> dict[str, Any]:
    st, fams, agree, best = Counter(), set(), [], Counter()
    eps = 0
    for p in sorted((out / "labels").glob("*.jsonl")):
        eps += 1
        for line in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            st[r["status"]] += 1
            best["+".join(r["best_teachers"])] += 1
            if r["status"] == "label":
                fams.add(r["family_id"])
                agree.append(r["student_agreement"])
    return {"version": VERSION, "episodes": eps, "states": sum(st.values()), "status": dict(st),
            "label_families": len(fams), "best_teacher_sets": dict(best.most_common()),
            "student_agreement_mean": round(sum(agree) / len(agree), 4) if agree else None}


def sft_examples(out: Path, dest: Path, *, family_weight: float = 1.0) -> dict[str, Any]:
    """Labelled states as SFT examples: per family one weight budget (`family_weight`) spread evenly over its
    states. Inconclusive and teacher-cap states are never examples."""
    from . import student_ingest as si
    labs = [json.loads(x) for p in sorted((out / "labels").glob("*.jsonl"))
            for x in p.read_text(encoding="utf-8").splitlines() if x]
    labs = [r for r in labs if r["status"] == "label"]
    per_fam = Counter(r["family_id"] for r in labs)
    by_fam = defaultdict(float)
    if dest.exists():
        raise DaggerError(f"{dest} exists (write-once)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        for r in labs:
            ex = {"observation": r["observation"], "actions": r["actions"], "weight": family_weight / per_fam[r["family_id"]]}
            ex = {"example_id": si.example_id(ex), **ex}
            by_fam[r["family_id"]] += ex["weight"]
            f.write(json.dumps(ex, sort_keys=True, separators=(",", ":")) + "\n")
    return {"examples": len(labs), "families": len(per_fam), "family_weight": family_weight,
            "weight_total": round(sum(by_fam.values()), 9),
            "file_sha256": "sha256:" + hashlib.sha256(dest.read_bytes()).hexdigest()}


def main(argv: Sequence[str] | None = None) -> int:
    from . import expand_collect as X
    ap = argparse.ArgumentParser(prog="python -m ember_v1.dagger")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("label")
    r.add_argument("--rollouts", type=Path, required=True)
    r.add_argument("--root", type=Path, default=Path(X.STORAGE_ROOT))
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--states", type=int, default=STATES)
    s = sub.add_parser("sft")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--dest", type=Path, required=True)
    s.add_argument("--family-weight", type=float, default=1.0)
    a = ap.parse_args(argv)
    if a.cmd == "label":
        print(json.dumps(run(rollouts=a.rollouts, root=a.root, out=a.out, workers=a.workers, states=a.states),
                         indent=1))
    else:
        print(json.dumps(sft_examples(a.out, a.dest, family_weight=a.family_weight), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
