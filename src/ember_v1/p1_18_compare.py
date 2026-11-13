"""P1-18 closed-loop comparison: students (rollout records) vs T0-T4 (verified groups) on the SAME scenarios.

Pairing is by scenario content hash. Per scenario and policy: people, evacuated, casualties, unresolved,
casualties + unresolved, blocked moves (every `blocked_*` simulator outcome), collisions (occupied / conflict /
cycle), failures. Uncertainty: the family is the unit (scenarios of one family are not independent); 95 %
intervals are percentile bootstraps over families (fixed seed) of the difference in the pooled per-person rate.
Only complete pairs count: a scenario missing for any compared policy is listed and left out, never imputed.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

VERSION = "p1-18-compare@1.0.0"
TEACHERS = ("T0", "T1", "T2", "T3", "T4")
COLLISIONS = ("blocked_occupied", "blocked_conflict", "blocked_cycle")
METRICS = ("people", "evacuated", "casualties", "unresolved", "cu", "blocked", "collisions", "failed")
BOOT = 2000
SEED = 20261110


def _row(people: int, evac: int, cas: int, unres: int, moves: Mapping[str, int], failed: bool) -> dict[str, int]:
    return {"people": people, "evacuated": evac, "casualties": cas, "unresolved": unres, "cu": cas + unres,
            "blocked": sum(v for k, v in moves.items() if k.startswith("blocked_")),
            "collisions": sum(moves.get(k, 0) for k in COLLISIONS), "failed": int(bool(failed))}


def teacher_rows(roots: Iterable[Path]) -> dict[str, dict[str, Any]]:
    """scenario_hash -> {family-free teacher metric rows} from every complete group under the roots."""
    out: dict[str, dict[str, Any]] = {}
    for root in roots:
        for gp in sorted((Path(root) / "groups").glob("*.json")):
            g = json.loads(gp.read_text(encoding="utf-8"))
            if not g.get("group_complete"):
                continue
            rows = {}
            for n, p in g["policies"].items():
                o = p["outcome"]
                rows[n] = _row(o["people"], o["evacuated"], o["casualties"], o["unresolved"],
                               o.get("movement_outcomes", {}), p.get("failure"))
            out[g["scenario_hash"]] = rows
    return out


def student_rows(path: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        r = json.loads(line)
        out[r["scenario_hash"]] = {"set": r["set"], "family_id": r["family_id"], "scenario_id": r["scenario_id"],
                                   "row": _row(r["people"], r["evacuated"], r["casualties"], r["unresolved"],
                                               r.get("movement_outcomes", {}), r.get("failure"))}
    return out


def _rate(rows: Sequence[Mapping[str, int]], k: str) -> float:
    ppl = sum(r["people"] for r in rows)
    return sum(r[k] for r in rows) / ppl if ppl else 0.0


def boot_diff(by_fam: Mapping[str, Sequence[tuple[Mapping[str, int], Mapping[str, int]]]], k: str,
              n: int = BOOT, seed: int = SEED) -> dict[str, Any]:
    """Pooled per-person rate of A minus B, with a family-cluster percentile bootstrap 95 % interval."""
    fams = sorted(by_fam)
    pairs = [p for f in fams for p in by_fam[f]]
    point = _rate([a for a, _ in pairs], k) - _rate([b for _, b in pairs], k)
    rng = random.Random(f"{seed}/{k}")
    stats = []
    for _ in range(n):
        pick = [rng.choice(fams) for _ in fams]
        ps = [p for f in pick for p in by_fam[f]]
        stats.append(_rate([a for a, _ in ps], k) - _rate([b for _, b in ps], k))
    stats.sort()
    lo, hi = stats[int(0.025 * n)], stats[int(0.975 * n) - 1]
    return {"diff": round(point, 6), "ci95": [round(lo, 6), round(hi, 6)], "families": len(fams),
            "scenarios": len(pairs), "excludes_zero": lo > 0 or hi < 0}


def list_meta(lists: Iterable[Path]) -> dict[str, dict[str, Any]]:
    """scenario_hash -> {set, family_id, scenario_id} from rollout lists (student_run.rollout_list)."""
    meta: dict[str, dict[str, Any]] = {}
    for p in lists:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if line:
                d = json.loads(line)
                meta[d["scenario_hash"]] = {k: d[k] for k in ("set", "family_id", "scenario_id")}
    return meta


def compare(lists: Sequence[Path], students: Mapping[str, Path], teacher_roots: Sequence[Path]) -> dict[str, Any]:
    """Per evaluation set (from the rollout lists): totals and per-person rates of every policy on the scenarios
    ALL of them finished, then each student minus each teacher, minus the per-scenario best of T1-T4, and minus
    every other student."""
    T = teacher_rows(teacher_roots)
    S = {name: student_rows(p) for name, p in students.items()}
    meta = list_meta(lists)
    by_set: dict[str, list[str]] = defaultdict(list)
    for h, m in meta.items():
        by_set[m["set"]].append(h)
    out: dict[str, Any] = {"version": VERSION, "unit": "family", "bootstrap": {"n": BOOT, "seed": SEED},
                           "students": {n: str(p) for n, p in students.items()}, "sets": {}}
    for st in sorted(by_set):
        hs = sorted(by_set[st])
        ok = [h for h in hs if h in T and set(TEACHERS) <= set(T[h]) and all(h in rows for rows in S.values())]
        missing = sorted(meta[h]["scenario_id"] for h in hs if h not in ok)
        pol: dict[str, list[tuple[str, Mapping[str, int]]]] = defaultdict(list)
        for h in ok:
            fam = meta[h]["family_id"]
            for t in TEACHERS:
                pol[t].append((fam, T[h][t]))
            best = min((T[h][t] for t in TEACHERS[1:]), key=lambda r: (r["cu"], r["casualties"]))
            pol["best_T1-T4"].append((fam, best))
            for name, rows in S.items():
                pol[name].append((fam, rows[h]["row"]))
        res: dict[str, Any] = {"scenarios": len(ok), "families": len({meta[h]["family_id"] for h in ok}),
                               "missing_pairs": missing, "policies": {}, "contrasts": {}}
        for name, rs in pol.items():
            rows = [r for _, r in rs]
            res["policies"][name] = {**{k: sum(r[k] for r in rows) for k in METRICS},
                                     **{f"{k}_rate": round(_rate(rows, k), 6)
                                        for k in ("cu", "casualties", "unresolved", "evacuated", "blocked",
                                                  "collisions")}}
        refs = [*TEACHERS, "best_T1-T4", *sorted(S)]
        for a in sorted(S):
            for b in refs:
                if a == b:
                    continue
                by_fam: dict[str, list] = defaultdict(list)
                for (fa, ra), (_, rb) in zip(pol[a], pol[b]):
                    by_fam[fa].append((ra, rb))
                res["contrasts"][f"{a}-{b}"] = {k: boot_diff(by_fam, k) for k in ("cu", "casualties", "unresolved")}
        out["sets"][st] = res
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ember_v1.p1_18_compare")
    ap.add_argument("--list", action="append", default=[], type=Path, help="rollout list(s): the scenario sets")
    ap.add_argument("--student", action="append", default=[], help="NAME=rollouts.jsonl")
    ap.add_argument("--teachers", action="append", default=[], type=Path, help="collection root(s)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    res = compare(a.list, {n: Path(p) for n, p in (s.split("=", 1) for s in a.student)}, a.teachers)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, sort_keys=True), encoding="utf-8", newline="\n")
    for st, r in res["sets"].items():
        print(st, r["scenarios"], {n: p["cu_rate"] for n, p in r["policies"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
