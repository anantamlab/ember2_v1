"""Independent student evaluation protocol: preregistered decision code (P1-15, issue #48).

Written BEFORE any student outcome exists. Nothing here generates a test set,
runs a student or reads a sealed suite. The code fixes, in advance:

- the comparator: the strongest single deployable teacher, named from a
  separate comparator-selection set, never from test outcomes;
- the oracle: per-scenario best teacher in hindsight, reported apart, never a baseline;
- the primary test: paired student-minus-comparator casualties+unresolved,
  averaged per family, one-sided family bootstrap upper bound < 0;
- guards: casualties and unresolved bounds, severe regressions, safety failures,
  zero teacher calls in the student path, latency limits;
- secondary: the same test against every teacher with Holm adjustment.

Matching T1 is not success. A shielded-arm improvement alone is not success.
"""
from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .teacher.registry import CANDIDATES

REPO = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO / "data" / "protocols" / "p1-15-student-evaluation.json"
SUPERIOR, NOT_SUPERIOR, INVALID = "SUPERIOR_TO_STRONGEST_TEACHER", "NOT_SUPERIOR", "INVALID_RUN"


class EvalProtocolError(ValueError):
    """Protocol, set, comparator or result breaks the preregistered rules."""


def load_protocol(path: str | Path = PROTOCOL_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def freeze_problems(p: Mapping[str, Any]) -> list[str]:
    """Reasons the protocol cannot be frozen yet (empty list = may be frozen after review)."""
    errs = []
    lat = p["guards"]["latency_ms"]
    errs += [f"latency {k} is not set" for k in ("p50_max", "p95_max", "p99_max") if lat.get(k) is None]
    if p.get("written_before_any_student_outcome") is not True:
        errs.append("protocol must be written before any student outcome")
    if p["secondary"]["claim_arm"] != "raw_student":
        errs.append("the claim arm must be the raw student")
    if p["comparator_rule"]["teacher_order"] != list(CANDIDATES):
        errs.append("teacher order must be the registry candidate order")
    return errs


def family_overlaps(sets: Mapping[str, Iterable[str]]) -> dict[str, list[str]]:
    """Pairwise overlaps of family (or geometry core) ids between named sets. Must be empty."""
    names = sorted(sets)
    vals = {n: set(sets[n]) for n in names}
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = sorted(vals[a] & vals[b])
            if both:
                out[f"{a}&{b}"] = both
    return out


def check_test_isolation(test: Iterable[str], others: Mapping[str, Iterable[str]]) -> None:
    ov = family_overlaps({"test": test, **others})
    bad = {k: v for k, v in ov.items() if k.startswith("test&") or k.endswith("&test")}
    if bad:
        raise EvalProtocolError(f"test families overlap other sets: {sorted(bad)}")


def select_comparator(totals: Mapping[str, Mapping[str, int]], *, set_kind: str,
                      protocol: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Strongest single deployable teacher from comparator-selection totals ONLY."""
    p = protocol or load_protocol()
    if set_kind != "comparator_selection":
        raise EvalProtocolError("the comparator is named from the comparator-selection set, never from test")
    if set(totals) != set(CANDIDATES):
        raise EvalProtocolError("all five teachers must be measured on the comparator-selection set")
    order = p["comparator_rule"]["teacher_order"]
    key = lambda t: (totals[t]["casualties_plus_unresolved"], totals[t]["casualties"], order.index(t))  # noqa: E731
    ranked = sorted(totals, key=key)
    return {"comparator": ranked[0], "ranking": ranked, "rule": p["comparator_rule"]["rank_by"],
            "set_kind": set_kind}


def oracle_reference(per_scenario: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    """Hindsight best teacher primary per scenario. NOT deployable, NOT a baseline."""
    best = {s: min(v.values()) for s, v in per_scenario.items()}
    return {"deployable": False, "per_scenario_best_primary": best, "total": sum(best.values()),
            "note": "retrospective oracle; reported separately"}


def family_means(diffs: Mapping[str, Sequence[float]]) -> list[float]:
    if not diffs or any(len(v) == 0 for v in diffs.values()):
        raise EvalProtocolError("every family needs at least one paired scenario")
    return [sum(v) / len(v) for _, v in sorted(diffs.items())]


def bootstrap_upper(diffs: Mapping[str, Sequence[float]], *, alpha: float, resamples: int, seed: int) -> float:
    """One-sided (1 - alpha) upper bound of the mean family difference; percentile family bootstrap."""
    m = family_means(diffs)
    rng = random.Random(seed)
    k = len(m)
    stats = sorted(sum(m[rng.randrange(k)] for _ in range(k)) / k for _ in range(resamples))
    return stats[min(resamples - 1, int((1 - alpha) * resamples))]


def holm(pvals: Mapping[str, float], alpha: float) -> dict[str, bool]:
    order = sorted(pvals, key=pvals.get)
    out, ok = {}, True
    for i, t in enumerate(order):
        ok = ok and pvals[t] <= alpha / (len(order) - i)
        out[t] = ok
    return out


def bootstrap_p(diffs: Mapping[str, Sequence[float]], *, resamples: int, seed: int) -> float:
    """One-sided bootstrap p for mean family difference >= 0 (centered resampling)."""
    m = family_means(diffs)
    mean = sum(m) / len(m)
    c = [x - mean for x in m]
    rng = random.Random(seed)
    k = len(c)
    hits = sum(sum(c[rng.randrange(k)] for _ in range(k)) / k <= mean for _ in range(resamples))
    return (hits + 1) / (resamples + 1)


def _diffs(rows: Sequence[Mapping[str, Any]], arm: str, ref: str, metric: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for r in rows:
        out.setdefault(r["family_id"], []).append(r[arm][metric] - r[ref][metric])
    return out


def decide(rows: Sequence[Mapping[str, Any]], comparator: Mapping[str, Any], run: Mapping[str, Any],
           protocol: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Preregistered verdict. `rows`: one per test scenario with `family_id`, `people`, `raw_student`, and
    each teacher name -> {casualties, unresolved, casualties_plus_unresolved}. `run`: safety failures,
    teacher calls in the student path, latency percentiles, and `set_kind`."""
    p = protocol or load_protocol()
    if comparator.get("set_kind") != "comparator_selection":
        raise EvalProtocolError("comparator was not named from the comparator-selection set")
    if run.get("set_kind") != "test":
        raise EvalProtocolError("decide() runs on the sealed test set only")
    if p.get("status") != "FROZEN":
        return {"verdict": INVALID, "reasons": ["protocol is not FROZEN"]}
    reasons = []
    g = p["guards"]
    if run["teacher_calls_in_student_path"] > g["teacher_calls_in_student_path_max"]:
        return {"verdict": INVALID, "reasons": ["a reference teacher ran inside the student path"]}
    if run["safety_failures"] > g["safety_failures_max"]:
        reasons.append("safety failures")
    for k in ("p50", "p95", "p99"):
        lim = g["latency_ms"][f"{k}_max"]
        if lim is None or run[f"latency_{k}_ms"] is None or run[f"latency_{k}_ms"] > lim:
            reasons.append(f"latency {k} limit not met or not measurable")
    ref, arm = comparator["comparator"], "raw_student"
    pr = p["primary"]
    kw = {"alpha": pr["alpha_one_sided"], "resamples": pr["bootstrap_resamples"], "seed": pr["bootstrap_seed"]}
    ub = {m: bootstrap_upper(_diffs(rows, arm, ref, m), **kw)
          for m in ("casualties_plus_unresolved", "casualties", "unresolved")}
    if not ub["casualties_plus_unresolved"] < 0:
        reasons.append("primary: upper bound of casualties+unresolved difference is not < 0")
    if ub["casualties"] > g["casualties_upper_bound_max"]:
        reasons.append("guard: casualties may be worse than the comparator")
    if ub["unresolved"] > g["unresolved_upper_bound_max"]:
        reasons.append("guard: unresolved may be worse than the comparator")
    severe = sum(r[arm]["casualties_plus_unresolved"] - r[ref]["casualties_plus_unresolved"]
                 >= max(3, 0.10 * r["people"]) for r in rows)
    if severe > g["severe_regression_max_fraction"] * len(rows):
        reasons.append(f"severe regressions {severe} of {len(rows)}")
    pv = {t: bootstrap_p(_diffs(rows, arm, t, "casualties_plus_unresolved"),
                         resamples=pr["bootstrap_resamples"], seed=pr["bootstrap_seed"]) for t in CANDIDATES}
    return {"verdict": NOT_SUPERIOR if reasons else SUPERIOR, "reasons": reasons, "comparator": ref,
            "upper_bounds": ub, "severe_regressions": severe,
            "secondary_beats_teacher_holm": holm(pv, pr["alpha_one_sided"]), "secondary_p": pv,
            "oracle": oracle_reference({r.get("scenario_id", str(i)): {t: r[t]["casualties_plus_unresolved"]
                                                                       for t in CANDIDATES}
                                        for i, r in enumerate(rows)})}
