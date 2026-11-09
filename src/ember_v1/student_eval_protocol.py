"""Independent student evaluation protocol: preregistered decision code (P1-15, issue #48).

Written BEFORE any student outcome exists. Nothing here generates a test set,
runs a student or reads a sealed suite. The code fixes, in advance:

- the comparator: the strongest single deployable teacher, named from a
  separate comparator-selection set, never from test outcomes;
- the oracle: per-scenario best teacher in hindsight, reported apart, never a baseline;
- the primary test: paired student-minus-comparator casualties+unresolved,
  averaged per family, one-sided family bootstrap upper bound < 0;
- guards: casualties and unresolved upper bounds AND point estimates <= 0, severe
  regressions per outcome (count and Clopper-Pearson rate bound), simulator-safety
  failures, model-input privacy audit, zero teacher calls in the student path, latency;
- claims: primary only, per outcome (Holm over 2), both outcomes (intersection-union);
- secondary: the same test against every teacher with Holm adjustment.

Matching T1 is not success. A shielded-arm improvement alone is not success.
"""
from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .sim.canonical import digest
from .teacher.registry import CANDIDATES

REPO = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO / "data" / "protocols" / "p1-15-student-evaluation.json"
SUPERIOR, NOT_SUPERIOR, INVALID = "SUPERIOR_TO_STRONGEST_TEACHER", "NOT_SUPERIOR", "INVALID_RUN"
# Family sets the sealed test must be disjoint from (protocol `sets.forbidden_sources`); every one must be given.
REQUIRED_ISOLATION = ("train", "dev", "calibration", "p1-09-selection", "counterfactual", "model_selection",
                      "comparator_selection")
PEOPLE_BANDS = ((0, 50), (51, 110), (111, 10 ** 9))
_OUTCOMES = ("casualties", "unresolved", "casualties_plus_unresolved")


class EvalProtocolError(ValueError):
    """Protocol, set, comparator or result breaks the preregistered rules."""


def load_protocol(path: str | Path = PROTOCOL_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def freeze_problems(p: Mapping[str, Any]) -> list[str]:
    """Reasons the protocol cannot be frozen yet (empty list = may be frozen after review)."""
    errs = []
    lat = p["guards"]["latency_ms"]
    errs += [f"latency {k} is not set" for k in ("p50_max", "p95_max", "p99_max") if lat.get(k) is None]
    if not lat.get("evidence"):
        errs.append("latency limits have no measured GPU evidence (preflight report hash)")
    g = p["guards"]
    if g["unresolved_upper_bound_max"] > 0 or g["casualties_upper_bound_max"] > 0:
        errs.append("a positive guard margin lets casualties or unresolved increase")
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


def protocol_hash(p: Mapping[str, Any]) -> str:
    return digest(dict(p))


def select_comparator(totals: Mapping[str, Mapping[str, int]], *, set_kind: str,
                      deployable: Mapping[str, bool], protocol: Mapping[str, Any] | None = None,
                      selection_families: Iterable[str] = ()) -> dict[str, Any]:
    """Strongest single DEPLOYABLE teacher from comparator-selection totals ONLY.

    `deployable[t]` is True when teacher t, running alone, met the frozen latency budget on the deployment
    hardware (measured on the comparator-selection set). A non-deployable teacher is ranked and reported but is
    never the comparator; it stays in the secondary comparisons. A per-scenario hindsight best is never one.

    The returned record is sealed (`comparator_hash`) and binds the selection totals, the selection families and
    the protocol hash, so `decide` can re-derive the comparator and prove it used no test outcome."""
    p = protocol or load_protocol()
    if set_kind != "comparator_selection":
        raise EvalProtocolError("the comparator is named from the comparator-selection set, never from test")
    if set(totals) != set(CANDIDATES) or set(deployable) != set(CANDIDATES):
        raise EvalProtocolError("all five teachers must be measured on the comparator-selection set")
    order = p["comparator_rule"]["teacher_order"]
    key = lambda t: (totals[t]["casualties_plus_unresolved"], totals[t]["casualties"], order.index(t))  # noqa: E731
    ranked = sorted(totals, key=key)
    ok = [t for t in ranked if deployable[t] is True]
    if not ok:
        raise EvalProtocolError("no teacher is deployable within the latency budget")
    body = {"comparator": ok[0], "ranking": ranked, "deployable": dict(deployable),
            "rule": p["comparator_rule"]["rank_by"], "set_kind": set_kind,
            "totals": {t: {k: totals[t][k] for k in ("casualties_plus_unresolved", "casualties")} for t in CANDIDATES},
            "selection_families": sorted(set(selection_families)), "protocol_hash": protocol_hash(p)}
    return {**body, "comparator_hash": digest(body)}


def comparator_problems(comparator: Mapping[str, Any], p: Mapping[str, Any]) -> list[str]:
    """Why a comparator record is not a valid, independent selection under protocol `p` (empty = valid)."""
    body = {k: v for k, v in comparator.items() if k != "comparator_hash"}
    if comparator.get("set_kind") != "comparator_selection":
        return ["comparator was not named from the comparator-selection set"]
    if digest(body) != comparator.get("comparator_hash"):
        return ["comparator record hash mismatch (edited or not produced by select_comparator)"]
    errs = []
    if comparator.get("protocol_hash") != protocol_hash(p):
        errs.append("comparator was selected under another protocol (or before the freeze)")
    if not comparator.get("selection_families"):
        errs.append("comparator record names no selection families")
    try:
        again = select_comparator(comparator["totals"], set_kind="comparator_selection",
                                  deployable=comparator["deployable"], protocol=p,
                                  selection_families=comparator["selection_families"])
    except (EvalProtocolError, KeyError, TypeError) as e:
        return errs + [f"comparator cannot be re-derived: {e}"]
    if again["comparator"] != comparator["comparator"] or again["ranking"] != comparator["ranking"]:
        errs.append("comparator does not follow the preregistered rule on its own selection totals")
    return errs


def row_problems(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Malformed paired rows (empty = well formed): every arm complete, counts consistent and within N."""
    errs, seen = [], set()
    if not rows:
        return ["no test rows"]
    for i, r in enumerate(rows):
        sid = r.get("scenario_id")
        if not isinstance(sid, str) or sid in seen:
            errs.append(f"row {i}: missing or duplicate scenario_id")
        seen.add(sid)
        n = r.get("people")
        if not isinstance(r.get("family_id"), str) or type(n) is not int or n < 1:
            errs.append(f"row {i}: family_id or people missing")
            continue
        for arm in ("raw_student", *CANDIDATES):
            o = r.get(arm)
            if not isinstance(o, Mapping) or any(type(o.get(k)) is not int or o[k] < 0 for k in _OUTCOMES):
                errs.append(f"row {i}: arm {arm} missing or not non-negative integers")
            elif o["casualties_plus_unresolved"] != o["casualties"] + o["unresolved"] or \
                    o["casualties_plus_unresolved"] > n:
                errs.append(f"row {i}: arm {arm} counts inconsistent with each other or with N")
    return errs


def tail_report(rows: Sequence[Mapping[str, Any]], arm: str, ref: str) -> dict[str, Any]:
    """Reported, not a gate (severe regressions gate the tail): worst family mean, worst scenarios, people bands."""
    fam = _diffs(rows, arm, ref, "casualties_plus_unresolved")
    per = sorted(((r[arm]["casualties_plus_unresolved"] - r[ref]["casualties_plus_unresolved"], r["scenario_id"])
                  for r in rows), reverse=True)
    bands = {}
    for lo, hi in PEOPLE_BANDS:
        sub = [r for r in rows if lo <= r["people"] <= hi]
        if sub:
            d = {m: sum(r[arm][m] - r[ref][m] for r in sub) / len(sub) for m in _OUTCOMES}
            bands[f"people {lo}-{hi}" if hi < 10 ** 9 else f"people >={lo}"] = {
                "scenarios": len(sub), "mean_diff": d, "point_estimate_worse": any(v > 0 for v in d.values())}
    return {"gate": False, "worst_family_mean_primary": max(sum(v) / len(v) for v in fam.values()),
            "worst_scenarios_primary": [{"scenario_id": s, "diff": x} for x, s in per[:5]], "people_bands": bands}


def clopper_pearson_upper(x: int, n: int, alpha: float) -> float:
    """One-sided (1 - alpha) exact upper bound of a binomial rate (bisection on the binomial CDF)."""
    if n <= 0 or not 0 <= x <= n:
        raise EvalProtocolError("invalid binomial count")
    if x == n:
        return 1.0

    def cdf(pr: float) -> float:
        return sum(math.comb(n, i) * pr ** i * (1 - pr) ** (n - i) for i in range(x + 1))
    lo, hi = x / n, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cdf(mid) > alpha else (lo, mid)
    return hi


def severe_regression(row: Mapping[str, Any], arm: str, ref: str, thresholds: Mapping[str, Any]) -> list[str]:
    """Outcomes on which ONE scenario regresses severely: worse by >= max(min_people, fraction * N)."""
    out = []
    for m, t in thresholds.items():
        if row[arm][m] - row[ref][m] >= max(t["min_people"], t["fraction_of_people"] * row["people"]):
            out.append(m)
    return out


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
    bad = row_problems(rows)
    if bad:
        raise EvalProtocolError(f"malformed test rows: {bad[:5]}")
    # Integrity of the run itself: exact frozen protocol, one-shot test use, independent comparator, isolation.
    if run.get("protocol_hash") != protocol_hash(p):
        return {"verdict": INVALID, "reasons": ["the run does not name the hash of this frozen protocol"]}
    if run.get("test_evaluations_before") != 0:
        return {"verdict": INVALID, "reasons": ["the sealed test set was evaluated before (or not recorded): "
                                                "no tuning or re-run on test"]}
    cp = comparator_problems(comparator, p)
    if cp:
        return {"verdict": INVALID, "reasons": [f"comparator: {e}" for e in cp]}
    iso = run.get("isolation")
    missing = [k for k in REQUIRED_ISOLATION if not isinstance(iso, Mapping) or k not in iso]
    if missing:
        return {"verdict": INVALID, "reasons": [f"family isolation not audited for: {missing}"]}
    if set(iso["comparator_selection"]) != set(comparator["selection_families"]):
        return {"verdict": INVALID, "reasons": ["comparator-selection families differ from the comparator record"]}
    try:
        check_test_isolation({r["family_id"] for r in rows}, {k: iso[k] for k in REQUIRED_ISOLATION})
    except EvalProtocolError as e:
        return {"verdict": INVALID, "reasons": [str(e)]}
    reasons = []
    g = p["guards"]
    if run["teacher_calls_in_student_path"] > g["teacher_calls_in_student_path_max"]:
        return {"verdict": INVALID, "reasons": ["a reference teacher ran inside the student path"]}
    if run.get("model_input_privacy_violations") is None or \
            run["model_input_privacy_violations"] > g["model_input_privacy_violations_max"]:
        return {"verdict": INVALID, "reasons": ["private simulator state reached the model input, or not audited"]}
    if run["safety_failures"] > g["safety_failures_max"]:
        reasons.append("simulator-safety failures (invalid joint maps, crashes, fail-closed steps)")
    for k in ("p50", "p95", "p99"):
        lim = g["latency_ms"][f"{k}_max"]
        if lim is None or run[f"latency_{k}_ms"] is None or run[f"latency_{k}_ms"] > lim:
            reasons.append(f"latency {k} limit not met or not measurable")
    ref, arm = comparator["comparator"], "raw_student"
    pr = p["primary"]
    kw = {"alpha": pr["alpha_one_sided"], "resamples": pr["bootstrap_resamples"], "seed": pr["bootstrap_seed"]}
    metrics = ("casualties_plus_unresolved", "casualties", "unresolved")
    diffs = {m: _diffs(rows, arm, ref, m) for m in metrics}
    ub = {m: bootstrap_upper(diffs[m], **kw) for m in metrics}
    est = {m: sum(family_means(diffs[m])) / len(diffs[m]) for m in metrics}
    if not ub["casualties_plus_unresolved"] < 0:
        reasons.append("primary: upper bound of casualties+unresolved difference is not < 0")
    if ub["casualties"] > g["casualties_upper_bound_max"] or est["casualties"] > 0:
        reasons.append("guard: casualties may be worse than the comparator")
    if ub["unresolved"] > g["unresolved_upper_bound_max"] or est["unresolved"] > 0:
        reasons.append("guard: unresolved may be worse than the comparator")
    sev = {r.get("scenario_id", str(i)): s for i, r in enumerate(rows)
           if (s := severe_regression(r, arm, ref, g["severe_regression"]))}
    sev_ub = clopper_pearson_upper(len(sev), len(rows), pr["alpha_one_sided"])
    if len(sev) > g["severe_regression_max_fraction"] * len(rows) or sev_ub > g["severe_regression_rate_upper_max"]:
        reasons.append(f"severe regressions {len(sev)} of {len(rows)} (upper bound {sev_ub:.3f})")
    # Separate outcome claims: each needs its own one-sided superiority; Holm over the two outcomes.
    po = {m: bootstrap_p(diffs[m], resamples=pr["bootstrap_resamples"], seed=pr["bootstrap_seed"])
          for m in ("casualties", "unresolved")}
    sep = holm(po, pr["alpha_one_sided"])
    claims = {"casualties_plus_unresolved_improved": not reasons,
              "casualties_improved": not reasons and sep["casualties"] and ub["casualties"] < 0,
              "unresolved_improved": not reasons and sep["unresolved"] and ub["unresolved"] < 0}
    # BOTH = intersection-union: each outcome superior at alpha on its own (no adjustment needed for "both").
    claims["both_improved"] = not reasons and ub["casualties"] < 0 and ub["unresolved"] < 0
    pv = {t: bootstrap_p(_diffs(rows, arm, t, "casualties_plus_unresolved"),
                         resamples=pr["bootstrap_resamples"], seed=pr["bootstrap_seed"]) for t in CANDIDATES}
    return {"verdict": NOT_SUPERIOR if reasons else SUPERIOR, "reasons": reasons, "comparator": ref,
            "upper_bounds": ub, "estimates": est, "severe_regressions": len(sev), "severe_detail": sev,
            "severe_rate_upper": sev_ub, "claims": claims, "separate_outcome_p": po,
            "secondary_beats_teacher_holm": holm(pv, pr["alpha_one_sided"]), "secondary_p": pv,
            "tail": tail_report(rows, arm, ref), "comparator_hash": comparator["comparator_hash"],
            "oracle": oracle_reference({r.get("scenario_id", str(i)): {t: r[t]["casualties_plus_unresolved"]
                                                                       for t in CANDIDATES}
                                        for i, r in enumerate(rows)})}
