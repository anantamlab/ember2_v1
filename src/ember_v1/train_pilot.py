"""P1-12 eight-scenario matched five-teacher TRAIN pilot: plan, authorization check and acceptance gate.

PREPARATION ONLY. Nothing here runs a teacher on a TRAIN scenario. `run_pilot` always refuses: the pilot needs
(1) this plan reviewed and frozen in its own commit, (2) a separate multi-teacher source authorization that
names the plan hash and every teacher fingerprint (`check_authorization`). The obsolete P1-09 T1 selection
receipt is not that authorization and is refused. Docs: `docs/P1_12_TRAIN_COLLECTION_PROTOCOL.md` §4.

The plan is a pure function of the committed TRAIN manifest, the held-out sets and a published salt:
hash-rank selection inside fixed strata, one scenario per family, the largest-N scenario forced in.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import s1_pipeline as sp
from .sim.canonical import digest
from .teacher.registry import CANDIDATES

PILOT_VERSION = "p1-12-train-pilot@0.1.0-draft"
PLAN_STATUS = "PROPOSED_NOT_FROZEN"
AUTH_KIND = "p1-12-multiteacher-train-pilot-authorization"
SALT = "p1-12-train-pilot@1"
SCENARIOS = 8
STRATA = {"feasible": 4, "difficult": 3, "unreachable": 1}          # generator label -> count
MAX_PER_FIRE_MODEL = 5                                               # at least 3 of each fire model
LIMITS = {"decision_cpu_s": {"T0": 5, "T1": 5, "T2": 30, "T3": 5, "T4": 5},
          "group_wall_s": 7200, "pilot_wall_s": 14400, "processes_per_teacher": 1, "retries": 0,
          "decisions_per_teacher": "scenario max_steps (one per step; no extra calls)"}
# P1-09 measured CPU over 72 scenarios (data/selection/p1-09-selection@1/outcome/selection_gate.json).
P1_09_CPU_S = {"T0": 7.09375, "T2": 7555.15625, "T3": 134.0, "T4": 68.75}
P1_09_SCENARIOS = 72
COST_SAFETY = 2.0
TIMEOUT_RULE = ("a decision over its CPU cap or a group over its wall cap is a recorded teacher failure: the group "
                "is incomplete, kept and reported, never retried or replaced; the pilot gate then fails")


class PilotBlocked(RuntimeError):
    """The real TRAIN pilot is not authorized."""


def _rank(salt: str, scenario_hash: str) -> str:
    return hashlib.sha256(f"{salt}|{scenario_hash}".encode()).hexdigest()


def select(entries: Sequence[Mapping[str, Any]], *, salt: str = SALT) -> list[dict[str, Any]]:
    """Deterministic stratified choice: largest N first, then hash rank; distinct families; fire-model cap."""
    pool = sorted(entries, key=lambda e: _rank(salt, e["scenario_hash"]))
    largest = max(pool, key=lambda e: (e["diagnostics"]["people"], _rank(salt, e["scenario_hash"])[::-1]))
    chosen: list[Mapping[str, Any]] = []
    need, fams, fire = dict(STRATA), set(), Counter()
    for e in [largest, *pool]:
        lab, fm = e["label"], e["diagnostics"]["fire_model"]
        if e in chosen or need.get(lab, 0) == 0 or e["family_id"] in fams or fire[fm] >= MAX_PER_FIRE_MODEL:
            continue
        chosen.append(e)
        need[lab] -= 1
        fams.add(e["family_id"])
        fire[fm] += 1
    if any(need.values()):
        raise ValueError(f"TRAIN cannot fill the pilot strata: missing {need}")
    return [{"scenario_id": e["scenario_id"], "scenario_hash": e["scenario_hash"], "family_id": e["family_id"],
             "lineage_id": e["lineage_id"], "core_hash": e.get("core_hash"), "label": e["label"],
             "fire_model": e["diagnostics"]["fire_model"], "people": e["diagnostics"]["people"],
             "max_steps": e["diagnostics"]["max_steps"], "rank": _rank(salt, e["scenario_hash"])}
            for e in chosen]


def cost_estimate(scenarios: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per = {t: c / P1_09_SCENARIOS for t, c in P1_09_CPU_S.items()}
    n = len(scenarios)
    t2 = per["T2"] * n * COST_SAFETY
    return {"basis": "P1-09 mean CPU per scenario (72 scenarios), x2 safety; T1 is about T0 scale",
            "per_scenario_cpu_s": {t: round(v, 2) for t, v in per.items()},
            "t2_cpu_s": round(t2, 1), "t2_cpu_s_per_scenario_limit": round(per["T2"] * COST_SAFETY, 1),
            "others_cpu_s": round((2 * per["T0"] + per["T3"] + per["T4"]) * n * COST_SAFETY, 1),   # T1 ~ T0
            "people_in_pilot": [s["people"] for s in scenarios],
            "caveat": "TRAIN N differs from P1-09; measured pilot T2 CPU decides the full-TRAIN plan",
            "modal_usd": 0}


def teacher_pins() -> dict[str, Any]:
    from .provenance import code_fingerprint
    from .teacher import registry as reg
    problems = reg.check_pins()
    out = {}
    for t, cls in reg.retained_classes().items():
        ident = reg.identity(t)
        out[t] = {"version": ident.version, "config_hash": ident.config_hash, "code_fingerprint": code_fingerprint(cls)}
    return {"teachers": out, "pin_problems": problems}


def build_plan(train: Mapping[str, Any], heldout: Mapping[str, Mapping[str, set]], *, salt: str = SALT,
               pins: Mapping[str, Any] | None = None, code: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The proposed pilot plan. Fails closed on any held-out overlap (whole TRAIN manifest audited)."""
    audit = sp.scenario_plan(train, heldout)["audit"]
    sc = select(train["scenarios"], salt=salt)
    pins = dict(pins if pins is not None else teacher_pins())
    if pins.get("pin_problems") or set(pins.get("teachers", {})) != set(CANDIDATES):
        raise ValueError(f"teacher pins are not exact: {pins.get('pin_problems')}")
    body = {
        "pilot_version": PILOT_VERSION, "status": PLAN_STATUS, "salt": salt,
        "source": {"split": "train", "manifest_hash": train["manifest_hash"], "spec_hash": train["spec_hash"],
                   "plan_hash": train["plan_hash"], "generator_revision": train["generator_revision"]},
        "selection_rule": {"strata": STRATA, "one_per_family": True, "largest_people_forced": True,
                           "max_per_fire_model": MAX_PER_FIRE_MODEL, "rank": "sha256(salt|scenario_hash)"},
        "scenarios": sc, "disjointness_audit": audit, "teacher_pins": pins, "code": dict(code or {}),
        "limits": LIMITS, "timeout_rule": TIMEOUT_RULE, "cost_estimate": cost_estimate(sc),
        "attempt_marker": {"path": "data/collection/p1-12-train-pilot@1/attempt-<n>.json", "write_once": True,
                           "written": "before the first teacher call; holds plan_hash, authorization hash, commit",
                           "rule": "an existing marker blocks a new run; attempt n+1 needs a new authorization"},
        "output_schema": "matched-multi-teacher@0.2.0 per group (multiteacher.collect report) + group manifest",
        "acceptance": acceptance_rule(),
    }
    return {**body, "plan_hash": digest(body)}


def acceptance_rule() -> dict[str, Any]:
    return {"groups": SCENARIOS, "verified_by": "multiteacher.verify_fixture equivalent on each group",
            "complete_groups": SCENARIOS, "incomplete_rate_per_stratum_max": 0.0,
            "regenerated_identical_min": 2, "leakage_failures_max": 0,
            "t2_cpu_mean_per_scenario_max": "cost_estimate.t2_cpu_s_per_scenario_limit",
            "group_wall_s_max": LIMITS["group_wall_s"], "pilot_wall_s_max": LIMITS["pilot_wall_s"],
            "on_failure": "keep and report; no substitution, no retry; re-plan before full TRAIN"}


def check_authorization(auth: Mapping[str, Any] | None, plan: Mapping[str, Any]) -> list[str]:
    """Problems with a multi-teacher pilot authorization (empty list = acceptable)."""
    from .receipt_review import _PLACEHOLDER
    if not auth:
        return ["no multi-teacher TRAIN pilot authorization"]
    errs = []
    text = json.dumps(auth, sort_keys=True).lower()
    if "p1-09" in text or str(auth.get("kind", "")).startswith(("s1", "selection")):
        errs.append("the P1-09 T1 selection receipt is not a multi-teacher authorization")
    if auth.get("kind") != AUTH_KIND:
        errs.append(f"kind must be {AUTH_KIND}")
    if plan.get("status") != "FROZEN":
        errs.append("the pilot plan is not FROZEN")
    if auth.get("plan_hash") != plan.get("plan_hash"):
        errs.append("authorization names another plan hash")
    if auth.get("scenarios") != SCENARIOS:
        errs.append(f"authorization must cover exactly {SCENARIOS} scenarios")
    if not isinstance(auth.get("attempt"), int) or auth["attempt"] < 1:
        errs.append("authorization must name the attempt number")
    want = {t: v["code_fingerprint"] for t, v in plan["teacher_pins"]["teachers"].items()}
    if auth.get("teacher_fingerprints") != want:
        errs.append("teacher fingerprints differ from the plan")
    rev = auth.get("reviewers") or []
    if auth.get("status") != "approved" or not rev or any(_PLACEHOLDER.match(str(r).strip()) for r in rev):
        errs.append("authorization is not approved by named reviewers")
    return errs


def pilot_acceptance(plan: Mapping[str, Any], groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Gate over the per-group summaries of a finished pilot (synthetic in tests; no real pilot exists)."""
    problems = []
    want = [s["scenario_id"] for s in plan["scenarios"]]
    got = [g["scenario_id"] for g in groups]
    if sorted(got) != sorted(want) or len(got) != len(set(got)):
        problems.append("groups differ from the plan's scenarios (substitution or duplicate)")
    label = {s["scenario_id"]: s["label"] for s in plan["scenarios"]}
    strata: dict[str, list[int]] = {}
    for g in groups:
        ok = g.get("verified") is True and g.get("group_complete") is True and not g.get("failed_teachers")
        strata.setdefault(label.get(g["scenario_id"], "unplanned"), [0, 0])
        strata[label.get(g["scenario_id"], "unplanned")][0] += 0 if ok else 1
        strata[label.get(g["scenario_id"], "unplanned")][1] += 1
        if not ok:
            problems.append(f"{g['scenario_id']}: not a verified complete group ({g.get('failed_teachers')})")
        if g.get("leakage_ok") is not True:
            problems.append(f"{g['scenario_id']}: leakage gate not passed")
        if not isinstance(g.get("wall_s"), (int, float)) or g["wall_s"] > LIMITS["group_wall_s"]:
            problems.append(f"{g['scenario_id']}: group wall cap exceeded or not measured")
    rate = {k: v[0] / v[1] for k, v in strata.items()}
    if any(r > 0 for r in rate.values()):
        problems.append(f"incomplete-group rate per stratum {rate}")
    if sum(1 for g in groups if g.get("regenerated_identical") is True) < 2:
        problems.append("fewer than 2 groups regenerated with an identical content hash")
    t2 = [g.get("t2_cpu_s") for g in groups]
    lim = plan["cost_estimate"]["t2_cpu_s_per_scenario_limit"]
    if any(not isinstance(x, (int, float)) for x in t2) or (t2 and sum(t2) / len(t2) > lim):
        problems.append(f"T2 CPU per scenario above {lim} s or not measured: re-plan before full TRAIN")
    walls = [g.get("wall_s") or 0 for g in groups]
    if sum(walls) > LIMITS["pilot_wall_s"]:
        problems.append("pilot wall cap exceeded")
    return {"pass": not problems, "problems": problems, "incomplete_rate_per_stratum": rate,
            "t2_cpu_mean_s": round(sum(x for x in t2 if isinstance(x, (int, float))) / max(1, len(t2)), 1)}


def check_attempt_marker(out_dir: str | Path, attempt: int) -> None:
    """Refuse when any attempt marker already exists, or the attempt number is not the next one."""
    existing = sorted(Path(out_dir).glob("attempt-*.json"))
    if existing:
        raise PilotBlocked(f"attempt marker exists ({existing[-1].name}); a new attempt needs a new authorization")
    if attempt != 1:
        raise PilotBlocked("the first attempt must be attempt 1")


def run_pilot(*_: Any, **__: Any) -> None:
    raise PilotBlocked("real TRAIN pilot execution is not authorized (plan not frozen; no multi-teacher authorization)")


def real_heldout(repo: Path) -> dict[str, dict[str, set]]:
    """Every historical set except TRAIN itself, plus the P1-09 confirmation set (regenerated, hash-checked)."""
    from .teacher import confirmation as C
    hist, _, _ = C.historical_sets()
    held = {ns: sp.heldout_from_entries(e) for ns, e in hist.items() if ns != "p1-02-v1/train"}
    conf = json.loads((repo / "data/selection/p1-09-selection@1/manifest.json").read_text(encoding="utf-8"))
    held["p1-09-selection@1"] = sp.heldout_from_manifest(conf)
    return held


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ember_v1.train_pilot")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("plan", help="write the PROPOSED pilot plan (runs no teacher)")
    pl.add_argument("--out", required=True)
    sub.add_parser("run", help="always refused in this draft")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run_pilot()
    repo = Path(__file__).resolve().parents[2]
    train = json.loads((repo / "data/splits/p1-02-v1/train.json").read_text(encoding="utf-8"))
    import subprocess
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    plan = build_plan(train, real_heldout(repo), code={"commit_at_proposal": commit})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(plan, indent=1, sort_keys=True, default=sorted) + "\n", encoding="utf-8",
                           newline="\n")
    sys.stdout.write(json.dumps({"plan_hash": plan["plan_hash"], "scenarios": [s["scenario_id"] for s in
                                 plan["scenarios"]], "t2_cpu_s": plan["cost_estimate"]["t2_cpu_s"]}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
