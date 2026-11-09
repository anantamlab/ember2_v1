"""P1-12 eight-scenario matched five-teacher TRAIN pilot: plan, gates, supervised execution, acceptance.

The real execution path exists but refuses unless EVERY gate passes (`run_preconditions`):
1. a FROZEN plan read from a trusted git commit, its hash self-consistent and equal to a live rebuild
   (scenario hashes, teacher pins, simulator/generator/collector fingerprints, leakage audits);
2. the eight TRAIN scenarios regenerate from the committed manifest with their recorded hashes;
3. all five teacher identities pinned (`registry.check_pins() == []`) and equal to the plan;
4. a SEPARATE multi-teacher authorization record, read from a trusted git commit, naming the exact plan
   hash, the eight scenarios, attempt 1, all five fingerprints and the execution commit;
5. that record's approval verified on GitHub: a merged pull request whose latest review by an allowlisted
   approver (not the PR author) is APPROVED on a commit with byte-identical authorization content.
   Free-text reviewer names are never evidence;
6. HEAD is the authorized execution commit, is on the trusted ref, descends from the freeze, and the
   checkout is clean;
7. no attempt marker, log, group output or evidence exists (no retries, no duplicates).
Then an atomic attempt marker is written BEFORE the first teacher call. Groups run one after another
under `pilot_supervisor` (one killable process per teacher). Docs: P1_12_TRAIN_COLLECTION_PROTOCOL.md §4.

The plan is a pure function of the committed TRAIN manifest, the held-out sets and a published salt:
hash-rank selection inside fixed strata, one scenario per family, the largest-N scenario forced in.
Selection reads only manifest fields (`SELECTION_INPUTS`); no teacher outcome exists or is consulted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import s1_pipeline as sp
from . import trust
from .sim.canonical import digest
from .teacher.registry import CANDIDATES

PILOT_VERSION = "p1-12-train-pilot@0.2.0-draft"
PLAN_STATUS = "PROPOSED_NOT_FROZEN"
FROZEN = "FROZEN"
AUTH_KIND = "p1-12-multiteacher-train-pilot-authorization"
AUTH_VERSION = "1.0.0"
AUTH_PENDING, AUTH_APPROVED = "PENDING", "APPROVED"
ATTEMPT_VERSION = "p1-12-train-pilot-attempt@1.0.0"
EVIDENCE_VERSION = "p1-12-train-pilot-evidence@1.0.0"
SALT = "p1-12-train-pilot@1"
SCENARIOS = 8
ATTEMPT = 1
STRATA = {"feasible": 4, "difficult": 3, "unreachable": 1}          # generator label -> count
MAX_PER_FIRE_MODEL = 5                                               # at least 3 of each fire model
SELECTION_INPUTS = ("scenario_hash", "label", "family_id", "diagnostics.fire_model", "diagnostics.people")
OUT_DIR = "data/collection/p1-12-train-pilot@1"
PLAN_FILE = f"{OUT_DIR}/plan_PROPOSED.json"
FROZEN_PLAN_FILE = f"{OUT_DIR}/plan.json"
AUTH_TEMPLATE_FILE = f"{OUT_DIR}/authorization_TEMPLATE.json"
AUTH_FILE = f"{OUT_DIR}/authorization.json"
LIMITS = {"decision_cpu_s": {"T0": 5, "T1": 5, "T2": 30, "T3": 5, "T4": 5},
          "group_wall_s": 7200, "pilot_wall_s": 14400, "processes_per_teacher": 1, "retries": 0,
          "decisions_per_teacher": "one per simulator step until termination (at most max_steps)",
          "process_model": "one spawned, killable process per teacher per group; groups run in plan order"}
FAILURE_POLICY = {
    "teacher_failures": ["teacher_exception", "invalid_proposal", "decision_cpu_cap", "group_wall_cap",
                         "pilot_wall_cap", "worker_crashed", "worker_unresponsive", "missing_policy_response",
                         "code_drift", "identity_drift", "interrupted", "supervisor_error"],
    "effect": "the matched group is incomplete: kept as diagnostic evidence with every streamed decision, "
              "never retried, never replaced, never training-eligible; the acceptance gate then fails",
    "integrity_error": "a worker stream that does not replay in the supervisor's simulator is a collector "
                       "fault: the group is recorded as integrity_error and the gate fails",
    "later_groups": "still run in plan order while the pilot deadline holds (diagnostic cost evidence); "
                    "after an interrupt or the pilot deadline no new group starts (recorded not_started)",
    "substitutes": "none", "retries": 0}
# Approval binding. Free-text names never count; GitHub records the review. GitHub forbids a PR author
# from approving their own PR, so author != approver holds by construction as well as by this check.
APPROVAL_POLICY = {
    "kind": "github_pr_review", "repository": "anantamlab/ember_v1", "base": "main",
    "approver_logins": ["sankalphs", "Sathvikar01"],
    "rules": ["the pull request that adds the authorization record is merged into main",
              "its merge commit is in the history of the trusted ref",
              "the latest non-comment review by an allowlisted approver is APPROVED",
              "the approver is not the pull request author",
              "the approved commit (review commit_id) holds a byte-identical authorization record",
              "the record names that approver login (a record cannot name its own commit)"],
    "trust_boundary": "binds approval to a GitHub account action; it cannot tell the owner from anyone "
                      "holding that account's token (see the protocol doc, Trust boundary)"}
P1_09_EVIDENCE = "data/selection/p1-09-selection@1/evidence.json"
N_BINS = ((0, 12), (12, 40), (40, 100), (100, 10 ** 6))
T2_BUDGET_S = 20.0                       # frozen T2 cpu_budget_ms / 1000 (internal fallback per decision)


class PilotBlocked(RuntimeError):
    """The real TRAIN pilot is not authorized or a gate failed."""


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


def design_check(sc: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The pilot design facts, recomputed from the selection (a test pins the expected values)."""
    return {"scenarios": len(sc), "families": len({s["family_id"] for s in sc}),
            "labels": dict(sorted(Counter(s["label"] for s in sc).items())),
            "fire_models": dict(sorted(Counter(s["fire_model"] for s in sc).items())),
            "people": sorted(s["people"] for s in sc), "max_steps_total": sum(s["max_steps"] for s in sc),
            "groups_per_scenario": 1, "fire_seeds_per_scenario": 1,
            "claim_scope": "pipeline feasibility; no population-level teacher comparison from 8 cases"}


def regeneration_ids(sc: Sequence[Mapping[str, Any]]) -> list[str]:
    """The two preregistered regeneration groups: smallest N, ties by rank (cheapest, no T2 budget hits)."""
    return [s["scenario_id"] for s in sorted(sc, key=lambda s: (s["people"], s["rank"]))[:2]]


# ----- cost ---------------------------------------------------------------------------------------------------
def p1_09_cost_basis(repo: Path) -> list[dict[str, Any]]:
    """Per-scenario P1-09 T2 CPU, people and decisions from the committed evidence (no teacher run)."""
    ev = json.loads((repo / P1_09_EVIDENCE).read_text(encoding="utf-8"))
    out = []
    for r in ev["rows"]:
        t2 = r["results"]["T2"]
        out.append({"people": t2["people"], "t2_cpu_s": t2["policy_cpu_s"], "decisions": t2["decisions"],
                    "max_decision_cpu_ms": t2["max_decision_cpu_ms"],
                    "budget_limited": t2["budget_limited_decisions"],
                    "others_cpu_s": sum(r["results"][t]["policy_cpu_s"] for t in r["results"] if t != "T2")})
    return out


def _bin(n: int) -> tuple[int, int]:
    return next(b for b in N_BINS if b[0] <= n < b[1])


def cost_estimate(scenarios: Sequence[Mapping[str, Any]], basis: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """T2 CPU stratified by people count (P1-09 per-scenario measurements in the same N bin).

    central = sum of bin means; upper = sum of bin maxima (the acceptance limit); worst case = every
    decision at the frozen 20 s T2 budget for max_steps decisions (not expected; the caps then act).
    """
    bins: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for r in basis:
        bins.setdefault(_bin(r["people"]), []).append(r)
    per, central, upper, others = [], 0.0, 0.0, 0.0
    for s in scenarios:
        b = bins.get(_bin(s["people"]))
        if not b:
            raise ValueError(f"no P1-09 measurement in the N bin of {s['scenario_id']}")
        cpu = [r["t2_cpu_s"] for r in b]
        row = {"scenario_id": s["scenario_id"], "people": s["people"], "n_bin": list(_bin(s["people"])),
               "bin_scenarios": len(b), "t2_cpu_s_mean": round(statistics.fmean(cpu), 1),
               "t2_cpu_s_max": round(max(cpu), 1),
               "t2_budget_limited_share": round(sum(1 for r in b if r["budget_limited"]) / len(b), 3),
               "t2_worst_case_s": s["max_steps"] * T2_BUDGET_S}
        per.append(row)
        central += row["t2_cpu_s_mean"]
        upper += row["t2_cpu_s_max"]
        others += statistics.fmean(r["others_cpu_s"] for r in b)
    v1 = 104.93 * len(scenarios) * 2
    return {"basis": f"P1-09 per-scenario T2 CPU ({len(basis)} scenarios, {P1_09_EVIDENCE}) in the same "
                     f"people bin {[list(b) for b in N_BINS]}",
            "per_scenario": per, "t2_cpu_s_central": round(central, 1), "t2_cpu_s_upper": round(upper, 1),
            "t2_cpu_s_total_limit": round(upper, 1), "others_cpu_s_central": round(others, 1),
            "t2_worst_case_s": sum(r["t2_worst_case_s"] for r in per),
            "wall_s_expected": round(central, 1),
            "superseded_v0_1_estimate": {"t2_cpu_s": round(v1, 1), "rule": "global mean x 8 x 2",
                                         "why_replaced": "ignored N: 3 of 8 pilot scenarios have N >= 100, "
                                                         "where P1-09 T2 CPU is about 4x its global mean"},
            "caveats": ["one host; T2 internal budget is thread CPU (20 s) so slower hosts hit it more",
                        "a T2 budget-limited decision depends on host speed: that group may not regenerate",
                        "worst case exceeds the pilot wall cap; then the caps end the pilot (recorded)"],
            "modal_usd": 0}


# ----- identities ---------------------------------------------------------------------------------------------
def teacher_pins() -> dict[str, Any]:
    from .provenance import code_fingerprint
    from .teacher import registry as reg
    problems = reg.check_pins()
    out = {}
    for t, cls in reg.retained_classes().items():
        ident = reg.identity(t)
        out[t] = {"version": ident.version, "config_hash": ident.config_hash,
                  "code_fingerprint": code_fingerprint(cls), "seed": cls().seed}
    return {"teachers": out, "pin_problems": problems}


def code_identities() -> dict[str, Any]:
    from . import multiteacher as mt
    from . import pilot_supervisor as ps
    from .gen.spec import GENERATOR_REVISION
    from .sim.simulator import OBSERVATION_VERSION
    fp = sp.module_fingerprint
    return {"simulator": {"observation_version": OBSERVATION_VERSION,
                          "module_fingerprint": fp("ember_v1.sim.simulator")},
            "generator": {"revision": GENERATOR_REVISION, "module_fingerprint": fp("ember_v1.gen.generator"),
                          "splits_fingerprint": fp("ember_v1.gen.splits")},
            "collector": {"schema_version": mt.SCHEMA_VERSION, "module_fingerprint": fp("ember_v1.multiteacher"),
                          "supervisor_version": ps.SUPERVISOR_VERSION,
                          "supervisor_fingerprint": fp("ember_v1.pilot_supervisor"),
                          "runner_fingerprint": fp("ember_v1.train_pilot")}}


def fingerprints_of(pins: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    return {t: {k: v[k] for k in ("version", "config_hash", "code_fingerprint")}
            for t, v in pins["teachers"].items()}


# ----- plan ---------------------------------------------------------------------------------------------------
def plan_body(plan: Mapping[str, Any]) -> dict[str, Any]:
    """The hashed part of a plan: everything except `status` and `plan_hash` (freezing keeps the hash)."""
    return {k: v for k, v in plan.items() if k not in ("status", "plan_hash")}


def build_plan(train: Mapping[str, Any], heldout: Mapping[str, Mapping[str, set]], *, salt: str = SALT,
               pins: Mapping[str, Any] | None = None, code: Mapping[str, Any] | None = None,
               geometry: Mapping[str, Any] | None = None, cost_basis: Sequence[Mapping[str, Any]] | None = None
               ) -> dict[str, Any]:
    """The proposed pilot plan. Fails closed on any held-out overlap (whole TRAIN manifest audited)."""
    audit = sp.scenario_plan(train, heldout)["audit"]
    sc = select(train["scenarios"], salt=salt)
    pins = dict(pins if pins is not None else teacher_pins())
    if pins.get("pin_problems") or set(pins.get("teachers", {})) != set(CANDIDATES):
        raise ValueError(f"teacher pins are not exact: {pins.get('pin_problems')}")
    if geometry is not None and not geometry.get("pass"):
        raise ValueError(f"geometry leakage audit failed: {geometry.get('hits')}")
    regen = regeneration_ids(sc)
    body = {
        "pilot_version": PILOT_VERSION, "salt": salt,
        "source": {"split": "train", "manifest_hash": train["manifest_hash"], "spec_hash": train["spec_hash"],
                   "plan_hash": train["plan_hash"], "generator_revision": train["generator_revision"]},
        "selection_rule": {"strata": STRATA, "one_per_family": True, "largest_people_forced": True,
                           "max_per_fire_model": MAX_PER_FIRE_MODEL, "rank": "sha256(salt|scenario_hash)",
                           "inputs": list(SELECTION_INPUTS), "teacher_outcomes_consulted": False},
        "scenarios": sc, "design": design_check(sc),
        "disjointness_audit": audit, "geometry_audit": dict(geometry or {"not_computed": True}),
        "identities": dict(code or {}), "teacher_pins": pins,
        "attempt": ATTEMPT, "limits": LIMITS, "failure_policy": FAILURE_POLICY,
        "regeneration": {"scenario_ids": regen, "rule": "the two smallest-N scenarios (ties by rank) are run a "
                         "second time inside the same attempt, after all eight groups, under the same caps; "
                         "both must reproduce content_hash", "counts_against": "pilot_wall_s"},
        "cost_estimate": cost_estimate(sc, cost_basis) if cost_basis else {"not_computed": True},
        "attempt_marker": {"path": f"{OUT_DIR}/attempt-<n>.json", "write_once": True, "atomic": "open(x)+fsync",
                           "written": "after every gate, before the first teacher call; holds the plan hash, "
                                      "authorization sha256 and commit, execution commit, limits, platform",
                           "rule": "any existing marker, log, group or evidence file blocks a run; attempt n+1 "
                                   "needs a new authorization naming n+1 and a new plan version"},
        "output": {"attempt_dir": f"{OUT_DIR}/attempt-<n>/", "log": "log.jsonl (append, fsync per line)",
                   "groups": "groups/<scenario_id>.json (write-once, matched-multi-teacher@0.2.0, scope "
                             "p1-12-train-pilot@1, training_eligible false)",
                   "regenerations": "regen/<scenario_id>.json", "evidence": "evidence.json (write-once)"},
        "acceptance": acceptance_rule(),
        "authorization_policy": APPROVAL_POLICY,
    }
    return {**body, "status": PLAN_STATUS, "plan_hash": digest(body)}


def acceptance_rule() -> dict[str, Any]:
    return {"groups": SCENARIOS, "verified_by": "multiteacher.verify_fixture on each group",
            "complete_groups": SCENARIOS, "incomplete_rate_per_stratum_max": 0.0,
            "regenerated_identical": "both regeneration.scenario_ids", "leakage_failures_max": 0,
            "t2_cpu_s_total_max": "cost_estimate.t2_cpu_s_total_limit",
            "group_wall_s_max": LIMITS["group_wall_s"], "pilot_wall_s_max": LIMITS["pilot_wall_s"],
            "on_failure": "keep and report; no substitution, no retry; re-plan before full TRAIN"}


def check_plan(plan: Mapping[str, Any], *, frozen: bool = True) -> list[str]:
    errs = []
    if digest(plan_body(plan)) != plan.get("plan_hash"):
        errs.append("plan hash does not match its content (tampered or stale)")
    if frozen and plan.get("status") != FROZEN:
        errs.append("the pilot plan is not FROZEN")
    sc = plan.get("scenarios") or []
    if len(sc) != SCENARIOS or len({s.get("scenario_id") for s in sc}) != SCENARIOS:
        errs.append(f"the plan must hold exactly {SCENARIOS} distinct scenarios")
    if plan.get("source", {}).get("split") != "train":
        errs.append("the plan source is not the TRAIN split")
    if plan.get("attempt") != ATTEMPT:
        errs.append(f"this plan version covers attempt {ATTEMPT} only")
    return errs


# ----- authorization ------------------------------------------------------------------------------------------
def authorization_template(plan: Mapping[str, Any]) -> dict[str, Any]:
    """The PENDING record the owner completes. Never approved here; no signature is invented."""
    return {"kind": AUTH_KIND, "authorization_version": AUTH_VERSION, "status": AUTH_PENDING,
            "plan_hash": plan["plan_hash"], "plan_version": plan["pilot_version"],
            "plan_freeze": {"commit": "<40-hex commit that adds the FROZEN plan>", "path": FROZEN_PLAN_FILE},
            "attempt": ATTEMPT,
            "scenarios": [{"scenario_id": s["scenario_id"], "scenario_hash": s["scenario_hash"]}
                          for s in plan["scenarios"]],
            "teacher_fingerprints": fingerprints_of(plan["teacher_pins"]),
            "execution_commit": "<40-hex reviewed commit on main to check out and run>",
            "limits": plan["limits"],
            "reviewer": {"github_login": "<approver login from the plan allowlist>"},
            "approval": {"kind": APPROVAL_POLICY["kind"], "repository": APPROVAL_POLICY["repository"],
                         "pull_request": None},
            "statement": "I reviewed the frozen plan and authorize exactly one attempt (attempt 1) of the "
                         "eight-scenario five-teacher TRAIN pilot on CPU. No Modal, GPU, training or benchmark."}


def check_authorization(auth: Mapping[str, Any] | None, plan: Mapping[str, Any]) -> list[str]:
    """Content problems of a multi-teacher pilot authorization (empty list = acceptable content).

    Content alone never proves approval; `verify_approval` checks the GitHub review.
    """
    if not auth:
        return ["no multi-teacher TRAIN pilot authorization"]
    errs = []
    text = json.dumps(auth, sort_keys=True).lower()
    if "p1-09" in text or str(auth.get("kind", "")).startswith(("s1", "selection")):
        errs.append("the P1-09 T1 selection receipt is not a multi-teacher authorization")
    if auth.get("kind") != AUTH_KIND or auth.get("authorization_version") != AUTH_VERSION:
        errs.append(f"kind must be {AUTH_KIND} version {AUTH_VERSION}")
    if plan.get("status") != FROZEN:
        errs.append("the pilot plan is not FROZEN")
    if auth.get("plan_hash") != plan.get("plan_hash"):
        errs.append("authorization names another plan hash")
    want_sc = [{"scenario_id": s["scenario_id"], "scenario_hash": s["scenario_hash"]} for s in plan["scenarios"]]
    if auth.get("scenarios") != want_sc or len(want_sc) != SCENARIOS:
        errs.append(f"authorization must list exactly the plan's {SCENARIOS} scenarios and hashes")
    if type(auth.get("attempt")) is not int or auth["attempt"] != ATTEMPT:
        errs.append(f"authorization must name attempt {ATTEMPT}")
    if auth.get("teacher_fingerprints") != fingerprints_of(plan["teacher_pins"]):
        errs.append("teacher fingerprints differ from the plan")
    if auth.get("limits") != plan.get("limits"):
        errs.append("authorization limits differ from the plan")
    for k in ("execution_commit",):
        if not isinstance(auth.get(k), str) or not trust._HEX40.match(auth[k]):
            errs.append(f"{k} must be a full 40-hex commit")
    pf = auth.get("plan_freeze") or {}
    if not isinstance(pf.get("commit"), str) or not trust._HEX40.match(pf["commit"]) \
            or pf.get("path") != FROZEN_PLAN_FILE:
        errs.append(f"plan_freeze must name a 40-hex commit and {FROZEN_PLAN_FILE}")
    rv, ap = auth.get("reviewer") or {}, auth.get("approval") or {}
    if auth.get("status") != AUTH_APPROVED:
        errs.append(f"authorization status is {auth.get('status')!r}, not {AUTH_APPROVED}")
    if rv.get("github_login") not in APPROVAL_POLICY["approver_logins"]:
        errs.append("reviewer login is not an allowlisted approver")
    if ap.get("kind") != APPROVAL_POLICY["kind"] or ap.get("repository") != APPROVAL_POLICY["repository"] \
            or type(ap.get("pull_request")) is not int:
        errs.append("approval must name the GitHub pull request that carries the record")
    return errs


def gh_api(path: str) -> Any:
    r = subprocess.run(["gh", "api", path], capture_output=True, timeout=60)
    if r.returncode != 0:
        raise ValueError(f"GitHub API {path}: {r.stderr.decode('utf-8', 'replace')[:200]}")
    return json.loads(r.stdout.decode("utf-8"))


def verify_approval(repo: str | Path, auth: Mapping[str, Any], auth_meta: Mapping[str, Any], *,
                    trusted_ref: str = "origin/main", gh: Callable[[str], Any] = gh_api) -> list[str]:
    """Problems with the GitHub review that must back the record (empty list = verified owner action)."""
    pol, errs = APPROVAL_POLICY, []
    n = (auth.get("approval") or {}).get("pull_request")
    login = (auth.get("reviewer") or {}).get("github_login")
    if type(n) is not int:
        return ["no pull request to verify"]
    try:
        pr = gh(f"repos/{pol['repository']}/pulls/{n}")
        reviews = gh(f"repos/{pol['repository']}/pulls/{n}/reviews?per_page=100")
    except (ValueError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as e:
        return [f"GitHub approval cannot be verified (fail closed): {e}"]
    author = (pr.get("user") or {}).get("login")
    if not pr.get("merged") or (pr.get("base") or {}).get("ref") != pol["base"]:
        errs.append(f"pull request #{n} is not merged into {pol['base']}")
    merge = pr.get("merge_commit_sha")
    if not merge or not trust.is_ancestor(repo, merge, trusted_ref):
        errs.append(f"the merge commit of #{n} is not in {trusted_ref}")
    if not trust.is_ancestor(repo, auth_meta["commit"], trusted_ref):
        errs.append("the authorization commit is not in the trusted history")
    if not isinstance(reviews, list) or len(reviews) >= 100:
        errs.append("review list missing or not complete")
        reviews = []
    latest: dict[str, Mapping[str, Any]] = {}
    for r in sorted(reviews, key=lambda r: (r.get("submitted_at") or "", r.get("id") or 0)):
        who = (r.get("user") or {}).get("login")
        if r.get("state") in ("APPROVED", "CHANGES_REQUESTED", "DISMISSED") and who:
            latest[who] = r
    ok = [w for w, r in latest.items() if r.get("state") == "APPROVED" and w in pol["approver_logins"]
          and w != author and (r.get("user") or {}).get("type") == "User"]
    commit = latest[login].get("commit_id") if login in ok else None
    if login not in ok:
        errs.append(f"no current APPROVED review by {login!r} (allowlisted, not the PR author {author!r}); "
                    f"verified approvers: {sorted(ok)}")
    elif not isinstance(commit, str) or not trust._HEX40.match(commit):
        errs.append("the approving review names no commit")
    else:
        shown = trust._git(repo, "show", f"{commit}:{auth_meta['path']}")
        if shown.returncode != 0:
            errs.append(f"the approved commit {commit[:12]} is not available locally (fetch pull/{n}/head)")
        elif trust.sha256_lf(shown.stdout) != auth_meta["sha256"]:
            errs.append("the approved commit holds a different authorization record")
    return errs


# ----- scenarios and held-out context -------------------------------------------------------------------------
def heldout_context(repo: Path) -> tuple[dict[str, dict[str, set]], list[Any], dict[str, Any]]:
    """(held-out id sets, held-out maps for the geometry audit, sources). Regenerated and hash-checked."""
    from .teacher import confirmation as C
    hist, families, _ = C.historical_sets()
    held = {ns: sp.heldout_from_entries(e) for ns, e in hist.items() if ns != "p1-02-v1/train"}
    conf = json.loads((repo / "data/selection/p1-09-selection@1/manifest.json").read_text(encoding="utf-8"))
    held["p1-09-selection@1"] = sp.heldout_from_manifest(conf)
    held_fams = set().union(*(h["family"] for h in held.values()))
    maps = [_Map(f"family:{f.family_id}", f.family_id, f.topology_hash, f.signature)
            for f in families if f.family_id in held_fams]
    from .gen.generator import generate_family
    from .gen.spec import GeneratorSpec
    spec = GeneratorSpec.from_dict(conf["plan"]["spec"])
    for f in conf["families"]:
        fam = generate_family(spec, f["family_seed"])[0]
        maps.append(_Map("p1-09-selection@1", fam.family_id, fam.topology_hash, fam.signature))
    p102 = len(maps) - len(conf["families"])
    maps.extend(C.pilot_maps())
    return held, maps, {"p1_02_heldout_family_maps": p102, "p1_09_family_maps": len(conf["families"]),
                        "pilot_set_maps": len(maps) - p102 - len(conf["families"])}


class _Map:
    __slots__ = ("set", "scenario_id", "topology_hash", "signature")

    def __init__(self, ns: str, sid: str, topo: str, sig: Any) -> None:
        self.set, self.scenario_id, self.topology_hash, self.signature = ns, sid, topo, sig


def _train_spec() -> Any:
    from .gen.splits import load_index, plan_from_index
    repo = Path(__file__).resolve().parents[2]
    return plan_from_index(load_index(repo / "data/splits/p1-02-v1/index.json")).spec


def train_families(train: Mapping[str, Any], family_ids: Sequence[str]) -> dict[str, Any]:
    """Regenerate the selected TRAIN families (map geometry) and check their ids and topology hashes."""
    from .gen.generator import build_family
    spec = _train_spec()
    if spec.spec_hash() != train["spec_hash"]:
        raise ValueError("TRAIN manifest spec differs from the committed index spec")
    fams = {f["family_id"]: f for f in train["families"]}
    out = {}
    for fid in family_ids:
        f = fams[fid]
        fam = build_family(spec, f["family_seed"], f["attempt"])
        if fam.family_id != fid or fam.topology_hash != f["topology_hash"]:
            raise ValueError(f"TRAIN family {fid} does not regenerate")
        out[fid] = fam
    return out


def geometry_audit(selected: Sequence[Mapping[str, Any]], fams: Mapping[str, Any], maps: Sequence[Any]
                   ) -> dict[str, Any]:
    """Every pilot map vs every held-out map and vs each other: exact topology (8 symmetries + exit renaming),
    near duplicate, wall near duplicate and their scale-free variants (`confirmation.pilot_match`)."""
    from .teacher.confirmation import _free_distance, pilot_match
    hits, nearest = [], {}
    for s in selected:
        fam = fams[s["family_id"]]
        best = None
        for m in maps:
            rule = pilot_match(fam, m)
            if rule:
                hits.append({"scenario_id": s["scenario_id"], "against": m.set, "id": m.scenario_id, "rule": rule})
            d = _free_distance(fam.signature, m.signature)
            if best is None or d < best[:2]:
                best = (*d, m)
        nearest[s["scenario_id"]] = {"against": best[2].set, "id": best[2].scenario_id,
                                     "wall_distance": round(best[0], 6), "exit_distance": round(best[1], 6)}
    ids = [s["family_id"] for s in selected]
    within = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            rule = pilot_match(fams[a], _Map("pilot", b, fams[b].topology_hash, fams[b].signature))
            if rule:
                within.append({"a": a, "b": b, "rule": rule})
    return {"maps_compared": len(maps), "sets": dict(sorted(Counter(m.set.split(":")[0] for m in maps).items())),
            "rules": ["topology_duplicate", "near_duplicate", "wall_near_duplicate", "scale_free_near_duplicate",
                      "scale_free_wall_near_duplicate"],
            "symmetries": "all 8 rotations/reflections; topology hash also ignores exit renaming",
            "hits": hits, "within_pilot_hits": within, "nearest": nearest, "pass": not hits and not within}


def pilot_scenarios(train: Mapping[str, Any], plan: Mapping[str, Any]) -> list[tuple[dict[str, Any], Any]]:
    """Regenerate the eight planned TRAIN scenarios from the committed manifest; check every hash."""
    from .gen.splits import materialize
    spec = _train_spec()
    want = {s["scenario_id"]: s for s in plan["scenarios"]}
    got = {}
    for e, g in materialize(train, spec):
        if e["scenario_id"] in want:
            s = want[e["scenario_id"]]
            if (g.scenario.content_hash(), e["lineage_id"], e["family_id"]) != \
                    (s["scenario_hash"], s["lineage_id"], s["family_id"]):
                raise ValueError(f"{e['scenario_id']}: regenerated scenario differs from the plan")
            got[e["scenario_id"]] = (dict(e), g.scenario)
    if set(got) != set(want):
        raise ValueError(f"planned scenarios missing from TRAIN: {sorted(set(want) - set(got))}")
    return [got[s["scenario_id"]] for s in plan["scenarios"]]


# ----- attempt and preconditions ------------------------------------------------------------------------------
def attempt_paths(out_dir: Path, attempt: int = ATTEMPT) -> dict[str, Path]:
    d = out_dir / f"attempt-{attempt}"
    return {"marker": out_dir / f"attempt-{attempt}.json", "dir": d, "log": d / "log.jsonl",
            "groups": d / "groups", "regen": d / "regen", "evidence": d / "evidence.json"}


def check_attempt_marker(out_dir: str | Path, attempt: int) -> None:
    """Refuse when any attempt marker or attempt output exists, or the attempt number is not 1."""
    out_dir = Path(out_dir)
    existing = sorted(out_dir.glob("attempt-*.json")) + sorted(p for p in out_dir.glob("attempt-*") if p.is_dir())
    if existing:
        raise PilotBlocked(f"attempt evidence exists ({existing[-1].name}); a new attempt needs a new authorization")
    if attempt != ATTEMPT:
        raise PilotBlocked(f"the first attempt must be attempt {ATTEMPT}")


def write_once(path: Path, doc: Mapping[str, Any]) -> None:
    """Atomic exclusive create + fsync; an existing file is never overwritten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        f = open(path, "x", encoding="utf-8", newline="\n")
    except FileExistsError:
        raise PilotBlocked(f"{path} exists: duplicate output refused") from None
    with f:
        f.write(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def parse_ref(text: str) -> dict[str, str]:
    commit, _, path = text.partition(":")
    return {"kind": "git", "commit": commit, "path": path}


def run_preconditions(repo: Path, plan_ref: Mapping[str, Any], auth_ref: Mapping[str, Any], *,
                      trusted_ref: str = "origin/main", out_dir: Path | None = None,
                      gh: Callable[[str], Any] = gh_api, live: Callable[[], Mapping[str, Any]] | None = None
                      ) -> dict[str, Any]:
    """Every gate before any teacher call. Raise PilotBlocked with all reasons."""
    from .teacher import confirmation as C
    from .teacher import registry as reg
    out_dir = out_dir or repo / OUT_DIR
    reasons: list[str] = []
    plan = auth = auth_meta = None
    try:
        plan, _ = trust.load_trusted_json(repo, plan_ref, trusted_ref=trusted_ref)
        if plan_ref.get("path") != FROZEN_PLAN_FILE:
            reasons.append(f"plan ref must point at {FROZEN_PLAN_FILE}")
        reasons += [f"plan: {e}" for e in check_plan(plan)]
    except (ValueError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as e:
        reasons.append(f"plan: {e}")
        plan = None
    try:
        auth, auth_meta = trust.load_trusted_json(repo, auth_ref, trusted_ref=trusted_ref)
    except (ValueError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as e:
        reasons.append(f"authorization: {e}")
    if plan is not None:
        reasons += [f"authorization: {e}" for e in check_authorization(auth, plan)]
        if auth and auth_meta and not reasons:
            reasons += [f"approval: {e}" for e in verify_approval(repo, auth, auth_meta, trusted_ref=trusted_ref,
                                                                    gh=gh)]
        if auth and (auth.get("plan_freeze") or {}).get("commit") != plan_ref.get("commit"):
            reasons.append("authorization names another plan freeze commit")
        if reg.check_pins() or reg.check_registry():
            reasons.append("teacher identity drift (registry pins)")
        try:
            fresh = (live or (lambda: live_plan(repo)))()
            if fresh["plan_hash"] != plan["plan_hash"]:
                diff = sorted(k for k in plan_body(plan) if plan_body(plan)[k] != fresh.get(k))
                reasons.append(f"live rebuild differs from the frozen plan (code, pins, data or audit drift): {diff}")
        except (ValueError, OSError, KeyError) as e:
            reasons.append(f"live plan rebuild failed: {e}")
    try:
        head = C._git(repo, "rev-parse", "HEAD")
    except subprocess.SubprocessError as e:
        head = ""
        reasons.append(f"git: {e}")
    dirty = C.tree_problems(repo)
    if dirty:
        reasons.append(f"working tree not clean: {dirty[:20]}{' ...' if len(dirty) > 20 else ''}")
    if auth and head != auth.get("execution_commit"):
        reasons.append("HEAD is not the authorized execution commit")
    if head and not trust.is_ancestor(repo, head, trusted_ref):
        reasons.append(f"HEAD is not in the history of {trusted_ref} (unreviewed code)")
    if plan_ref.get("commit") and head and not trust.is_ancestor(repo, plan_ref["commit"], head):
        reasons.append("HEAD does not descend from the plan freeze commit")
    try:
        check_attempt_marker(out_dir, ATTEMPT)
    except PilotBlocked as e:
        reasons.append(str(e))
    if reasons:
        raise PilotBlocked("; ".join(reasons))
    return {"plan": plan, "auth": auth, "auth_meta": auth_meta, "head": head}


def live_plan(repo: Path) -> dict[str, Any]:
    """Rebuild the plan from the current code, data and audits (the exact `plan` command computation)."""
    train = json.loads((repo / "data/splits/p1-02-v1/train.json").read_text(encoding="utf-8"))
    held, maps, src = heldout_context(repo)
    sc = select(train["scenarios"])
    fams = train_families(train, [s["family_id"] for s in sc])
    geo = {**geometry_audit(sc, fams, maps), "sources": src}
    return build_plan(train, held, code=code_identities(), geometry=geo, cost_basis=p1_09_cost_basis(repo))


# ----- execution ----------------------------------------------------------------------------------------------
def _jobs(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    from .teacher import registry as reg
    out = []
    for t in CANDIDATES:
        p = plan["teacher_pins"]["teachers"][t]
        ident = reg.identity(t, p["version"])
        out.append({"name": t, "identity": ident.key(), "seed": p["seed"], "code_fingerprint": p["code_fingerprint"]})
    return out


def _group_summary(sid: str, rep: Mapping[str, Any] | None, scenario: Any, run: Mapping[str, Any] | None,
                   status: str, error: str | None = None) -> dict[str, Any]:
    from . import multiteacher as mt
    row = {"scenario_id": sid, "status": status, "verified": False, "group_complete": False, "leakage_ok": False,
           "failed_teachers": list(CANDIDATES), "content_hash": None, "error": error,
           "wall_s": run["supervision"]["wall_s"] if run else None,
           "t2_cpu_s": run["supervision"]["process_cpu_s"]["T2"] if run else None,
           "budget_limited_decisions": run["supervision"]["budget_limited_decisions"] if run else None}
    if rep is None:
        return row
    try:
        mt.verify_fixture(rep, scenario)
        row["verified"] = True
    except mt.MultiTeacherError as e:
        row["error"] = f"verify: {e}"
    try:
        for r in rep["policies"].values():
            for d in r["decisions"]:
                mt.check_public_observation(d["observation"])
        row["leakage_ok"] = True
    except mt.MultiTeacherError as e:
        row["error"] = f"leakage: {e}"
    row.update(group_complete=rep["group_complete"], failed_teachers=rep["failed_teachers"],
               content_hash=rep["content_hash"])
    return row


def _run_one(plan, sid, scenario, jobs, pins, prov, deadline, log, target, attempt, kind):
    from . import pilot_supervisor as ps
    run = ps.run_group(scenario, jobs, decision_cpu_s=plan["limits"]["decision_cpu_s"],
                       group_wall_s=plan["limits"]["group_wall_s"], deadline=deadline, log=log, target=target)
    extra = {"pilot": {"plan_hash": plan["plan_hash"], "scenario_id": sid, "attempt": attempt}}
    try:
        rep = ps.build_group(scenario, run, pins["identities"], pins["seeds"], prov, extra=extra, run_kind=kind)
        return run, rep, None
    except ps.GroupIntegrityError as e:
        return run, None, f"integrity_error: {e}"


def execute(plan: Mapping[str, Any], scenarios: Sequence[tuple[Mapping[str, Any], Any]], out_dir: Path,
            marker_doc: Mapping[str, Any], *, target: Callable[..., None] | None = None,
            jobs: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """After every gate: marker, eight supervised groups, two regenerations, acceptance, evidence.

    `target` (tests only) replaces the worker entry point; `jobs` replaces the plan-derived worker jobs.
    """
    from . import multiteacher as mt
    from .teacher import registry as reg
    from .teacher.confirmation import AttemptLog
    paths = attempt_paths(out_dir, plan["attempt"])
    check_attempt_marker(out_dir, plan["attempt"])
    marker_body = {"attempt_version": ATTEMPT_VERSION, "attempt": plan["attempt"], "plan_hash": plan["plan_hash"],
                   **dict(marker_doc), "limits": plan["limits"],
                   "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "platform": {"system": platform.system(), "machine": platform.machine(),
                                "python": platform.python_version(), "cpu_count": os.cpu_count()}}
    marker = {**marker_body, "attempt_hash": digest(marker_body)}
    write_once(paths["marker"], marker)                     # BEFORE the first teacher call
    paths["dir"].mkdir(parents=True, exist_ok=False)
    log = AttemptLog(paths["log"])
    log("attempt_start", attempt_hash=marker["attempt_hash"])
    jobs = list(jobs if jobs is not None else _jobs(plan))
    pols = {j["name"]: reg.teacher_class(j["name"], j["identity"]["version"]) for j in jobs}
    insts = {n: (c() if j["seed"] is None else c(seed=j["seed"])) for (n, c), j in zip(pols.items(), jobs)}
    pins = {"identities": {n: p.identity.to_dict() for n, p in insts.items()}, "seeds": {j["name"]: j["seed"]
                                                                                         for j in jobs}}
    deadline = time.monotonic() + plan["limits"]["pilot_wall_s"]
    by_id = {e["scenario_id"]: (e, s) for e, s in scenarios}
    rows, reports, interrupted, halted = [], {}, False, False
    t0 = time.monotonic()
    for spec in plan["scenarios"]:
        sid = spec["scenario_id"]
        e, scn = by_id[sid]
        if interrupted or halted or time.monotonic() > deadline:
            rows.append(_group_summary(sid, None, scn, None, "not_started",
                                       "interrupted" if interrupted else "supervisor_error" if halted
                                       else "pilot_wall_cap"))
            log("group_not_started", scenario_id=sid)
            continue
        try:
            prov = mt.provenance(scn, insts, {"kind": "train", "manifest_hash": plan["source"]["manifest_hash"],
                                              "scenario_id": sid, "lineage_id": e["lineage_id"]})
            run, rep, err = _run_one(plan, sid, scn, jobs, pins, prov, deadline, log, target, plan["attempt"],
                                     "primary")
            interrupted = interrupted or run["interrupted"]
            doc = rep if rep is not None else {"scenario_id": sid, "integrity_error": err, "group_complete": False,
                                               "training_eligible": False,
                                               "runtime": {"supervision": run["supervision"]},
                                               "failures": run["failures"]}
            write_once(paths["groups"] / f"{sid}.json", doc)
            reports[sid] = rep
            rows.append(_group_summary(sid, rep, scn, run, "ran", err))
        except Exception as exc:          # noqa: BLE001 - supervisor fault: record it, start no new group
            msg = f"{type(exc).__name__}: {exc}"[:500]
            log("supervisor_error", scenario_id=sid, error=msg)
            rows.append(_group_summary(sid, None, scn, None, "supervisor_error", msg))
            halted = True
    regen = {}
    for sid in plan["regeneration"]["scenario_ids"]:
        rep = reports.get(sid)
        if interrupted or halted or time.monotonic() > deadline or rep is None or not rep["group_complete"]:
            regen[sid] = {"regenerated_identical": False, "reason": "not run (group incomplete, deadline or "
                                                                    "interrupt)"}
            continue
        e, scn = by_id[sid]
        run, again, err = _run_one(plan, sid, scn, jobs, pins, rep["provenance"], deadline, log, target,
                                   plan["attempt"], "regeneration")
        interrupted = interrupted or run["interrupted"]
        same = again is not None and again["content_hash"] == rep["content_hash"]
        write_once(paths["regen"] / f"{sid}.json", again if again is not None else {"integrity_error": err})
        regen[sid] = {"regenerated_identical": same, "content_hash": again["content_hash"] if again else None}
    for r in rows:
        r["regenerated_identical"] = regen.get(r["scenario_id"], {}).get("regenerated_identical")
    acc = pilot_acceptance(plan, rows, pilot_wall_s=round(time.monotonic() - t0, 3))
    body = {"evidence_version": EVIDENCE_VERSION, "plan_hash": plan["plan_hash"], "attempt": marker,
            "groups": rows, "regeneration": regen, "acceptance": acc, "interrupted": interrupted,
            "training_eligible": False,
            "note": "Pilot feasibility evidence only. Eight cases support no population-level teacher claim."}
    write_once(paths["evidence"], {**body, "evidence_hash": digest(body)})
    log("evidence_written", accepted=acc["pass"], interrupted=interrupted)
    if interrupted:
        raise KeyboardInterrupt
    return {"evidence": str(paths["evidence"]), "accepted": acc["pass"], "problems": acc["problems"]}


def pilot_acceptance(plan: Mapping[str, Any], groups: Sequence[Mapping[str, Any]], *,
                     pilot_wall_s: float | None = None) -> dict[str, Any]:
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
        st = strata.setdefault(label.get(g["scenario_id"], "unplanned"), [0, 0])
        st[0] += 0 if ok else 1
        st[1] += 1
        if not ok:
            problems.append(f"{g['scenario_id']}: not a verified complete group ({g.get('failed_teachers')})")
        if g.get("leakage_ok") is not True:
            problems.append(f"{g['scenario_id']}: leakage gate not passed")
        if not isinstance(g.get("wall_s"), (int, float)) or g["wall_s"] > LIMITS["group_wall_s"]:
            problems.append(f"{g['scenario_id']}: group wall cap exceeded or not measured")
    rate = {k: v[0] / v[1] for k, v in strata.items()}
    if any(r > 0 for r in rate.values()):
        problems.append(f"incomplete-group rate per stratum {rate}")
    regen = {g["scenario_id"]: g.get("regenerated_identical") for g in groups}
    for sid in plan["regeneration"]["scenario_ids"]:
        if regen.get(sid) is not True:
            problems.append(f"{sid}: preregistered regeneration not identical or not run")
    t2 = [g.get("t2_cpu_s") for g in groups]
    lim = plan["cost_estimate"]["t2_cpu_s_total_limit"]
    if any(not isinstance(x, (int, float)) for x in t2) or sum(t2) > lim:
        problems.append(f"T2 CPU total above {lim} s or not measured: re-plan before full TRAIN")
    walls = [g.get("wall_s") or 0 for g in groups]
    total_wall = pilot_wall_s if pilot_wall_s is not None else sum(walls)
    if total_wall > LIMITS["pilot_wall_s"]:
        problems.append("pilot wall cap exceeded")
    return {"pass": not problems, "problems": problems, "incomplete_rate_per_stratum": rate,
            "t2_cpu_total_s": round(sum(x for x in t2 if isinstance(x, (int, float))), 1),
            "budget_limited_groups": sorted(g["scenario_id"] for g in groups
                                            if any((g.get("budget_limited_decisions") or {}).values()))}


def run_pilot(repo: Path, plan_ref: Mapping[str, Any], auth_ref: Mapping[str, Any], *,
              trusted_ref: str = "origin/main") -> dict[str, Any]:
    """The one authorized attempt. Refuses (PilotBlocked) unless every gate passes."""
    pre = run_preconditions(repo, plan_ref, auth_ref, trusted_ref=trusted_ref)
    plan = pre["plan"]
    train = json.loads((repo / "data/splits/p1-02-v1/train.json").read_text(encoding="utf-8"))
    scenarios = pilot_scenarios(train, plan)
    return execute(plan, scenarios, repo / OUT_DIR, {
        "plan_freeze_commit": plan_ref["commit"], "authorization": {**pre["auth_meta"]},
        "execution_commit": pre["head"], "trusted_ref": trusted_ref,
        "rerun_policy": "one attempt only; attempt 2 needs a new authorization and plan version"})


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ember_v1.train_pilot")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("plan", help="write the PROPOSED pilot plan (runs no teacher)")
    pl.add_argument("--out", required=True)
    tp = sub.add_parser("template", help="write the PENDING authorization template for a plan file")
    tp.add_argument("--plan", required=True)
    tp.add_argument("--out", required=True)
    vp = sub.add_parser("verify-plan", help="rebuild the plan and compare with a plan file (runs no teacher)")
    vp.add_argument("--plan", required=True)
    rn = sub.add_parser("run", help="the authorized attempt; refused unless every gate passes")
    rn.add_argument("--plan-ref", required=True, help="<freeze commit>:" + FROZEN_PLAN_FILE)
    rn.add_argument("--auth-ref", required=True, help="<commit on main>:" + AUTH_FILE)
    rn.add_argument("--trusted-ref", default="origin/main")
    a = ap.parse_args(argv)
    repo = Path(__file__).resolve().parents[2]
    if a.cmd == "run":
        out = run_pilot(repo, parse_ref(a.plan_ref), parse_ref(a.auth_ref), trusted_ref=a.trusted_ref)
        sys.stdout.write(json.dumps(out) + "\n")
        return 0 if out["accepted"] else 1
    if a.cmd == "template":
        plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))
        if check_plan(plan, frozen=False):
            raise SystemExit(f"plan problems: {check_plan(plan, frozen=False)}")
        _write_json(Path(a.out), authorization_template(plan))
        return 0
    plan = live_plan(repo)
    if a.cmd == "verify-plan":
        old = json.loads(Path(a.plan).read_text(encoding="utf-8"))
        diff = sorted(k for k in plan_body(plan) if plan_body(old).get(k) != plan_body(plan)[k])
        sys.stdout.write(json.dumps({"same": old.get("plan_hash") == plan["plan_hash"], "file": old.get("plan_hash"),
                                     "rebuilt": plan["plan_hash"], "differs": diff}) + "\n")
        return 0 if old.get("plan_hash") == plan["plan_hash"] and not check_plan(old, frozen=False) else 1
    _write_json(Path(a.out), plan)
    sys.stdout.write(json.dumps({"plan_hash": plan["plan_hash"], "scenarios": [s["scenario_id"] for s in
                                 plan["scenarios"]], "t2_cpu_s_central": plan["cost_estimate"]["t2_cpu_s_central"],
                                 "geometry_pass": plan["geometry_audit"]["pass"]}) + "\n")
    return 0


def _write_json(path: Path, doc: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, sort_keys=True, default=sorted) + "\n", encoding="utf-8",
                    newline="\n")


if __name__ == "__main__":
    raise SystemExit(main())
