"""P1-12 TRAIN pilot: plan, authorization, approval binding, gates, killable supervision.

No test runs a teacher on a TRAIN scenario. Supervised execution runs real teachers on tiny
hand-built fixtures, plus fake workers that crash, hang, go silent, forge or emit invalid actions.
"""
from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest

import pilot_fakes as fk
from ember_v1 import multiteacher as mt
from ember_v1 import pilot_supervisor as ps
from ember_v1 import train_pilot as tp
from ember_v1.s1_pipeline import HELD_OUT_KEYS
from ember_v1.sim.canonical import digest
from ember_v1.teacher import registry as reg
from ember_v1.teacher.registry import CANDIDATES
from sim_helpers import scenario

REPO = Path(__file__).resolve().parents[1]
TRAIN = json.loads((REPO / "data/splits/p1-02-v1/train.json").read_text(encoding="utf-8"))
EMPTY = {"other": {k: set() for k in HELD_OUT_KEYS}}
PINS = {"teachers": {t: {"version": "x", "config_hash": "c", "code_fingerprint": f"fp-{t}", "seed": None}
                     for t in CANDIDATES}, "pin_problems": []}
GEO_OK = {"pass": True, "hits": [], "within_pilot_hits": []}
EXPECTED_IDS = ["g-a2993d188ad45093", "g-6dfef26f38a144d0", "g-b9f6d81f3b5bfce7", "g-35cca9f1d42320be",
                "g-0821b9cfcfbdc9a9", "g-0ee55267388889cc", "g-1a8b6230b6107b0e", "g-5082bdcde677aea4"]
H = "a" * 40


@pytest.fixture(scope="module")
def basis():
    return tp.p1_09_cost_basis(REPO)


@pytest.fixture(scope="module")
def plan(basis):
    return tp.build_plan(TRAIN, EMPTY, pins=PINS, geometry=GEO_OK, cost_basis=basis)


def frozen(plan):
    return {**plan, "status": tp.FROZEN}


# ----- plan and design ----------------------------------------------------------------------------------------
def test_plan_keeps_the_eight_preregistered_scenarios_and_design(plan):
    assert [s["scenario_id"] for s in plan["scenarios"]] == EXPECTED_IDS
    d = plan["design"]
    assert d["labels"] == {"difficult": 3, "feasible": 4, "unreachable": 1}
    assert d["fire_models"] == {"deterministic": 5, "stochastic": 3}
    assert d["people"] == [6, 8, 24, 33, 46, 114, 115, 120] and d["families"] == 8
    assert d["groups_per_scenario"] == 1 and d["fire_seeds_per_scenario"] == 1
    assert plan["status"] == tp.PLAN_STATUS and plan["attempt"] == 1
    assert plan["regeneration"]["scenario_ids"] == ["g-0821b9cfcfbdc9a9", "g-b9f6d81f3b5bfce7"]
    assert tp.build_plan(TRAIN, EMPTY, pins=PINS, geometry=GEO_OK, cost_basis=tp.p1_09_cost_basis(REPO)) == plan


def test_selection_reads_only_manifest_fields_and_no_teacher_outcome():
    """Scrambling every field outside SELECTION_INPUTS (plus pass-through ids) leaves the choice unchanged."""
    keep = {"scenario_hash", "label", "family_id", "scenario_id", "lineage_id", "core_hash"}
    stripped = []
    for e in TRAIN["scenarios"]:
        x = {k: v for k, v in e.items() if k in keep}
        x["diagnostics"] = {"fire_model": e["diagnostics"]["fire_model"], "people": e["diagnostics"]["people"],
                            "max_steps": e["diagnostics"]["max_steps"]}
        stripped.append(x)
    assert [s["scenario_id"] for s in tp.select(stripped)] == EXPECTED_IDS


def test_plan_fails_closed_on_heldout_overlap_bad_pins_and_geometry(basis):
    sid = TRAIN["scenarios"][0]
    bad = {"dev": {k: set() for k in HELD_OUT_KEYS}}
    bad["dev"]["family"] = {sid["family_id"]}
    with pytest.raises(ValueError, match="overlaps"):
        tp.build_plan(TRAIN, bad, pins=PINS, geometry=GEO_OK, cost_basis=basis)
    with pytest.raises(ValueError, match="pins"):
        tp.build_plan(TRAIN, EMPTY, pins={**PINS, "pin_problems": ["T2 drift"]}, geometry=GEO_OK, cost_basis=basis)
    with pytest.raises(ValueError, match="geometry"):
        tp.build_plan(TRAIN, EMPTY, pins=PINS, geometry={"pass": False, "hits": [1]}, cost_basis=basis)


def test_stratified_t2_cost_replaces_the_global_mean(plan):
    c = plan["cost_estimate"]
    big = [r for r in c["per_scenario"] if r["people"] >= 100]
    assert len(big) == 3 and all(r["bin_scenarios"] == 10 for r in big)
    assert c["t2_cpu_s_central"] == pytest.approx(sum(r["t2_cpu_s_mean"] for r in c["per_scenario"]), abs=0.2)
    assert c["t2_cpu_s_upper"] > c["superseded_v0_1_estimate"]["t2_cpu_s"] > c["t2_cpu_s_central"] * 0.9
    assert c["t2_cpu_s_total_limit"] == c["t2_cpu_s_upper"] < tp.LIMITS["pilot_wall_s"]
    assert c["t2_worst_case_s"] == 20 * plan["design"]["max_steps_total"] > tp.LIMITS["pilot_wall_s"]
    assert tp.LIMITS["decision_cpu_s"] == {"T0": 5, "T1": 5, "T2": 30, "T3": 5, "T4": 5}
    assert tp.LIMITS["group_wall_s"] == 7200 and tp.LIMITS["pilot_wall_s"] == 14400 and tp.LIMITS["retries"] == 0


def test_plan_hash_excludes_status_and_detects_tampering(plan):
    f = frozen(plan)
    assert f["plan_hash"] == plan["plan_hash"] and tp.check_plan(f) == []
    assert "not FROZEN" in " ".join(tp.check_plan(plan))
    for mutate in (lambda p: p["scenarios"][0].update(scenario_hash="sha256:" + "0" * 64),
                   lambda p: p["limits"]["decision_cpu_s"].update(T2=300),
                   lambda p: p["teacher_pins"]["teachers"]["T3"].update(code_fingerprint="drift")):
        t = copy.deepcopy(f)
        mutate(t)
        assert "tampered" in " ".join(tp.check_plan(t))


def test_committed_proposed_plan_matches_a_fresh_rebuild():
    """Never preserve a stale hash: the committed plan equals the live rebuild of code, data and audits."""
    committed = json.loads((REPO / tp.PLAN_FILE).read_text(encoding="utf-8"))
    fresh = json.loads(json.dumps(tp.live_plan(REPO), sort_keys=True, default=sorted))
    assert committed["status"] == tp.PLAN_STATUS and tp.check_plan(committed, frozen=False) == []
    assert committed == fresh, sorted(k for k in fresh if committed.get(k) != fresh[k])
    assert committed["geometry_audit"]["pass"] and committed["disjointness_audit"]["pass"]


def test_module_fingerprint_is_case_exact_on_every_platform():
    """CI (Linux) found Windows-only phantom modules: `from .sim import Scenario` resolved `ember_v1.sim.Scenario`
    to `sim/scenario.py` on a case-insensitive file system, so plan identities differed by platform."""
    from ember_v1 import s1_pipeline as sp
    assert sp._exact_module_path("ember_v1.sim.Scenario") is None
    assert sp._exact_module_path("ember_v1.sim.Simulator") is None
    assert sp._exact_module_path("ember_v1.sim.scenario") is not None
    assert sp._exact_module_path("ember_v1.sim") is not None


def test_freeze_candidate_doc_cites_the_committed_plan_exactly():
    committed = json.loads((REPO / tp.PLAN_FILE).read_text(encoding="utf-8"))
    doc = (REPO / "docs/P1_12_PILOT_FREEZE_CANDIDATE.md").read_text(encoding="utf-8")
    assert f"**Proposed plan hash: `{committed['plan_hash']}`**" in doc and "NOT FROZEN" in doc
    for s in committed["scenarios"]:
        assert f"| {s['scenario_id']} | {s['scenario_hash']} | {s['family_id']} |" in doc
    for t, v in committed["teacher_pins"]["teachers"].items():
        assert f"| {t} | {v['version']} | {v['config_hash']} | {v['code_fingerprint']} |" in doc
    assert not (REPO / tp.FROZEN_PLAN_FILE).exists() and not (REPO / tp.AUTH_FILE).exists()


# ----- authorization content ----------------------------------------------------------------------------------
def approved(plan, **kw):
    a = tp.authorization_template(frozen(plan))
    a.update(status=tp.AUTH_APPROVED, execution_commit=H, plan_freeze={"commit": H, "path": tp.FROZEN_PLAN_FILE},
             reviewer={"github_login": "Sathvikar01"},
             approval={**a["approval"], "pull_request": 77})
    a.update(kw)
    return a


def test_template_is_pending_and_never_acceptable(plan):
    t = tp.authorization_template(frozen(plan))
    assert t["status"] == "PENDING" and t["kind"] == tp.AUTH_KIND and t["attempt"] == 1
    assert len(t["scenarios"]) == 8 and set(t["teacher_fingerprints"]) == set(CANDIDATES)
    errs = " ".join(tp.check_authorization(t, frozen(plan)))
    assert "PENDING" in errs and "allowlisted" in errs and "pull request" in errs
    committed = REPO / tp.AUTH_TEMPLATE_FILE
    if committed.exists():
        c = json.loads(committed.read_text(encoding="utf-8"))
        assert c["status"] == "PENDING" and c["plan_hash"] == json.loads(
            (REPO / tp.PLAN_FILE).read_text(encoding="utf-8"))["plan_hash"]


@pytest.mark.parametrize("change,needle", [
    ({"plan_hash": "sha256:other"}, "another plan hash"),
    ({"attempt": 2}, "attempt 1"),
    ({"attempt": "1"}, "attempt 1"),
    ({"status": "approved by the owner"}, "not APPROVED"),
    ({"reviewer": {"github_login": "The Owner"}}, "allowlisted"),
    ({"reviewer": {"github_login": "Sankalp H S"}}, "allowlisted"),
    ({"approval": {"kind": "github_pr_review", "repository": "anantamlab/ember_v1", "pull_request": None}},
     "pull request"),
    ({"kind": "s1-train-authorization"}, "P1-09"),
    ({"note": "see p1-09 receipt"}, "P1-09"),
    ({"execution_commit": "HEAD"}, "execution_commit"),
    ({"limits": {**tp.LIMITS, "retries": 1}}, "limits"),
])
def test_authorization_content_negative_cases(plan, change, needle):
    assert tp.check_authorization(approved(plan), frozen(plan)) == []
    errs = tp.check_authorization(approved(plan, **change), frozen(plan))
    assert any(needle in e for e in errs), errs


def test_authorization_scenarios_and_fingerprints_must_match_exactly(plan):
    a = approved(plan)
    a["scenarios"] = a["scenarios"][:7]
    assert any("8 scenarios" in e for e in tp.check_authorization(a, frozen(plan)))
    a = approved(plan)
    a["scenarios"][3] = {**a["scenarios"][3], "scenario_hash": "sha256:" + "1" * 64}
    assert any("8 scenarios" in e for e in tp.check_authorization(a, frozen(plan)))
    a = approved(plan)
    a["teacher_fingerprints"]["T2"]["code_fingerprint"] = "drift"
    assert any("fingerprints" in e for e in tp.check_authorization(a, frozen(plan)))
    assert any("FROZEN" in e for e in tp.check_authorization(approved(plan), plan))
    assert tp.check_authorization(None, frozen(plan)) == ["no multi-teacher TRAIN pilot authorization"]


# ----- approval binding on a real git history + fake GitHub ---------------------------------------------------
def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          capture_output=True, text=True, check=True).stdout.strip()


def _commit(repo, rel, doc, msg):
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture()
def history(tmp_path, plan):
    """main: freeze commit (FROZEN plan) -> PR branch adds the authorization -> merge into main."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / ".gitattributes").write_text("* -text\n", encoding="utf-8")
    freeze = _commit(repo, tp.FROZEN_PLAN_FILE, frozen(plan), "freeze")
    _git(repo, "checkout", "-q", "-b", "auth")
    auth = approved(plan, plan_freeze={"commit": freeze, "path": tp.FROZEN_PLAN_FILE}, execution_commit=freeze)
    _commit(repo, tp.AUTH_FILE, {**auth, "statement": "draft"}, "auth draft")
    head = _commit(repo, tp.AUTH_FILE, auth, "auth final")
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--no-ff", "auth", "-m", "Merge PR #77")
    merge = _git(repo, "rev-parse", "HEAD")
    auth_ref = {"kind": "git", "commit": merge, "path": tp.AUTH_FILE}
    auth_doc, meta = tp.trust.load_trusted_json(repo, auth_ref, trusted_ref="main")
    gh = {"repos/anantamlab/ember_v1/pulls/77": {"merged": True, "base": {"ref": "main"},
                                                   "merge_commit_sha": merge, "user": {"login": "sankalphs"}},
          "repos/anantamlab/ember_v1/pulls/77/reviews?per_page=100": [
              {"id": 1, "state": "APPROVED", "commit_id": head, "submitted_at": "2026-11-10T10:00:00Z",
               "user": {"login": "Sathvikar01", "type": "User"}}]}
    return {"repo": repo, "freeze": freeze, "head": head, "merge": merge, "auth": auth_doc, "meta": meta,
            "auth_ref": auth_ref, "gh": gh, "plan_ref": {"kind": "git", "commit": freeze,
                                                          "path": tp.FROZEN_PLAN_FILE}}


def _fake(gh):
    def call(path):
        if path not in gh:
            raise ValueError(f"404 {path}")
        return copy.deepcopy(gh[path])
    return call


def test_verified_github_review_is_accepted(history):
    h = history
    assert tp.verify_approval(h["repo"], h["auth"], h["meta"], trusted_ref="main", gh=_fake(h["gh"])) == []


@pytest.mark.parametrize("mutate,needle", [
    (lambda g: g["repos/anantamlab/ember_v1/pulls/77"].update(merged=False), "not merged"),
    (lambda g: g["repos/anantamlab/ember_v1/pulls/77"].update(user={"login": "Sathvikar01"}), "not the PR author"),
    (lambda g: g["repos/anantamlab/ember_v1/pulls/77/reviews?per_page=100"].append(
        {"id": 2, "state": "CHANGES_REQUESTED", "commit_id": "c" * 40, "submitted_at": "2026-11-10T11:00:00Z",
         "user": {"login": "Sathvikar01", "type": "User"}}), "no current APPROVED"),
    (lambda g: g["repos/anantamlab/ember_v1/pulls/77/reviews?per_page=100"][0].update(
        user={"login": "someone-else", "type": "User"}), "no current APPROVED"),
    (lambda g: g["repos/anantamlab/ember_v1/pulls/77/reviews?per_page=100"][0].update(
        user={"login": "Sathvikar01", "type": "Bot"}), "no current APPROVED"),
    (lambda g: g["repos/anantamlab/ember_v1/pulls/77/reviews?per_page=100"].clear(), "no current APPROVED"),
    (lambda g: g.pop("repos/anantamlab/ember_v1/pulls/77"), "fail closed"),
])
def test_unverifiable_or_wrong_github_approval_is_refused(history, mutate, needle):
    h = history
    gh = copy.deepcopy(h["gh"])
    mutate(gh)
    errs = tp.verify_approval(h["repo"], h["auth"], h["meta"], trusted_ref="main", gh=_fake(gh))
    assert any(needle in e for e in errs), errs


def test_approval_of_other_bytes_is_refused(history):
    h = history
    gh = copy.deepcopy(h["gh"])
    draft = _git(h["repo"], "rev-parse", f"{h['head']}~1")             # approved the draft, not the final record
    gh["repos/anantamlab/ember_v1/pulls/77/reviews?per_page=100"][0]["commit_id"] = draft
    errs = tp.verify_approval(h["repo"], h["auth"], h["meta"], trusted_ref="main", gh=_fake(gh))
    assert any("different authorization record" in e for e in errs), errs


# ----- full preconditions -------------------------------------------------------------------------------------
def _pre(h, plan, **kw):
    args = {"trusted_ref": "main", "out_dir": h["repo"] / tp.OUT_DIR, "gh": _fake(h["gh"]),
            "live": lambda: frozen(plan)}
    args.update(kw)
    return tp.run_preconditions(h["repo"], h["plan_ref"], h["auth_ref"], **args)


def test_all_gates_pass_only_in_the_exact_authorized_state(history, plan):
    h = history
    _git(h["repo"], "checkout", "-q", h["freeze"])                     # the authorized execution commit
    out = _pre(h, plan)
    assert out["head"] == h["freeze"] and out["plan"]["plan_hash"] == plan["plan_hash"]


def test_gate_refusals(history, plan):
    h = history
    with pytest.raises(tp.PilotBlocked, match="execution commit"):     # HEAD is the merge, not the freeze
        _pre(h, plan)
    _git(h["repo"], "checkout", "-q", h["freeze"])
    with pytest.raises(tp.PilotBlocked, match="live rebuild differs"):  # code / pin / data drift
        _pre(h, plan, live=lambda: {**frozen(plan), "plan_hash": "sha256:drift"})
    with pytest.raises(tp.PilotBlocked, match="approval"):              # missing GitHub approval
        _pre(h, plan, gh=_fake({}))
    with pytest.raises(tp.PilotBlocked, match="trusted ref"):           # plan not on the trusted history
        _pre(h, plan, trusted_ref="refs/heads/none")
    (h["repo"] / "stray.txt").write_text("x", encoding="utf-8")
    with pytest.raises(tp.PilotBlocked, match="not clean"):
        _pre(h, plan)
    (h["repo"] / "stray.txt").unlink()
    marker = h["repo"] / tp.OUT_DIR / "attempt-1.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("{}", encoding="utf-8")
    with pytest.raises(tp.PilotBlocked, match="attempt evidence exists"):
        _pre(h, plan)


def test_tampered_or_unfrozen_plan_in_history_is_refused(history, plan):
    h = history
    t = frozen(plan)
    t["scenarios"][0]["scenario_hash"] = "sha256:" + "f" * 64
    bad = _commit(h["repo"], tp.FROZEN_PLAN_FILE, t, "tamper")
    with pytest.raises(tp.PilotBlocked, match="tampered"):
        tp.run_preconditions(
            h["repo"], {**h["plan_ref"], "commit": bad}, h["auth_ref"], trusted_ref="main",
            out_dir=h["repo"] / tp.OUT_DIR, gh=_fake(h["gh"]), live=lambda: frozen(plan))
    unf = _commit(h["repo"], tp.FROZEN_PLAN_FILE, plan, "unfreeze")
    with pytest.raises(tp.PilotBlocked, match="not FROZEN"):
        tp.run_preconditions(h["repo"], {**h["plan_ref"], "commit": unf}, h["auth_ref"], trusted_ref="main",
                             out_dir=h["repo"] / tp.OUT_DIR, gh=_fake(h["gh"]), live=lambda: frozen(plan))


def test_cli_run_is_refused_without_frozen_plan_or_authorization():
    with pytest.raises(tp.PilotBlocked):
        tp.main(["run", "--plan-ref", f"{'0' * 40}:{tp.FROZEN_PLAN_FILE}", "--auth-ref", f"{'0' * 40}:{tp.AUTH_FILE}"])


# ----- supervised execution on tiny fixtures (real teachers + fakes) ------------------------------------------
def tiny(i):
    return scenario(["E...#...", "....#...", "........", "..F.....", "......E."],
                    {"P1": (2, 1), "P2": (2, 2), "P3": (4, 5)}, fire="stochastic", p=0.5, interval=1, seed=5 + i,
                    max_steps=6, enter_fire="blocked", sid=f"p1-12-pilot-fixture-{i}")


@pytest.fixture(scope="module")
def mini_plan():
    pins = tp.teacher_pins()
    sc = [{"scenario_id": f"fx-{i}", "scenario_hash": tiny(i).content_hash(), "label": "feasible",
           "lineage_id": f"lin-{i}", "family_id": f"fam-{i}"} for i in range(2)]
    return {"plan_hash": "sha256:mini", "attempt": 1, "scenarios": sc, "teacher_pins": pins,
            "limits": {**tp.LIMITS, "group_wall_s": 120, "pilot_wall_s": 600},
            "regeneration": {"scenario_ids": ["fx-0"]}, "cost_estimate": {"t2_cpu_s_total_limit": 1e9},
            "source": {"manifest_hash": "fixture"}}


def mini_scenarios():
    return [({"scenario_id": f"fx-{i}", "lineage_id": f"lin-{i}"}, tiny(i)) for i in range(2)]


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def test_supervised_attempt_completes_verifies_regenerates_and_is_write_once(tmp_path, mini_plan):
    out = tp.execute(mini_plan, mini_scenarios(), tmp_path, {"execution_commit": H})
    assert out["accepted"], out["problems"]
    paths = tp.attempt_paths(tmp_path)
    marker, ev = load(paths["marker"]), load(paths["evidence"])
    assert marker["plan_hash"] == "sha256:mini" and marker["attempt"] == 1
    for sid, scn in (("fx-0", tiny(0)), ("fx-1", tiny(1))):
        rep = load(paths["groups"] / f"{sid}.json")
        assert mt.verify_fixture(rep, scn)["verified"] and rep["group_complete"] and rep["scope"] == mt.SCOPE_TRAIN_PILOT
        assert rep["training_eligible"] is False and list(rep["policies"]) == list(CANDIDATES)
        assert set(rep["runtime"]["supervision"]["cpu_measurement"].values()) <= {"os", "worker_reported"}
        # same content as the in-process fixture collector: the supervisor changes nothing but isolation
        classes = reg.retained_classes()
        ref = mt.collect_fixture(scn, {n: (lambda n=n: classes[n]()) for n in CANDIDATES})
        assert [r["final_transition_hash"] for r in ref["policies"].values()] == \
               [r["final_transition_hash"] for r in rep["policies"].values()]
    assert ev["regeneration"]["fx-0"]["regenerated_identical"] is True
    events = [json.loads(x)["event"] for x in paths["log"].read_text(encoding="utf-8").splitlines()]
    assert events[0] == "attempt_start" and events[-1] == "evidence_written"
    with pytest.raises(tp.PilotBlocked, match="attempt evidence exists"):
        tp.execute(mini_plan, mini_scenarios(), tmp_path, {"execution_commit": H})
    with pytest.raises(tp.PilotBlocked, match="duplicate output"):
        tp.write_once(paths["groups"] / "fx-0.json", {})


def _failed(tmp_path, mini_plan, target, plan_changes=None):
    plan = {**mini_plan, **(plan_changes or {})}
    out = tp.execute(plan, mini_scenarios(), tmp_path, {"execution_commit": H},
                     target=target)
    paths = tp.attempt_paths(tmp_path)
    return out, load(paths["groups"] / "fx-0.json"), load(paths["evidence"])


@pytest.mark.parametrize("target,teacher,kind,limits", [
    (fk.crash_t2, "T2", "worker_crashed", {}),
    (fk.hang_t3_in_decide, "T3", "decision_cpu_cap", {"decision_cpu_s": {**tp.LIMITS["decision_cpu_s"], "T3": 1}}),
    (fk.sleep_t1, "T1", "group_wall_cap", {"group_wall_s": 15}),
    (fk.silent_exit_t4, "T4", "missing_policy_response", {}),
    (fk.invalid_action_t0, "T0", "invalid_proposal", {}),
])
def test_teacher_failure_marks_the_group_incomplete_and_keeps_evidence(tmp_path, mini_plan, target, teacher, kind,
                                                                       limits):
    import time
    t0 = time.monotonic()
    out, rep, ev = _failed(tmp_path, mini_plan, target, {"limits": {**mini_plan["limits"], **limits}})
    assert time.monotonic() - t0 < 120                                   # killed long before the 60 s fakes end
    assert not out["accepted"] and rep["group_complete"] is False and rep["failed_teachers"] == [teacher]
    assert rep["policies"][teacher]["failure"]["kind"] == kind and rep["training_eligible"] is False
    assert mt.verify_fixture(rep, tiny(0))["verified"]                 # partial evidence still replays
    assert all(rep["policies"][n]["status"] == "complete" for n in CANDIDATES if n != teacher)
    assert ev["regeneration"]["fx-0"]["regenerated_identical"] is False  # never re-run an incomplete group
    assert any("not a verified complete group" in p for p in ev["acceptance"]["problems"])
    assert mt.outcome_table(rep)["comparable"] is False


def test_code_drift_in_a_worker_is_a_teacher_failure(tmp_path, mini_plan):
    jobs = tp._jobs(mini_plan)
    jobs[2] = {**jobs[2], "code_fingerprint": "sha256:not-the-plan"}
    tp.execute(mini_plan, mini_scenarios(), tmp_path, {"execution_commit": H}, jobs=jobs)
    rep = load(tp.attempt_paths(tmp_path)["groups"] / "fx-0.json")
    assert rep["failed_teachers"] == ["T2"] and rep["policies"]["T2"]["failure"]["kind"] == "code_drift"


def test_forged_worker_stream_is_an_integrity_error_not_a_result(tmp_path, mini_plan):
    out, doc, ev = _failed(tmp_path, mini_plan, fk.forged_observation_t1)
    assert "integrity_error" in doc and doc["group_complete"] is False and doc["training_eligible"] is False
    assert not out["accepted"] and ev["groups"][0]["status"] == "ran" and ev["groups"][0]["verified"] is False


def test_interrupted_attempt_keeps_marker_and_evidence_and_blocks_retry(tmp_path, mini_plan, monkeypatch):
    real = ps.run_group

    def interrupted(*a, **kw):
        r = real(*a, **kw)
        return {**r, "interrupted": True}
    monkeypatch.setattr(ps, "run_group", interrupted)
    with pytest.raises(KeyboardInterrupt):
        tp.execute(mini_plan, mini_scenarios(), tmp_path, {"execution_commit": H})
    ev = load(tp.attempt_paths(tmp_path)["evidence"])
    assert ev["interrupted"] and [g["status"] for g in ev["groups"]] == ["ran", "not_started"]
    assert ev["acceptance"]["pass"] is False
    monkeypatch.setattr(ps, "run_group", real)
    with pytest.raises(tp.PilotBlocked, match="attempt evidence exists"):
        tp.execute(mini_plan, mini_scenarios(), tmp_path, {"execution_commit": H})


def test_parent_fault_is_recorded_and_stops_new_groups(tmp_path, mini_plan, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("disk full")
    monkeypatch.setattr(mt, "provenance", boom)
    out = tp.execute(mini_plan, mini_scenarios(), tmp_path, {"execution_commit": H})
    ev = load(tp.attempt_paths(tmp_path)["evidence"])
    assert not out["accepted"] and [g["status"] for g in ev["groups"]] == ["supervisor_error", "not_started"]
    assert ev["interrupted"] is False and tp.attempt_paths(tmp_path)["marker"].exists()


def test_pilot_deadline_stops_new_groups(tmp_path, mini_plan):
    plan = {**mini_plan, "limits": {**mini_plan["limits"], "pilot_wall_s": -1}}
    out = tp.execute(plan, mini_scenarios(), tmp_path, {"execution_commit": H})
    ev = load(tp.attempt_paths(tmp_path)["evidence"])
    assert not out["accepted"] and [g["status"] for g in ev["groups"]] == ["not_started", "not_started"]


# ----- acceptance gate ----------------------------------------------------------------------------------------
def groups(plan):
    return [{"scenario_id": s["scenario_id"], "verified": True, "group_complete": True, "failed_teachers": [],
             "leakage_ok": True, "wall_s": 300.0, "t2_cpu_s": 150.0,
             "regenerated_identical": s["scenario_id"] in plan["regeneration"]["scenario_ids"]}
            for s in plan["scenarios"]]


def test_acceptance_gate(plan):
    assert tp.pilot_acceptance(plan, groups(plan))["pass"]
    g = groups(plan)
    g[1].update(group_complete=False, failed_teachers=["T2"])
    assert not tp.pilot_acceptance(plan, g)["pass"]
    g = groups(plan)
    g[-1] = {**g[-1], "scenario_id": "g-substitute"}
    assert any("substitution" in p for p in tp.pilot_acceptance(plan, g)["problems"])
    g = groups(plan)
    for x in g:
        x["regenerated_identical"] = False
    assert any("regeneration" in p for p in tp.pilot_acceptance(plan, g)["problems"])
    g = groups(plan)
    for x in g:
        x["t2_cpu_s"] = 600.0
    assert any("re-plan" in p for p in tp.pilot_acceptance(plan, g)["problems"])
