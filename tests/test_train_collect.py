"""P1-17 full-TRAIN collection: plan, authorization, evidence root, parallel sessions, resume, verification.

No test runs a teacher on a TRAIN scenario. Execution runs real teachers on tiny hand-built fixtures.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import pytest
# Slow tier (AGENTS.md): skipped unless --run-slow.
pytestmark = pytest.mark.slow

import pilot_fakes as fk
from ember_v1 import multiteacher as mt
from ember_v1 import pilot_supervisor as ps
from ember_v1 import train_collect as tc
from ember_v1 import train_pilot as tp
from ember_v1.sim.canonical import digest
from ember_v1.teacher.registry import CANDIDATES
from sim_helpers import scenario

REPO = Path(__file__).resolve().parents[1]
H = "b" * 40


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def live():
    return tc.live_plan(REPO)


# ----- plan ---------------------------------------------------------------------------------------------------
def test_plan_covers_every_train_scenario_once_in_lpt_order(live):
    train = load(REPO / tc.TRAIN_FILE)
    assert tc.check_plan(live, frozen=False) == []
    ids = [s["scenario_id"] for s in live["scenarios"]]
    assert sorted(ids) == sorted(e["scenario_id"] for e in train["scenarios"]) and len(ids) == 189
    d = live["design"]
    assert d["families"] == 64 and d["teacher_episodes"] == 945 and d["people_max"] == 120
    assert set(d["fire_models"]) == {"deterministic", "stochastic"}
    keys = [(-s["people"], -s["max_steps"], s["rank"]) for s in live["scenarios"]]
    assert keys == sorted(keys)
    assert live["selection_rule"]["teacher_outcomes_consulted"] is False
    assert live["disjointness_audit"]["pass"] and live["geometry_audit"]["pass"]


def test_regeneration_sample_is_two_lowest_rank_per_people_bin(live):
    ids = live["regeneration"]["scenario_ids"]
    assert len(ids) == 2 * len(tc.N_BINS) == len(set(ids))
    by = {s["scenario_id"]: s for s in live["scenarios"]}
    for b in tc.N_BINS:
        inb = sorted((s for s in live["scenarios"] if tc._bin(s["people"]) == b), key=lambda s: s["rank"])
        assert [s["scenario_id"] for s in inb[:2]] == [i for i in ids if tc._bin(by[i]["people"]) == b]


def test_pilot_overlap_is_fresh_collection_not_a_retry(live):
    pilot = load(REPO / tc.PILOT_SUMMARY_FILE)
    ov = live["pilot_overlap"]
    assert ov["decision"] == "fresh_collection" and "not attempt 2" in ov["not_a_retry"]
    assert ov["scenario_ids"] == sorted(g["scenario_id"] for g in pilot["groups"])
    assert set(ov["scenario_ids"]) <= {s["scenario_id"] for s in live["scenarios"]}


def test_storage_root_is_outside_the_checkout(live):
    root = Path(live["storage"]["root"])
    assert live["storage"]["in_git"] is False
    assert REPO.resolve() not in root.resolve().parents and root.resolve() != REPO.resolve()


def test_committed_proposed_plan_matches_rebuild_with_explicit_supervisor_revision(live):
    p = REPO / tc.PLAN_FILE
    committed = load(p)
    fresh = json.loads(json.dumps(live, sort_keys=True, default=sorted))
    # The completed P1-17 plan stays immutable. The code-drift race fix deliberately
    # changes the supervisor and its two importing fingerprints; it must invalidate
    # reuse of the old authorization. Pin BOTH revisions, rather than ignoring code
    # identity differences or rewriting the frozen historical experiment.
    revisions = {
        ("collector", "supervisor_fingerprint"): (
            "sha256:eae6a705ad0d4417a48af9f85e3d446ada9e6b38f62c2536bc3c364723b565c9",
            "sha256:b5a8e9d5d29f9f3981e7b0708a0d78a006e774a6161a6dfb0c82c982248503f5"),
        ("collector", "runner_fingerprint"): (
            "sha256:4494581159545a29978dee742492b3294788030ee6421d3e91ff8515437c0bcf",
            "sha256:046f51dd77e8d64cff0ce01d08fbc26b1a9b63b7520deb208656dad365c07c5d"),
        ("collection", "module_fingerprint"): (
            "sha256:6c392bd78427b7f454d57728d81973d7b1383ce674b0175b3e8635f669f8a020",
            "sha256:c23a65672c68e483ff8072609ca6d200ee867fe373942756a0cf533eb50b3001"),
    }
    assert committed["plan_hash"] != fresh["plan_hash"]
    assert tc.check_plan(committed, frozen=False) == tc.check_plan(fresh, frozen=False) == []
    for (section, key), (historical, corrected) in revisions.items():
        assert committed["identities"][section][key] == historical
        assert fresh["identities"][section][key] == corrected
        fresh["identities"][section][key] = historical
    fresh["plan_hash"] = digest({k: v for k, v in fresh.items() if k not in ("plan_hash", "status")})
    assert committed == fresh, sorted(k for k in fresh if committed.get(k) != fresh[k])


def test_frozen_plan_keeps_the_proposed_hash_if_present():
    f = REPO / tc.FROZEN_PLAN_FILE
    if not f.exists():
        pytest.skip("plan not frozen yet")
    plan, prop = load(f), load(REPO / tc.PLAN_FILE)
    assert plan["status"] == tc.FROZEN and tc.check_plan(plan) == []
    assert plan["plan_hash"] == prop["plan_hash"]


def test_plan_tamper_and_scope_checks(live):
    for mutate in (lambda p: p["scenarios"].pop(), lambda p: p.update(attempt=2),
                   lambda p: p["limits"].update(max_concurrent_groups=99)):
        t = copy.deepcopy(live)
        mutate(t)
        assert tc.check_plan(t, frozen=False)
    assert "not FROZEN" in " ".join(tc.check_plan(live))


def test_cost_estimate_and_lpt_wall():
    assert tc.lpt_wall([5, 4, 3, 3, 3], 2) == 10          # LPT, not the optimum 9
    assert tc.lpt_wall([10], 4) == 10
    sc = [{"scenario_id": "a", "people": 5, "max_steps": 10}, {"scenario_id": "b", "people": 110, "max_steps": 20}]
    basis = [{"people": 3, "t2_cpu_s": 2.0, "budget_limited": 0}, {"people": 101, "t2_cpu_s": 100.0,
                                                                  "budget_limited": 1}]
    c = tc.cost_estimate(sc, basis, 2)
    assert c["t2_cpu_s_central"] == 102.0 and c["wall_s_central_at_concurrency"] == 100.0
    assert c["t2_cpu_s_total_cap"] == 127.5 and c["t2_cpu_s_worst_case"] == 30 * tp.T2_BUDGET_S
    with pytest.raises(ValueError):
        tc.cost_estimate([{"scenario_id": "c", "people": 50, "max_steps": 1}], basis, 1)


# ----- pilot summary (Phase 1) --------------------------------------------------------------------------------
def test_committed_pilot_summary_is_self_consistent():
    s = load(REPO / tc.PILOT_SUMMARY_FILE)
    body = {k: v for k, v in s.items() if k != "summary_hash"}
    assert digest(body) == s["summary_hash"] and s["training_eligible"] is False
    assert len(s["groups"]) == 8 and s["people_total"] == 466 == sum(g["people"] for g in s["groups"])
    for t in CANDIDATES:
        tot = s["totals"][t]
        for k in ("casualties", "evacuated", "unresolved"):
            assert tot[k] == sum(g["teachers"][t][k] for g in s["groups"])
        assert tot["casualties"] + tot["evacuated"] + tot["unresolved"] == 466
    assert {t: s["totals"][t]["casualties_plus_unresolved"] for t in CANDIDATES} == \
        {"T0": 173, "T1": 126, "T2": 124, "T3": 111, "T4": 123}
    prov = s["provenance"]
    assert len(prov["executions"]) == 2 and "DEVIATION" in prov["protocol_compliance"]
    assert prov["technical_integrity"].startswith("PASS")


# ----- authorization ------------------------------------------------------------------------------------------
def _frozen(plan):
    body = tc.plan_body(plan)
    return {**body, "status": tc.FROZEN, "plan_hash": plan["plan_hash"]}


def _approved(plan):
    a = tc.authorization_template(plan)
    a.update(status=tp.AUTH_APPROVED, execution_commit=H, plan_freeze={"commit": H, "path": tc.FROZEN_PLAN_FILE},
             reviewer={"github_login": "sankalphs"})
    a["approval"].update(pull_request=1, owner_instruction="owner said run", recorded_by="agent")
    return a


def test_template_is_pending_and_never_valid(live):
    t = tc.authorization_template(live)
    assert t["status"] == tp.AUTH_PENDING and t["approval"]["pull_request"] is None
    assert len(t["scenarios"]) == 189 and t["storage_root"] == live["storage"]["root"]
    assert tc.check_authorization(t, _frozen(live))


def test_authorization_binds_every_field(live):
    plan = _frozen(live)
    assert tc.check_authorization(_approved(plan), plan) == []
    for k, v in (("plan_hash", "sha256:x"), ("attempt", 2), ("storage_root", "C:/elsewhere"),
                 ("limits", {}), ("manifest_hash", "m"), ("code_identities", {}), ("namespace", "p1-12")):
        a = _approved(plan)
        a[k] = v
        assert tc.check_authorization(a, plan), k
    a = _approved(plan)
    a["scenarios"] = a["scenarios"][:-1]
    assert tc.check_authorization(a, plan)
    a = _approved(plan)
    a["teacher_fingerprints"]["T2"]["code_fingerprint"] = "drift"
    assert tc.check_authorization(a, plan)
    pilot_auth = load(REPO / tp.AUTH_FILE)
    assert "kind must be" in " ".join(tc.check_authorization(pilot_auth, plan))


# ----- evidence root and sessions -----------------------------------------------------------------------------
def _root_plan(root, **lim):
    return {"plan_hash": "sha256:mini", "storage": {"root": str(root)},
            "limits": {**tc.BASE_LIMITS, "max_sessions": 2, **lim}}


def test_check_root_rules(tmp_path):
    root = tmp_path / "ev"
    plan = _root_plan(root)
    assert tc.check_root(root, plan, resume=False) == 1
    with pytest.raises(tc.CollectionBlocked, match="not the plan"):
        tc.check_root(tmp_path / "other", plan, resume=False)
    with pytest.raises(tc.CollectionBlocked, match="existing session 1"):
        tc.check_root(root, plan, resume=True)
    root.mkdir()
    (root / "session-1.json").write_text(json.dumps({"plan_hash": "sha256:mini"}), encoding="utf-8")
    with pytest.raises(tc.CollectionBlocked, match="not empty"):
        tc.check_root(root, plan, resume=False)
    assert tc.check_root(root, plan, resume=True) == 2
    (root / "STOP").write_text("", encoding="utf-8")
    with pytest.raises(tc.CollectionBlocked, match="STOP"):
        tc.check_root(root, plan, resume=True)
    (root / "STOP").unlink()
    (root / "session-2.json").write_text(json.dumps({"plan_hash": "sha256:mini"}), encoding="utf-8")
    with pytest.raises(tc.CollectionBlocked, match="max_sessions"):
        tc.check_root(root, plan, resume=True)
    (root / "session-2.json").write_text(json.dumps({"plan_hash": "sha256:other"}), encoding="utf-8")
    with pytest.raises(tc.CollectionBlocked, match="another plan"):
        tc.check_root(root, plan, resume=True)
    (root / "evidence.json").write_text("{}", encoding="utf-8")
    with pytest.raises(tc.CollectionBlocked, match="closed"):
        tc.check_root(root, plan, resume=True)


# ----- supervised execution on tiny fixtures ------------------------------------------------------------------
def tiny(i):
    return scenario(["E...#...", "....#...", "........", "..F.....", "......E."],
                    {"P1": (2, 1), "P2": (2, 2), "P3": (4, 5)}, fire="stochastic", p=0.5, interval=1, seed=11 + i,
                    max_steps=6, enter_fire="blocked", sid=f"p1-17-fixture-{i}")


N_FX = 3


def mini_plan(root, **lim):
    pins = tp.teacher_pins()
    sc = [{"scenario_id": f"fx-{i}", "scenario_hash": tiny(i).content_hash(), "label": "feasible",
           "lineage_id": f"lin-{i}", "family_id": f"fam-{i % 2}", "people": 3, "fire_model": "stochastic",
           "max_steps": 6} for i in range(N_FX)]
    return {"plan_hash": "sha256:mini", "attempt": 1, "namespace": tc.NAMESPACE, "scenarios": sc,
            "teacher_pins": pins, "storage": {"root": str(root)},
            "limits": {**tc.BASE_LIMITS, "group_wall_s": 120, "session_wall_s": 600, "max_concurrent_groups": 2,
                       "min_free_ram_gib_to_start": 0.0, "t2_cpu_s_total_cap": 1e9, **lim},
            "regeneration": {"scenario_ids": ["fx-0"]}, "source": {"manifest_hash": "fixture"}}


def mini_scenarios():
    return [({"scenario_id": f"fx-{i}", "lineage_id": f"lin-{i}"}, tiny(i)) for i in range(N_FX)]


def test_parallel_session_completes_verifies_regenerates_and_closes(tmp_path):
    root = tmp_path / "ev"
    plan = mini_plan(root)
    out = tc.execute(plan, mini_scenarios(), root, {"execution_commit": H}, poll_s=0.1)
    assert out["closed"] and out["coverage"]["verified_complete"] == N_FX and out["max_live_groups"] == 2
    ev = load(root / "evidence.json")
    assert ev["coverage"]["full_train_verified"] is False          # 3 fixtures are not the 189 TRAIN groups
    assert ev["regeneration"]["fx-0"]["identical"] and ev["regeneration_deterministic_ok"]
    for sid, scn in (("fx-0", tiny(0)), ("fx-2", tiny(2))):
        rep = load(root / "groups" / f"{sid}.json")
        mt.verify_fixture(rep, scn)
        assert rep["scope"] == mt.SCOPE_TRAIN_FULL and rep["collection"]["plan_hash"] == "sha256:mini"
        assert rep["training_eligible"] is False
        row = load(root / "rows" / f"{sid}.json")
        assert row["verified"] and row["file_sha256"] == tc._sha256_file(root / "groups" / f"{sid}.json")
        assert (root / "started" / f"{sid}.json").exists()
    with pytest.raises(tc.CollectionBlocked):
        tc.execute(plan, mini_scenarios(), root, {"execution_commit": H}, session=2)
    v = tc.verify_collection(plan, root, mini_scenarios())
    assert v["status_counts"] == {"verified_complete": N_FX} and v["full_train_verified"] is False
    assert v["teacher_episodes_complete"] == N_FX * 5
    assert sum(v["totals_verified_complete"]["T0"][k] for k in ("casualties", "evacuated", "unresolved")) == 3 * N_FX
    (root / "groups" / "fx-1.json").write_text("{}", encoding="utf-8")
    v2 = tc.verify_collection(plan, root, mini_scenarios())
    bad = next(g for g in v2["groups"] if g["scenario_id"] == "fx-1")
    assert bad["status"] != "verified_complete" and bad["problems"]


def test_resume_runs_only_never_started_groups_and_records_interrupted(tmp_path):
    root = tmp_path / "ev"
    plan = mini_plan(root, t2_cpu_s_total_cap=-1.0)          # admission closed at once: nothing starts
    out = tc.execute(plan, mini_scenarios(), root, {"execution_commit": H}, poll_s=0.1)
    assert not out["closed"] and out["halt_reason"] == "t2_cpu_total_cap"
    assert not list((root / "started").glob("*.json"))
    tc.write_once(root / "started" / "fx-1.json", {"scenario_id": "fx-1", "note": "simulated crash mid-group"})
    plan2 = mini_plan(root)
    plan2["limits"]["max_sessions"] = 3
    out2 = tc.execute(plan2, mini_scenarios(), root, {"execution_commit": H}, session=2, poll_s=0.1)
    assert out2["newly_interrupted"] == ["primary:fx-1"] and out2["closed"]
    assert not (root / "rows" / "fx-1.json").exists() and (root / "interrupted" / "primary-fx-1.json").exists()
    ev = load(root / "evidence.json")
    assert ev["coverage"]["interrupted"] == 1 and ev["coverage"]["verified_complete"] == N_FX - 1
    assert [s["session"] for s in ev["sessions"]] == [1, 2]


def test_resume_refuses_a_group_file_changed_after_its_row(tmp_path):
    root = tmp_path / "ev"
    plan = mini_plan(root, max_concurrent_groups=1)
    plan["scenarios"] = plan["scenarios"][:1]
    plan["regeneration"] = {"scenario_ids": []}
    out = tc.execute(plan, mini_scenarios()[:1], root, {"execution_commit": H}, poll_s=0.1)
    assert out["closed"]
    (root / "evidence.json").unlink()                         # pretend the attempt was still open
    g = root / "groups" / "fx-0.json"
    g.write_text(g.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(tc.CollectionBlocked, match="changed after its row"):
        tc.execute(plan, mini_scenarios()[:1], root, {"execution_commit": H}, session=2)


def test_operator_stop_file_closes_admission(tmp_path):
    root = tmp_path / "ev"
    plan = mini_plan(root)
    paths = tc.collection_paths(root)
    sched = tc._Scheduler(paths, lambda *a, **k: None, limit=1, min_free_gib=0.0, deadline_epoch=time.time() + 60,
                          t2_cap=1e9, t2_used=0.0, target=None)
    root.mkdir()
    paths["stop"].write_text("", encoding="utf-8")
    assert sched._admit_reason() == "operator_stop"
    sched.deadline = time.time() - 1
    paths["stop"].unlink()
    assert sched._admit_reason() == "session_wall_cap"
    assert plan["limits"]["max_concurrent_groups"] == 2


def test_supervisor_stop_hook_interrupts_live_teachers():
    scn = tiny(0)
    jobs = tp._jobs({"teacher_pins": tp.teacher_pins()})
    t0 = time.monotonic()
    run = ps.run_group(scn, jobs, decision_cpu_s=tp.LIMITS["decision_cpu_s"], group_wall_s=120,
                       target=fk.sleep_t1, stop_requested=lambda: time.monotonic() - t0 > 3)
    assert run["interrupted"] and run["failures"]["T1"]["kind"] == "interrupted"
    assert time.monotonic() - t0 < 60


def test_fire_entry_moves_flags_proposals_into_burning_cells():
    d = {"observation": {"burning": [[1, 2]], "people": [
        {"id": "P1", "position": [1, 1], "status": "active"}, {"id": "P2", "position": [0, 2], "status": "active"},
        {"id": "P3", "position": [2, 2], "status": "evacuated"}]},
         "proposal": {"actions": {"P1": "east", "P2": "north", "P3": "north"}}}
    assert tc.fire_entry_moves(d) == ["P1"]


def test_host_resource_probes_return_numbers():
    import os
    free = tc.host_free_ram_gib()
    assert free is None or free > 0
    rss = tc.peak_rss_mb(os.getpid())
    assert rss is None or rss > 1
