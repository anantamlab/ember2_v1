"""T2 CPU feasibility pilot and DEV gate (issue #9): cost model, gate logic, refusal paths.

No test here runs T2 on a DEV scenario.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ember_v1.contract import validate_observation
from ember_v1.sim import Simulator
from ember_v1.teacher import t2_eval, t2_pilot
from ember_v1.teacher.t2 import T2StochasticRollout

CFG = dict(T2StochasticRollout().identity.config)


def test_pilot_scenarios_are_synthetic_valid_and_repeatable():
    for topology, n, model in t2_pilot.pilot_cells():
        s = t2_pilot.pilot_scenario(topology, n, model)
        assert s == t2_pilot.pilot_scenario(topology, n, model)
        assert s.scenario_id.startswith("t2-pilot-") and s.fire_model == model
        assert 1 <= len(s.people) <= n
        validate_observation(Simulator(s).observe())
    assert {t for t, _, _ in t2_pilot.pilot_cells()} == set(t2_pilot.TOPOLOGIES)
    assert {n for _, n, m in t2_pilot.pilot_cells() if m == "stochastic"} == set(t2_pilot.SIZES)


def test_nominal_rollout_steps():
    assert t2_pilot.nominal_steps(CFG, "stochastic", 100, 0, 500) == 8 * (1 + 4 * 4) * 8
    assert t2_pilot.nominal_steps(CFG, "deterministic", 2, 0, 500) == 1 * (1 + 4 * 2) * 8
    assert t2_pilot.nominal_steps(CFG, "stochastic", 1, 97, 100) == 8 * 5 * 3


def _pilot_row(alpha, model="stochastic", limit=None, cpu=1000.0, blocked=()):
    return {"alpha": alpha, "fire_model": model, "limit_hit": limit, "t2_cpu_ms": cpu,
            "predicted_blocked_outcomes": list(blocked)}


def _profile(n, model="stochastic", c1=10.0, active=50):
    return [{"position": i % 3, "fire_model": model, "step": 0, "max_steps": 500, "area": 900,
             "active": active, "t1_wall_ms": c1} for i in range(n)]


def test_estimate_uses_unbudgeted_alpha_and_caps_at_budget():
    pilot = [_pilot_row(a) for a in (0.5, 0.6, 0.7, 0.8)] + [_pilot_row(9.0, limit="cpu_budget_ms")] + \
            [_pilot_row(a, "deterministic") for a in (0.4, 0.4, 0.4)]
    est = t2_pilot.estimate(CFG, pilot, _profile(10) + _profile(5, "deterministic", active=2))
    assert est["alpha"]["stochastic"]["n"] == 4                       # the budget-limited decision is excluded
    nom_s = t2_pilot.nominal_steps(CFG, "stochastic", 50, 0, 500)
    nom_d = t2_pilot.nominal_steps(CFG, "deterministic", 2, 0, 500)
    a_s = est["alpha"]["stochastic"]["median"]
    want = 10 * min(10 * (1 + a_s * nom_s), CFG["cpu_budget_ms"] + 10) + 5 * 10 * (1 + 0.4 * nom_d)
    assert est["central"]["t2_cpu_s_total"] == pytest.approx(want / 1000)
    assert est["upper"]["episode_factor"] == t2_pilot.EPISODE_FACTOR
    big = t2_pilot.estimate(CFG, pilot, _profile(10, c1=1000.0))
    assert big["upper"]["predicted_budget_hit_fraction"] == 1.0


def test_gate_decision_and_limiting_factors():
    est_ok = {"upper": {"t2_cpu_s_total": 3600.0, "predicted_budget_hit_fraction": 0.0}}
    rows = [_pilot_row(0.5)]
    g = t2_pilot.gate(CFG, rows, True, 100.0, est_ok)
    assert g["decision"] == "FULL_DEV_FEASIBLE" and g["limiting"] == []
    est_slow = {"upper": {"t2_cpu_s_total": t2_pilot.DEV_CPU_LIMIT_S + 1, "predicted_budget_hit_fraction": 0.2}}
    g = t2_pilot.gate(CFG, rows, True, 100.0, est_slow)
    assert g["decision"] == "BLOCKED_BY_COMPUTE" and g["limiting"] == ["G2_dev_cpu", "G3_budget_hits"]
    assert t2_pilot.gate(CFG, rows, False, 100.0, est_ok)["limiting"] == ["G1_bounded"]
    assert t2_pilot.gate(CFG, rows, True, None, est_ok)["limiting"] == ["G1_bounded"]
    over = [_pilot_row(0.5, cpu=CFG["cpu_budget_ms"] + t2_pilot.OVERRUN_MS + 1)]
    assert t2_pilot.gate(CFG, over, True, 100.0, est_ok)["limiting"] == ["G1_bounded"]
    blocked = [_pilot_row(0.5, blocked=["blocked_conflict"])]
    assert t2_pilot.gate(CFG, blocked, True, 100.0, est_ok)["limiting"] == ["G1_bounded"]
    assert t2_pilot.gate(CFG, rows, True, 100.0, {"error": "x"})["decision"] == "BLOCKED_BY_COMPUTE"


def test_peak_memory_is_measured():
    mb = t2_pilot.peak_rss_mb()
    assert mb is not None and 1 < mb < 16384


def _fake_pilot(decision="FULL_DEV_FEASIBLE", **over):
    t2 = T2StochasticRollout()
    rep = {"gate": {"decision": decision}, "teacher": t2.identity.to_dict(), "teacher_seed": t2.seed,
           "continuation": t2.continuation, "estimate": {"upper": {"t2_cpu_s_total": 1000.0}}}
    rep.update(over)
    return rep


def test_dev_harness_refuses_without_a_passing_matching_pilot(tmp_path):
    t2 = T2StochasticRollout()
    assert t2_eval.check_gate(_fake_pilot(), t2) == 1500.0
    for bad in (_fake_pilot("BLOCKED_BY_COMPUTE"), _fake_pilot(teacher={"config_hash": "sha256:other"}),
                _fake_pilot(teacher_seed=1), _fake_pilot(continuation={"name": "T0"}), {}):
        with pytest.raises(t2_eval.GateError):
            t2_eval.check_gate(bad, t2)
    path = tmp_path / "pilot.json"
    path.write_text(json.dumps(_fake_pilot("BLOCKED_BY_COMPUTE")), encoding="utf-8")
    with pytest.raises(SystemExit):
        t2_eval.main(["--pilot-report", str(path), "--out", str(tmp_path / "out.json")])
    assert not (tmp_path / "out.json").exists()


def test_committed_pilot_report_matches_frozen_t2_if_present():
    path = Path(t2_pilot.DEFAULT_OUT)
    if not path.exists():
        pytest.skip("pilot not run yet")
    rep = json.loads(path.read_text(encoding="utf-8"))
    t2 = T2StochasticRollout()
    assert rep["teacher"]["config_hash"] == t2.identity.config_hash and rep["teacher_seed"] == t2.seed
    assert rep["continuation"] == t2.continuation
    assert rep["gate"]["decision"] in ("FULL_DEV_FEASIBLE", "BLOCKED_BY_COMPUTE")
    assert rep["gate"] == t2_pilot.gate(rep["teacher"]["config"], rep["pilot"],
                                        rep["pilot_complete"] and rep["dev_profile_complete"],
                                        rep["resources"]["peak_rss_mb"], rep["estimate"])
    assert rep["dev_use"].startswith("frozen T1 replay")


def test_committed_dev_report_is_complete_and_consistent_if_present():
    """Reads the committed T2-vs-T1 DEV report only; T2 is not run here."""
    path = Path(t2_eval.DEFAULT_OUT)
    if not path.exists():
        pytest.skip("DEV comparison not run yet")
    rep = json.loads(path.read_text(encoding="utf-8"))
    t2 = T2StochasticRollout()
    idx = json.loads((t2_eval.ev.REPO / "data/splits/p1-02-v1/index.json").read_text(encoding="utf-8"))
    assert rep["eval_version"] == t2_eval.EVAL_VERSION and rep["complete"] is True
    assert rep["dev"]["split"] == "dev" and rep["dev"]["plan_id"] == "p1-02-v1"
    assert rep["dev"]["manifest_hash"] == idx["splits"]["dev"]["manifest_hash"]
    assert rep["scenarios_evaluated"] == len(rep["rows"]) == 48 and rep["families_evaluated"] == 16
    assert rep["environment"]["src_dirty"] is False
    assert rep["policies"]["T2"]["config_hash"] == t2.identity.config_hash and rep["teacher_seed"] == t2.seed
    assert rep["continuation"] == t2.continuation
    assert rep["policies"]["T1"]["config_hash"] == t2.continuation["config_hash"]
    pilot = json.loads(Path(t2_pilot.DEFAULT_OUT).read_text(encoding="utf-8"))
    assert rep["pilot_gate"] == pilot["gate"] and rep["wall_limit_s"] == t2_eval.check_gate(pilot, t2)
    det = [{k: (t2_eval.ev.deterministic_view(v) if k in t2_eval.POLICY_ORDER else v) for k, v in r.items()}
           for r in rep["rows"]]
    assert t2_eval.digest(det) == rep["deterministic_digest"]
    assert rep["paired_t2_minus_t1"] == t2_eval.t1_eval.paired(rep["rows"], "T2", "T1", t2_eval.BOOTSTRAP_SEED,
                                                               t2_eval.PAIRED_METRICS)
    # Frozen T1 inside the T2 run reproduces the P1-04 T1 episodes exactly.
    p104 = json.loads((t2_eval.ev.REPO / "data/baselines/p1-04-t1-dev-v1.json").read_text(encoding="utf-8"))
    old = {r["scenario_hash"]: r["T1"]["final_transition_hash"] for r in p104["rows"]}
    assert all(old[r["scenario_hash"]] == r["T1"]["final_transition_hash"] for r in rep["rows"])
    for r in rep["rows"]:
        assert r["T2"]["failure"] is None and r["T2"]["t2"]["decisions"] == r["T2"]["decisions"]
