"""T4 CPU feasibility pilot and DEV gate (issue #24): synthetic cells, cost model, gate logic, refusal paths.

No test here runs T4, A0 or A1 on a DEV scenario.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from ember_v1.contract import validate_observation
from ember_v1.sim import Simulator
from ember_v1.teacher import t4_eval, t4_pilot
from ember_v1.teacher.t4 import ABLATIONS, T4RiskPriority

CFG = dict(T4RiskPriority().identity.config)
ROOT = Path(__file__).resolve().parents[1]


def test_pilot_cells_are_synthetic_valid_repeatable_and_cover_the_preregistered_grid():
    cells = t4_pilot.pilot_cells()
    assert len(cells) == 52
    for c in cells:
        s = t4_pilot.pilot_scenario(c)
        assert s == t4_pilot.pilot_scenario(c)
        assert s.scenario_id.startswith("t4-pilot-") and s.fire_model == c["model"]
        assert 1 <= len(s.people) <= c["n"] and len(s.exits) == c["exits"]
        validate_observation(Simulator(s).observe())
    assert {c["n"] for c in cells} == set(t4_pilot.SIZES)
    assert {c["grid"] for c in cells} == {t4_pilot.SMALL, t4_pilot.LARGE}
    assert {c["model"] for c in cells} == {"stochastic", "deterministic"}
    assert {c["exits"] for c in cells} == {1, 3, 6}
    assert {c["topology"] for c in cells} == set(t4_pilot.TOPOLOGIES)
    assert {c["kind"] for c in cells} == {"base", "high_id_risk", "near_entrapment", "unreachable"}


def test_high_id_risk_cells_give_the_highest_id_to_the_person_nearest_the_fire():
    c = {"kind": "high_id_risk", "topology": "open_room", "grid": t4_pilot.SMALL, "n": 50, "model": "deterministic",
         "exits": 3}
    s = t4_pilot.pilot_scenario(c)
    d = t4_pilot._bfs(*t4_pilot.SMALL, frozenset(map(tuple, s.to_dict()["obstacles"])), tuple(s.ignitions[0]))
    ids = sorted(s.to_dict()["people"], key=lambda p: p["id"])
    dist = [d[tuple(p["position"])] for p in ids]
    assert dist == sorted(dist, reverse=True)


def test_unreachable_cells_start_people_without_any_exit_path():
    c = {"kind": "unreachable", "topology": "multi_room", "grid": t4_pilot.SMALL, "n": 50, "model": "stochastic",
         "exits": 3}
    obs = Simulator(t4_pilot.pilot_scenario(c)).observe()
    wl = t4_pilot.workload(obs)
    assert wl["routable"] < wl["active"]


def test_nominal_work_formula():
    wl = {"area": 400, "safe_exits": 3, "stochastic": True, "active": 10}
    assert t4_pilot.nominal_work(wl) == 400 * (1 + 3 * 4) + 400
    wl = {"area": 1000, "safe_exits": 1, "stochastic": False, "active": 200}
    assert t4_pilot.nominal_work(wl) == 1000 * 4 + 8000


def _row(v, alpha=0.001, limit=None, cpu=50.0, blocked=0, fallback=0, routable=10, mismatches=0, yields=0,
         eff=0):
    return {"variant": v, "alpha": alpha, "alpha_eval": 0.0001, "limit_hit": limit, "cpu_ms": cpu,
            "predicted_blocked": 0, "blocked_moves": blocked, "fallback_people": fallback, "route_people": routable,
            "prediction_mismatches": mismatches, "yields_over_id_rule": yields,
            "effective_yields_over_id_rule": eff}


def _good():
    return [_row("A0")] * 10 + [_row("A1")] * 10 + [_row("A2", yields=1, eff=1)] * 10


PROFILE = [{"position": 1, "area": 400, "safe_exits": 2, "stochastic": True, "active": 10},
           {"position": 2, "area": 400, "safe_exits": 2, "stochastic": False, "active": 10}]


def test_estimate_uses_unbudgeted_alpha_and_caps():
    pilot = _good() + [_row("A2", alpha=50.0, limit="cpu_budget_ms")]
    est = t4_pilot.estimate(CFG, pilot, PROFILE, {"t1_wall_s": 2.0})
    w = sum(t4_pilot.nominal_work(p) for p in PROFILE)
    assert est["alpha_wall_ms_per_work_unit"]["A2"]["n"] == 10
    assert est["central"]["variants"]["A2"]["cpu_s_total"] == pytest.approx(0.001 * w / 1000)
    assert est["central"]["comparison_cpu_s_total"] == pytest.approx(3 * 0.001 * w / 1000 + 4 * 0.0001 * w / 1000
                                                                     + 2.0)
    assert est["upper"]["episode_factor"] == 1.25
    cap = CFG["cpu_budget_ms"] + t4_pilot.OVERRUN_MS
    big = t4_pilot.estimate(CFG, [_row(v, alpha=100.0) for v in ("A0", "A1", "A2")], PROFILE, {"t1_wall_s": 0.0})
    assert big["upper"]["variants"]["A2"]["predicted_cap_decisions"] == 2
    assert big["upper"]["variants"]["A2"]["cpu_s_total"] == pytest.approx(2 * cap * 1.25 / 1000)
    assert "error" in t4_pilot.estimate(CFG, [_row("A0")], PROFILE, {"t1_wall_s": 0.0})


EST = {"upper": {"comparison_cpu_s_total": 100.0,
                 "variants": {v: {"predicted_cap_fraction": 0.0} for v in ("A0", "A1", "A2")}}}


def test_gate_decision_and_limiting_factors():
    g = t4_pilot.gate(CFG, _good(), True, 50.0, EST)
    assert g["decision"] == "FULL_DEV_FEASIBLE" and g["limiting"] == []
    hi_cap = {"upper": {"comparison_cpu_s_total": 1.0,
                        "variants": {v: {"predicted_cap_fraction": 0.5} for v in ("A0", "A1", "A2")}}}
    cases = [
        ((CFG, _good(), False, 50.0, EST), "G1_bounded"),
        ((CFG, _good() + [_row("A2", blocked=1)], True, 50.0, EST), "G1_bounded"),
        ((CFG, _good() + [_row("A2", mismatches=1)], True, 50.0, EST), "G1_bounded"),
        ((CFG, _good() + [_row("A2", cpu=1e6)], True, 50.0, EST), "G1_bounded"),
        ((CFG, _good(), True, 5000.0, EST), "G1_bounded"),
        ((CFG, _good(), True, 50.0, {"upper": {**EST["upper"], "comparison_cpu_s_total": 1e9}}), "G2_dev_cpu"),
        ((CFG, _good(), True, 50.0, hi_cap), "G3_cap_hits"),
        ((CFG, _good() + [_row("A1", fallback=10)] * 10, True, 50.0, EST), "G4_planner_health"),
        ((CFG, _good() + [_row("A0", limit="cpu_budget_ms")] * 10, True, 50.0, EST), "G4_planner_health"),
    ]
    for args, factor in cases:
        g = t4_pilot.gate(*args)
        assert g["decision"] == "BLOCKED_BY_COMPUTE" and factor in g["limiting"], factor
    no_yield = [r for r in _good() if r["variant"] != "A2"] + [_row("A2")] * 10
    assert t4_pilot.gate(CFG, no_yield, True, 50.0, EST)["decision"] == "BLOCKED_BY_MECHANISM_CHECK"
    a0_yield = _good() + [_row("A0", yields=1)]
    assert t4_pilot.gate(CFG, a0_yield, True, 50.0, EST)["limiting"] == ["G5_mechanism"]


def _fake_pilot(decision="FULL_DEV_FEASIBLE", teachers=None):
    teachers = teachers or {k: cls().identity.to_dict() for k, cls in ABLATIONS.items()}
    return {"gate": {"decision": decision}, "teachers": teachers,
            "estimate": {"upper": {"comparison_cpu_s_total": 1000.0}}}


def test_dev_harness_refuses_without_a_passing_matching_pilot(tmp_path):
    assert t4_eval.check_gate(_fake_pilot()) == 1500.0
    wrong = {k: cls().identity.to_dict() for k, cls in ABLATIONS.items()}
    wrong["A2"] = {**wrong["A2"], "config_hash": "sha256:other"}
    missing = {k: v for k, v in _fake_pilot()["teachers"].items() if k != "A1"}
    for bad in (_fake_pilot("BLOCKED_BY_COMPUTE"), _fake_pilot("BLOCKED_BY_MECHANISM_CHECK"), _fake_pilot(teachers=wrong),
                _fake_pilot(teachers=missing), {}):
        with pytest.raises(t4_eval.GateError):
            t4_eval.check_gate(bad)
    path = tmp_path / "pilot.json"
    path.write_text(json.dumps(_fake_pilot("BLOCKED_BY_COMPUTE")), encoding="utf-8")
    with pytest.raises(SystemExit):
        t4_eval.main(["--pilot-report", str(path), "--out", str(tmp_path / "out.json")])
    assert not (tmp_path / "out.json").exists()
    with pytest.raises(FileNotFoundError):
        t4_eval.evaluate(tmp_path / "missing.json")


def test_dev_wall_limit():
    assert t4_eval.wall_limit(0.0) == 900.0
    assert t4_eval.wall_limit(1000.0) == 1500.0
    assert t4_eval.wall_limit(1e9) == 6 * 3600.0


def test_dev_harness_compares_t1_and_the_three_frozen_variants_only():
    assert t4_eval.POLICY_ORDER == ("T1", "A0", "A1", "A2")
    assert t4_eval.COMPARISONS[0] == ("A2", "T1")
    pols = t4_eval.policies()
    assert {k: p.identity.name for k, p in pols.items()} == {"T1": "T1", "A0": "T4-A0", "A1": "T4-A1", "A2": "T4"}
    src = (ROOT / "src/ember_v1/teacher/t4_eval.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom):
            assert all(not a.name.startswith(("t0", "t2", "t3")) for a in node.names)
            assert not str(node.module).startswith(("t0", "t2", "t3"))


def _entry(ci, p, better, worse, direction="lower_is_better"):
    return {"direction": direction, "family_cluster_bootstrap_ci95": ci, "family_sign_test_p_two_sided": p,
            "family_a2_better": better, "family_a2_worse": worse}


def test_preregistered_decision_rule():
    v = t4_eval.verdict
    assert v(_entry([-2.0, -0.5], 0.01, 12, 1), "A2") == "better"
    assert v(_entry([0.5, 2.0], 0.01, 1, 12), "A2") == "worse"
    assert v(_entry([0.01, 0.2], 0.001, 14, 0, "higher_is_better"), "A2") == "better"
    assert v(_entry([-2.0, -0.5], 0.06, 10, 3), "A2") == "no_evidence_of_difference"
    assert v(_entry([-2.0, 0.0], 0.01, 12, 1), "A2") == "no_evidence_of_difference"
    assert v(_entry([-2.0, -0.5], 0.01, 1, 12), "A2") == "no_evidence_of_difference"
    assert v(_entry(None, 0.01, 12, 1), "A2") == "no_evidence_of_difference"


def test_step_metrics_and_person_outcomes_on_a_synthetic_episode():
    from test_teacher_t4 import CONTEST, det
    s = det(CONTEST)
    for v, saved in (("A0", 1), ("A2", 2)):
        rec, lat = t4_eval.run_episode(s, ABLATIONS[v](), CFG)
        assert rec["evacuated"] == saved and len(lat) == rec["steps"]
        assert sum(rec["initial_strata"].values()) == 2


def test_committed_pilot_report_matches_frozen_t4_if_present():
    path = Path(t4_pilot.DEFAULT_OUT)
    if not path.exists():
        pytest.skip("pilot not run yet")
    rep = json.loads(path.read_text(encoding="utf-8"))
    assert {k: v["config_hash"] for k, v in rep["teachers"].items()} == \
        {k: cls().identity.config_hash for k, cls in ABLATIONS.items()}
    assert rep["gate"]["decision"] in ("FULL_DEV_FEASIBLE", "BLOCKED_BY_COMPUTE", "BLOCKED_BY_MECHANISM_CHECK")
    assert rep["gate"] == t4_pilot.gate(rep["teachers"]["A2"]["config"], rep["pilot"],
                                        rep["pilot_complete"] and rep["dev_profile_complete"],
                                        rep["resources"]["peak_rss_mb"], rep["estimate"])
    assert rep["dev_use"].startswith("frozen T1 replay")
    assert not Path(t4_eval.DEFAULT_OUT).exists() or rep["gate"]["decision"] == "FULL_DEV_FEASIBLE"
