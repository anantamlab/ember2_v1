"""T3 CPU feasibility pilot and DEV gate (issue #23): cost model, gate logic, refusal paths.

No test here runs T3 on a DEV scenario; the committed DEV report is only read and recomputed.
"""
from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path

import pytest

from ember_v1.contract import validate_observation
from ember_v1.sim import Simulator
from ember_v1.sim.canonical import digest
from ember_v1.teacher import t3_eval, t3_pilot
from ember_v1.teacher.t3 import T3DynamicACO

CFG = dict(T3DynamicACO().identity.config)
FROZEN_T3_HASH = "sha256:5a8d4985b1f65d5de18a06c86d8dd522e2ca95e2accd583165b7e7d7ea1d02e0"


def test_pilot_scenarios_are_synthetic_valid_repeatable_and_cover_the_grid():
    cells = t3_pilot.pilot_cells()
    for c in cells:
        s = t3_pilot.pilot_scenario(c["topology"], c["n"], c["model"], c["grid"], c["exits"])
        assert s == t3_pilot.pilot_scenario(c["topology"], c["n"], c["model"], c["grid"], c["exits"])
        assert s.scenario_id.startswith("t3-pilot-") and s.fire_model == c["model"]
        assert 1 <= len(s.people) <= c["n"] and len(s.exits) == c["exits"]
        validate_observation(Simulator(s).observe())
    assert {c["n"] for c in cells} == set(t3_pilot.SIZES)
    assert {c["grid"] for c in cells} == {t3_pilot.SMALL, t3_pilot.LARGE}
    assert {c["model"] for c in cells} == {"stochastic", "deterministic"}
    assert {c["exits"] for c in cells} == {1, 3, 6}
    assert {"open_room", "corridor", "multi_room"} == {c["topology"] for c in cells}


def test_nominal_work_formula():
    wl = {"routable": 10, "dbar": 9.0, "area": 400, "active": 10}
    assert t3_pilot.nominal_work(CFG, wl) == 10 * (min(2 * 10, 64) + 10) * 10 + 400
    wl = {"routable": 200, "dbar": 4.0, "area": 1000, "active": 200}
    assert t3_pilot.nominal_work(CFG, wl) == 10 * (64 + 200) * 5 + 1000


def _row(alpha, limit=None, cpu=100.0, blocked=(), fallback=0, routable=10, mismatches=0):
    return {"alpha": alpha, "limit_hit": limit, "t3_cpu_ms": cpu, "predicted_blocked_outcomes": list(blocked),
            "fallback_people": fallback, "routable": routable, "prediction_mismatches": mismatches}


def test_estimate_uses_unbudgeted_alpha_and_caps():
    pilot = [_row(0.001), _row(0.002), _row(0.003), _row(9.0, limit="cpu_budget_ms")]
    profile = [{"position": 1, "routable": 10, "dbar": 9.0, "area": 400, "active": 10},
               {"position": 2, "routable": 10, "dbar": 9.0, "area": 400, "active": 10}]
    est = t3_pilot.estimate(CFG, pilot, profile, {"t0_wall_s": 1.0, "t1_wall_s": 2.0})
    w = t3_pilot.nominal_work(CFG, profile[0])
    assert est["alpha_wall_ms_per_work_unit"]["n"] == 3
    assert est["central"]["t3_cpu_s_total"] == pytest.approx(2 * 0.002 * w / 1000)
    assert est["central"]["comparison_cpu_s_total"] == pytest.approx(2 * 0.002 * w / 1000 + 3.0)
    assert est["upper"]["episode_factor"] == 1.25
    cap = CFG["cpu_budget_ms"] + t3_pilot.OVERRUN_MS
    assert est["hard_bound_t3_s"] == pytest.approx(2 * cap * 1.25 / 1000)
    big = t3_pilot.estimate(CFG, [_row(100.0)] * 3, profile, {"t0_wall_s": 0.0, "t1_wall_s": 0.0})
    assert big["upper"]["predicted_cap_decisions"] == 2
    assert big["upper"]["t3_cpu_s_total"] == pytest.approx(2 * cap * 1.25 / 1000)


def test_gate_decision_and_limiting_factors():
    good = [_row(0.001)] * 10
    est = {"upper": {"comparison_cpu_s_total": 100.0, "predicted_cap_fraction": 0.0}}
    g = t3_pilot.gate(CFG, good, True, 50.0, est)
    assert g["decision"] == "FULL_DEV_FEASIBLE" and g["limiting"] == []
    cases = [
        ((CFG, good, False, 50.0, est), "G1_bounded"),
        ((CFG, [_row(0.001, blocked=("blocked_conflict",))], True, 50.0, est), "G1_bounded"),
        ((CFG, [_row(0.001, mismatches=1)], True, 50.0, est), "G1_bounded"),
        ((CFG, [_row(0.001, cpu=1e6)], True, 50.0, est), "G1_bounded"),
        ((CFG, good, True, 5000.0, est), "G1_bounded"),
        ((CFG, good, True, 50.0, {"upper": {"comparison_cpu_s_total": 1e9, "predicted_cap_fraction": 0.0}}),
         "G2_dev_cpu"),
        ((CFG, good, True, 50.0, {"upper": {"comparison_cpu_s_total": 1.0, "predicted_cap_fraction": 0.5}}),
         "G3_cap_hits"),
        ((CFG, [_row(0.001, fallback=5)] * 10, True, 50.0, est), "G4_aco_health"),
        ((CFG, [_row(0.001, limit="cpu_budget_ms")] * 10, True, 50.0, est), "G4_aco_health"),
    ]
    for args, factor in cases:
        g = t3_pilot.gate(*args)
        assert g["decision"] == "BLOCKED_BY_COMPUTE" and factor in g["limiting"], factor


def _fake_pilot(decision="FULL_DEV_FEASIBLE", **over):
    t3 = T3DynamicACO()
    rep = {"gate": {"decision": decision}, "teacher": t3.identity.to_dict(), "teacher_seed": t3.seed,
           "estimate": {"upper": {"comparison_cpu_s_total": 1000.0}}}
    rep.update(over)
    return rep


def test_dev_harness_refuses_without_a_passing_matching_pilot(tmp_path):
    t3 = T3DynamicACO()
    assert t3_eval.check_gate(_fake_pilot(), t3) == 1500.0
    for bad in (_fake_pilot("BLOCKED_BY_COMPUTE"), _fake_pilot(teacher={"config_hash": "sha256:other"}),
                _fake_pilot(teacher_seed=1), {}):
        with pytest.raises(t3_eval.GateError):
            t3_eval.check_gate(bad, t3)
    path = tmp_path / "pilot.json"
    path.write_text(json.dumps(_fake_pilot("BLOCKED_BY_COMPUTE")), encoding="utf-8")
    with pytest.raises(SystemExit):
        t3_eval.main(["--pilot-report", str(path), "--out", str(tmp_path / "out.json")])
    assert not (tmp_path / "out.json").exists()
    with pytest.raises(FileNotFoundError):
        t3_eval.evaluate(tmp_path / "missing.json")


def test_eval_harness_runs_t0_t1_t3_and_never_t2():
    src = (Path(__file__).resolve().parents[1] / "src/ember_v1/teacher/t3_eval.py").read_text(encoding="utf-8")
    assert t3_eval.POLICY_ORDER == ("T0", "T1", "T3")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom):
            assert not str(node.module).startswith("t2") and all(not a.name.startswith("t2") for a in node.names)
    assert "p1-05" not in src.replace("P1-05 results are not read", "")


def test_committed_pilot_report_matches_frozen_t3_if_present():
    path = Path(t3_pilot.DEFAULT_OUT)
    if not path.exists():
        pytest.skip("pilot not run yet")
    rep = json.loads(path.read_text(encoding="utf-8"))
    t3 = T3DynamicACO()
    assert rep["teacher"]["config_hash"] == t3.identity.config_hash and rep["teacher_seed"] == t3.seed
    assert rep["gate"]["decision"] in ("FULL_DEV_FEASIBLE", "BLOCKED_BY_COMPUTE")
    assert rep["gate"] == t3_pilot.gate(rep["teacher"]["config"], rep["pilot"],
                                        rep["pilot_complete"] and rep["dev_profile_complete"],
                                        rep["resources"]["peak_rss_mb"], rep["estimate"])
    assert rep["dev_use"].startswith("frozen T0/T1 replay")


def test_dev_wall_limit_amendment_a1():
    """min(6 h, max(900 s, 1.5 x upper estimate)); the guard is elapsed wall, the estimate policy CPU only."""
    assert t3_eval.wall_limit(0.0) == 900.0
    assert t3_eval.wall_limit(187.6) == 900.0                    # the committed pilot's upper estimate
    assert t3_eval.wall_limit(600.0) == 900.0
    assert t3_eval.wall_limit(1000.0) == 1500.0
    assert t3_eval.wall_limit(14400.0) == 6 * 3600.0
    assert t3_eval.wall_limit(1e9) == 6 * 3600.0
    pilot = json.loads(Path(t3_pilot.DEFAULT_OUT).read_text(encoding="utf-8"))
    assert t3_eval.check_gate(pilot, T3DynamicACO()) == 900.0


def _entry(ci, p, better, worse, direction="lower_is_better"):
    return {"direction": direction, "family_cluster_bootstrap_ci95": ci, "family_sign_test_p_two_sided": p,
            "family_t3_better": better, "family_t3_worse": worse}


def test_preregistered_decision_rule():
    v = t3_eval.verdict
    assert v(_entry([-2.0, -0.5], 0.01, 12, 1)) == "better"
    assert v(_entry([0.5, 2.0], 0.01, 1, 12)) == "worse"
    assert v(_entry([0.01, 0.2], 0.001, 14, 0, "higher_is_better")) == "better"
    assert v(_entry([-2.0, -0.5], 0.06, 10, 3)) == "no_evidence_of_difference"     # sign test fails
    assert v(_entry([-2.0, 0.0], 0.01, 12, 1)) == "no_evidence_of_difference"      # CI touches 0
    assert v(_entry([-2.0, -0.5], 0.01, 1, 12)) == "no_evidence_of_difference"     # directions disagree
    assert v(_entry(None, 0.01, 12, 1)) == "no_evidence_of_difference"


def _assert_same(a, b, path="$"):
    """Equal structure; ints, strings, bools and None exact; floats within 1e-12 relative."""
    if isinstance(a, float) or isinstance(b, float):
        assert isinstance(a, (int, float)) and isinstance(b, (int, float)), path
        assert a == pytest.approx(b, rel=1e-12, abs=1e-15), path
    elif isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), path
        for k in a:
            _assert_same(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_same(x, y, f"{path}[{i}]")
    else:
        assert a == b and type(a) is type(b), path


def test_committed_dev_report_integrity_if_present():
    """Reads the committed T3 DEV report only; nothing is rerun here."""
    path = Path(t3_eval.DEFAULT_OUT)
    if not path.exists():
        pytest.skip("T3 DEV comparison not run yet")
    rep = json.loads(path.read_text(encoding="utf-8"))
    ev = t3_eval.ev
    t3 = T3DynamicACO()
    assert rep["eval_version"] == t3_eval.EVAL_VERSION and rep["complete"] is True
    idx = json.loads((ev.REPO / "data/splits/p1-02-v1/index.json").read_text(encoding="utf-8"))
    assert rep["dev"] == {"plan_id": "p1-02-v1", "split": "dev", "manifest_hash": idx["splits"]["dev"]["manifest_hash"],
                          "index_hash": idx["index_hash"], "scenarios": 48}
    rows = rep["rows"]
    assert rep["scenarios_evaluated"] == len(rows) == 48 and [r["position"] for r in rows] == list(range(48))
    assert rep["families_evaluated"] == len({r["family_id"] for r in rows}) == 16
    assert all(n == 3 for n in Counter(r["family_id"] for r in rows).values())
    assert rep["environment"]["src_dirty"] is False
    assert rep["policies"]["T3"] == t3.identity.to_dict() and rep["teacher_seed"] == t3.seed == 20261123
    assert rep["policies"]["T3"]["config_hash"] == FROZEN_T3_HASH
    pilot = json.loads(Path(t3_pilot.DEFAULT_OUT).read_text(encoding="utf-8"))
    assert rep["pilot_gate"] == pilot["gate"] and rep["wall_limit_s"] == t3_eval.check_gate(pilot, t3) == 900.0
    det = [{k: (ev.deterministic_view(v) if k in t3_eval.POLICY_ORDER else v) for k, v in r.items()} for r in rows]
    assert digest(det) == rep["deterministic_digest"]
    p31 = t3_eval.t1_eval.paired(rows, "T3", "T1", t3_eval.BOOTSTRAP_SEEDS["T1"], t3_eval.PAIRED_METRICS)
    p30 = t3_eval.t1_eval.paired(rows, "T3", "T0", t3_eval.BOOTSTRAP_SEEDS["T0"], t3_eval.PAIRED_METRICS)
    # Floats within 1e-12 relative: Python >= 3.12 sums floats with compensation, so a bootstrap mean can
    # differ in the last ulp between interpreter versions. Counts, p-values and verdicts must match exactly.
    _assert_same(rep["paired_t3_minus_t1"], p31)
    _assert_same(rep["paired_t3_minus_t0"], p30)
    assert rep["decision_rule"] == {"t3_minus_t1": t3_eval.verdicts(p31), "t3_minus_t0": t3_eval.verdicts(p30)}
    assert rep["t3_worse_than_t1"] == t3_eval.worse(rows, "T3", "T1")
    assert rep["t3_worse_than_t0"] == t3_eval.worse(rows, "T3", "T0")
    _assert_same(rep["strata"], t3_eval.strata(rows))
    # Frozen T0 and T1 reproduce their P1-03 / P1-04 episodes exactly.
    base = ev.REPO / "data/baselines"
    t0 = {r["scenario_hash"]: r["T0"]["final_transition_hash"]
          for r in json.loads((base / "p1-03-t0-dev-v1.json").read_text(encoding="utf-8"))["rows"]}
    t1 = {r["scenario_hash"]: r["T1"]["final_transition_hash"]
          for r in json.loads((base / "p1-04-t1-dev-v1.json").read_text(encoding="utf-8"))["rows"]}
    for r in rows:
        assert r["T0"]["final_transition_hash"] == t0[r["scenario_hash"]]
        assert r["T1"]["final_transition_hash"] == t1[r["scenario_hash"]]
    for name in t3_eval.POLICY_ORDER:
        agg = rep["aggregates"][name]
        assert agg["scenarios"] == 48 and agg["failures"] == 0
        assert agg["casualties"] == sum(r[name]["casualties"] for r in rows)
        for r in rows:
            x = r[name]
            assert x["failure"] is None and x["termination_reason"] in ("all_resolved", "max_steps")
            assert x["evacuated"] + x["casualties"] + x["unresolved"] == x["people"] > 0
            assert sum(x["casualty_context"].values()) == x["casualties"]
            assert x["decisions"] == x["steps"]
    for r in rows:
        d = r["T3"]["t3"]
        assert d["decisions"] == r["T3"]["decisions"] == sum(d["status"].values())
        assert set(d["status"]) <= {"complete", "budget_limited"}
        assert sum(d["limit_hit"].values()) == d["status"].get("budget_limited", 0)
