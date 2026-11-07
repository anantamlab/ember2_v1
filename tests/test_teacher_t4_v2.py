"""T4 v2 pilot and DEV harness (overnight P3): cells, exclusion, mechanics, gate logic and run-once guards.

These tests never run a v1.1 policy on a v2 pilot cell; they use random small scenarios and synthetic gate input.
"""
from __future__ import annotations

import random
from collections import Counter

import pytest

from ember_v1.sim import Simulator
from ember_v1.teacher import t4_v2_eval as v2e
from ember_v1.teacher import t4_v2_pilot as v2p
from ember_v1.teacher.t4_eval import GateError
from ember_v1.teacher.t4_pilot import pilot_cells, pilot_scenario
from ember_v1.teacher.t4_v11 import ABLATIONS_V11
from gen_helpers import random_scenario


def test_v2_cells_are_fixed_and_disjoint_from_the_v1_pilot():
    cells = v2p.pilot_cells_v2()
    assert len(cells) == 74
    assert len({repr(sorted(v2p._cell_json(c).items())) for c in cells}) == 74
    kept, rejected, excl = v2p.select_cells()
    assert excl == {"v1_scenarios_checked": len(pilot_cells()), "inspected_v1_layouts": 2}
    assert len(kept) + len(rejected) == 74
    v1 = [pilot_scenario(c) for c in pilot_cells()]
    v1_hash = {v2p.content_hash(s) for s in v1}
    v1_seed = {s.seed for s in v1}
    for _, s in kept:
        assert s.scenario_id.startswith("t4-pilot2-")
        assert v2p.content_hash(s) not in v1_hash and s.seed not in v1_seed
    assert len({v2p.content_hash(s) for _, s in kept}) == len(kept)


def test_scenarios_are_deterministic():
    c = v2p.pilot_cells_v2()[7]
    assert v2p.pilot_scenario_v2(c).to_dict() == v2p.pilot_scenario_v2(c).to_dict()


def _cell(mode="full_episode"):
    return {"kind": "test", "topology": "random", "grid": (0, 0), "n": 0, "model": "x", "exits": 0, "mode": mode}


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_run_cell_and_neutral_states_on_random_scenarios(seed):
    s = random_scenario(random.Random(seed))
    for v in v2p.VARIANTS:
        rows, ep, ok = v2p.run_cell(_cell(), s, v, float("inf"))
        assert ok and rows and ep["variant"] == v
        assert ep["evacuated"] + ep["casualties"] + ep["unresolved"] == len(s.people)
        for r in rows:
            assert r["blocked_moves"] == r["prediction_mismatches"] == r["verification_mismatches"] == 0
            assert r["precedence_audit_violations"] == r["precedence_inversions"] == 0
            if v == "A0":
                assert r["a0_id_inversions"] == r["a0_unexplained_over_id_yields"] == 0
    tot, cases, n, ok = v2p.run_neutral(_cell(), s, float("inf"))
    assert ok and n >= 1
    assert tot["A0/precedence_audit_violations"] == tot["A2/precedence_audit_violations"] == 0
    for c in cases:
        assert c["progress_a2"] > c["progress_a0"] and c["link"] in ("yield", "exit_reassignment")


def test_prefix_mode_stops_at_the_prefix():
    s = random_scenario(random.Random(5))
    rows, ep, ok = v2p.run_cell(_cell("prefix"), s, "A2", float("inf"))
    assert ok and ep == {} and len(rows) <= v2p.PREFIX_DECISIONS


def test_deadline_marks_the_cell_incomplete():
    s = random_scenario(random.Random(6))
    _, _, ok = v2p.run_cell(_cell(), s, "A0", 0.0)
    assert not ok
    assert v2p.run_neutral(_cell(), s, 0.0)[3] is False


def test_paired_state_progress_is_static_exit_distance():
    s = random_scenario(random.Random(21))
    obs = Simulator(s).observe()
    a0, a2 = ABLATIONS_V11["A0"]().decide(obs), ABLATIONS_V11["A2"]().decide(obs)
    from ember_v1.teacher.grid import PublicGrid
    from ember_v1.teacher.t4_risk import assess
    strata = {p: r["stratum"] for p, r in
              assess(PublicGrid.from_observation(obs), obs, dict(ABLATIONS_V11["A2"]().identity.config)).people.items()}
    out, _ = v2p.paired_state(obs, a0, a0, strata)
    assert out["at_risk/progress_a0"] == out["at_risk/progress_a2"] and out["g5c/yield"] == 0
    out2, _ = v2p.paired_state(obs, a0, a2, strata)
    for k in ("at_risk", "lower_risk"):
        assert out2[f"{k}/persons"] == out[f"{k}/persons"]


# ----- gate -----------------------------------------------------------------------------------------------
def _row(v, **kw):
    r = {"variant": v, "cpu_ms": 10.0, "predicted_blocked": 0, "blocked_moves": 0, "prediction_mismatches": 0,
         "verification_mismatches": 0, "route_people": 10, "fallback_people": 0, "limit_hit": None,
         "precedence_audit_violations": 0, "precedence_inversions": 0, "a0_id_inversions": 0,
         "a0_unexplained_over_id_yields": 0, "yields_over_id_rule": 2, "chain_yields_over_id_rule": 2}
    r.update(kw)
    return r


def _gate(pilot=None, neutral=None, complete=True, peak=100.0, upper=100.0, cap=0.0):
    cfg = dict(ABLATIONS_V11["A2"]().identity.config)
    pilot = pilot if pilot is not None else [_row(v) for v in v2p.VARIANTS]
    n = Counter({"g5c/yield": 1, "at_risk/persons": 5, "at_risk/progress_a0": 3, "at_risk/progress_a2": 4,
                 "neutral_decisions": 3})
    n.update(neutral or {})
    est = {"upper": {"comparison_cpu_s_total": upper,
                     "variants": {v: {"predicted_cap_fraction": cap} for v in v2p.VARIANTS}}}
    return v2p.gate_v2(cfg, pilot, n, complete, peak, est)


def test_gate_passes_only_when_every_check_holds():
    g = _gate()
    assert g["failed"] == [] and g["decision"] == "T4_V2_FULL_DEV_FEASIBLE"


@pytest.mark.parametrize("kw,failed", [
    ({"complete": False}, "G1_bounded"),
    ({"pilot": [_row(v, blocked_moves=1) for v in v2p.VARIANTS]}, "G1_bounded"),
    ({"pilot": [_row(v, cpu_ms=1e9) for v in v2p.VARIANTS]}, "G1_bounded"),
    ({"peak": 5000.0}, "G1_bounded"),
    ({"upper": 1e9}, "G2_dev_cpu"),
    ({"cap": 0.5}, "G3_cap_hits"),
    ({"pilot": [_row(v, fallback_people=5) for v in v2p.VARIANTS]}, "G4_planner_health"),
    ({"pilot": [_row(v, precedence_audit_violations=1) for v in v2p.VARIANTS]}, "G5a_precedence"),
    ({"neutral": {"A2/precedence_inversions": 1}}, "G5a_precedence"),
    ({"pilot": [_row(v, a0_id_inversions=1) for v in v2p.VARIANTS]}, "G5b_a0_integrity"),
    ({"pilot": [_row(v, a0_unexplained_over_id_yields=1) for v in v2p.VARIANTS]}, "G5b_a0_integrity"),
    ({"pilot": [_row(v) for v in ("A1", "A2")]}, "G5b_a0_integrity"),
    ({"neutral": {"g5c/yield": -1, "g5c/exit_reassignment": 9}}, "G5c_priority_effect"),
    ({"neutral": {"at_risk/progress_a2": -2}}, "G5c_priority_effect"),
])
def test_each_failed_check_blocks(kw, failed):
    g = _gate(**kw)
    assert failed in g["failed"] and g["decision"] == "T4_V2_BLOCKED"


# ----- DEV harness guards -----------------------------------------------------------------------------------
def _pilot(decision="T4_V2_FULL_DEV_FEASIBLE", hashes=None):
    t = {k: {"config_hash": cls().identity.config_hash} for k, cls in ABLATIONS_V11.items()}
    if hashes:
        t.update(hashes)
    return {"gate": {"decision": decision}, "teachers": t, "estimate": {"upper": {"comparison_cpu_s_total": 10.0}}}


def test_dev_gate_refuses_blocked_or_mismatched_pilots():
    assert v2e.check_gate_v2(_pilot()) == v2e.wall_limit(10.0)
    for bad in (_pilot("T4_V2_BLOCKED"), _pilot("FULL_DEV_FEASIBLE"),
                _pilot(hashes={"A2": {"config_hash": "sha256:" + "0" * 64}})):
        with pytest.raises(GateError):
            v2e.check_gate_v2(bad)


def test_dev_policies_are_v11_and_t1_reference_covers_dev():
    p = v2e.policies()
    assert [p[k].identity.version for k in ("A0", "A1", "A2")] == ["1.1.0"] * 3
    assert len(v2e.t1_reference()) == 48


def test_both_entry_points_refuse_to_overwrite(tmp_path):
    out = tmp_path / "r.json"
    out.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="runs once"):
        v2p.main(["--out", str(out)])
    with pytest.raises(SystemExit, match="runs once"):
        v2e.main(["--out", str(out)])
