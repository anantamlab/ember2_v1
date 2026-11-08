"""T4 v1.2.0: exact vacate infeasibility proof on top of the frozen v1.1.0 planner (docs/P1_T4_V12_NODE_BUDGET.md).

The tests check (1) the identity and the exact config change, (2) that every proved failure is a failure of
the v1.1.0 code with the state restored, (3) that decisions equal v1.1.0 whenever v1.1.0 hits no limit,
(4) the v1.1.0 invariants I1, I2, I4, I5 for v1.2.0, and (5) the pilot module (cells, disjointness, label).
"""
from __future__ import annotations

import copy
import json
import random
from collections import Counter

import pytest

from ember_v1.contract import validate_observation
from ember_v1.sim import Simulator
from ember_v1.sim.bench import room
from ember_v1.sim.movement import resolve_moves
from ember_v1.teacher import t4_v2_pilot as v2p
from ember_v1.teacher import t4_v12_pilot as v12p
from ember_v1.teacher.t4 import ABLATIONS, PRIORITY_EFFECTS
from ember_v1.teacher.t4_v11 import ABLATIONS_V11, TransactionalPlanner
from ember_v1.teacher.t4_v12 import ABLATIONS_V12, PLANNER_VERSION_V12, ProofPlanner, T4RiskPriorityV12
from gen_helpers import random_scenario
from test_teacher_t4 import FROZEN_HASHES
from test_teacher_t4_v11 import FIXTURES, V11_HASHES, _crowd, _graph_consistent, _probe, first_pass_blocked, obs_of

V12_HASHES = {
    "A0": "sha256:d95ac217d8dc61206c15fa508c6534ea89c2a259b5f6d964cb82ed69bcb047be",
    "A1": "sha256:f177bc8fe197dde7c45d1586b4f3e988baa4dc334d447519af7c8b721b042638",
    "A2": "sha256:79bd3322b9cf7ccd25928f3a70502089732496af5427f88f00ecd2ff6ba12504",
}


class Checked(_probe(ProofPlanner)):
    """Runs the frozen v1.1.0 vacate on a copy of the state for every proved failure and checks the claim."""
    proofs = 0

    def _state(self):
        return (dict(self.vres), dict(self.eres), {k: list(v) for k, v in self.plans.items()}, dict(self.origin),
                dict(self.pushed_by), {k: set(v) for k, v in self.forbid.items() if v},
                {k: list(v) for k, v in self.keys.items() if v})

    def plan_tx(self, pid, requester, root, depth):
        if requester is not None:
            self._limits()
            if self.limit_hit is None and depth <= self.cfg["push_depth_limit"] and not self._may_vacate(pid):
                snap = self._state()
                keep = (self.budget_left, self.limit_hit, self.tx.copy(), copy.deepcopy(self.info))
                self.budget_left = 10 ** 9                  # the claim holds for any node budget
                ok = TransactionalPlanner.plan_tx(self, pid, requester, root, depth)
                assert not ok, f"proved failure of {pid} succeeded in v1.1.0"
                assert self._state() == snap, f"v1.1.0 failure of {pid} did not restore the state"
                self.budget_left, self.limit_hit, self.tx = keep[:3]
                for p, d in keep[3].items():               # in place: outer frames hold these dicts
                    self.info[p].clear()
                    self.info[p].update(d)
                type(self).proofs += 1
        return super().plan_tx(pid, requester, root, depth)


def run12(obs, variant="A2", cls=Checked):
    pol = ABLATIONS_V12[variant]()
    tp = cls(pol.identity.config, validate_observation(obs), pol.scheme)
    acts, dg = tp.run()
    return acts, dg, tp


# ----- identity and scope -----------------------------------------------------------------------------------
def test_v12_identities_hashes_and_the_exact_config_change():
    ids = {k: cls().identity for k, cls in ABLATIONS_V12.items()}
    assert {k: v.config_hash for k, v in ids.items()} == V12_HASHES
    assert not set(V12_HASHES.values()) & (set(V11_HASHES.values()) | set(FROZEN_HASHES.values()))
    assert {k: cls().identity.config_hash for k, cls in ABLATIONS_V11.items()} == V11_HASHES     # v1.1 unchanged
    assert {k: cls().identity.config_hash for k, cls in ABLATIONS.items()} == FROZEN_HASHES       # v1.0 unchanged
    for k, i in ids.items():
        old = ABLATIONS_V11[k]().identity
        assert (i.name, i.policy, i.version) == (old.name, old.policy, "1.2.0")
        diff = {key for key in set(i.config) | set(old.config) if i.config.get(key) != old.config.get(key)}
        assert diff == {"planner", "vacate_infeasibility_proof"}
        assert i.config["planner"] == PLANNER_VERSION_V12


def test_v12_rejects_changes_to_descriptive_keys_and_numbers_out_of_range():
    with pytest.raises(ValueError, match="not configurable"):
        T4RiskPriorityV12({"vacate_infeasibility_proof": "off"})
    with pytest.raises(ValueError, match="push_depth_limit"):
        T4RiskPriorityV12({"push_depth_limit": 10 ** 6})


# ----- exactness ----------------------------------------------------------------------------------------------
def _limited_room():
    sim = Simulator(room(14, 120, 2, "stochastic"))
    for _ in range(4):
        sim.step(ABLATIONS_V11["A2"]().decide(sim.observe()).actions)
    return sim.observe()


def test_every_proved_failure_is_a_v11_failure_with_the_state_restored():
    Checked.proofs = 0
    rnd = random.Random(20261108)
    for k in range(40):
        s = _crowd(random_scenario(random.Random(rnd.getrandbits(32)), h=rnd.randint(4, 8), w=rnd.randint(4, 9)), rnd)
        for v in ("A0", "A2"):
            sim = Simulator(s)
            for _ in range(6):
                if sim.terminated:
                    break
                acts, _, _ = run12(sim.observe(), v)
                sim.step(acts)
    run12(_limited_room(), "A2")
    assert Checked.proofs > 100


def test_decisions_equal_v11_whenever_v11_hits_no_limit():
    rnd = random.Random(7)
    compared = Counter()
    scen = [_crowd(random_scenario(random.Random(rnd.getrandbits(32)), h=rnd.randint(4, 8), w=rnd.randint(4, 9)), rnd)
            for _ in range(25)] + [room(12, 50, 1, "deterministic"), room(16, 90, 2, "stochastic")]
    for s in scen:
        for v in ABLATIONS_V12:
            sim = Simulator(s)
            for _ in range(8):
                if sim.terminated:
                    break
                obs = sim.observe()
                d11, d12 = ABLATIONS_V11[v]().decide(obs), ABLATIONS_V12[v]().decide(obs)
                if d11.diagnostics["budget"]["limit_hit"] is None:
                    assert d12.actions == d11.actions
                    p11, p12 = d11.diagnostics["people"], d12.diagnostics["people"]
                    assert {p: x["plan_status"] for p, x in p12.items()} == {p: x["plan_status"] for p, x in p11.items()}
                    assert d12.diagnostics["expansions"] <= d11.diagnostics["expansions"]
                    compared[v] += 1
                sim.step(d11.actions)
    assert min(compared.values()) > 50


def test_proof_removes_the_node_limit_hit_of_a_crowded_room():
    obs = _limited_room()
    d11, d12 = T4RiskPriorityV11Ref().decide(obs), T4RiskPriorityV12().decide(obs)
    assert d11.diagnostics["budget"]["limit_hit"] == "node_budget_total"
    assert d12.diagnostics["budget"]["limit_hit"] is None
    assert d12.diagnostics["transaction"]["vacate_failed_infeasible"] > 0
    assert len(d12.diagnostics["fallback_people"]) < len(d11.diagnostics["fallback_people"])
    assert d12.diagnostics["expansions"] <= d12.diagnostics["budget"]["node_budget_total"]


def T4RiskPriorityV11Ref():
    return ABLATIONS_V11["A2"]()


# ----- v1.1.0 invariants hold for v1.2.0 ------------------------------------------------------------------------
@pytest.mark.parametrize("fixture", sorted(FIXTURES))
def test_invariants_I2_I4_on_the_defect_fixtures(fixture):
    _, dg, tp = run12(obs_of(FIXTURES[fixture]), "A0")
    assert tp.p_overwrites == [] and first_pass_blocked(tp) == [] and dg["converted_to_stay"] == []


def test_invariant_I1_I5_on_the_d2_fixture():
    _, dg, tp = run12(obs_of(FIXTURES["D2_genuine_a0_inversion"]), "A0")
    assert tp.precedence_audit == [] and not [p for p, x in dg["people"].items() if x["precedence_inversion"]]


def _check(obs, v):
    acts, dg, tp = run12(obs, v)
    active = {p["id"] for p in obs["people"] if p["status"] == "active"}
    assert set(acts) == active == set(dg["people"])
    assert tp.p_overwrites == [] and tp.precedence_audit == []
    assert first_pass_blocked(tp) == [] and dg["converted_to_stay"] == []
    _graph_consistent(tp)
    pg = tp.pg
    res = resolve_moves(height=pg.height, width=pg.width, walls=pg.walls, burning=pg.burning,
                        enter_fire=pg.enter_fire, positions=dict(pg.active), actions=acts)
    assert {p: o for p, (o, _) in res.items()} == {p: x["predicted_outcome"] for p, x in dg["people"].items()}
    assert all(x["priority_effect"] in PRIORITY_EFFECTS for x in dg["people"].values())
    assert dg["priority"]["precedence_inversions"] == 0
    if v == "A0":
        for pid, x in dg["people"].items():
            if x["yield_over_id_rule"]:
                assert x["yield_kind"] == "chain" and x["yield_origin"] < pid
    return acts


def test_randomized_small_grid_properties_v12():
    rnd = random.Random(20261109)
    decisions = Counter()
    for _ in range(60):
        s = _crowd(random_scenario(random.Random(rnd.getrandbits(32)), h=rnd.randint(3, 7), w=rnd.randint(3, 8)), rnd)
        for v in ABLATIONS_V12:
            sim = Simulator(s)
            while not sim.terminated:
                acts = _check(sim.observe(), v)
                decisions[v] += 1
                rec = sim.step(acts)
                assert all(m["outcome"] in ("moved", "stayed") for m in rec["movements"].values())
    assert min(decisions.values()) > 100


def test_dense_room_closed_loop_v12():
    for side, n, seed, m in ((14, 120, 2, "stochastic"), (20, 150, 3, "stochastic")):
        for v in ABLATIONS_V12:
            sim = Simulator(room(side, n, seed, m))
            for _ in range(8):
                if sim.terminated:
                    break
                sim.step(_check(sim.observe(), v))


@pytest.mark.parametrize("cfg,status,limit", [({"cpu_budget_ms": 0.0}, "cpu_budget_fallback", "cpu_budget_ms"),
                                              ({"node_budget_total": 0}, "node_budget_fallback", "node_budget_total")])
def test_budget_exhaustion_gives_valid_labeled_fallback_without_pushes(cfg, status, limit):
    sim = Simulator(room(24, 30, 3, "stochastic"))
    d = T4RiskPriorityV12(cfg).decide(sim.observe())
    dg = d.diagnostics
    assert dg["budget"]["limit_hit"] == limit and set(d.actions) == set(sim.active_ids())
    routed = [p for p in dg["people"].values() if p["route_found"]]
    assert routed and all(p["plan_status"] == status for p in routed)
    assert dg["transaction"].get("vacate_success", 0) == 0 and dg["converted_to_stay"] == []


def test_push_depth_limit_zero_still_consistent():
    obs = validate_observation(Simulator(room(16, 60, 7, "deterministic")).observe())
    pol = T4RiskPriorityV12({"push_depth_limit": 0})
    tp = Checked(pol.identity.config, obs, pol.scheme)
    _, dg = tp.run()
    assert dg["transaction"].get("vacate_success", 0) == 0
    assert first_pass_blocked(tp) == [] and tp.p_overwrites == []
    _graph_consistent(tp)


def test_deterministic_and_order_invariant_v12():
    for k in range(10):
        obs = Simulator(random_scenario(random.Random(1300 + k))).observe()
        for cls in ABLATIONS_V12.values():
            a, b = cls().decide(obs), cls().decide(obs)
            assert a.deterministic_dict() == b.deterministic_dict()
            assert cls().decide({**obs, "people": list(reversed(obs["people"]))}).actions == a.actions


# ----- pilot module -------------------------------------------------------------------------------------------
def test_v2_scenarios_are_unchanged_by_the_parametrization():
    rep = json.loads((v2p.ev.REPO / "data/baselines/p1-08-t4-v2-pilot.json").read_text(encoding="utf-8"))
    got = [{"cell": v2p._cell_json(c), "content_hash": v2p.content_hash(s)} for c, s in v2p.select_cells()[0]]
    assert got == rep["cell_content_hashes"]


def test_v12_pilot_cells_are_disjoint_from_v1_v2_and_the_exploratory_set():
    kept, rejected, excl = v12p.select_cells()
    assert len(kept) == len(v2p.pilot_cells_v2()) == 74 and rejected == []      # fixed before the run
    assert excl["v1_pilot_scenarios_checked"] > 0 and excl["v2_pilot_scenarios_checked"] == 74
    ex_kept, _, _ = v12p.select_cells(v12p.EXPLORATORY_NAMESPACE)
    pilot = {v2p.content_hash(s) for _, s in kept}
    explore = {v2p.content_hash(s) for _, s in ex_kept}
    v2c = {v2p.content_hash(v2p.pilot_scenario_v2(c)) for c in v2p.pilot_cells_v2()}
    assert not pilot & explore and not pilot & v2c
    assert all(s.scenario_id.startswith("t4-pilot3-") for _, s in kept)
    assert {s.seed for _, s in kept}.isdisjoint({s.seed for _, s in ex_kept})


def test_decision_label_is_ready_only_when_no_check_fails():
    assert v12p.decide({"failed": [], "decision": "T4_V2_FULL_DEV_FEASIBLE"})["decision"] == v12p.READY
    out = v12p.decide({"failed": ["G4_planner_health"], "decision": "T4_V2_BLOCKED"})
    assert out["decision"] == v12p.BLOCKED and out["v2_label"] == "T4_V2_BLOCKED"
    assert "SELECTED" not in v12p.READY.replace("SELECTION", "")


def test_pilot_entry_point_refuses_to_overwrite(tmp_path):
    out = tmp_path / "r.json"
    out.write_text("{}", encoding="utf-8")
    for extra in ([], ["--exploratory"]):
        with pytest.raises(SystemExit, match="exists"):
            v12p.main(["--out", str(out), *extra])


def test_committed_confirmation_report_is_reproducible_and_bound_to_the_freeze_commit():
    rep = json.loads((v2p.ev.REPO / "data/baselines/p1-08-t4-v12-pilot.json").read_text(encoding="utf-8"))
    assert rep["namespace"] == v12p.NAMESPACE and rep["pilot_version"] == v12p.PILOT_VERSION
    assert rep["evidence_class"] == "prospective_confirmation"
    assert {k: rep["teachers"][k]["config_hash"] for k in ABLATIONS_V12} == V12_HASHES
    assert rep["cells_run"] == 74 and rep["rejected_cells"] == [] and rep["errors"] == []
    assert rep["environment"]["git_sha"] == "b7ff040f8abe065ced5ed1331c0cd6854bd878a1"
    assert rep["environment"]["src_dirty"] is False
    assert rep["cell_content_hashes"] == [{"cell": v2p._cell_json(c), "content_hash": v2p.content_hash(s)}
                                          for c, s in v12p.select_cells()[0]]
    g = v2p.gate_v2(dict(ABLATIONS_V12["A2"]().identity.config), rep["pilot"], Counter(rep["neutral_paired_states"]),
                    rep["pilot_complete"] and rep["dev_profile_complete"], rep["resources"]["peak_rss_mb"],
                    rep["estimate"])
    assert v12p.decide(g) == rep["gate"]
    assert rep["gate"]["failed"] == [] and rep["gate"]["decision"] == v12p.READY
