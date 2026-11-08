"""T4 v1.0.0 repair defects (overnight P1) and the v1.1.0 transactional planner (overnight P2).

Defect fixtures were found by a seeded search over small ASCII scenarios and minimized
(docs/overnight/T4_V1_FAILURE_ANALYSIS.md section 3). Each fixture has:
  * a frozen-v1 reproduction test (passes: it documents the defect, v1 behavior unchanged);
  * invariant tests parametrized by planner version. The v1.0.0 case is `xfail(strict=True)`:
    it MUST fail, so the suite proves the defect without breaking the frozen v1 contract.
"""
from __future__ import annotations

import ast
import random
from collections import Counter
from pathlib import Path

import pytest

from ember_v1.contract import validate_observation
from ember_v1.sim import Simulator
from ember_v1.sim.bench import room
from ember_v1.sim.movement import resolve_moves
from ember_v1.teacher.t4 import ABLATIONS, PRIORITY_EFFECTS, YIELD_EFFECTS, _PriorityPlanner
from ember_v1.teacher.t4_v11 import (ABLATIONS_V11, ReservationConflict, T4NoPriorityV11, T4RiskPriorityV11,
                                     TransactionalPlanner)
from ember_v1.teacher.t4_v1_diag import TracedPlanner, classify
from gen_helpers import random_scenario
from test_teacher_t0 import ascii_scenario
from test_teacher_t4 import FROZEN_HASHES, STEP_ASIDE

ROOT = Path(__file__).resolve().parents[1]
V11_HASHES = {
    "A0": "sha256:97e134923cd63d55b79246c12e6e10b3800c4706bb2354e40fa8cc1588f675db",
    "A1": "sha256:d1c5acb82cfb693a0d1e6898c3af5293b7f405ada7c94cfe794b405cb2b7f326",
    "A2": "sha256:67a3517de6fd109d128c0cf31836867fa0a1ec30cb09844e7a1541acae237ad4",
}
# Minimized defect fixtures. Fire interval 1, exits not flammable, enter_fire "blocked", A0 (ID order).
FIXTURES = {
    # D1: Pb claims Pe's cell; Pe cannot vacate, stays, and its stay silently overwrites Pb's claim.
    "D1_stay_overwrites_claim": [".0.e.a*", "c.#b...", ".......", ".......", "......."],
    # D1b: Pe has no admissible move; the least-hazard fallback ignores Pd's exit claim, overwrites it,
    # and the same repair repeats for all 6 rounds before the move is converted to stay.
    "D1b_fallback_overwrite_repeats": ["e*....", "0.....", "d.....", "......", "......"],
    # D2: Pe is re-planned while LATER-ranked Pf still holds Pe's cell; Pe steps aside for Pf (A0 inversion).
    "D2_genuine_a0_inversion": ["....da", ".....0", ".....h", ".*..ce", ".....f"],
    # D3: a ring of vacates around a pillar; one refusal unravels one link per repair round.
    "D3_dependency_chain": ["dhc#", ".#eg", ".b#a", "*0jf"],
}
V10 = pytest.param("1.0.0", marks=pytest.mark.xfail(strict=True, reason="v1.0.0 repair defect (P1)"))
VERSIONS = [V10, "1.1.0"]


def obs_of(rows) -> dict:
    s = ascii_scenario(rows, interval=1, flammable=False, enter_fire="blocked", max_steps=30)
    return validate_observation(Simulator(s).observe())


def _probe(base):
    class Probe(base):
        def __init__(self, *a, **kw):
            self.p_overwrites, self.p_resolves = [], []
            super().__init__(*a, **kw)

        def reserve(self, pid, path, hold):
            W1, a = self.W + 1, self.g.area
            for tau in range(1, len(path)):
                c, n = path[tau - 1], path[tau]
                for table, k in ((self.vres, n * W1 + tau), (self.eres, (c * a + n) * W1 + tau if n != c else None)):
                    if k is not None and table.get(k) not in (None, pid):
                        self.p_overwrites.append((pid, table[k], tau))
            super().reserve(pid, path, hold)

        def resolve(self, actions):
            out = super().resolve(actions)
            self.p_resolves.append((dict(actions), out))
            return out
    return Probe


PROBES = {"1.0.0": _probe(TracedPlanner), "1.1.0": _probe(TransactionalPlanner)}


def run(version: str, obs: dict, variant: str = "A0"):
    pol = (ABLATIONS if version == "1.0.0" else ABLATIONS_V11)[variant]()
    tp = PROBES[version](pol.identity.config, obs, pol.scheme)
    acts, dg = tp.run()
    return acts, dg, tp


def first_pass_blocked(tp) -> list[str]:
    acts, res = tp.p_resolves[0]
    return sorted(p for p, a in acts.items() if a != "stay" and res[p][0] != "moved")


def genuine_inversions(version: str, tp, dg) -> list:
    if version == "1.0.0":
        return [e for e in classify(tp, dg) if e["class"] == "genuine_inversion"]
    return [p for p, x in dg["people"].items() if x["precedence_inversion"]]


# ----- P1: frozen v1.0.0 reproduction of each defect (must pass) --------------------------------------------
def test_v1_defect_fixtures_reproduce_and_tracing_is_passive():
    for name, rows in FIXTURES.items():
        obs = obs_of(rows)
        acts, dg, tp = run("1.0.0", obs)
        frozen = ABLATIONS["A0"]().decide(obs)
        assert acts == frozen.actions and dg == frozen.diagnostics, name      # tracing changes nothing
    _, dg, tp = run("1.0.0", obs_of(FIXTURES["D1_stay_overwrites_claim"]))
    assert [(o["new_owner"], o["lost_owner"], o["tau"], o["stay"]) for o in tp.t_overwrites] == [("Pe", "Pb", 1, True)]
    assert first_pass_blocked(tp) == ["Pb"] and dg["repair_rounds"] == 1
    _, dg, tp = run("1.0.0", obs_of(FIXTURES["D1b_fallback_overwrite_repeats"]))
    assert dg["repair_rounds"] == tp.cfg["repair_rounds"] == 6 and dg["converted_to_stay"] == ["Pe"]
    assert [r[1]["Pe"][0] for r in tp.p_resolves[:-1]] == ["blocked_conflict"] * 7    # same failed repair x7
    _, dg, tp = run("1.0.0", obs_of(FIXTURES["D2_genuine_a0_inversion"]))
    ev = classify(tp, dg)
    assert [(e["yielder"], e["beneficiary"], e["class"]) for e in ev] == [("Pe", "Pf", "genuine_inversion")]
    assert dg["people"]["Pe"]["priority_effect"] == "yielded_step_aside" and dg["people"]["Pe"]["repairs"] == 1
    _, dg, tp = run("1.0.0", obs_of(FIXTURES["D3_dependency_chain"]))
    assert len(tp.t_overwrites) == 7 and dg["repair_rounds"] == 6
    assert dg["converted_to_stay"] == ["Pd"] and dg["people"]["Pd"]["repairs"] == 0
    assert dg["predicted_outcomes"] == {"moved": 1, "stayed": 8}


# ----- P1/P2: intended invariants (v1.0.0 must fail, v1.1.0 must pass) -----------------------------------------
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("fixture", sorted(FIXTURES))
def test_invariant_I2_no_reservation_overwrite(fixture, version):
    _, _, tp = run(version, obs_of(FIXTURES[fixture]))
    assert tp.p_overwrites == []


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("fixture", sorted(FIXTURES))
def test_invariant_I4_planned_joint_action_agrees_with_the_resolver(fixture, version):
    _, dg, tp = run(version, obs_of(FIXTURES[fixture]))
    assert first_pass_blocked(tp) == [] and dg["converted_to_stay"] == [] and dg["repair_rounds"] == 0


@pytest.mark.parametrize("version", VERSIONS)
def test_invariant_I5_a0_has_no_genuine_precedence_inversion(version):
    _, dg, tp = run(version, obs_of(FIXTURES["D2_genuine_a0_inversion"]))
    assert genuine_inversions(version, tp, dg) == []


@pytest.mark.parametrize("version", VERSIONS)
def test_invariant_I1_later_ranked_reservations_never_constrain_a_replanner(version):
    _, dg, tp = run(version, obs_of(FIXTURES["D2_genuine_a0_inversion"]))
    if version == "1.0.0":
        assert not [c for c in tp.t_calls if c["next"] != c["counterfactual_next"]]
    else:
        assert tp.precedence_audit == []


# ----- P2: v1.1.0 identity and scope of the change ----------------------------------------------------------------
def test_v11_identities_hashes_and_the_exact_config_change():
    ids = {k: cls().identity for k, cls in ABLATIONS_V11.items()}
    assert {k: v.config_hash for k, v in ids.items()} == V11_HASHES
    assert not set(V11_HASHES.values()) & set(FROZEN_HASHES.values())
    assert {k: cls().identity.config_hash for k, cls in ABLATIONS.items()} == FROZEN_HASHES     # v1 unchanged
    for k, i in ids.items():
        v1 = ABLATIONS[k]().identity
        assert (i.name, i.policy, i.version) == (v1.name, v1.policy, "1.1.0")
        changed = {key for key in set(i.config) | set(v1.config) if i.config.get(key) != v1.config.get(key)}
        assert changed == {"planner", "yield_rule", "repair", "unrepaired_blocked_moves", "reservation_ownership",
                           "cycle_rule", "repair_rounds", "push_depth_limit"}
    shared = [{k: v for k, v in i.config.items() if k != "priority_scheme"} for i in ids.values()]
    assert shared[0] == shared[1] == shared[2]
    for bad in ({"repair_rounds": 6}, {"push_depth_limit": -1}, {"planner": "t4-priority-planner@1.0.0"}):
        with pytest.raises(ValueError):
            T4RiskPriorityV11(bad)


def test_v11_module_imports_only_the_frozen_t4_modules():
    tree = ast.parse((ROOT / "src/ember_v1/teacher/t4_v11.py").read_text(encoding="utf-8"))
    mods = {(n.level, n.module) for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    mods |= {(0, a.name) for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert mods <= {(0, "__future__"), (0, "math"), (0, "collections"), (0, "typing"), (1, "interface"), (1, "t4"),
                    (1, "t4_risk")}


def test_v11_keeps_the_v1_order_classes_urgency_and_exit_assignment():
    for k in range(40):
        obs = Simulator(random_scenario(random.Random(900 + k))).observe()
        for v in ABLATIONS:
            a = ABLATIONS[v]().decide(obs).diagnostics
            b = ABLATIONS_V11[v]().decide(obs).diagnostics
            assert a["priority"]["order"] == b["priority"]["order"]
            assert a["priority"]["classes"] == b["priority"]["classes"]
            for pid in a["people"]:
                pa, pb = a["people"][pid], b["people"][pid]
                assert (pa["risk"], pa["target_exit"], pa["priority_class"]) == (pb["risk"], pb["target_exit"],
                                                                               pb["priority_class"])


# ----- P2: reservation transaction units -----------------------------------------------------------------------
def test_reservation_overwrite_raises_instead_of_silently_replacing():
    obs = obs_of(FIXTURES["D1_stay_overwrites_claim"])
    pol = T4NoPriorityV11()
    tp = TransactionalPlanner(pol.identity.config, obs, pol.scheme)
    s_pe = tp.pos["Pe"]
    tp.reserve("Pb", [tp.pos["Pb"], s_pe], hold=False)
    with pytest.raises(ReservationConflict):
        tp.reserve("Pe", [s_pe, s_pe], hold=False)                  # Pe's stay may not take Pb's claim
    assert tp.vres[s_pe * (tp.W + 1) + 1] == "Pb"


def _graph_consistent(tp) -> None:
    W1, a = tp.W + 1, tp.g.area
    assert set(tp.plans) == set(tp.pos)                              # every person planned exactly once
    owners = set(tp.vres.values()) | set(tp.eres.values())
    assert owners <= set(tp.plans)
    for pid, path in tp.plans.items():
        for tau in range(1, len(path)):
            assert tp.vres[path[tau] * W1 + tau] == pid               # every planned step is owned
            if path[tau] != path[tau - 1]:
                assert tp.eres[(path[tau - 1] * a + path[tau]) * W1 + tau] == pid
    for pid, path in tp.plans.items():                               # tau = 1 cells are unique
        assert sum(q[1] == path[1] for q in tp.plans.values()) == 1


def test_failed_vacate_rolls_back_the_sub_chain_and_the_requester_replans():
    _, dg, tp = run("1.1.0", obs_of(FIXTURES["D2_genuine_a0_inversion"]))
    tx = dg["transaction"]
    assert tx["vacate_requests"] == 1 and tx["vacate_failed_no_full_plan"] == 1 and tx["push_rollbacks"] == 1
    assert tp.pos["Ph"] in tp.forbid["Pe"]                           # Pe forbids the cell Ph could not leave
    assert dg["people"]["Ph"]["push_refusals"] == 1 and dg["people"]["Ph"]["origin"] == "Ph"
    _graph_consistent(tp)


def test_dependent_chain_is_planned_with_the_root_origin():
    _, dg, tp = run("1.1.0", obs_of(FIXTURES["D3_dependency_chain"]))
    p = dg["people"]
    assert p["Pf"]["pushed_by"] == "Pa" and p["Pj"]["pushed_by"] == "Pf"
    assert p["Pf"]["origin"] == p["Pj"]["origin"] == "Pa"
    assert dg["predicted_outcomes"] == {"moved": 8, "stayed": 1}
    assert p["Pb"]["priority_effect"] == "yielded_wait" and p["Pb"]["yielded_to"] == "Pj"
    assert p["Pb"]["yield_kind"] == "chain" and p["Pb"]["yield_origin"] == "Pa" and not p["Pb"]["precedence_inversion"]
    _graph_consistent(tp)


# ----- P2: behavior cases --------------------------------------------------------------------------------------
def test_critical_person_still_takes_the_cell_a_lower_id_occupant_leaves():
    s = ascii_scenario(STEP_ASIDE, interval=1, flammable=True, enter_fire="casualty", max_steps=22)
    sim = Simulator(s)
    d = T4RiskPriorityV11().decide(sim.observe())
    rec = sim.step(d.actions)
    pa, pd = d.diagnostics["people"]["Pa"], d.diagnostics["people"]["Pd"]
    assert pd["priority_class"] == "critical" and pd["rank"] < pa["rank"]
    # v1.1 pushes Pa at once, so Pa can vacate along its own route (no detour needed)
    assert pa["vacated"] and pa["pushed_by"] == "Pd" and pa["priority_effect"] in (*YIELD_EFFECTS, "moved_as_intended")
    assert rec["movements"]["Pd"]["to"] == rec["movements"]["Pa"]["from"]


def test_protected_person_vacates_only_by_progress():
    seen = 0
    for k in range(60):
        sim = Simulator(room(14, 40, k, "stochastic"))
        for _ in range(6):
            if sim.terminated:
                break
            obs = validate_observation(sim.observe())
            _, dg, tp = run("1.1.0", obs, "A2")
            for pid, p in dg["people"].items():
                if p["vacated"] and tp.protected[pid]:
                    seen += 1
                    h, s, n = tp.hfield[pid], tp.pos[pid], tp.plans[pid][1]
                    assert tp.g.safe_exit[n] or h[n] < h[s]
            sim.step(T4RiskPriorityV11().act(obs))
        if seen >= 3:
            break
    assert seen >= 1


def test_follow_chain_moves_everyone_and_rotation_is_never_blocked():
    d = T4RiskPriorityV11().decide(Simulator(ascii_scenario(["0dcba....."], interval=100)).observe())
    assert set(d.actions.values()) == {"west"} and d.diagnostics["predicted_outcomes"] == {"moved": 4}
    for rows in (["ab.", "dc.", "..0"], ["ab", "dc", "0."], ["abc", "h.d", "gfe", "0.."]):
        for cls in ABLATIONS_V11.values():
            d = cls().decide(Simulator(ascii_scenario(rows, interval=100)).observe())
            assert set(d.diagnostics["predicted_outcomes"]) <= {"moved", "stayed"}


def test_dense_opposing_traffic_in_a_corridor_has_no_blocked_moves_or_swaps():
    rows = ["0abcdefghijklm1", "###############"]
    for cls in ABLATIONS_V11.values():
        sim = Simulator(ascii_scenario(rows, interval=100, max_steps=40))
        while not sim.terminated:
            d = cls().decide(sim.observe())
            r = sim.step(d.actions)
            assert all(m["outcome"] in ("moved", "stayed") for m in r["movements"].values())
            frm = {p: tuple(m["from"]) for p, m in r["movements"].items()}
            to = {p: tuple(m["to"]) for p, m in r["movements"].items() if m["outcome"] == "moved"}
            assert not any(to.get(q) == frm[p] and to.get(p) == frm[q] for p in to for q in to if p < q)
        assert all(p["status"] == "evacuated" for p in sim.private_state()["people"])


def test_unreachable_people_shelter_and_do_not_break_the_plan():
    rows = ["a#....0", "##.bc..", "...d..*"]
    d = T4RiskPriorityV11().decide(Simulator(ascii_scenario(rows, interval=2)).observe())
    p = d.diagnostics["people"]["Pa"]
    assert not p["route_found"] and p["priority_effect"] == "sheltering" and d.actions["Pa"] == "stay"
    assert set(d.diagnostics["predicted_outcomes"]) <= {"moved", "stayed"}


def test_deterministic_and_order_invariant():
    for k in range(20):
        obs = Simulator(random_scenario(random.Random(1300 + k))).observe()
        for cls in ABLATIONS_V11.values():
            a, b = cls().decide(obs), cls().decide(obs)
            assert a.deterministic_dict() == b.deterministic_dict()
            rev = {**obs, "people": list(reversed(obs["people"]))}
            assert cls().decide(rev).actions == a.actions


@pytest.mark.parametrize("cfg,status,limit", [({"cpu_budget_ms": 0.0}, "cpu_budget_fallback", "cpu_budget_ms"),
                                              ({"node_budget_total": 0}, "node_budget_fallback", "node_budget_total")])
def test_budget_exhaustion_gives_valid_labeled_fallback_without_pushes(cfg, status, limit):
    sim = Simulator(room(24, 30, 3, "stochastic"))
    obs = sim.observe()
    d = T4RiskPriorityV11(cfg).decide(obs)
    dg = d.diagnostics
    assert dg["budget"]["limit_hit"] == limit and set(d.actions) == set(sim.active_ids())
    routed = [p for p in dg["people"].values() if p["route_found"]]
    assert routed and all(p["plan_status"] == status for p in routed)
    assert dg["transaction"].get("vacate_success", 0) == 0 and dg["converted_to_stay"] == []
    assert set(dg["predicted_outcomes"]) <= {"moved", "stayed"}


def test_push_depth_limit_zero_disables_pushes_and_stays_consistent():
    obs = validate_observation(Simulator(room(16, 60, 7, "deterministic")).observe())
    pol = T4RiskPriorityV11({"push_depth_limit": 0})
    tp = _probe(TransactionalPlanner)(pol.identity.config, obs, pol.scheme)
    acts, dg = tp.run()
    assert dg["transaction"].get("vacate_success", 0) == 0
    assert first_pass_blocked(tp) == [] and tp.p_overwrites == []
    _graph_consistent(tp)


def test_200_people_decision_is_bounded():
    sim = Simulator(room(48, 200, 3, "stochastic"))
    for cls in ABLATIONS_V11.values():
        d = cls().decide(sim.observe())
        dg = d.diagnostics
        assert dg["budget"]["limit_hit"] is None and dg["expansions"] <= dg["budget"]["node_budget_total"]
        assert dg["transaction"].get("max_push_depth", 0) <= cls().identity.config["push_depth_limit"]
        assert set(dg["predicted_outcomes"]) <= {"moved", "stayed"} and dg["converted_to_stay"] == []


# ----- P2: randomized property test over small grids --------------------------------------------------------------
def _check_decision(obs, v):
    acts, dg, tp = run("1.1.0", validate_observation(obs), v)
    active = {p["id"] for p in obs["people"] if p["status"] == "active"}
    assert set(acts) == active == set(dg["people"])                       # ID conservation
    assert tp.p_overwrites == [] and tp.precedence_audit == []           # I2, I1
    assert first_pass_blocked(tp) == [] and dg["converted_to_stay"] == []  # I4
    _graph_consistent(tp)
    pg = tp.pg
    res = resolve_moves(height=pg.height, width=pg.width, walls=pg.walls, burning=pg.burning,
                        enter_fire=pg.enter_fire, positions=dict(pg.active), actions=acts)
    assert {p: o for p, (o, _) in res.items()} == {p: x["predicted_outcome"] for p, x in dg["people"].items()}
    assert all(x["priority_effect"] in PRIORITY_EFFECTS for x in dg["people"].values())
    assert dg["priority"]["precedence_inversions"] == 0
    if v == "A0":                                                          # I5
        for pid, x in dg["people"].items():
            if x["yield_over_id_rule"]:
                assert x["yield_kind"] == "chain" and x["yield_origin"] < pid
    return acts


def test_randomized_small_grid_properties():
    rnd = random.Random(20261108)
    decisions = Counter()
    for k in range(120):
        s = random_scenario(random.Random(rnd.getrandbits(32)), h=rnd.randint(3, 7), w=rnd.randint(3, 8))
        s = _crowd(s, rnd)
        for v in ABLATIONS_V11:
            sim = Simulator(s)
            while not sim.terminated:
                obs = sim.observe()
                acts = _check_decision(obs, v)
                decisions[v] += 1
                rec = sim.step(acts)
                assert all(m["outcome"] in ("moved", "stayed") for m in rec["movements"].values())
    assert min(decisions.values()) > 200


def _crowd(s, rnd):
    """Fill up to 60 % of the free cells with people so contests, pushes and chains are common."""
    from ember_v1.sim.scenario import Scenario
    d = s.to_dict()
    taken = {tuple(p) for p in d["obstacles"]} | {tuple(e["position"]) for e in d["exits"]} | \
            {tuple(p) for p in d["ignitions"]} | {tuple(p["position"]) for p in d["people"]}
    free = [(r, c) for r in range(d["grid"]["height"]) for c in range(d["grid"]["width"]) if (r, c) not in taken]
    rnd.shuffle(free)
    extra = free[: int(0.6 * len(free))]
    d["people"] = d["people"] + [{"id": f"Q{i:02d}", "position": list(p)} for i, p in enumerate(extra)]
    return Scenario.from_dict(d)


def test_dense_rooms_closed_loop_resolver_agreement():
    for side, n, seed, m in ((12, 50, 1, "deterministic"), (16, 90, 2, "stochastic"), (20, 150, 3, "stochastic")):
        for v in ABLATIONS_V11:
            sim = Simulator(room(side, n, seed, m))
            for _ in range(12):
                if sim.terminated:
                    break
                acts = _check_decision(sim.observe(), v)
                sim.step(acts)
