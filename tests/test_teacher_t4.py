"""T4 risk-based priority reference agent and its A0/A1 ablations (issue #24).

Covers: identity and independence, observation-only inputs, determinism and
ordering invariance, the public fire forecast and latest-departure fields
(against the simulator and a brute-force time-expanded search), urgency
components, strata, ID independence of the risk calculation, priority that
changes REAL resolved movement (yield over the lowest-ID rule, alternative
routing, step-aside), safety and fairness cases, budgets and fallbacks.
Predicted-vs-actual movement checks use the simulator's own resolver.
"""
from __future__ import annotations

import ast
import copy
import random
from collections import Counter
from pathlib import Path

import pytest

from ember_v1.sim import Scenario, Simulator
from ember_v1.sim.bench import room
from ember_v1.teacher.grid import PublicGrid
from ember_v1.teacher.t0 import T0ShortestPath
from ember_v1.teacher.t1 import T1CongestionAware
from ember_v1.teacher.t2 import T2StochasticRollout
from ember_v1.teacher.t3 import T3DynamicACO
from ember_v1.teacher.t4 import (ABLATIONS, PRIORITY_EFFECTS, YIELD_EFFECTS, T4DistancePriority, T4NoPriority,
                                 T4RiskPriority)
from ember_v1.teacher.t4_risk import INF, STRATA, assess, score
from gen_helpers import random_scenario
from test_teacher_t0 import PRIVATE_KEYS, ascii_scenario

FROZEN_HASHES = {
    "A0": "sha256:f561075ebc1c2cfbffeb55d196bc2675801b167d11e4c4b157079e33fc506243",
    "A1": "sha256:e9a70eab344df4237a41efdda2522a7547ac2f94b3549ce821b96c5bae49b9ff",
    "A2": "sha256:dbcd0af4752ffcaad00e89a38ffcba5d3867fa3e4de0cce9fd560533d4794195",
}
ROOT = Path(__file__).resolve().parents[1]
# Exit contest: Pb (low ID, no fire on its side) and Pz (high ID, fire behind) reach E0 in the same step.
CONTEST = ["b.0.z*"]
# Alternative routing: four urgent people queue at E0; low-risk Pb can use E1 instead.
ALT_ROUTE = ["1..b.0.zyxw*"]
# Step aside: lower-ID Pa must vacate the cell that critical Pd needs (found by a synthetic search).
STEP_ASIDE = [".#..#cb0", ".....1a.", "..*#..d#"]


def det(rows, **kw) -> Scenario:
    return ascii_scenario(rows, interval=kw.pop("interval", 1), flammable=kw.pop("flammable", False),
                          enter_fire=kw.pop("enter_fire", "blocked"), **kw)


def episode(s: Scenario, pol):
    sim = Simulator(s)
    log = []
    while not sim.terminated:
        d = pol.decide(sim.observe())
        log.append((d, sim.step(d.actions)))
    return sim, log


def statuses(sim) -> dict[str, str]:
    return {p["id"]: p["status"] for p in sim.private_state()["people"]}


def ra_of(obs, cfg=None):
    return assess(PublicGrid.from_observation(obs), obs, cfg or dict(T4RiskPriority().identity.config))


# ----- identity and independence ----------------------------------------------------------------------
def test_identities_and_frozen_config_hashes():
    ids = {k: cls().identity for k, cls in ABLATIONS.items()}
    assert {k: v.config_hash for k, v in ids.items()} == FROZEN_HASHES
    assert (ids["A2"].name, ids["A2"].policy, ids["A2"].version) == ("T4", "risk_based_priority", "1.0.0")
    assert (ids["A0"].name, ids["A1"].name) == ("T4-A0", "T4-A1")
    others = {T0ShortestPath().identity.config_hash, T1CongestionAware().identity.config_hash,
              T2StochasticRollout().identity.config_hash, T3DynamicACO().identity.config_hash}
    assert not set(FROZEN_HASHES.values()) & others
    shared = [{k: v for k, v in i.config.items() if k != "priority_scheme"} for i in ids.values()]
    assert shared[0] == shared[1] == shared[2]                    # one planner, one set of numbers
    assert [i.config["priority_scheme"] for i in ids.values()] == ["none", "fire_distance", "risk"]
    assert T4RiskPriority().seed is None


@pytest.mark.parametrize("cfg", [{"priority_scheme": "none"}, {"memory": "cross_step"}, {"w_margin": 0.5},
                                 {"q_early": 0.9}, {"reservation_horizon": 0}, {"cpu_budget_ms": float("nan")},
                                 {"stratum_high": 1.0}, {"nope": 1}])
def test_invalid_or_descriptive_config_is_rejected(cfg):
    with pytest.raises(ValueError):
        T4RiskPriority(cfg)


@pytest.mark.parametrize("module", ["t4.py", "t4_risk.py"])
def test_t4_modules_are_independent_of_other_teachers_and_private_state(module):
    allowed = {"ember_v1.sim.movement", "ember_v1.teacher.grid", "ember_v1.teacher.interface",
               "ember_v1.teacher.t4_risk", "__future__", "heapq", "math", "time", "collections", "typing",
               "dataclasses"}
    tree = ast.parse((ROOT / "src/ember_v1/teacher" / module).read_text(encoding="utf-8"))
    for node in ast.walk(tree):                     # drop docstrings: they name the forbidden fields on purpose
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            body[0] = ast.Pass()
    src = ast.unparse(tree)                         # code only, no comments
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            full = {1: f"ember_v1.teacher.{node.module}", 2: f"ember_v1.{node.module}"}.get(node.level, node.module)
            assert full in allowed, full
        elif isinstance(node, ast.Import):
            assert all(a.name in allowed for a in node.names)
    for word in ("emberbench/v1", "stream_seed", "private_state", "trapped", "random"):
        assert word not in src


@pytest.mark.parametrize("key", PRIVATE_KEYS)
def test_private_fields_are_rejected_not_ignored(key):
    obs = Simulator(ascii_scenario(["0.a"])).observe()
    for where in ("root", "map", "rules", "person"):
        bad = copy.deepcopy(obs)
        {"root": bad, "map": bad["map"], "rules": bad["rules"], "person": bad["people"][0]}[where][key] = 1
        with pytest.raises(ValueError):
            T4RiskPriority().decide(bad)


# ----- reproducibility ------------------------------------------------------------------------------------
@pytest.mark.parametrize("cls", list(ABLATIONS.values()))
def test_repeated_decisions_are_identical_and_stateless(cls):
    sim = Simulator(room(24, 30, 2, "stochastic"))
    sim.step(cls().act(sim.observe()))
    obs = sim.observe()
    pol = cls()
    a = pol.decide(obs).deterministic_dict()
    pol.decide(Simulator(room(24, 10, 9)).observe())
    assert pol.decide(obs).deterministic_dict() == a == cls().decide(obs).deterministic_dict()


def test_observation_list_order_does_not_change_the_decision():
    obs = Simulator(room(24, 30, 4, "stochastic")).observe()
    ref = T4RiskPriority().decide(obs).deterministic_dict()
    rnd = random.Random(1)
    for _ in range(3):
        o = copy.deepcopy(obs)
        for lst in (o["people"], o["map"]["exits"], o["map"]["obstacles"], o["burning"]):
            rnd.shuffle(lst)
        d = T4RiskPriority().decide(o).deterministic_dict()
        assert d["actions"] == ref["actions"] and d["diagnostics"] == ref["diagnostics"]


def test_evaluator_seed_and_scenario_id_do_not_change_the_decision():
    base = room(24, 20, 3, "stochastic").to_dict()
    decisions = []
    for seed, sid in ((1, "a"), (2, "b"), (99, "c")):
        d = {**base, "seed": seed, "scenario_id": sid}
        decisions.append(T4RiskPriority().decide(Simulator(Scenario.from_dict(d)).observe()).deterministic_dict())
    assert all(x["actions"] == decisions[0]["actions"] for x in decisions)


# ----- public fire forecast and deadline fields --------------------------------------------------------------
def _stay_burn_steps(s: Scenario) -> tuple[dict[tuple[int, int], int], int]:
    """First burning step of each cell while everyone stays, and the last step observed."""
    sim = Simulator(s)
    first: dict[tuple[int, int], int] = {tuple(b): 0 for b in sim.observe()["burning"]}
    t = 0
    while not sim.terminated:
        sim.step({pid: "stay" for pid in sim.active_ids()})
        t += 1
        for b in sim.observe()["burning"]:
            first.setdefault(tuple(b), t)
    return first, t


def test_deterministic_arrival_equals_the_public_rule():
    checked = 0
    for k in range(40):
        s = random_scenario(random.Random(k), model="deterministic", max_steps=30)
        obs = Simulator(s).observe()
        ra = ra_of(obs)
        truth, last = _stay_burn_steps(s)
        g = ra.cells
        for i in range(g.area):
            if g.wall[i]:
                continue
            t = ra.t_early[i]
            assert t == ra.t_late[i]
            p = g.rc(i)
            if p in truth:
                assert t == truth[p], (k, p)
            else:
                assert t > last, (k, p, t)         # not before the observed horizon
            checked += 1
    assert checked > 500


def test_stochastic_arrival_quantiles_are_ordered_and_layer1_exact():
    s = ascii_scenario(["0....", "..*..", "....a"], model="stochastic", interval=1, max_steps=100)
    d = s.to_dict()
    d["fire"]["spread_probability"] = 0.3
    obs = Simulator(Scenario.from_dict(d)).observe()
    ra = ra_of(obs)
    fc = ra.forecast
    for i in range(ra.cells.area):
        assert ra.t_early[i] <= ra.t_late[i]
        lay = fc.layer[i]
        if lay == 1:                                    # 1 - 0.7^n >= q exactly
            n_e = next(n for n in range(1, 99) if 1 - 0.7 ** (fc.kburn[i] * n) >= 0.25)
            n_l = next(n for n in range(1, 99) if 1 - 0.7 ** (fc.kburn[i] * n) >= 0.75)
            assert (ra.t_early[i], ra.t_late[i]) == (n_e, n_l)
        if lay >= 2 and lay < INF:
            assert ra.t_early[i] >= lay and ra.t_late[i] > ra.t_early[i]
    # probability is bounded and monotone in tau
    for i in range(ra.cells.area):
        ps = [fc.prob(i, t) for t in range(0, 12)]
        assert all(0.0 <= p <= 1.0 for p in ps) and ps == sorted(ps)


def test_spread_interval_shifts_the_arrival():
    for interval, step_expected in ((1, 2), (3, 6)):
        s = ascii_scenario(["0a..*"], interval=interval, max_steps=50)
        ra = ra_of(Simulator(s).observe())
        assert ra.t_early[2] == step_expected        # layer 2 -> second spread step


def _brute_latest(ra, eid: str, s: int, bound: int) -> int:
    g, T = ra.cells, ra.t_early
    e = dict(g.safe_exits)[eid]
    H = bound + g.area + 2
    reach = {c: (c == e and H <= T[e]) for c in range(g.area)}
    best = -INF
    for t in range(H - 1, -1, -1):
        nxt = {}
        for c in range(g.area):
            if not g.open[c] or (g.safe_exit[c] and c != e):
                nxt[c] = False
            elif c == e:
                nxt[c] = t <= T[e]
            else:
                nxt[c] = T[c] > t and (reach[c] or any(reach[n] for _, n in g.succ[c]))
        reach = nxt
        if t <= bound and reach[s] and best == -INF:
            best = t
    return best


def test_latest_departure_matches_a_brute_force_time_expanded_search():
    checked = 0
    for k in range(30):
        s = random_scenario(random.Random(100 + k), h=5, w=6, model="deterministic", max_steps=60)
        obs = Simulator(s).observe()
        ra = ra_of(obs)
        for eid, f in ra.fields.items():
            for pid, i in ra.pos.items():
                want = _brute_latest(ra, eid, i, 40)
                got = f.slack_early[i]
                if want == -INF:
                    assert got < 0, (k, eid, pid)
                elif want < 40:
                    assert got == want, (k, eid, pid, got, want)
                else:
                    assert got >= 40
                checked += 1
    assert checked > 40


def test_urgency_components_are_bounded_and_the_margin_identity_holds():
    for k in range(30):
        s = random_scenario(random.Random(300 + k))
        ra = ra_of(Simulator(s).observe())
        for r in ra.people.values():
            assert 0.0 <= r["urgency"] <= 1.0
            assert all(0.0 <= v <= 1.0 for v in r["components"].values())
            assert r["stratum"] in STRATA
            if r["margin"] is not None:
                assert r["margin"] == r["time_to_route_loss"] - r["evacuation_time"]


def _rec(**kw):
    base = {"options": 1, "congestion_delay": 0.0, "slack_early": 50, "route_len_early": 5, "fire_layer": 10,
            "exposure_early": 20, "shortest_viable": 5}
    base.update(kw)
    return base


@pytest.mark.parametrize("kw,stratum", [({"options": 0, "slack_early": -INF}, "no_viable_route"),
                                        ({"slack_early": -INF}, "critical"), ({"slack_early": 2}, "critical"),
                                        ({"slack_early": 9}, "high"), ({"slack_early": 12, "congestion_delay": 4.0},
                                                                       "high"),
                                        ({"slack_early": 24}, "elevated"), ({"slack_early": 25}, "low")])
def test_strata_thresholds(kw, stratum):
    r = _rec(**kw)
    score(r, dict(T4RiskPriority().identity.config))
    assert r["stratum"] == stratum


def test_risk_calculation_is_id_independent():
    rnd = random.Random(7)
    for k in range(10):
        s = room(20, 25, k + 1, "stochastic")
        d = s.to_dict()
        ids = [p["id"] for p in d["people"]]
        perm = ids[:]
        rnd.shuffle(perm)
        d2 = copy.deepcopy(d)
        for p, new in zip(d2["people"], perm):
            p["id"] = new
        s2 = Scenario.from_dict(d2)
        o1, o2 = Simulator(s).observe(), Simulator(s2).observe()
        r1, r2 = ra_of(o1), ra_of(o2)
        by_pos1 = {r1.pos[p]: r for p, r in r1.people.items()}
        by_pos2 = {r2.pos[p]: r for p, r in r2.people.items()}
        assert by_pos1 == by_pos2
        a1 = T4RiskPriority().decide(o1).diagnostics
        a2 = T4RiskPriority().decide(o2).diagnostics
        order1 = [r1.pos[p] for p in a1["priority"]["order"]]
        order2 = [r2.pos[p] for p in a2["priority"]["order"]]
        assert order1 == order2                          # planning order is a function of positions only


def test_urgency_rises_monotonically_as_the_fire_spreads():
    s = det(["0......a....*"], interval=1, max_steps=40)
    sim = Simulator(s)
    seen = []
    for _ in range(5):                                  # Pa burns at step 5
        ra = ra_of(sim.observe())
        seen.append((ra.people["Pa"]["urgency"], STRATA.index(ra.people["Pa"]["stratum"])))
        sim.step({"Pa": "stay"})
    us = [u for u, _ in seen]
    assert us == sorted(us) and us[-1] > us[0]
    assert [r for _, r in seen] == sorted([r for _, r in seen], reverse=True)   # severity never decreases


# ----- priority changes actual movement -------------------------------------------------------------------
def test_exit_contest_priority_beats_the_lowest_id_rule_through_coordination():
    s = det(CONTEST)
    out = {}
    for name, pol in (("T1", T1CongestionAware()), ("A0", T4NoPriority()), ("A1", T4DistancePriority()),
                      ("A2", T4RiskPriority())):
        sim, log = episode(s, pol)
        out[name] = statuses(sim)
    assert out["T1"]["Pz"] == out["A0"]["Pz"] == "casualty"           # the lowest-ID rule decides the contest
    assert out["A2"] == out["A1"] == {"Pb": "evacuated", "Pz": "evacuated"}
    sim, log = episode(s, T4RiskPriority())
    d, rec = log[1]                                                     # both one step from E0
    pb, pz = d.diagnostics["people"]["Pb"], d.diagnostics["people"]["Pz"]
    assert pz["priority_class"] == "critical" and pb["priority_class"] == "low" and pz["rank"] < pb["rank"]
    assert pb["priority_effect"] == "yielded_wait" and pb["yielded_to"] == "Pz" and pb["yield_over_id_rule"]
    assert d.actions == {"Pb": "stay", "Pz": "west"}
    assert rec["movements"]["Pz"]["outcome"] == "moved" and rec["evacuations"] == {"Pz": "E0"}
    assert d.diagnostics["priority"]["effective_yields_over_id_rule"] == 1


def test_low_risk_person_is_rerouted_to_another_exit():
    s = det(ALT_ROUTE)
    sim0, _ = episode(s, T4NoPriority())
    sim2, log = episode(s, T4RiskPriority())
    assert Counter(statuses(sim0).values())["casualty"] >= 1
    assert set(statuses(sim2).values()) == {"evacuated"}
    pb = log[0][0].diagnostics["people"]["Pb"]
    assert pb["target_exit"] == "E1" and pb["diverted"] and pb["nearest_exit"] == "E0"
    assert log[0][0].diagnostics["people"]["Pz"]["target_exit"] == "E0"


def test_lower_id_occupant_vacates_for_a_critical_person():
    s = ascii_scenario(STEP_ASIDE, interval=1, flammable=True, enter_fire="casualty", max_steps=22)
    sim = Simulator(s)
    obs = sim.observe()
    d = T4RiskPriority().decide(obs)
    rec = sim.step(d.actions)
    pa, pd = d.diagnostics["people"]["Pa"], d.diagnostics["people"]["Pd"]
    assert pd["priority_class"] == "critical" and pd["rank"] < pa["rank"]
    assert pa["vacate_requested_by"] == "Pd" and pa["vacated"] and pa["priority_effect"] in YIELD_EFFECTS
    assert pa["yielded_to"] == "Pd" and pa["yield_over_id_rule"]
    assert rec["movements"]["Pd"]["outcome"] == "moved"
    assert rec["movements"]["Pd"]["to"] == rec["movements"]["Pa"]["from"]       # Pd took the cell Pa left
    a0 = T4NoPriority().decide(obs).diagnostics["people"]
    assert a0["Pa"]["vacate_requested_by"] is None                               # no priority, no request


def test_a0_never_yields_over_the_id_rule_and_a2_sometimes_does():
    """True on these small scenarios. The CPU pilot found A0 yields over the ID rule after repair re-plans
    (docs/P1_T4_RISK_PRIORITY.md section 9.6); this test does not claim it never happens."""
    over = Counter()
    for k in range(120):
        s = random_scenario(random.Random(k), model="deterministic")
        for v in ("A0", "A2"):
            _, log = episode(s, ABLATIONS[v]())
            over[v] += sum(d.diagnostics["priority"]["effective_yields_over_id_rule"] for d, _ in log)
            over[v + "_any"] += sum(d.diagnostics["priority"]["yields_over_id_rule"] for d, _ in log)
    assert over["A0_any"] == 0
    assert over["A2"] >= 1


def test_distance_only_and_risk_priority_use_their_own_keys_and_differ():
    differ = 0
    for k in range(60):
        obs = Simulator(random_scenario(random.Random(700 + k))).observe()
        ra = ra_of(obs)
        a1 = T4DistancePriority().decide(obs).diagnostics["priority"]["order"]
        a2 = T4RiskPriority().decide(obs).diagnostics["priority"]["order"]
        lay = [INF if ra.people[p]["fire_layer"] is None else ra.people[p]["fire_layer"] for p in a1
               if ra.people[p]["static_distance"] is not None]
        assert lay == sorted(lay)                                    # A1: nearest fire first
        sev = [(STRATA.index(ra.people[p]["stratum"]), -ra.people[p]["urgency"]) for p in a2]
        assert sev == sorted(sev)                                    # A2: stratum, then urgency
        differ += a1 != a2
    assert differ >= 1


# ----- safety and fairness ----------------------------------------------------------------------------------
def test_two_high_risk_people_in_one_corridor_follow_each_other():
    sim, log = episode(det(["0..yz*"]), T4RiskPriority())
    assert set(statuses(sim).values()) == {"evacuated"}
    assert all(m["outcome"] in ("moved", "stayed") for _, r in log for m in r["movements"].values())


def test_narrow_corridors_with_opposing_flows_have_no_blocked_moves_or_swaps():
    from ember_v1.teacher.t4_pilot import pilot_scenario
    for n in (10, 50):
        cell = {"kind": "base", "topology": "corridor", "grid": (16, 20), "n": n, "model": "stochastic", "exits": 3}
        s = pilot_scenario(cell)
        for cls in ABLATIONS.values():
            sim, log = episode(s, cls())
            for d, r in log:
                assert all(m["outcome"] in ("moved", "stayed") for m in r["movements"].values())
                tgt = {pid: tuple(m["to"]) for pid, m in r["movements"].items() if m["outcome"] == "moved"}
                frm = {pid: tuple(m["from"]) for pid, m in r["movements"].items()}
                for p, t in tgt.items():
                    q = next((q for q, f in frm.items() if f == t and q != p), None)
                    assert q is None or tgt.get(q) != frm[p]
                assert all(d.diagnostics["people"][p]["predicted_outcome"] == m["outcome"]
                           for p, m in r["movements"].items())


def test_follow_chain_and_rotation_cycle():
    d = T4RiskPriority().decide(Simulator(det(["0dcba....."], interval=100)).observe())
    assert set(d.actions.values()) == {"west"} and d.diagnostics["predicted_outcomes"] == {"moved": 4}
    d2 = T4RiskPriority().decide(Simulator(det(["ab.", "dc.", "..0"], interval=100)).observe())
    assert set(d2.diagnostics["predicted_outcomes"]) <= {"moved", "stayed"}


def test_exit_that_will_burn_first_is_not_targeted():
    rows = ["0.....a......1", "*#############"]
    s = ascii_scenario(rows, interval=1, flammable=True, enter_fire="blocked", max_steps=40)
    obs = Simulator(s).observe()
    for cls in ABLATIONS.values():
        p = cls().decide(obs).diagnostics["people"]["Pa"]
        assert p["target_exit"] == "E1"
    sim, _ = episode(s, T4RiskPriority())
    assert statuses(sim) == {"Pa": "evacuated"}


def test_exit_burning_now_is_not_a_destination_and_no_safe_exit_shelters():
    rows = ["*0.a..", "......"]
    s = ascii_scenario(rows, interval=1, flammable=True, enter_fire="blocked", max_steps=30)
    sim = Simulator(s)
    sim.step({"Pa": "stay"})
    d = T4RiskPriority().decide(sim.observe())
    p = d.diagnostics["people"]["Pa"]
    assert d.diagnostics["safe_exits"] == [] and not p["route_found"] and p["priority_effect"] == "sheltering"
    assert p["risk"]["stratum"] == "no_viable_route" and d.actions["Pa"] in ("east", "south", "stay")


def test_priority_induced_waiting_is_bounded_and_low_risk_people_finish():
    s = det(["b.0.zyxw*"])
    sim, log = episode(s, T4RiskPriority())
    assert statuses(sim)["Pb"] == "evacuated"
    waits = sum(r["movements"]["Pb"]["outcome"] != "moved" for _, r in log if "Pb" in r["movements"])
    assert waits <= 4                                     # at most one wait per higher-priority person


def test_overloaded_single_exit_with_200_people_is_legal():
    from ember_v1.teacher.t4_pilot import pilot_scenario
    cell = {"kind": "base", "topology": "open_room", "grid": (48, 64), "n": 200, "model": "stochastic", "exits": 1}
    sim = Simulator(pilot_scenario(cell))
    for cls in ABLATIONS.values():
        d = cls().decide(sim.observe())
        assert set(d.actions) == set(sim.active_ids())
        assert set(d.diagnostics["predicted_outcomes"]) <= {"moved", "stayed"}
        assert d.diagnostics["budget"]["limit_hit"] is None


@pytest.mark.parametrize("n,side", [(1, 24), (10, 24), (50, 32), (100, 32), (200, 48)])
def test_complete_valid_maps_for_n_1_to_200(n, side):
    sim = Simulator(room(side, n, 5, "stochastic"))
    for _ in range(2):
        sim.step(T4RiskPriority().act(sim.observe()))
    obs = sim.observe()
    for cls in ABLATIONS.values():
        d = cls().decide(obs)
        assert set(d.actions) == set(sim.active_ids())
        assert set(d.diagnostics["predicted_outcomes"]) <= {"moved", "stayed"}
        assert all(p["priority_effect"] in PRIORITY_EFFECTS for p in d.diagnostics["people"].values())


def test_closed_loop_random_episodes_are_legal_and_predicted():
    steps = 0
    for k in range(40):
        s = random_scenario(random.Random(500 + k))
        for cls in ABLATIONS.values():
            sim, log = episode(s, cls())
            for d, r in log:
                steps += 1
                for pid, m in r["movements"].items():
                    assert m["outcome"] in ("moved", "stayed")
                    assert d.diagnostics["people"][pid]["predicted_outcome"] == m["outcome"]
    assert steps > 300


@pytest.mark.parametrize("cfg,status,limit", [({"cpu_budget_ms": 0.0}, "cpu_budget_fallback", "cpu_budget_ms"),
                                              ({"node_budget_total": 0}, "node_budget_fallback", "node_budget_total")])
def test_budget_exhaustion_gives_a_valid_labeled_fallback(cfg, status, limit):
    sim = Simulator(room(24, 30, 3, "stochastic"))
    obs = sim.observe()
    d = T4RiskPriority(cfg).decide(obs)
    dg = d.diagnostics
    assert set(d.actions) == set(sim.active_ids()) and dg["budget"]["limit_hit"] == limit
    routed = [p for p in dg["people"].values() if p["route_found"]]
    assert routed and all(p["plan_status"] == status and p["fallback"] and p["fallback_reason"] == status
                          for p in routed)
    assert set(dg["predicted_outcomes"]) <= {"moved", "stayed"}
    sim.step(d.actions)
    if status == "node_budget_fallback":
        assert T4RiskPriority(cfg).decide(obs).deterministic_dict() == d.deterministic_dict()


def test_a2_is_sometimes_worse_and_sometimes_better_than_a0_on_synthetic():
    def cas(k, v):
        sim, _ = episode(random_scenario(random.Random(k)), ABLATIONS[v]())
        return Counter(statuses(sim).values())["casualty"]
    assert any(cas(k, "A2") < cas(k, "A0") for k in (46, 75, 294, 912))
    assert any(cas(k, "A2") > cas(k, "A0") for k in (199, 435, 1042, 1528))
