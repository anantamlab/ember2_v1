"""Teacher selection protocol (#25): five separated views, fail-closed protocol, prospective gate, arbiter."""
from __future__ import annotations

import copy
import json

import pytest

from ember_v1 import s1
from ember_v1.sim import Simulator
from ember_v1.sim.canonical import digest
from ember_v1.teacher import arbiter as arb
from ember_v1.teacher import proposals as pr
from ember_v1.teacher import registry as reg
from ember_v1.teacher import selection as sel
from ember_v1.teacher.t0 import T0ShortestPath
from test_s1_gate import Repo
from test_teacher_t0 import ascii_scenario


# ----- historical inventory and status ------------------------------------------------------------------------
def test_inventory_of_committed_dev_reports_is_historical_only():
    inv = sel.evidence_inventory()
    c = inv["candidates"]
    assert inv["evidence_class"] == sel.HISTORICAL and inv["counts_as_confirmation"] is False
    assert [t for t in sel.CANDIDATES if c[t]["status"] == "dev_evidence"] == ["T0", "T1", "T2", "T3"]
    assert c["T4"]["status"] == "admitted_without_historical_evidence" and c["T4"]["scenarios"] == 0
    assert c["T4"]["admission"]["version"] == "1.2.0" and c["T4"]["admission"]["namespace"] == "t4-pilot@3"
    assert [b["failed_gate"] for b in c["T4"]["blocked_identities"]] == ["G5", "G4"]
    assert inv["cross_report_conflicts"] == [] and inv["same_dev_metadata"] and inv["same_positions"]
    for t in ("T0", "T1", "T2", "T3"):
        assert c[t]["scenarios"] == 48 and len(c[t]["config_hashes"]) == 1 and c[t]["failures"] == 0


def test_status_keeps_the_five_views_separate():
    st = sel.status()
    assert set(st["retained_identities"]["candidates"]) == set(sel.CANDIDATES)
    assert st["eligible"] == ["T0", "T1", "T2", "T3", "T4"] and st["excluded"] == {}
    t4 = st["retained_identities"]["candidates"]["T4"]
    assert [(x["version"], x["status"]) for x in t4] == [("1.0.0", "blocked"), ("1.1.0", "blocked"),
                                                          ("1.2.0", "retained")]
    assert st["historical_exploratory_evidence"]["T1"]["scenarios"] == 48
    assert st["prospective_confirmation_evidence"] is None and st["protocol_frozen"] is False
    assert st["decision"] == "NO_SELECTION" and st["operational_baseline"] == "T1"
    assert st["operational_baseline_basis"] == "historical_exploratory_provisional"


def test_value_teachers_are_shared_by_proposals_provenance_and_registry():
    from ember_v1 import provenance as pv
    assert pr.VALUE_TEACHERS is reg.VALUE_TEACHERS is pv.VALUE_TEACHERS


# ----- protocol ------------------------------------------------------------------------------------------------
def _frozen(**over):
    p = sel.draft_protocol()
    p["frozen"] = True
    p.update(over)
    return sel.freeze_protocol(p)


def test_draft_is_not_frozen_and_freezes_only_explicitly():
    d = sel.draft_protocol()
    assert d["frozen"] is False
    with pytest.raises(sel.ProtocolError, match="frozen"):
        sel.freeze_protocol(d)
    f = _frozen()
    assert sel.verify_frozen_protocol(f) == f
    with pytest.raises(sel.ProtocolError, match="hash"):
        sel.verify_frozen_protocol({**f, "protocol_id": "edited"})


def _elig(fn):
    e = copy.deepcopy(reg.default_eligibility())
    fn(e)
    return e


def _conf(**over):
    return {**sel.draft_protocol()["confirmation"], **over}


def _prim(**over):
    return {**sel.draft_protocol()["primary"], **over}


T4_OUT = {"status": "excluded", "reason": "excluded_by_protocol_rule", "rule": "test exclusion",
          "blocked_identities": reg.blocked_records("T4")}
FOUR = {"order": "lower_cpu_then_simplicity", "simplicity_order": ["T0", "T1", "T3", "T2"]}


def _t4_out(fn=lambda e: None):
    e = _elig(lambda e: e.update(T4=copy.deepcopy(T4_OUT)))
    fn(e)
    return {"eligibility": e, "t4_entry_rule": {**sel.t4_entry_rule(), "admitted": None}, "tie_rule": FOUR}


def _adm(**over):
    r = sel.t4_entry_rule()
    return {"t4_entry_rule": {**r, "admitted": {**r["admitted"], **over}}}


def test_draft_lists_all_five_candidates_with_t4_v12_admission_and_blocked_records():
    d = sel.draft_protocol()
    assert [c for c in sel.CANDIDATES if d["eligibility"][c]["status"] == "eligible"] == list(sel.CANDIDATES)
    t4r = d["t4_entry_rule"]
    assert t4r["admitted"] == reg.admission_record("T4") and t4r["admitted"]["version"] == "1.2.0"
    assert t4r["blocked_identities"] == reg.blocked_records("T4")
    assert d["tie_rule"]["simplicity_order"] == ["T0", "T1", "T3", "T4", "T2"]
    assert d["protocol_version"] == "teacher-selection-protocol@1.1.0"
    assert sel.verify_frozen_protocol(_frozen(**_t4_out()))["eligibility"]["T4"]["status"] == "excluded"


BAD_PROTOCOL = {
    "T4 eligible as blocked v1.1.0": ({"eligibility": _elig(lambda e: e.update(
        T4={"status": "eligible", "identity": reg.identity("T4", "1.1.0").key()}))}, "blocked"),
    "T4 excluded without blocked record": (_t4_out(lambda e: e["T4"].update(blocked_identities=[])),
                                           "blocked_identities"),
    "T4 excluded as blocked although v1.2.0 retained": ({"eligibility": _elig(lambda e: e.update(T4={
        "status": "excluded", "reason": "blocked_by_preregistered_gate",
        "blocked_identities": reg.blocked_records("T4")}))}, "retained identity"),
    "T4 excluded but admission kept": ({**_t4_out(), "t4_entry_rule": sel.t4_entry_rule()}, "admission"),
    "T4 eligible without admission": ({"t4_entry_rule": {**sel.t4_entry_rule(), "admitted": None}}, "admission"),
    "T4 admission wrong report hash": (_adm(report_sha256="sha256:" + "0" * 64), "admission"),
    "T4 admission wrong verdict": (_adm(verdict="T4_V2_BLOCKED"), "admission"),
    "T4 admission wrong namespace": (_adm(namespace="t4-pilot@2"), "admission"),
    "T4 admission names v1.1.0": (_adm(**reg.identity("T4", "1.1.0").key()), "admission"),
    "T4 entry rule drops blocked records": ({"t4_entry_rule": {**sel.t4_entry_rule(), "blocked_identities": []}},
                                            "blocked T4 records"),
    "T4 entry rule extra key": ({"t4_entry_rule": {**sel.t4_entry_rule(), "note": "x"}}, "t4_entry_rule"),
    "T4 missing": ({"eligibility": _elig(lambda e: e.pop("T4"))}, "exactly"),
    "four candidates": ({"candidates": ["T0", "T1", "T2", "T3"]}, "all five"),
    "T4 entry left open": ({"t4_entry_rule": {"deadline": "after_results"}}, "t4_entry_rule"),
    "tie rule without T4": ({"tie_rule": FOUR}, "tie rule"),
    "tie rule T4 last": ({"tie_rule": {"order": "lower_cpu_then_simplicity",
                                       "simplicity_order": ["T0", "T1", "T3", "T2", "T4"]}}, "tie rule"),
    "excluded T4 in tie rule": ({**_t4_out(), "tie_rule": sel.draft_protocol()["tie_rule"]}, "tie rule"),
    "baseline T2": ({"baseline": {"teacher": "T2", "basis": "historical_exploratory_provisional"}}, "baseline"),
    "baseline confirmed claim": ({"baseline": {"teacher": "T1", "basis": "confirmed"}}, "baseline"),
    "historical used as confirmation": ({"historical_evidence": {"use": "confirmation", "reports": []}},
                                        "descriptive_only"),
    "reuse historical namespace": ({"confirmation": _conf(namespace="p1-02-v1-dev")}, "new namespace"),
    "too few families": ({"confirmation": _conf(families=4)}, "10 families"),
    "isolation incomplete": ({"confirmation": _conf(isolation_against=["p1-02-v1/dev"])}, "isolation"),
    "two runs": ({"confirmation": _conf(runs_per_identity=2)}, "one confirmatory run"),
    "primary unresolved": ({"primary": _prim(metric="unresolved")}, "primary"),
    "few resamples": ({"primary": _prim(bootstrap_resamples=10)}, "primary"),
    "no safety guard": ({"safety": {"casualties_plus_unresolved": "no_net_deterioration"}}, "safety"),
    "budget over cap": ({"compute": {**sel.draft_protocol()["compute"], "usd_cap": 50}}, "compute"),
    "inline arbiter": ({"arbiter": {"equal_compute_to_best_single": True}}, "separately frozen"),
    "calibration not excluded": ({"excluded_data": ["locked_test", "blind_test", "emberbench"]}, "exclude"),
    "stop rule missing": ({"stop_rules": ["freeze_commit_missing"]}, "stop rules"),
    "unknown field": ({"note": "x"}, "unknown fields"),
}


@pytest.mark.parametrize("case", sorted(BAD_PROTOCOL))
def test_unsafe_or_ambiguous_protocols_are_refused(case):
    over, match = BAD_PROTOCOL[case]
    with pytest.raises(sel.ProtocolError, match=match):
        _frozen(**over)


def test_protocol_missing_a_field_is_refused():
    p = sel.draft_protocol()
    p["frozen"] = True
    del p["stop_rules"]
    with pytest.raises(sel.ProtocolError, match="misses"):
        sel.freeze_protocol(p)


# ----- prospective confirmation gate (synthetic numbers: gate logic only, no teacher run) ----------------------
ELIGIBLE = ("T0", "T1", "T2", "T3", "T4")
ARB_ELIGIBLE = ("T0", "T1", "T2", "T3")     # an arbiter protocol beside a T4-excluding selection protocol
NS = "p1-09-selection@1"


def _conf_manifest(fams=24, lin=3, prefix="CF"):
    sc = [{"scenario_id": f"{prefix}-{f}-{k}", "scenario_hash": digest(["s", prefix, f, k]),
           "core_hash": digest(["c", prefix, f, k]), "lineage_id": f"{prefix}L-{f}-{k}", "family_id": f"{prefix}-{f}"}
          for f in range(fams) for k in range(lin)]
    body = {"manifest_version": "1.0", "split": "selection_confirmation", "plan_id": NS,
            "plan_hash": "sha256:" + "1" * 64, "generator_revision": sel.GENERATOR_REVISION,
            "spec_hash": "sha256:" + "2" * 64, "families": [], "scenarios": sc, "failed_lineages": [],
            "lineage_rejections": {}}
    return {**body, "manifest_hash": digest(body)}


def _historical(**over):
    h = {name: [{"family_id": f"H-{i}", "lineage_id": f"HL-{i}", "scenario_hash": digest(["h", name, i]),
                 "core_hash": digest(["hc", name, i])}] for i, name in enumerate(sel.REQUIRED_ISOLATION)}
    h.update(over)
    return h


def _result(c, cas=5, unres=1, evac=10, cpu=1.0):
    return {"config_hash": reg.identity(c).config_hash, "casualties": cas, "evacuated": evac, "unresolved": unres,
            "people": cas + unres + evac, "failure": None, "final_transition_hash": digest([c, cas]),
            "policy_cpu_s": cpu, "max_decision_cpu_ms": 10.0}


def _evidence(protocol, pref, code_commit, manifest=None, per=None, extra=None, rehash=True, **over):
    m = manifest or _conf_manifest()
    per = per or {}
    rows = []
    for s in m["scenarios"]:
        res = {c: (per[c](s) if c in per else _result(c)) for c in ELIGIBLE}
        res.update(extra(s) if extra else {})
        rows.append({"scenario_id": s["scenario_id"], "family_id": s["family_id"], "scenario_hash": s["scenario_hash"],
                     "results": res})
    body = {"evidence_version": sel.EVIDENCE_VERSION, "evidence_class": sel.PROSPECTIVE,
            "protocol_id": protocol["protocol_id"], "protocol_hash": protocol["protocol_hash"],
            "protocol_commit": pref["commit"], "code_commit": code_commit, "manifest": m, "rows": rows,
            "complete": True, "errors": []}
    body.update(over)
    return {**body, "evidence_hash": digest(body) if rehash else "sha256:" + "0" * 64}


@pytest.fixture(scope="module")
def frozen_repo(tmp_path_factory):
    repo = Repo(tmp_path_factory.mktemp("sel") / "repo")
    p = _frozen()
    pref = repo.commit("selection/protocol.json", p)
    run = repo.commit("selection/run-marker.json", {"run": 1})["commit"]
    return repo, p, pref, run


def _gate(repo, p, pref, ev, historical=None):
    return sel.selection_gate(p, ev, protocol_ref=pref, repo=repo.path, historical=historical or _historical(),
                              trusted_ref="main")


def _no(g, match):
    assert g["decision"] == "NO_SELECTION" and g["selected"] is None and g["receipt_draft"] is None
    assert any(match in r for r in g["reasons"]), g["reasons"]


def test_gate_now_and_with_historical_dev_is_no_selection(frozen_repo):
    repo, p, pref, run = frozen_repo
    _no(_gate(repo, p, pref, None), "no prospective")
    _no(_gate(repo, p, pref, json.loads(sel.DEFAULT_REPORTS[1].read_text(encoding="utf-8"))), "historical DEV")
    _no(_gate(repo, p, pref, sel.evidence_inventory()), "historical DEV")
    relabelled = _evidence(p, pref, run, manifest={**_conf_manifest(), "plan_id": "p1-02-v1"})
    _no(_gate(repo, p, pref, relabelled), "historical DEV")
    _no(sel.selection_gate(None, None, protocol_ref=pref, repo=repo.path, historical={}), "protocol")
    _no(sel.selection_gate(sel.draft_protocol(), None, protocol_ref=pref, repo=repo.path, historical={}), "frozen")


def test_no_improvement_retains_t1_with_a_pending_receipt(frozen_repo):
    repo, p, pref, run = frozen_repo
    g = _gate(repo, p, pref, _evidence(p, pref, run))
    assert g["reasons"] == [] and g["decision"] == "RETAIN_T1_NO_IMPROVEMENT" and g["selected"] == "T1"
    r = g["receipt_draft"]
    assert r["review"]["status"] == "pending" and r["selected"]["name"] == "T1"
    assert r["eligibility"]["T4"] == {"status": "eligible", "identity": reg.identity("T4", "1.2.0").key()}
    assert all(not a["qualifies"] for a in g["analyses"].values())


def _better(c, cas=3, unres=1, evac=12, cpu=1.0):
    return lambda s: _result(c, cas=cas, unres=unres, evac=evac, cpu=cpu)


def test_superior_safe_feasible_candidate_is_proposed_and_aligns_with_s1(frozen_repo):
    repo, p, pref, run = frozen_repo
    g = _gate(repo, p, pref, _evidence(p, pref, run, per={"T3": _better("T3")}))
    assert g["decision"] == "SELECT_STANDALONE" and g["selected"] == "T3", g["reasons"]
    a = g["analyses"]["T3"]
    assert a["superior"] and a["ci95"][1] < 0 and a["sign_test_p"] < 0.05 and a["safety_pass"]
    draft = g["receipt_draft"]
    with pytest.raises(ValueError, match="not reviewed"):
        s1.verify_receipt(draft, p, pref)
    body = {k: v for k, v in draft.items() if k != "receipt_hash"}
    body["review"] = {"status": "approved", "reviewers": ["reviewer-a"]}
    reviewed = {**body, "receipt_hash": s1.receipt_hash(body)}
    assert s1.verify_receipt(reviewed, p, pref)["selected"]["name"] == "T3"


def test_s1_refuses_a_protocol_that_the_selection_validator_refuses(frozen_repo):
    repo, p, pref, run = frozen_repo
    weak = {"protocol_id": p["protocol_id"], "frozen": True, "eligibility": p["eligibility"]}
    weak["protocol_hash"] = s1.protocol_hash(weak)
    with pytest.raises(ValueError, match="selection protocol"):
        s1.verify_protocol(weak)


def test_two_qualifiers_prefer_lower_cpu_then_simplicity(frozen_repo):
    repo, p, pref, run = frozen_repo
    g = _gate(repo, p, pref, _evidence(p, pref, run, per={"T3": _better("T3", cpu=2.0),
                                                          "T0": _better("T0", cpu=1.0)}))
    assert g["selected"] == "T0"
    g = _gate(repo, p, pref, _evidence(p, pref, run, per={"T3": _better("T3"), "T2": _better("T2")}))
    assert g["selected"] == "T3"             # equal CPU: T3 before T2 in the frozen simplicity order
    g = _gate(repo, p, pref, _evidence(p, pref, run, per={"T4": _better("T4"), "T2": _better("T2")}))
    assert g["selected"] == "T4"             # equal CPU: T4 before T2
    g = _gate(repo, p, pref, _evidence(p, pref, run, per={"T4": _better("T4"), "T3": _better("T3")}))
    assert g["selected"] == "T3"             # equal CPU: T3 before T4
    g = _gate(repo, p, pref, _evidence(p, pref, run, per={"T4": _better("T4", cpu=0.5), "T3": _better("T3")}))
    assert g["selected"] == "T4"             # lower CPU first


def test_t4_v12_is_a_full_candidate_with_a_receipt_bound_to_its_code(frozen_repo):
    repo, p, pref, run = frozen_repo
    g = _gate(repo, p, pref, _evidence(p, pref, run, per={"T4": _better("T4")}))
    assert g["decision"] == "SELECT_STANDALONE" and g["selected"] == "T4", g["reasons"]
    sel_id = g["receipt_draft"]["selected"]
    assert sel_id["version"] == "1.2.0" and sel_id["config_hash"] == reg.identity("T4", "1.2.0").config_hash
    assert g["receipt_draft"]["review"]["status"] == "pending"


def test_missing_t4_v12_results_stop_selection(frozen_repo):
    repo, p, pref, run = frozen_repo
    ev = _evidence(p, pref, run)
    for r in ev["rows"][:1]:
        del r["results"]["T4"]
    body = {k: v for k, v in ev.items() if k != "evidence_hash"}
    _no(_gate(repo, p, pref, {**body, "evidence_hash": digest(body)}), "T4 result missing")


def test_casualty_gain_moved_into_unresolved_is_not_a_replacement(frozen_repo):
    repo, p, pref, run = frozen_repo
    g = _gate(repo, p, pref, _evidence(p, pref, run, per={"T2": _better("T2", cas=3, unres=4, evac=9)}))
    assert g["analyses"]["T2"]["superior"] and not g["analyses"]["T2"]["safety_pass"]
    assert g["decision"] == "RETAIN_T1_NO_IMPROVEMENT"


def test_mixed_family_results_are_not_superior(frozen_repo):
    repo, p, pref, run = frozen_repo
    half = lambda s: _result("T3", cas=3 if int(s["family_id"].split("-")[1]) % 2 else 7, evac=12)  # noqa: E731
    g = _gate(repo, p, pref, _evidence(p, pref, run, per={"T3": half}))
    assert not g["analyses"]["T3"]["superior"] and g["decision"] == "RETAIN_T1_NO_IMPROVEMENT"


def test_over_budget_candidate_is_not_a_replacement(frozen_repo):
    repo, p, pref, run = frozen_repo
    big = p["compute"]["replacement_cpu_s_cap"] / 72 + 1
    g = _gate(repo, p, pref, _evidence(p, pref, run, per={"T3": _better("T3", cpu=big)}))
    assert not g["analyses"]["T3"]["compute_feasible"] and g["decision"] == "RETAIN_T1_NO_IMPROVEMENT"


def test_blocked_t4_in_confirmation_evidence_stops_selection(frozen_repo, tmp_path):
    repo, p, pref, run = frozen_repo
    for v in ("1.0.0", "1.1.0"):
        ev = _evidence(p, pref, run, extra=lambda s, v=v: {"T4": {**_result("T1"), "config_hash":
                                                                  reg.identity("T4", v).config_hash}})
        _no(_gate(repo, p, pref, ev), "T4 identity differs from the protocol")
    repo2 = Repo(tmp_path / "repo")                 # a protocol that excludes T4 by rule
    p2 = _frozen(**_t4_out())
    pref2 = repo2.commit("selection/protocol.json", p2)
    run2 = repo2.commit("selection/run-marker.json", {"run": 1})["commit"]
    ev = _evidence(p2, pref2, run2)
    _no(_gate(repo2, p2, pref2, ev), "excluded candidate T4")


def test_evidence_before_or_beside_the_freeze_commit_is_refused(frozen_repo):
    repo, p, pref, run = frozen_repo
    root = repo.git("rev-list", "--max-parents=0", "HEAD")
    _no(_gate(repo, p, pref, _evidence(p, pref, root)), "does not descend")
    _no(_gate(repo, p, pref, _evidence(p, {**pref, "commit": root}, run)), "another protocol commit")
    _no(sel.selection_gate(p, _evidence(p, pref, run), protocol_ref={**pref, "path": "nope.json"},
                           repo=repo.path, historical=_historical(), trusted_ref="main"), "freeze commit")
    other = _frozen(protocol_id="other")
    _no(sel.selection_gate(other, _evidence(other, pref, run), protocol_ref=pref, repo=repo.path,
                           historical=_historical(), trusted_ref="main"), "differs from the committed")


def test_confirmation_must_be_family_disjoint_from_historical_sets(frozen_repo):
    repo, p, pref, run = frozen_repo
    m = _conf_manifest()
    leak = _historical(**{"p1-02-v1/dev": [{"family_id": m["scenarios"][0]["family_id"]}]})
    g = _gate(repo, p, pref, _evidence(p, pref, run, manifest=m), historical=leak)
    _no(g, "overlaps historical")
    assert g["isolation"]["overlap"]["p1-02-v1/dev"]["family"] == [m["scenarios"][0]["family_id"]]
    geo = _historical(**{"t4-pilot@2": [{"core_hash": m["scenarios"][5]["core_hash"]}]})
    _no(_gate(repo, p, pref, _evidence(p, pref, run, manifest=m), historical=geo), "overlaps historical")
    h = _historical()
    del h["t4-pilot@1"]
    _no(_gate(repo, p, pref, _evidence(p, pref, run), historical=h), "misses historical sets")


def _drop_last_row(p, pref, run):
    ev = _evidence(p, pref, run)
    ev = {**ev, "rows": ev["rows"][:-1]}
    ev["evidence_hash"] = digest({k: v for k, v in ev.items() if k != "evidence_hash"})
    return ev


BAD_EVIDENCE = {
    "incomplete rows": (_drop_last_row, "incomplete matched episodes"),
    "wrong design size": (lambda p, r, c: _evidence(p, r, c, manifest=_conf_manifest(fams=20)), "families x"),
    "identity drift": (lambda p, r, c: _evidence(p, r, c, per={"T2": lambda s: {**_result("T2"),
                                                                                "config_hash": "sha256:" + "9" * 64}}),
                       "identity differs"),
    "failure": (lambda p, r, c: _evidence(p, r, c, per={"T0": lambda s: {**_result("T0"), "failure": "timeout"}}),
                "failed"),
    "tampered": (lambda p, r, c: _evidence(p, r, c, rehash=False), "evidence hash"),
    "incomplete flag": (lambda p, r, c: _evidence(p, r, c, complete=False), "incomplete"),
    "other protocol": (lambda p, r, c: _evidence({**p, "protocol_hash": "sha256:" + "5" * 64}, r, c),
                       "another protocol"),
    "malformed result": (lambda p, r, c: _evidence(p, r, c, per={"T1": lambda s: {"casualties": 1}}),
                         "missing or malformed"),
    "decision cpu": (lambda p, r, c: _evidence(p, r, c, per={"T2": lambda s: {**_result("T2"),
                                                                              "max_decision_cpu_ms": 1e9}}),
                     "decision CPU"),
}


@pytest.mark.parametrize("case", sorted(BAD_EVIDENCE))
def test_incomplete_or_inconsistent_confirmation_is_refused(frozen_repo, case):
    repo, p, pref, run = frozen_repo
    make, match = BAD_EVIDENCE[case]
    _no(_gate(repo, p, pref, make(p, pref, run)), match)


# ----- arbiter contract ---------------------------------------------------------------------------------------
def _swap_obs():
    o = Simulator(ascii_scenario(["0.ab.1"])).observe()
    pos = {q["id"]: tuple(q["position"]) for q in o["people"]}
    left, right = sorted(pos, key=lambda i: pos[i][1])
    return o, left, right


def test_arbiter_never_selects_a_map_that_fails_the_resolver():
    o, left, right = _swap_obs()
    fb = pr.proposal(T0ShortestPath().decide(o))
    bad = {"label": "mixed", "actions": {left: "east", right: "west"}, "score": {"score_id": "s", "value": 9.0}}
    good = {"label": "T1", "actions": {left: "west", right: "east"}, "score": {"score_id": "s", "value": 1.0}}
    r = arb.select_joint_map(o, [bad, good], score_id="s", fallback=fb)
    assert r["chosen"] == "T1" and r["rejected"][0]["reason"] == "resolver blocked moves"
    r = arb.select_joint_map(o, [bad], score_id="s", fallback=fb)
    assert r["fallback_used"] and r["chosen"] == "fallback:T0"
    assert pr.joint_check(o, r["actions"])["valid_joint_plan"]
    with pytest.raises(arb.ArbiterError, match="fails the resolver"):
        arb.accept_joint_map(o, {"actions": bad["actions"]})
    assert arb.accept_joint_map(o, {"actions": good["actions"]}) == good["actions"]


def test_arbiter_refuses_unlike_scores_and_partial_maps():
    o, left, right = _swap_obs()
    fb = pr.proposal(T0ShortestPath().decide(o))
    a = {"label": "a", "actions": {left: "west", right: "east"}, "score": {"score_id": "s", "value": 1.0}}
    b = {"label": "b", "actions": {left: "west", right: "east"}, "score": {"score_id": "t2_mc_value", "value": 2.0}}
    with pytest.raises(arb.ArbiterError, match="never mixed or averaged"):
        arb.select_joint_map(o, [a, b], score_id="s", fallback=fb)
    part = {"label": "p", "actions": {left: "west"}, "score": {"score_id": "s", "value": 5.0}}
    assert arb.select_joint_map(o, [part, a], score_id="s", fallback=fb)["chosen"] == "a"
    with pytest.raises(arb.ArbiterError, match="non-numeric"):
        arb.select_joint_map(o, [{**a, "score": {"score_id": "s", "value": float("nan")}}], score_id="s",
                             fallback=fb)


def _arb_protocol(**over):
    body = {"protocol_version": arb.ARBITER_PROTOCOL_VERSION, "protocol_id": "arb", "score_id": "s",
            "candidates": ["T1", "T3"], "fallback_teacher": "T1", "observation_only": True,
            "independent_teacher_sampling": True, "equal_total_planning_cpu": True, "whole_map_only": True,
            "joint_resolver_valid_required": True, "no_score_averaging": True, "frozen": True}
    body.update(over)
    return {**body, "protocol_hash": digest(body)}


@pytest.mark.parametrize("over,match", [
    ({"no_score_averaging": False}, "no_score_averaging"), ({"whole_map_only": False}, "whole_map_only"),
    ({"joint_resolver_valid_required": False}, "joint_resolver"), ({"equal_total_planning_cpu": False}, "equal"),
    ({"observation_only": False}, "observation_only"), ({"candidates": ["T1", "T4"]}, "eligible"),
    ({"fallback_teacher": "T0"}, "fallback"), ({"score_id": ""}, "score_id"), ({"frozen": False}, "frozen"),
])
def test_arbiter_protocol_needs_every_safeguard(over, match):
    assert arb.validate_arbiter_protocol(_arb_protocol(), eligible=ARB_ELIGIBLE)["protocol_id"] == "arb"
    with pytest.raises(arb.ArbiterError, match=match):
        arb.validate_arbiter_protocol(_arb_protocol(**over), eligible=ARB_ELIGIBLE)
    with pytest.raises(arb.ArbiterError, match="hash"):
        arb.validate_arbiter_protocol({**_arb_protocol(), "score_id": "other"}, eligible=ARB_ELIGIBLE)
