"""P1-09 prospective confirmation set (p1-09-selection@1): regeneration, isolation, preflight, runner.

No test runs a teacher on a confirmation scenario: the runner is exercised on small ASCII scenarios only.
"""
from __future__ import annotations

import json

import pytest

from ember_v1.sim.canonical import digest
from ember_v1.teacher import confirmation as C
from ember_v1.teacher import registry as reg
from ember_v1.teacher import selection as sel
from test_s1_gate import Repo
from test_teacher_t0 import ascii_scenario


def _committed(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rebuilt():
    entries, fams, prov = C.historical_sets()
    m = C.build_manifest(entries, fams)
    return entries, fams, prov, m, C.isolation_report(m, entries, prov)


# ----- dataset -------------------------------------------------------------------------------------------------
def test_committed_manifest_and_audit_regenerate_exactly(rebuilt):
    *_, m, audit = rebuilt
    assert m == _committed(C.MANIFEST_PATH)
    assert audit == _committed(C.AUDIT_PATH)


def test_isolation_audit_passes_against_every_required_historical_set(rebuilt):
    entries, _, prov, m, audit = rebuilt
    assert set(C.HISTORICAL) == set(sel.REQUIRED_ISOLATION) == set(entries) == set(audit["historical_sets"])
    assert audit["pass"] and all(v == [] for o in audit["overlap"].values() for v in o.values())
    assert audit["families"] == C.FAMILIES and audit["lineages_per_family"] == [C.LINEAGES]
    assert audit["scenarios"] == C.FAMILIES * C.LINEAGES and all(audit["unique"].values())
    assert set(prov["sets"]) == set(C.HISTORICAL) and prov["reference_families"] > 0


def test_manifest_uses_a_fresh_namespace_and_records_every_exclusion():
    m = _committed(C.MANIFEST_PATH)
    assert (m["plan_id"], m["split"], m["plan_hash"]) == (C.NAMESPACE, C.SPLIT, C.plan_hash())
    assert m["manifest_hash"] == digest({k: v for k, v in m.items() if k != "manifest_hash"})
    assert m["manifest_hash"] == sel.draft_protocol()["confirmation"]["manifest_hash"]
    slots = sorted([f["pool_slot"] for f in m["families"]] + [x["pool_slot"] for x in m["excluded_families"]]
                   + [x["pool_slot"] for x in m["pool"]["family_failures"]])
    assert slots == list(range(m["pool"]["slots_used"]))          # no slot silently dropped
    assert all(x["reason"] in ("lineage_generation_failed", "isolation_collision") for x in m["excluded_families"])


def test_materialize_regenerates_every_scenario_and_refuses_tampering():
    m = _committed(C.MANIFEST_PATH)
    got = C.materialize(m)
    assert [e["scenario_id"] for e, _ in got] == [s["scenario_id"] for s in m["scenarios"]]
    assert all(s.content_hash() == e["scenario_hash"] for e, s in got)
    sc = [dict(s) for s in m["scenarios"]]
    sc[0]["lineage_seed"] += 1
    body = {k: v for k, v in m.items() if k != "manifest_hash"} | {"scenarios": sc}
    with pytest.raises(C.ConfirmationError, match="differs from the manifest"):
        C.materialize({**body, "manifest_hash": digest(body)})
    body = {k: v for k, v in m.items() if k != "manifest_hash"} | {"plan_hash": "sha256:" + "0" * 64}
    with pytest.raises(C.ConfirmationError, match="preregistered"):
        C.materialize({**body, "manifest_hash": digest(body)})


def test_a_historical_collision_excludes_the_whole_family():
    small = C.build_manifest({}, [], families=2, max_slots=8)
    first = small["scenarios"][0]
    hit = C.build_manifest({"fake-set": [{k: first[k] for k in C.ID_KEYS}]}, [], families=2, max_slots=8)
    ex = [x for x in hit["excluded_families"] if x["reason"] == "isolation_collision"]
    assert ex and ex[0]["family_id"] == first["family_id"] and ex[0]["historical_set"] == "fake-set"
    assert first["family_id"] not in {s["family_id"] for s in hit["scenarios"]}
    with pytest.raises(C.ConfirmationError, match="not filled"):
        C.build_manifest({}, [], families=2, max_slots=1)


# ----- protocol binding and compute -----------------------------------------------------------------------------
def test_draft_protocol_lists_five_identities_and_their_frozen_seeds():
    p = sel.draft_protocol()
    elig = [c for c in reg.CANDIDATES if p["eligibility"][c]["status"] == reg.ELIGIBLE]
    assert elig == list(reg.CANDIDATES)
    assert p["confirmation"]["policy_seeds"] == sel.policy_seeds() == {"T2": 20261107, "T3": 20261123}
    assert p["confirmation"]["namespace"] == C.NAMESPACE and p["frozen"] is False


def test_committed_protocol_candidate_is_the_frozen_draft():
    committed = _committed(C.OUT_DIR / "protocol.json")
    assert committed == sel.freeze_protocol({**sel.draft_protocol(), "frozen": True})
    assert sel.verify_frozen_protocol(committed)["confirmation"]["manifest_hash"] == \
        _committed(C.MANIFEST_PATH)["manifest_hash"]
    assert C.compute_preflight(committed)["pass"]


def test_compute_preflight_covers_every_eligible_teacher_and_flags_t2():
    pre = C.compute_preflight(sel.draft_protocol())
    assert pre["eligible"] == list(reg.CANDIDATES) and pre["missing_estimates"] == []
    assert pre["pass"] and pre["eligible_total_s"] <= pre["cpu_s_cap"]
    # Upper estimate only: T2 may exceed the replacement cap; the gate decides on measured CPU.
    assert pre["over_replacement_cap"] == ["T2"]
    assert max(pre["estimated_cpu_s"], key=pre["estimated_cpu_s"].get) == "T2"


def test_compute_preflight_fails_when_the_total_cap_is_too_small():
    p = sel.draft_protocol()
    p["compute"] = {**p["compute"], "cpu_s_cap": 100.0}
    assert not C.compute_preflight(p)["pass"]


# ----- runner on small non-confirmation scenarios ----------------------------------------------------------------
def _tiny():
    rows = [["0..a.", "....b", ".*..."], ["#0#..", "a...*", "..b.1"]]
    out = []
    for i, r in enumerate(rows):
        s = ascii_scenario(r, interval=2, max_steps=12, seed=900 + i, scenario_id=f"conf-test-{i}")
        out.append(({"scenario_id": s.scenario_id}, s))
    return out


def _identity(c):
    return dict(sel.draft_protocol()["eligibility"][c]["identity"])


@pytest.mark.parametrize("c", reg.CANDIDATES)
def test_run_teacher_is_matched_and_repeatable(c):
    seed = sel.policy_seeds().get(c)
    a = C.run_teacher(_identity(c), seed, _tiny())
    b = C.run_teacher(_identity(c), seed, _tiny())
    outcome = lambda r: {k: {f: v for f, v in x.items() if f not in ("policy_cpu_s", "max_decision_cpu_ms")}  # noqa: E731
                         for k, x in r["results"].items()}
    assert outcome(a) == outcome(b) and a["errors"] == [] and set(a["results"]) == {"conf-test-0", "conf-test-1"}
    for r in a["results"].values():
        assert r["config_hash"] == _identity(c)["config_hash"] and r["failure"] is None
        assert r["max_decision_cpu_ms"] >= 0 and r["policy_cpu_s"] >= 0


def test_run_teacher_refuses_a_wrong_seed_or_identity():
    with pytest.raises(C.ConfirmationError, match="seed or identity"):
        C.run_teacher(_identity("T2"), None, _tiny())     # T2 would fall back to its own default seed
    with pytest.raises(C.ConfirmationError, match="seed or identity"):
        C.run_teacher({**_identity("T0"), "config_hash": "sha256:" + "9" * 64}, None, _tiny())


# ----- run preconditions (no teacher outcome before every check passes) -----------------------------------------
def _frozen_protocol():
    return sel.freeze_protocol({**sel.draft_protocol(), "frozen": True})


@pytest.fixture
def repo(tmp_path):
    r = Repo(tmp_path / "repo")
    pref = r.commit("selection/protocol.json", _frozen_protocol())
    mref = r.commit("selection/manifest.json", _committed(C.MANIFEST_PATH))
    return r, pref, mref, tmp_path / "out" / "evidence.json"


def _pre(r, pref, mref, ev_path):
    return C.run_preconditions(r.path, pref, mref, trusted_ref="main", evidence_path=ev_path)


def test_preconditions_pass_for_the_frozen_protocol_and_its_manifest(repo):
    r, pref, mref, ev_path = repo
    got = _pre(r, pref, mref, ev_path)
    assert got["protocol"]["confirmation"]["manifest_hash"] == got["manifest"]["manifest_hash"]
    assert got["code_commit"] == r.git("rev-parse", "HEAD")


def test_preconditions_refuse_an_unfrozen_or_untrusted_protocol(repo):
    r, _, mref, ev_path = repo
    draft = r.commit("selection/draft.json", sel.draft_protocol())
    with pytest.raises(C.ConfirmationError, match="protocol: "):
        _pre(r, draft, mref, ev_path)
    r.git("checkout", "-q", "-b", "side")
    side = r.commit("selection/side.json", _frozen_protocol())
    r.git("checkout", "-q", "main")
    with pytest.raises(C.ConfirmationError, match="protocol: "):
        _pre(r, side, mref, ev_path)


def test_preconditions_refuse_another_manifest(repo):
    r, pref, _, ev_path = repo
    other = r.commit("selection/other.json", C.build_manifest({}, [], families=2, max_slots=8))
    with pytest.raises(C.ConfirmationError, match="not the one preregistered"):
        _pre(r, pref, other, ev_path)


def test_preconditions_refuse_a_dirty_tree_an_old_head_and_existing_evidence(repo):
    r, pref, mref, ev_path = repo
    (r.path / "selection/manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(C.ConfirmationError, match="uncommitted"):
        _pre(r, pref, mref, ev_path)
    r.git("checkout", "-q", "--", ".")
    r.git("checkout", "-q", "--detach", "HEAD~2")
    with pytest.raises(C.ConfirmationError, match="does not descend"):
        _pre(r, pref, mref, ev_path)
    r.git("checkout", "-q", "main")
    ev_path.parent.mkdir(parents=True)
    ev_path.write_text("{}", encoding="utf-8")
    with pytest.raises(C.ConfirmationError, match="evidence already exists"):
        _pre(r, pref, mref, ev_path)


def test_run_stops_before_any_teacher_when_a_precondition_fails(repo, monkeypatch):
    r, _, mref, ev_path = repo
    draft = r.commit("selection/draft.json", sel.draft_protocol())

    def boom(*a, **k):
        raise AssertionError("a teacher ran before the preconditions passed")
    monkeypatch.setattr(C, "run_rows", boom)
    with pytest.raises(C.ConfirmationError):
        C.run(r.path, draft, mref, trusted_ref="main", evidence_path=ev_path)
    assert not ev_path.exists()


def test_run_writes_evidence_once(repo, monkeypatch):
    r, pref, mref, ev_path = repo
    m = _committed(C.MANIFEST_PATH)
    rows = [{"scenario_id": s["scenario_id"], "family_id": s["family_id"], "scenario_hash": s["scenario_hash"],
             "results": {}} for s in m["scenarios"]]
    monkeypatch.setattr(C, "run_rows", lambda p, man: (rows, [], {}))      # synthetic: no teacher outcome
    out = C.run(r.path, pref, mref, trusted_ref="main", evidence_path=ev_path)
    doc = json.loads(ev_path.read_text(encoding="utf-8"))
    assert out["evidence"] == str(ev_path) and doc["evidence_class"] == sel.PROSPECTIVE
    assert doc["protocol_commit"] == pref["commit"] and doc["manifest"]["manifest_hash"] == m["manifest_hash"]
    assert doc["evidence_hash"] == digest({k: v for k, v in doc.items() if k != "evidence_hash"})
    with pytest.raises(C.ConfirmationError, match="evidence already exists"):
        C.run(r.path, pref, mref, trusted_ref="main", evidence_path=ev_path)
