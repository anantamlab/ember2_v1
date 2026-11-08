"""Teacher provenance and trust gates (issue #10): honest records verify; every forgery or tampering fails closed."""
from __future__ import annotations

import random

import pytest

from ember_v1 import provenance as pv
from ember_v1.gen.spec import GENERATOR_REVISION
from ember_v1.sim import Simulator
from ember_v1.sim.canonical import digest
from ember_v1.teacher.t0 import T0ShortestPath
from ember_v1.teacher.t1 import T1CongestionAware
from ember_v1.teacher.t2 import T2StochasticRollout
from gen_helpers import random_scenario


def _manifest(scenarios, split="train", **over):
    body = {"manifest_version": "1.0", "split": split, "plan_id": "test-plan", "plan_hash": "sha256:" + "1" * 64,
            "generator_revision": GENERATOR_REVISION, "spec_hash": "sha256:" + "2" * 64, "families": [],
            "scenarios": [{"scenario_id": s.scenario_id, "scenario_hash": s.content_hash(), "lineage_id": f"L-{i}",
                           "family_id": f"F-{split}-{i}"} for i, s in enumerate(scenarios)],
            "failed_lineages": [], "lineage_rejections": {}}
    body.update(over)
    return {**body, "manifest_hash": digest(body)}


@pytest.fixture(scope="module")
def world():
    s = random_scenario(random.Random(77))
    obs = Simulator(s).observe()
    m = _manifest([s])
    return s, obs, m


def _make(world, cls=T0ShortestPath, seed=None):
    s, obs, m = world
    pol = cls() if seed is None else cls(seed=seed)
    d = pol.decide(obs)
    rec = pv.provenance_record(pol, d, manifest=m, scenario_id=s.scenario_id)
    return rec, d


def _verify(world, rec, d=None, *, actions=None, values="same", obs=None, scenario_hash="same", manifests=None,
            allowed=("train",)):
    s, o, m = world
    vals = pv.decision_values(rec["teacher"]["name"], d) if values == "same" else values
    return pv.verify_provenance(rec, observation=obs or o, actions=actions or d.actions, values=vals,
                                scenario_hash=s.content_hash() if scenario_hash == "same" else scenario_hash,
                                manifests=manifests or {"train": m}, allowed_splits=allowed)


def _rehash(rec, **changes):
    body = {k: v for k, v in rec.items() if k != "record_hash"}
    for k, v in changes.items():
        body[k] = v
    return {**body, "record_hash": digest(body)}


def test_honest_records_verify_for_action_only_teachers(world):
    for cls in (T0ShortestPath, T1CongestionAware):
        rec, d = _make(world, cls)
        assert rec["values_digest"] is None
        assert _verify(world, rec, d)["status"] == "verified"


def test_honest_t2_record_binds_its_real_values(world):
    rec, d = _make(world, T2StochasticRollout, seed=5)
    vals = pv.decision_values("T2", d)
    assert vals and rec["values_digest"] == digest(vals)
    assert _verify(world, rec, d)["status"] == "verified"
    bad = {pid: dict(v) for pid, v in vals.items()}
    pid = next(iter(bad))
    k = next(a for a, x in bad[pid].items() if isinstance(x, (int, float)))
    bad[pid][k] = bad[pid][k] + 1.0
    with pytest.raises(pv.ProvenanceError, match="values do not match"):
        _verify(world, rec, d, values=bad)
    with pytest.raises(pv.ProvenanceError, match="values do not match"):
        _verify(world, rec, d, values=None)


def test_code_fingerprint_is_stable_and_teacher_specific():
    assert pv.code_fingerprint(T0ShortestPath) == pv.code_fingerprint(T0ShortestPath)
    assert pv.code_fingerprint(T0ShortestPath) != pv.code_fingerprint(T1CongestionAware)
    mods = set()
    todo = ["ember_v1.teacher.t1"]
    while todo:
        m = todo.pop()
        if m not in mods and pv._module_path(m):
            mods.add(m)
            todo.extend(pv._imports(m, pv._module_path(m)))
    assert {"ember_v1.teacher.t1", "ember_v1.teacher.grid", "ember_v1.teacher.interface"} <= mods
    assert "ember_v1.teacher.t3" not in mods


@pytest.mark.parametrize("field,value,match", [
    ("teacher", {"name": "T9", "policy": "x", "version": "1.0.0", "config_hash": "sha256:0"}, "unknown teacher"),
    ("code_fingerprint", "sha256:" + "0" * 64, "code fingerprint"),
    ("utilities", {"interface": "0", "grid": "0"}, "utility"),
    ("observation_digest", "sha256:" + "0" * 64, "observation"),
    ("actions_digest", "sha256:" + "0" * 64, "actions"),
    ("values_digest", "sha256:" + "0" * 64, "fabricated"),
    ("scenario", {"scenario_id": "nope", "scenario_hash": "x", "lineage_id": "x", "family_id": "x"}, "not in"),
    ("policy_seed", 5, "seed"),
    ("provenance_version", "teacher-provenance@0", "version"),
])
def test_forged_and_rehashed_fields_fail(world, field, value, match):
    rec, d = _make(world)
    with pytest.raises(pv.ProvenanceError, match=match):
        _verify(world, _rehash(rec, **{field: value}), d)


def test_forged_identity_fields_fail(world):
    rec, d = _make(world)
    for k, v in (("config_hash", "sha256:" + "0" * 64), ("version", "9.9.9"), ("policy", "other")):
        with pytest.raises(pv.ProvenanceError, match="identity differs|unknown teacher version"):
            _verify(world, _rehash(rec, teacher={**rec["teacher"], k: v}), d)
    with pytest.raises(pv.ProvenanceError, match="identity differs"):     # T0 label on a T1 decision
        t1 = _make(world, T1CongestionAware)[0]
        _verify(world, _rehash(t1, teacher={**t1["teacher"], "name": "T0"}), d)


def test_record_tampering_without_rehash_fails(world):
    rec, d = _make(world)
    with pytest.raises(pv.ProvenanceError, match="record hash"):
        _verify(world, {**rec, "policy_seed": 1}, d)
    with pytest.raises(pv.ProvenanceError, match="missing or extra"):
        _verify(world, {**rec, "extra": 1}, d)


def test_tampered_trace_fails(world):
    s, obs, m = world
    rec, d = _make(world)
    pid = sorted(d.actions)[0]
    other = next(a for a in obs["allowed_actions"] if a != d.actions[pid])
    with pytest.raises(pv.ProvenanceError, match="actions do not match"):
        _verify(world, rec, d, actions={**d.actions, pid: other})
    obs2 = Simulator(random_scenario(random.Random(78))).observe()
    with pytest.raises(pv.ProvenanceError, match="observation"):
        _verify(world, rec, d, obs=obs2, actions=T0ShortestPath().act(obs2))


def test_fabricated_values_for_an_action_only_teacher_fail(world):
    rec, d = _make(world)
    fake = {pid: {a: 0.0 for a in ("stay", "up")} for pid in d.actions}
    with pytest.raises(pv.ProvenanceError, match="fabricated"):
        _verify(world, rec, d, values=fake)


def test_manifest_binding_fails_closed(world):
    s, obs, m = world
    rec, d = _make(world)
    with pytest.raises(pv.ProvenanceError, match="manifest hash"):
        _verify(world, rec, d, manifests={"train": {**m, "plan_id": "other"}})
    m2 = _manifest([s], plan_id="other-plan")
    with pytest.raises(pv.ProvenanceError, match="different manifest"):
        _verify(world, rec, d, manifests={"train": m2})
    m3 = _manifest([s], generator_revision="gen-0.0.1")
    with pytest.raises(pv.ProvenanceError, match="generator revision"):
        _verify(world, rec, d, manifests={"train": m3})
    m4 = _manifest([random_scenario(random.Random(79))])
    with pytest.raises(pv.ProvenanceError):
        _verify(world, rec, d, manifests={"train": m4})
    with pytest.raises(pv.ProvenanceError, match="trace scenario"):
        _verify(world, rec, d, scenario_hash="sha256:" + "0" * 64)
    with pytest.raises(pv.ProvenanceError, match="no manifest"):
        _verify(world, rec, d, manifests={"dev": m})


def test_split_rules(world):
    s, obs, m = world
    dev = _manifest([s], split="dev")
    pol = T0ShortestPath()
    d = pol.decide(obs)
    rec = pv.provenance_record(pol, d, manifest=dev, scenario_id=s.scenario_id)
    with pytest.raises(pv.ProvenanceError, match="not allowed"):
        _verify(world, rec, d, manifests={"dev": dev})
    assert _verify(world, rec, d, manifests={"dev": dev}, allowed=("dev",))["split"] == "dev"
    locked = _manifest([s], split="locked_test")
    with pytest.raises(pv.ProvenanceError, match="locked"):
        pv.provenance_record(pol, d, manifest=locked, scenario_id=s.scenario_id)
    rec_l = _rehash(rec, split={**rec["split"], "split": "locked_test"})
    with pytest.raises(pv.ProvenanceError, match="not allowed"):
        _verify(world, rec_l, d, manifests={"locked_test": locked}, allowed=("train", "locked_test"))


def test_unregistered_teacher_class_cannot_create_records(world):
    s, obs, m = world

    class FakeT0(T0ShortestPath):
        def _plan(self, o):
            return {p["id"]: "stay" for p in o["people"] if p["status"] == "active"}, {}
    pol = FakeT0()
    with pytest.raises(pv.ProvenanceError, match="registered"):
        pv.provenance_record(pol, pol.decide(obs), manifest=m, scenario_id=s.scenario_id)


def test_consistent_forgery_passes_hashes_but_fails_regeneration(world):
    """A forger who writes a self-consistent record for fake actions is caught only by regeneration."""
    s, obs, m = world
    rec, d = _make(world)
    pid = sorted(d.actions)[0]
    fake = {**d.actions, pid: next(a for a in obs["allowed_actions"] if a != d.actions[pid])}
    forged = _rehash(rec, actions_digest=digest(fake))
    assert _verify(world, forged, d, actions=fake)["status"] == "verified"
    honest = {"record": rec, "observation": obs, "actions": d.actions, "values": None}
    bad = {"record": forged, "observation": obs, "actions": fake, "values": None}
    assert pv.regeneration_spot_check([honest], k=1, seed="t")["pass"] is True
    r = pv.regeneration_spot_check([honest, bad], k=2, seed="t")
    assert r["pass"] is False and r["mismatches"] == [{"index": 1, "reason": "actions differ on regeneration"}]


def test_spot_check_needs_a_sample_and_is_seeded(world):
    with pytest.raises(ValueError):
        pv.regeneration_spot_check([], k=0, seed="x")
    assert pv.regeneration_spot_check([], k=3, seed="x")["pass"] is False
    s, obs, m = world
    rec, d = _make(world)
    items = [{"record": rec, "observation": obs, "actions": d.actions, "values": None}] * 6
    a = pv.regeneration_spot_check(items, k=3, seed="x")["sampled"]
    assert a == pv.regeneration_spot_check(items, k=3, seed="x")["sampled"] and len(a) == 3


def _rec(split, fam, lin="L", obs="o", act="a", sh="s", teacher="T0"):
    return {"split": {"split": split}, "scenario": {"scenario_hash": sh, "lineage_id": lin, "family_id": fam},
            "observation_digest": obs, "actions_digest": act, "teacher": {"name": teacher}}


def test_leakage_audit():
    ok = pv.leakage_audit([_rec("train", "F1", "L1", "o1", sh="s1"), _rec("dev", "F2", "L2", "o2", sh="s2")])
    assert ok["pass"] and ok["duplicate_records"] == 0
    leak = pv.leakage_audit([_rec("train", "F1", "L1", "o1", sh="s1"), _rec("dev", "F1", "L2", "o2", sh="s2")])
    assert not leak["pass"] and leak["cross_split_leaks"]["family"] == ["F1"]
    obs_leak = pv.leakage_audit([_rec("train", "F1", "L1", "o", sh="s1"), _rec("dev", "F2", "L2", "o", sh="s2")])
    assert obs_leak["cross_split_leaks"]["observation"] == ["o"]
    dup = pv.leakage_audit([_rec("train", "F1"), _rec("train", "F1")])
    assert not dup["pass"] and dup["duplicate_records"] == 1
    assert pv.leakage_audit([])["pass"] is False


def _selection(**over):
    sel = {"name": "T1", "version": "1.0.0"}
    sel.update(over)
    return {"status": "verified", "selected": sel}


ALL_DEPS = {d: True for d in pv.REQUIRED_DEPENDENCIES}
GOOD = dict(dependencies=ALL_DEPS, provenance_failures=0, records_checked=100,
            spot_check={"pass": True, "checked": 50}, leakage={"pass": True})


def test_bulk_s1_is_blocked_now_and_only_authorized_when_every_gate_holds():
    now = pv.bulk_s1_gate(selection=None, dependencies={}, provenance_failures=0, records_checked=0,
                          spot_check=None, leakage=None)
    assert now["decision"] == "BULK_S1_BLOCKED" and len(now["reasons"]) >= 5
    assert pv.bulk_s1_gate(selection=_selection(), **GOOD)["decision"] == "BULK_S1_AUTHORIZED"
    assert "t4_24" not in pv.REQUIRED_DEPENDENCIES and "t4_24_resolved" in pv.REQUIRED_DEPENDENCIES
    bad_cases = [
        dict(selection=None),
        dict(selection={**_selection(), "status": "unverified"}),
        dict(selection=_selection(name="T9")),
        dict(selection=_selection(name="T4", version="1.1.0")),          # blocked T4 v1.1.0 (G4)
        dict(selection=_selection(name="T4", version="1.0.0")),          # blocked T4 v1.0.0 (G5)
        dict(selection=_selection(name="T4", version=None)),
        dict(selection=_selection(), dependencies={**ALL_DEPS, "t4_24_resolved": False}),
        dict(selection=_selection(), provenance_failures=1),
        dict(selection=_selection(), records_checked=0),
        dict(selection=_selection(), spot_check={"pass": True, "checked": 49}),
        dict(selection=_selection(), spot_check={"pass": False, "checked": 500}),
        dict(selection=_selection(), leakage={"pass": False}),
    ]
    for case in bad_cases:
        g = pv.bulk_s1_gate(**{**GOOD, **case})
        assert g["decision"] == "BULK_S1_BLOCKED" and g["reasons"], case


def test_blocked_t4_never_creates_or_verifies_records(world):
    from ember_v1.teacher.t4_v11 import T4RiskPriorityV11
    s, obs, m = world
    pol = T4RiskPriorityV11()
    with pytest.raises(pv.ProvenanceError, match="blocked"):
        pv.provenance_record(pol, pol.decide(obs), manifest=m, scenario_id=s.scenario_id)
    rec, d = _make(world)
    t4 = _rehash(rec, teacher={"name": "T4", "policy": "risk_based_priority", "version": "1.1.0",
                               "config_hash": pol.identity.config_hash})
    with pytest.raises(pv.ProvenanceError, match="blocked"):
        _verify(world, t4, d)
    with pytest.raises(pv.ProvenanceError, match="blocked"):
        pv.frozen_teacher("T4", None)
    assert "T4" not in pv.REGISTRY and pv.VALUE_TEACHERS == frozenset({"T2"})


# ----- ingestion (sources.ingest_zip) ---------------------------------------------------------------------
def _t1_traces(s, m):
    pol = T1CongestionAware()
    sim, traces, recs, obs_list = Simulator(s), [], [], []
    while not sim.terminated:
        obs = sim.observe()
        d = pol.decide(obs)
        obs_list.append((obs, d))
        recs.append(sim.step(d.actions))
    ep = {"scenario": s.to_dict(), "seed": None, "records": recs}
    for k, (obs, d) in enumerate(obs_list):
        traces.append({"observation": obs, "actions": d.actions, "group_id": "ep", "source": "T1", "episode": ep,
                       "step": k + 1, "provenance": pv.provenance_record(pol, d, manifest=m,
                                                                         scenario_id=s.scenario_id)})
    return traces


def _zipped(tmp_path, traces):
    import json
    import zipfile
    p = tmp_path / "s1.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("t.json", json.dumps({"traces": traces}))
    return p


def test_ingest_binds_provenance_and_training_requires_it(tmp_path, world):
    from ember_v1.sources import ingest_zip, training_records
    s, obs, m = world
    traces = _t1_traces(s, m)
    recs, _ = ingest_zip(_zipped(tmp_path, traces), source_id="s1", quality="teacher_verified",
                         manifests={"train": m})
    assert all(r["provenance"]["status"] == "verified" for r in recs)
    assert training_records(recs, require_provenance=True) == recs
    plain, _ = ingest_zip(_zipped(tmp_path, traces), source_id="s1", quality="teacher_verified")
    with pytest.raises(ValueError, match="provenance"):
        training_records(plain, require_provenance=True)
    with pytest.raises(ValueError, match="fixtures"):
        training_records(recs, allow_fixtures=True, require_provenance=True)
    no_prov = [{k: v for k, v in t.items() if k != "provenance"} for t in traces]
    with pytest.raises(ValueError, match="provenance record required"):
        ingest_zip(_zipped(tmp_path, no_prov), source_id="s1", quality="teacher_verified", manifests={"train": m})
    forged = [dict(t) for t in traces]
    forged[0]["provenance"] = _rehash(forged[0]["provenance"], teacher={**forged[0]["provenance"]["teacher"],
                                                                        "name": "T0"})
    with pytest.raises(ValueError, match="identity differs"):
        ingest_zip(_zipped(tmp_path, forged), source_id="s1", quality="teacher_verified", manifests={"train": m})
