"""P1-09 prospective confirmation set (p1-09-selection@1): regeneration, isolation, preflight, runner.

No test runs a teacher on a confirmation scenario: the runner is exercised on small ASCII scenarios only.
"""
from __future__ import annotations

import json

import pytest
# Slow tier (AGENTS.md): skipped unless --run-slow.
pytestmark = pytest.mark.slow

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
    pilots = C.pilot_maps()
    m = C.build_manifest(entries, fams, pilots=pilots)
    return entries, fams, prov, m, C.isolation_report(m, entries, prov, pilots)


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
    assert all(x["reason"] in ("lineage_generation_failed", "isolation_collision", "pilot_map_duplicate")
               for x in m["excluded_families"])


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
        assert set(r) == sel.RESULT_KEYS and 0 <= r["budget_limited_decisions"] <= r["decisions"]


def test_budget_limited_decisions_are_recognised_for_every_teacher_format():
    assert C.budget_limited({"budget_exhausted": True})                         # T1
    assert C.budget_limited({"budget": {"limit_hit": "rollouts"}})              # T2 / T3 / T4
    assert C.budget_limited({"status": "budget_limited"})                       # T3
    assert not C.budget_limited({"budget": {"limit_hit": None}, "budget_exhausted": False, "status": "complete"})
    assert not C.budget_limited(None)


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
    monkeypatch.setattr(C, "run_rows", lambda p, man, **k: (rows, [], _report()))    # synthetic: no outcome
    out = C.run(r.path, pref, mref, trusted_ref="main", evidence_path=ev_path)
    doc = json.loads(ev_path.read_text(encoding="utf-8"))
    assert out["evidence"] == str(ev_path) and doc["evidence_class"] == sel.PROSPECTIVE
    assert out["complete"] is False and doc["complete"] is False          # rows without results are partial
    assert doc["protocol_commit"] == pref["commit"] and doc["manifest"]["manifest_hash"] == m["manifest_hash"]
    assert doc["evidence_hash"] == digest({k: v for k, v in doc.items() if k != "evidence_hash"})
    with pytest.raises(C.ConfirmationError, match="evidence already exists"):
        C.run(r.path, pref, mref, trusted_ref="main", evidence_path=ev_path)


# ----- pilot-map near-duplicate screen (geometry only; no outcome) ----------------------------------------------
def _map(fam, h=None, w=None, walls=None, exits=None, sid="fake"):
    from types import SimpleNamespace
    return SimpleNamespace(height=h or fam.height, width=w or fam.width,
                           obstacles=tuple(fam.walls if walls is None else walls),
                           exits=tuple((f"E{i}", p) for i, p in enumerate(exits or fam.exits)), scenario_id=sid)


def _first_family():
    from ember_v1.gen.generator import generate_family
    from ember_v1.gen.spec import GeneratorSpec
    m = C.build_manifest({}, [], families=1, max_slots=8)
    return m, generate_family(GeneratorSpec.from_dict(m["plan"]["spec"]), m["families"][0]["family_seed"])[0]


def test_pilot_match_finds_copies_rotations_and_rescaled_maps():
    _, fam = _first_family()
    assert C.pilot_match(fam, C.PilotMap("p", _map(fam))) == "topology_duplicate"
    h, w = fam.height, fam.width
    rot = _map(fam, h=w, w=h, walls=[(c, h - 1 - r) for r, c in fam.walls],
               exits=[(c, h - 1 - r) for r, c in fam.exits])
    assert C.pilot_match(fam, C.PilotMap("p", rot)) == "topology_duplicate"          # symmetry invariant
    big = _map(fam, h=2 * h, w=2 * w,
               walls=[(2 * r + a, 2 * c + b) for r, c in fam.walls for a in (0, 1) for b in (0, 1)],
               exits=[(2 * r, 2 * c) for r, c in fam.exits])
    assert C.pilot_match(fam, C.PilotMap("p", big)).startswith("scale_free")         # 4x area: past the scale gate
    other = _map(fam, walls=[], exits=[(0, 0)])
    assert C.pilot_match(fam, C.PilotMap("p", other)) is None


def test_a_pilot_duplicate_excludes_the_whole_family_and_refills_in_pool_order():
    base, fam = _first_family()
    hit = C.build_manifest({}, [], families=1, max_slots=8, pilots=[C.PilotMap("t4-pilot@1", _map(fam, sid="x"))])
    ex = hit["excluded_families"][0]
    assert ex["family_id"] == fam.family_id and ex["reason"] == "pilot_map_duplicate"
    assert (ex["pilot_set"], ex["pilot_scenario_id"], ex["rule"]) == ("t4-pilot@1", "x", "topology_duplicate")
    assert fam.family_id not in {s["family_id"] for s in hit["scenarios"]}
    assert hit["families"][0]["pool_slot"] > base["families"][0]["pool_slot"]       # next slot, same order


def test_committed_audit_screens_every_pilot_map_without_hits(rebuilt):
    *_, audit = rebuilt
    g = audit["pilot_geometry"]
    assert set(g["maps_per_set"]) == set(C.PILOT_SETS) and g["pilot_maps"] == sum(g["maps_per_set"].values()) > 0
    fams = {f["family_id"] for f in _committed(C.MANIFEST_PATH)["families"]}
    assert g["hits"] == [] and set(g["nearest_by_family"]) == fams
    assert all(v["scale_free_wall_distance"] > 0.03 for v in g["nearest_by_family"].values())


# ----- clean checkout -------------------------------------------------------------------------------------------
def _ignore(r, *pats):
    (r.path / ".gitignore").write_text("\n".join(pats) + "\n", encoding="utf-8")
    r.git("add", ".gitignore")
    r.git("commit", "-q", "-m", "ignore")


def _write(r, rel, text="x"):
    f = r.path / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(text, encoding="utf-8")


def test_tree_problems_reject_untracked_sources_and_configs_but_allow_caches(tmp_path):
    r = Repo(tmp_path / "repo")
    _ignore(r, "__pycache__/", ".venv/", ".pytest_cache/", "results/", ".env")
    _write(r, "src/pkg/__pycache__/m.cpython-314.pyc")
    _write(r, ".venv/Lib/site.py")
    _write(r, ".pytest_cache/v/x")
    assert C.tree_problems(r.path) == []
    _write(r, "src/pkg/extra.py", "import os\n")
    _write(r, "configs/run.toml", "a = 1\n")
    got = C.tree_problems(r.path)
    assert "untracked src/pkg/extra.py" in got and "untracked configs/run.toml" in got
    _write(r, ".env", "SECRET=1\n")
    _write(r, "results/x.json", "{}")
    got = C.tree_problems(r.path)
    assert "ignored .env" in got and "ignored results/" in got                   # ignored, not allowlisted


def test_preconditions_refuse_untracked_python_and_configuration(repo):
    r, pref, mref, ev_path = repo
    _write(r, "src/ember_v1/teacher/shadow.py", "X = 1\n")
    with pytest.raises(C.ConfirmationError, match="untracked src/ember_v1/teacher/shadow.py"):
        _pre(r, pref, mref, ev_path)
    (r.path / "src/ember_v1/teacher/shadow.py").unlink()
    _write(r, "pyproject.local.toml", "[tool]\n")
    with pytest.raises(C.ConfirmationError, match="untracked pyproject.local.toml"):
        _pre(r, pref, mref, ev_path)


def test_preconditions_refuse_a_modified_manifest(repo):
    r, pref, mref, ev_path = repo
    m = _committed(C.MANIFEST_PATH)
    (r.path / mref["path"]).write_text(json.dumps({**m, "families": m["families"][:-1]}), encoding="utf-8")
    with pytest.raises(C.ConfirmationError, match="modified selection/manifest.json"):
        _pre(r, pref, mref, ev_path)


def test_preconditions_refuse_a_changed_policy_identity(repo, monkeypatch):
    r, pref, mref, ev_path = repo
    monkeypatch.setattr(reg, "check_pins", lambda *a, **k: ["T4 1.2.0 config hash differs from its pin"])
    with pytest.raises(C.ConfirmationError, match="identity drift"):
        _pre(r, pref, mref, ev_path)


def test_preconditions_refuse_a_missing_freeze_commit(repo):
    r, pref, mref, ev_path = repo
    with pytest.raises(C.ConfirmationError, match="protocol: "):
        _pre(r, {**pref, "commit": "f" * 40}, mref, ev_path)


def test_preconditions_refuse_existing_evidence_and_an_interrupted_attempt(repo):
    r, pref, mref, ev_path = repo
    marker, log = C.attempt_paths(ev_path)
    C.create_attempt_marker(marker, {"note": "interrupted"})           # a crash left only the marker
    with pytest.raises(C.ConfirmationError, match="previous confirmatory attempt exists"):
        _pre(r, pref, mref, ev_path)
    with pytest.raises(C.ConfirmationError, match="already started"):
        C.create_attempt_marker(marker, {"note": "second"})              # atomic exclusive create
    marker.unlink()
    log.write_text("{}\n", encoding="utf-8")                              # a log alone also blocks
    with pytest.raises(C.ConfirmationError, match="previous confirmatory attempt exists"):
        _pre(r, pref, mref, ev_path)
    log.unlink()
    ev_path.write_text("{}", encoding="utf-8")
    with pytest.raises(C.ConfirmationError, match="evidence already exists"):
        _pre(r, pref, mref, ev_path)


# ----- attempt marker and fail-closed evidence --------------------------------------------------------------------
def _report(stage="completed", abort=None):
    lim = {k: sel.draft_protocol()["compute"][k] for k in ("cpu_s_cap", "per_decision_cpu_ms_cap", "wall_s_cap")}
    return {"limits": lim, "stage": stage, "abort": abort}


def test_marker_is_written_before_the_first_scenario_and_a_crash_is_recorded_incomplete(repo, monkeypatch):
    r, pref, mref, ev_path = repo
    marker, log = C.attempt_paths(ev_path)

    def crash(p, man, **k):
        doc = json.loads(marker.read_text(encoding="utf-8"))
        assert doc["protocol_hash"] == p["protocol_hash"] and doc["code_commit"] == r.git("rev-parse", "HEAD")
        raise RuntimeError("worker pool died")
    monkeypatch.setattr(C, "run_rows", crash)
    out = C.run(r.path, pref, mref, trusted_ref="main", evidence_path=ev_path)
    doc = json.loads(ev_path.read_text(encoding="utf-8"))
    assert out["complete"] is False and out["stage"] == "supervisor_error" and doc["complete"] is False
    assert doc["supervision"]["abort"]["error"].startswith("RuntimeError") and doc["attempt"]["attempt_hash"]
    events = [json.loads(x)["event"] for x in log.read_text(encoding="utf-8").splitlines()]
    assert events == ["attempt_start", "evidence_written"]
    ev_path.unlink()                                                     # even without evidence: no second attempt
    with pytest.raises(C.ConfirmationError, match="previous confirmatory attempt exists"):
        C.run(r.path, pref, mref, trusted_ref="main", evidence_path=ev_path)


def test_a_supervisor_abort_is_never_complete(repo, monkeypatch):
    r, pref, mref, ev_path = repo
    m = _committed(C.MANIFEST_PATH)
    full = {c: {} for c in reg.CANDIDATES}
    rows = [{"scenario_id": s["scenario_id"], "family_id": s["family_id"], "scenario_hash": s["scenario_hash"],
             "results": full} for s in m["scenarios"]]
    abort = _report("total_cpu_cap", {"stage": "total_cpu_cap"})
    monkeypatch.setattr(C, "run_rows", lambda p, man, **k: (rows, [], abort))
    out = C.run(r.path, pref, mref, trusted_ref="main", evidence_path=ev_path)
    assert out["complete"] is False and out["stage"] == "total_cpu_cap"


# ----- live supervisor on synthetic non-confirmation scenarios ------------------------------------------------------
def tiny_loader(manifest):
    return _tiny()


def _burn(seconds):
    import time
    t = time.process_time()
    while time.process_time() - t < seconds:
        pass


def stuck_decision_worker(name, identity, seed, manifest, q, slot, loader=None):
    import time
    q.put(("start", name, 0))
    slot[C.DEC] = time.process_time()                                    # a decision that never returns
    _burn(60)


def cpu_hog_worker(name, identity, seed, manifest, q, slot, loader=None):
    q.put(("start", name, 0))
    _burn(60)


def sleeper_worker(name, identity, seed, manifest, q, slot, loader=None):
    import time
    q.put(("start", name, 0))
    time.sleep(60)


def crash_worker(name, identity, seed, manifest, q, slot, loader=None):
    import os
    os._exit(7)


LIMITS = {"cpu_s_cap": 600.0, "per_decision_cpu_ms_cap": 30000.0, "wall_s_cap": 120.0}


def _jobs():
    return [(c, _identity(c), sel.policy_seeds().get(c), {}) for c in ("T0", "T1")]


def test_supervisor_runs_real_workers_to_completion_on_synthetic_scenarios(tmp_path):
    log = C.AttemptLog(tmp_path / "log.jsonl")
    out = C.supervise(_jobs(), LIMITS, log=log, loader=tiny_loader)
    rep = out["report"]
    assert rep["stage"] == "completed" and rep["abort"] is None and out["errors"] == [], rep
    assert rep["exit_codes"] == {"T0": 0, "T1": 0} and rep["results_received"] == {"T0": 2, "T1": 2}
    assert set(out["results"]["T0"]) == {"conf-test-0", "conf-test-1"} and rep["total_process_cpu_s"] > 0
    events = [json.loads(x)["event"] for x in (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[0] == "supervisor_start" and events[-1] == "supervisor_end" and events.count("result") == 4


@pytest.mark.parametrize("target,limits,stage", [
    (stuck_decision_worker, {**LIMITS, "per_decision_cpu_ms_cap": 1500.0}, "per_decision_cpu_cap"),
    (cpu_hog_worker, {**LIMITS, "cpu_s_cap": 1.5}, "total_cpu_cap"),
    (sleeper_worker, {**LIMITS, "wall_s_cap": 3.0}, "wall_time_cap"),
    (crash_worker, LIMITS, "worker_crashed"),
])
def test_supervisor_stops_and_reaps_every_worker_at_the_first_breach(target, limits, stage):
    import time
    t0 = time.monotonic()
    out = C.supervise(_jobs(), limits, target=target)
    rep = out["report"]
    assert rep["stage"] == stage and rep["abort"]["stage"] == stage, rep
    assert all(code is not None for code in rep["exit_codes"].values())   # every worker reaped
    assert time.monotonic() - t0 < 45                                     # stopped long before the 60 s burn ends
