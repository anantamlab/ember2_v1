"""Mandatory S1 entrypoint (#10): trust root, receipt, eligibility, replay, provenance, regeneration, leakage.

The honest case below runs only on a temporary git repository and synthetic
records. It shows the gate logic; it does not authorize real S1 (no reviewed
receipt exists).
"""
from __future__ import annotations

import copy
import inspect
import json
import random
import subprocess
import zipfile

import pytest

from ember_v1 import provenance as pv
from ember_v1 import s1
from ember_v1.gen.spec import GENERATOR_REVISION
from ember_v1.sim import Simulator
from ember_v1.sim.canonical import digest
from ember_v1.sources import ingest_zip, training_records
from ember_v1.teacher import registry as reg
from ember_v1.teacher.t1 import T1CongestionAware
from gen_helpers import random_scenario


def _manifest(entries, split):
    body = {"manifest_version": "1.0", "split": split, "plan_id": "test-plan", "plan_hash": "sha256:" + "1" * 64,
            "generator_revision": GENERATOR_REVISION, "spec_hash": "sha256:" + "2" * 64, "families": [],
            "scenarios": entries, "failed_lineages": [], "lineage_rejections": {}}
    return {**body, "manifest_hash": digest(body)}


def _entry(sid, sh, lin, fam):
    return {"scenario_id": sid, "scenario_hash": sh, "lineage_id": lin, "family_id": fam}


@pytest.fixture(scope="module")
def world():
    s = random_scenario(random.Random(77))
    train = _manifest([_entry(s.scenario_id, s.content_hash(), "L-train-0", "F-train-0")], "train")
    dev = _manifest([_entry("d", "sha256:" + "d" * 64, "L-dev-0", "F-dev-0")], "dev")
    cal = _manifest([_entry("c", "sha256:" + "c" * 64, "L-cal-0", "F-cal-0")], "calibration")
    pol = T1CongestionAware()
    sim, recs, steps = Simulator(s), [], []
    while not sim.terminated:
        obs = sim.observe()
        d = pol.decide(obs)
        steps.append((obs, d))
        recs.append(sim.step(d.actions))
    ep = {"scenario": s.to_dict(), "seed": None, "records": recs}
    traces = [{"observation": o, "actions": d.actions, "group_id": "ep", "source": "T1", "episode": ep,
               "step": k + 1, "provenance": pv.provenance_record(pol, d, manifest=train, scenario_id=s.scenario_id)}
              for k, (o, d) in enumerate(steps)]
    assert len(traces) >= 3
    return {"s": s, "manifests": {"train": train, "dev": dev, "calibration": cal}, "traces": traces}


@pytest.fixture(autouse=True)
def small_spot_check(monkeypatch):
    monkeypatch.setattr(s1, "MIN_SPOT_CHECKS", 3)


def _zip(tmp_path, traces, name="s.zip"):
    p = tmp_path / name
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("t.json", json.dumps({"traces": traces}))
    return p


class Repo:
    """A temporary git repository whose `main` branch plays the reviewed trusted ref."""

    def __init__(self, path):
        self.path = path
        path.mkdir()
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.email", "t@example.invalid")
        self.git("config", "user.name", "t")
        self.git("commit", "-q", "--allow-empty", "-m", "root")

    def git(self, *a):
        return subprocess.run(["git", "-C", str(self.path), *a], check=True, capture_output=True,
                              text=True).stdout.strip()

    def commit(self, rel, obj):
        f = self.path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(obj), encoding="utf-8")
        self.git("add", rel)
        self.git("commit", "-q", "-m", rel)
        return {"kind": "git", "commit": self.git("rev-parse", "HEAD"), "path": rel}


def _protocol(eligibility=None, **over):
    from ember_v1.teacher import selection          # the full #25 protocol (s1 applies its validator too)
    body = {**selection.draft_protocol(), "protocol_id": "p1-09-test", "frozen": True,
            "eligibility": reg.default_eligibility() if eligibility is None else eligibility}
    body.update(over)
    return {**body, "protocol_hash": s1.protocol_hash(body)}


def _selected(cls=T1CongestionAware):
    i = cls().identity
    return {"name": i.name, "policy": i.policy, "version": i.version, "config_hash": i.config_hash,
            "code_fingerprint": pv.code_fingerprint(cls)}


CONF = {"namespace": "p1-09-selection@1", "manifest_hash": "sha256:" + "a" * 64, "evidence_hash": "sha256:" + "b" * 64,
        "family_ids": ["CF-0"], "lineage_ids": ["CL-0"], "scenario_hashes": ["sha256:" + "e" * 64]}


def _receipt(protocol, pref, rehash=True, **over):
    body = {"receipt_version": s1.RECEIPT_VERSION, "protocol_id": protocol["protocol_id"],
            "protocol_hash": protocol["protocol_hash"], "protocol_ref": pref, "decision": "RETAIN_T1_NO_IMPROVEMENT",
            "selected": _selected(), "eligibility": protocol["eligibility"], "confirmation": dict(CONF),
            "review": {"status": "approved", "reviewers": ["reviewer-a"]}}
    body.update(over)
    return {**body, "receipt_hash": s1.receipt_hash(body) if rehash else "sha256:" + "0" * 64}


def _run(tmp_path, world, *, protocol=None, receipt=None, receipt_over=None, traces=None, manifests=None,
         trusted_ref="main", repo=None):
    repo = repo or Repo(tmp_path / "repo")
    protocol = protocol or _protocol()
    pref = repo.commit("selection/protocol.json", protocol)
    receipt = receipt if receipt is not None else _receipt(protocol, pref, **(receipt_over or {}))
    rref = repo.commit("selection/receipt.json", receipt)
    return s1.build_s1_training_set([(_zip(tmp_path, world["traces"] if traces is None else traces), "src")],
                                    repo=repo.path, protocol_ref=pref, receipt_ref=rref,
                                    manifests=manifests or world["manifests"], trusted_ref=trusted_ref)


def _blocked(match, **kw):
    with pytest.raises(s1.S1GateError, match=match) as e:
        _run(**kw)
    assert e.value.report["decision"] == "BULK_S1_BLOCKED" and e.value.report["reasons"]
    return e.value.report


def test_honest_synthetic_case_passes_every_check(tmp_path, world):
    ts = _run(tmp_path, world)
    assert isinstance(ts, s1.S1TrainingSet) and len(ts.records) == len(world["traces"])
    r = ts.report
    assert r["decision"] == "BULK_S1_AUTHORIZED" and all(r["dependencies"].values())
    assert r["spot_check"]["pass"] and r["spot_check"]["checked"] >= 3 and r["leakage"]["pass"]
    assert all(x["replay"]["status"] == "complete" and x["provenance"]["status"] == "verified" for x in ts.records)


def test_training_set_cannot_be_built_outside_the_gate():
    with pytest.raises(s1.S1GateError, match="only come from build_s1_training_set"):
        s1.S1TrainingSet((), {})
    with pytest.raises(s1.S1GateError):
        s1.S1TrainingSet((), {}, object())


def test_entrypoint_has_no_opt_out():
    params = set(inspect.signature(s1.build_s1_training_set).parameters)
    assert params == {"sources", "repo", "protocol_ref", "receipt_ref", "manifests", "trusted_ref"}


def test_legacy_ingest_never_marks_s1_eligible(tmp_path, world):
    recs, entry = ingest_zip(_zip(tmp_path, world["traces"]), source_id="x", quality="teacher_verified")
    assert entry["s1_eligible"] is False and entry["provenance_verified"] is False
    assert training_records(recs)               # legacy replay-only filter still works for non-S1 use
    _, entry = ingest_zip(_zip(tmp_path, world["traces"]), source_id="x", quality="teacher_verified",
                          manifests={"train": world["manifests"]["train"]})
    assert entry["s1_eligible"] is False and entry["provenance_verified"] is True


# ----- bypasses ---------------------------------------------------------------------------------------------
def test_traces_without_provenance_are_blocked(tmp_path, world):
    plain = [{k: v for k, v in t.items() if k != "provenance"} for t in world["traces"]]
    _blocked("provenance record required", tmp_path=tmp_path, world=world, traces=plain)


def test_traces_without_full_replay_are_blocked(tmp_path, world):
    cut = [dict(t) for t in world["traces"]]
    cut[0] = {**cut[0], "episode": {**cut[0]["episode"], "records": cut[0]["episode"]["records"][:1]}}
    _blocked("replay|step inside", tmp_path=tmp_path, world=world, traces=cut)


def test_too_few_decisions_for_the_regeneration_check_are_blocked(tmp_path, world, monkeypatch):
    monkeypatch.setattr(s1, "MIN_SPOT_CHECKS", 10 ** 6)
    _blocked("spot check", tmp_path=tmp_path, world=world)


def test_consistent_forged_actions_fail_regeneration(tmp_path, world, monkeypatch):
    monkeypatch.setattr(s1, "SPOT_FRACTION", 1.0)          # check every decision
    obs = world["traces"][0]["observation"]
    honest = world["traces"][0]["actions"]
    pid = sorted(honest)[0]
    fake = {**honest, pid: next(a for a in obs["allowed_actions"] if a != honest[pid])}
    s = world["s"]
    sim, recs = Simulator(s), []
    first = True
    pol = T1CongestionAware()
    traces = []
    while not sim.terminated:
        o = sim.observe()
        d = pol.decide(o)
        acts = fake if first else d.actions
        rec = pv.provenance_record(pol, d, manifest=world["manifests"]["train"], scenario_id=s.scenario_id)
        if first:
            body = {k: v for k, v in rec.items() if k != "record_hash"}
            body["actions_digest"] = digest(acts)
            rec = {**body, "record_hash": digest(body)}
        traces.append((o, acts, rec))
        recs.append(sim.step(acts))
        first = False
    ep = {"scenario": s.to_dict(), "seed": None, "records": recs}
    forged = [{"observation": o, "actions": a, "group_id": "ep", "source": "T1", "episode": ep, "step": k + 1,
               "provenance": r} for k, (o, a, r) in enumerate(traces)]
    rep = _blocked("spot check", tmp_path=tmp_path, world=world, traces=forged)
    assert rep["spot_check"]["pass"] is False


def test_records_of_another_teacher_are_blocked(tmp_path, world):
    p = _protocol()
    from ember_v1.teacher.t0 import T0ShortestPath
    _blocked("not the selected identity", tmp_path=tmp_path, world=world, protocol=p,
             receipt_over={"decision": "SELECT_STANDALONE", "selected": _selected(T0ShortestPath)})


# ----- trust root -------------------------------------------------------------------------------------------
def test_unknown_trust_root_kind_is_blocked(tmp_path, world):
    repo = Repo(tmp_path / "repo")
    pref = repo.commit("p.json", _protocol())
    bad = [{**pref, "kind": "https"}, {**pref, "commit": pref["commit"][:12]}, {**pref, "path": "../p.json"},
           {**pref, "path": "/p.json"}, {k: v for k, v in pref.items() if k != "kind"}]
    for ref in bad:
        with pytest.raises(s1.S1GateError, match="trust root"):
            s1.build_s1_training_set([], repo=repo.path, protocol_ref=ref, receipt_ref=pref,
                                     manifests=world["manifests"], trusted_ref="main")


def test_receipt_outside_the_trusted_history_is_blocked(tmp_path, world):
    repo = Repo(tmp_path / "repo")
    p = _protocol()
    pref = repo.commit("p.json", p)
    repo.git("checkout", "-q", "-b", "unreviewed")
    rref = repo.commit("r.json", _receipt(p, pref))
    repo.git("checkout", "-q", "main")
    with pytest.raises(s1.S1GateError, match="not in the history of trusted ref"):
        s1.build_s1_training_set([(_zip(tmp_path, world["traces"]), "s")], repo=repo.path, protocol_ref=pref,
                                 receipt_ref=rref, manifests=world["manifests"], trusted_ref="main")
    with pytest.raises(s1.S1GateError, match="trust root"):          # default ref does not exist here
        s1.build_s1_training_set([], repo=repo.path, protocol_ref=pref, receipt_ref=pref,
                                 manifests=world["manifests"])


# ----- forged or wrong receipts ----------------------------------------------------------------------------
def _p():
    return _protocol()


FORGED_RECEIPTS = {
    "edited without rehash": ("hash does not match", lambda p, r: _receipt(p, r, rehash=False)),
    "selects blocked T4 v1.1.0": ("not an eligible identity|blocked", lambda p, r: _receipt(
        p, r, decision="SELECT_STANDALONE",
        selected={**reg.identity("T4", "1.1.0").key(), "code_fingerprint": "sha256:" + "0" * 64})),
    "selects blocked T4 v1.0.0": ("not an eligible identity|blocked", lambda p, r: _receipt(
        p, r, decision="SELECT_STANDALONE",
        selected={**reg.identity("T4", "1.0.0").key(), "code_fingerprint": "sha256:" + "0" * 64})),
    "wrong code fingerprint": ("code fingerprint", lambda p, r: _receipt(
        p, r, selected={**_selected(), "code_fingerprint": "sha256:" + "0" * 64})),
    "wrong protocol hash": ("another protocol", lambda p, r: _receipt(p, r, protocol_hash="sha256:" + "0" * 64)),
    "wrong protocol ref": ("another protocol commit", lambda p, r: _receipt(p, {**r, "path": "other.json"})),
    "not reviewed": ("not reviewed", lambda p, r: _receipt(p, r, review={"status": "pending", "reviewers": []})),
    "review without reviewers": ("not reviewed", lambda p, r: _receipt(p, r, review={"status": "approved",
                                                                                   "reviewers": [" "]})),
    "arbiter": ("arbiter", lambda p, r: _receipt(p, r, decision="SELECT_ARBITER")),
    "retain T1 but selects T2": ("must select T1", lambda p, r: _receipt(
        p, r, selected=_selected(reg.teacher_class("T2")))),
    "unknown decision": ("unknown receipt decision", lambda p, r: _receipt(p, r, decision="SELECTED")),
    "receipt claims T4 eligible": ("eligibility differs", lambda p, r: _receipt(
        p, r, eligibility={**p["eligibility"], "T4": {"status": "eligible",
                                                      "identity": reg.identity("T4", "1.1.0").key()}})),
    "empty confirmation": ("confirmation", lambda p, r: _receipt(p, r, confirmation={**CONF, "family_ids": []})),
    "extra field": ("missing or extra", lambda p, r: {**_receipt(p, r), "note": "x"}),
    "wrong version": ("unsupported receipt", lambda p, r: _receipt(p, r, receipt_version="teacher-selection@0")),
}


@pytest.mark.parametrize("case", sorted(FORGED_RECEIPTS))
def test_forged_or_wrong_receipt_is_blocked(tmp_path, world, case):
    match, make = FORGED_RECEIPTS[case]
    repo = Repo(tmp_path / "repo")
    p = _p()
    pref = repo.commit("selection/protocol.json", p)
    rref = repo.commit("selection/receipt.json", make(p, pref))
    with pytest.raises(s1.S1GateError, match=match):
        s1.build_s1_training_set([(_zip(tmp_path, world["traces"]), "s")], repo=repo.path, protocol_ref=pref,
                                 receipt_ref=rref, manifests=world["manifests"], trusted_ref="main")


# ----- protocol eligibility ---------------------------------------------------------------------------------
def _elig(fn):
    e = copy.deepcopy(reg.default_eligibility())
    fn(e)
    return e


BAD_PROTOCOLS = {
    "T4 eligible as blocked v1.1.0": _protocol(_elig(lambda e: e.update(
        T4={"status": "eligible", "identity": reg.identity("T4", "1.1.0").key()}))),
    "T4 silently missing": _protocol(_elig(lambda e: e.pop("T4"))),
    "T4 excluded without blocked records": _protocol(_elig(lambda e: e["T4"].update(blocked_identities=[]))),
    "T4 excluded with wrong verdict": _protocol(_elig(
        lambda e: e["T4"]["blocked_identities"][0].update(verdict="PASSED"))),
    "T4 excluded without reason": _protocol(_elig(lambda e: e["T4"].pop("reason"))),
    "not frozen": _protocol(frozen=False),
    "hash mismatch": {**_protocol(), "protocol_id": "edited"},
}


@pytest.mark.parametrize("case", sorted(BAD_PROTOCOLS))
def test_wrong_excluded_candidate_status_in_the_protocol_is_blocked(tmp_path, world, case):
    _blocked("selection receipt", tmp_path=tmp_path, world=world, protocol=BAD_PROTOCOLS[case])


# ----- manifests and leakage -------------------------------------------------------------------------------
def test_missing_or_mixed_manifests_are_blocked(tmp_path, world):
    m = world["manifests"]
    _blocked("manifests", tmp_path=tmp_path, world=world, manifests={"train": m["train"], "dev": m["dev"]})
    tmp2 = tmp_path / "b"
    tmp2.mkdir()
    swapped = {**m, "calibration": m["dev"]}
    _blocked("manifests", tmp_path=tmp2, world=world, manifests=swapped)


def test_training_family_in_the_selection_confirmation_set_is_blocked(tmp_path, world):
    conf = {**CONF, "family_ids": ["F-train-0"]}
    rep = _blocked("leakage", tmp_path=tmp_path, world=world, receipt_over={"confirmation": conf})
    assert rep["leakage"]["heldout"]["overlap"]["selection_confirmation"]["family"] == ["F-train-0"]


def test_training_scenario_in_dev_is_blocked(tmp_path, world):
    m = world["manifests"]
    tr = m["train"]["scenarios"][0]
    dev = _manifest([_entry("d", tr["scenario_hash"], "L-dev-0", "F-dev-0")], "dev")
    rep = _blocked("leakage", tmp_path=tmp_path, world=world, manifests={**m, "dev": dev})
    assert rep["leakage"]["heldout"]["overlap"]["dev"]["scenario"] == [tr["scenario_hash"]]
