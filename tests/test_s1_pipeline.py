"""S1 dataset pipeline (#10): receipt-first gate, TRAIN-only plan, leakage, provenance, replay, diagnostics.

Every test runs on a synthetic split plan (`gen_helpers.small_plan`) or a hand-built scenario, in a temporary
git repository whose `main` plays the trusted ref. Nothing here labels a p1-02-v1 scenario.
"""
from __future__ import annotations

import copy
import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from ember_v1 import s1, sources
from ember_v1 import s1_pipeline as sp
from ember_v1.gen.splits import build_all
from ember_v1.sim.canonical import digest
from ember_v1.teacher.t1 import T1CongestionAware
from gen_helpers import small_plan
from sim_helpers import scenario as ascii_scenario
from test_s1_gate import CONF, Repo, _protocol, _receipt

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def plan():
    build, manifests, _, _ = build_all(small_plan(seed=31, plan_id="s1-pipeline-test"))
    scen = [g.scenario for g in build.scenarios["train"]]
    assert manifests["train"]["split"] == "train" and len(scen) >= 6
    return {"manifests": {k: manifests[k] for k in ("train", "dev", "calibration")}, "scenarios": scen}


@pytest.fixture(autouse=True)
def small_spot_check(monkeypatch):
    monkeypatch.setattr(s1, "MIN_SPOT_CHECKS", 3)


def _refs(tmp_path, *, receipt_over=None, pending=False):
    k = 0
    while (tmp_path / f"repo{k}").exists():
        k += 1
    repo = Repo(tmp_path / f"repo{k}")
    protocol = _protocol()
    pref = repo.commit("selection/protocol.json", protocol)
    over = dict(receipt_over or {})
    if pending:
        over["review"] = {"status": "pending", "reviewers": []}
    rref = repo.commit("selection/receipt.json", _receipt(protocol, pref, **over))
    return {"repo": repo.path, "protocol_ref": pref, "receipt_ref": rref, "trusted_ref": "main"}


def _build(tmp_path, plan, *, name="shard-0", out=None, manifests=None, scenarios=None, refs=None, **kw):
    refs = refs or _refs(tmp_path)
    kw.setdefault("max_episodes", 100)
    return sp.build_shard(out or tmp_path / "out", shard_id=name, manifests=manifests or plan["manifests"],
                          scenarios=plan["scenarios"] if scenarios is None else scenarios, **refs, **kw)


def _rehash_manifest(m):
    body = {k: v for k, v in m.items() if k != "manifest_hash"}
    return {**body, "manifest_hash": digest(body)}


def test_honest_shard_passes_every_gate_and_is_deterministic(tmp_path, plan):
    refs = _refs(tmp_path)
    a = _build(tmp_path, plan, out=tmp_path / "a", refs=refs)
    b = _build(tmp_path, plan, out=tmp_path / "b", refs=refs)
    assert a["s1_gate"]["decision"] == "BULK_S1_AUTHORIZED", a["s1_gate"]
    assert a["s1_gate"]["spot_check"]["pass"] and a["s1_gate"]["leakage_pass"]
    assert a["manifest_hash"] == b["manifest_hash"]
    assert a["files"]["zip"]["sha256"] == b["files"]["zip"]["sha256"]
    assert sp.verify_shard(tmp_path / "a" / "shard-0.manifest.json")["status"] == "verified"
    n = len(plan["manifests"]["train"]["scenarios"])
    assert a["quality"]["episodes"] == n and a["quality"]["episode_status"] == {"labelled": n}
    assert a["s1_gate"]["records"] == a["quality"]["decisions"]
    assert a["teacher"]["name"] == "T1" and a["teacher"]["version"] == "1.0.0"
    assert a["heldout_audit"]["pass"] and set(a["heldout_audit"]["heldout_sets"]) == {
        "dev", "calibration", "selection_confirmation"}
    for k in ("pipeline_fingerprint", "sources_fingerprint", "s1_gate_fingerprint"):
        assert a["code"][k].startswith("sha256:")


def test_traces_carry_real_actions_and_no_values_or_scores(tmp_path, plan):
    _build(tmp_path, plan)
    with zipfile.ZipFile(tmp_path / "out" / "shard-0.zip") as zf:
        docs = [json.loads(zf.read(n)) for n in sorted(zf.namelist())]
    pol = T1CongestionAware()
    for doc in docs:
        assert set(doc) == {"member_format", "episode", "traces"}
        for k, t in enumerate(doc["traces"]):
            assert set(t) == {"observation", "actions", "group_id", "source", "step", "provenance"}
            assert t["provenance"]["values_digest"] is None and t["source"] == "T1@1.0.0"
            assert t["actions"] == doc["episode"]["records"][k]["actions"]
        assert pol.decide(doc["traces"][0]["observation"]).actions == doc["traces"][0]["actions"]


def test_pending_receipt_blocks_before_any_teacher_call(tmp_path, plan, monkeypatch):
    def boom(self, obs):
        raise AssertionError("the teacher ran before the receipt gate")
    monkeypatch.setattr(T1CongestionAware, "decide", boom)
    with pytest.raises(sp.S1PipelineError, match="not reviewed") as e:
        _build(tmp_path, plan, refs=_refs(tmp_path, pending=True))
    assert e.value.report["decision"] == "BULK_S1_BLOCKED"
    assert not (tmp_path / "out").exists()


def test_receipt_for_another_teacher_or_forged_hash_is_blocked(tmp_path, plan):
    from test_s1_gate import _selected
    from ember_v1.teacher.t0 import T0ShortestPath
    with pytest.raises(sp.S1PipelineError, match="selection receipt"):
        _build(tmp_path, plan, refs=_refs(tmp_path, receipt_over={"selected": _selected(T0ShortestPath)}))


def test_train_family_in_dev_is_blocked(tmp_path, plan):
    m = copy.deepcopy(plan["manifests"])
    leak = dict(m["dev"]["scenarios"][0], family_id=m["train"]["scenarios"][0]["family_id"])
    m["dev"]["scenarios"][0] = leak
    m["dev"] = _rehash_manifest(m["dev"])
    with pytest.raises(sp.S1PipelineError, match="overlaps held-out"):
        _build(tmp_path, plan, manifests=m)


def test_train_core_geometry_in_an_extra_heldout_set_is_blocked(tmp_path, plan):
    core = plan["manifests"]["train"]["scenarios"][3]["core_hash"]
    extra = {"locked_test": {"family": set(), "lineage": set(), "scenario": set(), "core": {core}}}
    with pytest.raises(sp.S1PipelineError, match="overlaps held-out"):
        _build(tmp_path, plan, extra_heldout=extra)


def test_train_family_in_the_selection_confirmation_set_is_blocked(tmp_path, plan):
    fam = plan["manifests"]["train"]["scenarios"][0]["family_id"]
    conf = dict(CONF, family_ids=[fam])
    with pytest.raises(sp.S1PipelineError, match="selection_confirmation"):
        _build(tmp_path, plan, refs=_refs(tmp_path, receipt_over={"confirmation": conf}))


def test_only_the_train_split_is_labelled(plan):
    with pytest.raises(ValueError, match="only the train split"):
        sp.scenario_plan(plan["manifests"]["dev"], {"x": {k: set() for k in sp.HELD_OUT_KEYS}})


def test_max_episodes_guard_and_missing_scenarios(tmp_path, plan):
    with pytest.raises(sp.S1PipelineError, match="max_episodes"):
        _build(tmp_path, plan, max_episodes=2)
    with pytest.raises(sp.S1PipelineError, match="no scenario with the manifest hash"):
        _build(tmp_path, plan, scenarios=plan["scenarios"][1:])


def test_teacher_failure_is_a_diagnostic_never_a_label(tmp_path, plan, monkeypatch):
    from ember_v1.sim import Simulator
    bad = plan["scenarios"][0].scenario_id
    first = digest(Simulator(plan["scenarios"][0]).observe())
    real = T1CongestionAware.decide

    def flaky(self, obs):
        if digest(obs) == first:
            raise RuntimeError("synthetic teacher failure")
        return real(self, obs)
    monkeypatch.setattr(T1CongestionAware, "decide", flaky)
    # The regeneration spot check re-runs the teacher; keep the patched teacher for labelling only.
    monkeypatch.setattr(s1, "build_s1_training_set", lambda *a, **k: (_ for _ in ()).throw(
        s1.S1GateError({"reasons": ["not run in this test"]})))
    man = _build(tmp_path, plan)
    st = {e["scenario_id"]: e["status"] for e in man["episodes"]}
    assert st[bad] == "teacher_error"
    with zipfile.ZipFile(tmp_path / "out" / "shard-0.zip") as zf:
        assert not any(bad in n for n in zf.namelist())
    diags = [json.loads(x) for x in (tmp_path / "out" / "shard-0.diagnostics.jsonl").read_text().splitlines()]
    d = next(x for x in diags if x["scenario"]["scenario_id"] == bad)
    assert d["status"] == "teacher_error" and "synthetic teacher failure" in d["error"]
    assert sp.verify_shard(tmp_path / "out" / "shard-0.manifest.json")["status"] == "verified"


def _hand_built():
    # a: trapped in a walled pocket next to the fire (casualty). b: free path to the exit.
    return ascii_scenario(["#F#.......E",
                           "#.#........",
                           "###........"], {"a": (1, 1), "b": (2, 3)}, sid="hand-negative", max_steps=30)


def _hand_manifests(s):
    from test_s1_gate import _entry, _manifest
    e = dict(_entry(s.scenario_id, s.content_hash(), "L-hand", "F-hand"), core_hash="sha256:" + "9" * 64)
    return {"train": _manifest([e], "train"),
            "dev": _manifest([_entry("d", "sha256:" + "d" * 64, "L-dev", "F-dev")], "dev"),
            "calibration": _manifest([_entry("c", "sha256:" + "c" * 64, "L-cal", "F-cal")], "calibration")}


def test_hand_built_negative_episode_is_labelled_and_diagnosed(tmp_path):
    s = _hand_built()
    man = sp.build_shard(tmp_path / "out", shard_id="hand", manifests=_hand_manifests(s), scenarios=[s],
                         max_episodes=1, **_refs(tmp_path))
    assert man["s1_gate"]["decision"] == "BULK_S1_AUTHORIZED", man["s1_gate"]
    (d,) = [json.loads(x) for x in (tmp_path / "out" / "hand.diagnostics.jsonl").read_text().splitlines()]
    o = d["outcome"]
    assert d["status"] == "labelled" and d["negative"] is True
    assert o["casualties"] == len(o["casualty_events"]) >= 1 and o["casualty_events"][0]["person"] == "a"
    assert o["people"] == o["evacuated"] + o["casualties"] + o["unresolved"] and o["evacuated"] == 1
    assert man["quality"]["negative_episodes"] == 1 and man["quality"]["values_written"] == 0


def test_write_once_and_tamper_detection(tmp_path, plan):
    refs = _refs(tmp_path)
    _build(tmp_path, plan, refs=refs)
    with pytest.raises(sp.S1PipelineError, match="write-once"):
        _build(tmp_path, plan, refs=refs)
    mp = tmp_path / "out" / "shard-0.manifest.json"
    zp = tmp_path / "out" / "shard-0.zip"
    raw = bytearray(zp.read_bytes())
    raw[-30] ^= 1
    zp.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="zip differs"):
        sp.verify_shard(mp)
    man = json.loads(mp.read_text())
    man["quality"]["decisions"] += 1
    mp.write_text(json.dumps(man))
    with pytest.raises(ValueError, match="manifest hash"):
        sp.verify_shard(mp)


def test_forged_actions_in_a_rehashed_shard_fail_the_s1_gate(tmp_path, plan):
    """A shard edited after the pipeline: replay or provenance must refuse it."""
    refs = _refs(tmp_path)
    _build(tmp_path, plan, refs=refs)
    zp = tmp_path / "out" / "shard-0.zip"
    with zipfile.ZipFile(zp) as zf:
        members = [(n, json.loads(zf.read(n))) for n in sorted(zf.namelist())]
    doc = members[0][1]
    t = doc["traces"][0]
    pid = next(iter(t["actions"]))
    t["actions"][pid] = "stay" if t["actions"][pid] != "stay" else "up"
    forged = tmp_path / "forged.zip"
    sp.write_zip(forged, [(n, json.dumps(d).encode()) for n, d in members])
    with pytest.raises(s1.S1GateError):
        s1.build_s1_training_set([(forged, "forged")], repo=refs["repo"], protocol_ref=refs["protocol_ref"],
                                 receipt_ref=refs["receipt_ref"], manifests=plan["manifests"], trusted_ref="main")


# ----- ingestion format and bounds ------------------------------------------------------------------------
def _zip(tmp_path, members, name="z.zip"):
    p = tmp_path / name
    sp.write_zip(p, [(n, json.dumps(d).encode()) for n, d in members])
    return p


def test_shared_episode_member_rules(tmp_path):
    with pytest.raises(ValueError, match="shared-episode member"):
        sources.ingest_zip(_zip(tmp_path, [("a.json", {"member_format": "other@0", "episode": {}, "traces": []})]),
                           source_id="x", quality="teacher_verified")
    s = _hand_built()
    from ember_v1.sim import Simulator
    sim, recs, obs = Simulator(s), [], []
    pol = T1CongestionAware()
    while not sim.terminated:
        o = sim.observe()
        d = pol.decide(o)
        obs.append((o, d.actions))
        recs.append(sim.step(d.actions))
    ep = {"scenario": s.to_dict(), "seed": None, "records": recs}
    tr = [{"observation": o, "actions": a, "group_id": "g", "source": "T1", "step": k + 1}
          for k, (o, a) in enumerate(obs)]
    ok = {"member_format": sources.SHARED_EPISODE_FORMAT, "episode": ep, "traces": tr}
    recs_out, entry = sources.ingest_zip(_zip(tmp_path, [("a.json", ok)], "ok.zip"), source_id="x",
                                         quality="teacher_verified")
    assert len(recs_out) == len(tr) and all(r["replay"]["status"] == "complete" for r in recs_out)
    bad = copy.deepcopy(ok)
    bad["traces"][0]["episode"] = ep
    with pytest.raises(ValueError, match="must not carry its own episode"):
        sources.ingest_zip(_zip(tmp_path, [("a.json", bad)], "bad.zip"), source_id="x", quality="teacher_verified")


def test_zip_bounds(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="unsafe member path"):
        sources.ingest_zip(_zip(tmp_path, [("../a.json", {})]), source_id="x", quality="teacher_verified")
    monkeypatch.setattr(sources, "_MAX_MEMBERS", 2)
    with pytest.raises(ValueError, match="too many zip members"):
        sources.ingest_zip(_zip(tmp_path, [(f"{i}.json", {}) for i in range(3)], "m.zip"), source_id="x",
                           quality="teacher_verified")
    monkeypatch.setattr(sources, "_MAX_MEMBERS", 100_000)
    p = tmp_path / "bomb.zip"
    sp.write_zip(p, [("a.json", b" " * (3 * 2**20))])
    with pytest.raises(ValueError, match="compression ratio"):
        sources.ingest_zip(p, source_id="x", quality="teacher_verified")


# ----- storage estimate and status ------------------------------------------------------------------------
def test_storage_estimate_is_an_upper_bound_model(plan):
    # exact model bytes/step = 100 + 50 people + 10 walls, one sample 10 % above it
    def smp(p, w, steps, extra=1.0):
        b = (100 + 50 * p + 10 * w) * steps * extra
        return {"people": p, "walls": w, "steps": steps, "max_steps": 4 * steps, "member_bytes": b,
                "zip_bytes": b / 10}
    samples = [smp(4, 10, 10), smp(2, 30, 5), smp(9, 3, 7), smp(5, 5, 8)]
    est = sp.storage_estimate(samples, plan["manifests"]["train"])
    assert est["coef_uncompressed"] == pytest.approx([100, 50, 10], abs=1e-6)
    sc = [e["diagnostics"] for e in plan["manifests"]["train"]["scenarios"]]
    exact = sum((100 + 50 * e["people"] + 10 * e["walls"]) * e["max_steps"] for e in sc)
    assert est["uncompressed_bytes"]["upper"] == pytest.approx(exact, rel=1e-6)
    noisy = sp.storage_estimate(samples + [smp(5, 6, 8, extra=1.2)], plan["manifests"]["train"])
    assert noisy["cover_ratio_uncompressed"] > 1.0        # scaled so that every sample is covered
    assert est["uncompressed_bytes"]["upper"] >= est["uncompressed_bytes"]["central"]
    assert est["decisions"]["upper"] == sum(e["max_steps"] for e in sc)
    with pytest.raises(ValueError):
        sp.storage_estimate(samples[:2], plan["manifests"]["train"])


def _have_origin_main():
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--verify", "origin/main"],
                          capture_output=True).returncode == 0


@pytest.mark.skipif(not _have_origin_main(), reason="needs origin/main")
def test_real_repository_status_is_blocked():
    rep = sp.status(REPO)
    assert rep["decision"] == "BULK_S1_BLOCKED"
    assert all(v.startswith("refused") for v in rep["receipts"].values())


def test_project_repository_cannot_make_single_teacher_labels_after_pivot(tmp_path, plan, monkeypatch):
    # Even with otherwise valid refs, the project repository is refused before any receipt read or teacher call.
    monkeypatch.setattr(T1CongestionAware, "decide", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    refs = {**_refs(tmp_path), "repo": REPO}
    with pytest.raises(sp.S1PipelineError, match="superseded by the five-teacher direction"):
        _build(tmp_path, plan, refs=refs, out=tmp_path / "pivot-out")
    assert not (tmp_path / "pivot-out").exists()
