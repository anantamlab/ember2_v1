"""Run 19: private HF artifact store, workstation execution interface and the Phase 8 policy arms (CPU only).

The Hub is replaced by an in-memory double with commits, so publish / verify / fetch / immutability are
checked end to end without network. The GPU adapter is replaced by a deterministic scorer."""
from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import random
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from ember_v1 import hfstore as hs
from ember_v1 import policy_arms as pa
from ember_v1 import student_run as sr
from ember_v1 import student_train as st
from ember_v1 import workstation as ws
from ember_v1.sim import Simulator
from ember_v1.sim.movement import resolve_moves
from ember_v1.teacher.grid import PublicGrid
from gen_helpers import random_scenario


# ----- doubles ----------------------------------------------------------------------------------------------
class FakeHf:
    """Commits = full snapshots {repo_path: bytes}; oids are fake 40-hex ids."""

    def __init__(self) -> None:
        self.snaps: list[dict[str, bytes]] = [{}]
        self.oids = ["0" * 40]
        self.folder_uploads = 0

    def _commit(self, files: dict[str, bytes]) -> SimpleNamespace:
        snap = dict(self.snaps[-1])
        snap.update(files)
        self.snaps.append(snap)
        self.oids.append(hashlib.sha1(f"c{len(self.snaps)}".encode()).hexdigest())
        return SimpleNamespace(oid=self.oids[-1])

    def _snap(self, rev: str | None) -> dict[str, bytes]:
        return self.snaps[-1] if rev in (None, "main") else self.snaps[self.oids.index(rev)]

    def whoami(self) -> dict:
        return {"name": "tester"}

    def create_repo(self, *a, **k) -> None:
        pass

    def repo_info(self, *a, **k) -> SimpleNamespace:
        return SimpleNamespace(private=True, sha=self.oids[-1])

    def file_exists(self, rid, rel, repo_type=None) -> bool:
        return rel in self.snaps[-1]

    def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type, commit_message):
        data = path_or_fileobj if isinstance(path_or_fileobj, bytes) else Path(path_or_fileobj).read_bytes()
        return self._commit({path_in_repo: data})

    def upload_folder(self, *, repo_id, repo_type, folder_path, path_in_repo, ignore_patterns, commit_message):
        self.folder_uploads += 1
        files = {}
        for p in sorted(Path(folder_path).rglob("*")):
            rel = p.relative_to(folder_path).as_posix()
            if p.is_file() and not any(fnmatch.fnmatch(rel, pat) for pat in ignore_patterns):
                files[f"{path_in_repo}/{rel}"] = p.read_bytes()
        return self._commit(files)

    def list_repo_tree(self, rid, path, *, recursive, revision, repo_type):
        for k, v in sorted(self._snap(revision).items()):
            if k.startswith(path + "/"):
                big = len(v) > 300
                yield SimpleNamespace(path=k, size=len(v), blob_id=None if big else
                                      hashlib.sha1(b"blob %d\0" % len(v) + v).hexdigest(),
                                      lfs={"sha256": hashlib.sha256(v).hexdigest()} if big else None)

    def get_paths_info(self, rid, paths, *, expand, repo_type):
        rel = paths[0]
        last = None
        for i, snap in enumerate(self.snaps):
            if rel in snap and (i == 0 or self.snaps[i - 1].get(rel) != snap[rel]):
                last = self.oids[i]
        return [SimpleNamespace(last_commit=SimpleNamespace(oid=last))] if last else []

    def hf_hub_download(self, rid, filename, *, repo_type, revision, cache_dir=None, local_dir=None):
        snap = self._snap(revision)
        if filename not in snap:
            raise FileNotFoundError(filename)
        out = Path(local_dir or cache_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(snap[filename])
        return str(out)


class Scorer:
    """Deterministic stand-in for ClefLoraAdapter: log-probs from a hash of (state, person, salt)."""

    def __init__(self, salt: str = "", fail_after: int | None = None) -> None:
        self.salt, self.calls, self.fail_after = salt, 0, fail_after

    def scores(self, row):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise KeyboardInterrupt("simulated stop")
        out = {}
        for pid in row["questions"]:
            h = hashlib.sha256(f"{row['state']}|{pid}|{self.salt}".encode()).digest()
            z = [h[i] / 40.0 for i in range(5)]
            m = max(z)
            lse = m + math.log(sum(math.exp(v - m) for v in z))
            out[pid] = {a: z[j] - lse for j, a in enumerate(st.ACTION_ORDER)}
        return out

    def predict(self, row):
        return {p: min(s, key=lambda a: (-s[a], a)) for p, s in self.scores(row).items()}

    def memory_bytes(self):
        return 0


# ----- hfstore ----------------------------------------------------------------------------------------------
def _payload(d: Path) -> Path:
    (d / "sub").mkdir(parents=True)
    (d / "a.json").write_text('{"x": 1}\n', encoding="utf-8")
    (d / "sub" / "big.bin").write_bytes(bytes(range(256)) * 8)
    (d / "skip.tmp").write_text("temp", encoding="utf-8")
    return d


def _man(files, path="grp/art@1", aid="art@1"):
    return hs.build_manifest(artifact_id=aid, kind="test", status="COMPLETE", repo_id="tester/ember-v1-datasets",
                             repo_type="dataset", path=path, files=files, source={"git_commit": None})


def test_publish_verify_fetch_roundtrip_and_idempotence(tmp_path):
    src = _payload(tmp_path / "src")
    hf = FakeHf()
    files = hs.file_table(src)
    assert [f["path"] for f in files] == ["a.json", "sub/big.bin"]           # *.tmp never uploaded
    e = hs.publish(hf, _man(files), src, log=lambda s: None)
    assert e["verification"] == "PASS" and e["verified_commit"] and e["manifest_commit"] in hf.oids
    assert "grp/art@1/skip.tmp" not in hf.snaps[-1]
    again = hs.publish(hf, _man(files), src, log=lambda s: None)               # no second upload
    assert hf.folder_uploads == 1 and again["manifest_sha256"] == e["manifest_sha256"]
    r = hs.fetch(hf, repo_id="r", repo_type="dataset", path="grp/art@1", revision=e["manifest_commit"],
                 dest=tmp_path / "dl", manifest_sha256=e["manifest_sha256"], log=lambda s: None)
    assert r["local_verification"]["result"] == "PASS"
    with pytest.raises(hs.StoreError, match="manifest"):
        hs.fetch(hf, repo_id="r", repo_type="dataset", path="grp/art@1", revision=e["manifest_commit"],
                 dest=tmp_path / "dl2", manifest_sha256="sha256:" + "0" * 64, log=lambda s: None)


def test_published_versions_are_immutable(tmp_path):
    src = _payload(tmp_path / "src")
    hf = FakeHf()
    hs.publish(hf, _man(hs.file_table(src)), src, log=lambda s: None)
    (src / "a.json").write_text('{"x": 2}\n', encoding="utf-8")
    with pytest.raises(hs.StoreError, match="immutable"):
        hs.publish(hf, _man(hs.file_table(src)), src, log=lambda s: None)


def test_remote_verification_detects_tampering(tmp_path):
    src = _payload(tmp_path / "src")
    hf = FakeHf()
    man = _man(hs.file_table(src))
    e = hs.publish(hf, man, src, log=lambda s: None)
    mb = hs.dumps(man)
    for key, data in (("grp/art@1/sub/big.bin", b"\x00" * 2048), ("grp/art@1/a.json", b'{"x": 9}\n')):
        bad = FakeHf()
        bad.snaps, bad.oids = [dict(s) for s in hf.snaps], list(hf.oids)
        bad.snaps[bad.oids.index(e["manifest_commit"])][key] = data
        res = hs.verify_remote(bad, man, mb, e["manifest_commit"], full=True, log=lambda s: None)
        assert res["result"] == "FAIL"
    hf.snaps[hf.oids.index(e["manifest_commit"])]["grp/art@1/extra.txt"] = b"x"
    assert any("extra" in p for p in hs.verify_remote(hf, man, mb, e["manifest_commit"], log=lambda s: None)["problems"])


def test_credentials_are_never_uploaded(tmp_path):
    d = tmp_path / "p"
    d.mkdir()
    (d / "log.txt").write_text("token=hf_" + "A1b2C3d4" * 5 + "\n", encoding="utf-8")
    with pytest.raises(hs.StoreError, match="credential"):
        hs.file_table(d)


def test_local_checks_cross_check_existing_hash_lists(tmp_path):
    d = _payload(tmp_path / "p")
    files = hs.file_table(d)
    by = {f["path"]: f["sha256"] for f in files}
    (d / "SHA256SUMS").write_text(f"{by['a.json']} *./a.json\n{by['sub/big.bin']} *./sub/big.bin\n", encoding="utf-8")
    (d / "sft_manifest.json").write_text(json.dumps({"files": {"a.json": "sha256:" + by["a.json"]}}), encoding="utf-8")
    files = hs.file_table(d)
    ok = hs.local_checks(files, d, [{"file": "SHA256SUMS", "type": "sha256sums"},
                                    {"file": "sft_manifest.json", "type": "sft_manifest"}])
    assert ok["result"] == "PASS"
    (d / "SHA256SUMS").write_text("0" * 64 + " *./a.json\n", encoding="utf-8")
    assert hs.local_checks(hs.file_table(d), d, [{"file": "SHA256SUMS", "type": "sha256sums"}])["result"] == "FAIL"


# ----- workstation ------------------------------------------------------------------------------------------
@pytest.fixture()
def world(tmp_path):
    """A records artifact (real simulator episodes of the deterministic scorer) published to a fake Hub, plus a
    tiny checkout with run configs and the registry."""
    proto = st.load_protocol()
    cfg = st.config_from_protocol(proto)
    rec_dir = tmp_path / "records"
    (rec_dir / "rollouts").mkdir(parents=True)
    lst = rec_dir / "rollouts" / "list.jsonl"
    with open(lst, "w", encoding="utf-8", newline="\n") as f:
        for i in range(3):
            s = random_scenario(random.Random(900 + i))
            f.write(json.dumps({"set": "dev", "scenario_id": f"s{i}", "family_id": f"f{i}",
                                "scenario": s.to_dict(), "scenario_hash": s.content_hash()}, sort_keys=True) + "\n")
    g = SimpleNamespace(wall_cap_s=1e9, usd_cap=1e9)
    (rec_dir / "eval-out").mkdir()
    sr.rollouts(Scorer(), cfg, lst, rec_dir / "eval-out" / "rollouts.jsonl", label="V1", guards=lambda: g,
                log=lambda s: None)
    model_dir = tmp_path / "model"
    (model_dir / "final").mkdir(parents=True)
    (model_dir / "final" / "adapter_config.json").write_text("{}", encoding="utf-8")
    (model_dir / "identity.json").write_text(json.dumps({
        "checkpoint_dir": "final", "protocol": {"path": "data/protocols/p1-14-student-train.json",
                                                "protocol_hash": proto["protocol_hash"]},
        "training": {"final_weights_digest": "sha256:x"}, "base_model": {}}), encoding="utf-8")
    hf = FakeHf()
    reg = {"registry_version": hs.VERSION, "namespace": "tester", "repos": {}, "artifacts": {}}
    for aid, d, repo, path in (("rec@1", rec_dir, "experiments", "v1-evaluation/rec@1"),
                               ("model@1", model_dir, "models", "v1/model@1")):
        rid, rtype = hs.repo_ids("tester")[repo]
        man = hs.build_manifest(artifact_id=aid, kind="t", status="COMPLETE", repo_id=rid, repo_type=rtype,
                                path=path, files=hs.file_table(d), source={"git_commit": None})
        hs.record(reg, hs.publish(hf, man, d, log=lambda s: None))
    repo = tmp_path / "checkout"
    (repo / "data" / "hf").mkdir(parents=True)
    (repo / "data" / "runs").mkdir(parents=True)
    (repo / "data" / "protocols").mkdir(parents=True)
    (repo / "infra").mkdir()
    (repo / "infra" / "p0b-requirements.lock.txt").write_text("torch==2.8.0 \\\n", encoding="utf-8")
    src_proto = Path(st.__file__).resolve().parents[2] / "data" / "protocols" / "p1-14-student-train.json"
    (repo / "data" / "protocols" / "p1-14-student-train.json").write_bytes(src_proto.read_bytes())
    hs.save_registry(reg, repo / "data" / "hf" / "registry.json")

    def inp(aid, repo_key):
        e = reg["artifacts"][aid]
        return {"repo": repo_key, "artifact_id": aid, "path": e["path"], "manifest_sha256": e["manifest_sha256"]}
    auth = {"status": "AUTHORIZED", "by": "test"}
    base = {"config_version": "ember-run-config@1", "authorization": auth, "max_wall_s": 3600}
    params = {"records": "eval-out/rollouts.jsonl", "scenarios": "rollouts/list.jsonl", "scenario_ids": "ALL",
              "question_style": cfg["question_style"]}
    confs = {"replay": {**base, "run_id": "t-replay@1", "kind": "trace-replay", "inputs": {"records": inp("rec@1", "experiments")},
                        "params": params, "result_path": "student-rollouts/t-replay@1"},
             "repro": {**base, "run_id": "t-repro@1", "kind": "v1-reproduce",
                       "inputs": {"model": inp("model@1", "models"), "records": inp("rec@1", "experiments")},
                       "params": {**params, "max_steps": 4}, "result_path": "v1-evaluation/t-repro@1"},
             "arms": {**base, "run_id": "t-arms@1", "kind": "v1-policy-ablation",
                      "authorization": {"status": "NOT_AUTHORIZED"},
                      "inputs": {"model": inp("model@1", "models"), "records": inp("rec@1", "experiments")},
                      "params": {"scenarios": "rollouts/list.jsonl", "sets": ["dev"], "arms": list(pa.ARMS)},
                      "result_path": "v1-policy-ablation/t-arms@1"}}
    for k, v in confs.items():
        (repo / "data" / "runs" / f"{k}.json").write_text(json.dumps(v), encoding="utf-8")
    revs = {"experiments": reg["artifacts"]["rec@1"]["manifest_commit"],
            "models": reg["artifacts"]["model@1"]["manifest_commit"]}
    return SimpleNamespace(hf=hf, repo=repo, revs=revs, root=tmp_path / "ws", reg=reg, cfg=cfg, rec_dir=rec_dir)


def _prep(w, name, monkeypatch=None, attempt=1, revs=None):
    return ws.prepare_run(config=f"data/runs/{name}.json", commit="a" * 40, revisions=revs or w.revs, root=w.root,
                          attempt=attempt, repo=w.repo, hf=w.hf, skip_code_check=True, log=lambda s: None)


@pytest.fixture()
def no_gpu_checks(monkeypatch):
    monkeypatch.setattr(ws, "check_environment", lambda env, gpu: None)
    monkeypatch.setattr(ws, "fetch_base_model", lambda root, log: {"repo": "stub", "revision": "0" * 40})
    real = ws.environment
    monkeypatch.setattr(ws, "environment", lambda repo, gpu: {**real(repo, gpu=False), "gpu": {"device": "stub"}})


def test_trace_replay_prepare_execute_publish_fetch(world, tmp_path):
    rd = _prep(world, "replay")
    man = json.loads((rd / "execution_manifest.json").read_text(encoding="utf-8"))
    assert man["inputs"]["records"]["revision"] == world.revs["experiments"]
    res = ws.execute_run(rd, repo=world.repo, skip_code_check=True)
    assert res["verdict"] == "PASS" and res["matched"] == 3 and (rd / "final" / "replay.jsonl").exists()
    assert not (rd / "partial").exists() and ws._state(rd)["status"] == "COMPLETE"
    with pytest.raises(ws.WsError, match="never executed again"):
        ws.execute_run(rd, repo=world.repo, skip_code_check=True)
    e = ws.publish_results(rd, hf=world.hf, log=lambda s: None)
    assert e["verification"] == "PASS" and e["path"] == "student-rollouts/t-replay@1/attempt-1"
    rv = ws.fetch_results(repo_id="tester/ember-v1-experiments", path=e["path"], revision=e["manifest_commit"],
                          dest=tmp_path / "laptop", hf=world.hf, log=lambda s: None)
    assert rv["verdict_rechecked"] == "PASS" and rv["agreement"]["matched"] == 3
    assert _prep(world, "replay") == rd                                   # same inputs: already prepared
    with pytest.raises(ws.WsError, match="other inputs"):
        _prep(world, "replay", revs={**world.revs, "experiments": "f" * 40})


def test_refuses_unauthorized_and_unknown_manifest(world):
    with pytest.raises(ws.WsError, match="NOT AUTHORIZED"):
        _prep(world, "arms")
    cfg = json.loads((world.repo / "data/runs/replay.json").read_text(encoding="utf-8"))
    cfg["inputs"]["records"]["manifest_sha256"] = "sha256:" + "1" * 64
    (world.repo / "data/runs/bad.json").write_text(json.dumps({**cfg, "run_id": "t-bad@1"}), encoding="utf-8")
    with pytest.raises(ws.WsError, match="registry"):
        _prep(world, "bad")
    with pytest.raises(ws.WsError, match="inside the checkout"):
        ws.prepare_run(config="../x.json", commit="a" * 40, revisions=world.revs, root=world.root,
                       repo=world.repo, hf=world.hf, skip_code_check=True)


def test_reproduce_matches_and_detects_a_different_model(world, no_gpu_checks):
    rd = _prep(world, "repro")
    res = ws.execute_run(rd, repo=world.repo, skip_code_check=True, adapter_factory=lambda c, checkpoint=None: Scorer())
    assert res["verdict"] == "REPRODUCED" and res["steps"] == res["steps_exact"] > 0
    rd2 = _prep(world, "repro", attempt=2)
    res2 = ws.execute_run(rd2, repo=world.repo, skip_code_check=True,
                          adapter_factory=lambda c, checkpoint=None: Scorer(salt="other"))
    assert res2["verdict"] == "NOT_REPRODUCED" and res2["n_mismatches"] > 0


def test_interrupted_run_needs_explicit_resume_and_skips_done_steps(world, no_gpu_checks):
    rd = _prep(world, "repro")
    with pytest.raises(KeyboardInterrupt):
        ws.execute_run(rd, repo=world.repo, skip_code_check=True,
                       adapter_factory=lambda c, checkpoint=None: Scorer(fail_after=3))
    assert ws._state(rd)["status"] == "INTERRUPTED" and (rd / "partial" / "steps.jsonl").exists()
    n_before = len((rd / "partial" / "steps.jsonl").read_text().splitlines())
    with pytest.raises(ws.WsError, match="--resume"):
        ws.execute_run(rd, repo=world.repo, skip_code_check=True, adapter_factory=lambda c, checkpoint=None: Scorer())
    with pytest.raises(ws.WsError, match="allow-incomplete"):
        ws.publish_results(rd, hf=world.hf)
    res = ws.execute_run(rd, resume=True, repo=world.repo, skip_code_check=True,
                         adapter_factory=lambda c, checkpoint=None: Scorer())
    rows = [json.loads(x) for x in (rd / "final" / "steps.jsonl").read_text().splitlines()]
    assert res["verdict"] == "REPRODUCED" and n_before > 0
    assert len(rows) == len({(r["scenario_id"], r["step"]) for r in rows})          # nothing done twice


def test_code_identity_requires_exact_clean_commit(tmp_path):
    def g(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)
    g("init", "-q")
    g("config", "user.email", "t@t")
    g("config", "user.name", "t")
    (tmp_path / "f.txt").write_text("1")
    g("add", ".")
    g("commit", "-qm", "x")
    head = ws.git(tmp_path, "rev-parse", "HEAD")
    assert ws.code_identity(tmp_path, head)["clean"]
    with pytest.raises(ws.WsError, match="not the requested"):
        ws.code_identity(tmp_path, "b" * 40)
    with pytest.raises(ws.WsError, match="40-hex"):
        ws.code_identity(tmp_path, "main")
    (tmp_path / "f.txt").write_text("2")
    with pytest.raises(ws.WsError, match="local changes"):
        ws.code_identity(tmp_path, head)


# ----- policy arms ------------------------------------------------------------------------------------------
def _obs_and_scores(seed):
    s = random_scenario(random.Random(seed))
    sim = Simulator(s)
    obs = sim.observe()
    rnd = random.Random(seed)
    sc = {}
    for p in obs["people"]:
        if p["status"] == "active":
            z = {a: rnd.uniform(-3, 0) for a in st.ACTION_ORDER}
            sc[p["id"]] = z
    return obs, sc


def test_coordinator_leaves_no_blocked_move_and_mask_only_admissible():
    from ember_v1.shield import admissible_actions
    checked = 0
    for seed in range(60):
        obs, sc = _obs_and_scores(seed)
        if not sc:
            continue
        adm = admissible_actions(obs)
        m = pa.static_mask(obs, sc)
        assert all(m[p] in adm[p] for p in m)
        joint, info = pa.coordinate(obs, sc)
        pg = PublicGrid.from_observation(obs)
        res = resolve_moves(height=pg.height, width=pg.width, walls=pg.walls, burning=pg.burning,
                            enter_fire=pg.enter_fire, positions=dict(pg.active), actions=joint)
        assert not [p for p, (o, _) in res.items() if o.startswith("blocked_")]
        assert all(joint[p] in adm[p] for p in joint)
        checked += 1
    assert checked > 20


def test_arm_a_equals_the_raw_student_and_arms_record_raw_and_executed(world, tmp_path):
    lst = world.rec_dir / "rollouts" / "list.jsonl"
    g = SimpleNamespace(wall_cap_s=1e9, usd_cap=1e9)
    res = pa.run_arms(Scorer(), world.cfg, lst, tmp_path / "arms", arms=pa.ARMS, label="V1", guards=lambda: g,
                      block=2, log=lambda s: None)
    assert res["stopped"] is None and all(r["finished"] == 3 for r in res["arms"].values())
    base = {json.loads(x)["scenario_id"]: json.loads(x)
            for x in (world.rec_dir / "eval-out" / "rollouts.jsonl").read_text().splitlines()}
    arms = {a: {json.loads(x)["scenario_id"]: json.loads(x)
                for x in (tmp_path / "arms" / f"rollouts-{a}.jsonl").read_text().splitlines()} for a in pa.ARMS}
    for sid, r in arms["A_raw"].items():
        assert r["trace"] == r["raw_trace"] == base[sid]["trace"]          # arm A = the unchanged student
        assert r["postprocess_changes"]["steps_changed"] == 0
    for a in ("B_static_mask", "C_joint_coord"):
        for r in arms[a].values():
            assert len(r["trace"]) == len(r["raw_trace"]) and r["arm"] == a
    assert sum(r["blocked"] for r in arms["C_joint_coord"].values()) == 0
    items = {json.loads(x)["scenario_id"]: json.loads(x) for x in lst.read_text().splitlines()}
    from ember_v1.sim import Scenario
    for r in arms["C_joint_coord"].values():                              # executed trace replays to the record
        rp = sr.replay_trace(Scenario.from_dict(items[r["scenario_id"]]["scenario"]), r["trace"],
                             {"question_style": world.cfg["question_style"]})
        assert {k: rp[k] for k in ("evacuated", "casualties", "unresolved")} == \
            {k: r[k] for k in ("evacuated", "casualties", "unresolved")}


def test_committed_run_configs_are_consistent_with_the_registry():
    repo = Path(ws.REPO_ROOT)
    reg = hs.load_registry(repo / hs.REGISTRY)
    for p in sorted((repo / "data" / "runs").glob("*.json")):
        cfg = json.loads(p.read_text(encoding="utf-8"))
        assert cfg["config_version"] == "ember-run-config@1" and cfg["kind"] in ws.KINDS
        for spec in cfg["inputs"].values():
            assert reg["artifacts"][spec["artifact_id"]]["manifest_sha256"] == spec["manifest_sha256"]
            assert reg["artifacts"][spec["artifact_id"]]["verification"] == "PASS"
        if cfg["kind"] == "v1-policy-ablation":
            assert cfg["authorization"]["status"] == "NOT_AUTHORIZED"
