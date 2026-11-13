"""student_run: token measurement, published epoch, and the GPU phase runner on a CPU mock adapter."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from pathlib import Path

import pytest

from ember_v1 import student_ingest as si
from ember_v1 import student_run as sr
from ember_v1 import student_train as st
from ember_v1.sim import Simulator
from ember_v1.sim.canonical import digest
from gen_helpers import random_scenario
from ember_v1.teacher.t0 import T0ShortestPath


def _tok(s: str) -> tuple[int, ...]:
    return tuple(range(len(s) // 4 + 1))


def _sha(p: Path) -> str:
    return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def sft(tmp_path_factory):
    d = tmp_path_factory.mktemp("sft")
    pol = T0ShortestPath()
    out = {"train": [], "val": []}
    for i in range(6):
        sim = Simulator(random_scenario(random.Random(100 + i)))
        for _ in range(3):
            if sim.terminated:
                break
            obs = sim.observe()
            dec = pol.decide(obs)
            ex = {"observation": obs, "actions": dict(dec.actions), "weight": 0.5 + i / 10}
            ex = {"example_id": si.example_id(ex), **ex}
            out["val" if i == 5 else "train"].append(ex)
            sim.step(dec.actions)
    for split, exs in out.items():
        (d / f"{split}.jsonl").write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in exs),
                                          encoding="utf-8", newline="\n")
    body = {"k": 100, "dataset_hash": "sha256:" + "0" * 64,
            "files": {n: _sha(d / n) for n in ("train.jsonl", "val.jsonl")}}
    (d / "sft_manifest.json").write_text(json.dumps({**body, "manifest_hash": digest(body)}), encoding="utf-8")
    return d


@pytest.fixture(scope="module")
def tokens(sft, tmp_path_factory, monkeypatch_module):
    monkeypatch_module.setattr(sr, "hf_tokenizer", lambda p: _tok)
    out = tmp_path_factory.mktemp("t") / "tok"
    rep = sr.measure(sft, Path("unused"), out, log=lambda s: None)
    assert rep["all_rows_ok"] and (out / "token_summary.json").exists()
    return out


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def frozen():
    return st.load_protocol()


class ScoringMock(st.MockDecisionAdapter):
    def scores(self, row):
        out = {}
        for pid in row["questions"]:
            z = self._logits(self._key(row, pid))
            m = max(z)
            lse = m + math.log(sum(math.exp(v - m) for v in z))
            out[pid] = {a: z[j] - lse for j, a in enumerate(st.ACTION_ORDER)}
        return out

    def load_state(self, path):
        self.w = ScoringMock.load(Path(path)).w

    @classmethod
    def load(cls, path):
        d = json.loads((Path(path) / "weights.json").read_text(encoding="utf-8"))
        m = cls(d["seed"])
        m.w = {k: list(v) for k, v in d["w"].items()}
        return m


def _factory(cfg, checkpoint=None):
    return ScoringMock(0) if checkpoint is None else ScoringMock.load(checkpoint)


def test_measure_counts_every_row_and_refuses_overlong(sft, tokens, tmp_path, monkeypatch):
    summ = json.loads((tokens / "token_summary.json").read_text(encoding="utf-8"))
    lines = (tokens / "token_counts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == summ["splits"]["train"]["examples"] + summ["splits"]["val"]["examples"]
    monkeypatch.setattr(sr, "hf_tokenizer", lambda p: (lambda s: tuple(range(len(s) * 10))))
    with pytest.raises(sr.RunError, match="not OK"):
        sr.measure(sft, Path("unused"), tmp_path / "x", log=lambda s: None)


def test_epoch_is_a_prefix_for_a_smaller_budget(sft, tokens, tmp_path):
    big = sr.build_epoch(sft, tokens, seed="s", token_budget=200_000, out=tmp_path / "big.json")
    small = sr.build_epoch(sft, tokens, seed="s", token_budget=20_000, out=tmp_path / "small.json")
    b = json.loads((tmp_path / "big.json").read_text())["draws"]
    s = json.loads((tmp_path / "small.json").read_text())["draws"]
    assert big["n_draws"] > small["n_draws"] > 0 and b[:len(s)] == s


def _spec(sft, tokens, tmp_path, budget, name):
    sr.build_epoch(sft, tokens, seed="s", token_budget=budget, out=tmp_path / f"{name}.json")
    return {"sft_dir": str(sft), "epoch": str(tmp_path / f"{name}.json"), "out": str(tmp_path / name),
            "token_counts": str(tokens / "token_counts.jsonl"), "tokenize": _tok, "probe_val_examples": 2,
            "full_pass_tokens": 1000}


def test_pilot_phase_trains_reloads_and_projects(sft, tokens, frozen, tmp_path):
    rep = sr.gpu_phase("pilot", frozen, _spec(sft, tokens, tmp_path, 30_000, "p"), adapter_factory=_factory,
                       log=lambda s: None)
    assert rep["data"]["host_vs_local_token_mismatch_rows"] == 0
    assert rep["train"]["status"] == "COMPLETED" and rep["reload"]["pass"] is True
    assert rep["go"]["go"] in (True, False) and (tmp_path / "p" / "pilot_report.json").exists()


@pytest.mark.parametrize("cuda_available", [False, True])
def test_pilot_resets_cuda_stats_only_when_available(sft, tokens, frozen, tmp_path, monkeypatch, cuda_available):
    torch = pytest.importorskip("torch")
    resets = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    def reset_peak():
        assert cuda_available, "CPU mock runs must not reset CUDA allocator statistics"
        resets.append(True)

    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset_peak)
    rep = sr.gpu_phase("pilot", frozen, _spec(sft, tokens, tmp_path, 30_000, "p"),
                       adapter_factory=_factory, log=lambda s: None)
    assert rep["reload"]["pass"] is True
    assert len(resets) == (1 if cuda_available else 0)


def test_pilot_refuses_a_draft_protocol(sft, tokens, frozen, tmp_path):
    d = {**copy.deepcopy(frozen), "status": "DRAFT_NOT_FROZEN", "frozen": False}
    d["protocol_hash"] = st.protocol_hash(d)
    with pytest.raises(st.StudentTrainProtocolError, match="not frozen"):
        sr.gpu_phase("pilot", d, _spec(sft, tokens, tmp_path, 30_000, "d"), adapter_factory=_factory)


def test_train_resumes_from_its_checkpoint_to_the_same_weights(sft, tokens, frozen, tmp_path, monkeypatch):
    cfg = st.config_from_protocol(frozen)
    monkeypatch.setattr(st, "config_from_protocol", lambda p: {**cfg, "grad_accum_draws": 2, "checkpoint_every": 2})
    spec = _spec(sft, tokens, tmp_path, 200_000, "full")
    ref = sr.gpu_phase("train", frozen, spec, adapter_factory=_factory, log=lambda s: None)
    # interrupted run: stop right after the first checkpoint, then resume in a "new container"
    spec2 = {**spec, "out": str(tmp_path / "cut")}
    real_train = st.train

    def cut(*a, on_checkpoint=None, **k):
        def stop(c):
            on_checkpoint(c)
            raise KeyboardInterrupt
        return real_train(*a, on_checkpoint=stop, **k)
    monkeypatch.setattr(st, "train", cut)
    with pytest.raises(KeyboardInterrupt):
        sr.gpu_phase("train", frozen, spec2, adapter_factory=_factory, log=lambda s: None)
    monkeypatch.setattr(st, "train", real_train)
    res = sr.gpu_phase("train", frozen, spec2, adapter_factory=_factory, log=lambda s: None)
    assert res["resumed_from"].endswith("ckpt-000002")
    assert res["train"]["final_weights_digest"] == ref["train"]["final_weights_digest"]
    assert res["train"]["meter"]["draws"] == ref["train"]["meter"]["draws"]


def test_eval_phase_scores_val_and_rolls_out_resumably(sft, tokens, frozen, tmp_path):
    rep = sr.gpu_phase("pilot", frozen, _spec(sft, tokens, tmp_path, 30_000, "p"), adapter_factory=_factory,
                       log=lambda s: None)
    ck = str(tmp_path / "p" / "run" / "final")
    lst = tmp_path / "scen.jsonl"
    with open(lst, "w", encoding="utf-8") as f:
        for i in range(3):
            s = random_scenario(random.Random(500 + i))
            f.write(json.dumps({"set": "t", "scenario_id": f"s{i}", "family_id": f"f{i}",
                                "scenario": s.to_dict(), "scenario_hash": s.content_hash()}) + "\n")
    spec = {"sft_dir": str(sft), "checkpoint": ck, "out": str(tmp_path / "ev"), "rollouts": str(lst),
            "rollout_limit": 2, "label": "mock"}
    ev = sr.gpu_phase("eval", frozen, spec, adapter_factory=_factory, log=lambda s: None)
    assert ev["val"]["people"] > 0 and ev["rollouts"]["finished"] == 2 and ev["rollouts"]["stopped"] is None
    recs = [json.loads(x) for x in (tmp_path / "ev" / "rollouts.jsonl").read_text().splitlines()]
    assert all(r["people"] == r["evacuated"] + r["casualties"] + r["unresolved"] for r in recs)
    from ember_v1.sim import Scenario
    items = {json.loads(x)["scenario_id"]: json.loads(x) for x in lst.read_text().splitlines()}
    for r in recs:                     # the trace rebuilds every student-visited state and the same outcome
        seen = []
        rp = sr.replay_trace(Scenario.from_dict(items[r["scenario_id"]]["scenario"]), r["trace"],
                             {"question_style": frozen["data"]["question_style"]},
                             visit=lambda t, o, a: seen.append(t))
        assert rp["steps"] == r["decisions"] == len(seen)
        assert {k: rp[k] for k in ("evacuated", "casualties", "unresolved")} == \
            {k: r[k] for k in ("evacuated", "casualties", "unresolved")}
    ev2 = sr.gpu_phase("eval", frozen, {**spec, "rollout_limit": 3}, adapter_factory=_factory, log=lambda s: None)
    assert ev2["rollouts"]["ran_this_container"] == 1 and ev2["rollouts"]["finished"] == 3
    assert rep["reload"]["pass"]
