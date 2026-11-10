"""P1-14 multi-teacher trainer on CPU: mock model, five-teacher synthetic fixture, no GPU, no real data.

Ported from the P1-11 harness tests of PR #44 (f9696a4). The T1-only S1 training set is gone: rows come only
from `student_ingest` (validated, weighted class-A whole joint maps of all five teachers).
"""
import copy
import json

import pytest

from ember_v1 import chunk_joint as cj
from ember_v1 import student_ingest as si
from ember_v1 import student_objective as so
from ember_v1 import student_train as st

from test_multiteacher_dataset import built  # noqa: F401  (module fixture: five-teacher synthetic room)


def tok(text):
    return text.split()


def _cost(e):
    return 10 + 3 * len(e["actions"])


@pytest.fixture(scope="module")
def frozen_file():
    return st.load_protocol()


@pytest.fixture(scope="module")
def proto(frozen_file):
    """A draft copy of the frozen file (in memory): the draft-refusal paths stay tested."""
    d = {**copy.deepcopy(frozen_file), "status": "DRAFT_NOT_FROZEN", "frozen": False,
         "authorizes": "nothing: draft copy"}
    return {**d, "protocol_hash": st.protocol_hash(d)}


@pytest.fixture(scope="module")
def cfg(proto):
    return st.config_from_protocol(proto)


@pytest.fixture()
def small_cfg(cfg):
    return {**cfg, "chunk_k": 2, "grad_accum_draws": 2, "checkpoint_every": 0}


@pytest.fixture()
def prepared(built):  # noqa: F811
    return si.prepare_examples(built[2])


@pytest.fixture()
def epoch(prepared):
    return si.sample_epoch(prepared["examples"], seed="train-test", token_budget=300, cost=_cost)


@pytest.fixture()
def rows(prepared, epoch, small_cfg):
    return st.rows_from_epoch(prepared, epoch, tok, small_cfg)


def _guards(**kw):
    return st.Guards(kw.get("wall", 1e9), kw.get("usd", 1e9), 1.545, **({"clock": kw["clock"]} if "clock" in kw else {}))


# ----- protocol and configuration ----------------------------------------------------------------------------
def test_protocol_file_is_frozen_five_teacher_with_measured_tokens(frozen_file, proto):
    rep = st.check_protocol(frozen_file, require_frozen=True)
    assert rep["protocol_hash"] == frozen_file["protocol_hash"] == st.protocol_hash(frozen_file)
    assert frozen_file["status"] == "FROZEN" and frozen_file["frozen"] is True
    assert frozen_file["teachers"]["demonstrations_from"] == ["T0", "T1", "T2", "T3", "T4"]
    tm = frozen_file["data"]["token_measurement"]
    assert tm["all_rows_ok"] is True and tm["max_row_tokens"] < frozen_file["pins"]["max_len"]
    assert frozen_file["data"]["token_budget_per_epoch"].startswith("RULE")
    with pytest.raises(st.StudentTrainProtocolError, match="not frozen"):
        st.check_protocol(proto, require_frozen=True)


@pytest.mark.parametrize("mutate,match", [
    (lambda p: p["teachers"].update(demonstrations_from=["T1"]), "five teachers"),
    (lambda p: p["data"].update(source="s1"), "student_ingest"),
    (lambda p: p.update(status="FROZEN"), "inconsistent"),
    (lambda p: p["compute"]["phases"]["train"].update(usd_cap=1.0), "wall cap costs more"),
    (lambda p: p["compute"].update(total_usd_cap=5.0), "exceed total_usd_cap"),
])
def test_protocol_checks_fail_closed(proto, mutate, match):
    p = copy.deepcopy(proto)
    mutate(p)
    p["protocol_hash"] = st.protocol_hash(p)
    with pytest.raises(st.StudentTrainProtocolError, match=match):
        st.check_protocol(p, require_frozen=False)
    q = copy.deepcopy(proto)
    q["training"]["lr"] = 1.0                 # content change without a new hash
    with pytest.raises(st.StudentTrainProtocolError, match="protocol_hash"):
        st.check_protocol(q, require_frozen=False)


def test_config_is_the_protocol(cfg, proto):
    assert cfg["model"] == "unsloth/Qwen3.5-2B" and cfg["model_commit"] == proto["pins"]["model_commit"]
    assert cfg["lora"] == {"r": 16, "lora_alpha": 16, "lora_dropout": 0.0, "target_modules": "all-linear", "bias": "none"}
    assert (cfg["epochs"], cfg["micro_batch_rows"], cfg["grad_accum_draws"], cfg["checkpoint_every"]) == (1, 1, 8, 250)
    assert (cfg["chunk_k"], cfg["max_len"], cfg["reservations"]) == (100, 16384, False)
    for k in ("tokenizer_json_sha256", "requirements_lock_sha256", "unsloth_wheel_sha256", "clef_sha256"):
        assert cfg[k] == proto["pins"][k]


def test_schedule_counts_draws(cfg):
    sch = st.schedule(1000, cfg)
    assert sch == {"draws": 1000, "optimizer_steps": 125, "warmup_steps": 4}
    assert st.lr_at(0, 1.0, sch) == pytest.approx(0.25) and st.lr_at(3, 1.0, sch) == pytest.approx(1.0)
    assert st.lr_at(124, 1.0, sch) < 1e-3 and st.lr_at(64, 1.0, sch) < st.lr_at(10, 1.0, sch)
    assert st.schedule(10, cfg, 3)["optimizer_steps"] == 4


# ----- data path ---------------------------------------------------------------------------------------------
def test_only_ingested_examples_are_accepted(prepared, epoch, small_cfg, built):  # noqa: F811
    for bad in (built[2], {"examples": prepared["examples"]}, None, {**prepared, "version": "s1"}):
        with pytest.raises(st.TrainError, match="ingest_gate"):
            st.rows_from_epoch(bad, epoch, tok, small_cfg)
    with pytest.raises(st.TrainError, match="ingest_gate"):
        st.rows_from_epoch(prepared, {**epoch, "draws": list(reversed(epoch["draws"])) + epoch["draws"][:1]},
                           tok, small_cfg)
    with pytest.raises(st.TrainError, match="reservations"):
        st.rows_from_epoch(prepared, epoch, tok, small_cfg, reservation_fn=lambda o, d: "")


def test_rows_are_the_objective_rows_with_questions(rows, prepared, epoch, small_cfg):
    base = si.epoch_rows(prepared["examples"], epoch, k=small_cfg["chunk_k"])
    assert len(rows) == len(base)
    for r, b in zip(rows, base):
        assert (r["state"], r["gold"], r["draw"], r["example_id"]) == (b["state"], b["targets"], b["draw"], b["example_id"])
        assert list(r["questions"]) == b["ids"] and r["tokens"] < 16384
        assert "teacher" not in json.dumps(r) and "weight" not in r
    for g in st._draw_groups(rows):
        assert sum(r["row_share"] for r in g) == pytest.approx(1.0)
    with pytest.raises(st.TrainError, match="row_status_not_ok"):
        st.rows_from_epoch(prepared, epoch, tok, {**small_cfg, "max_len": 50})


def test_trainer_loss_equals_the_exact_sft_objective(rows, prepared, small_cfg, tmp_path):
    """One epoch with lr 0: the reported first loss is the mean over the first grad_accum draws of
    student_objective.sft_nll under the same (mock) logits."""
    m = st.MockDecisionAdapter(3)
    by = {e["example_id"]: e for e in prepared["examples"]}
    groups = st._draw_groups(rows)

    def scorer(row):
        return {pid: m._logits(m._key({"state": row["state"]}, pid)) for pid in row["ids"]}
    want = []
    for g in groups[:small_cfg["grad_accum_draws"]]:
        e = by[g[0]["example_id"]]
        want.append(so.sft_nll(scorer, e["observation"], e["actions"], si.id_chunks(e["observation"], 2))["loss"])
    rep = st.train(m, rows, {**small_cfg, "lr": 0.0, "head_lr": 0.0}, guards=_guards(), out_dir=tmp_path / "x",
                   mode="s0_fixture")
    assert rep["meter"]["first_loss"] == pytest.approx(sum(want) / len(want), abs=1e-12)
    assert rep["meter"]["draws"] == len(groups) and rep["meter"]["rows"] == len(rows)


# ----- training loop (from #44) ------------------------------------------------------------------------------
def test_s0_overfit_reaches_full_accuracy(rows, small_cfg, tmp_path):
    m = st.MockDecisionAdapter(0)
    rep = st.train(m, rows, {**small_cfg, "head_lr": 20.0}, guards=_guards(), out_dir=tmp_path / "s0",
                   mode="s0_fixture", epochs=40)
    assert rep["status"] == "COMPLETED"
    acc = sum(m.predict(r)[i] == r["gold"][i] for r in rows for i in r["gold"]) / sum(len(r["gold"]) for r in rows)
    assert acc == 1.0
    assert rep["meter"]["last_loss"] < rep["meter"]["first_loss"] / 10
    assert rep["meter"]["tokens"] == 40 * sum(r["tokens"] for r in rows) and rep["meter"]["tokens_per_s"] > 0
    assert rep["meter"]["peak_memory_bytes"] > 0
    assert json.loads((tmp_path / "s0" / "train_report.json").read_text())["status"] == "COMPLETED"


def test_one_seed_is_reproducible(prepared, small_cfg, tmp_path):
    def run(seed, i):
        ep = si.sample_epoch(prepared["examples"], seed=seed, token_budget=300, cost=_cost)
        rows = st.rows_from_epoch(prepared, ep, tok, small_cfg)
        return st.train(st.MockDecisionAdapter(0), rows, small_cfg, guards=_guards(), out_dir=tmp_path / f"r{i}",
                        mode="s0_fixture", epochs=3)["final_weights_digest"]
    assert run("a", 0) == run("a", 1) != run("b", 2)


def test_protocol_mode_refuses_the_draft(rows, cfg, proto, tmp_path):
    with pytest.raises(st.StudentTrainProtocolError, match="not frozen"):
        st.train(st.MockDecisionAdapter(), rows, cfg, guards=_guards(), out_dir=tmp_path / "p", mode="protocol",
                 protocol=proto)
    assert not (tmp_path / "p").exists()


def test_s0_mode_is_cpu_and_small_only(rows, small_cfg, tmp_path):
    many = [{**r, "draw": r["draw"] + 1000 * k} for k in range(70) for r in rows]
    with pytest.raises(st.TrainError, match="s0_fixture"):
        st.train(st.MockDecisionAdapter(), many, small_cfg, guards=_guards(), out_dir=tmp_path / "a", mode="s0_fixture")
    gpu = st.MockDecisionAdapter()
    gpu.device = "cuda"
    with pytest.raises(st.TrainError, match="s0_fixture"):
        st.train(gpu, rows, small_cfg, guards=_guards(), out_dir=tmp_path / "b", mode="s0_fixture")
    with pytest.raises(st.TrainError, match="padding gate"):
        st.train(st.MockDecisionAdapter(), rows, {**small_cfg, "micro_batch_rows": 2}, guards=_guards(),
                 out_dir=tmp_path / "c", mode="s0_fixture")


def test_partial_or_shuffled_draws_are_refused(rows, small_cfg, tmp_path):
    multi = [g for g in st._draw_groups(rows) if len(g) > 1]
    assert multi, "fixture must have a decision split into several chunk rows"
    with pytest.raises(st.TrainError, match="cover all"):
        st.train(st.MockDecisionAdapter(), [r for r in rows if r is not multi[0][0]], small_cfg, guards=_guards(),
                 out_dir=tmp_path / "p", mode="s0_fixture")
    g0 = multi[0]
    other = [r for r in rows if r["draw"] != g0[0]["draw"]]
    with pytest.raises(st.TrainError, match="contiguous"):
        st.train(st.MockDecisionAdapter(), [g0[0]] + other + g0[1:], small_cfg, guards=_guards(),
                 out_dir=tmp_path / "q", mode="s0_fixture")


class _NaN(st.MockDecisionAdapter):
    def loss_and_backward(self, row, scale):
        return float("nan")


def test_nonfinite_loss_writes_failure_report(rows, small_cfg, tmp_path):
    with pytest.raises(st.TrainError, match="loss_not_finite"):
        st.train(_NaN(), rows, small_cfg, guards=_guards(), out_dir=tmp_path / "n", mode="s0_fixture")
    rep = json.loads((tmp_path / "n" / "failure_report.json").read_text())
    assert rep["status"] == "FAILED" and rep["reason"] == "loss_not_finite" and rep["stage"] == "train"


@pytest.mark.parametrize("kw,reason", [({"wall": 10.0}, "wall_cap_reached"), ({"usd": 0.001}, "usd_cap_reached")])
def test_wall_and_cost_guards_stop(rows, small_cfg, tmp_path, kw, reason):
    t = iter(range(0, 10 ** 6, 5))
    g = _guards(clock=lambda: float(next(t)), **kw)
    with pytest.raises(st.TrainError, match=reason):
        st.train(st.MockDecisionAdapter(), rows, small_cfg, guards=g, out_dir=tmp_path / "g", mode="s0_fixture", epochs=5)
    assert json.loads((tmp_path / "g" / "failure_report.json").read_text())["reason"] == reason


def test_checkpoints_and_reload_integrity(rows, small_cfg, tmp_path):
    m = st.MockDecisionAdapter(0)
    rep = st.train(m, rows, {**small_cfg, "checkpoint_every": 2}, guards=_guards(), out_dir=tmp_path / "c",
                   mode="s0_fixture", epochs=3)
    names = [c["path"] for c in rep["checkpoints"]]
    assert names[-1] == "final" and names[0] == "ckpt-000002"
    assert all(c["files"]["weights.json"].startswith("sha256:") for c in rep["checkpoints"])
    assert st.save_reload_check(m, st.MockDecisionAdapter.load, tmp_path / "reload", rows)["pass"]

    def broken(path):
        x = st.MockDecisionAdapter.load(path)
        k = next(iter(x.w))
        x.w[k] = [v + 1.0 if j == 0 else v for j, v in enumerate(x.w[k])]
        return x
    with pytest.raises(st.TrainError, match="reload_integrity"):
        st.save_reload_check(m, broken, tmp_path / "reload2", rows)


def test_write_once_output(rows, small_cfg, tmp_path):
    (tmp_path / "w").mkdir()
    with pytest.raises(st.TrainError, match="write-once"):
        st.train(st.MockDecisionAdapter(), rows, small_cfg, guards=_guards(), out_dir=tmp_path / "w", mode="s0_fixture")


def test_student_joint_validation(prepared):
    obs = prepared["examples"][0]["observation"]
    active = [p["id"] for p in obs["people"] if p["status"] == "active"]
    m = st.MockDecisionAdapter(1)
    rep = st.validate_student_joint(obs, 2, m.predict, tok)
    assert rep["chunks"] == -(-len(active) // 2) and set(rep["joint"]) == set(active)
    assert sum(rep["outcomes"].values()) == len(rep["joint"])

    def drops_one(row):
        p = m.predict(row)
        p.pop(next(iter(p)))
        return p
    with pytest.raises(cj.ChunkError):
        st.validate_student_joint(obs, 2, drops_one, tok)


# ----- cost and GPU gates -------------------------------------------------------------------------------------
def test_pilot_go_rule(proto):
    ok = st.pilot_go_rule(proto, tokens_per_epoch_actual=50_000_000, measured_tokens_per_s=3700,
                          peak_memory_gb=12.0, loss_finite=True)
    assert ok["go"] and ok["projected_train_usd"] == pytest.approx(5.80, abs=0.01)
    assert not st.pilot_go_rule(proto, tokens_per_epoch_actual=200_000_000, measured_tokens_per_s=1900,
                                peak_memory_gb=12.0, loss_finite=True)["go"]
    assert not st.pilot_go_rule(proto, tokens_per_epoch_actual=1, measured_tokens_per_s=1, peak_memory_gb=23.0,
                                loss_finite=True)["go"]


def _frozen(proto):
    p = copy.deepcopy(proto)
    p["status"], p["frozen"] = "FROZEN", True
    p["protocol_hash"] = st.protocol_hash(p)
    return p


def test_gpu_plan_needs_frozen_protocol_and_authorization(proto):
    with pytest.raises(st.StudentTrainProtocolError, match="not frozen"):
        st.gpu_train_plan("pilot", proto, authorize="pilot", code_commit="a" * 40)
    fz = _frozen(proto)                       # in-memory only: never written, never committed
    with pytest.raises(PermissionError):
        st.gpu_train_plan("train", fz, authorize="pilot", code_commit="a" * 40)
    with pytest.raises(st.TrainError, match="config"):
        st.gpu_train_plan("pilot", fz, authorize="pilot", code_commit="dirty")
    plan = st.gpu_train_plan("pilot", fz, authorize="pilot", code_commit="a" * 40)
    assert plan["timeout_s"] == 3600 and plan["retries"] == 0 and plan["max_containers"] == 1
    assert plan["billing_cap_enforced_by_modal"] is False


def test_t1_only_protocol_cannot_reach_gpu_even_if_check_is_bypassed(proto, monkeypatch):
    monkeypatch.setattr(st, "check_protocol", lambda *a, **k: {})
    fz = _frozen(proto)
    fz["teacher"] = {"sole_teacher": True}
    with pytest.raises(st.TrainError, match="superseded_training_goal"):
        st.gpu_train_plan("pilot", fz, authorize="pilot", code_commit="a" * 40)


def test_runtime_pins(cfg):
    meta = {"model_name": cfg["model"], "model_revision": f"{cfg['model']}@{cfg['model_commit']}",
            "tokenizer_sha256": cfg["tokenizer_state_sha256"], "encoder_config_sha256": cfg["encoder_config_sha256"],
            "head_sha256": cfg["head_init_sha256"], "precision": "bf16", "token_limit": 16384,
            "upstream_contract": {"clef_matches_audited": True}}
    st.verify_runtime_pins(meta, cfg, image_lock_sha256=cfg["requirements_lock_sha256"])
    for k, v in (("model_revision", "0" * 40), ("tokenizer_sha256", "x"), ("token_limit", 8192)):
        with pytest.raises(st.TrainError, match="identity_or_pin_drift"):
            st.verify_runtime_pins({**meta, k: v}, cfg, image_lock_sha256=cfg["requirements_lock_sha256"])
    with pytest.raises(st.TrainError, match="image_lock"):
        st.verify_runtime_pins(meta, cfg, image_lock_sha256="0" * 64)


def test_pins_match_the_audited_head():
    from ember_v1 import p0b
    p = st.load_protocol()
    assert p["pins"]["model"] == p0b.PINNED_MODEL and p["pins"]["clef_sha256"] == p0b.AUDITED_CLEF_SHA256


def test_v2_protocol_changes_only_data_and_caps():
    v1 = st.load_protocol()
    v2 = st.load_protocol(st.REPO / "data/protocols/p1-18-student-v2.json")
    st.check_protocol(v2, require_frozen=True)
    a, b = st.config_from_protocol(v1), st.config_from_protocol(v2)
    diff = {k for k in a if a[k] != b[k]}
    assert diff == {"protocol_hash", "epoch_seed", "phases"}
    assert v2["based_on"]["protocol_hash"] == v1["protocol_hash"]
    assert sum(ph["usd_cap"] for ph in v2["compute"]["phases"].values()) <= 70.0
    assert "128,701,152" in v2["data"]["token_budget_per_epoch"]
