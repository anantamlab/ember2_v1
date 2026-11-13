"""Issue #2 preparation (CPU only): install lock, install check, GPU spend guard, inspect records.

No Modal, GPU, torch or model download is used.
"""
import re
import sys
import types
from pathlib import Path

import pytest

import p0b_fakes as fakes
from ember_v1 import p0b
from p0b_fakes import CharTokenizer, FakeModel, provenance

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "infra" / "p0b-requirements.lock.txt"
LOCK_INPUT = ROOT / "infra" / "p0b-requirements.in"
CLEAN = "f" * 40


@pytest.fixture(autouse=True)
def _no_cuda(monkeypatch):
    monkeypatch.setattr(p0b, "_sync", lambda: False)


def _norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def _lock_entries():
    """{normalized name: (version, hash count)} of the hash-locked requirements file."""
    entries, current = {}, None
    for line in LOCK.read_text(encoding="utf-8").splitlines():
        if match := re.match(r"^([A-Za-z0-9_.\-]+)==(\S+)", line):
            current = _norm(match.group(1))
            entries[current] = [match.group(2), 0]
        elif "--hash=sha256:" in line and current:
            entries[current][1] += 1
    return {k: tuple(v) for k, v in entries.items()}


# ------------------------------------------------------------ dependency lock
def test_lock_pins_every_package_with_a_hash():
    entries = _lock_entries()
    assert len(entries) > 50
    assert all(count >= 1 for _, count in entries.values()), "every locked package needs a sha256"
    for line in LOCK.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith((" ", "#", "-")):
            assert "==" in line and "@" not in line, f"not an exact index pin: {line}"


def test_lock_matches_pinned_runtime_and_inputs():
    entries = _lock_entries()
    for pkg, version in p0b.PINNED_RUNTIME.items():
        assert entries[_norm(pkg)][0] == version, pkg
    for line in LOCK_INPUT.read_text(encoding="utf-8").splitlines():
        if match := re.match(r"^([A-Za-z0-9_.\-]+)(?:\[[^\]]+\])?==(\S+)$", line.strip()):
            assert entries[_norm(match.group(1))][0] == match.group(2), line


def test_torchao_is_built_for_the_locked_torch():
    # torchao 0.16.0 targets torch 2.10 and skips its C++ extensions on torch 2.8.
    assert _lock_entries()["torch"][0] == "2.8.0" and _lock_entries()["torchao"][0] in ("0.13.0", "0.14.0")


# ------------------------------------------------------------ install check
def test_contract_without_installed_unsloth_never_matches(monkeypatch):
    monkeypatch.setattr(p0b, "_installed_file", lambda relative: None)
    contract = p0b._upstream_contract()
    assert contract["all_match_audited"] is False
    assert all(contract[f"{name}_sha256"] is None for name in p0b.AUDITED_UPSTREAM_FILES)
    assert all(contract[f"{name}_matches_audited"] is False for name in p0b.AUDITED_UPSTREAM_FILES)
    assert set(p0b.AUDITED_UPSTREAM_FILES) <= {k.removesuffix("_matches_audited") for k in contract}


def _installed(monkeypatch, versions=None, contract=None):
    pins = {**p0b.PINNED_RUNTIME, **(versions or {})}
    monkeypatch.setattr(p0b, "_version", lambda pkg: pins.get(pkg))
    monkeypatch.setattr(p0b, "_upstream_contract", lambda: contract or fakes.audited_contract())


def test_verify_install_passes_only_on_the_exact_stack(monkeypatch, capsys):
    _installed(monkeypatch)
    assert p0b.verify_install()["status"] == "PASS"
    assert p0b.main(["verify-install"]) == 0 and '"status": "PASS"' in capsys.readouterr().out


@pytest.mark.parametrize("versions, contract", [
    ({"transformers": "5.3.0"}, None),
    ({"unsloth": None}, None),
    ({}, {**fakes.audited_contract(), "clef_matches_audited": False, "all_match_audited": False}),
])
def test_verify_install_fails_on_drift(monkeypatch, versions, contract):
    _installed(monkeypatch, versions, contract)
    assert p0b.verify_install()["status"] == "FAIL"
    assert p0b.main(["verify-install"]) == 2


def test_verify_install_cli_rejects_other_arguments():
    assert p0b.main([]) == 64 and p0b.main(["inspect"]) == 64


# ------------------------------------------------------------ GPU spend guard
def test_inspect_plan_keeps_the_reviewed_resource_limits():
    plan = p0b.gpu_run_plan("inspect", "L4", p0b.PINNED_MODEL, "bf16", "inspect", CLEAN)
    assert plan["model"] == "unsloth/Qwen3.5-2B" and plan["revision"] == "fffbc0d8711cf87df6d1f97938c5e50ef94aadc7"
    assert plan["gpu"] == "L4" and plan["code_commit"] == CLEAN
    assert (plan["startup_timeout_s"], plan["timeout_s"]) == (900, 600)
    assert (plan["retries"], plan["max_containers"], plan["scaledown_window_s"]) == (0, 1, 2)
    assert "worst_case_usd" not in plan and "hard_cap_usd" not in plan  # no cap wording


def test_cost_estimate_is_labeled_and_separated():
    est = p0b.gpu_run_plan("inspect", "L4", p0b.PINNED_MODEL, "bf16", "inspect", CLEAN)["cost_estimate"]
    assert est["kind"] == "ESTIMATE_NOT_A_BILLING_CAP" and "2026-11-08" in est["prices"]
    assert est["configured_container_s"] == 1500
    assert est["gpu_usd"] == pytest.approx(1500 * 0.000222, abs=1e-3)
    assert est["cpu_usd_assumed"] == pytest.approx(1500 * 4 * 0.0000131, abs=1e-3)
    assert est["memory_usd_assumed"] == pytest.approx(1500 * 32 * 0.00000222, abs=1e-3)
    assert est["envelope_usd"] == pytest.approx(est["gpu_usd"] + est["cpu_usd_assumed"] + est["memory_usd_assumed"],
                                                abs=0.01)
    assert est["envelope_usd"] == 0.52 < p0b.P0B_CAP_USD
    excluded = " ".join(est["excludes"])
    assert "image build" in excluded and "storage" in excluded  # charged apart, not estimated here
    assert any("retries=0" in a for a in est["assumes"]) and any("preemption" in a for a in est["assumes"])


def test_l40s_estimate_uses_its_own_price():
    l4 = p0b.cost_estimate("inspect", "L4")
    l40s = p0b.cost_estimate("inspect", "L40S")
    assert l40s["gpu_usd"] == pytest.approx(1500 * 0.000542, abs=1e-3) and l40s["envelope_usd"] > l4["envelope_usd"]


def test_a10_is_allowed_between_l4_and_l40s_with_its_own_price():
    assert p0b.ALLOWED_GPUS == ("L4", "A10", "L40S")
    a10 = p0b.gpu_run_plan("inspect", "A10", p0b.PINNED_MODEL, "bf16", "inspect", CLEAN)["cost_estimate"]
    assert a10["gpu_usd"] == pytest.approx(1500 * 1.10 / 3600, abs=1e-3)
    env = [p0b.cost_estimate("inspect", g)["envelope_usd"] for g in p0b.ALLOWED_GPUS]
    assert env == sorted(env) and env[-1] < 2.0          # one bounded attempt on any allowed GPU < $2 estimate
    for g in ("A100", "H100", "T4", "a10", "A10G"):
        with pytest.raises(ValueError, match="gpu must be one of"):
            p0b.gpu_run_plan("inspect", g, p0b.PINNED_MODEL, "bf16", "inspect", CLEAN)


def test_allocated_gpu_comes_from_cuda_metadata_only():
    assert p0b.allocated_gpu("A10", {"gpu": "NVIDIA A10G", "gpu_capability": "8.6"})["matches_request"] is True
    assert p0b.allocated_gpu("L4", {"gpu": "NVIDIA L40S", "gpu_capability": "8.9"})["matches_request"] is False
    none = p0b.allocated_gpu("L4", None)
    assert none["matches_request"] is None and none["device_name"] is None and none["source"] == "none"
    assert set(p0b.GPU_DEVICE_NAMES) == set(p0b.ALLOWED_GPUS)


def test_modal_app_uses_the_plan_limits():
    # Source check (CI has no modal): the decorator and with_options take the shared constants.
    src = (ROOT / "infra" / "modal_app.py").read_text(encoding="utf-8")
    for needle in ("startup_timeout=STARTUP_TIMEOUT_S", "retries=RETRIES", "max_containers=MAX_CONTAINERS",
                   "scaledown_window=SCALEDOWN_WINDOW_S", 'timeout=plan["timeout_s"]', 'retries=plan["retries"]',
                   'max_containers=plan["max_containers"]', 'scaledown_window=plan["scaledown_window_s"]',
                   "watch_remote_call(", "get_current_stats", "app_id=app.app_id", "write_json_atomic(path, result)"):
        assert needle in src, needle
    assert "batched_forward: bool = modal.parameter(default=False)" in src
    # Harness tasks are submitted once with spawn() (bounded wait), never with a blocking remote().
    for task in ("inspect", "smoke", "throughput", "padding"):
        assert f"harness.{task}.spawn(" in src and f"harness.{task}.remote(" not in src


VERIFY_CMD = '.run_commands("PYTHONPATH=/root python -m ember_v1.p0b verify-install")'


def test_modal_image_runs_verify_install_with_root_on_pythonpath():
    # First inspect (2026-11-03) failed: the builder RUN step does not have /root on sys.path.
    src = (ROOT / "infra" / "modal_app.py").read_text(encoding="utf-8")
    copy = src.index('add_local_python_source("ember_v1", copy=True)')
    assert src.count("verify-install") == 1 and src.index(VERIFY_CMD) > copy


def test_modal_app_reads_local_files_only_on_the_client():
    # A10 attempt (2026-11-08): the container imports /root/modal_app.py, where the lock file does not
    # exist; a module-level read failed every startup. Source check, for CI without modal.
    import ast
    src = (ROOT / "infra" / "modal_app.py").read_text(encoding="utf-8")
    reads = {"read_bytes", "read_text", "open"}
    top = [n for n in ast.parse(src).body if not isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    calls = [c for n in top for c in ast.walk(n) if isinstance(c, ast.Call)]
    assert not [c for c in calls if getattr(c.func, "attr", getattr(c.func, "id", None)) in reads]
    assert src.count("LOCK_FILE.read_bytes()") == 1
    guard = src.index("    if modal.is_local():\n")
    assert guard < src.index("LOCK_FILE.read_bytes()") < src.index("    return env\n")


def test_modal_app_imports_in_the_container_without_the_lock_file(tmp_path):
    # Behaviour check where modal is installed: a lone copy of modal_app.py, as in /root on Modal.
    pytest.importorskip("modal")
    import os
    import subprocess
    (tmp_path / "modal_app.py").write_bytes((ROOT / "infra" / "modal_app.py").read_bytes())
    code = ("import modal, runpy, sys; modal.is_local = lambda: False; "
            "g = runpy.run_path(sys.argv[1]); env = g['_image_env'](); "
            "assert 'EMBER_IMAGE_LOCK_SHA256' not in env and env['HF_HOME'] == '/cache/hf'; print('ok')")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    out = subprocess.run([sys.executable, "-c", code, str(tmp_path / "modal_app.py")], cwd=tmp_path, env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and out.stdout.strip().endswith("ok"), out.stderr[-2000:]


def _find_p0b(cwd, pythonpath):
    import os
    import subprocess
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    if pythonpath:
        env["PYTHONPATH"] = str(pythonpath)
    code = "import importlib.util as u; print(u.find_spec('ember_v1.p0b').origin)"
    out = subprocess.run([sys.executable, "-c", code], cwd=cwd, env=env, capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else None


def test_verify_install_build_step_imports_from_pythonpath(tmp_path):
    """The package copied to <root>/ember_v1 imports from another working directory with PYTHONPATH=<root>.

    This checks only the import path on the CPU. It does not replace a real Modal image build.
    """
    import shutil
    root, elsewhere = tmp_path / "root", tmp_path / "elsewhere"
    elsewhere.mkdir()
    shutil.copytree(Path(p0b.__file__).parent, root / "ember_v1", ignore=shutil.ignore_patterns("__pycache__"))
    copied = root / "ember_v1" / "p0b.py"
    with_path = _find_p0b(elsewhere, root)
    assert with_path is not None and Path(with_path).resolve() == copied.resolve()
    without = _find_p0b(elsewhere, None)  # no copy in the working directory: never the copied file
    assert without is None or Path(without).resolve() != copied.resolve()


@pytest.mark.parametrize("args, error", [
    (("inspect", "L4", p0b.PINNED_MODEL, "bf16", "", CLEAN), PermissionError),
    (("smoke", "L4", p0b.PINNED_MODEL, "bf16", "inspect", CLEAN), PermissionError),
    (("train", "L4", p0b.PINNED_MODEL, "bf16", "train", CLEAN), ValueError),
    (("inspect", "H100", p0b.PINNED_MODEL, "bf16", "inspect", CLEAN), ValueError),
    (("inspect", "L4", "unsloth/Qwen3.5-4B", "bf16", "inspect", CLEAN), ValueError),
    (("inspect", "L4", p0b.PINNED_MODEL, "4bit", "inspect", CLEAN), ValueError),
    (("inspect", "L4", p0b.PINNED_MODEL, "bf16", "inspect", CLEAN + "-dirty"), ValueError),
    (("inspect", "L4", p0b.PINNED_MODEL, "bf16", "inspect", ""), ValueError),
])
def test_gpu_run_is_refused_unless_authorized_pinned_and_clean(args, error):
    with pytest.raises(error):
        p0b.gpu_run_plan(*args)


# ------------------------------------------------------------ model revision pin
def test_load_model_passes_the_pinned_revision(monkeypatch):
    calls = {}

    class Fast:
        @staticmethod
        def from_pretrained(**kwargs):
            calls.update(kwargs)
            return "model", "tok"

        @staticmethod
        def for_inference(model):
            return model

    monkeypatch.setitem(sys.modules, "unsloth", types.SimpleNamespace(FastDecisionModel=Fast))
    assert p0b.load_model(p0b.PINNED_MODEL, 16384, "bf16", revision=p0b.PINNED_MODEL_COMMIT) == ("model", "tok")
    assert calls["revision"] == p0b.PINNED_MODEL_COMMIT and calls["load_in_4bit"] is False


@pytest.mark.parametrize("revision", ["main", "fffbc0d", "F" * 40])
def test_load_model_rejects_a_non_commit_revision_before_import(monkeypatch, revision):
    monkeypatch.setitem(sys.modules, "unsloth", None)  # an import would raise ImportError
    with pytest.raises(ValueError):
        p0b.load_model(p0b.PINNED_MODEL, 16384, "bf16", revision=revision)


def test_model_pin_never_passes_an_unknown_revision():
    expected = p0b.model_revision(p0b.PINNED_MODEL, p0b.PINNED_MODEL_COMMIT)
    assert p0b.model_pin_status(provenance(), expected) == "PASS"
    assert p0b.model_pin_status(provenance(model_revision=None), expected) == "UNVERIFIED"
    assert p0b.model_pin_status(None, expected) == "UNVERIFIED"
    assert p0b.model_pin_status(provenance(model_revision=f"{p0b.PINNED_MODEL}@{'0' * 40}"), expected) == "FAIL"


# ------------------------------------------------------------ inspect records
def _inspect(monkeypatch, kernels=None, model=None):
    fakes.install(monkeypatch)
    monkeypatch.setattr(p0b, "batched_probabilities", lambda m, t, records, qs: [
        {pid: a["probabilities"] for pid, a in fakes.answers_for(q).items()} for q in qs])
    monkeypatch.setattr(p0b, "_profile_kernels", kernels or (
        lambda fn: {"executed": ["fake_kernel"], "evidence": "torch_profiler"}))
    return p0b.task_inspect(model or FakeModel(), CharTokenizer(), "compact", metadata=provenance())


def test_inspect_records_head_association_preservation_and_environment(monkeypatch):
    report = _inspect(monkeypatch)
    assert report["status"] == "PASS" and report["checks"]["model_pin"] == "PASS"
    assert report["decision_head"]["forward_missing_params"] == []
    assert "records" in report["decision_head"]["forward_signature"]
    assert [q["question_id"] for q in report["question_options"]] == list(p0b.synthetic_row(3, "compact")["questions"])
    assert all(q["option_ids"] == sorted(q["option_ids"]) and len(q["option_ids"]) == 5
               for q in report["question_options"])
    probe = {p["n"]: p for p in report["preservation_probe"]}
    assert list(probe) == list(p0b.PRESERVATION_PROBE_NS)
    assert probe[1]["status"] == "OK" and probe[1]["truncated"] is False
    # One token per character: N=200 does not fit 16384 here, and the probe records it.
    assert probe[200]["status"] == "ERROR" and probe[200]["truncated"] is None
    assert report["kernels_native_predict"]["evidence"] == "torch_profiler"
    assert isinstance(report["pip_freeze"], list) and report["upstream_contract"]["all_match_audited"] is True


def test_native_predict_profile_is_required(monkeypatch):
    def kernels(fn):
        # Only the native predict() profile fails; the batched adapter profile succeeds.
        name = getattr(fn, "__code__", None)
        native = name is not None and "predict_answers" in name.co_names
        return ({"executed": None, "evidence": "profiler_failed: RuntimeError"} if native
                else {"executed": ["k"], "evidence": "torch_profiler"})

    report = _inspect(monkeypatch, kernels=kernels)
    assert report["checks"]["kernel_profile"] == "UNVERIFIED" and report["status"] == "UNVERIFIED"


def test_preservation_probe_is_recorded_but_not_gating(monkeypatch):
    # The limit fits the gated N=3 row (and N=1) but not the N=200 probe row.
    full3 = len(fakes.encode_record(CharTokenizer(), p0b.synthetic_row(3, "compact"), max_length=10**9).input_ids)
    report = _inspect(monkeypatch, model=FakeModel(max_len=full3 + 1))
    probe = {p["n"]: p for p in report["preservation_probe"]}
    assert probe[1]["status"] == "OK" and probe[200]["truncated"] is not False
    assert probe[200]["status"] != "OK" and report["checks"]["encoding"] == "PASS"
    assert report["status"] == "PASS"


def test_kernel_markers_label_deltanet_kernels():
    names = ["chunk_gated_delta_rule_fwd_kernel_h", "causal_conv1d_fwd_kernel", "triton_poi_fused_add_0",
             "aten::mm"]
    markers = p0b.kernel_markers(names)
    assert markers["fla_gated_delta"] == names[:1] and markers["causal_conv1d"] == names[1:2]
    assert markers["triton"] == names[2:3]
    assert p0b.kernel_markers(["aten::mm"]) == {k: [] for k in p0b.KERNEL_MARKERS}
