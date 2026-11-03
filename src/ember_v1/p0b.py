"""P0B GPU feasibility measurements for Unsloth FastDecisionModel (no training).

Runs on any CUDA host; `infra/modal_app.py` runs it on Modal L4/L40S.
Imports torch/unsloth lazily so the CPU package and tests stay dependency-free.

Upstream contract, read from source at `unslothai/unsloth@AUDITED_UNSLOTH_COMMIT`
(vendored Clef `AUDITED_CLEF_REVISION`). Details: `docs/P0B_PREFLIGHT.md`.
- `FastDecisionModel.build_dataset(rows, tokenizer, model) -> (items, report)`.
  Clef items are dicts: `item["record"]` is a Clef `EncodedRecord` (`input_ids`,
  `questions[*].question_id/option_ids`), `item["source"]` is the encoded
  `{"state", "questions"}`. `report` has `total`, `skipped` (per question) and
  `truncated` (count of items with `len(input_ids) >= max_len`).
- Clef `encode_record` cuts ONLY the end of the state: `state_ids[:max_length - fixed]`.
- `FastDecisionModel.predict` re-encodes at `max(max_len, CLEF_SERVE_MAX_LEN)` and
  returns only the answers: its own `truncated` flag never reaches the caller.
- `ClefDecisionModel.forward(input_ids=, attention_mask=, records=)` returns
  `(flat_padded_logits, None)`: one row per question, in record order. The vendored
  `collate_records` RIGHT-pads; the head uses `attention_mask.sum()` as the length.

Truncation proof (`encode_one`): re-encode the same source with the hash-pinned
vendored `encode_record` and no length bound. The row is preserved only if the
bounded record is identical and the full length is below
`min(max_len, CLEF_SERVE_MAX_LEN)`. Anything not proven is UNVERIFIED and never
passes a gate.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import secrets
import sys
import time
import weakref
from pathlib import Path
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .contract import ACTIONS
from .dataset import QUESTION_STYLES, build_decision_row, question_style_of
from .padding_gate import assess_padding_parity, validate_provenance
from .synthetic import make_synthetic_observation

AUDITED_UNSLOTH_COMMIT = "fa2a5e714955ecc5ffcae555432036deeb52c37c"
AUDITED_CLEF_REVISION = "6a3ac5fc030a3f33f7fa860a1271886b64a6606d"
# sha256 of unsloth/_vendor/clef/joint_schema_model.py; equals the entry in its clef_manifest.json.
AUDITED_CLEF_SHA256 = "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3"
AUDITED_DECISION_SHA256 = "784a0a36f50f82c1608b6ef53392249a585e8d16859c2ec3cfe7f4d64fbf56b0"
AUDITED_SERVE_MAX_LEN = 16384  # unsloth/models/_decision_common.py CLEF_SERVE_MAX_LEN
# sha256 of every upstream file on the audited encode/load/predict path. The PyPI wheel
# unsloth==2026.10.11 has these exact bytes (checked 2026-11-03, docs/P0B_GPU_INSPECT.md).
AUDITED_UPSTREAM_FILES = {
    "clef": ("unsloth/_vendor/clef/joint_schema_model.py", AUDITED_CLEF_SHA256),
    "decision": ("unsloth/models/decision.py", AUDITED_DECISION_SHA256),
    # A plain LM (Qwen3.5-2B) loads here: fresh Clef head, seeded with random_state=3407.
    "decision_from_lm": ("unsloth/models/decision_from_lm.py",
                         "1236b0ec42c05687702e3ef813eaf0ae44b9280477d2ec6df2a1828754455153"),
    "decision_common": ("unsloth/models/_decision_common.py",
                        "6dbfab4ed887fff660cdeda7fcfdafc035566ad1da3ad5b6e91e212a7bde8f4c"),
}

# D1 backbone lock. P0B loads only this exact Hugging Face commit (HF API sha, 2026-11-03).
PINNED_MODEL = "unsloth/Qwen3.5-2B"
PINNED_MODEL_COMMIT = "fffbc0d8711cf87df6d1f97938c5e50ef94aadc7"
# Exact versions the image lock installs (infra/p0b-requirements.lock.txt).
PINNED_RUNTIME = {"unsloth": "2026.10.11", "unsloth_zoo": "2026.10.11", "torch": "2.8.0", "transformers": "5.2.0",
                  "trl": "0.22.2", "xformers": "0.0.32.post2", "triton": "3.4.0", "torchao": "0.14.0"}

# ---- GPU spend guard (infra/modal_app.py). Pure Python so CPU CI tests it without Modal.
ALLOWED_GPUS = ("L4", "L40S")
# Per-call limit. Modal applies it separately to container start (download + load) and to the
# call, so one command holds the GPU for at most STARTUP_TIMEOUT_S + TASK_TIMEOUT_S[task].
STARTUP_TIMEOUT_S = 900
TASK_TIMEOUT_S = {"inspect": 600, "smoke": 1800, "padding": 1800, "throughput": 3600}
# modal.com/pricing, read 2026-11-03 (GPU only; CPU and memory are billed on top).
GPU_USD_PER_S = {"L4": 0.000222, "L40S": 0.000542}
CPU_CORE_USD_PER_S = 0.0000131
MEM_GIB_USD_PER_S = 0.00000222
# Bound for the CPU/memory part of the estimate: what the container may hold while loading.
EST_CORES, EST_MEM_GIB = 4, 32
P0B_CAP_USD = 10.0  # HLD 5.3


def gpu_run_plan(task: str, gpu: str, model: str, precision: str, authorize: str,
                 code_commit: str) -> dict[str, Any]:
    """Refuse a GPU run unless it is explicitly authorized, on the pinned model, from a clean
    commit, and bounded. Returns the limits the Modal entrypoint applies and the worst-case cost.
    """
    if task not in TASK_TIMEOUT_S:
        raise ValueError(f"unknown task {task!r}: use {'|'.join(TASK_TIMEOUT_S)}")
    if authorize != task:
        raise PermissionError(f"GPU spend needs explicit authorization: pass --authorize {task}")
    if not _COMMIT.match(code_commit or ""):
        raise ValueError("GPU runs need a clean git checkout (no uncommitted changes): results must map to a commit")
    if gpu not in ALLOWED_GPUS:
        raise ValueError(f"gpu must be one of {ALLOWED_GPUS}")
    if model != PINNED_MODEL:
        raise ValueError(f"D1 backbone lock: P0B runs only {PINNED_MODEL}")
    if precision != "bf16":
        raise ValueError("P0B runs bf16 first (HLD D5)")
    worst_s = STARTUP_TIMEOUT_S + TASK_TIMEOUT_S[task]
    per_s = GPU_USD_PER_S[gpu] + EST_CORES * CPU_CORE_USD_PER_S + EST_MEM_GIB * MEM_GIB_USD_PER_S
    return {"task": task, "gpu": gpu, "model": PINNED_MODEL, "revision": PINNED_MODEL_COMMIT, "code_commit": code_commit,
            "startup_timeout_s": STARTUP_TIMEOUT_S, "timeout_s": TASK_TIMEOUT_S[task], "retries": 0,
            "max_containers": 1, "worst_case_gpu_s": worst_s, "worst_case_usd": round(worst_s * per_s, 2),
            "phase_cap_usd": P0B_CAP_USD}

PADDING_ORDERS = ("short_first", "short_last", "short_middle")
# Stack identity that batched-forward evidence must share with the serving runtime.
EVIDENCE_MATCH_KEYS = ("model_name", "model_revision", "tokenizer_revision", "tokenizer_sha256",
                       "encoder_config_sha256", "head_sha256", "decision_config", "ember_v1_source_sha256",
                       "runtime_versions", "cuda", "gpu", "gpu_capability", "precision", "max_seq", "token_limit",
                       "head_path", "upstream_contract")
# Implementation identity that must be KNOWN (non-null) before the fast path may start.
REQUIRED_IDENTITY_KEYS = ("model_revision", "tokenizer_sha256", "encoder_config_sha256", "head_sha256",
                          "ember_v1_source_sha256")
P95_MIN_SAMPLES = 20
P99_MIN_SAMPLES = 100
# predict() rounds each probability to 4 decimals: worst-case sum error is 5 x 5e-5.
PROB_SUM_TOL = 1e-3
_COMMIT = re.compile(r"^[0-9a-f]{40}$")

LEFT_UNSUPPORTED_REASON = (
    "HLD v3 decision D3: the Clef head takes attention_mask.sum() as the sequence "
    "length and uses absolute question/option spans, so only right padding is "
    "representable without a custom head."
)


# ---------------------------------------------------------------- loading
def load_model(model_name: str, max_seq: int, precision: str, revision: str | None = None):
    """`revision` (40-hex) pins the Hugging Face commit; upstream passes it to the backbone load."""
    if precision not in ("bf16", "4bit"):
        raise ValueError("precision must be bf16 or 4bit")
    if revision is not None and not _COMMIT.match(revision):
        raise ValueError("revision must be a 40-hex commit")
    from unsloth import FastDecisionModel

    model, tokenizer = FastDecisionModel.from_pretrained(
        model_name=model_name, max_seq_length=max_seq, load_in_4bit=precision == "4bit",
        **({"revision": revision} if revision else {}))
    FastDecisionModel.for_inference(model)
    return model, tokenizer


def _version(pkg: str) -> str | None:
    import importlib.metadata as md

    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def _sha256_file(path: Any) -> str | None:
    if not path:
        return None
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return None


def model_revision(model_name: str, commit: Any) -> str | None:
    """`name@<40-hex commit>` or None. Never substitute a guessed or placeholder commit."""
    return f"{model_name}@{commit}" if isinstance(commit, str) and _COMMIT.match(commit) else None


def _installed_file(relative: str) -> str | None:
    """Path of a file of the installed `unsloth` distribution, found WITHOUT importing it.

    Importing unsloth needs a GPU; this lookup also works in a CPU image build step.
    """
    import importlib.metadata as md

    try:
        path = md.distribution("unsloth").locate_file(relative)
    except md.PackageNotFoundError:
        return None
    return str(path) if Path(str(path)).is_file() else None


def _upstream_contract() -> dict[str, Any]:
    out: dict[str, Any] = {"audited_unsloth_commit": AUDITED_UNSLOTH_COMMIT,
                           "audited_clef_revision": AUDITED_CLEF_REVISION}
    for name, (relative, audited) in AUDITED_UPSTREAM_FILES.items():
        digest = _sha256_file(_installed_file(relative))
        out[f"{name}_sha256"] = digest
        out[f"{name}_matches_audited"] = digest == audited
    out["all_match_audited"] = all(out[f"{name}_matches_audited"] for name in AUDITED_UPSTREAM_FILES)
    return out


def verify_install() -> dict[str, Any]:
    """CPU check of the installed stack: audited upstream bytes and exact pinned versions.

    Runs as the last Modal image build step, so a drifted install fails the build
    before any GPU container starts. `status` is PASS only if everything matches.
    """
    import platform

    versions = {pkg: _version(pkg) for pkg in PINNED_RUNTIME}
    contract = _upstream_contract()
    mismatched = sorted(pkg for pkg, want in PINNED_RUNTIME.items() if versions[pkg] != want)
    ok = contract["all_match_audited"] and not mismatched
    return {"task": "verify_install", "status": "PASS" if ok else "FAIL", "python": platform.python_version(),
            "runtime_versions": versions, "pinned_runtime": dict(PINNED_RUNTIME),
            "version_mismatches": mismatched, "upstream_contract": contract}


def package_source_sha256() -> str:
    """Identity of the ember_v1 code that encodes, serves and gates (all package .py files).

    CRLF is normalized so a Windows and a Linux checkout of one commit hash the same.
    """
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        digest.update(path.name.encode("utf-8") + b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()


def _sha256_text(text: Any) -> str | None:
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if isinstance(text, str) and text else None


def _encoder_config_sha256(config: Any) -> str | None:
    try:
        return _sha256_text(config.to_json_string(use_diff=False))
    except Exception:  # noqa: BLE001 - unknown identity stays None
        return None


def _head_sha256(head: Any) -> str | None:
    """Hash of the decision-head weights (names, shapes, dtypes, bytes)."""
    import torch

    try:
        digest = hashlib.sha256()
        for name, tensor in sorted(head.state_dict().items()):
            t = tensor.detach().cpu().contiguous()
            digest.update(f"{name}|{tuple(t.shape)}|{t.dtype}".encode("utf-8"))
            digest.update(t.view(-1).view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()
    except Exception:  # noqa: BLE001 - unknown identity stays None
        return None


def _tokenizer_sha256(tokenizer: Any) -> str | None:
    """Hash of the fast tokenizer's full serialized state (vocab, merges, added tokens)."""
    try:
        return _sha256_text(_tok(tokenizer).backend_tokenizer.to_str())
    except Exception:  # noqa: BLE001 - unknown identity stays None
        return None


def _host_runtime() -> dict[str, Any]:
    """Host facts that are not package versions. Unknown values stay None."""
    import os
    import platform
    import subprocess

    import torch

    driver = None
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=30, check=True).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        cudnn = torch.backends.cudnn.version()
    except Exception:  # noqa: BLE001 - unknown stays None
        cudnn = None
    return {"python": platform.python_version(), "platform": platform.platform(), "nvidia_driver": driver,
            "cudnn": cudnn, "image_lock_sha256": os.environ.get("EMBER_IMAGE_LOCK_SHA256") or None}


def _pip_freeze() -> list[str]:
    """Sorted `name==version` of every installed distribution (the environment lock to freeze)."""
    import importlib.metadata as md

    return sorted({f"{d.metadata['Name']}=={d.version}" for d in md.distributions() if d.metadata["Name"]},
                  key=str.lower)


def runtime_metadata(model: Any, tokenizer: Any, model_name: str, precision: str, max_seq: int, *,
                     evidence_source: str, code_commit: str | None = None,
                     requested_revision: str | None = None) -> dict[str, Any]:
    """Provenance for result files. Unknown values stay None; the gate rejects them."""
    import importlib.util

    import torch

    config = getattr(model, "config", None) or getattr(getattr(model, "encoder", None), "config", None)
    init_kwargs = getattr(_tok(tokenizer), "init_kwargs", None) or {}
    decision_config = getattr(model, "decision_config", None)
    gpu, capability = None, None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu, capability = props.name, f"{props.major}.{props.minor}"
    head = getattr(model, "head", None)
    return {
        "schema": "ember_v1.p0b.runtime/2",
        "model_name": model_name,
        "model_revision": model_revision(model_name, getattr(config, "_commit_hash", None)),
        "tokenizer_revision": model_revision(model_name, init_kwargs.get("_commit_hash")),
        "tokenizer_sha256": _tokenizer_sha256(tokenizer),
        "encoder_config_sha256": _encoder_config_sha256(config),
        "head_sha256": _head_sha256(head) if head is not None else None,
        "decision_config": json.loads(json.dumps(dict(decision_config), sort_keys=True, default=str))
        if isinstance(decision_config, Mapping) else None,
        "ember_v1_source_sha256": package_source_sha256(),
        "runtime_versions": {p: _version(p) for p in (
            "unsloth", "unsloth_zoo", "transformers", "tokenizers", "torch", "torchvision", "triton", "xformers",
            "peft", "accelerate", "torchao", "flash-linear-attention", "causal-conv1d", "bitsandbytes")},
        "cuda": torch.version.cuda,
        "gpu": gpu,
        "gpu_capability": capability,
        "head_path": f"{type(model).__module__}.{type(model).__name__}"
                     f"+{type(head).__module__}.{type(head).__name__}+clef.collate_records(right_pad)",
        # Detected = importable. Executed = seen by torch.profiler (filled by the padding/inspect tasks).
        "kernel_backend": {"detected_packages": [k for k in ("fla", "causal_conv1d") if importlib.util.find_spec(k)],
                           "executed": None, "evidence": "import_check_only"},
        "precision": precision,
        "max_seq": max_seq,
        "token_limit": _token_limit(model),
        "upstream_contract": _upstream_contract(),
        "code_commit": code_commit,
        "evidence_source": evidence_source,
        # What was asked for; `model_revision` above is what the loaded config reports.
        "requested_model_revision": model_revision(model_name, requested_revision),
        "backbone": _backbone_info(model),
        "host": _host_runtime(),
        "pip_freeze_sha256": _sha256_text("\n".join(_pip_freeze())),
    }


def _backbone_info(model: Any) -> dict[str, Any]:
    encoder = getattr(model, "encoder", None)
    dtypes: set[str] = set()
    try:
        dtypes = {str(p.dtype) for p in encoder.parameters()} if encoder is not None else set()
    except Exception:  # noqa: BLE001 - unknown stays empty
        pass
    return {"class": f"{type(encoder).__module__}.{type(encoder).__name__}" if encoder is not None else None,
            "param_dtypes": sorted(dtypes),
            "unsloth_forced_float32": getattr(model, "_unsloth_forced_float32", None),
            "unsloth_fast_backbone": getattr(model, "_unsloth_fast_backbone", None)}


# ---------------------------------------------------------------- encoding
def synthetic_row(n: int, question_style: str) -> dict[str, Any]:
    obs = make_synthetic_observation(n)
    return build_decision_row(obs, {f"P{i}": "stay" for i in range(n)}, group_id=f"synthetic-{n}",
                              source="synthetic-contract-only", question_style=question_style)


def _tok(tokenizer: Any) -> Any:
    return getattr(tokenizer, "tokenizer", tokenizer)


def _pad_id(tokenizer: Any) -> int:
    tok = _tok(tokenizer)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if pad is None:
        raise RuntimeError("tokenizer has no pad or eos token id")
    return pad


QUESTION_KEYS = frozenset(("type", "instructions", "criteria"))


def validate_questions(questions: Any) -> dict[str, Any]:
    """Exactly one five-action choice question per person ID. No repair.

    Only `type`, `instructions` and `criteria` are allowed: upstream `_clef_question`
    drops any other key, so an extra key would be silently ignored, not encoded.
    """
    if not isinstance(questions, Mapping) or not questions:
        raise ValueError("questions must be a nonempty mapping of person ID to question")
    for pid, question in questions.items():
        if not isinstance(pid, str) or not pid:
            raise ValueError(f"invalid person ID {pid!r}")
        if not isinstance(question, Mapping) or question.get("type") != "choice":
            raise ValueError(f"question for {pid} is not a choice question")
        if set(question) != QUESTION_KEYS:
            raise ValueError(f"question for {pid} must have exactly the keys {sorted(QUESTION_KEYS)}")
        if type(question["instructions"]) is not str or not question["instructions"]:
            # An empty instruction is replaced by the ID inside encode_record.
            raise ValueError(f"question for {pid} needs nonempty string instructions")
        criteria = question["criteria"]
        if not isinstance(criteria, Mapping) or set(criteria) != set(ACTIONS):
            raise ValueError(f"question for {pid} must offer exactly the options {list(ACTIONS)}")
        if any(type(text) is not str for text in criteria.values()):
            raise ValueError(f"option descriptions for {pid} must be strings")
    return dict(questions)


def _native_questions(questions: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The request in the form `build_dataset` hands to `encode_record`.

    Upstream (`_decision_common._clef_question`, audited commit): a `choice` with dict
    criteria becomes `{"type", "instructions", "criteria"}` with the same criteria
    mapping, keyed by `str(name)`, in request order.
    """
    return {str(pid): {"type": q["type"], "instructions": q["instructions"], "criteria": dict(q["criteria"])}
            for pid, q in questions.items()}


def _question_changes(source_questions: Any, expected: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Exact comparison of the encoded questions with the request: IDs, order, type,
    instructions, option IDs, option order and every option description."""
    if not isinstance(source_questions, Mapping):
        return ["encoded questions are not a mapping"]
    if list(source_questions) != list(expected):
        return [f"question IDs/order {list(source_questions)} != {list(expected)}"]
    changes = []
    for pid, want in expected.items():
        got = source_questions[pid]
        if not isinstance(got, Mapping) or set(got) != QUESTION_KEYS:
            changes.append(f"{pid}: keys differ")
            continue
        for key in ("type", "instructions"):
            if type(got[key]) is not str or got[key] != want[key]:
                changes.append(f"{pid}: {key} differs")
        criteria = got["criteria"]
        if not isinstance(criteria, Mapping) or list(criteria.items()) != list(want["criteria"].items()) or any(
                type(text) is not str for text in criteria.values()):
            changes.append(f"{pid}: options or option descriptions differ")
    return changes


def inference_row(request: Mapping[str, Any]) -> dict[str, Any]:
    """Row for `build_dataset`, which skips questions without gold. The placeholder
    label is never an action: Clef `encode_record` reads only state and questions."""
    if not isinstance(request, Mapping):
        raise ValueError("request must be a mapping")
    state = request.get("state")
    if not isinstance(state, str) or not state:
        raise ValueError("state must be a nonempty string")
    questions = validate_questions(request.get("questions"))
    return {"state": state, "questions": questions, "gold": {pid: "stay" for pid in questions}}


def _row_digest(row: Mapping[str, Any]) -> str:
    """Order-preserving digest of everything the model reads from a request.

    Nested lists only (no dict, no `sort_keys`), so question order and option order
    stay significant: [state, [[id, type, instructions, [[option, description], ...]], ...]].
    """
    questions = [[pid, q.get("type"), q.get("instructions"), [[option, text] for option, text in
                                                             q.get("criteria", {}).items()]]
                 for pid, q in row["questions"].items()]
    payload = json.dumps([row["state"], questions], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# Per-process key: evidence is reusable only inside the process whose encode_one made it.
# A dict typed by a caller (status="OK", truncated=False, matching digest) has no valid seal.
_EVIDENCE_KEY = secrets.token_bytes(32)


def _seal_of(evidence: Mapping[str, Any]) -> str:
    body = {k: v for k, v in evidence.items() if k != "seal"}
    payload = json.dumps(body, sort_keys=True, default=repr, ensure_ascii=False).encode("utf-8")
    return hmac.new(_EVIDENCE_KEY, payload, hashlib.sha256).hexdigest()


def _sealed(evidence: dict[str, Any]) -> dict[str, Any]:
    return {**evidence, "seal": _seal_of(evidence)}


def _check_seal(evidence: Any) -> None:
    if not isinstance(evidence, Mapping) or not isinstance(evidence.get("seal"), str) or not hmac.compare_digest(
            evidence["seal"], _seal_of(evidence)):
        raise ValueError("encoding evidence was not produced by encode_one in this process (or was modified)")


# id(obj) -> (weakref, random token). The weakref check means a reused id of a dead
# object never inherits the old token. Objects that can not be weak-referenced get None.
_OBJECT_TOKENS: dict[int, tuple[Any, str]] = {}


def _object_token(obj: Any) -> str | None:
    entry = _OBJECT_TOKENS.get(id(obj))
    if entry is not None and entry[0]() is obj:
        return entry[1]
    key = id(obj)
    try:
        ref = weakref.ref(obj, lambda _r, k=key: _OBJECT_TOKENS.pop(k, None))
    except TypeError:
        return None
    token = secrets.token_hex(16)
    _OBJECT_TOKENS[key] = (ref, token)
    return token


def _encoding_binding(model: Any, tokenizer: Any) -> dict[str, Any] | None:
    """Cheap identity of what produced an encoding: these model and tokenizer objects
    and the encoding-relevant config (token limit, decision_config). No weight hashing."""
    model_token, tokenizer_token = _object_token(model), _object_token(tokenizer)
    if model_token is None or tokenizer_token is None:
        return None
    config = getattr(model, "decision_config", None)
    config_text = json.dumps(dict(config), sort_keys=True, default=repr) if isinstance(config, Mapping) else None
    return {"model": model_token, "tokenizer": tokenizer_token, "token_limit": _token_limit(model),
            "decision_config_sha256": _sha256_text(config_text)}


def _clef_encoder() -> tuple[Callable[..., Any] | None, str]:
    """Vendored Clef `encode_record`, only if its source is byte-identical to the audited file."""
    try:
        from unsloth._vendor.clef import joint_schema_model as jsm
    except Exception as exc:  # noqa: BLE001 - any import failure means no evidence
        return None, f"UNVERIFIED: vendored Clef import failed ({type(exc).__name__})"
    digest = _sha256_file(getattr(jsm, "__file__", None))
    if digest != AUDITED_CLEF_SHA256:
        return None, f"UNVERIFIED: vendored Clef sha256 {digest} is not the audited {AUDITED_CLEF_SHA256}"
    return jsm.encode_record, "MATCHES_AUDITED"


def _token_limit(model: Any) -> int | None:
    """Lowest context any audited encode path uses: build_dataset (max_len) or predict (serve)."""
    config = getattr(model, "decision_config", None)
    max_len = config.get("max_len") if isinstance(config, Mapping) else None
    if type(max_len) is not int or max_len <= 0:
        return None
    return min(max_len, AUDITED_SERVE_MAX_LEN)


def _record(item: Any) -> Any | None:
    record = item.get("record") if isinstance(item, Mapping) else None
    if record is None or not hasattr(record, "input_ids") or not hasattr(record, "questions"):
        return None
    return record


def encoding_evidence(record: Any, reference: Any, questions: Mapping[str, Any], *, limit: int,
                      report: Mapping[str, Any]) -> dict[str, Any]:
    """Positive evidence that the entire state and every question survived encoding.

    `reference` is the same source encoded by the same audited `encode_record` with
    no length bound. Equal records prove no cut at `max_len`; a full length below
    `limit` proves `predict()` (which re-encodes) does not cut either.
    """
    base = {"input_tokens": len(record.input_ids), "full_tokens": len(reference.input_ids),
            "token_limit": limit, "report": dict(report)}
    ids = [getattr(q, "question_id", None) for q in record.questions]
    if len(ids) != len(set(ids)) or set(ids) != set(questions):
        return {**base, "status": "FAIL_QUESTION_MISMATCH", "truncated": None,
                "reason": f"encoded IDs {sorted(map(str, ids))} differ from the request"}
    bad = [q.question_id for q in record.questions if sorted(getattr(q, "option_ids", ())) != sorted(ACTIONS)]
    if bad:
        return {**base, "status": "FAIL_OPTION_MISMATCH", "truncated": None, "reason": f"bad options for {bad}"}
    if tuple(record.input_ids) != tuple(reference.input_ids) or tuple(record.questions) != tuple(reference.questions):
        return {**base, "status": "FAIL_TRUNCATED", "truncated": True,
                "reason": "encoded record differs from the unbounded encoding: the state was cut"}
    if len(reference.input_ids) >= limit:
        # Same rule as upstream _clef_truncated: one spare token tells a cut from an exact fit.
        return {**base, "status": "FAIL_NO_SPARE_TOKEN", "truncated": None,
                "reason": f"{len(reference.input_ids)} tokens leave no spare token under {limit}; raise max_seq"}
    if report.get("truncated"):
        return {**base, "status": "UNVERIFIED_REPORT_CONFLICT", "truncated": None,
                "reason": "build_dataset reports truncation that the reference check does not show"}
    return {**base, "status": "OK", "truncated": False,
            "evidence": f"unbounded encode_record reference, clef sha256 {AUDITED_CLEF_SHA256[:12]}"}


def encode_one(row: Mapping[str, Any], tokenizer: Any, model: Any) -> tuple[Any, dict[str, Any]]:
    """Encode ONE row. Returns (record, evidence); record is None unless status == "OK"."""
    from unsloth import FastDecisionModel

    questions = validate_questions(row.get("questions"))
    # Every evidence dict names its row and its encoding environment (model, tokenizer, config).
    stamp = {"row_sha256": _row_digest(row), "binding": _encoding_binding(model, tokenizer)}
    expected = _native_questions(questions)
    encode_record, contract = _clef_encoder()
    items, report = FastDecisionModel.build_dataset([dict(row)], tokenizer, model)
    report = dict(report)
    if report.get("skipped") or len(items) != 1:
        return None, _sealed({"status": "FAIL_ENCODE_SKIPPED", "truncated": None, "report": report,
                              **stamp})
    item, record, limit = items[0], _record(items[0]), _token_limit(model)
    source = item.get("source") if isinstance(item, Mapping) else None
    reasons = [r for r, bad in (
        (contract, encode_record is None),
        ("build_dataset item is not {'record': EncodedRecord, 'source': ...}",
         record is None or not isinstance(source, Mapping)),
        ("model.decision_config['max_len'] is unreadable", limit is None),
        ("build_dataset changed the state text",
         isinstance(source, Mapping) and source.get("state") != row["state"]),
    ) if bad]
    if reasons:
        return None, _sealed({"status": "UNVERIFIED", "truncated": None, "reason": "; ".join(reasons),
                              "report": report, **stamp})
    changes = _question_changes(source.get("questions"), expected)
    if changes:
        return None, _sealed({"status": "FAIL_QUESTION_CHANGED", "truncated": None, "report": report,
                              "reason": "; ".join(changes)[:500], **stamp})
    try:
        # Reference from the REQUEST (not from item["source"]): equal records then prove
        # that the requested state and questions were encoded, uncut.
        reference = encode_record(_tok(tokenizer), {"state": row["state"], "questions": expected},
                                  max_length=sys.maxsize)
    except Exception as exc:  # noqa: BLE001 - no reference means no proof
        return None, _sealed({"status": "UNVERIFIED", "truncated": None, "report": report, **stamp,
                              "reason": f"reference encoding failed: {type(exc).__name__}: {exc}"[:300]})
    evidence = {**encoding_evidence(record, reference, questions, limit=limit, report=report), **stamp}
    return (record if evidence["status"] == "OK" else None), _sealed(evidence)


def _question_names(record: Any, questions: Mapping[str, Any]) -> list[str]:
    names = [getattr(q, "question_id", None) for q in record.questions]
    if not all(isinstance(n, str) for n in names) or len(set(names)) != len(names) or set(names) != set(questions):
        raise RuntimeError("encoded question IDs differ from the active person IDs")
    return names


# ---------------------------------------------------------------- inference
def _answers(result: Any, questions: Mapping[str, Any]) -> Mapping[str, Any]:
    """Normalize predict() output: {pid: {...}} or {"answers": {pid: {...}}, ...}."""
    if not isinstance(result, Mapping):
        raise ValueError("predict() did not return a mapping")
    if "answers" not in questions and isinstance(result.get("answers"), Mapping):
        return result["answers"]
    return result


def validate_answers(answers: Any, questions: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Exact ID set, five actions, finite normalized probabilities, answer = top action.

    Raises ValueError on any defect; never repairs or substitutes an action.
    """
    if not isinstance(answers, Mapping):
        raise ValueError("prediction is not a mapping")
    missing, extra = sorted(set(questions) - set(answers)), sorted(map(str, set(answers) - set(questions)))
    if missing or extra:
        raise ValueError(f"prediction ID set differs from the active IDs: missing={missing} extra={extra}")
    out = {}
    for pid in questions:
        result = answers[pid]
        if not isinstance(result, Mapping):
            raise ValueError(f"prediction for {pid} is not a mapping")
        answer, probs = result.get("answer"), result.get("probabilities")
        if not isinstance(answer, str) or answer not in ACTIONS:
            raise ValueError(f"prediction for {pid} has invalid answer {answer!r}")
        if not isinstance(probs, Mapping) or set(probs) != set(ACTIONS):
            raise ValueError(f"prediction for {pid} must score exactly {list(ACTIONS)}")
        values = {}
        for action in ACTIONS:
            p = probs[action]
            if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1:
                raise ValueError(f"invalid probability {p!r} for {pid}/{action}")
            values[action] = float(p)
        if abs(sum(values.values()) - 1.0) > PROB_SUM_TOL:
            raise ValueError(f"probabilities for {pid} are not normalized (sum={sum(values.values())})")
        if values[answer] < max(values.values()):
            raise ValueError(f"answer {answer} for {pid} is not the top-probability action")
        out[pid] = {"answer": answer, "probabilities": values}
    return out


def predict_answers(model: Any, tokenizer: Any, request: Mapping[str, Any], *,
                    evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Native `predict()` behind positive preservation evidence.

    `evidence` may be reused only if it is the unmodified result of `encode_one` in
    this process for the same state and questions (HMAC seal + ordered row digest)
    and for the same model and tokenizer objects and encoding config (binding);
    otherwise this raises. If the objects can not be tracked, the proof is recomputed. Returns `answers` only when `status == "OK"`; otherwise
    `answers` is None and `predict()` is not called. Malformed predictions raise.
    """
    from unsloth import FastDecisionModel

    row = inference_row(request)
    if evidence is None:
        _, evidence = encode_one(row, tokenizer, model)
    else:
        _check_seal(evidence)
        if evidence.get("row_sha256") != _row_digest(row):
            raise ValueError("encoding evidence belongs to a different state/questions")
        current = _encoding_binding(model, tokenizer)
        if evidence.get("binding") is None or current is None:
            _, evidence = encode_one(row, tokenizer, model)  # identity not trackable: prove again
        elif evidence["binding"] != current:
            raise ValueError("encoding evidence was produced for a different model/tokenizer/encoding configuration")
    evidence = dict(evidence)
    if evidence.get("status") != "OK" or evidence.get("truncated") is not False:
        return {"status": evidence.get("status") or "UNVERIFIED", "truncated": evidence.get("truncated"),
                "answers": None, "encoding": evidence}
    raw = FastDecisionModel.predict(model, tokenizer, row["state"], row["questions"])
    answers = _answers(raw, row["questions"])
    if answers is not raw:  # a wrapped result carries its own evidence; it must agree
        if "truncated" in raw and raw["truncated"] is not False:
            return {"status": "FAIL_TRUNCATED_PREDICT", "truncated": raw["truncated"], "answers": None,
                    "encoding": evidence}
        if "input_tokens" in raw and raw["input_tokens"] != evidence.get("full_tokens"):
            raise RuntimeError("predict() input_tokens differs from the verified full encoding")
    return {"status": "OK", "truncated": False, "answers": validate_answers(answers, row["questions"]),
            "encoding": evidence}


def _amp_dtype(model: Any, device: Any) -> Any:
    # Same autocast dtype as predict(); private upstream helper, valid only on the audited stack.
    from unsloth.models.decision import _clef_amp_dtype

    return _clef_amp_dtype(model, device)


def batched_probabilities(model: Any, tokenizer: Any, records: Sequence[Any],
                          questions: Sequence[Mapping[str, Any]]) -> list[dict[str, dict[str, float]]]:
    """One right-padded GPU forward for many environments (softmax at T=1, uncalibrated)."""
    import torch
    from unsloth._vendor.clef.joint_schema_model import collate_records

    if len(records) != len(questions) or not records:
        raise ValueError("records and questions must be nonempty and aligned")
    names = [_question_names(record, qs) for record, qs in zip(records, questions)]
    for record in records:
        for q in record.questions:
            if sorted(q.option_ids) != sorted(ACTIONS):
                raise RuntimeError(f"unexpected option set for {q.question_id}: {sorted(q.option_ids)}")
    device = next(model.parameters()).device
    batch = collate_records(list(records), _pad_id(tokenizer), device)
    if batch.get("media"):
        raise RuntimeError("media inputs are out of scope for the text-only P0B adapter")
    amp = _amp_dtype(model, device)
    with torch.inference_mode(), torch.autocast(device.type, dtype=amp, enabled=amp is not None):
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], records=batch["records"])
    logits = out[0] if isinstance(out, tuple) else out
    total = sum(len(n) for n in names)
    if not torch.is_tensor(logits) or logits.dim() != 2 or logits.shape[0] != total or logits.shape[1] < len(ACTIONS):
        raise RuntimeError(f"forward returned {getattr(logits, 'shape', type(logits))}; expected ({total}, >=5)")
    rows = iter(logits.float().cpu())
    results = []
    for record, record_names in zip(records, names):
        per = {}
        for q, name in zip(record.questions, record_names):
            probs = torch.softmax(next(rows)[: len(q.option_ids)], -1).tolist()
            per[name] = dict(zip(q.option_ids, probs))
        results.append(per)
    return results


# ---------------------------------------------------------------- serving
def _argmax(probs: Mapping[str, float]) -> str:
    # Upstream picks the first maximum over sorted option IDs: alphabetical tie-break.
    return min(probs, key=lambda a: (-probs[a], a))


def verify_batched_forward_evidence(evidence: Any, runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Allow the right-padded fast path only on recomputed, complete, matching evidence.

    `evidence` is the full `padding` task result. Stored `status`/`verdicts` are
    ignored: every gate input is re-assessed here. Raises ValueError otherwise.
    """
    if not isinstance(evidence, Mapping) or evidence.get("task") != "padding":
        raise ValueError("batched_forward evidence must be the full `padding` task result, not a verdict")
    validate_provenance(dict(runtime), require_kernel_profile=False)
    unknown = [k for k in REQUIRED_IDENTITY_KEYS if not runtime.get(k)]
    if unknown:
        raise ValueError(f"serving implementation identity unknown: {unknown}; the fast path stays off")
    if not (runtime.get("upstream_contract") or {}).get("clef_matches_audited"):
        raise ValueError("installed Clef encoder is not the audited file; the fast path stays off")
    style = evidence.get("question_style")
    if style not in QUESTION_STYLES:
        raise ValueError("padding evidence does not record a known question_style")
    docs = evidence.get("gate_inputs")
    if not isinstance(docs, Mapping) or set(docs) != set(PADDING_ORDERS):
        raise ValueError(f"padding evidence needs raw gate inputs for exactly {list(PADDING_ORDERS)}")
    agreement = evidence.get("adapter_agreement_with_predict")
    if not isinstance(agreement, Mapping) or not agreement or any(
            type(v) not in (int, float) or v != 1 for v in agreement.values()):
        raise ValueError("batched adapter does not agree 100% with predict()")
    encodings = evidence.get("encodings")
    if not isinstance(encodings, list) or not encodings or any(
            not isinstance(e, Mapping) or e.get("status") != "OK" or e.get("truncated") is not False
            for e in encodings):
        raise ValueError("padding evidence contains unverified or truncated encodings")
    verdicts, tokens, batch_sizes = {}, [], []
    for order in PADDING_ORDERS:
        verdict = assess_padding_parity(docs[order])  # raises on invalid evidence
        if verdict["status"] != "PASS":
            raise ValueError(f"padding parity FAILS for order {order}")
        meta = docs[order]["metadata"]
        diff = [k for k in EVIDENCE_MATCH_KEYS if meta.get(k) != runtime.get(k)]
        detected = (meta.get("kernel_backend") or {}).get("detected_packages")
        if detected != (runtime.get("kernel_backend") or {}).get("detected_packages"):
            diff.append("kernel_backend.detected_packages")
        if diff:
            raise ValueError(f"padding evidence was measured on a different stack: {diff}")
        verdicts[order] = verdict["status"]
        tokens += [c["input_tokens"] for c in docs[order]["cases"]]
        batch_sizes.append(verdict["environments"])
    canonical = json.dumps(evidence, sort_keys=True, default=str).encode("utf-8")
    return {"status": "VERIFIED", "orders": verdicts,
            # Only this regime was measured; decide_batch refuses anything outside it.
            "scope": {"question_style": style, "min_input_tokens": min(tokens), "max_input_tokens": max(tokens),
                      "max_batch_size": min(batch_sizes)},
            "evidence_sha256": hashlib.sha256(canonical).hexdigest()}


def _check_batched_scope(scope: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
                         encodings: Sequence[Mapping[str, Any]]) -> None:
    """Refuse (never silently serve) requests outside the measured padding regime."""
    if len(rows) > scope["max_batch_size"]:
        raise RuntimeError(f"batch of {len(rows)} exceeds the {scope['max_batch_size']} environments "
                           "the padding evidence measured")
    for i, (row, entry) in enumerate(zip(rows, encodings)):
        if question_style_of(row["questions"]) != scope["question_style"]:
            raise RuntimeError(f"request {i}: question style is not the validated {scope['question_style']!r}")
        if not scope["min_input_tokens"] <= entry["input_tokens"] <= scope["max_input_tokens"]:
            raise RuntimeError(f"request {i}: {entry['input_tokens']} tokens outside the validated range "
                               f"[{scope['min_input_tokens']}, {scope['max_input_tokens']}]")


def decide_batch(model: Any, tokenizer: Any, requests: Sequence[Mapping[str, Any]], *,
                 metadata: Mapping[str, Any], batched_forward: bool = False,
                 batched_evidence: Mapping[str, Any] | None = None,
                 sync: Callable[[], bool] | None = None) -> list[dict[str, Any]]:
    """Serve one Modal batch. Fails closed: any unverified request raises for the batch."""
    if batched_forward and (not isinstance(batched_evidence, Mapping) or batched_evidence.get("status") != "VERIFIED"
                            or not isinstance(batched_evidence.get("scope"), Mapping)):
        raise RuntimeError("batched_forward needs evidence from verify_batched_forward_evidence; refusing")
    sync = sync or _sync
    rows = [inference_row(req) for req in requests]
    synced = sync()
    t0 = time.perf_counter()
    if not batched_forward:
        out = []
        for i, row in enumerate(rows):
            res = predict_answers(model, tokenizer, row)
            if res["status"] != "OK":
                raise RuntimeError(f"request {i}: {res['status']}: fail closed, no fallback action")
            out.append({"answers": res["answers"], "path": "predict", "probability_scale": "served_temperature",
                        "encoding": {k: res["encoding"].get(k) for k in ("status", "input_tokens", "full_tokens")}})
    else:
        records, encodings = [], []
        for i, row in enumerate(rows):
            record, entry = encode_one(row, tokenizer, model)
            if entry["status"] != "OK":
                raise RuntimeError(f"request {i}: {entry['status']}: fail closed, no fallback action")
            records.append(record)
            encodings.append(entry)
        _check_batched_scope(batched_evidence["scope"], rows, encodings)
        probs = batched_probabilities(model, tokenizer, records, [r["questions"] for r in rows])
        out = []
        for row, per, entry in zip(rows, probs, encodings):
            answers = validate_answers({pid: {"answer": _argmax(p), "probabilities": p} for pid, p in per.items()},
                                       row["questions"])
            out.append({"answers": answers, "path": "batched_right_pad", "probability_scale": "softmax_T1_uncalibrated",
                        "encoding": {k: entry.get(k) for k in ("status", "input_tokens", "full_tokens")}})
    sync()
    exec_s = time.perf_counter() - t0
    timing = {"scope": "server_batch_execution", "batch_exec_s": exec_s, "batch_size": len(rows),
              "amortized_per_request_s": exec_s / max(1, len(rows)), "cuda_synchronized": bool(synced),
              "includes": "preservation check, encoding, inference, output validation",
              "excludes": "Modal batching wait (wait_ms), queueing, network, cold start; measure client-side"}
    for item in out:
        item.update(status="OK", truncated=False, model_revision=metadata.get("model_revision"), timing=timing)
    return out


# ---------------------------------------------------------------- timing
def _sync() -> bool:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return True
    return False


def _pct(sorted_values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile."""
    return sorted_values[max(0, math.ceil(q * len(sorted_values)) - 1)]


def latency_summary(times: Sequence[float], **extra: Any) -> dict[str, Any]:
    """Tail percentiles are withheld (None) when the sample is too small to estimate them."""
    s = sorted(times)
    n = len(s)
    if n < 2:
        raise ValueError("at least two timed repetitions")
    out = {"n": n, "method": "nearest_rank", "min_s": s[0], "p50_s": _pct(s, .5), "max_s": s[-1],
           "mean_s": sum(s) / n,
           "p95_s": _pct(s, .95) if n >= P95_MIN_SAMPLES else None,
           "p99_s": _pct(s, .99) if n >= P99_MIN_SAMPLES else None, **extra}
    if n < P99_MIN_SAMPLES:
        out["tail_note"] = (f"n={n}: p95 needs n>={P95_MIN_SAMPLES}, p99 needs n>={P99_MIN_SAMPLES}; "
                            "withheld values are not tail-latency evidence, max_s is descriptive only")
    return out


def _timed(fn: Callable[[], Any], repeats: int, warmup: int = 2,
           sync: Callable[[], bool] | None = None) -> dict[str, Any]:
    sync = sync or _sync
    if repeats < 2:
        raise ValueError("at least two repetitions")
    for _ in range(warmup):
        fn()
    synced = sync()
    times = []
    for _ in range(repeats):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append(time.perf_counter() - t0)
    return latency_summary(times, warmup=warmup, cuda_synchronized=bool(synced), clock="time.perf_counter")


# Name patterns of the optional DeltaNet fast kernels (HLD A5). Heuristic labels over the
# profiled kernel names; the full name list is the evidence, not these labels.
KERNEL_MARKERS = {"fla_gated_delta": re.compile(r"gated_delta|fused_recurrent|chunk_\w*fwd", re.I),
                  "causal_conv1d": re.compile(r"causal_conv1d", re.I),
                  "triton": re.compile(r"triton", re.I)}


def kernel_markers(names: Sequence[str]) -> dict[str, list[str]]:
    """Profiled kernel names that match each marker (at most 10 per marker)."""
    return {label: [n for n in names if pattern.search(n)][:10] for label, pattern in KERNEL_MARKERS.items()}


def _profile_kernels(fn: Callable[[], Any], top: int = 25) -> dict[str, Any]:
    """Device ops/kernels that actually ran, ranked by device time (torch.profiler)."""
    try:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            fn()
            _sync()
        events = prof.key_averages()
        key = "device_time_total" if events and hasattr(events[0], "device_time_total") else "cuda_time_total"
        ran = sorted((e for e in events if getattr(e, key, 0) > 0), key=lambda e: -getattr(e, key, 0))
        names = [e.key[:120] for e in ran[:top]]
        all_names = [e.key[:200] for e in ran]
        return {"executed": names, "evidence": "torch_profiler" if names else "torch_profiler_no_device_events",
                "device_event_count": len(all_names), "markers": kernel_markers(all_names)}
    except Exception as exc:  # noqa: BLE001 - report, never invent kernel evidence
        return {"executed": None, "evidence": f"profiler_failed: {type(exc).__name__}: {exc}"[:300]}


def _overall(statuses: Sequence[str]) -> str:
    if statuses and all(s == "PASS" for s in statuses):
        return "PASS"
    return "FAIL" if any(s.startswith("FAIL") for s in statuses) else "UNVERIFIED"


def result_exit_code(result: Any) -> int:
    """Process exit code for a task result: 0 only for an explicit PASS, else 2."""
    return 0 if isinstance(result, Mapping) and result.get("status") == "PASS" else 2


# Every check must be PASS for inspect to PASS; a missing check is UNVERIFIED.
INSPECT_CHECKS = ("item_shape", "clef_encoder", "upstream_contract", "decision_head", "encoding", "native_predict",
                  "batched_adapter", "adapter_agreement", "kernel_profile", "provenance", "model_pin")
# Audited ClefDecisionModel.forward(input_ids, attention_mask, records, **media).
HEAD_FORWARD_PARAMS = ("input_ids", "attention_mask", "records")
# Encoding-only probe sizes (no GPU work): records full-state preservation beyond the inspected row.
PRESERVATION_PROBE_NS = (1, 200)


def inspection_status(checks: Mapping[str, str]) -> str:
    values = [checks.get(name) for name in INSPECT_CHECKS]
    if any(v == "FAIL" for v in values):
        return "FAIL"
    return "PASS" if all(v == "PASS" for v in values) else "UNVERIFIED"


def _check_of(status: Any) -> str:
    if status == "OK":
        return "PASS"
    return "FAIL" if isinstance(status, str) and status.startswith("FAIL") else "UNVERIFIED"


def _cuda_mem_reserved_gb() -> float | None:
    try:
        import torch
    except ImportError:
        return None
    return round(torch.cuda.memory_reserved() / 2**30, 3) if torch.cuda.is_available() else None


def _signature(fn: Any) -> Any:
    import inspect

    try:
        return inspect.signature(fn)
    except (TypeError, ValueError):
        return None


def head_compatibility(model: Any) -> tuple[str, dict[str, Any]]:
    """Decision-head facts and a status: the audited forward signature and a head + encoder."""
    head, encoder = getattr(model, "head", None), getattr(model, "encoder", None)
    forward = _signature(getattr(type(model), "forward", None))
    info: dict[str, Any] = {
        "head_class": f"{type(head).__module__}.{type(head).__name__}" if head is not None else None,
        "encoder_class": f"{type(encoder).__module__}.{type(encoder).__name__}" if encoder is not None else None,
        "forward_signature": str(forward) if forward is not None else None,
        "head_forward_signature": str(sig) if (sig := _signature(getattr(type(head), "forward", None))) else None,
    }
    try:
        state = head.state_dict()
        info["head_param_count"] = int(sum(t.numel() for t in state.values()))
        info["head_param_dtypes"] = sorted({str(t.dtype) for t in state.values()})
        info["head_state_shapes"] = {k: list(t.shape) for k, t in list(sorted(state.items()))[:40]}
    except Exception as exc:  # noqa: BLE001 - recorded, the status below decides
        info["head_state_error"] = f"{type(exc).__name__}: {exc}"[:200]
    if head is None or encoder is None:
        return "FAIL", info
    if forward is None:
        return "UNVERIFIED", info
    missing = [p for p in HEAD_FORWARD_PARAMS if p not in forward.parameters]
    info["forward_missing_params"] = missing
    return ("FAIL" if missing else "PASS"), info


def model_pin_status(metadata: Mapping[str, Any] | None, expected: str) -> str:
    """PASS only if the loaded config reports the pinned model commit. Unknown never passes."""
    if metadata is None or metadata.get("model_revision") is None:
        return "UNVERIFIED"
    return "PASS" if metadata.get("model_revision") == expected else "FAIL"


# ---------------------------------------------------------------- tasks
def task_inspect(model: Any, tokenizer: Any, question_style: str = "compact",
                 metadata: Mapping[str, Any] | None = None,
                 expected_revision: str | None = None) -> dict[str, Any]:
    """Check the upstream internals the harness relies on.

    Returns the full report with per-check PASS/FAIL/UNVERIFIED and an overall
    `status`. Errors are recorded, never raised, so the report survives a failure.
    `expected_revision` defaults to the D1 lock `PINNED_MODEL@PINNED_MODEL_COMMIT`.
    """
    checks: dict[str, str] = {}
    errors: dict[str, str] = {}
    info: dict[str, Any] = {"task": "inspect", "question_style": question_style, "checks": checks, "errors": errors,
                            "required_checks": list(INSPECT_CHECKS)}

    def failed(name: str, exc: Exception) -> None:
        checks[name] = "FAIL"
        errors[name] = f"{type(exc).__name__}: {exc}"[:500]

    def attrs(obj: Any) -> list[str]:
        return sorted(a for a in dir(obj) if not a.startswith("_"))[:60]

    row = synthetic_row(3, question_style)
    info["model_type"] = f"{type(model).__module__}.{type(model).__name__}"
    info["has_head"], info["has_encoder"] = hasattr(model, "head"), hasattr(model, "encoder")
    checks["decision_head"], info["decision_head"] = head_compatibility(model)
    info["forward_signature"] = info["decision_head"]["forward_signature"]
    info["decision_config"] = {k: str(v)[:100] for k, v in dict(getattr(model, "decision_config", None) or {}).items()}
    try:
        from unsloth import FastDecisionModel

        items, report = FastDecisionModel.build_dataset([row], tokenizer, model)
        item = items[0] if len(items) == 1 else None
        record = _record(item)
        info.update(report={k: str(v)[:200] for k, v in dict(report).items()},
                    item_type=type(item).__name__ if item is not None else None,
                    item_keys=sorted(map(str, item)) if isinstance(item, Mapping) else None,
                    record_attrs=attrs(record) if record is not None else None,
                    question_attrs=attrs(record.questions[0]) if record is not None and record.questions else None)
        checks["item_shape"] = "PASS" if record is not None and isinstance(item.get("source"), Mapping) else "FAIL"
    except Exception as exc:  # noqa: BLE001 - recorded in the report
        failed("item_shape", exc)
    info["upstream_contract"] = _upstream_contract()
    # Every audited upstream file must be byte-identical; a drifted install is unverified, not failed.
    checks["upstream_contract"] = "PASS" if info["upstream_contract"].get("all_match_audited") else "UNVERIFIED"
    encoder, info["clef_encoder"] = _clef_encoder()
    checks["clef_encoder"] = "PASS" if encoder is not None else "FAIL"

    rec, evidence, checked = None, None, None
    try:
        rec, evidence = encode_one(row, tokenizer, model)
        info["encoding"] = evidence
        checks["encoding"] = _check_of(evidence["status"])
        if rec is not None:
            info["record_type"] = f"{type(rec).__module__}.{type(rec).__name__}"
            info["question_type"] = type(rec.questions[0]).__name__ if rec.questions else None
            # Question/option association as encoded: question order and each option order.
            info["question_options"] = [{"question_id": str(q.question_id), "option_ids": list(map(str, q.option_ids))}
                                        for q in rec.questions]
    except Exception as exc:  # noqa: BLE001
        failed("encoding", exc)
    if evidence is not None:
        try:
            checked = predict_answers(model, tokenizer, row, evidence=evidence)
            info["predict_status"] = checked["status"]
            checks["native_predict"] = _check_of(checked["status"])
        except Exception as exc:  # noqa: BLE001 - malformed native output is a FAIL
            failed("native_predict", exc)
    if rec is not None and checked is not None and checked["status"] == "OK":
        try:
            probs = batched_probabilities(model, tokenizer, [rec], [row["questions"]])
            checks["batched_adapter"] = "PASS"
            equal = all(_argmax(p) == checked["answers"][pid]["answer"] for pid, p in probs[0].items())
            info["batched_vs_predict_argmax_equal"] = equal
            checks["adapter_agreement"] = "PASS" if equal else "FAIL"
        except Exception as exc:  # noqa: BLE001
            failed("batched_adapter", exc)
        if checks.get("batched_adapter") == "PASS":
            # Both serving paths: path A (native predict) and path B (batched adapter).
            info["kernels_native_predict"] = _profile_kernels(
                lambda: predict_answers(model, tokenizer, row, evidence=evidence))
            info["kernels"] = _profile_kernels(
                lambda: batched_probabilities(model, tokenizer, [rec], [row["questions"]]))
            profiled = all(info[k]["evidence"] == "torch_profiler" for k in ("kernels", "kernels_native_predict"))
            checks["kernel_profile"] = "PASS" if profiled else "UNVERIFIED"
    if metadata is None:
        checks["provenance"] = "UNVERIFIED"
    else:
        try:
            info["provenance_gaps"] = validate_provenance(dict(metadata), require_kernel_profile=False)
            checks["provenance"] = "PASS"
        except ValueError as exc:
            failed("provenance", exc)
    info["expected_model_revision"] = expected_revision or model_revision(PINNED_MODEL, PINNED_MODEL_COMMIT)
    checks["model_pin"] = model_pin_status(metadata, info["expected_model_revision"])
    # Encoding only (tokenizer work, no forward pass): recorded, not gating; `smoke` owns N<=200.
    info["preservation_probe"] = []
    for n in PRESERVATION_PROBE_NS:
        try:
            _, probe = encode_one(synthetic_row(n, question_style), tokenizer, model)
            info["preservation_probe"].append({"n": n, **{k: probe.get(k) for k in (
                "status", "truncated", "input_tokens", "full_tokens", "token_limit")}})
        except Exception as exc:  # noqa: BLE001 - recorded only
            info["preservation_probe"].append({"n": n, "status": "ERROR", "truncated": None,
                                               "error": f"{type(exc).__name__}: {exc}"[:300]})
    info["pip_freeze"] = _pip_freeze()  # the environment to freeze after a PASS (issue #2)
    info["cuda_mem_reserved_gb"] = _cuda_mem_reserved_gb()
    info["status"] = inspection_status(checks)
    return info


def task_smoke(model: Any, tokenizer: Any, ns: Sequence[int], repeats: int, question_style: str) -> dict[str, Any]:
    """Native single-state path: preservation proof, ID binding, predict() latency."""
    if repeats < 2:
        raise ValueError("at least two repetitions")
    results = []
    for n in ns:
        row = synthetic_row(n, question_style)
        t0 = time.perf_counter()
        _, evidence = encode_one(row, tokenizer, model)
        entry = {"n": n, **evidence, "preservation_check_s": time.perf_counter() - t0}
        if evidence["status"] == "OK":
            first = predict_answers(model, tokenizer, row, evidence=evidence)
            entry["status"] = "PASS" if first["status"] == "OK" else first["status"]
            if first["status"] == "OK":
                entry["predict_latency"] = {
                    "scope": "FastDecisionModel.predict + output validation, one state, evidence precomputed",
                    **_timed(lambda: predict_answers(model, tokenizer, row, evidence=evidence), repeats)}
        results.append(entry)
    return {"task": "smoke", "question_style": question_style, "results": results,
            "status": _overall([r["status"] for r in results]),
            "note": "Synthetic shape + single-request latency. PASS needs positive preservation evidence per row."}


def task_padding(model: Any, tokenizer: Any, ns: Sequence[int], question_style: str,
                 metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Singleton vs right-padded mixed-length batches -> padding_gate inputs."""
    if len(set(ns)) < 2:
        raise ValueError("need at least two crowd sizes of different length")
    rows, records, enc = [], [], []
    for n in sorted(set(ns)):
        row = synthetic_row(n, question_style)
        record, entry = encode_one(row, tokenizer, model)
        if entry["status"] != "OK" or record is None:
            raise RuntimeError(f"encoding not verified for n={n}: {entry['status']}: {entry.get('reason')}")
        rows.append(row)
        records.append(record)
        enc.append(entry)
    qs = [r["questions"] for r in rows]
    single = [batched_probabilities(model, tokenizer, [rec], [q])[0] for rec, q in zip(records, qs)]
    # Adapter check: batched T=1 argmax must equal predict() answers (temperature keeps order).
    agreement = {}
    for row, probs, entry in zip(rows, single, enc):
        res = predict_answers(model, tokenizer, row, evidence=entry)
        if res["status"] != "OK":
            raise RuntimeError(f"predict() not verified for {row['group_id']}: {res['status']}")
        agreement[row["group_id"]] = sum(
            _argmax(p) == res["answers"][pid]["answer"] for pid, p in probs.items()) / len(probs)
    kernels = _profile_kernels(lambda: batched_probabilities(model, tokenizer, records, qs))
    measured = {**metadata, "kernel_backend": {**dict(metadata.get("kernel_backend") or {}), **kernels}}
    k = len(rows)
    orders = {"short_first": list(range(k)), "short_last": list(range(k))[::-1],
              "short_middle": [k - 1] + list(range(k - 1)) if k > 2 else [1, 0]}
    docs = {}
    for name, order in orders.items():
        batched = batched_probabilities(model, tokenizer, [records[i] for i in order], [qs[i] for i in order])
        by_env = {order[j]: batched[j] for j in range(k)}
        cases = []
        for i in range(k):
            env = f"n{len(qs[i])}"
            for variant, probs in (("single", single[i]), ("right", by_env[i])):
                cases.append({"env_id": env, "variant": variant, "input_tokens": enc[i]["input_tokens"],
                              "truncated": enc[i]["truncated"], "probabilities": probs})
        docs[name] = {"metadata": {**measured, "order": [f"n{len(qs[i])}" for i in order],
                                   "left_unsupported_reason": LEFT_UNSUPPORTED_REASON,
                                   "packed_varlen_verified": False},
                      "cases": cases}
    return {"task": "padding", "question_style": question_style, "crowd_sizes": sorted(set(ns)),
            "encodings": enc, "adapter_agreement_with_predict": agreement, "gate_inputs": docs}


def throughput_status(results: Sequence[Mapping[str, Any]]) -> str:
    """PASS = at least one measured batch; OOM only as the recorded capacity limit."""
    statuses = [r.get("status") for r in results]
    return "PASS" if "OK" in statuses and set(statuses) <= {"OK", "OOM"} else "FAIL"


def task_throughput(model: Any, tokenizer: Any, batch_sizes: Sequence[int], ns: Sequence[int],
                    repeats: int, question_style: str) -> dict[str, Any]:
    """Real cross-environment GPU batching: one forward per batch of pre-encoded states."""
    import torch

    pool = []
    for n in ns:
        row = synthetic_row(n, question_style)
        record, entry = encode_one(row, tokenizer, model)
        if entry["status"] != "OK" or record is None:
            raise RuntimeError(f"encoding not verified for n={n}: {entry['status']}: {entry.get('reason')}")
        pool.append((record, row["questions"], entry["input_tokens"]))
    results = []
    for b in batch_sizes:
        chosen = [pool[i % len(pool)] for i in range(b)]
        recs, qs, lens = zip(*chosen)
        torch.cuda.reset_peak_memory_stats()
        try:
            lat = _timed(lambda: batched_probabilities(model, tokenizer, recs, qs), repeats)
        except torch.cuda.OutOfMemoryError:
            results.append({"batch": b, "status": "OOM"})
            torch.cuda.empty_cache()
            break
        results.append({
            "batch": b, "status": "OK", "latency": lat,
            "env_steps_per_s_at_p50": round(b / lat["p50_s"], 2),
            "real_tokens": sum(lens), "padded_token_ratio": round(1 - sum(lens) / (b * max(lens)), 4),
            "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 3),
        })
    return {"task": "throughput", "question_style": question_style, "crowd_sizes": list(ns), "results": results,
            "status": throughput_status(results),
            "scope": "one right-padded forward per batch on pre-encoded records; excludes tokenization, "
                     "preservation checks, Modal queueing and network. Path B is not parity-cleared by this task."}


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m ember_v1.p0b verify-install`: CPU-only install check (Modal image build step)."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args != ["verify-install"]:
        print("usage: python -m ember_v1.p0b verify-install", file=sys.stderr)
        return 64
    result = verify_install()
    print(json.dumps(result, indent=2))
    return result_exit_code(result)


if __name__ == "__main__":
    raise SystemExit(main())
