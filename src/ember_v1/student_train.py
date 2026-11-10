"""P1-14 student trainer: weighted multi-teacher SFT on complete joint-action demonstrations (stage 1 baseline).

Qwen3.5-2B + Clef joint-schema head, LoRA r16. Ported from the P1-11 CPU harness of PR #44 (f9696a4) with the
T1-only parts removed: no `s1` training set, no `student_protocol` (P1-11) gate, no hard single-teacher labels.

Status: CPU harness. Nothing here has trained a model. The GPU adapter (`ClefLoraAdapter`) is NOT EXECUTED;
its upstream API assumptions are listed in `GPU_API_ASSUMPTIONS` and checked by `preflight_api` before any
step. Every entry point that could spend money refuses unless `check_protocol(require_frozen=True)` passes for
`data/protocols/p1-14-student-train.json` (now `DRAFT_NOT_FROZEN`) and the caller passes the exact phase name as
`authorize`.

Layers (all CPU-testable with `MockDecisionAdapter` and synthetic fixtures):
* `config_from_protocol`: every pin and hyperparameter comes from the protocol file; nothing is free.
* `rows_from_epoch`: the ONLY data path. `student_ingest.prepare_examples` output (validated, weighted class-A
  whole joint maps of all five teachers; no teacher name) and a `student_ingest.sample_epoch` manifest become
  Clef rows: `student_ingest.epoch_rows` (= `student_objective.rows_for`) plus the person questions. Every row
  must encode `OK` at `max_len`. The record weight is used once, by sampling.
* `train`: rows in epoch draw order (the seeded weighted draw IS the data order). Per row the adapter returns the
  row-mean cross-entropy; the trainer scales it by the row's share of the decision's people, so one draw
  contributes exactly `student_objective.sft_nll` = -log pi(A|s) / N. Gradient accumulation counts draws.
  Cosine schedule with warmup, finite-loss, wall-clock and projected-cost guards, periodic checkpoints,
  throughput and memory tracking, and a failure report on any stop.
* `save_reload_check`, `validate_student_joint`, `pilot_go_rule`, `gpu_train_plan`.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from . import chunk_joint as cj
from . import student_ingest as si
from . import token_budget as tb
from .contract import ACTIONS
from .dataset import decision_questions
from .sim.canonical import digest

TRAIN_VERSION = "p1-14-student-train@0.2.0"
ACTION_ORDER = tuple(sorted(ACTIONS))              # Clef option order (sorted option IDs)
S0_MAX_ROWS = 64
REPO = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO / "data/protocols/p1-14-student-train.json"
TEACHERS = ("T0", "T1", "T2", "T3", "T4")
GPU_PHASES = ("pilot", "train", "eval")
STATUSES = ("DRAFT_NOT_FROZEN", "FROZEN")


class TrainError(RuntimeError):
    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


# ----- protocol --------------------------------------------------------------------------------------------
class StudentTrainProtocolError(ValueError):
    pass


def load_protocol(path: str | Path = PROTOCOL_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def protocol_hash(p: Mapping[str, Any]) -> str:
    return digest({k: v for k, v in p.items() if k != "protocol_hash"})


def hourly_usd(compute: Mapping[str, Any]) -> float:
    r, c = compute["rates_usd_per_h"], compute["container"]
    return r["gpu"] + r["cpu_core"] * c["cpu_cores"] + r["memory_gib"] * c["memory_gib"]


def check_protocol(p: Mapping[str, Any] | None = None, *, require_frozen: bool) -> dict[str, Any]:
    """Structural checks always; `require_frozen` (any paid or real-data run) also needs status FROZEN."""
    p = load_protocol() if p is None else p
    errs = []
    if p.get("protocol_id") != "p1-14-student-train":
        errs.append("not the P1-14 student training protocol")
    if p.get("status") not in STATUSES or (p.get("status") == "FROZEN") is not (p.get("frozen") is True):
        errs.append(f"status {p.get('status')!r} / frozen {p.get('frozen')!r} inconsistent")
    if p.get("protocol_hash") != protocol_hash(p):
        errs.append("protocol_hash does not match the protocol content")
    if tuple((p.get("teachers") or {}).get("demonstrations_from") or ()) != TEACHERS:
        errs.append("demonstrations must come from all five teachers T0..T4 (no single-teacher imitation)")
    if (p.get("data") or {}).get("source") != "student_ingest.epoch_rows":
        errs.append("training rows must come from student_ingest.epoch_rows")
    comp = p.get("compute") or {}
    phases = comp.get("phases") or {}
    if set(phases) != set(GPU_PHASES):
        errs.append(f"compute phases must be {GPU_PHASES}")
    elif sum(ph["usd_cap"] for ph in phases.values()) > comp.get("total_usd_cap", 0):
        errs.append("phase dollar caps exceed total_usd_cap")
    else:
        h = hourly_usd(comp)
        errs += [f"{k}: wall cap costs more than its dollar cap" for k, ph in phases.items()
                 if ph["wall_cap_s"] / 3600 * h > ph["usd_cap"] + 1e-9]
    if require_frozen and p.get("status") != "FROZEN":
        errs.append("protocol is not frozen: no real-data training and no GPU spend is allowed")
    if errs:
        raise StudentTrainProtocolError("; ".join(errs))
    return {"protocol_hash": p["protocol_hash"], "hourly_usd": round(hourly_usd(comp), 4)}


# ----- configuration ---------------------------------------------------------------------------------------
def config_from_protocol(p: Mapping[str, Any]) -> dict[str, Any]:
    pins, tr, d, comp = p["pins"], p["training"], p["data"], p["compute"]
    ck = tr["checkpoints"]
    every = int(ck.split("every ")[1].split(" ")[0]) if "every " in ck else 0
    return {
        "protocol_id": p["protocol_id"], "protocol_hash": p["protocol_hash"],
        "model": pins["model"], "model_commit": pins["model_commit"],
        "tokenizer_json_sha256": pins["tokenizer_json_sha256"], "tokenizer_state_sha256": pins["tokenizer_state_sha256"],
        "requirements_lock_sha256": pins["requirements_lock_sha256"], "clef_sha256": pins["clef_sha256"],
        "unsloth_wheel_sha256": pins["unsloth_wheel_sha256"], "unsloth_zoo_wheel_sha256": pins["unsloth_zoo_wheel_sha256"],
        "precision": pins["precision"], "max_len": pins["max_len"], "gpu": pins["gpu"],
        "encoder_config_sha256": pins["encoder_config_sha256_a10"], "head_init_sha256": pins["head_init_sha256_a10"],
        "lora": lora_config(tr["adapter"]), "lr": tr["lr"], "head_lr": tr["head_lr"], "schedule": tr["schedule"],
        "warmup_ratio": tr["warmup_ratio"], "weight_decay": tr["weight_decay"], "epochs": tr["epochs"],
        "micro_batch_rows": tr["micro_batch_rows"], "grad_accum_draws": tr["grad_accum_draws"], "seed": tr["seed"],
        "epoch_seed": tr["epoch_seed"], "checkpoint_every": every, "gradient_checkpointing":
        tr["gradient_checkpointing"], "chunk_k": d["chunk_k"], "encoding": d["encoding"],
        "question_style": d["question_style"], "reservations": d["reservations"],
        "hourly_usd": hourly_usd(comp), "phases": comp["phases"],
    }


def lora_config(adapter: Mapping[str, Any]) -> dict[str, Any]:
    if adapter.get("type") != "lora" or adapter.get("r") != 16:
        raise TrainError("config", "the protocol fixes LoRA r = 16")
    return {"r": adapter["r"], "lora_alpha": adapter["alpha"], "lora_dropout": adapter["dropout"],
            "target_modules": "all-linear", "bias": "none"}


def schedule(n_draws: int, cfg: Mapping[str, Any], epochs: int | None = None) -> dict[str, int]:
    """Optimizer steps over `epochs` passes of `n_draws` decisions, `grad_accum_draws` decisions per step."""
    if n_draws < 1:
        raise TrainError("config", "no training draws")
    steps = math.ceil(n_draws * (epochs or cfg["epochs"]) / cfg["grad_accum_draws"])
    return {"draws": n_draws, "optimizer_steps": steps, "warmup_steps": max(1, math.ceil(cfg["warmup_ratio"] * steps))}


def lr_at(step: int, base: float, sched: Mapping[str, int]) -> float:
    """Linear warmup to `base`, then cosine to 0 at the last optimizer step (step is 0-based)."""
    w, n = sched["warmup_steps"], sched["optimizer_steps"]
    if step < w:
        return base * (step + 1) / w
    if n <= w:
        return base
    return base * 0.5 * (1 + math.cos(math.pi * (step - w) / max(1, n - w)))


# ----- dataset ---------------------------------------------------------------------------------------------
def rows_from_epoch(prepared: Mapping[str, Any], epoch: Mapping[str, Any], tokenize: tb.Tokenize,
                    cfg: Mapping[str, Any], reservation_fn: Callable[..., str] | None = None) -> list[dict[str, Any]]:
    """Clef rows for one sampled epoch. `prepared` is `student_ingest.prepare_examples` output (which refuses an
    invalid or held-out-touching dataset whole); `epoch` is its `sample_epoch` manifest (hash-checked)."""
    if not isinstance(prepared, Mapping) or prepared.get("version") != si.INGEST_VERSION \
            or not isinstance(prepared.get("examples"), list):
        raise TrainError("ingest_gate", "training rows come only from student_ingest.prepare_examples output")
    if bool(cfg["reservations"]) is not (reservation_fn is not None):
        raise TrainError("config", "the protocol's reservations setting and reservation_fn disagree")
    examples = prepared["examples"]
    by_id = {e["example_id"]: e for e in examples}
    try:
        base = si.epoch_rows(examples, epoch, k=cfg["chunk_k"], reservation_fn=reservation_fn,
                             encoding=cfg["encoding"])
    except si.IngestError as e:
        raise TrainError("ingest_gate", str(e)) from e
    people = {}
    for r in base:
        people[r["draw"]] = people.get(r["draw"], 0) + len(r["ids"])
    qcache: dict[str, dict[str, Any]] = {}
    out = []
    for r in base:
        eid = r["example_id"]
        if eid not in qcache:
            qcache[eid] = decision_questions(by_id[eid]["observation"], style=cfg["question_style"])
        questions = {i: qcache[eid][i] for i in r["ids"]}
        cnt = tb.clef_counts(tokenize, r["state"], questions, cfg["max_len"])
        if cnt["status"] != "OK":
            raise TrainError("row_status_not_ok", f"{eid} row {r['index']}: {cnt['status']} ({cnt['total']} tokens)")
        if any(a not in ACTIONS for a in r["targets"].values()):
            raise TrainError("label", "a label is not an allowed action")
        out.append({"state": r["state"], "questions": questions, "gold": dict(r["targets"]), "draw": r["draw"],
                    "index": r["index"], "example_id": eid, "row_share": len(r["ids"]) / people[r["draw"]],
                    "tokens": cnt["total"]})
    return out


def _draw_groups(rows: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    """Rows grouped by draw, in draw order; each draw's row shares must sum to 1 (all its people)."""
    groups: list[list[Mapping[str, Any]]] = []
    for r in rows:
        if groups and groups[-1][0]["draw"] == r["draw"]:
            groups[-1].append(r)
        else:
            groups.append([r])
    if len({g[0]["draw"] for g in groups}) != len(groups):
        raise TrainError("config", "rows of one draw are not contiguous")
    if any(abs(sum(r["row_share"] for r in g) - 1.0) > 1e-9 for g in groups):
        raise TrainError("config", "a draw's rows do not cover all of its people")
    return groups


# ----- model interface -------------------------------------------------------------------------------------
class Adapter(Protocol):
    device: str

    def loss_and_backward(self, row: Mapping[str, Any], scale: float) -> float: ...
    def optimizer_step(self, lr: float, head_lr: float) -> None: ...
    def predict(self, row: Mapping[str, Any]) -> dict[str, str]: ...
    def save(self, path: Path) -> None: ...
    def weights_digest(self) -> str: ...
    def memory_bytes(self) -> int: ...


class MockDecisionAdapter:
    """CPU stand-in with real cross-entropy gradients: one logit vector per (state, person) key. It can
    memorise a fixture (the S0 overfit check of the training loop); it says nothing about the real model."""
    device = "cpu"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.w: dict[str, list[float]] = {}
        self.g: dict[str, list[float]] = {}

    def _key(self, row: Mapping[str, Any], pid: str) -> str:
        return hashlib.sha256(f"{row['state']}|{pid}".encode()).hexdigest()[:24]

    def _logits(self, key: str) -> list[float]:
        if key not in self.w:
            r = random.Random(f"{self.seed}|{key}")
            self.w[key] = [r.gauss(0, 0.01) for _ in ACTION_ORDER]
        return self.w[key]

    def loss_and_backward(self, row, scale):
        total = 0.0
        for pid, gold in row["gold"].items():
            k = self._key(row, pid)
            z = self._logits(k)
            m = max(z)
            e = [math.exp(v - m) for v in z]
            s = sum(e)
            p = [v / s for v in e]
            y = ACTION_ORDER.index(gold)
            total += -math.log(max(p[y], 1e-300))
            g = self.g.setdefault(k, [0.0] * len(z))
            for j in range(len(z)):
                g[j] += scale * (p[j] - (1.0 if j == y else 0.0)) / len(row["gold"])
        return total / len(row["gold"])

    def optimizer_step(self, lr, head_lr):
        for k, g in self.g.items():
            self.w[k] = [w - head_lr * gi for w, gi in zip(self.w[k], g)]
        self.g = {}

    def predict(self, row):
        out = {}
        for pid in row["questions"]:
            z = self._logits(self._key(row, pid))
            out[pid] = min(range(len(z)), key=lambda j: (-z[j], ACTION_ORDER[j]))  # alphabetical tie-break
            out[pid] = ACTION_ORDER[out[pid]]
        return out

    def save(self, path):
        path.mkdir(parents=True, exist_ok=False)
        (path / "weights.json").write_text(json.dumps({"seed": self.seed, "w": self.w}, sort_keys=True),
                                           encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, path):
        d = json.loads((path / "weights.json").read_text(encoding="utf-8"))
        m = cls(d["seed"])
        m.w = {k: list(v) for k, v in d["w"].items()}
        return m

    def weights_digest(self):
        return digest({"w": self.w})

    def memory_bytes(self):
        return 8 * 5 * len(self.w)


# ----- guards ----------------------------------------------------------------------------------------------
class Guards:
    """Wall-clock and projected-cost stops. The projection is elapsed x hourly rate: an estimate, not billing."""

    def __init__(self, wall_cap_s: float, usd_cap: float, hourly_usd: float, *,
                 clock: Callable[[], float] = time.monotonic, stop_fraction: float = 0.95) -> None:
        self.wall_cap_s, self.usd_cap, self.hourly = wall_cap_s, usd_cap, hourly_usd
        self.clock, self.stop_fraction = clock, stop_fraction
        self.t0 = clock()

    def elapsed(self) -> float:
        return self.clock() - self.t0

    def projected_usd(self) -> float:
        return self.elapsed() / 3600 * self.hourly

    def check(self) -> None:
        if self.elapsed() >= self.stop_fraction * self.wall_cap_s:
            raise TrainError("wall_cap_reached", f"{self.elapsed():.0f} s")
        if self.projected_usd() >= self.usd_cap:
            raise TrainError("usd_cap_reached", f"projected ${self.projected_usd():.2f}")


# ----- training loop ---------------------------------------------------------------------------------------
def _write(path: Path, doc: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def train(adapter: Adapter, rows: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any], *, guards: Guards,
          out_dir: str | Path, mode: str, protocol: Mapping[str, Any] | None = None, epochs: int | None = None,
          max_rows: int | None = None, resume: Mapping[str, Any] | None = None,
          on_checkpoint: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """`mode="protocol"`: the real run; the protocol must be FROZEN and the epoch count is the protocol's.
    `mode="gpu_pilot"`: the bounded GPU pilot (frozen protocol, `max_rows` rows, the protocol schedule).
    `mode="s0_fixture"`: CPU fixture only (<= 64 rows, cpu adapter); `epochs` may repeat for the overfit check.
    `rows` come from `rows_from_epoch`; their order is the epoch's seeded draw order.
    `resume` = {"path", "meter", "checkpoints"} of a periodic checkpoint of the SAME run (same rows, config):
    the adapter state is restored and the draws already consumed are skipped. `on_checkpoint` is called after
    every saved checkpoint (e.g. to persist a volume) with a progress record."""
    out = Path(out_dir)
    if mode in ("protocol", "gpu_pilot"):
        check_protocol(protocol, require_frozen=True)    # raises StudentTrainProtocolError on a draft
        if epochs not in (None, cfg["epochs"]):
            raise TrainError("config", "the protocol run uses the protocol epoch count")
        if mode == "protocol" and max_rows is not None:
            raise TrainError("config", "the protocol run uses every row")
        if mode == "gpu_pilot" and not max_rows:
            raise TrainError("config", "the GPU pilot is bounded by max_rows")
    elif mode == "s0_fixture":
        if adapter.device != "cpu" or len(rows) > S0_MAX_ROWS:
            raise TrainError("config", f"s0_fixture is a CPU check on <= {S0_MAX_ROWS} rows")
    else:
        raise TrainError("config", f"unknown mode {mode!r}")
    if cfg["micro_batch_rows"] != 1:
        raise TrainError("config", "batch > 1 needs a padding gate PASS")
    if out.exists() and resume is None:
        raise TrainError("config", f"{out} exists (write-once)")
    n_epochs = epochs or cfg["epochs"]
    rows = list(rows)[:max_rows] if max_rows else list(rows)
    if max_rows and rows:           # whole draws only: drop a trailing partial draw
        last = rows[-1]["draw"]
        if abs(sum(r["row_share"] for r in rows if r["draw"] == last) - 1.0) > 1e-9:
            rows = [r for r in rows if r["draw"] != last]
    groups = _draw_groups(rows)
    sched = schedule(len(groups), cfg, n_epochs)
    meter = {"rows": 0, "draws": 0, "tokens": 0, "optimizer_steps": 0, "peak_memory_bytes": 0, "losses": []}
    ckpts: list[dict[str, Any]] = []
    skip = 0
    if resume is not None:
        adapter.load_state(Path(resume["path"]))          # type: ignore[attr-defined]
        meter = {**resume["meter"], "losses": list(resume["meter"]["losses"])}
        ckpts = list(resume["checkpoints"])
        skip = meter["draws"]
        if meter["optimizer_steps"] * cfg["grad_accum_draws"] != skip:
            raise TrainError("config", "a resume checkpoint must sit on an optimizer-step boundary")
    stage = "train"
    try:
        accum, run_loss = 0, 0.0
        seen = 0
        for _ep in range(n_epochs):
            for g in groups:
                if seen < skip:
                    seen += 1
                    continue
                seen += 1
                draw_loss = 0.0
                for row in g:            # one draw = one decision: -log pi(A|s) / N over its rows (chain rule)
                    loss = adapter.loss_and_backward(row, row["row_share"] / cfg["grad_accum_draws"])
                    if not math.isfinite(loss):
                        raise TrainError("loss_not_finite", f"draw {row['draw']} row {row['index']}")
                    draw_loss += row["row_share"] * loss
                    meter["rows"] += 1
                    meter["tokens"] += row.get("tokens", 0)
                run_loss += draw_loss
                accum += 1
                meter["draws"] += 1
                if accum == cfg["grad_accum_draws"]:
                    step = meter["optimizer_steps"]
                    if step == 0 and hasattr(adapter, "grad_norms"):
                        meter["first_step_grad_norms"] = adapter.grad_norms()
                        if not all(math.isfinite(v) and v > 0 for v in meter["first_step_grad_norms"].values()):
                            raise TrainError("no_gradient", json.dumps(meter["first_step_grad_norms"]))
                    adapter.optimizer_step(lr_at(step, cfg["lr"], sched), lr_at(step, cfg["head_lr"], sched))
                    meter["optimizer_steps"] += 1
                    meter["losses"].append(run_loss / accum)
                    accum, run_loss = 0, 0.0
                    meter["peak_memory_bytes"] = max(meter["peak_memory_bytes"], adapter.memory_bytes())
                    guards.check()
                    if cfg["checkpoint_every"] and meter["optimizer_steps"] % cfg["checkpoint_every"] == 0:
                        stage = "checkpoint"
                        ckpts.append(_checkpoint(adapter, out, meter["optimizer_steps"]))
                        if on_checkpoint is not None:
                            on_checkpoint({"path": str(out / ckpts[-1]["path"]), "meter": dict(meter),
                                           "checkpoints": list(ckpts), "schedule": sched})
                        stage = "train"
        if accum:
            step = meter["optimizer_steps"]
            adapter.optimizer_step(lr_at(step, cfg["lr"], sched), lr_at(step, cfg["head_lr"], sched))
            meter["optimizer_steps"] += 1
            meter["losses"].append(run_loss / accum)
        stage = "final_checkpoint"
        ckpts.append(_checkpoint(adapter, out, meter["optimizer_steps"], final=True))
    except Exception as e:  # noqa: BLE001 - every stop writes a failure report, then re-raises
        reason = e.reason if isinstance(e, TrainError) else type(e).__name__
        _write(out / "failure_report.json", {"train_version": TRAIN_VERSION, "status": "FAILED", "stage": stage,
                                              "reason": reason, "detail": str(e)[:2000], "meter": _meter(meter, guards),
                                              "checkpoints": ckpts, "traceback": traceback.format_exc()[-4000:]})
        raise
    report = {"train_version": TRAIN_VERSION, "status": "COMPLETED", "mode": mode, "schedule": sched,
              "config_hash": digest(dict(cfg)), "meter": _meter(meter, guards), "checkpoints": ckpts,
              "final_weights_digest": adapter.weights_digest()}
    _write(out / "train_report.json", report)
    return report


def _meter(m: Mapping[str, Any], g: Guards) -> dict[str, Any]:
    el = g.elapsed()
    return {**{k: v for k, v in m.items() if k != "losses"}, "elapsed_s": round(el, 3), "losses": list(m["losses"]),
            "tokens_per_s": round(m["tokens"] / el, 2) if el > 0 else None,
            "projected_usd": round(g.projected_usd(), 4), "first_loss": m["losses"][0] if m["losses"] else None,
            "last_loss": m["losses"][-1] if m["losses"] else None}


def _checkpoint(adapter: Adapter, out: Path, step: int, final: bool = False) -> dict[str, Any]:
    path = out / ("final" if final else f"ckpt-{step:06d}")
    adapter.save(path)
    files = {p.relative_to(path).as_posix(): "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(path.rglob("*")) if p.is_file()}
    return {"path": path.name, "step": step, "weights_digest": adapter.weights_digest(), "files": files}


# ----- integrity and validation ----------------------------------------------------------------------------
def save_reload_check(adapter: Adapter, load: Callable[[Path], Adapter], path: str | Path,
                      probe_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Save, reload into a fresh object, and compare weights and predictions on probe rows."""
    p = Path(path)
    adapter.save(p)
    fresh = load(p)
    same_w = fresh.weights_digest() == adapter.weights_digest()
    diff = sum(fresh.predict(r) != adapter.predict(r) for r in probe_rows)
    rep = {"weights_equal": same_w, "prediction_mismatch_rows": diff, "probe_rows": len(probe_rows),
           "pass": same_w and diff == 0}
    if not rep["pass"]:
        raise TrainError("reload_integrity", json.dumps(rep))
    return rep


def validate_student_joint(obs: Mapping[str, Any], k: int, predict_chunk: Callable[[Mapping[str, Any]], Mapping[str, str]],
                           tokenize: tb.Tokenize | None, *, limit: int = tb.SERVE_MAX_LEN) -> dict[str, Any]:
    """One student decision: every chunk row checked, one prediction per row, the joint map assembled (fail
    closed) and resolved by the public movement resolver."""
    cj.check_rows(obs, k, tokenize, limit=limit)
    chunks = cj.chunk_ids(obs, k)
    rows = tb.chunk_rows(obs, {i: "stay" for c in chunks for i in c}, k=k, group_id="student-eval",
                         source="student", encoding="ascii", question_style="compact")
    joint = cj.assemble(obs, chunks, [dict(predict_chunk(r)) for r in rows])
    res = cj.resolve(obs, joint)
    outcomes: dict[str, int] = {}
    for o, _ in res.values():
        outcomes[o] = outcomes.get(o, 0) + 1
    return {"joint": joint, "outcomes": outcomes, "chunks": len(chunks),
            "cross_chunk_blocked": cj.cross_chunk_conflicts(obs, chunks, joint, res)}


# ----- cost gates ------------------------------------------------------------------------------------------
def pilot_go_rule(protocol: Mapping[str, Any], *, tokens_per_epoch_actual: int, measured_tokens_per_s: float,
                  peak_memory_gb: float, loss_finite: bool) -> dict[str, Any]:
    """The protocol's pilot go-rule on MEASURED numbers. Estimates never satisfy it."""
    comp = protocol["compute"]
    h = hourly_usd(comp)
    cap = comp["phases"]["train"]["usd_cap"]
    if measured_tokens_per_s <= 0:
        raise TrainError("pilot", "measured throughput must be positive")
    proj = tokens_per_epoch_actual / measured_tokens_per_s / 3600 * h
    hours = tokens_per_epoch_actual / measured_tokens_per_s / 3600
    go = proj <= 0.9 * cap and peak_memory_gb <= comp["memory_estimate_gb"]["a10_usable"] and loss_finite \
        and hours * 3600 <= comp["phases"]["train"]["wall_cap_s"]
    return {"projected_train_usd": round(proj, 2), "projected_train_h": round(hours, 2), "limit_usd": 0.9 * cap,
            "peak_memory_gb": peak_memory_gb, "go": go,
            "note": "planning projection; the Modal workspace has no enforced dollar cap"}


def gpu_train_plan(phase: str, protocol: Mapping[str, Any], *, authorize: str, code_commit: str) -> dict[str, Any]:
    """Modal limits for one GPU phase. Refuses a draft protocol, a missing authorization or a dirty commit."""
    check_protocol(protocol, require_frozen=True)
    # Second, independent guard: the superseded P1-11 T1-only protocol can never reach this point.
    if protocol.get("teacher", {}).get("sole_teacher") is True or protocol.get("protocol_id") == "p1-11-student":
        raise TrainError("superseded_training_goal", "T1-only GPU training is disabled")
    if phase not in GPU_PHASES:
        raise TrainError("config", f"unknown phase {phase!r}")
    if authorize != phase:
        raise PermissionError(f"GPU spend needs explicit owner authorization: pass --authorize {phase}")
    if not (isinstance(code_commit, str) and len(code_commit) == 40):
        raise TrainError("config", "GPU runs need a clean 40-hex commit")
    enf, ph = protocol["compute"]["enforcement"], protocol["compute"]["phases"][phase]
    return {"phase": phase, "gpu": protocol["pins"]["gpu"], "timeout_s": ph["wall_cap_s"],
            "watchdog_stop_s": round(0.95 * ph["wall_cap_s"]), "usd_cap": ph["usd_cap"],
            "retries": enf["retries"], "max_containers": enf["max_concurrent_containers"],
            "code_commit": code_commit, "billing_cap_enforced_by_modal": False}


def verify_runtime_pins(meta: Mapping[str, Any], cfg: Mapping[str, Any], *, image_lock_sha256: str | None) -> None:
    """Container identities (`p0b.runtime_metadata`) before the first step. Any difference stops the run."""
    # p0b.runtime_metadata reports `model_revision` as "<name>@<40-hex commit>" (p0b.model_revision).
    want = {"model_name": cfg["model"], "model_revision": f"{cfg['model']}@{cfg['model_commit']}",
            "tokenizer_sha256": cfg["tokenizer_state_sha256"], "encoder_config_sha256": cfg["encoder_config_sha256"],
            "head_sha256": cfg["head_init_sha256"], "precision": cfg["precision"]}
    bad = {k: meta.get(k) for k, v in want.items() if meta.get(k) != v}
    if (meta.get("upstream_contract") or {}).get("clef_matches_audited") is not True:
        bad["clef_matches_audited"] = (meta.get("upstream_contract") or {}).get("clef_matches_audited")
    if (meta.get("token_limit") or 0) < cfg["max_len"]:
        bad["token_limit"] = meta.get("token_limit")
    if image_lock_sha256 != cfg["requirements_lock_sha256"]:
        bad["image_lock_sha256"] = image_lock_sha256
    if bad:
        raise TrainError("identity_or_pin_drift", json.dumps(bad, sort_keys=True, default=str))


# ----- GPU adapter (NOT EXECUTED) --------------------------------------------------------------------------
GPU_API_ASSUMPTIONS = (
    "unsloth.FastDecisionModel.from_pretrained(model_name, max_seq_length, load_in_4bit, revision)",
    "unsloth.FastDecisionModel.get_peft_model(model, r, lora_alpha, lora_dropout, target_modules, ...)",
    "unsloth.FastDecisionModel.build_dataset(rows, tokenizer, model) -> (items, report)",
    "unsloth.models.decision.ClefDataCollator(pad_id)(items) -> batch with target / marker_mask",
    "model(input_ids=, attention_mask=, records=) -> (flat_logits[num_questions, >=5], None)",
    "model.save_pretrained(dir, tokenizer) (adapters + head); unsloth.models.decision._load_clef_checkpoint",
    "unsloth.models._block_compile._causal_conv1d_adapter(original, depthwise_conv) (wrapped: x= call)",
)


def preflight_api() -> dict[str, Any]:
    """Run inside the GPU container before any step: every assumed name must exist, else stop."""
    import importlib
    found = {}
    for mod, name in (("unsloth", "FastDecisionModel"), ("unsloth.models.decision", "ClefDataCollator"),
                      ("unsloth.models.decision", "_load_clef_checkpoint"),
                      ("unsloth.models.decision", "_clef_amp_dtype"), ("unsloth.models.clef", "question_options"),
                      ("unsloth.models._block_compile", "_causal_conv1d_adapter")):
        try:
            found[f"{mod}.{name}"] = hasattr(importlib.import_module(mod), name)
        except ImportError:
            found[f"{mod}.{name}"] = False
    if not all(found.values()):
        raise TrainError("gpu_api", json.dumps(found))
    return found


def _conv_x_shim() -> None:  # pragma: no cover - GPU only
    """P1-18 pilot@2 fault: unsloth's block-compile conv adapter (`_block_compile._causal_conv1d_adapter`)
    takes `hidden_states`, but transformers 5.2's Qwen3.5 GatedDeltaNet calls `causal_conv1d_fn(x=...)`.
    After a training-mode forward routes the conv, a no-grad training-mode call raised TypeError. This
    wrapper accepts either name and forwards positionally; it changes no math."""
    import functools

    from unsloth.models import _block_compile as bc
    orig = bc._causal_conv1d_adapter
    if getattr(orig, "_ember_x_kw", False):
        return

    def adapter(original, depthwise_conv):
        fn = orig(original, depthwise_conv)

        @functools.wraps(fn, updated=())
        def causal_conv1d_fn(hidden_states=None, weight=None, bias=None, activation=None, x=None, **kwargs):
            return fn(x if hidden_states is None else hidden_states, weight, bias, activation, **kwargs)
        causal_conv1d_fn._unsloth_block_adapter = True
        return causal_conv1d_fn
    adapter._ember_x_kw = True
    bc._causal_conv1d_adapter = adapter


class ClefLoraAdapter:  # pragma: no cover - GPU only; the P1-18 A10 pilot is its first run
    """Upstream Clef LoRA: `FastDecisionModel.get_peft_model` puts LoRA r16 on the language layers of
    `model.encoder` (the Qwen3.5 backbone); the fresh `model.head` (float32) trains in full. Loss = the
    upstream `_soft_cross_entropy` over the row's questions (mean over people = our row-mean CE); the
    marker mask confines it to each question's options. bf16 autocast, unsloth gradient checkpointing,
    batch 1. Every upstream name it uses is in GPU_API_ASSUMPTIONS and checked by `preflight_api` first.

    `metadata` (p0b.runtime_metadata of the freshly loaded model, BEFORE LoRA) is kept for pin checks."""

    def __init__(self, cfg: Mapping[str, Any], *, checkpoint: str | Path | None = None) -> None:
        import torch
        from unsloth import FastDecisionModel

        from . import p0b
        preflight_api()
        _conv_x_shim()
        self.torch, self.device, self.cfg = torch, "cuda", cfg
        if checkpoint is None:
            model, tok = p0b.load_model(cfg["model"], cfg["max_len"], cfg["precision"], revision=cfg["model_commit"])
            self.metadata = p0b.runtime_metadata(model, tok, cfg["model"], cfg["precision"], cfg["max_len"],
                                                 evidence_source="modal:StudentTrainer",
                                                 requested_revision=cfg["model_commit"])
            lc = cfg["lora"]
            model = FastDecisionModel.get_peft_model(
                model, r=lc["r"], lora_alpha=lc["lora_alpha"], lora_dropout=lc["lora_dropout"],
                target_modules=lc["target_modules"], bias=lc["bias"],
                use_gradient_checkpointing="unsloth" if cfg["gradient_checkpointing"] else False,
                random_state=cfg["seed"])
        else:  # a saved adapter folder: base at the pinned revision + LoRA + head (inference / evaluation)
            model, tok = FastDecisionModel.from_pretrained(model_name=str(checkpoint),
                                                           max_seq_length=cfg["max_len"], load_in_4bit=False)
            self.metadata = None
        self.model, self.tok = model, tok
        self.pad_id = p0b._pad_id(tok)
        torch.manual_seed(cfg["seed"])
        enc = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder.")]
        head = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("head.")]
        other = [n for n, p in model.named_parameters() if p.requires_grad
                 and not (n.startswith("encoder.") or n.startswith("head."))]
        if checkpoint is None and (not enc or not head or other
                                   or any("lora_" not in n for n, p in model.named_parameters()
                                          if p.requires_grad and n.startswith("encoder."))):
            raise TrainError("gpu_api", f"trainable set is not LoRA + head (enc {len(enc)}, head {len(head)}, "
                                        f"other {other[:3]})")
        self.trainable = {"lora_tensors": len(enc), "lora_params": sum(p.numel() for p in enc),
                          "head_tensors": len(head), "head_params": sum(p.numel() for p in head)}
        self.opt = torch.optim.AdamW([{"params": enc, "lr": cfg["lr"]}, {"params": head, "lr": cfg["head_lr"]}],
                                     weight_decay=cfg["weight_decay"]) if checkpoint is None else None
        FastDecisionModel.for_training(model) if checkpoint is None else FastDecisionModel.for_inference(model)
        from unsloth.models.decision import ClefDataCollator, _clef_amp_dtype
        self.collate = ClefDataCollator(self.pad_id)
        self.amp = _clef_amp_dtype(model, torch.device("cuda"))

    def item(self, row):
        """The upstream Clef item of one row; any skip or cut stops the run (no silent truncation)."""
        from unsloth import FastDecisionModel
        items, rep = FastDecisionModel.build_dataset([{k: row[k] for k in ("state", "questions", "gold")}],
                                                     self.tok, self.model)
        if rep.get("skipped") or rep.get("truncated") or len(items) != 1 \
                or len(items[0]["targets"]) != len(row["questions"]):
            raise TrainError("truncation_detected", json.dumps(rep, default=str))
        return items[0]

    def _forward(self, item):
        from unsloth.models.decision import _no_cudnn_attention
        b = self.collate([item])
        target, mask = b.pop("target").cuda(), b.pop("marker_mask").cuda()
        b.pop("ordinal", None)
        inputs = {k: (v.cuda() if self.torch.is_tensor(v) else v) for k, v in b.items()}
        with _no_cudnn_attention(), self.torch.autocast("cuda", dtype=self.amp, enabled=self.amp is not None):
            z, _ = self.model(**inputs)
        return z.float().masked_fill(~mask, -1e4), target

    def loss_and_backward(self, row, scale):
        z, target = self._forward(self.item(row))
        loss = -(target * self.torch.log_softmax(z, -1)).sum(-1).mean()
        (loss * scale).backward()
        return float(loss.detach())

    def optimizer_step(self, lr, head_lr):
        self.opt.param_groups[0]["lr"], self.opt.param_groups[1]["lr"] = lr, head_lr
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)

    def grad_norms(self) -> dict[str, float]:
        """Norms of the accumulated gradients (call before optimizer_step): both groups must be > 0."""
        out = {}
        for name, g in (("lora", self.opt.param_groups[0]), ("head", self.opt.param_groups[1])):
            s = sum(float(p.grad.float().pow(2).sum()) for p in g["params"] if p.grad is not None)
            out[name] = math.sqrt(s)
        return out

    def scores(self, row) -> dict[str, dict[str, float]]:
        """Per person: log-probabilities over that person's options (upstream option order)."""
        from unsloth.models.clef import question_options
        it = self.item(row)
        was_training = self.model.training
        self.model.eval()  # upstream `_clef_logits` scores in eval mode under no_grad (no dropout)
        try:
            with self.torch.no_grad():
                z, _ = self._forward(it)
                lp = self.torch.log_softmax(z, -1).cpu()
        finally:
            self.model.train(was_training)
        out = {}
        for i, (qid, q) in enumerate(it["source"]["questions"].items()):
            keys = [k for k, _ in question_options(q)]
            out[qid] = {k: float(lp[i, j]) for j, k in enumerate(keys)}
        return out

    def predict(self, row):
        return {pid: min(s, key=lambda a: (-s[a], a)) for pid, s in self.scores(row).items()}

    def save(self, path):
        path.mkdir(parents=True, exist_ok=False)
        self.model.save_pretrained(str(path), self.tok)
        if self.opt is not None:
            self.torch.save(self.opt.state_dict(), path / "optimizer.pt")

    def load_state(self, path):
        """Resume: adapters + head (upstream loader) and the optimizer state of a periodic checkpoint."""
        from unsloth.models.decision import _load_clef_checkpoint
        _load_clef_checkpoint(self.model, Path(path))
        self.opt.load_state_dict(self.torch.load(Path(path) / "optimizer.pt", map_location="cuda"))

    def weights_digest(self):
        h = hashlib.sha256()
        for n, p in sorted(self.model.named_parameters()):
            if p.requires_grad or "lora_" in n or n.startswith("head."):
                h.update(n.encode())
                h.update(p.detach().float().cpu().numpy().tobytes())
        return "sha256:" + h.hexdigest()

    def memory_bytes(self):
        return int(self.torch.cuda.max_memory_allocated())
