"""DAgger for the human-view student: automatic teacher labels on the states the student visits.

Round 0 rolls the belief teacher alone (plain imitation data). Later rounds let the student act
with probability 1 - beta and the teacher with probability beta, per person and step; the teacher
labels EVERY visited view. Training on the union of all rounds removes the covariate-shift gap
that plain imitation leaves. No human gives feedback at any point.

The student is any object with the V1 adapter interface (`scores`, `loss_and_backward`,
`optimizer_step`, `save`): `MockDecisionAdapter` on CPU, `ClefLoraAdapter` on the GPU workstation.

Guards: training rows come from the `train` split only; every output path is write-once.

  python -m ember_v1.human_dagger collect --namespace N --count 256 --out DIR [--beta 0.5 --student CKPT]
  python -m ember_v1.human_dagger fit     --data DIR [DIR ...] --out CKPT [--resume CKPT]
  python -m ember_v1.human_dagger eval    --namespace N --split select --count 32 --out FILE [--student CKPT]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .contract import ACTIONS
from .human_encoding import QUESTION_ID, view_row
from .human_env import EpisodeStats, HumanEvacuationEnv, HumanRewardConfig, run_episode, summarize
from .human_teacher import LABEL_MODES, belief_policy, labels, random_policy
from .human_view import ViewConfig, admissible, validate_view
from .rl_online_scenarios import build_scenarios
from .student_train import lr_at, schedule

DAGGER_VERSION = "1.0"


def _uniform(*key: Any) -> float:
    return int(hashlib.sha256("|".join(map(str, key)).encode()).hexdigest()[:13], 16) / 16 ** 13


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, doc: Any) -> None:
    with open(path, "x", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(doc, sort_keys=True, indent=1) + "\n")


def topology(scenario: Any) -> str:
    """Symmetry-invariant hash of the static map (walls and exits): the family identity."""
    from .gen.analysis import topology_hash
    return topology_hash(scenario.height, scenario.width, scenario.obstacles, [p for _, p in scenario.exits])


def adapter_policy(adapter: Any, masked: bool = True) -> Callable:
    """Greedy student policy. `masked` drops moves the person can see are wasted (wall, fire);
    ties break alphabetically, as in `shield.shielded_choice`."""
    def act(views, routes=None):
        out = {}
        for pid, view in sorted(views.items()):
            scores = adapter.scores(view_row(view, ACTIONS[0]))[QUESTION_ID]
            if set(scores) != set(ACTIONS) or not all(math.isfinite(v) for v in scores.values()):
                raise ValueError("the student must score all five actions with finite values")
            out[pid] = min(admissible(view) if masked else ACTIONS, key=lambda a: (-scores[a], a))
        return out
    return act


def collect(scenarios: Sequence[Any], *, view: ViewConfig = ViewConfig(), beta: float = 1.0, seed: int = 0,
            student: Callable | None = None, label_mode: str = "belief",
            privileged: Callable[[], Any] | None = None, round_index: int = 0
            ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Roll every scenario once and label each visited view.

    beta = probability that the teacher's action is executed (1.0 = teacher only). `student` is a
    policy `(views, routes) -> actions`. `privileged()` builds a fresh full-map teacher per episode
    (an object with `.act(observation)`); the privileged/consistent label modes need it. The full
    observation goes to that teacher only, never into a view.
    """
    if not 0.0 <= beta <= 1.0 or (beta < 1.0 and student is None):
        raise ValueError("beta must be in [0, 1]; a student is required when beta < 1")
    if label_mode not in LABEL_MODES or (label_mode != "belief") != (privileged is not None):
        raise ValueError("privileged/consistent labels need a full-map teacher; belief labels take none")
    rows, episodes = [], []
    for scenario in scenarios:
        env = HumanEvacuationEnv(scenario, view, HumanRewardConfig(eta=0.0))
        views, info = env.reset()
        full = privileged() if privileged is not None else None
        stats, done = EpisodeStats(scenario.scenario_id), False
        while not done:
            routes = env.observer.plans()
            gold = labels(views, label_mode, routes=routes,
                          privileged=full.act(env.observation) if full is not None else None)
            acted = student(views, routes) if beta < 1.0 else {}
            actions = {}
            for pid, v in sorted(views.items()):
                by_teacher = beta >= 1.0 or _uniform(seed, scenario.scenario_id, v["step"], pid) < beta
                actions[pid] = gold[pid] if by_teacher else acted[pid]
                rows.append({"scenario_id": scenario.scenario_id, "round": round_index, "view": v,
                             "label": gold[pid], "executed": actions[pid],
                             "actor": "teacher" if by_teacher else "student"})
            views, _, done, info = env.step(actions)
            stats.add(info["record"])
        episodes.append(stats.result(info))
    return rows, {"rows": len(rows), "labels": dict(sorted(Counter(r["label"] for r in rows).items())),
                  "executed_by": dict(sorted(Counter(r["actor"] for r in rows).items())),
                  "label_equals_executed": sum(r["label"] == r["executed"] for r in rows),
                  "episodes": summarize(episodes)}


# ----- datasets (write-once, hashed) -----------------------------------------------------------------
def write_dataset(out_dir: str | Path, rows: Sequence[Mapping[str, Any]], meta: Mapping[str, Any]) -> dict[str, Any]:
    out = Path(out_dir)
    if not rows:
        raise ValueError("no rows to write")
    out.mkdir(parents=True, exist_ok=False)     # write-once: a started collection is never overwritten
    with open(out / "rows.jsonl", "x", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n")
    manifest = {"dagger_version": DAGGER_VERSION, "rows": len(rows),
                "rows_sha256": _sha256(out / "rows.jsonl"), **dict(meta)}
    _write_json(out / "manifest.json", manifest)
    return manifest


def read_dataset(data_dir: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    d = Path(data_dir)
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dagger_version") != DAGGER_VERSION or manifest.get("split") != "train":
        raise ValueError(f"{d}: not a train-split DAgger dataset of this version")
    if _sha256(d / "rows.jsonl") != manifest["rows_sha256"]:
        raise ValueError(f"{d}: rows.jsonl does not match its manifest")
    rows = [json.loads(line) for line in (d / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(rows) != manifest["rows"]:
        raise ValueError(f"{d}: row count differs from the manifest")
    for r in rows:
        validate_view(r["view"])
        if r["label"] not in ACTIONS:
            raise ValueError(f"{d}: unsupported label")
    return rows, manifest


def aggregate(datasets: Sequence[Sequence[Mapping[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Union of all rounds as training rows. Exact duplicates (same text, same label) are dropped.
    A text with two different labels is kept twice (the loss then learns the split) and counted."""
    seen: dict[str, set[str]] = {}
    out, total = [], 0
    for rows in datasets:
        for r in rows:
            total += 1
            row = view_row(r["view"], r["label"])
            known = seen.setdefault(row["state"], set())
            if r["label"] not in known:
                known.add(r["label"])
                out.append(row)
    return out, {"input_rows": total, "rows": len(out), "duplicates_dropped": total - len(out),
                 "conflicting_states": sum(len(v) > 1 for v in seen.values())}


# ----- training and metrics --------------------------------------------------------------------------
def fit(adapter: Any, rows: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any], *, seed: int = 0) -> dict[str, Any]:
    """Cross-entropy on teacher labels. cfg: epochs, grad_accum_draws, lr, head_lr, warmup_ratio.
    Warmup + cosine schedule (`student_train.lr_at`). Stops on a non-finite loss."""
    sched = schedule(len(rows), cfg, cfg["epochs"])
    accum = cfg["grad_accum_draws"]
    losses, pending, run, step = [], 0, 0.0, 0
    for epoch in range(cfg["epochs"]):
        order = list(range(len(rows)))
        random.Random(f"{seed}|{epoch}").shuffle(order)
        for n, i in enumerate(order):
            last = epoch == cfg["epochs"] - 1 and n == len(order) - 1
            loss = adapter.loss_and_backward(rows[i], 1.0 / accum)
            if not math.isfinite(loss):
                raise FloatingPointError(f"non-finite imitation loss at row {i}")
            run, pending = run + loss, pending + 1
            if pending == accum or last:
                adapter.optimizer_step(lr_at(step, cfg["lr"], sched), lr_at(step, cfg["head_lr"], sched))
                losses.append(run / pending)
                pending, run, step = 0, 0.0, step + 1
    return {"rows": len(rows), "epochs": cfg["epochs"], "optimizer_steps": step, "schedule": sched,
            "first_loss": losses[0], "last_loss": losses[-1], "losses": losses}


def imitation_metrics(adapter: Any, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Raw (unmasked) agreement with the teacher, and how often the raw choice is a visible legal
    move. `legal_rate` is the direct check of the old wall-hit failure."""
    if not rows:
        raise ValueError("no rows")
    raw = adapter_policy(adapter, masked=False)
    agree = legal = 0
    by_label: dict[str, list[int]] = {}
    for r in rows:
        a = raw({"p": r["view"]})["p"]
        agree += a == r["label"]
        legal += a in admissible(r["view"])
        hit = by_label.setdefault(r["label"], [0, 0])
        hit[0] += a == r["label"]
        hit[1] += 1
    return {"rows": len(rows), "accuracy": agree / len(rows), "legal_rate": legal / len(rows),
            "accuracy_by_label": {k: v[0] / v[1] for k, v in sorted(by_label.items())}}


def evaluate(scenarios: Sequence[Any], view: ViewConfig, student: Any | None = None, *, seed: int = 0
             ) -> dict[str, Any]:
    """Every arm on the same scenarios. Arms that use rules outside the model are named as such."""
    arms: dict[str, Callable] = {"belief_teacher": belief_policy, "random_masked": random_policy(seed)}
    if student is not None:
        arms["student_raw"] = adapter_policy(student, masked=False)
        arms["student_masked"] = adapter_policy(student, masked=True)
    return {name: summarize([run_episode(s, policy, view) for s in scenarios]) for name, policy in arms.items()}


# ----- command line ----------------------------------------------------------------------------------
def load_adapter(kind: str, path: Path | None = None, *, train: bool = False) -> Any:
    if kind == "mock":
        from .student_train import MockDecisionAdapter
        return MockDecisionAdapter.load(path) if path else MockDecisionAdapter()
    if kind == "clef":  # pragma: no cover - GPU only
        from .student_train import ClefLoraAdapter, config_from_protocol, load_protocol
        cfg = config_from_protocol(load_protocol())
        if not train:
            return ClefLoraAdapter(cfg, checkpoint=path)
        adapter = ClefLoraAdapter(cfg)      # frozen base + fresh LoRA + fresh head
        if path:
            adapter.load_state(path)        # continue an earlier human-view round
        return adapter
    raise ValueError(f"unknown adapter kind {kind!r}")


def _view_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--radius", type=int, default=4)
    p.add_argument("--exit-knowledge", choices=("seen", "one", "all"), default="seen")
    p.add_argument("--no-memory", action="store_true")
    p.add_argument("--no-line-of-sight", action="store_true")
    p.add_argument("--no-hints", action="store_true")
    p.add_argument("--max-steps", type=int, default=48)
    p.add_argument("--kind", choices=("mock", "clef"), default="clef")
    p.add_argument("--seed", type=int, default=0)


def _view(a: argparse.Namespace) -> ViewConfig:
    return ViewConfig(radius=a.radius, line_of_sight=not a.no_line_of_sight, memory=not a.no_memory,
                      exit_knowledge=a.exit_knowledge, hints=not a.no_hints)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ember_v1.human_dagger")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="roll TRAIN scenarios and label every visited view")
    c.add_argument("--namespace", required=True)
    c.add_argument("--count", type=int, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--beta", type=float, default=1.0)
    c.add_argument("--round", type=int, default=0)
    c.add_argument("--student", type=Path)
    c.add_argument("--unmasked", action="store_true")
    c.add_argument("--label-mode", choices=LABEL_MODES, default="belief")
    c.add_argument("--privileged", default="T1", help="full-map teacher for privileged/consistent labels")
    _view_args(c)
    f = sub.add_parser("fit", help="train the student on the union of collected rounds")
    f.add_argument("--data", type=Path, nargs="+", required=True)
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--resume", type=Path)
    f.add_argument("--epochs", type=int, default=1)
    f.add_argument("--grad-accum", type=int, default=8)
    f.add_argument("--lr", type=float, default=2e-4)
    f.add_argument("--head-lr", type=float, default=1e-4)
    f.add_argument("--warmup-ratio", type=float, default=0.03)
    f.add_argument("--kind", choices=("mock", "clef"), default="clef")
    f.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("eval", help="compare arms on one split (select for choices, report once at the end)")
    e.add_argument("--namespace", required=True)
    e.add_argument("--split", choices=("select", "report"), required=True)
    e.add_argument("--count", type=int, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--student", type=Path)
    e.add_argument("--train-data", type=Path, nargs="*", default=[],
                   help="collected datasets; the run stops if an evaluation map is one of their families")
    _view_args(e)
    a = ap.parse_args(argv)
    if a.out.exists():
        ap.error(f"{a.out} exists (outputs are write-once)")

    if a.cmd == "collect":
        student = adapter_policy(load_adapter(a.kind, a.student), masked=not a.unmasked) if a.student else None
        privileged = None
        if a.label_mode != "belief":
            from .teacher.registry import teacher_class
            privileged = teacher_class(a.privileged)
        scenarios = [s for s, _ in build_scenarios(a.namespace, "train", a.count, a.max_steps)]
        rows, stats = collect(scenarios, view=_view(a), beta=a.beta, seed=a.seed, student=student,
                              label_mode=a.label_mode, privileged=privileged, round_index=a.round)
        manifest = write_dataset(a.out, rows, {
            "namespace": a.namespace, "split": "train", "count": a.count, "max_steps": a.max_steps,
            "view": _view(a).to_dict(), "beta": a.beta, "seed": a.seed, "round": a.round,
            "label_mode": a.label_mode, "privileged": a.privileged if privileged else None,
            "student": str(a.student) if a.student else None, "student_masked": not a.unmasked, "stats": stats,
            "topology_hashes": sorted({topology(s) for s in scenarios})})
        print(json.dumps({k: manifest[k] for k in ("rows", "rows_sha256")} | {"episodes": stats["episodes"]},
                         sort_keys=True))
    elif a.cmd == "fit":
        loaded = [read_dataset(d) for d in a.data]
        if len({json.dumps(m["view"], sort_keys=True) for _, m in loaded}) != 1:
            ap.error("the datasets were collected with different view settings")
        rows, info = aggregate([r for r, _ in loaded])
        adapter = load_adapter(a.kind, a.resume, train=True)
        cfg = {"epochs": a.epochs, "grad_accum_draws": a.grad_accum, "lr": a.lr, "head_lr": a.head_lr,
               "warmup_ratio": a.warmup_ratio}
        report = fit(adapter, rows, cfg, seed=a.seed)
        adapter.save(a.out)
        _write_json(a.out / "human_fit_report.json", {
            "dagger_version": DAGGER_VERSION, "config": cfg, "seed": a.seed, "aggregate": info,
            "datasets": [{"path": str(d), "rows_sha256": m["rows_sha256"], "round": m["round"]}
                         for d, (_, m) in zip(a.data, loaded)],
            "resume": str(a.resume) if a.resume else None, "report": report,
            "train_metrics": imitation_metrics(adapter, [{"view": r["view"], "label": r["label"]}
                                                         for r in loaded[-1][0][:2000]]),
            "weights_digest": adapter.weights_digest()})
        print(json.dumps({k: report[k] for k in ("rows", "optimizer_steps", "first_loss", "last_loss")}, sort_keys=True))
    else:
        student = load_adapter(a.kind, a.student) if a.student else None
        scenarios = [s for s, _ in build_scenarios(a.namespace, a.split, a.count, a.max_steps)]
        trained = {h for d in a.train_data for h in read_dataset(d)[1]["topology_hashes"]}
        if trained & {topology(s) for s in scenarios}:
            ap.error("an evaluation map shares its topology with a training dataset")
        result = {"dagger_version": DAGGER_VERSION, "namespace": a.namespace, "split": a.split, "count": a.count,
                  "max_steps": a.max_steps, "view": _view(a).to_dict(), "student": str(a.student) if a.student else None,
                  "train_data_checked": [str(d) for d in a.train_data],
                  "arms": evaluate(scenarios, _view(a), student, seed=a.seed)}
        _write_json(a.out, result)
        print(json.dumps({k: {m: round(v[m], 4) for m in ("evacuated_rate", "casualty_rate", "utility",
                                                           "blocked_per_attempt")}
                          for k, v in result["arms"].items()}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
