"""Run 19 Phase 8: three closed-loop evaluation arms of ONE Student V1 checkpoint (prepared, NOT authorized).

  A_raw          the neural argmax, unchanged (`student_run.StudentPolicy`).
  B_static_mask  per person, the best-scored option among `shield.admissible_actions` (drops a move off the
                 grid, into a wall or into a burning cell; "stay" always stays). Single-person and static.
  C_joint_coord  B, then a movement-aware joint repair with the EXACT public resolver
                 (`sim.movement.resolve_moves`, simultaneous moves, lowest-id conflict rule, chains, cycles):
                 while any mover is blocked, the one blocked person whose block is not explained by another
                 blocked mover (else any blocked person) with the smallest score regret takes its next-best
                 admissible option. Options below "stay" are never used ("stay" is never blocked), so the loop
                 ends with no blocked move. Exits and evacuation are the simulator's, applied after movement.

Every arm records the RAW model argmax per step (`raw_trace`) next to the EXECUTED actions (`trace`), and
counts every person whose action the post-processor changed. B and C are policy corrections, not model
outputs; reports must keep them apart. The fire trajectory depends only on the scenario seed (sim/fire.py), so
the arms meet identical fire on each scenario.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

from .shield import admissible_actions
from .sim.movement import DELTAS, resolve_moves
from .student_run import StudentPolicy, encode_step
from .teacher.grid import PublicGrid

VERSION = "p1-19-policy-arms@1.0.0"
ARMS = ("A_raw", "B_static_mask", "C_joint_coord")


def _argmax(s: Mapping[str, float]) -> str:
    return min(s, key=lambda a: (-s[a], a))          # same tie-break as ClefLoraAdapter.predict


class ScoreTap:
    """Adapter proxy: `predict` = argmax of `scores` (exactly ClefLoraAdapter.predict) and keeps the scores."""

    def __init__(self, inner: Any) -> None:
        self.inner, self.last = inner, {}

    def scores(self, row: Mapping[str, Any]) -> dict[str, dict[str, float]]:
        return self.inner.scores(row)

    def predict(self, row: Mapping[str, Any]) -> dict[str, str]:
        s = self.inner.scores(row)
        self.last.update(s)
        return {pid: _argmax(v) for pid, v in s.items()}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def preference(obs: Mapping[str, Any], scores: Mapping[str, Mapping[str, float]]) -> dict[str, list[str]]:
    """Admissible options per active person, best score first, cut after "stay"."""
    adm = admissible_actions(obs)
    out = {}
    for pid, opts in adm.items():
        s = scores[pid]
        order = sorted(opts, key=lambda a: (-s[a], a))
        out[pid] = order[:order.index("stay") + 1]
    return out


def static_mask(obs: Mapping[str, Any], scores: Mapping[str, Mapping[str, float]]) -> dict[str, str]:
    return {pid: pref[0] for pid, pref in preference(obs, scores).items()}


def coordinate(obs: Mapping[str, Any], scores: Mapping[str, Mapping[str, float]]) -> tuple[dict[str, str],
                                                                                          dict[str, Any]]:
    """Joint repair of the masked choice; returns (actions, {iterations, repaired})."""
    pref = preference(obs, scores)
    idx = {pid: 0 for pid in pref}
    pg = PublicGrid.from_observation(obs)
    pos = dict(pg.active)
    it = 0
    while True:
        joint = {pid: pref[pid][idx[pid]] for pid in sorted(pref)}
        res = resolve_moves(height=pg.height, width=pg.width, walls=pg.walls, burning=pg.burning,
                            enter_fire=pg.enter_fire, positions=pos, actions=joint)
        blocked = [pid for pid, (out, _) in res.items() if out.startswith("blocked_")]
        if not blocked:
            return joint, {"iterations": it, "repaired": sum(1 for p in idx if idx[p])}
        occ = {v: k for k, v in pos.items()}
        bset = set(blocked)

        def target(pid: str) -> tuple[int, int]:
            d = DELTAS[joint[pid]]
            return (pos[pid][0] + d[0], pos[pid][1] + d[1])
        primary = [p for p in blocked if occ.get(target(p)) not in bset]
        cand = primary or blocked

        def regret(pid: str) -> tuple[float, str]:
            s = scores[pid]
            return (s[pref[pid][idx[pid]]] - s[pref[pid][idx[pid] + 1]], pid)
        pick = min(cand, key=regret)
        idx[pick] += 1                                   # never past "stay": stay is never blocked
        it += 1


class ArmPolicy(StudentPolicy):
    """StudentPolicy + one arm's post-processing. `trace` = executed actions (replayable), `raw_trace` = model."""

    def __init__(self, adapter: Any, cfg: Mapping[str, Any], label: str, arm: str) -> None:
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}")
        tap = adapter if isinstance(adapter, ScoreTap) else ScoreTap(adapter)
        super().__init__(tap, cfg, label)
        self.tap, self.arm = tap, arm               # the arm is part of `label`, hence of the identity digest
        self.reset()

    def reset(self) -> None:
        self.raw_trace: list[Any] = []
        self.changed = {"mask": 0, "coord": 0, "steps_changed": 0, "coord_iterations": 0}

    def decide(self, observation: Mapping[str, Any]) -> Any:
        self.tap.last = {}
        d = super().decide(observation)
        raw = dict(d.actions)
        active = sorted(raw)
        self.raw_trace.append(self.trace[-1])
        final = raw
        if self.arm != "A_raw" and active:
            sc = {p: self.tap.last[p] for p in active}
            masked = static_mask(observation, sc)
            final = masked
            self.changed["mask"] += sum(masked[p] != raw[p] for p in active)
            if self.arm == "C_joint_coord":
                final, info = coordinate(observation, sc)
                self.changed["coord"] += sum(final[p] != masked[p] for p in active)
                self.changed["coord_iterations"] += info["iterations"]
            self.changed["steps_changed"] += int(final != raw)
            self.trace[-1] = encode_step(active, final)
        return SimpleNamespace(actions=final, diagnostics={"arm": self.arm}, runtime=d.runtime)

    def record_extra(self) -> dict[str, Any]:
        return {"arm": self.arm, "arms_version": VERSION, "raw_trace": list(self.raw_trace),
                "postprocess_changes": dict(self.changed),
                "trace_semantics": "trace = EXECUTED actions; raw_trace = model argmax before any post-processing"}


def run_arms(adapter: Any, cfg: Mapping[str, Any], scenarios_path: Any, out_dir: Any, *, arms: Sequence[str],
             label: str, guards: Any, block: int = 4, persist: Any = lambda: None,
             log: Any = print) -> dict[str, Any]:
    """Interleaved, resumable arms: per block of `block` scenarios, every arm in order (an interruption leaves
    the arms on matched scenarios). One rollouts file per arm."""
    from pathlib import Path

    from .student_run import rollouts
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = sum(1 for line in Path(scenarios_path).read_text(encoding="utf-8").splitlines() if line)
    res: dict[str, Any] = {}
    for end in range(min(block, n), n + block, block):
        end = min(end, n)
        for arm in arms:
            pol = ArmPolicy(adapter, cfg, f"{label}/{arm}", arm)
            res[arm] = rollouts(adapter, cfg, Path(scenarios_path), out_dir / f"rollouts-{arm}.jsonl",
                                label=f"{label}/{arm}", guards=guards, persist=persist, limit=end, log=log,
                                policy=pol)
            if res[arm]["stopped"]:
                return {"arms": res, "stopped": res[arm]["stopped"], "scenarios": n}
        if end == n:
            break
    return {"arms": res, "stopped": None, "scenarios": n}
