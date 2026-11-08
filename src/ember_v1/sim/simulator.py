"""Independent CPU simulator for the public EMBERBench V1 step contract.

Source: EMBERBench `docs/simulation-contract.md` (simulator version 0.2.0) and
`docs/v1-fairness-spec.md`, commit 423143a. Written from the documents only.
It imports nothing from EMBERBench, the decision model or Modal.

Phase order of one `step(actions)`, with t = completed steps + 1:
  1 observation (pre-step)  2 action validation  3-5 movement (movement.py)
  6 evacuation on exits not burning at the start of the step
  7 fire spread for step t  8 casualties (entered_fire / fire_spread), trapped set
  9 termination (all_resolved wins over max_steps)

`observe()` returns only `Observation.to_dict()` fields. The seed, the RNG
state, the scenario ID and the trapped set never leave the simulator.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .canonical import canonical_json, digest
from .fire import Terrain, neighbours, spread
from .movement import ACTIONS, resolve_moves
from .rng import SplitMix64, draws_between, stream_seed
from .scenario import Pos, Scenario

OBSERVATION_VERSION = "1.0"


class SimulationError(RuntimeError):
    """Step on a terminated simulator, or a fire model contract violation."""


class InvalidActionError(ValueError):
    """Action mapping does not have exactly one allowed action per active person."""


def _cells(cells: Any) -> list[list[int]]:
    return [list(p) for p in sorted(cells)]


class Simulator:
    def __init__(self, scenario: Scenario, seed: int | None = None) -> None:
        if not isinstance(scenario, Scenario):
            raise TypeError("scenario must be a Scenario (use Scenario.from_dict)")
        self._scenario = scenario
        self._seed = scenario.seed if seed is None else seed
        self._rng = SplitMix64(stream_seed(self._seed, "fire"))
        self._terrain = Terrain(scenario.height, scenario.width, frozenset(scenario.obstacles),
                                frozenset(p for _, p in scenario.exits), scenario.exits_flammable)
        self._exit_at = {p: i for i, p in scenario.exits}
        self._pos: dict[str, Pos] = dict(scenario.people)
        self._status = {pid: "active" for pid in self._pos}
        self._burning = frozenset(scenario.ignitions)
        self._step = 0
        self._terminated = False
        self._reason: str | None = None
        self._trapped = self._compute_trapped()
        self._chain = digest({"scenario_hash": scenario.content_hash(), "seed": self._seed,
                              "state": self._private_state()})

    # ----- read-only views -------------------------------------------------
    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def termination_reason(self) -> str | None:
        return self._reason

    @property
    def chain_hash(self) -> str:
        return self._chain

    def active_ids(self) -> list[str]:
        return sorted(pid for pid, s in self._status.items() if s == "active")

    def observe(self) -> dict[str, Any]:
        """Fresh `Observation.to_dict()`: public fields only, canonical order."""
        s = self._scenario
        return {
            "observation_version": OBSERVATION_VERSION,
            "step": self._step,
            "map": {"width": s.width, "height": s.height, "obstacles": _cells(s.obstacles),
                    "exits": [{"id": i, "position": list(p)} for i, p in s.exits]},
            "rules": {"fire_model": s.fire_model, "spread_interval": s.spread_interval,
                      "spread_probability": s.spread_probability, "exits_flammable": s.exits_flammable,
                      "enter_fire": s.enter_fire, "conflict_resolution": "lowest_id_wins",
                      "max_steps": s.max_steps},
            "people": [{"id": pid, "position": list(self._pos[pid]), "status": self._status[pid]}
                       for pid in sorted(self._pos)],
            "burning": _cells(self._burning),
            "allowed_actions": list(ACTIONS),
        }

    def private_state(self) -> dict[str, Any]:
        """Evaluator-private snapshot (includes RNG position and trapped). Never give to a policy."""
        return self._private_state()

    # ----- transition ------------------------------------------------------
    def step(self, actions: Mapping[str, str]) -> dict[str, Any]:
        if self._terminated:
            raise SimulationError("simulator is terminated")
        acts = self._validate_actions(actions)
        t = self._step + 1
        burning0 = self._burning
        active = {pid: self._pos[pid] for pid in acts}
        moves = resolve_moves(height=self._scenario.height, width=self._scenario.width,
                              walls=self._terrain.walls, burning=burning0,
                              enter_fire=self._scenario.enter_fire, positions=active, actions=acts)

        rng_before = self._rng.state
        ignited = spread(self._terrain, burning0, t, self._scenario.fire_model,
                         self._scenario.spread_interval, self._scenario.spread_probability, self._rng)
        if ignited & burning0 or not all(self._terrain.flammable(p) for p in ignited):
            raise SimulationError("fire model returned a burning or non-flammable cell")
        draws = draws_between(rng_before, self._rng.state)
        # Spread is computed before the state changes, but its result is applied in phase 7:
        # the fire model never sees people or actions, so the order is equivalent.

        for pid, (_, end) in moves.items():                       # phase 5
            self._pos[pid] = end
        evacuations = {pid: self._exit_at[self._pos[pid]] for pid in acts   # phase 6
                       if self._pos[pid] in self._exit_at and self._pos[pid] not in burning0}
        for pid in evacuations:
            self._status[pid] = "evacuated"
        self._burning = burning1 = burning0 | ignited             # phase 7
        casualties = {pid: ("entered_fire" if self._pos[pid] in burning0 else "fire_spread")
                      for pid in acts if self._status[pid] == "active" and self._pos[pid] in burning1}
        for pid in casualties:                                    # phase 8
            self._status[pid] = "casualty"
        self._trapped = self._compute_trapped()
        self._step = t
        if not self.active_ids():                                 # phase 9
            self._terminated, self._reason = True, "all_resolved"
        elif t >= self._scenario.max_steps:
            self._terminated, self._reason = True, "max_steps"

        record = {
            "step": t,
            "actions": dict(sorted(acts.items())),
            "movements": {pid: {"outcome": o, "from": list(active[pid]), "to": list(end)}
                          for pid, (o, end) in sorted(moves.items())},
            "ignitions": _cells(ignited),
            "fire_draws": draws,
            "evacuations": dict(sorted(evacuations.items())),
            "casualties": dict(sorted(casualties.items())),
            "termination_reason": self._reason,
            "state_hash": digest(self._private_state()),
        }
        self._chain = digest({"prev": self._chain, "record": record})
        record["transition_hash"] = self._chain
        return record

    # ----- internals -------------------------------------------------------
    def _validate_actions(self, actions: Mapping[str, str]) -> dict[str, str]:
        expected = set(self.active_ids())
        if not isinstance(actions, Mapping) or set(actions) != expected:
            raise InvalidActionError(f"step {self._step}: actions must contain exactly the active ids")
        bad = sorted(pid for pid, a in actions.items() if type(a) is not str or a not in ACTIONS)
        if bad:
            raise InvalidActionError(f"step {self._step}: unsupported action for {bad}")
        return {pid: actions[pid] for pid in sorted(expected)}

    def _compute_trapped(self) -> frozenset[str]:
        """Active people with no 4-path to a non-burning exit over non-wall, non-burning cells."""
        open_cell = lambda p: self._terrain.in_grid(p) and p not in self._terrain.walls and p not in self._burning  # noqa: E731
        start = [p for p in self._exit_at if p not in self._burning]
        reach, queue = set(start), deque(start)
        while queue:
            for n in neighbours(queue.popleft()):
                if n not in reach and open_cell(n):
                    reach.add(n)
                    queue.append(n)
        return frozenset(pid for pid in self.active_ids() if self._pos[pid] not in reach)

    def _private_state(self) -> dict[str, Any]:
        return {
            "step": self._step,
            "people": [{"id": pid, "position": list(self._pos[pid]), "status": self._status[pid]}
                       for pid in sorted(self._pos)],
            "burning": _cells(self._burning),
            "trapped": sorted(self._trapped),
            "rng": {"fire": self._rng.state},
            "terminated": self._terminated,
            "termination_reason": self._reason,
        }


Policy = Callable[[dict[str, Any]], Mapping[str, str]]


def run_episode(scenario: Scenario, policy: Policy, seed: int | None = None) -> list[dict[str, Any]]:
    """Run to termination. The policy sees only a fresh `observe()` dict each step."""
    sim = Simulator(scenario, seed)
    records = []
    while not sim.terminated:
        records.append(sim.step(policy(sim.observe())))
    return records


def replay(scenario: Scenario, seed: int | None, action_log: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    sim = Simulator(scenario, seed)
    return [sim.step(actions) for actions in action_log]


@dataclass(frozen=True)
class ReplayVerdict:
    """Result of `verify_replay`.

    status:
      "complete"  every record matches and the replay terminates exactly at the last record.
      "partial"   every record matches, the replay has not terminated, and the caller
                  declared the trace partial. A partial trace is never proof of an episode.
      "diverged"  see `reason` and `step` (the first step that fails).
    reason (diverged only): "malformed_record", "invalid_actions", "step_after_termination",
      "record_mismatch" or "incomplete" (records end before termination).
    """
    status: str
    steps_verified: int
    step: int | None = None
    reason: str | None = None
    termination_reason: str | None = None
    final_transition_hash: str | None = None

    @property
    def complete(self) -> bool:
        return self.status == "complete"


def verify_replay(scenario: Scenario, seed: int | None, records: Sequence[Mapping[str, Any]], *,
                  partial: bool = False) -> ReplayVerdict:
    """Replay the recorded actions and check every record byte-for-byte.

    With `partial=False` (the default, and the only mode that may support
    `teacher_verified`) the records must cover the whole episode: the replay must
    terminate at the last record, and no record may follow termination. With
    `partial=True` a matching prefix gives status "partial", never "complete".
    """
    sim = Simulator(scenario, seed)
    done = 0
    for rec in records:
        t = done + 1
        if sim.terminated:
            return ReplayVerdict("diverged", done, t, "step_after_termination")
        if not isinstance(rec, Mapping) or not isinstance(rec.get("actions"), Mapping):
            return ReplayVerdict("diverged", done, t, "malformed_record")
        try:
            again = sim.step(rec["actions"])
        except InvalidActionError:
            return ReplayVerdict("diverged", done, t, "invalid_actions")
        try:
            same = again == rec and canonical_json(again) == canonical_json(rec)   # type-strict: 1 != 1.0
        except (TypeError, ValueError):
            return ReplayVerdict("diverged", done, t, "malformed_record")
        if not same:
            return ReplayVerdict("diverged", done, t, "record_mismatch")
        done = t
    if sim.terminated:
        return ReplayVerdict("complete", done, None, None, sim.termination_reason, sim.chain_hash)
    if partial:
        return ReplayVerdict("partial", done, None, None, None, sim.chain_hash)
    return ReplayVerdict("diverged", done, done + 1, "incomplete")


def first_divergence(scenario: Scenario, seed: int | None, records: Sequence[Mapping[str, Any]]) -> int | None:
    """Prefix check: the first step whose record differs, else None.

    None does NOT mean the episode was replayed in full: a valid prefix also gives
    None. Use `verify_replay(..., partial=False)` for full-episode proof.
    """
    verdict = verify_replay(scenario, seed, records, partial=True)
    return verdict.step if verdict.status == "diverged" else None
