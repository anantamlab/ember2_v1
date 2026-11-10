"""CPU-only five-teacher matched rollout primitives (P1-12, issue #45).

THIS IS A FIXTURE/SCHEMA IMPLEMENTATION, NOT A TRAINING DATA EXPORTER.
It intentionally refuses real TRAIN/DEV/holdout labelling until a separately
reviewed multi-teacher source protocol, leakage gate and cost budget exist
(`docs/P1_12_TRAIN_COLLECTION_PROTOCOL.md`, status DRAFT_NOT_FROZEN).

Every policy gets its own fresh simulator from the SAME independent scenario.
Only a copy of observe() enters a teacher. The scenario/fire seed and private
state are evaluator-only. Fire spread is action-independent (simulator phase
7 never sees people or actions); unequal fire histories over overlapping steps
invalidate matched comparisons and raise.

A rollout is a whole policy trajectory. A same-step observation from different
policy trajectories is NOT a common decision state: `common_state_steps` lists
the only steps where all five trajectories really had one identical public
observation. Use probe_shared_state() (or counterfactual.py) for a valid
same-state disagreement record. Negative outcomes here are descriptive, not
causal labels for individual actions.

Evidence layers of one report:
- `content_hash`: deterministic content only (observations, proposals without
  timing, transitions, outcomes, diagnostics, provenance). Regeneration with
  the same code, scenario and policies must reproduce it.
- `evidence_hash`: everything, including host-dependent runtime cost.
A teacher exception, invalid proposal or CPU-cap breach is a recorded teacher
FAILURE, the group is then incomplete and not comparable. A broken replay,
unmatched fire or identity substitution is a collector integrity error and
raises instead.
"""
from __future__ import annotations

import copy
import time
from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any

from .sim import Scenario, Simulator, verify_replay
from .sim.canonical import digest
from .sim.simulator import OBSERVATION_VERSION
from .teacher import proposals
from .teacher import registry as reg
from .teacher.registry import CANDIDATES

SCHEMA_VERSION = "matched-multi-teacher@0.2.0"
SCOPE_FIXTURE = "synthetic_fixture"
# Real TRAIN pilot groups; written only by `train_pilot.execute` after every gate passed. Still never
# training-eligible in this schema: an incomplete group stays diagnostic evidence for ever.
SCOPE_TRAIN_PILOT = "p1-12-train-pilot@1"
SCOPE_TRAIN_FULL = "p1-17-train-full@1"          # full five-teacher TRAIN collection (P1-17)
SCOPES = (SCOPE_FIXTURE, SCOPE_TRAIN_PILOT, SCOPE_TRAIN_FULL)
SUPERVISION_NOT_ASSESSED = "not_assessed: needs a frozen P1-14 dataset protocol (#47)"
CAUSALITY_NOTE = (
    "Closed-loop outcomes are associated with complete policy trajectories; "
    "they do not prove that any one action caused a casualty."
)
PUBLIC_OBSERVATION_KEYS = frozenset({"observation_version", "step", "map", "rules", "people", "burning",
                                     "allowed_actions"})
PRIVATE_KEYS = frozenset({"seed", "rng", "trapped", "scenario_id", "state_hash", "terminated"})
_DELTA = {"north": (-1, 0), "south": (1, 0), "west": (0, -1), "east": (0, 1), "stay": (0, 0)}
# Runtime (host-dependent) fields; excluded from content_hash only.
_RUNTIME_KEYS = frozenset({"compute", "runtime"})


class MultiTeacherError(ValueError):
    """Untrusted, incomplete, unpaired or over-budget trajectory evidence."""


class _TeacherFailure(Exception):
    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def _new_policy(factory: Callable[[], Any], expected: str) -> Any:
    policy = factory()
    if policy.identity.name != expected:
        raise MultiTeacherError(f"expected policy {expected}, got {policy.identity.name}")
    if not callable(getattr(policy, "decide", None)):
        raise MultiTeacherError(f"{expected} lacks decide()")
    return policy


def _check_factories(factories: Mapping[str, Callable[[], Any]]) -> None:
    if not isinstance(factories, Mapping) or set(factories) != set(CANDIDATES):
        raise MultiTeacherError(f"exactly five independent factories required: {CANDIDATES}")
    if any(not callable(f) for f in factories.values()):
        raise MultiTeacherError("every policy factory must be callable")


def deterministic_view(x: Any) -> Any:
    """Copy of a report without host-dependent runtime fields (the content_hash input)."""
    if isinstance(x, Mapping):
        return {k: deterministic_view(v) for k, v in x.items() if k not in _RUNTIME_KEYS}
    if isinstance(x, list):
        return [deterministic_view(v) for v in x]
    return x


def _private_key_paths(x: Any, path: str = "") -> list[str]:
    out: list[str] = []
    if isinstance(x, Mapping):
        for k, v in x.items():
            if k in PRIVATE_KEYS:
                out.append(f"{path}/{k}")
            out.extend(_private_key_paths(v, f"{path}/{k}"))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            out.extend(_private_key_paths(v, f"{path}[{i}]"))
    return out


def check_public_observation(obs: Mapping[str, Any]) -> None:
    """Leakage guard: a policy input has exactly the public keys and no private field anywhere."""
    if not isinstance(obs, Mapping) or set(obs) != PUBLIC_OBSERVATION_KEYS:
        raise MultiTeacherError("policy input is not exactly the public observation")
    leaked = _private_key_paths(obs)
    if leaked:
        raise MultiTeacherError(f"private simulator fields in policy input: {leaked[:5]}")


def step_diagnostics(obs: Mapping[str, Any], actions: Mapping[str, str], record: Mapping[str, Any]) -> dict[str, Any]:
    """Public-state diagnostics of one transition. Computable from the observation and the record only."""
    pos = {p["id"]: tuple(p["position"]) for p in obs["people"] if p["status"] == "active"}
    burning = {tuple(c) for c in obs["burning"]}
    targets = Counter((pos[pid][0] + _DELTA[a][0], pos[pid][1] + _DELTA[a][1])
                      for pid, a in actions.items() if a != "stay")
    near_fire = sum(any((r + dr, c + dc) in burning for dr, dc in _DELTA.values()) for r, c in pos.values())
    outcomes = Counter(m["outcome"] for m in record["movements"].values())
    return {
        "active_before": len(pos),
        "movement_outcomes": dict(sorted(outcomes.items())),
        "blocked_moves": sum(v for k, v in outcomes.items() if k.startswith("blocked_")),
        "contested_targets": sum(n > 1 for n in targets.values()),
        "max_target_contention": max(targets.values(), default=0),
        "active_adjacent_to_fire": near_fire,
        "evacuated": len(record["evacuations"]),
        "casualties": len(record["casualties"]),
        "exit_use": dict(sorted(Counter(str(e) for e in record["evacuations"].values()).items())),
        "new_ignitions": len(record["ignitions"]),
    }


def _decision_status(d: Any) -> str:
    s = d.diagnostics.get("status") if isinstance(d.diagnostics, Mapping) else None
    return s if isinstance(s, str) else "unreported"


def _outcome(scenario: Scenario, sim: Simulator, records: list[dict[str, Any]],
             diags: list[dict[str, Any]]) -> dict[str, Any]:
    final = sim.private_state()           # evaluator-only; never passed into policy
    statuses = Counter(p["status"] for p in final["people"])
    causes = Counter(c for rec in records for c in rec["casualties"].values())
    moves = Counter()
    for d in diags:
        moves.update(d["movement_outcomes"])
    return {
        "people": len(final["people"]), "casualties": statuses["casualty"],
        "evacuated": statuses["evacuated"], "unresolved": statuses["active"],
        "casualties_plus_unresolved": statuses["casualty"] + statuses["active"],
        "casualty_causes": dict(sorted(causes.items())),
        "movement_outcomes": dict(sorted(moves.items())),
        "blocked_moves": sum(d["blocked_moves"] for d in diags),
        "contested_targets": sum(d["contested_targets"] for d in diags),
        "peak_target_contention": max((d["max_target_contention"] for d in diags), default=0),
        "steps": len(records), "termination_reason": sim.termination_reason,
        "max_steps": scenario.max_steps,
    }


def reported_cpu_ms(d: Any) -> float | None:
    r = d.runtime.get("cpu_ms") if isinstance(getattr(d, "runtime", None), Mapping) else None
    return r if isinstance(r, (int, float)) and not isinstance(r, bool) else None


class Trajectory:
    """One teacher's closed-loop trajectory, built step by step in the collector's own simulator.

    Shared by in-process fixture collection (`_rollout`) and the killable pilot supervisor
    (`pilot_supervisor`), which replays the proposals a worker process streamed. The collector's
    simulator is the source of truth for every observation, transition and outcome.
    """

    def __init__(self, scenario: Scenario, name: str, *, clock: str) -> None:
        self.scenario, self.name, self.clock = scenario, name, clock
        self.sim = Simulator(scenario)
        self.decisions: list[dict[str, Any]] = []
        self.records: list[dict[str, Any]] = []
        self.diags: list[dict[str, Any]] = []
        self.cpu: list[float] = []
        self.wall: list[float] = []
        self.reported: list[float | None] = []
        self.status_counts: Counter[str] = Counter()
        self.failure: dict[str, Any] | None = None

    @property
    def done(self) -> bool:
        return self.sim.terminated or self.failure is not None

    def observation(self) -> dict[str, Any]:
        obs = self.sim.observe()
        check_public_observation(obs)
        return obs

    def apply(self, obs: Mapping[str, Any], raw_proposal: Mapping[str, Any], *, status: str, cpu_ms: float,
              wall_ms: float, reported: float | None) -> None:
        """Validate one proposal against `obs` (raise `_TeacherFailure` if invalid) and step the simulator."""
        try:
            prop = proposals.validate_proposal(obs, raw_proposal)
        except proposals.ProposalError as exc:
            raise _TeacherFailure("invalid_proposal", str(exc)[:500]) from None
        if prop["teacher"].get("name") != self.name:
            raise MultiTeacherError("decision teacher identity mismatch")
        self.cpu.append(round(cpu_ms, 6))
        self.wall.append(round(wall_ms, 6))
        self.reported.append(reported)
        self.status_counts[status] += 1
        rec = self.sim.step(prop["actions"])
        diag = step_diagnostics(obs, prop["actions"], rec)
        self.decisions.append({"observation": dict(obs), "observation_digest": digest(obs), "proposal": prop,
                               "teacher_status": status, "record": rec, "diagnostics": diag})
        self.records.append(rec)
        self.diags.append(diag)

    def fail(self, kind: str, message: str) -> None:
        if self.failure is None:
            self.failure = {"kind": kind, "step": self.sim.observe()["step"], "message": str(message)[:500]}

    def finish(self, identity: Mapping[str, Any], policy_seed: Any) -> dict[str, Any]:
        """The rollout record. A trajectory without a failure must be terminal; replay is re-verified."""
        if self.failure is None and not self.sim.terminated:
            self.fail("missing_policy_response", "trajectory ended before the episode terminated")
        failure = self.failure
        verdict = verify_replay(self.scenario, None, self.records, partial=failure is not None)
        want = "partial" if failure is not None else "complete"
        if verdict.status != want or verdict.final_transition_hash != self.sim.chain_hash:
            raise MultiTeacherError(f"{self.name} replay failed: {verdict.reason}")
        return {
            "identity": dict(identity), "policy_seed": policy_seed,
            "status": "failed" if failure else "complete", "failure": failure,
            "decisions": self.decisions,
            "outcome": None if failure else _outcome(self.scenario, self.sim, self.records, self.diags),
            "decision_status_counts": dict(sorted(self.status_counts.items())),
            "final_transition_hash": self.sim.chain_hash,
            "replay": {"status": verdict.status, "steps": verdict.steps_verified},
            "runtime": {"collector_cpu_ms": self.cpu, "collector_wall_ms": self.wall,
                        "teacher_reported_cpu_ms": self.reported, "total_cpu_ms": round(sum(self.cpu), 6),
                        "clock": self.clock},
        }


def _rollout(scenario: Scenario, policy: Any, *, cpu_ms_cap: float) -> dict[str, Any]:
    traj = Trajectory(scenario, policy.identity.name, clock="time.thread_time_ns around decide()")
    while not traj.done:
        obs = traj.observation()
        try:
            w0, c0 = time.perf_counter_ns(), time.thread_time_ns()
            try:
                d = policy.decide(copy.deepcopy(obs))      # a teacher can never alter the stored record
            except Exception as exc:  # noqa: BLE001 - recorded as a teacher failure, never swallowed silently
                raise _TeacherFailure("teacher_exception", f"{type(exc).__name__}: {exc}"[:500]) from None
            c1, w1 = time.thread_time_ns(), time.perf_counter_ns()
            ms = (c1 - c0) / 1e6
            if d.teacher.get("name") != policy.identity.name:
                raise MultiTeacherError("decision teacher identity mismatch")
            try:
                raw = proposals.proposal(d)
            except proposals.ProposalError as exc:
                raise _TeacherFailure("invalid_proposal", str(exc)[:500]) from None
            # Post-hoc check on collector-measured thread CPU (the teacher's own figure is kept, not trusted).
            # In-process collection cannot preempt one decide(); the TRAIN pilot uses `pilot_supervisor`.
            if ms > cpu_ms_cap or sum(traj.cpu) + ms > cpu_ms_cap:
                raise _TeacherFailure("cpu_cap", f"{ms:.1f} ms decision, cap {cpu_ms_cap} ms per episode")
            traj.apply(obs, raw, status=_decision_status(d), cpu_ms=ms, wall_ms=(w1 - w0) / 1e6,
                       reported=reported_cpu_ms(d))
        except _TeacherFailure as f:
            traj.fail(f.kind, str(f))
    return traj.finish(policy.identity.to_dict(), policy.seed)


def _teacher_provenance(policy: Any) -> dict[str, Any]:
    from .provenance import code_fingerprint
    ident = policy.identity
    try:
        pinned = reg.identity(ident.name, ident.version).key() == {
            "name": ident.name, "policy": ident.policy, "version": ident.version, "config_hash": ident.config_hash}
        status = reg.identity(ident.name, ident.version).status
    except reg.RegistryError:
        pinned, status = False, "unregistered"
    return {"identity": ident.to_dict(), "code_fingerprint": code_fingerprint(type(policy)),
            "registry_pinned": pinned, "registry_status": status if pinned else "unregistered"}


def provenance(scenario: Scenario, policies: Mapping[str, Any], source: Mapping[str, Any] | None) -> dict[str, Any]:
    """Exact identities: simulator, collector, scenario source and every teacher's code fingerprint."""
    from .s1_pipeline import module_fingerprint
    return {
        "simulator": {"observation_version": OBSERVATION_VERSION,
                      "module_fingerprint": module_fingerprint("ember_v1.sim.simulator")},
        "collector": {"schema_version": SCHEMA_VERSION,
                      "module_fingerprint": module_fingerprint(__name__)},
        "scenario": {"content_hash": scenario.content_hash(),
                     "source": dict(source) if source else {"kind": SCOPE_FIXTURE, "generator": None}},
        "teachers": {n: _teacher_provenance(p) for n, p in policies.items()},
    }


def _common_state_steps(result: Mapping[str, Mapping[str, Any]]) -> list[int]:
    seqs = [[d["observation_digest"] for d in r["decisions"]] for r in result.values()]
    out = []
    for i in range(min(len(s) for s in seqs)):
        if len({s[i] for s in seqs}) == 1:
            out.append(i)
    return out


def collect_fixture(
    scenario: Scenario, factories: Mapping[str, Callable[[], Any]], *,
    scope: str = SCOPE_FIXTURE, cpu_ms_per_teacher_cap: float = 30000,
    max_people: int = 200, max_steps: int = 30, source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Gather full paired trajectories from a tiny independent synthetic fixture.

    Cannot be called for TRAIN, DEV, calibration or held-out scenarios.
    This deliberately cannot create an S1-valid source or training label.
    A later source gate must use the same core with actual manifest and trusted
    authorization rather than silently changing this fixture-only contract.
    """
    if scope != SCOPE_FIXTURE:
        raise MultiTeacherError("real split collection requires a new reviewed multi-teacher source gate")
    if not isinstance(scenario, Scenario):
        raise MultiTeacherError("expected a validated independent Scenario")
    _check_factories(factories)
    if type(max_people) is not int or type(max_steps) is not int or max_people < 1 or max_steps < 1:
        raise MultiTeacherError("invalid fixture resource bounds")
    if isinstance(cpu_ms_per_teacher_cap, bool) or not isinstance(cpu_ms_per_teacher_cap, (int, float)) \
            or cpu_ms_per_teacher_cap <= 0:
        raise MultiTeacherError("CPU cap must be positive")
    if len(scenario.people) > max_people or scenario.max_steps > max_steps:
        raise MultiTeacherError("scenario exceeds fixture bounds (no silent truncation)")
    policies = {name: _new_policy(factories[name], name) for name in CANDIDATES}
    result = {name: _rollout(scenario, policies[name], cpu_ms_cap=cpu_ms_per_teacher_cap)
              for name in CANDIDATES}
    return assemble_group(scenario, result, provenance(scenario, policies, source), scope=scope)


def assemble_group(scenario: Scenario, result: Mapping[str, Mapping[str, Any]], prov: Mapping[str, Any], *,
                   scope: str, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Seal one matched five-teacher group. Never training-eligible here (P1-14 decides, from complete groups)."""
    if scope not in SCOPES:
        raise MultiTeacherError(f"unknown collection scope {scope!r}")
    if list(result) != list(CANDIDATES):
        raise MultiTeacherError("a group holds exactly the five teachers in registry order")
    check_matched_fire(result)
    complete = all(r["status"] == "complete" for r in result.values())
    body = {
        "schema_version": SCHEMA_VERSION, "scope": scope, "training_eligible": False,
        "supervision_eligibility": SUPERVISION_NOT_ASSESSED,
        "group_complete": complete, "comparison_eligible": complete,
        "failed_teachers": sorted(n for n, r in result.items() if r["status"] != "complete"),
        "scenario_hash": scenario.content_hash(),
        "scenario_seed_evaluator_only": scenario.seed,
        "matched_initial_state": True, "matched_exogenous_fire_prefix": True,
        "common_state_steps": _common_state_steps(result),
        "teacher_order": list(CANDIDATES), "provenance": dict(prov),
        "policies": dict(result), "causality": CAUSALITY_NOTE, **dict(extra or {}),
    }
    return seal(body)


def seal(body: Mapping[str, Any]) -> dict[str, Any]:
    body = {k: v for k, v in body.items() if k not in ("content_hash", "evidence_hash")}
    content = digest(deterministic_view(body))
    return {**body, "content_hash": content, "evidence_hash": digest({**body, "content_hash": content})}


def check_matched_fire(result: Mapping[str, Mapping[str, Any]]) -> None:
    """Fire is compared on the common time prefix; differing personal outcomes do not affect fire."""
    names = list(result)
    reference = result[names[0]]["decisions"]
    for name in names[1:]:
        other = result[name]["decisions"]
        for i in range(min(len(reference), len(other))):
            a, b = reference[i]["record"], other[i]["record"]
            if a["ignitions"] != b["ignitions"] or a["fire_draws"] != b["fire_draws"]:
                raise MultiTeacherError(f"unmatched exogenous fire for {name} at step {i + 1}")


def verify_fixture(report: Mapping[str, Any], scenario: Scenario) -> dict[str, Any]:
    """Independent integrity check of a stored report WITHOUT running any teacher.

    Re-executes every recorded joint action in a fresh simulator, regenerates
    each public observation, record, diagnostic and outcome, re-validates each
    proposal against its regenerated observation and re-hashes the report.
    """
    if seal(report)["content_hash"] != report.get("content_hash") or \
            seal(report)["evidence_hash"] != report.get("evidence_hash"):
        raise MultiTeacherError("report hash mismatch: content was altered")
    if report.get("scenario_hash") != scenario.content_hash():
        raise MultiTeacherError("report belongs to another scenario")
    if report.get("schema_version") != SCHEMA_VERSION or report.get("training_eligible") is not False \
            or report.get("scope") not in SCOPES:
        raise MultiTeacherError("unsupported schema, scope or eligibility claim")
    if list(report["policies"]) != list(CANDIDATES):
        raise MultiTeacherError("report does not hold exactly the five teachers in order")
    for name, roll in report["policies"].items():
        sim = Simulator(scenario)
        diags = []
        for i, step in enumerate(roll["decisions"]):
            obs = sim.observe()
            if step["observation"] != obs or step["observation_digest"] != digest(obs):
                raise MultiTeacherError(f"{name} step {i}: stored observation is not the regenerated state")
            prop = proposals.validate_proposal(obs, step["proposal"])
            if prop["teacher"]["name"] != name:
                raise MultiTeacherError(f"{name} step {i}: proposal of another teacher")
            rec = sim.step(prop["actions"])
            if rec != step["record"]:
                raise MultiTeacherError(f"{name} step {i}: transition does not regenerate")
            diag = step_diagnostics(obs, prop["actions"], rec)
            if diag != step["diagnostics"]:
                raise MultiTeacherError(f"{name} step {i}: diagnostics do not regenerate")
            diags.append(diag)
        if sim.chain_hash != roll["final_transition_hash"]:
            raise MultiTeacherError(f"{name}: chain hash mismatch")
        if roll["status"] == "complete":
            if not sim.terminated:
                raise MultiTeacherError(f"{name}: 'complete' trajectory is not terminal")
            if _outcome(scenario, sim, [d["record"] for d in roll["decisions"]], diags) != roll["outcome"]:
                raise MultiTeacherError(f"{name}: outcome does not regenerate")
        elif roll["failure"] is None or roll["outcome"] is not None:
            raise MultiTeacherError(f"{name}: failed trajectory without a failure record")
    check_matched_fire(report["policies"])
    complete = all(r["status"] == "complete" for r in report["policies"].values())
    if report["group_complete"] is not complete or report["comparison_eligible"] is not complete:
        raise MultiTeacherError("group completeness flag does not match the trajectories")
    return {"verified": True, "content_hash": report["content_hash"]}


def regenerate(report: Mapping[str, Any], scenario: Scenario,
               factories: Mapping[str, Callable[[], Any]], **kw: Any) -> dict[str, Any]:
    """Re-run all five teachers. Same code, scenario and policies must reproduce `content_hash`.

    A time-based teacher budget (T2/T3 `budget_limited`) can make this fail by design; then the
    report is not regenerable and must not be used as supervised evidence.
    """
    again = collect_fixture(scenario, factories, **kw)
    if again["content_hash"] != report["content_hash"]:
        raise MultiTeacherError("regeneration produced different deterministic content")
    return {"regenerated": True, "content_hash": again["content_hash"]}


def outcome_table(report: Mapping[str, Any]) -> dict[str, Any]:
    """Descriptive per-teacher closed-loop outcomes on one matched scenario. No ranking, no labels."""
    if not report.get("comparison_eligible"):
        return {"descriptive_only": True, "comparable": False, "reason": "incomplete five-teacher group",
                "failed_teachers": report.get("failed_teachers", [])}
    keys = ("casualties", "unresolved", "evacuated", "casualties_plus_unresolved", "blocked_moves",
            "contested_targets", "steps")
    return {"descriptive_only": True, "comparable": True, "causality": CAUSALITY_NOTE,
            "supervision_eligibility": SUPERVISION_NOT_ASSESSED,
            "rows": {n: {k: r["outcome"][k] for k in keys} for n, r in report["policies"].items()}}


def probe_shared_state(observation: Mapping[str, Any],
                       factories: Mapping[str, Callable[[], Any]]) -> dict[str, Any]:
    """All five actions from ONE identical public observation (not a rollout step join).

    Fixture-only; no outcome judgement and no teacher selection. Each teacher gets its own copy.
    Teachers are Markov by contract (decision = f(identity, seed, observation)), so a fresh
    instance gives the decision the teacher would make in any trajectory that reaches this state.
    """
    _check_factories(factories)
    check_public_observation(observation)
    props = {}
    for name in CANDIDATES:
        p = _new_policy(factories[name], name)
        prop = proposals.proposal(p.decide(copy.deepcopy(dict(observation))))
        proposals.validate_proposal(observation, prop)
        props[name] = prop
    delta = proposals.disagreement(observation, list(props.values()))
    distinct = Counter(digest(p["actions"]) for p in props.values())
    body = {"schema_version": SCHEMA_VERSION, "scope": SCOPE_FIXTURE,
            "training_eligible": False, "observation_digest": digest(observation),
            "proposals": props, "disagreement": delta, "distinct_joint_actions": len(distinct),
            "note": "Same public state; agreement alone is not quality or causal evidence."}
    return {**body, "probe_hash": digest(deterministic_view(body))}
