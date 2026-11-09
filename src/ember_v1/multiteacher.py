"""CPU-only five-teacher matched rollout primitives (P1-12, issue #45).

THIS IS A FIXTURE/SCHEMA IMPLEMENTATION, NOT A TRAINING DATA EXPORTER.
It intentionally refuses real TRAIN/DEV/holdout labelling until a separately
reviewed multi-teacher source protocol, leakage gate and cost budget exist.

Every policy gets its own fresh simulator from the SAME independent scenario.
Only observe() enters a teacher. The scenario/fire seed and private state are
evaluator-only. Fire spread is action-independent; unequal fire histories over
overlapping steps invalidate matched comparisons.

A rollout is a whole policy trajectory. A same-step observation from different
policy trajectories is NOT a common decision state. Use probe_shared_state()
for a valid same-state disagreement record. Negative outcomes here are
descriptive, not causal labels for individual actions.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any

from .sim import Scenario, Simulator, verify_replay
from .sim.canonical import digest
from .teacher import proposals
from .teacher.registry import CANDIDATES

SCHEMA_VERSION = "matched-multi-teacher@0.1.0"
SCOPE_FIXTURE = "synthetic_fixture"
CAUSALITY_NOTE = (
    "Closed-loop outcomes are associated with complete policy trajectories; "
    "they do not prove that any one action caused a casualty."
)


class MultiTeacherError(ValueError):
    """Untrusted, incomplete, unpaired or over-budget trajectory evidence."""


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


def _rollout(scenario: Scenario, policy: Any, *, cpu_ms_cap: float) -> dict[str, Any]:
    sim = Simulator(scenario)
    transitions: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    outcomes: Counter[str] = Counter()
    casualties: Counter[str] = Counter()
    decision_cpu_ms = 0.0
    while not sim.terminated:
        obs = sim.observe()
        d = policy.decide(obs)
        if d.teacher["name"] != policy.identity.name:
            raise MultiTeacherError("decision teacher identity mismatch")
        prop = proposals.proposal(d)
        proposals.validate_proposal(obs, prop)
        ms = d.runtime.get("cpu_ms", 0.0)
        if isinstance(ms, bool) or not isinstance(ms, (int, float)) or not (0 <= ms <= cpu_ms_cap):
            raise MultiTeacherError(f"{policy.identity.name} exceeded per-decision CPU cap")
        decision_cpu_ms += ms
        if decision_cpu_ms > cpu_ms_cap:
            raise MultiTeacherError(f"{policy.identity.name} exceeded episode CPU cap")
        rec = sim.step(prop["actions"])
        transitions.append({"observation": obs, "proposal": prop, "record": rec})
        records.append(rec)
        outcomes.update(m["outcome"] for m in rec["movements"].values())
        casualties.update(rec["casualties"].values())
    verdict = verify_replay(scenario, None, records, partial=False)
    if not verdict.complete or verdict.final_transition_hash != sim.chain_hash:
        raise MultiTeacherError(f"{policy.identity.name} full replay failed: {verdict.reason}")
    final = sim.private_state()           # evaluator-only; never passed into policy
    statuses = Counter(p["status"] for p in final["people"])
    return {
        "identity": policy.identity.to_dict(), "policy_seed": policy.seed,
        "decisions": transitions,
        "outcome": {
            "people": len(final["people"]), "casualties": statuses["casualty"],
            "evacuated": statuses["evacuated"], "unresolved": statuses["active"],
            "casualty_causes": dict(sorted(casualties.items())),
            "movement_outcomes": dict(sorted(outcomes.items())),
            "blocked_moves": sum(v for k, v in outcomes.items() if k.startswith("blocked_")),
            "steps": len(records), "termination_reason": sim.termination_reason,
        },
        "policy_cpu_ms": round(decision_cpu_ms, 6),
        "final_transition_hash": sim.chain_hash,
        "replay": {"status": verdict.status, "steps": verdict.steps_verified},
    }


def collect_fixture(
    scenario: Scenario, factories: Mapping[str, Callable[[], Any]], *,
    scope: str = SCOPE_FIXTURE, cpu_ms_per_teacher_cap: float = 30000,
    max_people: int = 200, max_steps: int = 30,
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
    if not isinstance(cpu_ms_per_teacher_cap, (int, float)) or cpu_ms_per_teacher_cap <= 0:
        raise MultiTeacherError("CPU cap must be positive")
    if len(scenario.people) > max_people or scenario.max_steps > max_steps:
        raise MultiTeacherError("scenario exceeds fixture bounds (no silent truncation)")
    result = {name: _rollout(scenario, _new_policy(factories[name], name),
                             cpu_ms_cap=cpu_ms_per_teacher_cap)
              for name in CANDIDATES}
    reference = result[CANDIDATES[0]]["decisions"]
    # Each policy may terminate at a different time. Fire history is compared
    # only for the common prefix; differing personal outcomes do not affect fire.
    for name in CANDIDATES[1:]:
        other = result[name]["decisions"]
        for i in range(min(len(reference), len(other))):
            a, b = reference[i]["record"], other[i]["record"]
            if a["ignitions"] != b["ignitions"] or a["fire_draws"] != b["fire_draws"]:
                raise MultiTeacherError(f"unmatched exogenous fire for {name} at step {i + 1}")
    body = {
        "schema_version": SCHEMA_VERSION, "scope": scope, "training_eligible": False,
        "scenario_hash": scenario.content_hash(),
        "scenario_seed_evaluator_only": scenario.seed,
        "matched_initial_state": True, "matched_exogenous_fire_prefix": True,
        "teacher_order": list(CANDIDATES), "policies": result, "causality": CAUSALITY_NOTE,
    }
    return {**body, "evidence_hash": digest(body)}


def probe_shared_state(observation: Mapping[str, Any],
                       factories: Mapping[str, Callable[[], Any]]) -> dict[str, Any]:
    """All five actions from ONE identical public observation (not a rollout step join).

    Fixture-only; no outcome judgement and no teacher selection.
    """
    _check_factories(factories)
    props = {}
    for name in CANDIDATES:
        p = _new_policy(factories[name], name)
        prop = proposals.proposal(p.decide(observation))
        proposals.validate_proposal(observation, prop)
        props[name] = prop
    delta = proposals.disagreement(observation, list(props.values()))
    body = {"schema_version": SCHEMA_VERSION, "scope": SCOPE_FIXTURE,
            "training_eligible": False, "observation_digest": digest(observation),
            "proposals": props, "disagreement": delta,
            "note": "Same public state; agreement alone is not quality or causal evidence."}
    return {**body, "probe_hash": digest(body)}
