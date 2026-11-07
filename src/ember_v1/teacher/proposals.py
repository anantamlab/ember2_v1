"""Common teacher proposal record and joint-action checks for the five-teacher gate (issue #25).

A proposal is what one teacher says for one public observation. It is built
from a `PolicyDecision` and carries: teacher identity, observation digest, the
full ID -> action map, per-action values only when the teacher really computes
them (T2), compute cost, the teacher seed and a proposal hash.

`validate_proposal` rejects partial, invalid or fabricated proposals.
`joint_check` runs a joint action map (one teacher's, or a combination) through
the exact public resolver and reports blocked moves; combining per-person
actions of different teachers can create a blocked or harmful joint plan.
`disagreement` describes how proposals differ. Agreement is NOT a quality
measure (docs/P1_TEACHER_DIRECTION.md section 6).

Nothing here chooses a teacher or merges scores. There is no weighted average
of unlike scores.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any

from ..contract import validate_actions, validate_observation
from ..sim.canonical import digest
from ..sim.movement import resolve_moves
from .grid import PublicGrid
from .interface import PolicyDecision

PROPOSAL_VERSION = "teacher-proposal@1.0.0"
VALUE_TEACHERS = frozenset({"T2"})          # only T2 computes per-action values
_KEYS = {"proposal_version", "teacher", "observation_digest", "step", "actions", "values", "policy_seed",
         "compute", "proposal_hash"}


class ProposalError(ValueError):
    """A proposal is partial, invalid, fabricated or does not match its observation."""


def _values(name: str, decision: PolicyDecision) -> dict[str, Any] | None:
    if name not in VALUE_TEACHERS:
        return None
    people = decision.diagnostics.get("people", {})
    vals = {pid: p["values"] for pid, p in sorted(people.items()) if p.get("values") is not None}
    return vals or None


def proposal(decision: PolicyDecision) -> dict[str, Any]:
    """Proposal record of one teacher decision."""
    t = decision.teacher
    body = {"proposal_version": PROPOSAL_VERSION,
            "teacher": {k: t[k] for k in ("name", "policy", "version", "config_hash")},
            "observation_digest": decision.reproducibility["observation_digest"],
            "step": decision.reproducibility["step"], "actions": dict(sorted(decision.actions.items())),
            "values": _values(t["name"], decision), "policy_seed": decision.reproducibility["policy_seed"],
            "compute": {"cpu_ms": decision.runtime.get("cpu_ms"), "wall_ms": decision.runtime.get("wall_ms")}}
    return {**body, "proposal_hash": digest({k: v for k, v in body.items() if k != "compute"})}


def validate_proposal(obs: Mapping[str, Any], prop: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on a partial, invalid, tampered or fabricated proposal. Return the validated record."""
    if not isinstance(prop, Mapping) or set(prop) != _KEYS:
        raise ProposalError("proposal has missing or extra fields")
    if prop["proposal_version"] != PROPOSAL_VERSION:
        raise ProposalError("unsupported proposal version")
    body = {k: v for k, v in prop.items() if k not in ("compute", "proposal_hash")}
    if digest(body) != prop["proposal_hash"]:
        raise ProposalError("proposal hash does not match its content")
    o = validate_observation(obs)
    if digest(o) != prop["observation_digest"] or o["step"] != prop["step"]:
        raise ProposalError("proposal was made for another observation")
    active = {p["id"] for p in o["people"] if p["status"] == "active"}
    acts = prop["actions"]
    if not isinstance(acts, Mapping) or set(acts) != active:
        missing, extra = sorted(active - set(acts or {})), sorted(set(acts or {}) - active)
        raise ProposalError(f"partial proposal: missing {missing[:5]}, unknown {extra[:5]}")
    try:
        validate_actions(o, dict(acts))
    except ValueError as e:
        raise ProposalError(f"invalid action: {e}") from None
    name = prop["teacher"]["name"]
    if prop["values"] is not None and name not in VALUE_TEACHERS:
        raise ProposalError(f"{name} computes no per-action values; values are fabricated")
    if prop["values"] is not None:
        allowed = set(o["allowed_actions"])
        for pid, v in prop["values"].items():
            if pid not in active or not isinstance(v, Mapping) or not set(v) <= allowed:
                raise ProposalError(f"values for {pid!r} are not over allowed actions of an active person")
    return dict(prop)


def joint_check(obs: Mapping[str, Any], actions: Mapping[str, str]) -> dict[str, Any]:
    """Resolve one joint action map with the exact public resolver. Count outcomes and blocked moves."""
    o = validate_observation(obs)
    acts = validate_actions(o, dict(actions))
    pg = PublicGrid.from_observation(o)
    res = resolve_moves(height=pg.height, width=pg.width, walls=pg.walls, burning=pg.burning,
                        enter_fire=pg.enter_fire, positions=dict(pg.active), actions=acts)
    outcomes = Counter(out for out, _ in res.values())
    blocked = sorted(pid for pid, (out, _) in res.items() if out.startswith("blocked_"))
    return {"outcomes": dict(sorted(outcomes.items())), "blocked": blocked, "valid_joint_plan": not blocked}


def combine(obs: Mapping[str, Any], choice: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Joint action map from per-person teacher choices {pid: proposal}, then the exact joint check.

    Every chosen proposal must be valid for `obs`. The result records which
    teacher gave each action, so an arbiter decision is never attributed to a
    single teacher.
    """
    o = validate_observation(obs)
    active = {p["id"] for p in o["people"] if p["status"] == "active"}
    if set(choice) != active:
        raise ProposalError("a combined plan needs a choice for every active person")
    for p in {id(v): v for v in choice.values()}.values():
        validate_proposal(o, p)
    acts = {pid: choice[pid]["actions"][pid] for pid in sorted(active)}
    src = {pid: choice[pid]["teacher"]["name"] for pid in sorted(active)}
    return {"actions": acts, "sources": src, "source_proposals": sorted({choice[p]["proposal_hash"] for p in active}),
            "joint": joint_check(o, acts)}


def disagreement(obs: Mapping[str, Any], props: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Descriptive disagreement between valid proposals (not a quality measure)."""
    o = validate_observation(obs)
    for p in props:
        validate_proposal(o, p)
    names = [p["teacher"]["name"] for p in props]
    if len(set(names)) != len(names):
        raise ProposalError("one proposal per teacher")
    active = sorted(p["id"] for p in o["people"] if p["status"] == "active")
    unanimous = sum(len({p["actions"][pid] for p in props}) == 1 for pid in active)
    pair = {f"{a['teacher']['name']}-{b['teacher']['name']}":
            (sum(a["actions"][pid] == b["actions"][pid] for pid in active) / len(active) if active else None)
            for a, b in combinations(props, 2)}
    return {"people": len(active), "teachers": names, "unanimous_people": unanimous,
            "unanimous_fraction": unanimous / len(active) if active else None, "pairwise_agreement": pair,
            "note": "agreement is descriptive only; it is not evidence of label quality"}
