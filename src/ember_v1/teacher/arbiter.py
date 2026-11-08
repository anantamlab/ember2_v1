"""Joint-map arbiter contract (issue #25, protocol draft section 4 S1). Not run; no arbiter is frozen.

An arbiter may be compared with standalone teachers only under its own frozen
protocol (`validate_arbiter_protocol`), referenced by the selection protocol.
`select_joint_map` is the only selection rule it may use:

- it chooses among WHOLE joint action maps (a teacher's full proposal, or a
  combined map from `proposals.combine`), never per-person directions;
- every candidate map must pass the exact public resolver
  (`proposals.joint_check`); a map with any blocked move is rejected and
  recorded, even when every single action is legal;
- all scores must carry the one predeclared `score_id`; scores of another kind
  are refused, and no score is ever averaged or mixed;
- when no candidate map is valid, the predeclared fallback teacher's full
  proposal is used, and it must be resolver-valid too.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..contract import validate_actions, validate_observation
from ..sim.canonical import digest
from . import registry as reg
from .proposals import ProposalError, joint_check, validate_proposal

ARBITER_PROTOCOL_VERSION = "teacher-arbiter-protocol@1.0.0"
ARBITER_FIELDS = ("protocol_version", "protocol_id", "score_id", "candidates", "fallback_teacher",
                  "observation_only", "independent_teacher_sampling", "equal_total_planning_cpu",
                  "whole_map_only", "joint_resolver_valid_required", "no_score_averaging", "frozen")


class ArbiterError(ValueError):
    """Invalid arbiter protocol, incompatible scores, or no resolver-valid joint map."""


def validate_arbiter_protocol(ap: Any, *, eligible: Sequence[str]) -> dict[str, Any]:
    """Fail closed unless every arbiter safeguard is predeclared. Check the hash."""
    if not isinstance(ap, Mapping) or set(ap) != set(ARBITER_FIELDS) | {"protocol_hash"}:
        raise ArbiterError(f"arbiter protocol needs exactly {ARBITER_FIELDS} and protocol_hash")
    if ap["protocol_version"] != ARBITER_PROTOCOL_VERSION:
        raise ArbiterError("unsupported arbiter protocol version")
    for k in ("observation_only", "independent_teacher_sampling", "equal_total_planning_cpu", "whole_map_only",
              "joint_resolver_valid_required", "no_score_averaging", "frozen"):
        if ap[k] is not True:
            raise ArbiterError(f"arbiter protocol needs {k}: true")
    if not isinstance(ap["score_id"], str) or not ap["score_id"]:
        raise ArbiterError("arbiter needs one predeclared score_id")
    cands = list(ap["candidates"])
    if not cands or not set(cands) <= set(eligible) or len(set(cands)) != len(cands):
        raise ArbiterError("arbiter candidates must be distinct eligible teachers")
    if ap["fallback_teacher"] not in cands:
        raise ArbiterError("fallback teacher must be one of the arbiter candidates")
    for c in cands:
        if c not in reg.CANDIDATES:
            raise ArbiterError(f"unknown candidate {c}")
    if digest({k: v for k, v in ap.items() if k != "protocol_hash"}) != ap["protocol_hash"]:
        raise ArbiterError("arbiter protocol hash does not match its content")
    return dict(ap)


def accept_joint_map(obs: Mapping[str, Any], combined: Mapping[str, Any]) -> dict[str, str]:
    """The actions of a combined map only if the exact resolver accepts the whole map; else raise."""
    o = validate_observation(obs)
    acts = validate_actions(o, dict(combined["actions"]))
    if not joint_check(o, acts)["valid_joint_plan"]:
        raise ArbiterError("combined joint map fails the resolver; it cannot be selected")
    return acts


def select_joint_map(obs: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]], *, score_id: str,
                     fallback: Mapping[str, Any]) -> dict[str, Any]:
    """Choose the best-scored resolver-valid whole map, else the valid fallback proposal.

    Each candidate is {"label", "actions" (full map), "score": {"score_id", "value"}}.
    """
    o = validate_observation(obs)
    active = {p["id"] for p in o["people"] if p["status"] == "active"}
    valid, rejected = [], []
    for i, c in enumerate(candidates):
        sc = c.get("score") or {}
        if sc.get("score_id") != score_id:
            raise ArbiterError(f"candidate {c.get('label', i)!r} has score {sc.get('score_id')!r}, "
                               f"not the predeclared {score_id!r}; unlike scores are never mixed or averaged")
        v = sc.get("value")
        if not isinstance(v, (int, float)) or isinstance(v, bool) or v != v:
            raise ArbiterError(f"candidate {c.get('label', i)!r} has a non-numeric score")
        acts = c.get("actions")
        if not isinstance(acts, Mapping) or set(acts) != active:
            rejected.append({"label": c.get("label", i), "reason": "not a whole joint map"})
            continue
        try:
            acts = validate_actions(o, dict(acts))
        except ValueError as e:
            rejected.append({"label": c.get("label", i), "reason": f"invalid action: {e}"})
            continue
        jc = joint_check(o, acts)
        if not jc["valid_joint_plan"]:
            rejected.append({"label": c.get("label", i), "reason": "resolver blocked moves", "blocked": jc["blocked"]})
            continue
        valid.append((v, -i, c.get("label", i), acts))
    if valid:
        v, _, label, acts = max(valid)
        return {"actions": acts, "chosen": label, "score": v, "fallback_used": False, "rejected": rejected}
    try:
        fb = validate_proposal(o, fallback)
    except ProposalError as e:
        raise ArbiterError(f"fallback proposal invalid: {e}") from None
    if not joint_check(o, fb["actions"])["valid_joint_plan"]:
        raise ArbiterError("fallback proposal fails the resolver; stop (no valid joint map)")
    return {"actions": dict(fb["actions"]), "chosen": f"fallback:{fb['teacher']['name']}", "score": None,
            "fallback_used": True, "rejected": rejected}
