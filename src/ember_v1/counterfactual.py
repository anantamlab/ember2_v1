"""Controlled joint-action counterfactual fixtures (P1-13, issue #46).

Only hand-built synthetic scenarios are accepted: NO TRAIN/DEV/selection data,
no teacher or student training preferences, no GPU or external benchmark.

Two interventions begin at the SAME replay-validated simulator state, change
the COMPLETE action map for ONE step, then use a FRESH common T1 continuation.
The simulator gives the same independent fire seed to each branch; the acting
policy sees only public observe(). Matching fire is verified over common steps.
This estimates a *joint-action intervention under that continuation*, NOT
whether an individual action directly caused someone's later death.

Censored horizons must not be turned into success/preference labels.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .contract import validate_actions
from .sim import Scenario, Simulator, verify_replay
from .sim.canonical import digest
from .teacher import proposals
from .teacher.t1 import T1CongestionAware

CF_VERSION = "joint-counterfactual-fixture@0.1.0"
SCOPE = "synthetic_fixture"
INTERVENTION_SCOPE = "one full legal joint action; identical T1 continuation"
CAUSAL_LIMIT = (
    "Conditional paired outcome of a complete joint action under a fixed T1 "
    "continuation and one simulator fire seed. Not an individual-action death attribution "
    "or a population causal effect. No preference training eligibility."
)


class CounterfactualError(ValueError):
    """Invalid prefix, shared-state intervention, provenance or resource bound."""


def _replay_prefix(scenario: Scenario, records: Sequence[Mapping[str, Any]]) -> Simulator:
    sim = Simulator(scenario)
    for t, record in enumerate(records, 1):
        if sim.terminated or not isinstance(record, Mapping) or "actions" not in record:
            raise CounterfactualError(f"invalid prefix at step {t}")
        actual = sim.step(record["actions"])
        if actual != record:
            raise CounterfactualError(f"prefix diverged at step {t} (same action is not enough)")
    if sim.terminated:
        raise CounterfactualError("prefix is already terminal; cannot intervene")
    check = verify_replay(scenario, None, records, partial=True)
    if check.status != "partial" or check.steps_verified != len(records):
        raise CounterfactualError(f"prefix not replay-verified: {check.status}/{check.reason}")
    return sim


def _statuses(sim: Simulator) -> dict[str, int]:
    # Evaluator-only statistics, never added to an actor's observation.
    c = Counter(p["status"] for p in sim.private_state()["people"])
    return {"casualties": c["casualty"], "evacuated": c["evacuated"], "unresolved": c["active"]}


def counterfactual_fixture(
    scenario: Scenario,
    prefix_records: Sequence[Mapping[str, Any]],
    alternatives: Mapping[str, Mapping[str, str]], *,
    scope: str = SCOPE, horizon: int = 4, max_horizon: int = 8,
    continuation_factory: Callable[[], Any] = T1CongestionAware,
    cpu_ms_per_branch_cap: float = 30000,
) -> dict[str, Any]:
    """Branch two to five validated, different joint actions at one common state.

    The horizon counts the intervention step plus continuation decisions.
    All branches are paired on the exogenous fire stream through their shared
    time prefix. If a branch remains active at the limit, its outcome is
    CENSORED; do not rank it as an independently verified positive.
    """
    if scope != SCOPE:
        raise CounterfactualError("real split counterfactuals require reviewed source authorization")
    if not isinstance(scenario, Scenario):
        raise CounterfactualError("scenario must be a validated Scenario")
    if type(horizon) is not int or type(max_horizon) is not int or not 1 <= horizon <= max_horizon <= 8:
        raise CounterfactualError("invalid or unbounded horizon")
    if not isinstance(cpu_ms_per_branch_cap, (int, float)) or cpu_ms_per_branch_cap <= 0:
        raise CounterfactualError("invalid branch CPU cap")
    if not isinstance(alternatives, Mapping) or not 2 <= len(alternatives) <= 5:
        raise CounterfactualError("requires two to five joint-action alternatives")
    common = _replay_prefix(scenario, prefix_records)
    obs = common.observe()
    validated = {}
    for name, action_map in alternatives.items():
        if not isinstance(name, str) or not name or not isinstance(action_map, Mapping):
            raise CounterfactualError("every named intervention must be a joint-action mapping")
        try:
            validated[name] = validate_actions(obs, dict(action_map))
        except ValueError as exc:
            raise CounterfactualError(f"{name}: incomplete or invalid joint map: {exc}") from None
    if len({digest(a) for a in validated.values()}) != len(validated):
        raise CounterfactualError("identical alternatives cannot make a causal contrast")
    before = _statuses(common)
    branches = {}
    for name in sorted(validated):
        sim = _replay_prefix(scenario, prefix_records)
        joint = proposals.joint_check(sim.observe(), validated[name])
        records = [sim.step(validated[name])]
        policy = continuation_factory()
        if policy.identity.name != "T1":
            raise CounterfactualError("fixture continuation must be frozen T1")
        cpu = 0.0
        while not sim.terminated and len(records) < horizon:
            d = policy.decide(sim.observe())  # public fields ONLY
            ms = d.runtime.get("cpu_ms", 0.0)
            if not isinstance(ms, (int, float)) or ms < 0:
                raise CounterfactualError("invalid continuation CPU measurement")
            cpu += ms
            if cpu > cpu_ms_per_branch_cap:
                raise CounterfactualError("continuation CPU budget exceeded")
            prop = proposals.proposal(d)
            proposals.validate_proposal(sim.observe(), prop)
            records.append(sim.step(prop["actions"]))
        chain = [*prefix_records, *records]
        proof = verify_replay(scenario, None, chain, partial=not sim.terminated)
        if proof.status != ("complete" if sim.terminated else "partial"):
            raise CounterfactualError(f"{name}: branch replay failed: {proof.reason}")
        after = _statuses(sim)
        branches[name] = {
            "intervention_actions": validated[name], "joint_resolver": joint,
            "intervention_transition": records[0],
            "horizon_records": records, "continuation": policy.identity.to_dict(),
            "continuation_cpu_ms": round(cpu, 6),
            "outcome": {**after, "new_casualties": after["casualties"] - before["casualties"],
                        "new_evacuations": after["evacuated"] - before["evacuated"],
                        "censored": not sim.terminated},
            "replay": proof.status, "chain_hash": sim.chain_hash,
        }
    names = sorted(branches)
    for b in names[1:]:
        a_rec, b_rec = branches[names[0]]["horizon_records"], branches[b]["horizon_records"]
        for t in range(min(len(a_rec), len(b_rec))):
            if a_rec[t]["ignitions"] != b_rec[t]["ignitions"] or \
                    a_rec[t]["fire_draws"] != b_rec[t]["fire_draws"]:
                raise CounterfactualError(f"future fire differs at common step {t + 1}")
    contrasts = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ca, cb = branches[a]["outcome"], branches[b]["outcome"]
            contrasts.append({
                "a": a, "b": b,
                "delta_b_minus_a_new_casualties": cb["new_casualties"] - ca["new_casualties"],
                "delta_b_minus_a_unresolved": cb["unresolved"] - ca["unresolved"],
                "delta_b_minus_a_new_evacuations": cb["new_evacuations"] - ca["new_evacuations"],
                "censored": ca["censored"] or cb["censored"],
                "inference": "paired descriptive intervention under fixed T1 continuation; no population CI",
                "training_preference_allowed": False,
            })
    body = {
        "counterfactual_version": CF_VERSION, "scope": SCOPE, "training_eligible": False,
        "scenario_hash": scenario.content_hash(), "common_observation_digest": digest(obs),
        "prefix_hash": digest(list(prefix_records)), "prefix_steps": len(prefix_records),
        "horizon": horizon, "matched_exogenous_fire_prefix": True,
        "continuation": "T1 v1.0.0 common observation-only policy",
        "branches": branches, "contrasts": contrasts, "causality_limit": CAUSAL_LIMIT,
    }
    return {**body, "report_hash": digest(body)}
