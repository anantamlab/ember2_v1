"""Controlled joint-action counterfactual fixtures (P1-13, issue #46).

Only hand-built synthetic scenarios are accepted: NO TRAIN/DEV/selection data,
no teacher or student training preferences, no GPU or external benchmark.

Procedure (`shared_state_study` runs it end to end):
1. replay a verified trajectory prefix to ONE common simulator state;
2. ask all five teachers for a decision on that same public observation
   (`multiteacher.probe_shared_state`; teachers are Markov in the observation);
3. group identical joint actions; a real disagreement needs two or more;
4. every alternative is a COMPLETE legal joint action (a per-person mix of
   teachers is allowed only with no blocked move in the whole map);
5. execute each alternative for one step from the identical state;
6. future fire is matched: replicate 0 is the factual stream, replicates
   1..K-1 fork the future fire stream (`Simulator.fork_fire`) with the SAME
   fork seed for every alternative;
7. continue under the same frozen, registry-pinned continuation (default T1);
8. measure casualties, evacuations, unresolved, blocked moves and contention;
9. record horizon, censoring, per-replicate deltas and a conservative verdict;
10. bind the prefix, probe, alternatives and fork seeds into the report hash.

This estimates a *joint-action intervention under that continuation*, NOT
whether an individual action directly caused someone's later death. A
censored, mixed-sign or under-replicated contrast is INCONCLUSIVE, never a
preference. Nothing here is training-eligible.
"""
from __future__ import annotations

import copy
import math
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from . import multiteacher as mt
from .contract import validate_actions
from .sim import Scenario, Simulator, verify_replay
from .sim.canonical import digest
from .sim.rng import stream_seed
from .teacher import proposals
from .teacher import registry as reg
from .teacher.t1 import T1CongestionAware

CF_VERSION = "joint-counterfactual-fixture@0.2.0"
SCOPE = "synthetic_fixture"
INTERVENTION_SCOPE = "one full legal joint action; identical frozen continuation; matched future fire"
CAUSAL_LIMIT = (
    "Conditional paired outcome of a complete joint action under a fixed frozen "
    "continuation and matched future fire replicates. Not an individual-action death attribution "
    "or a population causal effect. No preference training eligibility."
)
HARD_MAX_HORIZON = 64
MAX_REPLICATES = 16
# Preregistered verdict rule (fixture version; P1-14 must freeze its own copy before use).
VERDICT_RULE = {
    "primary": "casualties_plus_unresolved at branch end",
    "guard": "casualties at branch end",
    "min_replicates_stochastic": 3,
    "better": "every replicate: primary and guard not worse, and primary strictly better in one or more",
    "inconclusive": ["censored", "insufficient_replicates", "mixed_sign"],
    "no_difference": "every replicate: primary and guard equal",
}
VERDICTS = ("a_better", "b_better", "no_difference", "inconclusive")


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


def fork_seeds(scenario: Scenario, prefix_hash: str, replicates: int) -> list[int | None]:
    """Evaluator-only future-fire seeds. Replicate 0 is the factual (unforked) stream."""
    return [None] + [stream_seed(scenario.seed, f"p1-13/fork/{prefix_hash}/{k}") for k in range(1, replicates)]


def _check_continuation(policy: Any) -> dict[str, Any]:
    ident = policy.identity
    try:
        pinned = reg.identity(ident.name, ident.version)
    except reg.RegistryError:
        pinned = None
    key = {"name": ident.name, "policy": ident.policy, "version": ident.version, "config_hash": ident.config_hash}
    if pinned is None or pinned.status != reg.RETAINED or pinned.key() != key:
        raise CounterfactualError("continuation must be a frozen, retained, registry-pinned teacher")
    return ident.to_dict()


def _run_branch(scenario: Scenario, prefix: Sequence[Mapping[str, Any]], actions: Mapping[str, str],
                fork_seed: int | None, horizon: int, continuation_factory: Callable[[], Any],
                cpu_cap: float) -> dict[str, Any]:
    sim = _replay_prefix(scenario, prefix)
    before = _statuses(sim)
    if fork_seed is not None:
        sim.fork_fire(fork_seed)
    obs = sim.observe()
    joint = proposals.joint_check(obs, actions)
    rec = sim.step(dict(actions))
    records, diags = [rec], [mt.step_diagnostics(obs, actions, rec)]
    policy = continuation_factory()
    ident = _check_continuation(policy)
    cpu = 0.0
    while not sim.terminated and len(records) < horizon:
        obs = sim.observe()
        c0 = time.thread_time_ns()
        d = policy.decide(copy.deepcopy(obs))            # a copy of public fields ONLY
        cpu += (time.thread_time_ns() - c0) / 1e6
        if cpu > cpu_cap:
            raise CounterfactualError("continuation CPU budget exceeded")
        prop = proposals.validate_proposal(obs, proposals.proposal(d))
        rec = sim.step(prop["actions"])
        records.append(rec)
        diags.append(mt.step_diagnostics(obs, prop["actions"], rec))
    after = _statuses(sim)
    return {"sim": sim, "joint": joint, "records": records, "continuation": ident, "cpu_ms": round(cpu, 6),
            "outcome": {**after, "casualties_plus_unresolved": after["casualties"] + after["unresolved"],
                        "new_casualties": after["casualties"] - before["casualties"],
                        "new_evacuations": after["evacuated"] - before["evacuated"],
                        "blocked_moves": sum(x["blocked_moves"] for x in diags),
                        "contested_targets": sum(x["contested_targets"] for x in diags),
                        "steps_run": len(records), "censored": not sim.terminated,
                        "termination_reason": sim.termination_reason}}


def replay_branch(scenario: Scenario, prefix: Sequence[Mapping[str, Any]], fork_seed: int | None,
                  records: Sequence[Mapping[str, Any]]) -> str:
    """Re-execute a stored (possibly forked) branch; return its chain hash or raise."""
    sim = _replay_prefix(scenario, prefix)
    if fork_seed is not None:
        sim.fork_fire(fork_seed)
    for t, r in enumerate(records):
        if sim.step(r["actions"]) != r:
            raise CounterfactualError(f"branch does not regenerate at horizon step {t + 1}")
    if fork_seed is None:
        proof = verify_replay(scenario, None, [*prefix, *records], partial=not sim.terminated)
        if proof.status != ("complete" if sim.terminated else "partial"):
            raise CounterfactualError(f"branch replay failed: {proof.reason}")
    return sim.chain_hash


def sign_test_p(wins: int, losses: int) -> float | None:
    """Exact two-sided sign test on non-tied replicates (descriptive only)."""
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def classify(per_rep: Sequence[Mapping[str, Any]], *, stochastic: bool) -> dict[str, Any]:
    """Conservative verdict for one pair from per-replicate outcomes (b minus a)."""
    if any(r["censored"] for r in per_rep):
        return {"verdict": "inconclusive", "reason": "censored"}
    dp = [r["delta_casualties_plus_unresolved"] for r in per_rep]
    dg = [r["delta_casualties"] for r in per_rep]
    if all(x == 0 for x in dp) and all(x == 0 for x in dg):
        return {"verdict": "no_difference", "reason": "equal in every replicate"}
    if stochastic and len(per_rep) < VERDICT_RULE["min_replicates_stochastic"]:
        return {"verdict": "inconclusive", "reason": "insufficient_replicates"}
    if all(x <= 0 for x in dp) and all(x <= 0 for x in dg) and any(x < 0 for x in dp):
        return {"verdict": "b_better", "reason": "weakly dominant in every replicate"}
    if all(x >= 0 for x in dp) and all(x >= 0 for x in dg) and any(x > 0 for x in dp):
        return {"verdict": "a_better", "reason": "weakly dominant in every replicate"}
    return {"verdict": "inconclusive", "reason": "mixed_sign"}


def counterfactual_fixture(
    scenario: Scenario,
    prefix_records: Sequence[Mapping[str, Any]],
    alternatives: Mapping[str, Mapping[str, str]], *,
    scope: str = SCOPE, horizon: int = 4, max_horizon: int = 8,
    continuation_factory: Callable[[], Any] = T1CongestionAware,
    cpu_ms_per_branch_cap: float = 30000, replicates: int = 1,
    alternative_sources: Mapping[str, Any] | None = None,
    prefix_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Branch two to five validated, different joint actions at one common state.

    The horizon counts the intervention step plus continuation decisions.
    All alternatives share each replicate's future fire stream. If a branch remains
    active at the limit, its outcome is CENSORED; it is never ranked.
    """
    if scope != SCOPE:
        raise CounterfactualError("real split counterfactuals require reviewed source authorization")
    if not isinstance(scenario, Scenario):
        raise CounterfactualError("scenario must be a validated Scenario")
    if type(horizon) is not int or type(max_horizon) is not int or \
            not 1 <= horizon <= max_horizon <= HARD_MAX_HORIZON:
        raise CounterfactualError("invalid or unbounded horizon")
    if isinstance(cpu_ms_per_branch_cap, bool) or not isinstance(cpu_ms_per_branch_cap, (int, float)) \
            or cpu_ms_per_branch_cap <= 0:
        raise CounterfactualError("invalid branch CPU cap")
    stochastic = scenario.fire_model != "deterministic"
    if type(replicates) is not int or not 1 <= replicates <= MAX_REPLICATES:
        raise CounterfactualError("invalid replicate count")
    if not stochastic and replicates != 1:
        raise CounterfactualError("deterministic fire has one exact future; use replicates=1")
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
    prefix_hash = digest(list(prefix_records))
    seeds = fork_seeds(scenario, prefix_hash, replicates)
    before = _statuses(common)
    names = sorted(validated)
    branches: dict[str, Any] = {}
    runs: dict[str, list[dict[str, Any]]] = {}
    for name in names:
        runs[name] = [_run_branch(scenario, prefix_records, validated[name], s, horizon,
                                  continuation_factory, cpu_ms_per_branch_cap) for s in seeds]
    for k in range(replicates):                     # matched fire within each replicate
        for b in names[1:]:
            a_rec, b_rec = runs[names[0]][k]["records"], runs[b][k]["records"]
            for t in range(min(len(a_rec), len(b_rec))):
                if a_rec[t]["ignitions"] != b_rec[t]["ignitions"] or \
                        a_rec[t]["fire_draws"] != b_rec[t]["fire_draws"]:
                    raise CounterfactualError(f"future fire differs at common step {t + 1}, replicate {k}")
    for name in names:
        r0 = runs[name][0]
        branches[name] = {
            "intervention_actions": validated[name], "joint_resolver": r0["joint"],
            "sources": (alternative_sources or {}).get(name),
            "intervention_transition": r0["records"][0],
            "horizon_records": r0["records"], "continuation": r0["continuation"],
            "outcome": r0["outcome"],
            "replay": "complete" if r0["sim"].terminated else "partial", "chain_hash": r0["sim"].chain_hash,
            "replicates": [{"replicate": k, "fork_seed_evaluator_only": seeds[k], "records": r["records"],
                            "outcome": r["outcome"], "chain_hash": r["sim"].chain_hash} for k, r in
                           enumerate(runs[name])],
            "runtime": {"continuation_cpu_ms": [r["cpu_ms"] for r in runs[name]]},
        }
    contrasts = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            per_rep = []
            for k in range(replicates):
                ca, cb = runs[a][k]["outcome"], runs[b][k]["outcome"]
                per_rep.append({
                    "replicate": k, "censored": ca["censored"] or cb["censored"],
                    "delta_casualties": cb["casualties"] - ca["casualties"],
                    "delta_unresolved": cb["unresolved"] - ca["unresolved"],
                    "delta_casualties_plus_unresolved": cb["casualties_plus_unresolved"]
                    - ca["casualties_plus_unresolved"],
                    "delta_evacuated": cb["evacuated"] - ca["evacuated"],
                    "delta_blocked_moves": cb["blocked_moves"] - ca["blocked_moves"],
                })
            ca, cb = branches[a]["outcome"], branches[b]["outcome"]
            dp = [r["delta_casualties_plus_unresolved"] for r in per_rep]
            verdict = classify(per_rep, stochastic=stochastic)
            contrasts.append({
                "a": a, "b": b, **verdict,
                "delta_b_minus_a_new_casualties": cb["new_casualties"] - ca["new_casualties"],
                "delta_b_minus_a_unresolved": cb["unresolved"] - ca["unresolved"],
                "delta_b_minus_a_new_evacuations": cb["new_evacuations"] - ca["new_evacuations"],
                "censored": any(r["censored"] for r in per_rep),
                "per_replicate": per_rep,
                "summary": {"replicates": replicates, "b_better": sum(x < 0 for x in dp),
                            "a_better": sum(x > 0 for x in dp), "ties": sum(x == 0 for x in dp),
                            "mean_delta_primary": sum(dp) / len(dp),
                            "sign_test_p_two_sided": sign_test_p(sum(x < 0 for x in dp), sum(x > 0 for x in dp))},
                "inference": "paired matched-fire replicates of one state under one frozen continuation; "
                             "uncertainty is over future fire only, not over states or populations",
                "preference_candidate": verdict["verdict"] in ("a_better", "b_better"),
                "training_preference_allowed": False,
            })
    body = {
        "counterfactual_version": CF_VERSION, "scope": SCOPE, "training_eligible": False,
        "scenario_hash": scenario.content_hash(), "common_observation_digest": digest(obs),
        "prefix_hash": prefix_hash, "prefix_steps": len(prefix_records),
        "prefix_source": dict(prefix_source) if prefix_source else None,
        "prefix_records": list(prefix_records),
        "people_active_at_intervention": len(common.active_ids()), "status_before": before,
        "horizon": horizon, "replicates": replicates, "fire_model": scenario.fire_model,
        "matched_exogenous_fire_prefix": True, "verdict_rule": VERDICT_RULE,
        "continuation": branches[names[0]]["continuation"],
        "branches": branches, "contrasts": contrasts, "causality_limit": CAUSAL_LIMIT,
    }
    return {**body, "report_hash": digest(mt.deterministic_view(body))}


def verify_counterfactual(report: Mapping[str, Any], scenario: Scenario) -> dict[str, Any]:
    """Re-execute every stored branch and replicate without running any policy; re-derive verdicts."""
    body = {k: v for k, v in report.items() if k != "report_hash"}
    if digest(mt.deterministic_view(body)) != report.get("report_hash"):
        raise CounterfactualError("report hash mismatch: content was altered")
    if report.get("scenario_hash") != scenario.content_hash() or report.get("training_eligible") is not False:
        raise CounterfactualError("report belongs to another scenario or claims eligibility")
    prefix = report["prefix_records"]
    if digest(list(prefix)) != report["prefix_hash"]:
        raise CounterfactualError("prefix hash mismatch")
    seeds = fork_seeds(scenario, report["prefix_hash"], report["replicates"])
    obs = _replay_prefix(scenario, prefix).observe()
    if digest(obs) != report["common_observation_digest"]:
        raise CounterfactualError("common observation does not regenerate")
    runs = {}
    for name, br in report["branches"].items():
        runs[name] = []
        for k, rep in enumerate(br["replicates"]):
            if rep["fork_seed_evaluator_only"] != seeds[k]:
                raise CounterfactualError(f"{name}: fork seed is not the preregistered derivation")
            if rep["records"][0]["actions"] != br["intervention_actions"]:
                raise CounterfactualError(f"{name}: replicate {k} did not apply the intervention")
            if replay_branch(scenario, prefix, seeds[k], rep["records"]) != rep["chain_hash"]:
                raise CounterfactualError(f"{name}: replicate {k} chain hash mismatch")
            runs[name].append(rep["outcome"])
    for c in report["contrasts"]:
        got = classify(c["per_replicate"], stochastic=report["fire_model"] != "deterministic")
        if got["verdict"] != c["verdict"]:
            raise CounterfactualError(f"{c['a']} vs {c['b']}: verdict does not follow the rule")
        for r in c["per_replicate"]:
            ca, cb = runs[c["a"]][r["replicate"]], runs[c["b"]][r["replicate"]]
            if r["delta_casualties_plus_unresolved"] != cb["casualties_plus_unresolved"] - \
                    ca["casualties_plus_unresolved"]:
                raise CounterfactualError("per-replicate delta does not match branch outcomes")
    return {"verified": True, "report_hash": report["report_hash"]}


# ----- alternatives from five same-state teacher proposals -----------------------------------------
def alternatives_from_probe(probe: Mapping[str, Any]) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    """Distinct complete joint actions of the five teachers at one state, named by proposer set."""
    groups: dict[str, list[str]] = {}
    acts: dict[str, dict[str, str]] = {}
    for name, prop in probe["proposals"].items():
        h = digest(prop["actions"])
        groups.setdefault(h, []).append(name)
        acts[h] = dict(prop["actions"])
    alts, sources = {}, {}
    for h, who in groups.items():
        label = "+".join(sorted(who))
        alts[label] = acts[h]
        sources[label] = {"kind": "teacher_joint_action", "proposed_by": sorted(who),
                          "proposal_hashes": sorted(probe["proposals"][n]["proposal_hash"] for n in who)}
    return alts, sources


def composite_alternative(observation: Mapping[str, Any], probe: Mapping[str, Any],
                          choice: Mapping[str, str]) -> tuple[dict[str, str], dict[str, Any]]:
    """Per-person mix of teacher actions. Refused unless the WHOLE joint map has no blocked move."""
    try:
        comb = proposals.combine(observation, {pid: probe["proposals"][t] for pid, t in choice.items()})
    except (KeyError, proposals.ProposalError) as exc:
        raise CounterfactualError(f"composite is not a complete valid map: {exc}") from None
    if not comb["joint"]["valid_joint_plan"]:
        raise CounterfactualError(f"composite joint map blocks {comb['joint']['blocked']}; not a coordinated plan")
    return comb["actions"], {"kind": "composite", "per_person_source": comb["sources"],
                             "source_proposals": comb["source_proposals"], "joint": comb["joint"]}


def shared_state_study(scenario: Scenario, fixture_report: Mapping[str, Any], teacher: str, step: int,
                       factories: Mapping[str, Callable[[], Any]], **kw: Any) -> dict[str, Any]:
    """Steps 1-10 on a verified P1-12 fixture: prefix of one teacher's trajectory, five-way probe,
    distinct teacher joint actions as interventions. No disagreement -> no contrast (recorded)."""
    mt.verify_fixture(fixture_report, scenario)
    decisions = fixture_report["policies"][teacher]["decisions"]
    if type(step) is not int or not 0 <= step < len(decisions):
        raise CounterfactualError("step is outside the source trajectory")
    prefix = [d["record"] for d in decisions[:step]]
    obs = decisions[step]["observation"]
    probe = mt.probe_shared_state(obs, factories)
    alts, sources = alternatives_from_probe(probe)
    src = {"kind": "p1-12-fixture", "content_hash": fixture_report["content_hash"], "teacher": teacher,
           "step": step, "probe_hash": probe["probe_hash"]}
    if len(alts) < 2:
        return {"counterfactual_version": CF_VERSION, "status": "no_disagreement", "prefix_source": src,
                "probe": probe, "training_eligible": False}
    report = counterfactual_fixture(scenario, prefix, alts, alternative_sources=sources, prefix_source=src, **kw)
    return {"counterfactual_version": CF_VERSION, "status": "contrasted", "probe": probe, "report": report,
            "training_eligible": False}
