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
# Real-scope studies (P1-17 Phase 7, preregistered batch): the verified TRAIN collection only, run to
# termination (horizon = the scenario's max_steps, at most REAL_MAX_HORIZON). DEV/EMBERBench never.
REAL_SCOPES = ("p1-17-train-full@1",)
REAL_MAX_HORIZON = 300
STEP0_SOURCE = "matched-group-step0"
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


def horizon_cap(scope: str) -> int:
    return REAL_MAX_HORIZON if scope in REAL_SCOPES else HARD_MAX_HORIZON


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


def _replay_branch_outcome(scenario: Scenario, prefix: Sequence[Mapping[str, Any]], fork_seed: int | None,
                           records: Sequence[Mapping[str, Any]]) -> tuple[Simulator, dict[str, Any]]:
    """Re-execute a stored branch WITHOUT any policy; re-derive its outcome exactly as `_run_branch` does."""
    sim = _replay_prefix(scenario, prefix)
    before = _statuses(sim)
    if fork_seed is not None:
        sim.fork_fire(fork_seed)
    if not records:
        raise CounterfactualError("branch has no intervention step")
    diags = []
    for t, r in enumerate(records):
        if sim.terminated:
            raise CounterfactualError(f"branch continues after termination at horizon step {t + 1}")
        obs = sim.observe()
        if sim.step(r["actions"]) != r:
            raise CounterfactualError(f"branch does not regenerate at horizon step {t + 1}")
        diags.append(mt.step_diagnostics(obs, r["actions"], r))
    if fork_seed is None:
        proof = verify_replay(scenario, None, [*prefix, *records], partial=not sim.terminated)
        if proof.status != ("complete" if sim.terminated else "partial"):
            raise CounterfactualError(f"branch replay failed: {proof.reason}")
    after = _statuses(sim)
    return sim, {**after, "casualties_plus_unresolved": after["casualties"] + after["unresolved"],
                 "new_casualties": after["casualties"] - before["casualties"],
                 "new_evacuations": after["evacuated"] - before["evacuated"],
                 "blocked_moves": sum(x["blocked_moves"] for x in diags),
                 "contested_targets": sum(x["contested_targets"] for x in diags),
                 "steps_run": len(records), "censored": not sim.terminated,
                 "termination_reason": sim.termination_reason}


def replay_branch(scenario: Scenario, prefix: Sequence[Mapping[str, Any]], fork_seed: int | None,
                  records: Sequence[Mapping[str, Any]]) -> str:
    """Re-execute a stored (possibly forked) branch; return its chain hash or raise."""
    return _replay_branch_outcome(scenario, prefix, fork_seed, records)[0].chain_hash


def _per_replicate(ca: Mapping[str, Any], cb: Mapping[str, Any], k: int) -> dict[str, Any]:
    return {"replicate": k, "censored": ca["censored"] or cb["censored"],
            "delta_casualties": cb["casualties"] - ca["casualties"],
            "delta_unresolved": cb["unresolved"] - ca["unresolved"],
            "delta_casualties_plus_unresolved": cb["casualties_plus_unresolved"] - ca["casualties_plus_unresolved"],
            "delta_evacuated": cb["evacuated"] - ca["evacuated"],
            "delta_blocked_moves": cb["blocked_moves"] - ca["blocked_moves"]}


def _summary(per_rep: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    dp = [r["delta_casualties_plus_unresolved"] for r in per_rep]
    return {"replicates": len(per_rep), "b_better": sum(x < 0 for x in dp), "a_better": sum(x > 0 for x in dp),
            "ties": sum(x == 0 for x in dp), "mean_delta_primary": sum(dp) / len(dp),
            "sign_test_p_two_sided": sign_test_p(sum(x < 0 for x in dp), sum(x > 0 for x in dp))}


def _check_matched_fire(runs: Mapping[str, Sequence[Sequence[Mapping[str, Any]]]], replicates: int) -> None:
    """Within each replicate, every alternative sees the same ignitions and fire draws at every common step."""
    names = sorted(runs)
    for k in range(replicates):
        for b in names[1:]:
            a_rec, b_rec = runs[names[0]][k], runs[b][k]
            for t in range(min(len(a_rec), len(b_rec))):
                if a_rec[t]["ignitions"] != b_rec[t]["ignitions"] or a_rec[t]["fire_draws"] != b_rec[t]["fire_draws"]:
                    raise CounterfactualError(f"future fire differs at common step {t + 1}, replicate {k}")


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
    if scope != SCOPE and scope not in REAL_SCOPES:
        raise CounterfactualError("real split counterfactuals require reviewed source authorization")
    if not isinstance(scenario, Scenario):
        raise CounterfactualError("scenario must be a validated Scenario")
    if type(horizon) is not int or type(max_horizon) is not int or \
            not 1 <= horizon <= max_horizon <= horizon_cap(scope):
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
    _check_matched_fire({n: [r["records"] for r in runs[n]] for n in names}, replicates)
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
            per_rep = [_per_replicate(runs[a][k]["outcome"], runs[b][k]["outcome"], k) for k in range(replicates)]
            ca, cb = branches[a]["outcome"], branches[b]["outcome"]
            verdict = classify(per_rep, stochastic=stochastic)
            contrasts.append({
                "a": a, "b": b, **verdict,
                "delta_b_minus_a_new_casualties": cb["new_casualties"] - ca["new_casualties"],
                "delta_b_minus_a_unresolved": cb["unresolved"] - ca["unresolved"],
                "delta_b_minus_a_new_evacuations": cb["new_evacuations"] - ca["new_evacuations"],
                "censored": any(r["censored"] for r in per_rep),
                "per_replicate": per_rep,
                "summary": _summary(per_rep),
                "inference": "paired matched-fire replicates of one state under one frozen continuation; "
                             "uncertainty is over future fire only, not over states or populations",
                "preference_candidate": verdict["verdict"] in ("a_better", "b_better"),
                "training_preference_allowed": False,
            })
    body = {
        "counterfactual_version": CF_VERSION, "scope": scope, "training_eligible": False,
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


def _check_continuation_record(ident: Any) -> None:
    if not isinstance(ident, Mapping):
        raise CounterfactualError("continuation identity missing")
    try:
        pinned = reg.identity(ident.get("name"), ident.get("version"))
    except (reg.RegistryError, TypeError):
        pinned = None
    key = {k: ident.get(k) for k in ("name", "policy", "version", "config_hash")}
    if pinned is None or pinned.status != reg.RETAINED or pinned.key() != key:
        raise CounterfactualError("continuation must be a frozen, retained, registry-pinned teacher")


def _rerun_continuation(scenario: Scenario, prefix: Sequence[Mapping[str, Any]], fork_seed: int | None,
                        records: Sequence[Mapping[str, Any]], factory: Callable[[], Any]) -> None:
    """Optional provenance check: the pinned continuation, re-run, chooses every stored post-intervention action."""
    sim = _replay_prefix(scenario, prefix)
    if fork_seed is not None:
        sim.fork_fire(fork_seed)
    sim.step(records[0]["actions"])
    policy = factory()
    for t, r in enumerate(records[1:], 2):
        obs = sim.observe()
        prop = proposals.validate_proposal(obs, proposals.proposal(policy.decide(copy.deepcopy(obs))))
        if prop["actions"] != r["actions"]:
            raise CounterfactualError(f"continuation does not reproduce the stored action at horizon step {t}")
        sim.step(r["actions"])


def verify_counterfactual(report: Mapping[str, Any], scenario: Scenario, *,
                          continuation_factory: Callable[[], Any] | None = None) -> dict[str, Any]:
    """Re-execute every stored branch and replicate without running any policy, then RE-DERIVE every
    outcome, per-replicate delta, censoring flag, summary and verdict from the regenerated simulators.
    Stored labels are never trusted: a re-hashed report with an edited guard delta, censoring flag or
    outcome fails. With `continuation_factory`, the pinned continuation is also re-run and must choose
    every stored post-intervention action (provenance of the continuation, not only replayability)."""
    body = {k: v for k, v in report.items() if k != "report_hash"}
    if digest(mt.deterministic_view(body)) != report.get("report_hash"):
        raise CounterfactualError("report hash mismatch: content was altered")
    if report.get("scenario_hash") != scenario.content_hash() or report.get("training_eligible") is not False:
        raise CounterfactualError("report belongs to another scenario or claims eligibility")
    if report.get("scope") not in (SCOPE, *REAL_SCOPES) or report.get("counterfactual_version") != CF_VERSION:
        raise CounterfactualError("unsupported counterfactual scope or version")
    if report.get("fire_model") != scenario.fire_model or report.get("verdict_rule") != VERDICT_RULE:
        raise CounterfactualError("fire model or verdict rule differs from the preregistered rule")
    stochastic = scenario.fire_model != "deterministic"
    n_rep, horizon = report.get("replicates"), report.get("horizon")
    if type(n_rep) is not int or not 1 <= n_rep <= MAX_REPLICATES or (not stochastic and n_rep != 1):
        raise CounterfactualError("invalid replicate count")
    if type(horizon) is not int or not 1 <= horizon <= horizon_cap(report["scope"]):
        raise CounterfactualError("invalid horizon")
    prefix = report["prefix_records"]
    if digest(list(prefix)) != report["prefix_hash"] or len(prefix) != report.get("prefix_steps"):
        raise CounterfactualError("prefix hash mismatch")
    seeds = fork_seeds(scenario, report["prefix_hash"], n_rep)
    common = _replay_prefix(scenario, prefix)
    obs = common.observe()
    if digest(obs) != report["common_observation_digest"]:
        raise CounterfactualError("common observation does not regenerate")
    if report.get("status_before") != _statuses(common) or \
            report.get("people_active_at_intervention") != len(common.active_ids()):
        raise CounterfactualError("status before the intervention does not regenerate")
    _check_continuation_record(report.get("continuation"))
    branches = report["branches"]
    if not isinstance(branches, Mapping) or not 2 <= len(branches) <= 5:
        raise CounterfactualError("requires two to five joint-action alternatives")
    if len({digest(br["intervention_actions"]) for br in branches.values()}) != len(branches):
        raise CounterfactualError("identical alternatives cannot make a causal contrast")
    outcomes: dict[str, list[dict[str, Any]]] = {}
    records: dict[str, list[Sequence[Mapping[str, Any]]]] = {}
    for name, br in branches.items():
        try:
            if validate_actions(obs, dict(br["intervention_actions"])) != br["intervention_actions"]:
                raise ValueError("not in canonical validated form")
        except ValueError as exc:
            raise CounterfactualError(f"{name}: intervention is not a complete legal joint map: {exc}") from None
        if br.get("continuation") != report["continuation"]:
            raise CounterfactualError(f"{name}: continuation differs from the report continuation")
        reps = br["replicates"]
        if len(reps) != n_rep or [r.get("replicate") for r in reps] != list(range(n_rep)):
            raise CounterfactualError(f"{name}: replicate set is not 0..{n_rep - 1}")
        outcomes[name], records[name] = [], []
        for k, rep in enumerate(reps):
            if rep["fork_seed_evaluator_only"] != seeds[k]:
                raise CounterfactualError(f"{name}: fork seed is not the preregistered derivation")
            if not rep["records"] or rep["records"][0]["actions"] != br["intervention_actions"]:
                raise CounterfactualError(f"{name}: replicate {k} did not apply the intervention")
            if len(rep["records"]) > horizon:
                raise CounterfactualError(f"{name}: replicate {k} runs past the horizon")
            sim, out = _replay_branch_outcome(scenario, prefix, seeds[k], rep["records"])
            if sim.chain_hash != rep["chain_hash"]:
                raise CounterfactualError(f"{name}: replicate {k} chain hash mismatch")
            if not sim.terminated and len(rep["records"]) != horizon:
                raise CounterfactualError(f"{name}: replicate {k} stops before the horizon without termination")
            if rep["outcome"] != out:
                raise CounterfactualError(f"{name}: replicate {k} outcome does not regenerate")
            if continuation_factory is not None:
                _rerun_continuation(scenario, prefix, seeds[k], rep["records"], continuation_factory)
            outcomes[name].append(out)
            records[name].append(rep["records"])
        r0 = reps[0]
        if br["outcome"] != r0["outcome"] or br["horizon_records"] != r0["records"] or \
                br["intervention_transition"] != r0["records"][0] or br["chain_hash"] != r0["chain_hash"] or \
                br["replay"] != ("complete" if not r0["outcome"]["censored"] else "partial"):
            raise CounterfactualError(f"{name}: branch summary differs from replicate 0 (the factual stream)")
    _check_matched_fire(records, n_rep)
    names = sorted(branches)
    want_pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    if [(c.get("a"), c.get("b")) for c in report["contrasts"]] != want_pairs:
        raise CounterfactualError("contrasts must cover every pair of alternatives exactly once, in order")
    for c in report["contrasts"]:
        a, b = c["a"], c["b"]
        per_rep = [_per_replicate(outcomes[a][k], outcomes[b][k], k) for k in range(n_rep)]
        if c["per_replicate"] != per_rep:
            raise CounterfactualError(f"{a} vs {b}: per-replicate delta does not match branch outcomes")
        got = classify(per_rep, stochastic=stochastic)
        if got["verdict"] != c["verdict"] or got["reason"] != c.get("reason"):
            raise CounterfactualError(f"{a} vs {b}: verdict does not follow the rule")
        ca, cb = outcomes[a][0], outcomes[b][0]
        if (c.get("delta_b_minus_a_new_casualties"), c.get("delta_b_minus_a_unresolved"),
                c.get("delta_b_minus_a_new_evacuations")) != (cb["new_casualties"] - ca["new_casualties"],
                                                               cb["unresolved"] - ca["unresolved"],
                                                               cb["new_evacuations"] - ca["new_evacuations"]):
            raise CounterfactualError(f"{a} vs {b}: factual-stream deltas do not match branch outcomes")
        if c.get("censored") != any(r["censored"] for r in per_rep) or c.get("summary") != _summary(per_rep):
            raise CounterfactualError(f"{a} vs {b}: censoring or summary does not match the replicates")
        if c.get("preference_candidate") != (got["verdict"] in ("a_better", "b_better")) or \
                c.get("training_preference_allowed") is not False:
            raise CounterfactualError(f"{a} vs {b}: unsupported preference label")
    return {"verified": True, "report_hash": report["report_hash"],
            "continuation_reproduced": continuation_factory is not None}


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
    if report["common_observation_digest"] != probe["observation_digest"]:
        raise CounterfactualError("probe observation is not the replayed common state")
    return {"counterfactual_version": CF_VERSION, "status": "contrasted", "probe": probe, "report": report,
            "training_eligible": False}


def stored_step0_probe(group: Mapping[str, Any]) -> dict[str, Any]:
    """The five teachers' STORED step-0 proposals of a verified matched group, as a probe.

    Step 0 is the matched initial state: every teacher decided on the identical observation, so the stored
    proposals are the same-state alternatives without any live teacher call (no T2 budget-dependent
    re-decision). Same fields as `multiteacher.probe_shared_state`, with the group's scope and source."""
    props, digests = {}, set()
    for name in mt.CANDIDATES:
        d = group["policies"][name]["decisions"][0]
        props[name] = d["proposal"]
        digests.add(d["observation_digest"])
    if len(digests) != 1:
        raise CounterfactualError("step-0 observations differ: not a matched initial state")
    obs = group["policies"][mt.CANDIDATES[0]]["decisions"][0]["observation"]
    for p in props.values():
        proposals.validate_proposal(obs, p)
    delta = proposals.disagreement(obs, list(props.values()))
    distinct = Counter(digest(p["actions"]) for p in props.values())
    body = {"schema_version": mt.SCHEMA_VERSION, "scope": group["scope"], "source": STEP0_SOURCE,
            "group_content_hash": group["content_hash"], "training_eligible": False,
            "observation_digest": digest(obs), "proposals": props, "disagreement": delta,
            "distinct_joint_actions": len(distinct),
            "note": "Stored step-0 proposals on the matched initial state; agreement alone is not quality evidence."}
    return {**body, "probe_hash": digest(mt.deterministic_view(body))}


def step0_study(scenario: Scenario, group: Mapping[str, Any], *, family_id: str, lineage_id: str,
                **kw: Any) -> dict[str, Any]:
    """Real-scope study at the matched initial state of a verified group (P1-17 Phase 7): the distinct stored
    step-0 teacher joint maps are the interventions; empty prefix; continuation and replicates from `kw`."""
    mt.verify_fixture(group, scenario)
    if group.get("scope") not in REAL_SCOPES or not group.get("group_complete"):
        raise CounterfactualError("step-0 studies need a complete group of an authorized real scope")
    probe = stored_step0_probe(group)
    alts, sources = alternatives_from_probe(probe)
    src = {"kind": STEP0_SOURCE, "content_hash": group["content_hash"], "teacher": None, "step": 0,
           "probe_hash": probe["probe_hash"], "family_id": family_id, "lineage_id": lineage_id}
    if len(alts) < 2:
        return {"counterfactual_version": CF_VERSION, "status": "no_disagreement", "prefix_source": src,
                "probe": probe, "training_eligible": False}
    report = counterfactual_fixture(scenario, [], alts, scope=group["scope"], alternative_sources=sources,
                                    prefix_source=src, **kw)
    if report["common_observation_digest"] != probe["observation_digest"]:
        raise CounterfactualError("stored step-0 observation is not the scenario's initial state")
    return {"counterfactual_version": CF_VERSION, "status": "contrasted", "probe": probe, "report": report,
            "training_eligible": False}


def _verify_step0_study(study: Mapping[str, Any], scenario: Scenario, group: Mapping[str, Any],
                        continuation_factory: Callable[[], Any] | None) -> dict[str, Any]:
    probe = study["probe"]
    if probe != stored_step0_probe(group):
        raise CounterfactualError("probe is not the group's stored step-0 proposals")
    src = study.get("prefix_source") if study.get("status") == "no_disagreement" else study["report"]["prefix_source"]
    if src.get("content_hash") != group["content_hash"] or src.get("probe_hash") != probe["probe_hash"] or \
            src.get("step") != 0:
        raise CounterfactualError("study does not bind the source group and probe")
    alts, sources = alternatives_from_probe(probe)
    if study["status"] == "no_disagreement":
        if len(alts) >= 2:
            raise CounterfactualError("the probe disagrees; 'no_disagreement' is not supported")
        return {"verified": True, "status": "no_disagreement", "probe_hash": probe["probe_hash"]}
    if study["status"] != "contrasted":
        raise CounterfactualError(f"unknown study status {study.get('status')!r}")
    rep = study["report"]
    if rep["prefix_records"] != [] or rep["scope"] != group["scope"]:
        raise CounterfactualError("a step-0 study has an empty prefix and the group's scope")
    got = {n: (b["intervention_actions"], b["sources"]) for n, b in rep["branches"].items()}
    if got != {n: (alts[n], sources[n]) for n in alts}:
        raise CounterfactualError("alternatives are not exactly the stored step-0 teacher joint actions")
    out = verify_counterfactual(rep, scenario, continuation_factory=continuation_factory)
    return {**out, "status": "contrasted", "probe_hash": probe["probe_hash"], "teachers_reasked": False}


def verify_shared_state_study(study: Mapping[str, Any], scenario: Scenario, fixture_report: Mapping[str, Any], *,
                              factories: Mapping[str, Callable[[], Any]] | None = None,
                              continuation_factory: Callable[[], Any] | None = None) -> dict[str, Any]:
    """Complete shared-state verification of a `shared_state_study` result, without trusting stored labels.

    Binds: the verified source fixture; the prefix = the source teacher's first `step` records; the probe
    (re-hashed) was taken on exactly the replayed common observation; the alternatives are exactly the
    distinct teacher joint actions of that probe with their proposer sets; then `verify_counterfactual`.
    With `factories`, all five teachers are re-asked on the common observation and must give the stored
    proposals (Markov teachers; fixture data only)."""
    mt.verify_fixture(fixture_report, scenario)
    if study.get("training_eligible") is not False or study.get("counterfactual_version") != CF_VERSION:
        raise CounterfactualError("study claims eligibility or has another version")
    probe = study["probe"]
    pbody = {k: v for k, v in probe.items() if k != "probe_hash"}
    if digest(mt.deterministic_view(pbody)) != probe.get("probe_hash") or probe.get("training_eligible") is not False:
        raise CounterfactualError("probe hash mismatch: content was altered")
    if set(probe["proposals"]) != set(mt.CANDIDATES):
        raise CounterfactualError("the probe must hold one proposal from each of the five teachers")
    src = study.get("prefix_source") if study.get("status") == "no_disagreement" else study["report"]["prefix_source"]
    if src.get("kind") == STEP0_SOURCE:
        if factories is not None:
            raise CounterfactualError("step-0 studies use stored proposals; teachers are not re-asked")
        return _verify_step0_study(study, scenario, fixture_report, continuation_factory)
    teacher, step = src["teacher"], src["step"]
    if src.get("kind") != "p1-12-fixture" or src.get("content_hash") != fixture_report["content_hash"] or \
            src.get("probe_hash") != probe["probe_hash"]:
        raise CounterfactualError("study does not bind the source fixture and probe")
    decisions = fixture_report["policies"][teacher]["decisions"]
    if type(step) is not int or not 0 <= step < len(decisions):
        raise CounterfactualError("step is outside the source trajectory")
    common = _replay_prefix(scenario, [d["record"] for d in decisions[:step]]).observe()
    if probe["observation_digest"] != digest(common) or decisions[step]["observation_digest"] != digest(common):
        raise CounterfactualError("probe observation is not the replayed common state")
    for name, prop in probe["proposals"].items():
        proposals.validate_proposal(common, prop)
    if factories is not None:
        again = mt.probe_shared_state(common, factories)
        if again["probe_hash"] != probe["probe_hash"]:
            raise CounterfactualError("teachers re-asked on the common state give other proposals")
    alts, sources = alternatives_from_probe(probe)
    if study["status"] == "no_disagreement":
        if len(alts) >= 2:
            raise CounterfactualError("the probe disagrees; 'no_disagreement' is not supported")
        return {"verified": True, "status": "no_disagreement", "probe_hash": probe["probe_hash"]}
    if study["status"] != "contrasted":
        raise CounterfactualError(f"unknown study status {study.get('status')!r}")
    rep = study["report"]
    if [d["record"] for d in decisions[:step]] != rep["prefix_records"]:
        raise CounterfactualError("prefix is not the source teacher's trajectory prefix")
    got = {n: (b["intervention_actions"], b["sources"]) for n, b in rep["branches"].items()}
    if got != {n: (alts[n], sources[n]) for n in alts}:
        raise CounterfactualError("alternatives are not exactly the probe's distinct teacher joint actions")
    out = verify_counterfactual(rep, scenario, continuation_factory=continuation_factory)
    return {**out, "status": "contrasted", "probe_hash": probe["probe_hash"],
            "teachers_reasked": factories is not None}
