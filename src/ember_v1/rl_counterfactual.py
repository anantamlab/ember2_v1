"""Evaluator-private, same-state counterfactuals for fresh online TRAIN experience.

Policies receive only independent public observations. Teacher scores never define value.
This module deliberately does not import the historical preference datasets or benchmark.
"""
from __future__ import annotations

import copy
import math
import random
import statistics
import time
from collections import Counter

from .contract import validate_actions
from .rl_env import outcomes
from .sim import Scenario, Simulator
from .sim.canonical import digest
from .sim.rng import SplitMix64

VERSION = "online-counterfactual@1"


def snapshot(sim):
    """Complete JSON-safe evaluator snapshot; never supply this object to a policy."""
    body = {"version": VERSION, "scenario": sim._scenario.to_dict(), "seed": sim._seed,
            "state": copy.deepcopy(sim.private_state()), "chain": sim.chain_hash,
            "public_digest": digest(sim.observe())}
    return {**body, "snapshot_hash": digest(body)}


def restore(saved):
    body = {k: v for k, v in saved.items() if k != "snapshot_hash"}
    if saved["snapshot_hash"] != digest(body) or saved["version"] != VERSION:
        raise ValueError("counterfactual snapshot identity mismatch")
    sim = Simulator(Scenario.from_dict(saved["scenario"]), saved["seed"])
    state = saved["state"]
    sim._pos = {p["id"]: tuple(p["position"]) for p in state["people"]}
    sim._status = {p["id"]: p["status"] for p in state["people"]}
    sim._burning = frozenset(tuple(p) for p in state["burning"])
    sim._trapped = frozenset(state["trapped"])
    sim._rng = SplitMix64(state["rng"]["fire"])
    sim._step, sim._terminated = state["step"], state["terminated"]
    sim._reason, sim._chain = state["termination_reason"], saved["chain"]
    if sim.private_state() != state or sim._compute_trapped() != sim._trapped or digest(sim.observe()) != saved["public_digest"]:
        raise ValueError("counterfactual restore does not reproduce the saved state")
    return sim


def terminal_metrics(sim, records):
    obs = sim.observe()
    result = outcomes(obs)
    result.update(utility=(result["evacuated"]-2*result["casualties"]-.75*result["unresolved"])/result["people"],
                  steps_run=len(records), restricted_completion_time=obs["step"],
                  completion_time=obs["step"] if not result["unresolved"] else None,
                  completion_censored=bool(result["unresolved"]))
    movements = [m for r in records for m in r["movements"].values()]
    result.update(blocked=sum(m["outcome"].startswith("blocked_") for m in movements),
                  conflicts=sum(m["outcome"] in {"blocked_conflict", "blocked_occupied", "blocked_cycle"} for m in movements))
    return result


def paired_estimate(values, seed=0, repeats=2000):
    """Empirical paired-fire uncertainty, conditional on this one state/continuation."""
    mean = statistics.mean(values)
    if len(values) == 1:
        return {"mean": mean, "se": 0.0, "lower_95": mean, "upper_95": mean, "replicates": 1}
    rng = random.Random(seed)
    boot = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(repeats))
    return {"mean": mean, "se": statistics.stdev(values)/math.sqrt(len(values)),
            "lower_95": boot[int(.025*repeats)], "upper_95": boot[int(.975*repeats)], "replicates": len(values)}


def estimates(branches):
    reference = branches["student"]
    result = {}
    for name, runs in branches.items():
        deltas = {metric: [b["outcome"][metric]-a["outcome"][metric] for a, b in zip(reference, runs)]
                  for metric in ("utility", "evacuated", "casualties", "unresolved", "restricted_completion_time")}
        metrics = {key: paired_estimate(vals, int(digest([name, key])[-16:], 16)) for key, vals in deltas.items()}
        eligible = name != "student" and metrics["utility"]["lower_95"] > 0 and max(deltas["casualties"]) <= 0
        result[name] = {"paired": metrics, "per_replicate_deltas": deltas,
                        "training_eligible": eligible,
                        "limitation": "Empirical future-fire uncertainty for one state under the named continuation; exploratory candidate selection, not population significance."}
    return result


def compare(saved, candidates, *, continuation, continuation_identity, fork_seeds, source):
    if source.get("split") != "train":
        raise ValueError("counterfactual learning accepts TRAIN visited states only")
    common = restore(saved)
    if common.terminated or "student" not in candidates or not 2 <= len(candidates) <= 12:
        raise ValueError("counterfactual needs a live state and 2–12 full joint candidates")
    if not 1 <= len(fork_seeds) <= 16 or fork_seeds[0] is not None or len(set(fork_seeds)) != len(fork_seeds):
        raise ValueError("invalid matched future-fire realizations")
    if common._scenario.fire_model == "deterministic" and len(fork_seeds) != 1:
        raise ValueError("deterministic fire has one exact future")
    obs = common.observe()
    candidates = {n: validate_actions(obs, dict(a)) for n, a in candidates.items()}
    if len({digest(a) for a in candidates.values()}) != len(candidates):
        raise ValueError("deduplicate identical interventions before comparison")
    branches = {}
    for name, actions in candidates.items():
        runs = []
        for seed in fork_seeds:
            sim = restore(saved)  # a fresh restoration, never a previously executed branch
            if seed is not None:
                sim.fork_fire(seed)
            pre_state, pre_chain = digest(sim.private_state()), sim.chain_hash
            policy = copy.deepcopy(continuation)
            records = [sim.step(actions)]
            while not sim.terminated:
                public = copy.deepcopy(sim.observe())
                decision = policy(public)
                records.append(sim.step(decision.actions if hasattr(decision, "actions") else decision))
            runs.append({"fork_seed_evaluator_only": seed, "pre_state_digest": pre_state,
                         "pre_chain": pre_chain, "records": records, "outcome": terminal_metrics(sim, records),
                         "final_chain": sim.chain_hash})
        branches[name] = runs
    body = {"version": VERSION, "source": copy.deepcopy(source), "snapshot_evaluator_only": saved,
            "public_observation": obs, "common_observation_digest": digest(obs),
            "candidates": candidates, "continuation": copy.deepcopy(continuation_identity),
            "fork_seeds_evaluator_only": fork_seeds, "branches": branches,
            "estimates": estimates(branches), "matched_future_fire": True,
            "training_record_kind": "offline_counterfactual_joint_action", "on_policy_episode": False}
    study = {**body, "study_hash": digest(body)}
    verify_study(study, continuation=continuation)
    return study


def verify_study(study, *, continuation=None):
    if digest({k: v for k, v in study.items() if k != "study_hash"}) != study["study_hash"]:
        raise ValueError("counterfactual study hash mismatch")
    saved = study["snapshot_evaluator_only"]
    if restore(saved).observe() != study["public_observation"] or study["source"].get("split") != "train":
        raise ValueError("counterfactual common state/source mismatch")
    count = 0
    for name, runs in study["branches"].items():
        if len(runs) != len(study["fork_seeds_evaluator_only"]):
            raise ValueError("counterfactual replicate count mismatch")
        for i, run in enumerate(runs):
            seed = study["fork_seeds_evaluator_only"][i]
            sim = restore(saved)
            if seed is not None:
                sim.fork_fire(seed)
            if run["pre_state_digest"] != digest(sim.private_state()) or run["pre_chain"] != sim.chain_hash:
                raise ValueError("candidate did not start from the identical restored state")
            if run["records"][0]["actions"] != study["candidates"][name]:
                raise ValueError("intervention differs from named candidate")
            policy = copy.deepcopy(continuation) if continuation else None
            for step, row in enumerate(run["records"]):
                if step and policy is not None:
                    decision = policy(copy.deepcopy(sim.observe()))
                    expected = decision.actions if hasattr(decision, "actions") else decision
                    if row["actions"] != expected:
                        raise ValueError("branch does not use the identified fixed continuation")
                if sim.step(row["actions"]) != row:
                    raise ValueError("counterfactual branch replay differs")
                count += 1
            if not sim.terminated or sim.chain_hash != run["final_chain"] or terminal_metrics(sim, run["records"]) != run["outcome"]:
                raise ValueError("counterfactual terminal outcome differs")
            reference = study["branches"]["student"][i]
            for a, b in zip(reference["records"], run["records"]):
                if a["ignitions"] != b["ignitions"] or a["fire_draws"] != b["fire_draws"]:
                    raise ValueError("counterfactual future fire is not matched")
    if estimates(study["branches"]) != study["estimates"]:
        raise ValueError("counterfactual paired advantages differ")
    return {"status": "PASS", "branches": sum(map(len, study["branches"].values())), "transitions": count,
            "continuation_recomputed": continuation is not None}


def evaluate_job(job):
    """CPU worker: verified visited-state prefix, all teachers, and matched branches."""
    saved = job["snapshot"]
    factual = Simulator(Scenario.from_dict(saved["scenario"]), saved["seed"])
    for row in job["prefix_records"]:
        if factual.step(row["actions"]) != row:
            raise ValueError("counterfactual source prefix is not a factual on-policy replay")
    if snapshot(factual) != saved or factual.observe() != job["factual_visit"]["observation"]:
        raise ValueError("counterfactual is not the selected visited state")
    candidates, origins, diagnostics = candidate_actions(factual.observe(), job["student"], job["greedy"],
        seed=job["seed"], teacher_limits=job.get("teacher_limits"))
    study = compare(saved, candidates, continuation=fixed_t1,
                    continuation_identity=job["continuation_identity"], fork_seeds=job["fork_seeds"], source=job["source"])
    study.update(candidate_origins=origins, teacher_diagnostics=diagnostics,
                 prefix_records=job["prefix_records"], factual_visit=job["factual_visit"])
    study["study_hash"] = digest({k: v for k, v in study.items() if k != "study_hash"})
    return study


def fixed_t1(public):
    from .teacher.t1 import T1CongestionAware
    return T1CongestionAware().decide(public).actions


def candidate_actions(public, student, greedy, *, seed, teacher_limits=None):
    """All policy calls get public observations; teacher identities and limits are explicit."""
    from .teacher.registry import CANDIDATES, teacher_class
    from .provenance import code_fingerprint
    limits = teacher_limits or {"T2": {"cpu_budget_ms": 500.0, "max_rollouts": 24},
                                "T3": {"cpu_budget_ms": 200.0, "max_ant_steps": 20000}}
    candidates, origins, diagnostics = {}, {}, {}
    def add(name, actions, origin):
        actions = validate_actions(public, dict(actions))
        prior = next((n for n, a in candidates.items() if a == actions), None)
        if prior:
            origins[prior].append(origin)
        else:
            candidates[name], origins[name] = actions, [origin]
    add("student", student, {"name": "student", "type": "sampled_on_policy"})
    add("student-greedy", greedy, {"name": "student-greedy", "type": "current_policy"})
    for name in CANDIDATES:
        started = time.monotonic()
        try:
            cls = teacher_class(name)
            policy = cls(limits.get(name))
            decision = policy.decide(copy.deepcopy(public))
            add(name, decision.actions, {"name": name, "identity": policy.identity.to_dict(),
                                       "code_fingerprint": code_fingerprint(cls), "limits": limits.get(name, {})})
            diagnostics[name] = {"status": decision.diagnostics.get("status"), "seconds": time.monotonic()-started}
        except Exception as exc:
            diagnostics[name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                                 "seconds": time.monotonic()-started}
    rng = random.Random(seed)
    people = sorted(student)
    options = list(public["allowed_actions"])
    for i in range(4):
        actions = dict(student)
        for pid in rng.sample(people, min(2, len(people))):
            actions[pid] = rng.choice([a for a in options if a != actions[pid]])
        add(f"local-{i}", actions, {"name": f"local-{i}", "type": "two_person_variation", "seed": seed})
    return candidates, origins, diagnostics


def curriculum_stage(stage, history, episodes_in_stage):
    if stage >= 4 or episodes_in_stage < 16:
        return stage
    rows = history[-32:]
    n = sum(r["people"] for r in rows)
    if episodes_in_stage >= 32 or (sum(r["evacuated"] for r in rows)/n >= .55 and sum(r["casualties"] for r in rows)/n <= .30):
        return stage+1
    return stage
