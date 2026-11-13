"""Student joint-action probability model, exact log-likelihoods and training objectives (P1-14, CPU reference).

No torch, no model. A `scorer` stands in for the Clef head: it maps one row to raw logits. The GPU trainer
must compute exactly these quantities from the head's logits; the tests here are the reference values.

Audited Clef head (unsloth 2026.10.11 wheel, `_vendor/clef/joint_schema_model.py`, sha256 `0e304cf7…dc3a3`):

* One row = one token sequence: system prompt, STATE (+ any reservation block), the schema of the row's
  questions. `JointSchemaHead.forward` returns, per question, one UNNORMALIZED logit per option of that
  question (`option_ids` sorted). The upstream probability is `softmax` over that question's options.
* Question fields cross-attend only to the token memory. No answer of the row is an input. Given the row's
  token sequence, the answers of the row are conditionally independent. The logits of a person DO depend on
  the whole sequence: the state, the other questions of the row and their order (causal backbone).
* Every active person has exactly the 5 contract options `ACTIONS`. A move into a wall, the boundary or fire
  is an allowed contract action that the simulator resolves as blocked, so the distribution is over the 5
  contract actions, not over "useful" moves.

Hence, for a decision split into rows r_1..r_m (chunks C_1..C_m) and a joint map A:

    log pi(A | s) = sum_j sum_{i in C_j} log softmax(z_i(x_j(s, A_{C<j})))[a_i]

where x_j is the exact row input. Without reservations x_j = x_j(s) does not depend on A. With reservations
x_j depends on the actions of earlier chunks, so the SAME person with the SAME action can have a different
term under two joint maps (no cancellation). The chain rule makes the sum exact in both cases, and the
probabilities of all 5^N joint maps sum to 1 (tested by enumeration).

`pi` is the RAW model policy. A deterministic shield or corrector applied after argmax induces another
distribution: P_shield(B | s) = sum over raw maps A with shield(A) = B of pi(A | s). It is not computed by
any trainer; `induced_shield_probability` computes it only by exact enumeration for tiny N and refuses
otherwise. Preference and SFT losses are defined on raw joint maps only.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .contract import ACTIONS, validate_actions, validate_observation
from .encoding import encode_observation

ACTION_ORDER = tuple(sorted(ACTIONS))           # Clef option order (sorted option IDs)
OBJECTIVE_VERSION = "p1-14-student-objective@0.1.0"
Row = dict[str, Any]
Scorer = Callable[[Row], Mapping[str, Sequence[float]]]
ReservationFn = Callable[[Mapping[str, Any], Mapping[str, str]], str]


class ObjectiveError(ValueError):
    """An objective input breaks the probability model. Training must stop."""


# ----- rows ----------------------------------------------------------------------------------------------
def _partition(obs: Mapping[str, Any], chunks: Sequence[Sequence[str]]) -> list[list[str]]:
    active = sorted(p["id"] for p in obs["people"] if p["status"] == "active")
    flat = [i for c in chunks for i in c]
    if sorted(flat) != active or len(flat) != len(set(flat)) or any(not c for c in chunks):
        raise ObjectiveError("chunks must partition the active people into non-empty parts")
    return [list(c) for c in chunks]


def rows_for(obs: Mapping[str, Any], actions: Mapping[str, str], chunks: Sequence[Sequence[str]],
             reservation_fn: ReservationFn | None = None, *, encoding: str = "ascii") -> list[Row]:
    """The exact rows under which `actions` is scored: row j's input carries the reservation block of
    `actions` restricted to chunks < j (teacher forcing on the scored map itself, never on another map)."""
    obs = validate_observation(obs)
    actions = validate_actions(obs, dict(actions))
    parts = _partition(obs, chunks)
    state = encode_observation(obs, mode=encoding)
    rows, decided = [], {}
    for j, part in enumerate(parts):
        extra = reservation_fn(obs, dict(decided)) if reservation_fn else ""
        rows.append({"index": j, "state": state + extra, "ids": part, "reserved_ids": sorted(decided),
                     "targets": {i: actions[i] for i in part}})
        decided.update({i: actions[i] for i in part})
    return rows


def _log_softmax(z: Sequence[float]) -> list[float]:
    if len(z) != len(ACTION_ORDER) or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in z):
        raise ObjectiveError(f"a question needs {len(ACTION_ORDER)} finite logits, got {list(z)[:6]}")
    m = max(z)
    lse = m + math.log(sum(math.exp(x - m) for x in z))
    return [x - lse for x in z]


def score_rows(scorer: Scorer, rows: Sequence[Row]) -> dict[str, list[float]]:
    """Per-person log-probabilities over ACTION_ORDER, row by row, exactly as the scorer conditions them."""
    out: dict[str, list[float]] = {}
    for r in rows:
        logits = scorer({k: r[k] for k in ("index", "state", "ids", "reserved_ids")})
        if set(logits) != set(r["ids"]):
            raise ObjectiveError(f"row {r['index']}: scorer keys differ from the row's people")
        for pid in r["ids"]:
            out[pid] = _log_softmax(logits[pid])
    return out


def joint_logprob(scorer: Scorer, obs: Mapping[str, Any], actions: Mapping[str, str],
                  chunks: Sequence[Sequence[str]], reservation_fn: ReservationFn | None = None) -> dict[str, Any]:
    """Exact log pi(A | s) under the inference conditioning (one forward per row of THIS map)."""
    rows = rows_for(obs, actions, chunks, reservation_fn)
    lp = score_rows(scorer, rows)
    per = {pid: lp[pid][ACTION_ORDER.index(a)] for r in rows for pid, a in r["targets"].items()}
    return {"logp": sum(per.values()), "per_person": per, "row_inputs": [r["state"] for r in rows]}


def cancelling_persons(obs: Mapping[str, Any], chosen: Mapping[str, str], rejected: Mapping[str, str],
                       chunks: Sequence[Sequence[str]], reservation_fn: ReservationFn | None = None) -> set[str]:
    """Persons whose log-probability term is provably identical in both maps: same action AND an identical row
    input. Without reservations, every same-action person cancels. With reservations, a same-action person in a
    row after the first differing chunk does NOT cancel."""
    rw = rows_for(obs, chosen, chunks, reservation_fn)
    rl = rows_for(obs, rejected, chunks, reservation_fn)
    return {pid for a, b in zip(rw, rl) if a["state"] == b["state"] for pid in a["ids"] if chosen[pid] == rejected[pid]}


# ----- losses --------------------------------------------------------------------------------------------
def _softplus(x: float) -> float:
    return max(x, 0.0) + math.log1p(math.exp(-abs(x)))


def sft_nll(scorer: Scorer, obs: Mapping[str, Any], actions: Mapping[str, str], chunks: Sequence[Sequence[str]],
            reservation_fn: ReservationFn | None = None, *, weight: float = 1.0) -> dict[str, Any]:
    """Weighted SFT: weight * (-log pi(A|s)) / N (per-person mean, as the harness cross-entropy)."""
    jl = joint_logprob(scorer, obs, actions, chunks, reservation_fn)
    n = len(jl["per_person"])
    return {"loss": weight * -jl["logp"] / n, "joint_logp": jl["logp"], "people": n}


def set_nll(scorer: Scorer, obs: Mapping[str, Any], alternatives: Sequence[Mapping[str, str]],
            chunks: Sequence[Sequence[str]], reservation_fn: ReservationFn | None = None) -> dict[str, Any]:
    """Option B (set likelihood): -log sum_m pi(A_m | s) over several complete, distinct valid joint maps.
    Each alternative is scored as a whole map; no per-person label is mixed between alternatives."""
    from .sim.canonical import digest
    if len(alternatives) < 2 or len({digest(dict(a)) for a in alternatives}) != len(alternatives):
        raise ObjectiveError("set likelihood needs at least two distinct complete joint maps")
    lps = [joint_logprob(scorer, obs, a, chunks, reservation_fn)["logp"] for a in alternatives]
    m = max(lps)
    return {"loss": -(m + math.log(sum(math.exp(x - m) for x in lps))), "alternative_logps": lps}


def preference_margin(policy: Scorer, reference_logps: tuple[float, float], obs: Mapping[str, Any],
                      chosen: Mapping[str, str], rejected: Mapping[str, str], chunks: Sequence[Sequence[str]],
                      reservation_fn: ReservationFn | None = None) -> dict[str, Any]:
    """h = [log pi(w) - log ref(w)] - [log pi(l) - log ref(l)], all four EXACT joint log-probabilities under the
    same row conditioning. `reference_logps` = (log ref(w), log ref(l)) from the frozen Stage-A adapter, computed
    with the same rows (`joint_logprob`)."""
    if dict(chosen) == dict(rejected):
        raise ObjectiveError("chosen and rejected are identical")
    w = joint_logprob(policy, obs, chosen, chunks, reservation_fn)
    lo = joint_logprob(policy, obs, rejected, chunks, reservation_fn)
    rw, rl = reference_logps
    return {"margin": (w["logp"] - rw) - (lo["logp"] - rl), "logp_chosen": w["logp"], "logp_rejected": lo["logp"]}


def dpo_loss(margin: float, beta: float) -> float:
    """-log sigmoid(beta * h)."""
    if not beta > 0:
        raise ObjectiveError("beta must be positive")
    return _softplus(-beta * margin)


def ipo_loss(margin: float, tau: float) -> float:
    """(h - 1/(2 tau))^2 (Azar et al. IPO)."""
    if not tau > 0:
        raise ObjectiveError("tau must be positive")
    return (margin - 1.0 / (2.0 * tau)) ** 2


# ----- shield -------------------------------------------------------------------------------------------
def induced_shield_probability(scorer: Scorer, obs: Mapping[str, Any], target: Mapping[str, str],
                               chunks: Sequence[Sequence[str]], shield: Callable[[dict[str, str]], dict[str, str]],
                               reservation_fn: ReservationFn | None = None, *, max_maps: int = 3125) -> float:
    """P_shield(target | s) = sum over raw maps A with shield(A) == target of pi(A | s), by exact enumeration.
    Refused when 5^N > max_maps: the shielded distribution is then NOT known and must not be called exact."""
    import itertools
    ids = sorted(p["id"] for p in obs["people"] if p["status"] == "active")
    if len(ACTION_ORDER) ** len(ids) > max_maps:
        raise ObjectiveError(f"shielded probability needs {len(ACTION_ORDER)}^{len(ids)} raw maps; not computed")
    total = 0.0
    for combo in itertools.product(ACTION_ORDER, repeat=len(ids)):
        raw = dict(zip(ids, combo))
        if shield(dict(raw)) == dict(target):
            total += math.exp(joint_logprob(scorer, obs, raw, chunks, reservation_fn)["logp"])
    return total


# ----- preference admission -----------------------------------------------------------------------------
def admit_preference(record: Mapping[str, Any]) -> tuple[Mapping[str, str], Mapping[str, str]]:
    """Only an APPROVED class C record at one shared state enters a preference loss. Candidates, censored or
    inconclusive contrasts (class D) and shield/corrector outputs are refused."""
    if record.get("class") != "C":
        raise ObjectiveError(f"class {record.get('class')!r} never enters a preference loss")
    appr = record.get("approval") or {}
    if appr.get("status") != "approved":
        raise ObjectiveError(f"preference candidate not approved: {appr.get('status')!r}")
    ev = record["evidence"]
    if ev.get("censored") or ev.get("verdict") not in ("a_better", "b_better"):
        raise ObjectiveError("censored or inconclusive comparison")
    for side in ("chosen", "rejected"):
        src = record[side].get("sources") or {}
        if isinstance(src, Mapping) and src.get("kind") in ("shielded", "corrected"):
            raise ObjectiveError("a shielded or corrected map is not a raw policy action")
    return record["chosen"]["actions"], record["rejected"]["actions"]
