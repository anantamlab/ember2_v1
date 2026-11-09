"""Exact joint-action likelihood under the Clef conditioning, DPO/IPO margins, set likelihood, shield refusal and
preference admission (P1-14 §1). Mock scorers only; CPU."""
import hashlib
import itertools
import math

import pytest

from ember_v1 import student_objective as so
from ember_v1.sim import Scenario, Simulator


def _obs(n=3):
    sc = Scenario.from_dict({
        "schema_version": "1.0", "scenario_id": "objective-fixture", "grid": {"width": 6, "height": 3},
        "obstacles": [], "exits": [{"id": "E0", "position": [1, 5]}],
        "people": [{"id": f"P{i:03d}", "position": [i % 3, i // 3]} for i in range(n)],
        "ignitions": [[2, 4]],
        "fire": {"model": "deterministic", "spread_interval": 9, "exits_flammable": False},
        "movement": {"conflict_resolution": "lowest_id_wins", "enter_fire": "blocked"},
        "termination": {"max_steps": 5}, "seed": 3})
    return Simulator(sc).observe()


def scorer(row):
    """Deterministic logits that depend on the whole row input (state + reservations + the row's people)."""
    out = {}
    for pid in row["ids"]:
        h = hashlib.sha256(f"{row['state']}|{','.join(row['ids'])}|{pid}".encode()).digest()
        out[pid] = [(b - 128) / 40 for b in h[:5]]
    return out


def ref_scorer(row):
    return {pid: [0.1 * j for j in range(5)] for pid in row["ids"]}


def reserve(obs, decided):
    return "" if not decided else "\nR:" + ";".join(f"{k}={v}" for k, v in sorted(decided.items()))


CHUNKS = [["P000", "P001"], ["P002"]]


@pytest.mark.parametrize("res", [None, reserve])
def test_joint_distribution_is_normalized_over_all_contract_maps(res):
    obs = _obs()
    ids = ["P000", "P001", "P002"]
    total = sum(math.exp(so.joint_logprob(scorer, obs, dict(zip(ids, c)), CHUNKS, res)["logp"])
                for c in itertools.product(so.ACTION_ORDER, repeat=3))
    assert total == pytest.approx(1.0, abs=1e-12)


def test_joint_logprob_is_the_sum_of_row_conditioned_terms():
    obs = _obs()
    a = {"P000": "east", "P001": "stay", "P002": "north"}
    jl = so.joint_logprob(scorer, obs, a, CHUNKS, reserve)
    assert jl["logp"] == pytest.approx(sum(jl["per_person"].values()))
    assert jl["row_inputs"][0] + "\nR:P000=east;P001=stay" == jl["row_inputs"][1]


def test_unreserved_same_actions_cancel_exactly():
    obs = _obs()
    w = {"P000": "east", "P001": "stay", "P002": "north"}
    lo = {"P000": "south", "P001": "stay", "P002": "north"}
    assert so.cancelling_persons(obs, w, lo, CHUNKS) == {"P001", "P002"}
    full = so.preference_margin(scorer, (0.0, 0.0), obs, w, lo, CHUNKS)["margin"]
    pw = so.joint_logprob(scorer, obs, w, CHUNKS)["per_person"]
    pl = so.joint_logprob(scorer, obs, lo, CHUNKS)["per_person"]
    assert full == pytest.approx(pw["P000"] - pl["P000"])


def test_reserved_same_actions_after_a_differing_chunk_do_not_cancel():
    obs = _obs()
    w = {"P000": "east", "P001": "stay", "P002": "north"}
    lo = {"P000": "south", "P001": "stay", "P002": "north"}
    assert so.cancelling_persons(obs, w, lo, CHUNKS, reserve) == {"P001"}
    exact = so.preference_margin(scorer, (0.0, 0.0), obs, w, lo, CHUNKS, reserve)["margin"]
    pw = so.joint_logprob(scorer, obs, w, CHUNKS, reserve)["per_person"]
    pl = so.joint_logprob(scorer, obs, lo, CHUNKS, reserve)["per_person"]
    naive = pw["P000"] - pl["P000"]                     # "differing persons only": wrong under reservations
    assert pw["P002"] != pl["P002"]
    assert exact == pytest.approx(naive + pw["P002"] - pl["P002"]) and exact != pytest.approx(naive)


def test_dpo_and_ipo_equal_hand_computed_values():
    obs = _obs()
    w = {"P000": "east", "P001": "stay", "P002": "north"}
    lo = {"P000": "south", "P001": "west", "P002": "north"}
    ref = (so.joint_logprob(ref_scorer, obs, w, CHUNKS, reserve)["logp"],
           so.joint_logprob(ref_scorer, obs, lo, CHUNKS, reserve)["logp"])
    m = so.preference_margin(scorer, ref, obs, w, lo, CHUNKS, reserve)
    h = (m["logp_chosen"] - ref[0]) - (m["logp_rejected"] - ref[1])
    assert m["margin"] == pytest.approx(h)
    assert so.dpo_loss(h, 0.1) == pytest.approx(-math.log(1 / (1 + math.exp(-0.1 * h))))
    assert so.ipo_loss(h, 0.5) == pytest.approx((h - 1.0) ** 2)
    with pytest.raises(so.ObjectiveError):
        so.dpo_loss(h, 0)
    with pytest.raises(so.ObjectiveError):
        so.preference_margin(scorer, ref, obs, w, w, CHUNKS)


def test_set_likelihood_scores_whole_maps_only():
    obs = _obs()
    a1 = {"P000": "east", "P001": "east", "P002": "east"}
    a2 = {"P000": "south", "P001": "north", "P002": "east"}
    s = so.set_nll(scorer, obs, [a1, a2], CHUNKS, reserve)
    p1, p2 = (math.exp(so.joint_logprob(scorer, obs, a, CHUNKS, reserve)["logp"]) for a in (a1, a2))
    assert s["loss"] == pytest.approx(-math.log(p1 + p2))
    composite = {"P000": "east", "P001": "north", "P002": "east"}           # a mix no alternative proposed
    assert math.exp(so.joint_logprob(scorer, obs, composite, CHUNKS, reserve)["logp"]) not in (p1, p2)
    with pytest.raises(so.ObjectiveError):
        so.set_nll(scorer, obs, [a1, dict(a1)], CHUNKS)


def test_sampled_whole_maps_need_person_level_conditioning_or_decode_to_a_mixture():
    """Why `sample_whole_maps` is not active for Stage A. Weighted SFT on three whole maps at one state:
    X=(north,north) 0.4, Y=(east,east) 0.3, Z=(east,west) 0.3. The SFT optimum of conditionally independent heads
    is the per-person marginals; its most likely joint map is (east,north), which no teacher proposed. With one
    person per reservation chunk the optimum is the exact mixture and its mode is the whole map X."""
    obs = _obs(2)
    maps = {("north", "north"): 0.4, ("east", "east"): 0.3, ("east", "west"): 0.3}
    ids = ["P000", "P001"]

    def logits(dist):
        return [math.log(dist.get(a, 0.0) + 1e-12) for a in so.ACTION_ORDER]

    def marg(i, cond=None):
        d = {}
        for m, w in maps.items():
            if cond is None or m[0] == cond:
                d[m[i]] = d.get(m[i], 0.0) + w
        z = sum(d.values())
        return {a: v / z for a, v in d.items()}

    def independent(row):
        return {pid: logits(marg(ids.index(pid))) for pid in row["ids"]}

    def chained(row):
        out = {}
        for pid in row["ids"]:
            if pid == "P000":
                out[pid] = logits(marg(0))
            else:
                prior = row["state"].split("R:P000=")[1]
                out[pid] = logits(marg(1, prior))
        return out

    def mode(sc, chunks, res=None):
        return max(itertools.product(so.ACTION_ORDER, repeat=2),
                   key=lambda c: so.joint_logprob(sc, obs, dict(zip(ids, c)), chunks, res)["logp"])

    assert mode(independent, [ids]) == ("east", "north") and ("east", "north") not in maps   # per-person mixture
    one_each = [["P000"], ["P001"]]
    assert mode(chained, one_each, reserve) == ("north", "north")                            # a whole teacher map
    for m, w in maps.items():
        p = math.exp(so.joint_logprob(chained, obs, dict(zip(ids, m)), one_each, reserve)["logp"])
        assert p == pytest.approx(w, abs=1e-9)


def test_invalid_scorer_outputs_fail_closed():
    obs = _obs()
    a = {"P000": "east", "P001": "stay", "P002": "north"}
    with pytest.raises(so.ObjectiveError):
        so.joint_logprob(lambda r: {p: [0.0] * 4 for p in r["ids"]}, obs, a, CHUNKS)
    with pytest.raises(so.ObjectiveError):
        so.joint_logprob(lambda r: {p: [0, 0, 0, 0, float("nan")] for p in r["ids"]}, obs, a, CHUNKS)
    with pytest.raises(so.ObjectiveError):
        so.joint_logprob(lambda r: {"P000": [0] * 5}, obs, a, CHUNKS)
    with pytest.raises(so.ObjectiveError):
        so.joint_logprob(scorer, obs, a, [["P000"], ["P002"]])


def test_shielded_probability_is_enumerated_or_refused():
    obs = _obs(2)
    chunks = [["P000"], ["P001"]]

    def shield(m):                                         # e.g. turn every "west" into "stay"
        return {k: ("stay" if v == "west" else v) for k, v in m.items()}
    tgt = {"P000": "stay", "P001": "east"}
    p = so.induced_shield_probability(scorer, obs, tgt, chunks, shield, reserve)
    raw = sum(math.exp(so.joint_logprob(scorer, obs, {"P000": a, "P001": "east"}, chunks, reserve)["logp"])
              for a in ("stay", "west"))
    assert p == pytest.approx(raw)
    assert p > math.exp(so.joint_logprob(scorer, obs, tgt, chunks, reserve)["logp"])
    with pytest.raises(so.ObjectiveError):
        so.induced_shield_probability(scorer, _obs(6), tgt, chunks, shield)


def test_only_approved_same_state_preferences_are_admitted():
    base = {"class": "C", "approval": {"status": "approved"}, "evidence": {"censored": False, "verdict": "a_better"},
            "chosen": {"actions": {"P000": "east"}, "sources": {"kind": "teacher_joint_action"}},
            "rejected": {"actions": {"P000": "stay"}, "sources": {"kind": "composite"}}}
    assert so.admit_preference(base) == ({"P000": "east"}, {"P000": "stay"})
    bad = [{**base, "class": "D"}, {**base, "approval": {"status": "candidate_not_approved"}},
           {**base, "evidence": {"censored": True, "verdict": "a_better"}},
           {**base, "evidence": {"censored": False, "verdict": "inconclusive"}},
           {**base, "chosen": {"actions": {"P000": "east"}, "sources": {"kind": "shielded"}}}]
    for r in bad:
        with pytest.raises(so.ObjectiveError):
            so.admit_preference(r)
