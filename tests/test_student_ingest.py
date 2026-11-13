"""P1-14 ingestion interface: validated records -> name-free examples -> seeded weighted epoch -> exact rows.

Synthetic fixtures only (the five-teacher test room and generated large-N observations). No TRAIN data.
"""
from __future__ import annotations

import copy
import json
from collections import Counter

import pytest

from ember_v1 import multiteacher_dataset as ds
from ember_v1 import student_ingest as si
from ember_v1 import student_objective as so
from ember_v1.sim import Scenario, Simulator
from test_multiteacher_dataset import _conflict, _reseal, built  # noqa: F401  (module fixture)


def _cost(e):
    return 10 + 3 * len(e["actions"])


def test_only_selected_weighted_class_a_records_become_examples(built):  # noqa: F811
    _, _, recs = built
    prep = si.prepare_examples(recs)
    ex = prep["examples"]
    want = [r for r in recs if r["class"] == "A" and r["selection"]["selected"] and r["weight"]]
    assert len(ex) == len(want) > 0 and prep["report"]["sft_examples"] == len(want)
    assert abs(prep["report"]["weight_total"] - sum(r["weight"] for r in want)) < 1e-9
    assert all(set(e) == {"observation", "actions", "weight", "example_id"} for e in ex)
    text = json.dumps(ex)
    for leaked in ("teachers", "provenance", "proposal_hash", "teacher_values", "selection", "evidence"):
        assert leaked not in text
    assert [e["example_id"] for e in ex] == sorted(e["example_id"] for e in ex)


def test_invalid_or_heldout_datasets_are_refused_whole(built):  # noqa: F811
    _, _, recs = built
    bad = copy.deepcopy(recs)
    bad[0]["weight"] = 5.0                      # tampered without reseal -> integrity failure
    with pytest.raises(si.IngestError, match="dataset refused"):
        si.prepare_examples(bad)
    fam = recs[0]["family_id"]
    with pytest.raises(si.IngestError, match="held-out"):
        si.prepare_examples(recs, heldout_families={fam})
    with pytest.raises(si.IngestError, match="held-out"):
        si.prepare_examples(recs, heldout_scenarios={recs[0]["scenario_hash"]})


def test_epoch_is_seeded_reproducible_and_within_budget(built):  # noqa: F811
    ex = si.prepare_examples(built[2])["examples"]
    a = si.sample_epoch(ex, seed="s1", token_budget=5000, cost=_cost)
    assert a == si.sample_epoch(ex, seed="s1", token_budget=5000, cost=_cost)
    assert a["draws"] != si.sample_epoch(ex, seed="s2", token_budget=5000, cost=_cost)["draws"]
    by = {e["example_id"]: e for e in ex}
    assert a["tokens_used"] == sum(_cost(by[d]) for d in a["draws"]) <= 5000
    assert a["stopped"] == "next_draw_exceeds_budget"
    # the draw sequence is a prefix property: a larger budget extends, never reorders, the epoch
    b = si.sample_epoch(ex, seed="s1", token_budget=9000, cost=_cost)
    assert b["draws"][:a["n_draws"]] == a["draws"]


def test_uniform_is_platform_independent():
    # pinned values: SHA-256 based, no dependency on random.Random or the Python version
    assert [si._uniform("pin", i) for i in range(3)] == [0.16703451224658605, 0.6828911397574569, 0.028977016894602525]
    assert 0.0 <= min(si._uniform("pin", i) for i in range(1000)) and max(si._uniform("pin", i) for i in range(1000)) < 1


def test_draw_frequencies_follow_weights():
    ex = [{"example_id": f"e{i}", "weight": w, "observation": {}, "actions": {}} for i, w in enumerate((1, 2, 7))]
    ep = si.sample_epoch(ex, seed="freq", token_budget=20000, cost=lambda e: 1)
    c = Counter(ep["draws"])
    assert ep["n_draws"] == 20000
    for i, w in enumerate((0.1, 0.2, 0.7)):
        assert abs(c[f"e{i}"] / 20000 - w) < 0.015


def test_epoch_request_errors_fail_closed():
    ex = [{"example_id": "e0", "weight": 1.0, "observation": {}, "actions": {}}]
    for kw, msg in (({"token_budget": 0}, "positive"), ({"seed": ""}, "seed"),
                    ({"token_budget": 5, "cost": lambda e: 9}, "exceeds"),
                    ({"cost": lambda e: 0}, "positive integer")):
        args = {"seed": "x", "token_budget": 100, "cost": lambda e: 1, **kw}
        with pytest.raises(si.IngestError, match=msg):
            si.sample_epoch(ex, **args)
    with pytest.raises(si.IngestError, match="no SFT"):
        si.sample_epoch([], seed="x", token_budget=1, cost=lambda e: 1)


def test_rows_apply_the_weight_once_and_match_the_objective(built):  # noqa: F811
    ex = si.prepare_examples(built[2])["examples"]
    ep = si.sample_epoch(ex, seed="rows", token_budget=800, cost=_cost)
    rows = si.epoch_rows(ex, ep, k=2)
    assert rows and all(r["loss_weight"] == 1.0 for r in rows)      # weight already used by sampling
    assert all(set(r) == si.ROW_KEYS for r in rows)
    by = {e["example_id"]: e for e in ex}
    for n, eid in enumerate(ep["draws"]):
        mine = [r for r in rows if r["draw"] == n]
        e = by[eid]
        assert mine == [{**r, "loss_weight": 1.0, "example_id": eid, "draw": n}
                        for r in so.rows_for(e["observation"], e["actions"], si.id_chunks(e["observation"], 2))]
        assert {i: a for r in mine for i, a in r["targets"].items()} == dict(e["actions"])
    tampered = {**ep, "draws": list(reversed(ep["draws"]))}
    if tampered["draws"] != ep["draws"]:
        with pytest.raises(si.IngestError, match="manifest hash"):
            si.epoch_rows(ex, tampered, k=2)


def test_teacher_names_change_nothing_downstream(built):  # noqa: F811
    a, b = _conflict(built)
    swap = {"T2": "T1", "T1": "T2"}
    renamed = []
    for x in (a, b):
        y = copy.deepcopy(x)
        y["teachers"] = [swap.get(t, t) for t in y["teachers"]]
        y["evidence"]["trajectory_outcomes"] = {swap.get(t, t): o for t, o in y["evidence"]["trajectory_outcomes"].items()}
        renamed.append(_reseal(y))

    def rows_of(recs):
        w = ds.assign_weights(ds.select_joint_targets(recs))
        ex = [{**ds.model_example(r), "example_id": si.example_id(ds.model_example(r))}
              for r in w if r["class"] == "A" and r["selection"]["selected"] and r["weight"]]
        ep = si.sample_epoch(ex, seed="names", token_budget=500, cost=_cost)
        return ep, si.epoch_rows(ex, ep, k=3)
    assert rows_of([a, b]) == rows_of(renamed)


def _large_obs(n):
    w = 40
    sc = Scenario.from_dict({
        "schema_version": "1.0", "scenario_id": f"ingest-large-{n}", "grid": {"width": w, "height": 12},
        "obstacles": [], "exits": [{"id": "E0", "position": [0, 0]}, {"id": "E1", "position": [11, w - 1]}],
        "people": [{"id": f"P{i:03d}", "position": [1 + i // (w - 2), 1 + i % (w - 2)]} for i in range(n)],
        "ignitions": [[11, 0]],
        "fire": {"model": "deterministic", "spread_interval": 9, "exits_flammable": False},
        "movement": {"conflict_resolution": "lowest_id_wins", "enter_fire": "blocked"},
        "termination": {"max_steps": 5}, "seed": 5})
    return Simulator(sc).observe()


@pytest.mark.parametrize("n,k", [(100, 40), (200, 40), (150, 64)])
def test_large_n_chunk_rows_partition_people_and_reserve_earlier_chunks(n, k):
    obs = _large_obs(n)
    acts = {p["id"]: so.ACTION_ORDER[i % 5] for i, p in enumerate(obs["people"]) if p["status"] == "active"}
    ex = [{"observation": obs, "actions": acts, "weight": 1.0}]
    ex[0]["example_id"] = si.example_id(ex[0])
    ep = si.sample_epoch(ex, seed="big", token_budget=1, cost=lambda e: 1)
    rows = si.epoch_rows(ex, ep, k=k, reservation_fn=lambda o, d: "" if not d else f"\nR:{len(d)}")
    assert len(rows) == -(-n // k)
    assert sorted(i for r in rows for i in r["ids"]) == sorted(acts)
    assert [len(r["reserved_ids"]) for r in rows] == [j * k for j in range(len(rows))]
    assert all(r["targets"] == {i: acts[i] for i in r["ids"]} for r in rows)
    assert rows[0]["state"] != rows[-1]["state"]          # reservation block of earlier chunks is in the input


# ----- run 12: exact chain rule at N>100, CPU mock training loop, model and head pins -----------------------
def _hash_scorer(row):
    """Deterministic toy logits that depend on the WHOLE row input (state incl. reservation block) and the person."""
    import hashlib
    out = {}
    for pid in row["ids"]:
        h = hashlib.sha256(f"{row['state']}|{pid}".encode()).digest()
        out[pid] = [(b - 128) / 64 for b in h[:5]]
    return out


def test_large_n_chain_rule_is_exact_and_covers_every_person():
    import math
    obs = _large_obs(120)
    active = sorted(p["id"] for p in obs["people"] if p["status"] == "active")
    assert len(active) == 120
    acts = {pid: so.ACTION_ORDER[i % 5] for i, pid in enumerate(active)}
    chunks = si.id_chunks(obs, 40)
    res = lambda o, d: "" if not d else "\nRES " + ",".join(f"{k}={v}" for k, v in sorted(d.items()))  # noqa: E731
    jl = so.joint_logprob(_hash_scorer, obs, acts, chunks, res)
    assert sorted(jl["per_person"]) == active                       # one term per active person, no duplicate
    manual, decided = 0.0, {}
    for part in chunks:
        row = {"index": 0, "state": so.encode_observation(obs, mode="ascii") + res(obs, decided), "ids": part,
               "reserved_ids": sorted(decided)}
        z = _hash_scorer(row)
        for pid in part:
            m = max(z[pid])
            manual += z[pid][so.ACTION_ORDER.index(acts[pid])] - (m + math.log(sum(math.exp(x - m) for x in z[pid])))
        decided.update({pid: acts[pid] for pid in part})
    assert jl["logp"] == pytest.approx(manual, abs=1e-9)
    # change one action in chunk 0: later chunks see another reservation block, so their terms do not cancel
    other = {**acts, active[0]: so.ACTION_ORDER[(so.ACTION_ORDER.index(acts[active[0]]) + 1) % 5]}
    keep = so.cancelling_persons(obs, acts, other, chunks, res)
    assert keep == set(active[1:40])                                # only same-row, same-action persons cancel
    lo = so.joint_logprob(_hash_scorer, obs, other, chunks, res)["per_person"]
    assert any(lo[p] != jl["per_person"][p] for p in active[40:])


def test_cpu_mock_training_loop_reduces_the_exact_objective(built):  # noqa: F811
    """A 5-parameter toy head (shared action bias), plain gradient descent on the exact SFT objective of the
    ingested epoch. Checks the data path end to end on CPU. Not the student; no Qwen, no GPU."""
    import math
    ex = si.prepare_examples(built[2])["examples"]
    ep = si.sample_epoch(ex, seed="mock-train", token_budget=600, cost=_cost)
    by = {e["example_id"]: e for e in ex}
    theta = [0.0] * 5

    def scorer(row):
        return {pid: list(theta) for pid in row["ids"]}

    def epoch_loss():
        return sum(so.sft_nll(scorer, by[e]["observation"], by[e]["actions"],
                              si.id_chunks(by[e]["observation"], 2))["loss"] for e in ep["draws"]) / len(ep["draws"])

    rows = si.epoch_rows(ex, ep, k=2)
    people = {n: sum(len(r["ids"]) for r in rows if r["draw"] == n) for n in range(len(ep["draws"]))}
    losses = [epoch_loss()]
    for _ in range(25):
        m = max(theta)
        p = [math.exp(t - m) for t in theta]
        p = [x / sum(p) for x in p]
        grad = [0.0] * 5
        for r in rows:
            for a in r["targets"].values():
                for j in range(5):
                    grad[j] += (p[j] - (j == so.ACTION_ORDER.index(a))) / people[r["draw"]] / len(ep["draws"])
        theta = [t - 0.5 * g for t, g in zip(theta, grad)]
        losses.append(epoch_loss())
    assert all(b <= a + 1e-12 for a, b in zip(losses, losses[1:])) and losses[-1] < losses[0]
    assert losses[0] == pytest.approx(math.log(5))                  # uniform head: -log(1/5) per person


def test_student_model_and_head_pins_are_preserved():
    from ember_v1 import p0b
    assert p0b.PINNED_MODEL == "unsloth/Qwen3.5-2B"
    sha = p0b.AUDITED_CLEF_SHA256
    assert len(sha) == 64 and f"`{sha[:8]}…{sha[-5:]}`" in so.__doc__
