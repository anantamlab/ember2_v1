"""Training-dataset ingestion interface (P1-14 -> student trainer). CPU reference; no model, no real data.

The ONLY path from multi-teacher dataset records to trainer rows:

1. `prepare_examples` refuses the whole dataset unless `validate_dataset` passes (integrity, coherence, source
   drift, family caps) and no record touches a held-out family or scenario (DEV, calibration, locked test,
   selection sets, P1-15 test). Only selected, weighted class-A records become SFT examples, through
   `model_example` (observation, whole action map, weight). Teacher names and provenance never pass.
2. `sample_epoch` draws examples with probability proportional to weight, with replacement, until a fixed token
   budget is used. Draws come from SHA-256 of (seed, draw index), so the epoch is identical on every platform and
   Python version. The manifest publishes the seed, the budget and every draw.
3. `epoch_rows` turns each draw into the exact rows of `student_objective.rows_for` (teacher forcing on the
   drawn map itself). **The record weight is applied once, by sampling: every row has `loss_weight` 1.** Using the
   weight again in the loss would square it.

Real use stays blocked until the P1-14 protocol is frozen and TRAIN data is authorized (`multiteacher_dataset
.export`). This module only defines and tests the interface on synthetic fixtures.
"""
from __future__ import annotations

import hashlib
from bisect import bisect_right
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from . import multiteacher_dataset as ds
from . import student_objective as so
from .sim.canonical import digest

INGEST_VERSION = "p1-14-ingest@0.1.0"
ROW_KEYS = frozenset({"index", "state", "ids", "reserved_ids", "targets", "loss_weight", "example_id", "draw"})
CostFn = Callable[[Mapping[str, Any]], int]


class IngestError(ValueError):
    """The dataset or the epoch request is not acceptable for training. Training must not start."""


# ----- examples ------------------------------------------------------------------------------------------------
def example_id(example: Mapping[str, Any]) -> str:
    """Name-free identity of an SFT example: digest of the observation and the whole target map only."""
    return digest({"observation": example["observation"], "actions": dict(example["actions"])})


def prepare_examples(records: Iterable[Mapping[str, Any]], protocol: Mapping[str, Any] | None = None, *,
                     heldout_families: Iterable[str] = (), heldout_scenarios: Iterable[str] = ()
                     ) -> dict[str, Any]:
    """Validated SFT examples plus an ingestion report. Any problem refuses the whole dataset (no partial use)."""
    protocol = protocol or ds.load_protocol()
    records = list(records)
    v = ds.validate_dataset(records, protocol)
    if not v["ok"]:
        raise IngestError("dataset refused: " + "; ".join(v["problems"][:5]))
    fam, scen = set(heldout_families), set(heldout_scenarios)
    leaks = sorted({r["family_id"] for r in records if r["family_id"] in fam}
                   | {r["scenario_hash"] for r in records if r["scenario_hash"] in scen})
    if leaks:
        raise IngestError(f"held-out families or scenarios in the dataset: {leaks[:5]}")
    examples, classes = [], Counter()
    for r in records:
        sft = r["class"] == "A" and (r.get("selection") or {}).get("selected") and r.get("weight")
        classes[(r["class"], "sft" if sft else "not_sft")] += 1
        if sft:
            ex = ds.model_example(r)
            if not (isinstance(ex["weight"], (int, float)) and ex["weight"] > 0):
                raise IngestError("an SFT example needs a positive numeric weight")
            examples.append({**ex, "example_id": example_id(ex)})
    ids = Counter(e["example_id"] for e in examples)
    if any(n > 1 for n in ids.values()):
        raise IngestError("two weighted SFT examples share one observation and map (validator gap)")
    examples.sort(key=lambda e: e["example_id"])
    return {"version": INGEST_VERSION, "examples": examples,
            "report": {"records": len(records), "sft_examples": len(examples),
                       "classes": {f"{k}:{u}": n for (k, u), n in sorted(classes.items())},
                       "weight_total": round(sum(e["weight"] for e in examples), 12),
                       "review": v.get("review", []), "dataset_digest": digest(sorted(r["record_hash"]
                                                                                       for r in records))}}


# ----- epoch sampling ------------------------------------------------------------------------------------------
def _uniform(seed: str, i: int) -> float:
    """Platform-independent uniform in [0, 1): 53 bits of SHA-256(seed, i)."""
    h = hashlib.sha256(f"{INGEST_VERSION}|{seed}|{i}".encode()).digest()
    return (int.from_bytes(h[:8], "big") >> 11) / float(1 << 53)


def sample_epoch(examples: Sequence[Mapping[str, Any]], *, seed: str, token_budget: int, cost: CostFn,
                 max_draws: int = 10 ** 7) -> dict[str, Any]:
    """Draws ∝ weight, with replacement, in a fixed order, until the next draw does not fit the token budget.

    Stopping at the first draw that does not fit keeps the draw sequence independent of example size (skipping
    a large draw and continuing would favour small examples). The manifest is the published epoch."""
    if type(token_budget) is not int or token_budget < 1:
        raise IngestError("token_budget must be a positive integer")
    if not isinstance(seed, str) or not seed:
        raise IngestError("seed must be a non-empty published string")
    if not examples:
        raise IngestError("no SFT examples")
    ex = sorted(examples, key=lambda e: e["example_id"])
    costs = []
    for e in ex:
        c = cost(e)
        if type(c) is not int or c < 1:
            raise IngestError(f"cost of {e['example_id'][:19]} must be a positive integer")
        if c > token_budget:
            raise IngestError(f"example {e['example_id'][:19]} alone ({c}) exceeds the token budget")
        costs.append(c)
    cum, total = [], 0.0
    for e in ex:
        total += float(e["weight"])
        cum.append(total)
    draws, used, i = [], 0, 0
    while i < max_draws:
        j = min(bisect_right(cum, _uniform(seed, i) * total), len(ex) - 1)
        if used + costs[j] > token_budget:
            break
        draws.append(ex[j]["example_id"])
        used += costs[j]
        i += 1
    counts = Counter(draws)
    return {"version": INGEST_VERSION, "seed": seed, "token_budget": token_budget, "tokens_used": used,
            "draws": draws, "n_draws": len(draws), "distinct": len(counts),
            "examples": len(ex), "weight_total": round(total, 12),
            "stopped": "next_draw_exceeds_budget" if i < max_draws else "max_draws",
            "manifest_hash": digest({"seed": seed, "token_budget": token_budget, "draws": draws})}


# ----- rows ----------------------------------------------------------------------------------------------------
def id_chunks(observation: Mapping[str, Any], k: int) -> list[list[str]]:
    """Active people in ID order, at most k per chunk (the order `token_budget.chunk_rows` uses)."""
    if type(k) is not int or k < 1:
        raise IngestError("k must be a positive integer")
    ids = sorted(p["id"] for p in observation["people"] if p["status"] == "active")
    return [ids[s:s + k] for s in range(0, len(ids), k)]


def epoch_rows(examples: Sequence[Mapping[str, Any]], epoch: Mapping[str, Any], *, k: int,
               reservation_fn: so.ReservationFn | None = None, encoding: str = "ascii") -> list[dict[str, Any]]:
    """Trainer rows for every draw, in draw order. Rows carry no teacher, provenance or weight field."""
    by_id = {e["example_id"]: e for e in examples}
    missing = sorted(set(epoch["draws"]) - set(by_id))
    if missing:
        raise IngestError(f"epoch names unknown examples: {missing[:3]}")
    if digest({"seed": epoch["seed"], "token_budget": epoch["token_budget"], "draws": epoch["draws"]}) \
            != epoch["manifest_hash"]:
        raise IngestError("epoch manifest hash does not match its draws")
    out = []
    for n, eid in enumerate(epoch["draws"]):
        e = by_id[eid]
        for row in so.rows_for(e["observation"], e["actions"], id_chunks(e["observation"], k), reservation_fn,
                               encoding=encoding):
            row = {**row, "loss_weight": 1.0, "example_id": eid, "draw": n}
            if set(row) != ROW_KEYS:
                raise IngestError(f"row keys {sorted(set(row) ^ ROW_KEYS)} differ from the trainer contract")
            out.append(row)
    return out
