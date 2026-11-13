"""Chunked inference correctness study (P1-11 prerequisite R7). CPU only; no model, no GPU.

A chunked decision splits the active people into rows of at most k questions (`token_budget.chunk_rows`).
Every row carries the full public state, but a row decides jointly only for its own people. This module:

* checks every chunk row of an observation: each active person exactly once, the full public state in every
  row, and an encoded length (audited Clef layout, real tokenizer) strictly below the 16384-token bound;
* assembles one joint action map from per-chunk predictions and refuses any gap, overlap, foreign ID or
  disallowed action (fail closed: there is no fallback action);
* validates the complete map with the public movement resolver and counts outcomes;
* runs closed-loop episodes on independent synthetic scenarios and checks EVERY observation, not only step 0.

The chunk "predictors" are teacher proxies, not the student. `t1_joint` is T1 on the whole state (the
unchunked reference). `t1_chunk_static` runs T1 per chunk and treats people of other chunks as static
obstacles (they see the state but not the other chunks' moves). `t1_chunk_blind` runs T1 per chunk without
the other people (a worst case). The proxies measure the coordination loss that chunking can cause in a
cooperative planner. They do not show that a chunked student keeps T1's joint decision quality.
"""
from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from . import token_budget as tb
from .contract import ACTIONS, validate_observation
from .dataset import decision_questions
from .encoding import encode_observation
from .sim import Scenario, Simulator
from .sim.movement import resolve_moves
from .teacher.grid import PublicGrid
from .teacher.t1 import T1CongestionAware

STUDY_VERSION = "p1-chunk-joint-map@1.0.0"
RESERVED_STUDY_VERSION = "p1-chunk-reserved@1.0.0"
Tokenize = tb.Tokenize
Predictor = Callable[[Mapping[str, Any], Sequence[str]], Mapping[str, str]]


class ChunkError(ValueError):
    """A chunk row or a joint map is invalid. The decision fails closed."""


# ----- rows ------------------------------------------------------------------------------------------------
def chunk_ids(obs: Mapping[str, Any], k: int) -> list[list[str]]:
    """Active IDs in the encoder's question order, split into chunks of at most k (= chunk_rows)."""
    if type(k) is not int or k < 1:
        raise ChunkError("k must be a positive integer")
    ids = list(decision_questions(obs, style="compact"))
    return [ids[s:s + k] for s in range(0, len(ids), k)]


def check_rows(obs: Mapping[str, Any], k: int, tokenize: Tokenize | None, *, limit: int = tb.SERVE_MAX_LEN,
               question_style: str = "compact", encoding: str = "ascii",
               chunks: Sequence[Sequence[str]] | None = None, joint: Mapping[str, str] | None = None,
               reservations: bool = False) -> dict[str, Any]:
    """Every row of one observation. Raise ChunkError on any violation; return the token counts.

    With `reservations`, the rows are the sequential rows the model sees: row j = full state + the reservation
    block of `joint` restricted to rows < j (`joint` = the decisions of the earlier rows; at inference these are
    the model's own outputs, never a teacher's)."""
    obs = validate_observation(obs)
    full_state = encode_observation(obs, mode=encoding, local_neighbors=False)
    qs = decision_questions(obs, style=question_style)
    active = sorted(p["id"] for p in obs["people"] if p["status"] == "active")
    if reservations and joint is None:
        raise ChunkError("reserved rows need the decisions of the earlier rows")
    # The real row builder (training and inference use it); dummy "stay" golds when no decision is given.
    gold = dict(joint) if joint is not None else {i: "stay" for i in active}
    try:
        built = tb.chunk_rows(obs, gold, k=k, group_id="chunk-study", source="synthetic", encoding=encoding,
                              question_style=question_style, chunks=chunks, reservations=reservations)
    except ValueError as e:
        raise ChunkError(str(e)) from e
    got = [r["chunk"]["ids"] for r in built]
    if chunks is None and got != chunk_ids(obs, k):
        raise ChunkError("chunk order differs from the encoder question order")
    if chunks is not None and [sorted(c) for c in got] != [sorted(c) for c in chunks]:
        raise ChunkError("chunk membership differs from the requested chunks")
    chunks = got
    seen = [i for c in chunks for i in c]
    if sorted(seen) != active or len(seen) != len(set(seen)):
        raise ChunkError("an active person is missing or appears twice")
    rows, decided = [], {}
    for c, row in zip(chunks, built):
        block = tb.reservation_block(obs, decided) if reservations else ""
        if row["state"] != full_state + block:
            raise ChunkError("a chunk row does not carry the full public state (plus its reservations)")
        decided.update({i: gold[i] for i in c})
        if list(row["questions"]) != c or any(row["questions"][i] != qs[i] for i in c):
            raise ChunkError("a chunk row's questions differ from its people")
        if tokenize is not None:
            cnt = tb.clef_counts(tokenize, row["state"], row["questions"], limit)
            if cnt["status"] != "OK":
                raise ChunkError(f"chunk of {len(c)} encodes to {cnt['total']} tokens: {cnt['status']}")
            rows.append({"people": len(c), "total": cnt["total"], "state": cnt["state"], "fixed": cnt["fixed"],
                         "reserved": len(row["chunk"]["reserved_ids"])})
        else:
            rows.append({"people": len(c), "reserved": len(row["chunk"]["reserved_ids"])})
    unchunked = None
    if tokenize is not None:
        u = tb.clef_counts(tokenize, full_state, qs, limit)
        unchunked = {"total": u["total"], "status": u["status"]}
    return {"active": len(active), "rows": rows, "unchunked": unchunked}


# ----- joint map -------------------------------------------------------------------------------------------
def assemble(obs: Mapping[str, Any], chunks: Sequence[Sequence[str]],
             predictions: Sequence[Mapping[str, str]]) -> dict[str, str]:
    """One joint action map from per-chunk predictions. Fail closed on any gap, overlap or bad action."""
    if len(chunks) != len(predictions):
        raise ChunkError("one prediction per chunk is required")
    active = {p["id"] for p in obs["people"] if p["status"] == "active"}
    joint: dict[str, str] = {}
    for ids, pred in zip(chunks, predictions):
        if set(pred) != set(ids):
            raise ChunkError(f"chunk prediction keys differ from its people: "
                             f"missing {sorted(set(ids) - set(pred))[:3]}, extra {sorted(set(pred) - set(ids))[:3]}")
        for pid, a in pred.items():
            if pid in joint:
                raise ChunkError(f"{pid} is decided by two chunks")
            if a not in obs["allowed_actions"] or a not in ACTIONS:
                raise ChunkError(f"{pid}: action {a!r} is not allowed")
            joint[pid] = a
    if set(joint) != active:
        raise ChunkError("the joint map does not cover exactly the active people")
    return dict(sorted(joint.items()))


def resolve(obs: Mapping[str, Any], joint: Mapping[str, str]) -> dict[str, tuple[str, tuple[int, int]]]:
    pg = PublicGrid.from_observation(obs)
    return resolve_moves(height=pg.height, width=pg.width, walls=pg.walls, burning=pg.burning,
                         enter_fire=pg.enter_fire, positions=dict(pg.active), actions=joint)


def cross_chunk_conflicts(obs: Mapping[str, Any], chunks: Sequence[Sequence[str]], joint: Mapping[str, str],
                          res: Mapping[str, tuple[str, Any]]) -> int:
    """Blocked movers whose blocking cell is targeted or held by a person of ANOTHER chunk."""
    chunk_of = {pid: c for c, ids in enumerate(chunks) for pid in ids}
    pos = {p["id"]: tuple(p["position"]) for p in obs["people"] if p["status"] == "active"}
    occ = {v: k for k, v in pos.items()}
    from .sim.movement import DELTAS
    tgt = {pid: (pos[pid][0] + DELTAS[a][0], pos[pid][1] + DELTAS[a][1]) for pid, a in joint.items() if a != "stay"}
    by_target: dict[tuple[int, int], list[str]] = {}
    for pid, t in tgt.items():
        by_target.setdefault(t, []).append(pid)
    n = 0
    for pid, (outcome, _) in res.items():
        if outcome not in ("blocked_occupied", "blocked_conflict", "blocked_cycle"):
            continue
        t = tgt[pid]
        others = {chunk_of[o] for o in by_target.get(t, []) if o != pid}
        if t in occ:
            others.add(chunk_of[occ[t]])
        if others - {chunk_of[pid]}:
            n += 1
    return n


# ----- teacher proxies -------------------------------------------------------------------------------------
def _sub_observation(obs: Mapping[str, Any], ids: Sequence[str], others: str) -> dict[str, Any]:
    keep = set(ids)
    sub = {**obs, "map": dict(obs["map"])}
    if others == "static":
        extra = [p["position"] for p in obs["people"] if p["status"] == "active" and p["id"] not in keep]
        sub["map"]["obstacles"] = sorted({tuple(x) for x in obs["map"]["obstacles"]} | {tuple(x) for x in extra})
        sub["map"]["obstacles"] = [list(x) for x in sub["map"]["obstacles"]]
    sub["people"] = [p for p in obs["people"] if p["id"] in keep]
    return sub


def _reserved_sub_observation(obs: Mapping[str, Any], ids: Sequence[str], decided: Mapping[str, str],
                              exits: Sequence[str] | None = None) -> tuple[dict[str, Any], set]:
    """Sub-observation of one sequential chunk: undecided people of later chunks are static obstacles at their
    cells; people decided by earlier chunks are obstacles at their reserved cells (`token_budget.reserved_target`)
    — exactly the information of the reservation block. A reserved cell that is an exit cannot be an obstacle
    in a valid observation; it is returned in the reserved set instead. `exits` restricts the exits."""
    keep = set(ids)
    reserved = {tb.reserved_target(obs, pid, a) for pid, a in decided.items()}
    exit_cells = {tuple(x["position"]) for x in obs["map"]["exits"]}
    later = {tuple(p["position"]) for p in obs["people"] if p["status"] == "active"
             and p["id"] not in keep and p["id"] not in decided}
    obst = {tuple(x) for x in obs["map"]["obstacles"]} | later | (reserved - exit_cells)
    sub = {**obs, "map": {**obs["map"], "obstacles": [list(x) for x in sorted(obst)]}}
    if exits is not None:
        sub["map"]["exits"] = [x for x in obs["map"]["exits"] if x["id"] in set(exits)]
    sub["people"] = [p for p in obs["people"] if p["id"] in keep]
    return sub, reserved


def _respect_reservations(obs: Mapping[str, Any], actions: Mapping[str, str], reserved: set) -> dict[str, str]:
    """A chunk may not target a reserved cell (here: a reserved exit cell, the only kind the sub-observation
    cannot encode as an obstacle). Such a move becomes `stay`."""
    return {pid: ("stay" if a != "stay" and tb.reserved_target(obs, pid, a) in reserved else a)
            for pid, a in actions.items()}


def exit_allocation(obs: Mapping[str, Any]) -> dict[str, str | None]:
    """Simple public, teacher-free global exit allocation: capacity ceil(active / reachable safe exits) per exit;
    (distance, person ID, exit ID) pairs in ascending order are assigned greedily. A person with no safe exit
    path gets None. Recomputed at every observation (no hidden state)."""
    from dataclasses import replace

    from .teacher.grid import exit_distance_field
    pg = PublicGrid.from_observation(obs)
    fields = {eid: exit_distance_field(replace(pg, exits=((eid, p),))) for eid, p in pg.safe_exits()}
    pos = dict(pg.active)
    pairs = sorted((f[pos[pid]][0], pid, eid) for eid, f in fields.items() for pid in pos if pos[pid] in f)
    reachable = sorted({eid for _, _, eid in pairs})
    cap = -(-len(pos) // len(reachable)) if reachable else 0
    load: Counter = Counter()
    out: dict[str, str | None] = {pid: None for pid in pos}
    for _, pid, eid in pairs:
        if out[pid] is None and load[eid] < cap:
            out[pid], load[eid] = eid, load[eid] + 1
    return out


def exit_chunks(obs: Mapping[str, Any], k: int) -> list[list[str]]:
    """Chunks grouped by allocated exit (exit-ID order, unassigned last), question order inside, at most k."""
    alloc = exit_allocation(obs)
    order = list(decision_questions(obs, style="compact"))
    groups: dict[str, list[str]] = {}
    for pid in order:
        groups.setdefault(alloc[pid] or "~none", []).append(pid)
    return [g[s:s + k] for key in sorted(groups) for g in [groups[key]] for s in range(0, len(g), k)]


def predictor(name: str) -> Callable[[Mapping[str, Any], Sequence[Sequence[str]]], list[dict[str, str]]]:
    """Chunk-decision proxies. `t1_joint` is the full T1 planner (the unchunked REFERENCE; it is never called
    inside a chunked method when there is more than one chunk). Every other method calls T1 only on one
    chunk's sub-observation, as a stand-in for one student row; no method reads another chunk's plan except
    through the public reservation block."""
    t1 = T1CongestionAware()

    def joint(obs, chunks):
        a = t1.act(obs)
        return [{i: a[i] for i in c} for c in chunks]

    def per_chunk(others):
        def f(obs, chunks):
            if len(chunks) == 1:
                return joint(obs, chunks)
            out = []
            for c in chunks:
                a = t1.act(_sub_observation(obs, c, others))
                out.append({i: a[i] for i in c})
            return out
        return f

    def reserved(exit_restricted):
        def f(obs, chunks):
            if len(chunks) == 1 and not exit_restricted:
                return joint(obs, chunks)
            alloc = exit_allocation(obs) if exit_restricted else {}
            out, decided = [], {}
            for c in chunks:
                ex = None
                if exit_restricted:
                    got = {alloc[i] for i in c}
                    ex = None if None in got or len(got) != 1 else sorted(got)
                sub, res = _reserved_sub_observation(obs, c, decided, ex)
                planned = t1.act(sub)
                a = _respect_reservations(obs, {i: planned[i] for i in c}, res)
                out.append(a)
                decided.update(a)
            return out
        return f

    table = {"t1_joint": joint, "t1_chunk_static": per_chunk("static"), "t1_chunk_blind": per_chunk("blind"),
             "t1_chunk_reserved": reserved(False), "t1_chunk_reserved_fix": reserved(False),
             "t1_chunk_exitalloc": reserved(True)}
    if name not in table:
        raise ValueError(f"unknown predictor {name}")
    return table[name]


METHOD_CHUNKER = {"t1_chunk_exitalloc": "exit"}            # every other method: ID-order chunks
METHOD_RESERVED = {"t1_chunk_reserved", "t1_chunk_reserved_fix", "t1_chunk_exitalloc"}
METHOD_CORRECTED = {"t1_chunk_reserved_fix"}


def method_chunks(name: str, obs: Mapping[str, Any], k: int) -> list[list[str]]:
    return exit_chunks(obs, k) if METHOD_CHUNKER.get(name) == "exit" else chunk_ids(obs, k)


def correct_conflicts(obs: Mapping[str, Any], chunks: Sequence[Sequence[str]], joint: Mapping[str, str], *,
                      max_changes: int) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Bounded, deterministic, teacher-free conflict correction on the assembled RAW map (public grid only).

    For each mover blocked by another person (occupied, conflict or cycle; inside or across chunks), in ID order
    and at most `max_changes`: take the first of the other moves,
    in contract action order, whose cell is passable, not an end cell of anyone in the raw resolution, not
    already taken by an earlier correction, and strictly nearer to a safe exit. If the corrected map resolves to
    more blocked moves than the raw map, the whole correction is reverted. Returns (map, change log); the raw map
    is never modified, so raw and corrected actions stay separately recorded."""
    from .sim.movement import DELTAS
    from .teacher.grid import exit_distance_field
    res = resolve(obs, joint)
    pg = PublicGrid.from_observation(obs)
    field = exit_distance_field(pg)
    pos = dict(pg.active)
    chunk_of = {pid: c for c, ids in enumerate(chunks) for pid in ids}
    taken = {end for _, end in res.values()}
    claims: dict[tuple[int, int], list[str]] = {}
    for pid, a in joint.items():
        if a != "stay":
            claims.setdefault((pos[pid][0] + DELTAS[a][0], pos[pid][1] + DELTAS[a][1]), []).append(pid)
    occ = {v: k for k, v in pos.items()}
    out, log = dict(joint), []
    for pid in sorted(joint):
        if len(log) >= max_changes:
            break
        if res[pid][0] not in BLOCKED or pos[pid] not in field:
            continue
        t0 = (pos[pid][0] + DELTAS[joint[pid]][0], pos[pid][1] + DELTAS[joint[pid]][1])
        others = {chunk_of[o] for o in claims.get(t0, []) if o != pid} | ({chunk_of[occ[t0]]} if t0 in occ else set())
        cross = bool(others - {chunk_of[pid]})
        here = field[pos[pid]][0]
        for a in ACTIONS:
            if a in ("stay", joint[pid]):
                continue
            t = (pos[pid][0] + DELTAS[a][0], pos[pid][1] + DELTAS[a][1])
            if pg.passable(t) and t not in taken and t in field and field[t][0] < here:
                out[pid] = a
                taken.add(t)
                log.append({"id": pid, "raw": joint[pid], "corrected": a, "cross_chunk": cross})
                break
    if log and sum(o in BLOCKED for o, _ in resolve(obs, out).values()) > sum(o in BLOCKED for o, _ in res.values()):
        return dict(joint), [{"reverted": len(log)}]
    return out, log


# ----- synthetic scenarios ---------------------------------------------------------------------------------
def synthetic_scenario(n: int, *, seed: int, fire_model: str, crowded: bool = True,
                       max_steps: int = 60, tag: str = STUDY_VERSION) -> Scenario:
    """Independent synthetic room: not a TRAIN, DEV, held-out or benchmark scenario.

    A room with a partition wall with two gaps (a bottleneck), three exits on the far wall and n people packed
    in the near half (crowded) or spread over the room. `tag` seeds the generator: the 1.0.0 study keeps its
    default; the reserved-chunk study uses its own tag, so its rooms are new and independent."""
    rng = random.Random(f"{tag}|{n}|{seed}|{fire_model}|{crowded}")
    h = 20 if n <= 120 else 24
    w = 30 if n <= 120 else 36
    mid = w // 2
    gaps = {h // 3, 2 * h // 3}
    walls = [[r, mid] for r in range(h) if r not in gaps]
    exits = [{"id": f"E{j}", "position": [r, w - 1]} for j, r in enumerate((1, h // 2, h - 2))]
    ign = [[rng.randrange(1, h - 1), rng.randrange(1, mid - 1)]]
    blocked = {tuple(x) for x in walls} | {tuple(x["position"]) for x in exits} | {tuple(x) for x in ign}
    cols = range(0, mid) if crowded else range(0, w - 1)
    cells = [(r, c) for r in range(h) for c in cols if (r, c) not in blocked]
    if n > len(cells):
        raise ValueError("room too small for n")
    people = [{"id": f"P{i:03d}", "position": list(p)} for i, p in enumerate(rng.sample(cells, n))]
    fire = {"model": fire_model, "spread_interval": 3, "exits_flammable": False}
    if fire_model == "stochastic":
        fire["spread_probability"] = 0.35
    return Scenario.from_dict({
        "schema_version": "1.0",
        "scenario_id": f"chunk-{'study' if tag == STUDY_VERSION else 'rsv'}-n{n}-{fire_model[:3]}-{seed}",
        "grid": {"width": w, "height": h}, "obstacles": walls, "exits": exits, "people": people,
        "ignitions": ign, "fire": fire, "movement": {"conflict_resolution": "lowest_id_wins", "enter_fire": "blocked"},
        "termination": {"max_steps": max_steps}, "seed": rng.randrange(2 ** 63),
        "description": "synthetic chunk-study room (not a benchmark scenario)"})


# ----- closed loop -----------------------------------------------------------------------------------------
BLOCKED = ("blocked_occupied", "blocked_conflict", "blocked_cycle")


def run_episode(scenario: Scenario, pred_name: str, k: int, tokenize: Tokenize | None, *,
                limit: int = tb.SERVE_MAX_LEN, reference: str = "t1_joint") -> dict[str, Any]:
    """Drive the episode with `pred_name` chunked at k. At EVERY observation: check rows, assemble, resolve,
    and compare with the unchunked reference decision on the same observation."""
    import time
    sim = Simulator(scenario)
    pred, ref = predictor(pred_name), predictor(reference)
    reserved, corrected = pred_name in METHOD_RESERVED, pred_name in METHOD_CORRECTED
    totals: Counter = Counter()
    max_tokens, max_unchunked, steps, errors = 0, 0, 0, []
    unchunked_status: Counter = Counter()
    step_tokens: list[list[int]] = []          # per step: the encoded total of every chunk row, in row order
    step_ms: list[float] = []                  # per step: wall ms of chunking + chunk decisions + correction
    raw_log: list[dict[str, Any]] = []         # per step with a correction: raw and corrected actions
    cpu0 = time.process_time()
    decision_cpu = 0.0
    while not sim.terminated:
        obs = sim.observe()
        t_wall, t_cpu = time.perf_counter(), time.process_time()
        chunks = method_chunks(pred_name, obs, k)
        try:
            raw = assemble(obs, chunks, pred(obs, chunks))
        except ChunkError as e:
            errors.append({"step": obs["step"], "error": str(e)})
            break
        joint, fixes = raw, []
        if corrected:
            joint, fixes = correct_conflicts(obs, chunks, raw, max_changes=max(1, len(raw) // 10))
        step_ms.append(round((time.perf_counter() - t_wall) * 1000, 2))
        decision_cpu += time.process_time() - t_cpu
        try:      # the rows the model would see: built from the RAW decisions of the earlier rows
            rows = check_rows(obs, k, tokenize, limit=limit, chunks=chunks, joint=raw, reservations=reserved)
        except ChunkError as e:
            errors.append({"step": obs["step"], "error": str(e)})
            break
        if tokenize is not None:
            step_tokens.append([r["total"] for r in rows["rows"]])
            max_tokens = max(max_tokens, max(r["total"] for r in rows["rows"]))
            max_unchunked = max(max_unchunked, rows["unchunked"]["total"])
            unchunked_status[rows["unchunked"]["status"]] += 1
        if fixes:
            raw_log.append({"step": obs["step"], "changes": fixes})
            totals["corrections"] += sum(1 for f in fixes if "corrected" in f)
            totals["correction_reverts"] += sum(f.get("reverted", 0) > 0 for f in fixes)
        raw_res = resolve(obs, raw)
        totals["raw_blocked"] += sum(1 for o, _ in raw_res.values() if o in BLOCKED)
        totals["raw_cross_chunk_blocked"] += cross_chunk_conflicts(obs, chunks, raw, raw_res)
        totals["reservation_conflicts"] += reservation_conflicts(obs, chunks, raw) if reserved else 0
        res = resolve(obs, joint)
        refj = assemble(obs, [list(joint)], ref(obs, [list(joint)]))
        ref_res = resolve(obs, refj)
        out = Counter(o for o, _ in res.values())
        totals["decisions"] += 1
        totals["person_decisions"] += len(joint)
        totals["multi_chunk_decisions"] += len(chunks) > 1
        totals["moves"] += sum(a != "stay" for a in joint.values())
        totals["blocked"] += sum(out[o] for o in BLOCKED)
        totals["ref_blocked"] += sum(1 for o, _ in ref_res.values() if o in BLOCKED)
        totals["cross_chunk_blocked"] += cross_chunk_conflicts(obs, chunks, joint, res)
        totals["action_differs_from_unchunked"] += sum(joint[i] != refj[i] for i in joint)
        for o in BLOCKED:
            totals[o] += out[o]
        sim.step(joint)
        steps += 1
    final = sim.observe()
    st = Counter(p["status"] for p in final["people"])
    for key in ("decisions", "person_decisions", "multi_chunk_decisions", "moves", "blocked", "ref_blocked",
                "cross_chunk_blocked", "action_differs_from_unchunked", "raw_blocked", "raw_cross_chunk_blocked",
                "reservation_conflicts", "corrections", "correction_reverts", *BLOCKED):
        totals.setdefault(key, 0)          # a run that failed closed at step 0 still has every counter
    return {"scenario": scenario.scenario_id, "n": len(scenario.people), "fire": scenario.fire_model,
            "predictor": pred_name, "k": k, "steps": steps, "errors": errors, "complete": not errors,
            "termination": sim.termination_reason, "evacuated": st["evacuated"], "casualties": st["casualty"],
            "unresolved": st["active"], "max_row_tokens": max_tokens, "max_unchunked_tokens": max_unchunked,
            "unchunked_status_counts": dict(unchunked_status), "reserved_rows": reserved,
            "corrected": corrected, "step_row_tokens": step_tokens, "step_decision_ms": step_ms,
            "decision_cpu_s": round(decision_cpu, 3), "episode_cpu_s": round(time.process_time() - cpu0, 3),
            "corrections_log": raw_log, "joint_map_valid_steps": totals["decisions"], **dict(totals)}


def severe_regression_threshold(n: int) -> int:
    """Preregistered for the reserved-chunk study: casualties+unresolved worse by at least max(3, 5 % of N)."""
    return max(3, -(-5 * n // 100))


def paired_summary(runs: Sequence[Mapping[str, Any]], baseline: str = "t1_joint") -> list[dict[str, Any]]:
    """Paired (same scenario hash, same fire stream) differences method - baseline, per (N, k, method).

    Deltas are reported separately for casualties, unresolved, casualties+unresolved and evacuated; a method is
    never summarized by the combined number alone. Runs that failed closed are counted, never dropped."""
    per_k = baseline != "t1_joint"            # a chunked baseline is paired at the same k; t1_joint has one run
    key = (lambda r: (r["scenario_hash"], r["k"])) if per_k else (lambda r: r["scenario_hash"])
    base = {key(r): r for r in runs if r["predictor"] == baseline and r["complete"]}
    cells: dict[tuple, list[Mapping[str, Any]]] = {}
    for r in runs:
        if r["predictor"] != baseline:
            cells.setdefault((r["n"], r["k"], r["predictor"]), []).append(r)
    out = []
    for (n, k, p), rs in sorted(cells.items()):
        done = [r for r in rs if r["complete"] and key(r) in base]
        if per_k and not done and not any(key(r) in base for r in rs):
            continue                          # no paired baseline at this k (e.g. k >= N for a chunked baseline)
        d = {m: [r[m] - base[key(r)][m] for r in done]
             for m in ("casualties", "unresolved", "evacuated")}
        cu = [a + b for a, b in zip(d["casualties"], d["unresolved"])]
        thr = severe_regression_threshold(n)
        out.append({
            "n": n, "k": k, "method": p, "baseline": baseline, "scenarios": len(rs), "failed_closed": len(rs) - len(done),
            "sum_delta_casualties": sum(d["casualties"]), "sum_delta_unresolved": sum(d["unresolved"]),
            "sum_delta_cas_plus_unres": sum(cu), "sum_delta_evacuated": sum(d["evacuated"]),
            "worse_cas_plus_unres": sum(x > 0 for x in cu), "better_cas_plus_unres": sum(x < 0 for x in cu),
            "equal_cas_plus_unres": sum(x == 0 for x in cu),
            "unresolved_increased": sum(x > 0 for x in d["unresolved"]),
            "severe_regression_threshold": thr,
            "severe_regressions": [r["scenario"] for r, x in zip(done, cu) if x >= thr],
            "cross_chunk_blocked": sum(r["cross_chunk_blocked"] for r in done),
            "raw_cross_chunk_blocked": sum(r["raw_cross_chunk_blocked"] for r in done),
            "reservation_conflicts": sum(r["reservation_conflicts"] for r in done),
            "corrections": sum(r.get("corrections", 0) for r in done),
            "max_row_tokens": max((r["max_row_tokens"] for r in done), default=0),
        })
    return out


def reservation_conflicts(obs: Mapping[str, Any], chunks: Sequence[Sequence[str]], raw: Mapping[str, str]) -> int:
    """Raw decisions of a later chunk whose target cell equals a cell reserved by an earlier chunk of the same
    step (the reservation block was visible and was not respected). Counts violations; it changes nothing."""
    n, reserved = 0, set()
    for c in chunks:
        for pid in c:
            if raw[pid] != "stay" and tb.reserved_target(obs, pid, raw[pid]) in reserved:
                n += 1
        reserved |= {tb.reserved_target(obs, pid, raw[pid]) for pid in c}
    return n
