"""CPU-only token budget of the audited Clef encoding (P0B G10, the N=200 preservation skip).

`clef_counts` re-states the token layout of the audited vendored Clef `encode_record`
(unsloth 2026.10.11 wheel, `unsloth/_vendor/clef/joint_schema_model.py`, sha256 `CLEF_SHA256`):

    input = prefix(system prompt, "STATE:") + state + schema(questions) + suffix
    fixed = prefix + schema + suffix
    fixed > max_length          -> ValueError -> build_dataset skips the row (FAIL_ENCODE_SKIPPED)
    state cut to max_length - fixed tokens otherwise (silent; p0b detects it as FAIL_TRUNCATED)

Every text piece is tokenized separately, as upstream does (`tokenizer(text, add_special_tokens=False)`).
The counts are exact for a tokenizer that behaves like the pinned one; they are not a GPU result and do not
replace the `smoke` preservation evidence. This module never truncates people or questions: it only counts
and classifies. Chunking (`chunk_rows`) splits the questions over several rows and keeps every person.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .contract import validate_actions, validate_observation
from .dataset import decision_questions
from .encoding import encode_observation

CLEF_SHA256 = "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3"   # = p0b.AUDITED_CLEF_SHA256
UNSLOTH_WHEEL = ("unsloth-2026.10.11-py3-none-any.whl",
                 "sha256:baff520eacd8da49a8d211a49fb62e407cddd80a72570fd2ffe596a68150992e")
SERVE_MAX_LEN = 16384                                                               # = p0b.AUDITED_SERVE_MAX_LEN
SYSTEM_PROMPT = ("Read the complete state and schema. Decide every field jointly. Each answer "
                 "must be exactly one of that field's allowed options.")
Tokenize = Callable[[str], Sequence[int]]


def _render(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                                                           sort_keys=True)


def _options(question: Mapping[str, Any]) -> list[tuple[str, Any]]:
    crit = question["criteria"]
    if isinstance(crit, list):                       # upstream _clef_question: list -> dict.fromkeys
        crit = dict.fromkeys(crit)
    return sorted((str(k), v) for k, v in crit.items())


def schema_tokens(tokenize: Tokenize, questions: Mapping[str, Mapping[str, Any]]) -> list[int]:
    """Per-question schema token counts (choice questions only, as the decision rows use)."""
    out = []
    for i, (qid, q) in enumerate(questions.items()):
        if q.get("type") != "choice":
            raise ValueError("only choice questions are counted")
        n = len(tokenize(f"\nFIELD {i + 1}\nID: {qid}\nTYPE: {q['type']}\nINSTRUCTION: "))
        ins = q.get("instructions")
        n += len(tokenize(_render(str(qid) if ins is None or ins == "" else ins)))
        n += len(tokenize("\nALLOWED OPTIONS:\n"))
        for j, (oid, desc) in enumerate(_options(q)):
            sem = {"option_id": oid} if desc is None else {"option_id": oid, "description": desc}
            n += len(tokenize(f"OPTION {j + 1}: ")) + len(tokenize(_render(sem))) + len(tokenize("\n"))
        n += len(tokenize("END FIELD\n"))
        out.append(n)
    return out


def clef_counts(tokenize: Tokenize, state: Any, questions: Mapping[str, Mapping[str, Any]],
                limit: int = SERVE_MAX_LEN) -> dict[str, Any]:
    prefix = len(tokenize(f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"))
    suffix = len(tokenize("\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"))
    per_q = schema_tokens(tokenize, questions)
    schema = len(tokenize("\n\nSCHEMA FIELDS:\n")) + sum(per_q)
    st = len(tokenize(_render(state)))
    fixed = prefix + schema + suffix
    total = fixed + st
    return {"prefix": prefix, "state": st, "schema": schema, "suffix": suffix, "fixed": fixed, "total": total,
            "questions": len(per_q), "per_question_max": max(per_q, default=0), "limit": limit,
            "status": classify(fixed, total, limit)}


def classify(fixed: int, total: int, limit: int) -> str:
    """Upstream outcome at `limit`, with p0b's spare-token rule."""
    if fixed > limit:
        return "SCHEMA_TOO_LONG"          # encode_record raises; build_dataset skips (FAIL_ENCODE_SKIPPED)
    if total > limit:
        return "STATE_CUT"                # state silently cut; p0b reference check -> FAIL_TRUNCATED
    if total == limit:
        return "NO_SPARE_TOKEN"           # p0b -> FAIL_NO_SPARE_TOKEN
    return "OK"


def chunk_rows(observation: Mapping[str, Any], actions: Mapping[str, str], *, k: int, group_id: str,
               source: str, encoding: str = "ascii", question_style: str = "compact",
               local_neighbors: bool = False) -> list[dict[str, Any]]:
    """Per-chunk rows: the FULL shared global state, and the questions of at most `k` people per row.

    Every active person is in exactly one row, in ID order; nothing is dropped. Chunks decide jointly only
    within a chunk; cross-chunk coordination comes from the shared state alone. This is an inference-protocol
    change to be evaluated, not a parity claim.
    """
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    obs = validate_observation(observation)
    gold = validate_actions(obs, actions)
    state = encode_observation(obs, mode=encoding, local_neighbors=local_neighbors)
    qs = decision_questions(obs, style=question_style)
    ids = list(qs)
    rows = []
    for c, start in enumerate(range(0, len(ids), k)):
        part = ids[start:start + k]
        rows.append({"state": state, "questions": {i: qs[i] for i in part}, "gold": {i: gold[i] for i in part},
                     "group_id": group_id, "source": source, "chunk": {"index": c, "of": -(-len(ids) // k),
                                                                       "ids": part}})
    for r in rows:
        r["chunk"]["of"] = len(rows)
    assert sorted(i for r in rows for i in r["questions"]) == sorted(ids)
    return rows


def max_chunk(tokenize: Tokenize, state: str, questions: Mapping[str, Mapping[str, Any]],
              limit: int = SERVE_MAX_LEN) -> int:
    """Binary-search k such that every chunk of k questions (in ID order) fits with a spare token; 0 if none.

    The returned k is always verified. Chunk cost is near-monotone in k (the last, partial chunk can break
    strict monotonicity), so k is a safe size, not a proven maximum.
    """
    ids = list(questions)
    best = 0
    lo, hi = 1, len(ids)
    while lo <= hi:                       # chunk cost is monotone in k for prefix chunks of the same question set
        mid = (lo + hi) // 2
        ok = all(clef_counts(tokenize, state, {i: questions[i] for i in ids[s:s + mid]}, limit)["status"] == "OK"
                 for s in range(0, len(ids), mid))
        if ok:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best
