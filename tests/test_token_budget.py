"""token_budget: exact Clef layout counts (fake tokenizers), outcome classes, lossless chunking."""
import json

import pytest

from ember_v1 import p0b
from ember_v1 import token_budget as tb
from ember_v1.dataset import build_decision_row
from ember_v1.synthetic import make_synthetic_observation


def chars(text):                      # one token per character: counts equal text lengths
    return list(text)


def words(text):                      # a coarser tokenizer, to show the count is per piece
    return text.split() or ([0] if text else [])


def _row(n, style="compact"):
    obs = make_synthetic_observation(n)
    return obs, build_decision_row(obs, {f"P{i}": "stay" for i in range(n)}, group_id="g", source="s",
                                   question_style=style)


def _full_text(state, questions):
    """Independent re-statement of upstream encode_record's text (unsloth/_vendor/clef/joint_schema_model.py)."""
    parts = [f"<|im_start|>system\n{tb.SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n", state,
             "\n\nSCHEMA FIELDS:\n"]
    for i, (qid, q) in enumerate(questions.items()):
        parts.append(f"\nFIELD {i + 1}\nID: {qid}\nTYPE: choice\nINSTRUCTION: {q['instructions']}\nALLOWED OPTIONS:\n")
        for j, (oid, desc) in enumerate(sorted(q["criteria"].items())):
            sem = json.dumps({"description": desc, "option_id": oid}, separators=(",", ":"), sort_keys=True)
            parts.append(f"OPTION {j + 1}: {sem}\n")
        parts.append("END FIELD\n")
    parts.append("\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:")
    return "".join(parts)


def test_pins_equal_p0b_audit():
    assert tb.CLEF_SHA256 == p0b.AUDITED_CLEF_SHA256
    assert tb.SERVE_MAX_LEN == p0b.AUDITED_SERVE_MAX_LEN


@pytest.mark.parametrize("style", ["compact", "verbose"])
@pytest.mark.parametrize("n", [1, 3, 17])
def test_char_count_equals_full_text_length(n, style):
    _, row = _row(n, style)
    c = tb.clef_counts(chars, row["state"], row["questions"])
    assert c["total"] == len(_full_text(row["state"], row["questions"]))
    assert c["state"] == len(row["state"]) and c["fixed"] == c["total"] - c["state"]
    assert c["questions"] == n and c["status"] == "OK"


def test_counts_are_per_piece():
    _, row = _row(3)
    c = tb.clef_counts(words, row["state"], row["questions"])
    assert c["total"] == c["prefix"] + c["state"] + c["schema"] + c["suffix"]
    assert c["schema"] == len(words("\n\nSCHEMA FIELDS:\n")) + sum(tb.schema_tokens(words, row["questions"]))


def test_classify_boundaries():
    assert tb.classify(11, 20, 10) == "SCHEMA_TOO_LONG"
    assert tb.classify(10, 11, 10) == "STATE_CUT"
    assert tb.classify(5, 10, 10) == "NO_SPARE_TOKEN"
    assert tb.classify(5, 9, 10) == "OK"


def test_status_from_limit():
    _, row = _row(3)
    c = tb.clef_counts(chars, row["state"], row["questions"])
    assert tb.clef_counts(chars, row["state"], row["questions"], c["fixed"] - 1)["status"] == "SCHEMA_TOO_LONG"
    assert tb.clef_counts(chars, row["state"], row["questions"], c["total"] - 1)["status"] == "STATE_CUT"
    assert tb.clef_counts(chars, row["state"], row["questions"], c["total"])["status"] == "NO_SPARE_TOKEN"
    assert tb.clef_counts(chars, row["state"], row["questions"], c["total"] + 1)["status"] == "OK"


def test_list_criteria_empty_instruction_and_type():
    q = {"a": {"type": "choice", "instructions": "", "criteria": ["x", "y"]}}
    assert tb.schema_tokens(chars, q)[0] == len("\nFIELD 1\nID: a\nTYPE: choice\nINSTRUCTION: a\nALLOWED OPTIONS:\n"
                                               'OPTION 1: {"option_id":"x"}\nOPTION 2: {"option_id":"y"}\n'
                                               "END FIELD\n")
    with pytest.raises(ValueError):
        tb.schema_tokens(chars, {"a": {"type": "score", "criteria": ["x"]}})


@pytest.mark.parametrize("k", [1, 4, 7, 10, 11])
def test_chunk_rows_keep_every_person(k):
    obs, row = _row(10)
    rows = tb.chunk_rows(obs, row["gold"], k=k, group_id="g", source="s")
    ids = [i for r in rows for i in r["questions"]]
    assert ids == list(row["questions"])                     # each person once, in ID order
    assert all(r["state"] == row["state"] for r in rows)     # full shared state in every row
    assert all(r["gold"] == {i: row["gold"][i] for i in r["questions"]} for r in rows)
    assert all(r["chunk"]["of"] == len(rows) == -(-10 // k) for r in rows)
    assert all(len(r["questions"]) <= k for r in rows)


@pytest.mark.parametrize("k", [0, -1, 1.0, True])
def test_chunk_rows_bad_k(k):
    obs, row = _row(2)
    with pytest.raises(ValueError):
        tb.chunk_rows(obs, row["gold"], k=k, group_id="g", source="s")


def test_chunk_rows_refuses_bad_actions():
    obs, row = _row(3)
    gold = dict(row["gold"])
    gold.pop("P0")
    with pytest.raises(ValueError):
        tb.chunk_rows(obs, gold, k=2, group_id="g", source="s")


def test_max_chunk_is_tight():
    _, row = _row(30)
    one = tb.clef_counts(chars, row["state"], dict(list(row["questions"].items())[:8]))["total"]
    limit = one + 1                                          # 8 questions fit with a spare token
    k = tb.max_chunk(chars, row["state"], row["questions"], limit)
    qs, ids = row["questions"], list(row["questions"])
    fits = lambda m: all(tb.clef_counts(chars, row["state"], {i: qs[i] for i in ids[s:s + m]}, limit)["status"]
                         == "OK" for s in range(0, len(ids), m))
    assert 1 <= k <= 8 and fits(k) and not fits(k + 1)       # later IDs (P10..) cost more, so k may be < 8


def test_max_chunk_zero_when_nothing_fits():
    _, row = _row(3)
    assert tb.max_chunk(chars, row["state"], row["questions"], 10) == 0
