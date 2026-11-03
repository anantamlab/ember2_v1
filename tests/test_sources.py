"""Zip ingestion and source-quality policy (CPU only)."""
import json
import zipfile
from pathlib import Path

import pytest

from ember_v1.sources import ingest_zip, training_records

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _zip(tmp_path, members):
    path = tmp_path / "src.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for name, text in members.items():
            zf.writestr(name, text)
    return path


def test_scaffold_layout_yields_one_fixture_trace(tmp_path):
    path = _zip(tmp_path, {
        "pkg/examples/teacher_trace.json": (EXAMPLES / "teacher_trace.json").read_text(encoding="utf-8"),
        "pkg/examples/observation.json": (EXAMPLES / "observation.json").read_text(encoding="utf-8"),
        "pkg/src/code.py": "raise SystemExit('never executed')",
    })
    records, entry = ingest_zip(path, source_id="p0a-fixtures", quality="synthetic_fixture")
    assert len(records) == 1 and entry["decisions"] == 2
    assert entry["non_trace_json_members"] == 1 and entry["training_eligible"] is False
    assert records[0]["group_id"] == "p0a-fixtures/example-independent-layout-family-A"
    assert training_records(records) == []
    assert training_records(records, allow_fixtures=True) == records


def test_jsonl_and_list_members(tmp_path):
    trace = json.loads((EXAMPLES / "teacher_trace.json").read_text(encoding="utf-8"))
    path = _zip(tmp_path, {"a.jsonl": json.dumps(trace) + "\n" + json.dumps(trace) + "\n",
                           "b.json": json.dumps({"traces": [trace]})})
    records, entry = ingest_zip(path, source_id="s", quality="teacher_verified")
    assert entry["traces"] == 3 and len(training_records(records)) == 3


@pytest.mark.parametrize("bad", ["missing_action", "privileged_field"])
def test_invalid_trace_fails_closed(tmp_path, bad):
    trace = json.loads((EXAMPLES / "teacher_trace.json").read_text(encoding="utf-8"))
    if bad == "missing_action":
        trace["actions"].pop("P2")
    else:
        trace["observation"]["seed"] = 7
    with pytest.raises(ValueError):
        ingest_zip(_zip(tmp_path, {"t.json": json.dumps(trace)}), source_id="s", quality="synthetic_fixture")


def test_bad_quality_or_source_id(tmp_path):
    path = _zip(tmp_path, {})
    with pytest.raises(ValueError):
        ingest_zip(path, source_id="s", quality="gold")
    with pytest.raises(ValueError):
        ingest_zip(path, source_id="a/b", quality="synthetic_fixture")
