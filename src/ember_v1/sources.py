"""Training-data source registry and zip ingestion (no extraction to disk).

Every source has a `quality` tier. Only `teacher_verified` traces (simulator-
replayed teacher output, HLD v3 section 7) are training-eligible by default.
A `teacher_verified` trace must carry its whole episode, and the episode must
pass full replay (`ember_v1.sim.verify_replay(partial=False)`); see
`verify_teacher_decision`.
`synthetic_fixture` sources (hand-written example labels) feed the pipeline
smoke split only, so made-up labels never count as teacher evidence.

Zip members are read in memory and parsed as JSON only; nothing is executed or
written outside the requested output file.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .contract import validate_actions, validate_observation
from .sim import Scenario, Simulator, verify_replay
from .sim.canonical import canonical_json, digest

QUALITIES = ("teacher_verified", "teacher_unverified", "synthetic_fixture")
TRAINING_QUALITIES = frozenset({"teacher_verified"})
_TRACE_KEYS = {"observation", "actions", "group_id", "source"}
_MAX_MEMBER_BYTES = 50 * 2**20  # guard against oversized / zip-bomb members
_MAX_TOTAL_BYTES = 2 * 2**30    # declared uncompressed total of one zip
_MAX_MEMBERS = 100_000
_MAX_RATIO = 200                # declared uncompressed / compressed size, for members above 1 MiB


def verify_teacher_decision(trace: Mapping[str, Any], cache: dict[str, Any] | None = None) -> dict[str, Any]:
    """Full-episode replay evidence for one `teacher_verified` decision. Fail closed.

    The trace must carry `episode` = {"scenario", "seed", "records"} and `step`
    (1-based). The whole episode must replay byte-for-byte to termination
    (`verify_replay(..., partial=False)`); a valid prefix is not enough. The
    decision's observation must equal the replayed observation before `step`,
    and its actions must equal that step's recorded actions.
    """
    ep, step = trace.get("episode"), trace.get("step")
    if not isinstance(ep, Mapping) or set(ep) != {"scenario", "seed", "records"}:
        raise ValueError("teacher_verified needs episode = {scenario, seed, records}")
    records = ep["records"]
    if not isinstance(records, list) or type(step) is not int or not 1 <= step <= len(records):
        raise ValueError("teacher_verified needs an integer step inside the episode")
    # A shared-episode member attaches one episode object to many traces: hash it once per object. The cache
    # keeps the object alive, so its id cannot be reused while the cache exists.
    memo = cache.get(("id", id(ep))) if cache is not None else None
    if memo is not None and memo[0] is ep:
        key = memo[1]
    else:
        key = digest(dict(ep))
        if cache is not None:
            cache[("id", id(ep))] = (ep, key)
    if cache is not None and key in cache:
        verdict, observations = cache[key]
    else:
        scenario = Scenario.from_dict(ep["scenario"])
        verdict = verify_replay(scenario, ep["seed"], records, partial=False)
        if not verdict.complete:
            raise ValueError(f"episode replay not complete: {verdict.reason} at step {verdict.step}")
        sim, observations = Simulator(scenario, ep["seed"]), []
        for rec in records:
            observations.append(sim.observe())
            sim.step(rec["actions"])
        if cache is not None:
            cache[key] = (verdict, observations)
    if canonical_json(trace["observation"]) != canonical_json(observations[step - 1]):   # type-strict
        raise ValueError(f"observation differs from the replayed observation before step {step}")
    if canonical_json(trace["actions"]) != canonical_json(records[step - 1]["actions"]):
        raise ValueError(f"actions differ from the replayed actions of step {step}")
    return {"status": "complete", "episode_sha256": key, "steps": verdict.steps_verified, "step": step,
            "termination_reason": verdict.termination_reason, "final_transition_hash": verdict.final_transition_hash}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


SHARED_EPISODE_FORMAT = "s1-shard-member@1.0.0"


def _traces(data: Any) -> Iterable[Mapping[str, Any]]:
    """A member holds one trace object, a list of them, or {"traces": [...]}.

    Shared-episode member (`SHARED_EPISODE_FORMAT`): {"member_format", "episode", "traces"}. The episode is
    stored once and attached to every trace; a trace in such a member must not carry its own episode. This
    keeps a shard linear in episode length (a per-trace episode copy is quadratic).
    """
    shared = None
    if isinstance(data, Mapping) and "member_format" in data:
        if data["member_format"] != SHARED_EPISODE_FORMAT or set(data) != {"member_format", "episode", "traces"} \
                or not isinstance(data["traces"], list):
            raise ValueError("unsupported or malformed shared-episode member")
        shared = data["episode"]
    if isinstance(data, Mapping) and isinstance(data.get("traces"), list):
        data = data["traces"]
    items = data if isinstance(data, list) else [data]
    for item in items:
        if isinstance(item, Mapping) and set(item) >= _TRACE_KEYS:
            if shared is not None:
                if "episode" in item:
                    raise ValueError("a trace in a shared-episode member must not carry its own episode")
                item = {**item, "episode": shared}
            yield item


def ingest_zip(zip_path: Path, *, source_id: str, quality: str,
               manifests: Mapping[str, Mapping[str, Any]] | None = None,
               allowed_splits: tuple[str, ...] = ("train",)) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate every teacher trace in a zip. Return (records, manifest entry).

    Fail closed: an invalid trace raises; it is never repaired or skipped.
    JSON members without trace keys (e.g. a bare observation) are counted only.
    With `manifests` (split -> verified split manifest), every `teacher_verified`
    trace must also carry a `provenance` record (and `values` when its teacher
    computes them) that passes `provenance.verify_provenance` (issue #10).
    """
    if quality not in QUALITIES:
        raise ValueError(f"quality must be one of {QUALITIES}")
    if not source_id or "/" in source_id:
        raise ValueError("source_id must be nonempty and contain no '/'")
    records: list[dict[str, Any]] = []
    other_json = 0
    replay_cache: dict[str, Any] = {}
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
        if len(infos) > _MAX_MEMBERS:
            raise ValueError(f"too many zip members: {len(infos)} > {_MAX_MEMBERS}")
        if sum(i.file_size for i in infos) > _MAX_TOTAL_BYTES:
            raise ValueError("zip expands beyond the total size bound")
        for info in sorted(infos, key=lambda i: i.filename):
            name = info.filename
            if name.startswith("/") or "\\" in name or ":" in name or ".." in name.split("/"):
                raise ValueError(f"unsafe member path: {name!r}")
            if info.is_dir() or not name.lower().endswith((".json", ".jsonl")):
                continue
            if info.file_size > _MAX_MEMBER_BYTES:
                raise ValueError(f"member too large: {info.filename}")
            if info.file_size > _MAX_RATIO * max(info.compress_size, 1) and info.file_size > 2**20:
                raise ValueError(f"member compression ratio too high: {info.filename}")
            raw = zf.read(info)
            text = raw.decode("utf-8")
            docs = ([json.loads(line) for line in text.splitlines() if line.strip()]
                    if info.filename.lower().endswith(".jsonl") else [json.loads(text)])
            found = 0
            for doc in docs:
                for i, trace in enumerate(_traces(doc)):
                    obs = validate_observation(trace["observation"])
                    actions = validate_actions(obs, trace["actions"])
                    if not isinstance(trace["group_id"], str) or not trace["group_id"]:
                        raise ValueError(f"{info.filename}: nonempty group_id required")
                    if not isinstance(trace["source"], str) or not trace["source"]:
                        raise ValueError(f"{info.filename}: teacher source/version required")
                    rec = {
                        "source_id": source_id, "quality": quality,
                        "member": info.filename, "member_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
                        "index": i,
                        # Prefix keeps lineage unique across sources for group splits.
                        "group_id": f"{source_id}/{trace['group_id']}",
                        "teacher": trace["source"], "observation": obs, "actions": actions,
                    }
                    if quality == "teacher_verified":
                        try:
                            rec["replay"] = verify_teacher_decision(trace, replay_cache)
                            if manifests is not None:
                                rec["provenance"] = _verify_trace_provenance(trace, manifests, allowed_splits)
                                rec["provenance_record"] = trace["provenance"]
                                rec["values"] = trace.get("values")
                        except ValueError as e:
                            raise ValueError(f"{info.filename}[{i}]: {e}") from None
                    records.append(rec)
                    found += 1
            other_json += found == 0
    manifest = {
        "source_id": source_id, "quality": quality, "path": str(zip_path.name),
        "sha256": sha256_file(zip_path), "traces": len(records),
        "decisions": sum(len(r["actions"]) for r in records),
        "groups": sorted({r["group_id"] for r in records}),
        "non_trace_json_members": other_json,
        "training_eligible": quality in TRAINING_QUALITIES,      # quality tier only (legacy meaning)
        "provenance_verified": manifests is not None and quality == "teacher_verified",
        "s1_eligible": False,     # only ember_v1.s1.build_s1_training_set can authorize S1 (issue #10)
    }
    return records, manifest


def _verify_trace_provenance(trace: Mapping[str, Any], manifests: Mapping[str, Mapping[str, Any]],
                             allowed_splits: tuple[str, ...]) -> dict[str, Any]:
    from . import provenance                      # lazy: provenance imports the teacher package
    if not isinstance(trace.get("provenance"), Mapping):
        raise ValueError("provenance record required")
    scenario_hash = Scenario.from_dict(trace["episode"]["scenario"]).content_hash()
    return provenance.verify_provenance(trace["provenance"], observation=trace["observation"],
                                        actions=trace["actions"], values=trace.get("values"),
                                        scenario_hash=scenario_hash, manifests=manifests,
                                        allowed_splits=allowed_splits)


def training_records(records: Iterable[Mapping[str, Any]], *, allow_fixtures: bool = False,
                     require_provenance: bool = False) -> list[Mapping[str, Any]]:
    """Legacy filter to quality-eligible records. `allow_fixtures` is for pipeline smoke only.

    This is NOT an S1 entrypoint and its output never authorizes bulk S1:
    use `ember_v1.s1.build_s1_training_set`, which has no opt-out.

    A `teacher_verified` record without complete full-replay evidence raises (fail closed).
    With `require_provenance` (required for S1 training, issue #10), a `teacher_verified`
    record without verified provenance raises too, and fixtures are never allowed.
    """
    if require_provenance and allow_fixtures:
        raise ValueError("fixtures are never allowed together with require_provenance")
    allowed = TRAINING_QUALITIES | ({"synthetic_fixture"} if allow_fixtures else set())
    out = [r for r in records if r["quality"] in allowed]
    for r in out:
        if r["quality"] == "teacher_verified" and (r.get("replay") or {}).get("status") != "complete":
            raise ValueError("teacher_verified record lacks complete full-episode replay evidence")
        if require_provenance and r["quality"] == "teacher_verified" and \
                (r.get("provenance") or {}).get("status") != "verified":
            raise ValueError("teacher_verified record lacks verified teacher provenance")
    return out
