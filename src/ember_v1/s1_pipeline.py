"""S1 distillation dataset pipeline for the selected teacher (issue #10). Fail closed.

This module PRODUCES S1 shards. It never authorizes training: `s1.build_s1_training_set` stays the only S1
entrypoint, and the pipeline runs it on every shard it writes (end-to-end gate).

Order of work (`build_shard`):

1 Receipt gate first. The frozen protocol and the selection receipt are read from the trusted git history
  (`trust.load_trusted_json`) and verified by `s1.verify_receipt`. A pending, unreviewed or forged receipt
  stops the run before any teacher call: `BULK_S1_BLOCKED`, no label is made.
2 Teacher pin. The teacher is the receipt's selected identity, built by `provenance.frozen_teacher` from the
  registry; identity drift (`registry.check_pins`, `check_registry`) stops the run. Only seedless teachers
  (T1 v1.0.0) are supported here.
3 Scenario plan. TRAIN manifest only. Every TRAIN family, lineage, scenario and core geometry must be disjoint
  from every held-out set (DEV, calibration, the receipt's selection-confirmation set and any extra set, for
  example locked_test and the pilot namespaces). Any overlap stops the run; nothing is dropped silently.
  Each supplied scenario must hash to its manifest entry.
4 Labelling. One episode per scenario, to termination. Every decision gets a `provenance.provenance_record`.
  The recorded actions are the teacher's real actions; no probability, score or value is written (T1 is
  action-only, `registry.VALUE_TEACHERS`). The episode is then replayed in full (`verify_replay(partial=False)`).
5 Separation. A failed episode (teacher exception, replay failure) gives NO training trace, only a diagnostic
  record. Every episode, including ones with casualties or unresolved people ("negative" episodes), gets a
  diagnostic record with its outcome and every casualty event. Training traces are not filtered by outcome.
6 Shard. Deterministic zip (fixed member order and timestamps) of shared-episode members
  (`sources.SHARED_EPISODE_FORMAT`), a diagnostics JSONL file and a hash-bound shard manifest with every
  source/config hash, quality statistics and storage figures.
7 End-to-end gate. `s1.build_s1_training_set` ingests the shard (full replay + provenance), runs the
  regeneration spot check and the leakage audit; its decision is written to the shard manifest.

No people or actions are truncated anywhere: a member above the ingestion bound stops the run.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import statistics
import sys
import zipfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import provenance as pv
from . import s1, sources, trust
from .sim import Scenario, Simulator, verify_replay
from .sim.canonical import canonical_json, digest
from .teacher import registry as reg

PIPELINE_VERSION = "s1-pipeline@1.0.0"
SHARD_MANIFEST_VERSION = "s1-shard-manifest@1.0.0"
DIAG_VERSION = "s1-diagnostic@1.0.0"
ZIP_DATE = (1980, 1, 1, 0, 0, 0)
HELD_OUT_KEYS = ("family", "lineage", "scenario", "core")


class S1PipelineError(RuntimeError):
    """The pipeline is blocked. `report` holds every reason. No shard is written."""

    def __init__(self, report: Mapping[str, Any]):
        super().__init__("BULK_S1_BLOCKED: " + "; ".join(report.get("reasons", [])))
        self.report = dict(report)


# ----- fingerprints ---------------------------------------------------------------------------------------
def _exact_module_path(mod: str) -> Path | None:
    """`provenance._module_path`, but every path part must match the directory listing exactly. On a
    case-insensitive file system (Windows) `from .sim import Scenario` otherwise resolves the phantom module
    `ember_v1.sim.Scenario` to `sim/scenario.py`, and the digest differs from Linux."""
    path = pv._module_path(mod)
    if path is None:
        return None
    d = pv._PKG
    for part in path.relative_to(pv._PKG).parts:
        if part not in {p.name for p in d.iterdir()}:
            return None
        d = d / part
    return path


def module_fingerprint(module: str) -> str:
    """Digest of one `ember_v1` module and its transitive `ember_v1` imports (same rule as teacher fingerprints;
    module paths are case-exact, so the digest is the same on Windows and Linux)."""
    seen: dict[str, str] = {}
    todo = [module]
    while todo:
        mod = todo.pop()
        path = _exact_module_path(mod)
        if mod in seen or path is None:
            continue
        seen[mod] = "sha256:" + hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        todo.extend(pv._imports(mod, path) - set(seen))
    return digest(dict(sorted(seen.items())))


# ----- held-out sets and the scenario plan ----------------------------------------------------------------
def heldout_from_manifest(m: Mapping[str, Any]) -> dict[str, set]:
    sc = m.get("scenarios", ())
    return {"family": {s["family_id"] for s in sc}, "lineage": {s["lineage_id"] for s in sc},
            "scenario": {s["scenario_hash"] for s in sc}, "core": {s["core_hash"] for s in sc if s.get("core_hash")}}


def heldout_from_entries(entries: Iterable[Mapping[str, Any]]) -> dict[str, set]:
    entries = list(entries)
    return {"family": {e["family_id"] for e in entries}, "lineage": {e["lineage_id"] for e in entries},
            "scenario": {e["scenario_hash"] for e in entries},
            "core": {e["core_hash"] for e in entries if e.get("core_hash")}}


def heldout_from_confirmation(conf: Mapping[str, Any], manifest: Mapping[str, Any] | None = None) -> dict[str, set]:
    """Receipt confirmation lists; core hashes too when the confirmation manifest (same hash) is given."""
    out = {"family": set(conf["family_ids"]), "lineage": set(conf["lineage_ids"]),
           "scenario": set(conf["scenario_hashes"]), "core": set()}
    if manifest is not None:
        if manifest.get("manifest_hash") != conf["manifest_hash"] or \
                digest({k: v for k, v in manifest.items() if k != "manifest_hash"}) != manifest["manifest_hash"]:
            raise ValueError("confirmation manifest differs from the receipt's manifest hash")
        out["core"] = {s["core_hash"] for s in manifest["scenarios"] if s.get("core_hash")}
    return out


def scenario_plan(train: Mapping[str, Any], heldout: Mapping[str, Mapping[str, set]], *,
                  scenario_ids: Sequence[str] | None = None) -> dict[str, Any]:
    """TRAIN entries to label and the overlap audit. Raise ValueError on any overlap (fail closed)."""
    trust.check_manifest(train)
    if train.get("split") != "train":
        raise ValueError(f"only the train split is labelled, not {train.get('split')!r}")
    entries = list(train["scenarios"])
    if scenario_ids is not None:
        by_id = {e["scenario_id"]: e for e in entries}
        missing = sorted(set(scenario_ids) - set(by_id))
        if missing:
            raise ValueError(f"scenarios not in the train manifest: {missing[:5]}")
        entries = [by_id[s] for s in sorted(set(scenario_ids))]
    if not entries:
        raise ValueError("no train scenarios to label")
    if not heldout or any(set(h) != set(HELD_OUT_KEYS) for h in heldout.values()):
        raise ValueError(f"held-out sets need keys {HELD_OUT_KEYS}")
    # The whole TRAIN manifest is audited, not only the selected entries: a family shared with a held-out set
    # taints every lineage of it.
    mine = heldout_from_manifest(train)
    overlap = {name: {k: sorted(mine[k] & h[k]) for k in HELD_OUT_KEYS} for name, h in heldout.items()}
    ok = not any(v for o in overlap.values() for v in o.values())
    audit = {"heldout_sets": sorted(heldout), "train_counts": {k: len(mine[k]) for k in HELD_OUT_KEYS},
             "heldout_counts": {n: {k: len(h[k]) for k in HELD_OUT_KEYS} for n, h in heldout.items()},
             "overlap": overlap, "pass": ok}
    if not ok:
        raise ValueError(f"train manifest overlaps held-out sets: {json.dumps(overlap, sort_keys=True)[:400]}")
    return {"entries": entries, "audit": audit}


# ----- labelling ------------------------------------------------------------------------------------------
def label_episode(policy, scenario: Scenario, entry: Mapping[str, Any], manifest: Mapping[str, Any]
                  ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """(shared-episode member or None, diagnostic record). Never raises for a teacher or replay failure."""
    sim, records, steps = Simulator(scenario), [], []
    ident = policy.identity
    diag: dict[str, Any] = {"diag_version": DIAG_VERSION,
                            "scenario": {k: entry[k] for k in ("scenario_id", "scenario_hash", "lineage_id",
                                                               "family_id")},
                            "teacher": {"name": ident.name, "version": ident.version, "config_hash": ident.config_hash},
                            "people": len(scenario.people)}
    cpu_ms: list[float] = []
    try:
        while not sim.terminated:
            obs = sim.observe()
            d = policy.decide(obs)
            if pv.decision_values(ident.name, d) is not None:
                raise RuntimeError("an action-only pipeline received per-action values")
            prov = pv.provenance_record(policy, d, manifest=manifest, scenario_id=entry["scenario_id"])
            steps.append((obs, d.actions, prov))
            cpu_ms.append(float(d.runtime.get("cpu_ms", 0.0)))
            records.append(sim.step(d.actions))
    except Exception as e:                    # noqa: BLE001 - a failure becomes a diagnostic, never a label
        diag.update(status="teacher_error", error=f"{type(e).__name__}: {e}", decisions=len(steps))
        return None, diag
    verdict = verify_replay(scenario, None, records, partial=False)
    outcome = _outcome(scenario, records)
    diag.update(outcome=outcome, decisions=len(steps), person_decisions=sum(len(a) for _, a, _ in steps),
                negative=outcome["casualties"] > 0 or outcome["unresolved"] > 0,
                movement_outcomes=dict(sorted(Counter(m["outcome"] for r in records
                                                      for m in r["movements"].values()).items())),
                replay={"complete": verdict.complete, "steps": verdict.steps_verified,
                        "reason": None if verdict.complete else verdict.reason,
                        "final_transition_hash": verdict.final_transition_hash},
                runtime={"cpu_ms_total": round(sum(cpu_ms), 3), "cpu_ms_max": max(cpu_ms, default=0.0)})
    if not verdict.complete:
        diag["status"] = "replay_failure"
        return None, diag
    diag["status"] = "labelled"
    group = f"{entry['family_id']}/{entry['lineage_id']}"
    source = f"{ident.name}@{ident.version}"
    traces = [{"observation": o, "actions": a, "group_id": group, "source": source, "step": k + 1, "provenance": p}
              for k, (o, a, p) in enumerate(steps)]
    member = {"member_format": sources.SHARED_EPISODE_FORMAT,
              "episode": {"scenario": scenario.to_dict(), "seed": None, "records": records}, "traces": traces}
    return member, diag


def _outcome(scenario: Scenario, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    last_action: dict[str, str] = {}
    events = []
    evac = 0
    for r in records:
        last_action.update(r["actions"])
        evac += len(r["evacuations"])
        events.extend({"person": pid, "step": r["step"], "cause": cause, "last_action": r["actions"][pid]}
                      for pid, cause in r["casualties"].items())
    people = len(scenario.people)
    return {"people": people, "evacuated": evac, "casualties": len(events), "unresolved": people - evac - len(events),
            "steps": len(records), "termination_reason": records[-1]["termination_reason"] if records else None,
            "casualty_events": events}


def diag_hash(d: Mapping[str, Any]) -> str:
    """Digest of a diagnostic record (deterministic: host timing is never in it)."""
    return digest({k: v for k, v in d.items() if k != "diag_hash"})


# ----- shard files ----------------------------------------------------------------------------------------
def _member_name(i: int, entry: Mapping[str, Any]) -> str:
    return f"episodes/{i:05d}-{entry['scenario_id']}.json"


def write_zip(path: Path, members: Sequence[tuple[str, bytes]]) -> str:
    """Deterministic zip: given order, fixed timestamps, deflate. Return its sha256."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name, raw in members:
            info = zipfile.ZipInfo(name, date_time=ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, raw, compresslevel=6)
    data = buf.getvalue()
    path.write_bytes(data)
    return "sha256:" + hashlib.sha256(data).hexdigest()


def quality_report(members: Sequence[Mapping[str, Any]], diags: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    acts = Counter(a for m in members for t in m["traces"] for a in t["actions"].values())
    per_dec = [len(t["actions"]) for m in members for t in m["traces"]]
    status = Counter(d["status"] for d in diags)
    labelled = [d for d in diags if d["status"] == "labelled"]
    return {
        "episodes": len(diags), "episode_status": dict(sorted(status.items())),
        "decisions": len(per_dec), "person_decisions": sum(per_dec),
        "action_counts": dict(sorted(acts.items())),
        "action_fractions": {a: round(c / max(sum(acts.values()), 1), 6) for a, c in sorted(acts.items())},
        "active_people_per_decision": {"min": min(per_dec, default=0), "max": max(per_dec, default=0),
                                       "mean": round(statistics.fmean(per_dec), 3) if per_dec else 0.0},
        "negative_episodes": sum(d["negative"] for d in labelled),
        "outcome_totals": {k: sum(d["outcome"][k] for d in labelled)
                           for k in ("people", "evacuated", "casualties", "unresolved")},
        "termination": dict(sorted(Counter(d["outcome"]["termination_reason"] for d in labelled).items())),
        "values_written": 0, "probabilities_written": 0,
    }


def build_shard(out_dir: str | Path, *, shard_id: str, repo: str | Path, protocol_ref: Mapping[str, Any],
                receipt_ref: Mapping[str, Any], manifests: Mapping[str, Mapping[str, Any]],
                scenarios: Iterable[Scenario], extra_heldout: Mapping[str, Mapping[str, set]] | None = None,
                confirmation_manifest: Mapping[str, Any] | None = None, scenario_ids: Sequence[str] | None = None,
                max_episodes: int, trusted_ref: str = "origin/main") -> dict[str, Any]:
    """Label TRAIN scenarios with the receipt's teacher and write one verified shard. Return the shard manifest.

    `max_episodes` is a required guard against an accidental bulk run: a plan larger than it stops the run.
    """
    reasons: list[str] = []
    head: dict[str, Any] = {"pipeline_version": PIPELINE_VERSION, "shard_id": shard_id, "trusted_ref": trusted_ref}

    def blocked(msg: str) -> S1PipelineError:
        reasons.append(msg)
        return S1PipelineError({**head, "decision": "BULK_S1_BLOCKED", "reasons": reasons})

    if not shard_id or "/" in shard_id or "\\" in shard_id:
        raise blocked("shard_id must be nonempty and contain no path separator")
    # 0 research-direction gate (2026-11-09, #45/#47): the project repository may not make single-teacher
    # S1 labels, even after its T1 receipt is approved. Multi-teacher data uses the P1-12/P1-14 gates.
    if Path(repo).resolve() == reg.REPO.resolve():
        raise blocked("single-teacher S1 labelling of the project repository is superseded by the "
                      "five-teacher direction (#45/#47)")
    # 1 receipt gate (before any teacher call)
    try:
        protocol, p_trust = trust.load_trusted_json(repo, protocol_ref, trusted_ref=trusted_ref)
        receipt, r_trust = trust.load_trusted_json(repo, receipt_ref, trusted_ref=trusted_ref)
        sel = s1.verify_receipt(receipt, protocol, protocol_ref)
    except (ValueError, KeyError, TypeError, OSError) as e:
        raise blocked(f"selection receipt: {e}") from None
    head["trust"] = {"protocol": p_trust, "receipt": r_trust}
    # 2 teacher pin
    drift = reg.check_pins() + reg.check_registry()
    if drift:
        raise blocked(f"identity drift: {drift}")
    ident = sel["selected"]
    try:
        policy = pv.frozen_teacher(ident["name"], None, ident["version"])
    except pv.ProvenanceError as e:
        raise blocked(f"teacher: {e} (only seedless teachers are supported)") from None
    if pv.code_fingerprint(type(policy)) != ident["code_fingerprint"] or \
            policy.identity.config_hash != ident["config_hash"]:
        raise blocked("teacher code or config differs from the receipt")
    # 3 scenario plan
    try:
        for s in s1.REQUIRED_MANIFESTS:
            trust.check_manifest(manifests[s])
            if manifests[s].get("split") != s:
                raise ValueError(f"manifest for {s!r} is the {manifests[s].get('split')!r} split")
        if len({(manifests[s]["plan_hash"], manifests[s]["spec_hash"]) for s in s1.REQUIRED_MANIFESTS}) != 1:
            raise ValueError("manifests come from different split plans")
        heldout = {"dev": heldout_from_manifest(manifests["dev"]),
                   "calibration": heldout_from_manifest(manifests["calibration"]),
                   "selection_confirmation": heldout_from_confirmation(sel["confirmation"], confirmation_manifest),
                   **(extra_heldout or {})}
        plan = scenario_plan(manifests["train"], heldout, scenario_ids=scenario_ids)
    except (ValueError, KeyError, TypeError) as e:
        raise blocked(f"scenario plan: {e}") from None
    entries = plan["entries"]
    if len(entries) > max_episodes:
        raise blocked(f"plan has {len(entries)} episodes > max_episodes {max_episodes} (no silent truncation)")
    by_hash = {s.content_hash(): s for s in scenarios}
    missing = [e["scenario_id"] for e in entries if e["scenario_hash"] not in by_hash]
    if missing:
        raise blocked(f"no scenario with the manifest hash for {missing[:5]}")
    # 4-5 labelling
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    zip_path, diag_path, man_path = (out / f"{shard_id}.zip", out / f"{shard_id}.diagnostics.jsonl",
                                     out / f"{shard_id}.manifest.json")
    for p in (zip_path, diag_path, man_path):
        if p.exists():
            raise blocked(f"{p.name} exists (write-once)")
    zmembers, members, diags, runtime = [], [], [], []
    for i, e in enumerate(entries):
        member, diag = label_episode(policy, by_hash[e["scenario_hash"]], e, manifests["train"])
        runtime.append(diag.pop("runtime", None))          # host timing: kept out of every hashed file
        diag["diag_hash"] = diag_hash(diag)
        diags.append(diag)
        if member is None:
            continue
        raw = canonical_json(member).encode("utf-8")
        if len(raw) > sources._MAX_MEMBER_BYTES:
            raise blocked(f"episode {e['scenario_id']} member is {len(raw)} bytes > ingestion bound (no truncation)")
        zmembers.append((_member_name(i, e), raw))
        members.append(member)
    if not members:
        raise blocked("no episode was labelled")
    # 6 shard files
    zip_sha = write_zip(zip_path, zmembers)
    diag_text = "".join(canonical_json(d) + "\n" for d in diags)
    diag_path.write_text(diag_text, encoding="utf-8", newline="\n")
    # 7 end-to-end S1 gate on the written shard
    try:
        ts = s1.build_s1_training_set([(zip_path, shard_id)], repo=repo, protocol_ref=protocol_ref,
                                      receipt_ref=receipt_ref, manifests=manifests, trusted_ref=trusted_ref)
        gate = {k: ts.report[k] for k in ("decision", "reasons", "records", "spot_check", "dependencies")}
        gate["leakage_pass"] = ts.report["leakage"]["pass"]
    except s1.S1GateError as err:
        gate = {"decision": "BULK_S1_BLOCKED", "reasons": err.report.get("reasons", [])}
    uncompressed = sum(len(r) for _, r in zmembers)
    body = {
        "shard_manifest_version": SHARD_MANIFEST_VERSION, **head,
        "receipt": {"receipt_hash": sel["receipt_hash"], "decision": sel["decision"], "ref": dict(receipt_ref)},
        "protocol": {"protocol_hash": protocol["protocol_hash"], "ref": dict(protocol_ref)},
        "teacher": {**ident, "policy_seed": None, "values": "none (action-only)"},
        "code": {"pipeline_fingerprint": module_fingerprint(__name__),
                 "sources_fingerprint": module_fingerprint(sources.__name__),
                 "s1_gate_fingerprint": module_fingerprint(s1.__name__), "s1_gate_version": s1.S1_GATE_VERSION,
                 "provenance_version": pv.PROVENANCE_VERSION, "member_format": sources.SHARED_EPISODE_FORMAT},
        "split": {k: manifests["train"][k] for k in ("split", "plan_id", "plan_hash", "manifest_hash",
                                                     "generator_revision", "spec_hash")},
        "heldout_audit": plan["audit"],
        "episodes": [{"member": _member_name(i, e), "scenario_id": e["scenario_id"],
                      "scenario_hash": e["scenario_hash"], "status": d["status"], "diag_hash": d["diag_hash"]}
                     for i, (e, d) in enumerate(zip(entries, diags))],
        "files": {"zip": {"name": zip_path.name, "sha256": zip_sha, "bytes": zip_path.stat().st_size,
                          "uncompressed_bytes": uncompressed},
                  "diagnostics": {"name": diag_path.name, "sha256": trust.sha256_lf(diag_text.encode("utf-8")),
                                  "records": len(diags)}},
        "quality": quality_report(members, diags),
        "s1_gate": gate,
    }
    cpu = [r["cpu_ms_total"] for r in runtime if r]
    man = {**body, "manifest_hash": digest(body),
           "runtime": {"note": "host timing, excluded from manifest_hash", "teacher_cpu_s": round(sum(cpu) / 1e3, 3),
                       "max_decision_cpu_ms": max((r["cpu_ms_max"] for r in runtime if r), default=0.0)}}
    man_path.write_text(json.dumps(man, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return man


def verify_shard(manifest_path: str | Path) -> dict[str, Any]:
    """Re-hash a shard and its diagnostics against the shard manifest. Raise ValueError on any difference."""
    mp = Path(manifest_path)
    man = json.loads(mp.read_text(encoding="utf-8"))
    if man.get("shard_manifest_version") != SHARD_MANIFEST_VERSION:
        raise ValueError("unsupported shard manifest version")
    if digest({k: v for k, v in man.items() if k not in ("manifest_hash", "runtime")}) != man.get("manifest_hash"):
        raise ValueError("shard manifest hash does not match its content")
    f = man["files"]
    zp = mp.parent / f["zip"]["name"]
    if sources.sha256_file(zp) != f["zip"]["sha256"]:
        raise ValueError("shard zip differs from its manifest")
    text = (mp.parent / f["diagnostics"]["name"]).read_bytes()
    if trust.sha256_lf(text) != f["diagnostics"]["sha256"]:
        raise ValueError("diagnostics differ from the manifest")
    diags = [json.loads(line) for line in text.decode("utf-8").splitlines() if line.strip()]
    if [d["diag_hash"] for d in diags] != [e["diag_hash"] for e in man["episodes"]] or \
            any(diag_hash(d) != d["diag_hash"] for d in diags):
        raise ValueError("diagnostic records differ from the manifest")
    with zipfile.ZipFile(zp) as zf:
        names = sorted(zf.namelist())
    want = sorted(e["member"] for e in man["episodes"] if e["status"] == "labelled")
    if names != want:
        raise ValueError("shard members differ from the labelled episodes")
    return {"status": "verified", "manifest_hash": man["manifest_hash"], "members": len(names),
            "s1_gate": man["s1_gate"]["decision"]}


# ----- storage estimate -----------------------------------------------------------------------------------
def storage_estimate(samples: Sequence[Mapping[str, Any]], train: Mapping[str, Any]) -> dict[str, Any]:
    """Upper and central storage estimate for a TRAIN manifest from measured synthetic episodes.

    `samples`: per episode {people, walls, steps, max_steps, member_bytes, zip_bytes}. Model (least squares):
    bytes per step = a + b * people + c * walls (each observation repeats the map and every person). The model
    is scaled by the largest sample over-prediction ratio, so every sample is at or below it. Upper bound: every
    person stays active for max_steps. Central: the mean sample steps / max_steps fraction.
    """
    if len(samples) < 3:
        raise ValueError("need at least 3 samples")
    X = [(1.0, float(s["people"]), float(s["walls"])) for s in samples]

    def fit(key: str) -> tuple[list[float], float]:
        y = [s[key] / s["steps"] for s in samples]
        coef = _ols3(X, y)
        pred = [max(sum(c * x for c, x in zip(coef, row)), 1e-9) for row in X]
        return coef, max(1.0, max(v / p for v, p in zip(y, pred)))

    (c_raw, k_raw), (c_zip, k_zip) = fit("member_bytes"), fit("zip_bytes")
    per_step = lambda c, k, e: k * (c[0] + c[1] * e["people"] + c[2] * e["walls"])  # noqa: E731
    frac = statistics.fmean(s["steps"] / s["max_steps"] for s in samples)
    sc = [e["diagnostics"] for e in train["scenarios"]]
    up_raw = [per_step(c_raw, k_raw, e) * e["max_steps"] for e in sc]
    up_zip = [per_step(c_zip, k_zip, e) * e["max_steps"] for e in sc]
    big = max(up_raw)
    return {"samples": len(samples), "model": "bytes/step = a + b*people + c*walls, scaled to cover every sample",
            "coef_uncompressed": [round(c, 3) for c in c_raw], "cover_ratio_uncompressed": round(k_raw, 4),
            "coef_zip": [round(c, 3) for c in c_zip], "cover_ratio_zip": round(k_zip, 4),
            "mean_steps_fraction_of_max": round(frac, 4), "train_scenarios": len(sc),
            "decisions": {"upper": sum(e["max_steps"] for e in sc),
                          "central": sum(math.ceil(frac * e["max_steps"]) for e in sc)},
            "uncompressed_bytes": {"upper": round(sum(up_raw)), "central": round(frac * sum(up_raw))},
            "zip_bytes": {"upper": round(sum(up_zip)), "central": round(frac * sum(up_zip))},
            "largest_member_upper_bytes": round(big), "member_bound_bytes": sources._MAX_MEMBER_BYTES,
            "largest_member_fits": big <= sources._MAX_MEMBER_BYTES,
            "per_trace_episode_format_largest_member_upper_bytes": round(big * max(e["max_steps"] for e in sc))}


def _ols3(X: Sequence[Sequence[float]], y: Sequence[float]) -> list[float]:
    """Least squares for 3 coefficients (normal equations, Gaussian elimination with pivoting)."""
    A = [[sum(r[i] * r[j] for r in X) for j in range(3)] + [sum(r[i] * v for r, v in zip(X, y))] for i in range(3)]
    for i in range(3):
        p = max(range(i, 3), key=lambda r: abs(A[r][i]))
        A[i], A[p] = A[p], A[i]
        if abs(A[i][i]) < 1e-12:
            raise ValueError("samples do not determine the model (vary people and walls)")
        for r in range(3):
            if r != i:
                f = A[r][i] / A[i][i]
                A[r] = [a - f * b for a, b in zip(A[r], A[i])]
    return [A[i][3] / A[i][i] for i in range(3)]


# ----- status (real repository) ---------------------------------------------------------------------------
RECEIPT_CANDIDATES = ("data/selection/p1-09-selection@1/receipt.json",
                      "data/selection/p1-09-selection@1/outcome/receipt_DRAFT_PENDING_REVIEW.json")
PROTOCOL_REF = {"kind": "git", "commit": "64b2d9873953ba750fd6c08fb84761f1bcd8e0bb",
                "path": "data/selection/p1-09-selection@1/protocol.json"}


def status(repo: str | Path, trusted_ref: str = "origin/main") -> dict[str, Any]:
    """Receipt gate of the real repository at `trusted_ref`. Never labels anything."""
    import subprocess
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", trusted_ref], capture_output=True, text=True)
    commit = head.stdout.strip()
    out = {"pipeline_version": PIPELINE_VERSION, "trusted_ref": trusted_ref, "commit": commit, "receipts": {}}
    for path in RECEIPT_CANDIDATES:
        try:
            protocol, _ = trust.load_trusted_json(repo, PROTOCOL_REF, trusted_ref=trusted_ref)
            receipt, _ = trust.load_trusted_json(repo, {"kind": "git", "commit": commit, "path": path},
                                                 trusted_ref=trusted_ref)
            s1.verify_receipt(receipt, protocol, PROTOCOL_REF)
            out["receipts"][path] = "verified"
        except (ValueError, KeyError, TypeError, OSError) as e:
            out["receipts"][path] = f"refused: {e}"
    ok = any(v == "verified" for v in out["receipts"].values())
    out["decision"] = "RECEIPT_GATE_PASSED_OTHER_GATES_PENDING" if ok else "BULK_S1_BLOCKED"
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ember_v1.s1_pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("status", help="receipt gate of the real repository (labels nothing)")
    st.add_argument("--repo", default=".")
    st.add_argument("--trusted-ref", default="origin/main")
    args = ap.parse_args(argv)
    rep = status(args.repo, args.trusted_ref)
    sys.stdout.write(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    return 0 if rep["decision"] == "BULK_S1_BLOCKED" else 3


if __name__ == "__main__":
    raise SystemExit(main())
