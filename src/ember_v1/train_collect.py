"""P1-17 full-TRAIN five-teacher collection: 189 matched groups x T0-T4, parallel, crash-safe, resumable.

Reuses the P1-12 pilot machinery (`train_pilot` gates and audits, `pilot_supervisor` killable workers) and adds:

- a plan over EVERY TRAIN scenario (no selection), ordered longest-expected-first (LPT) for parallel groups;
- a host-bound evidence root OUTSIDE every checkout, named in the plan and the authorization, so a fresh
  clone sees the same attempt markers (the P1-12 rerun bypass cannot happen again);
- one group-runner process per matched group (it spawns the five teacher processes), at most
  `max_concurrent_groups` at once, admitted only while host free RAM >= `min_free_ram_gib_to_start`;
- write-once per-group checkpoints: `started/<sid>.json` BEFORE any teacher call, then `groups/<sid>.json`
  and `rows/<sid>.json`. A late crash loses at most the groups that were running;
- sessions: session 1 needs an empty root; a resume (session n <= max_sessions) re-verifies finished groups,
  records every started-but-unfinished group as `interrupted/<sid>.json` (terminal, never rerun: a retry needs
  a new authorization) and runs only groups that never started;
- a read-only `verify` command (full replay of every stored group; the completion-triggered verifier).

Gates (`run_preconditions`) are the pilot's: FROZEN plan from a trusted commit equal to a live rebuild, all
189 scenarios regenerate with their hashes, exact teacher pins, a single-owner authorization merged on GitHub
naming plan hash, scenarios, fingerprints, limits, concurrency, storage root and execution commit, clean HEAD
on the trusted ref. Docs: docs/P1_17_TRAIN_COLLECTION.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import s1_pipeline as sp
from . import train_pilot as tp
from . import trust
from .sim.canonical import digest
from .teacher.registry import CANDIDATES

VERSION = "p1-17-train-full@1.0.0"
NAMESPACE = "p1-17-train-full@1"
SALT = NAMESPACE
ATTEMPT = 1
PLAN_STATUS, FROZEN = "PROPOSED_NOT_FROZEN", "FROZEN"
AUTH_KIND = "p1-17-train-full-collection-authorization"
AUTH_VERSION = "1.0.0"
SESSION_VERSION = "p1-17-train-full-session@1.0.0"
EVIDENCE_VERSION = "p1-17-train-full-evidence@1.0.0"
OUT_DIR = f"data/collection/{NAMESPACE}"
PLAN_FILE = f"{OUT_DIR}/plan_PROPOSED.json"
FROZEN_PLAN_FILE = f"{OUT_DIR}/plan.json"
AUTH_FILE = f"{OUT_DIR}/authorization.json"
AUTH_TEMPLATE_FILE = f"{OUT_DIR}/authorization_TEMPLATE.json"
CALIBRATION_FILE = f"{OUT_DIR}/calibration.json"
PILOT_SUMMARY_FILE = "data/collection/p1-12-train-pilot@1/result/attempt-1-summary.json"
STORAGE_ROOT = "D:/Coding/ember-evidence/p1-17-train-full@1"      # host-bound; outside every checkout
TRAIN_FILE = "data/splits/p1-02-v1/train.json"
SCENARIOS = 189
REGEN_PER_BIN = 2
N_BINS = tp.N_BINS
POLL_S = 2.0
STAGGER_S = 1.0
PROGRESS_EVERY_S = 30.0

# Fixed protocol limits; `max_concurrent_groups` comes from the committed calibration.
BASE_LIMITS = {"decision_cpu_s": dict(tp.LIMITS["decision_cpu_s"]), "group_wall_s": 7200,
               "session_wall_s": 129600, "max_sessions": 4, "processes_per_teacher": 1, "retries": 0,
               "min_free_ram_gib_to_start": 2.0,
               "decisions_per_teacher": tp.LIMITS["decisions_per_teacher"],
               "process_model": "one group-runner process per matched group, which spawns one killable process "
                                "per teacher; at most max_concurrent_groups runners at once"}
FAILURE_POLICY = {
    "teacher_failures": list(tp.FAILURE_POLICY["teacher_failures"]),
    "effect": "the matched group is incomplete: kept as diagnostic evidence with every streamed decision, "
              "never retried, never replaced, never training-eligible",
    "integrity_error": "a worker stream that does not replay is a collector fault: recorded integrity_error",
    "runner_crashed": "a group runner that exits without a row: recorded runner_crashed, kept, never rerun",
    "interrupted": "a group that started (started/<sid>.json) but has no row when a later session opens is "
                   "recorded interrupted/<sid>.json; it is never rerun in this attempt",
    "not_started": "groups that never started (stop request, session wall cap, T2 CPU total cap) may run in a "
                   "later session of the same attempt, at most max_sessions sessions",
    "retry": "a retry of an incomplete, crashed or interrupted group needs a NEW authorization that names the "
             "group ids; this plan version allows none",
    "substitutes": "none", "fresh_clone": "the evidence root is outside the checkout and bound by the plan; a "
                                          "fresh clone sees the same markers and is refused"}
PILOT_OVERLAP = {
    "decision": "fresh_collection",
    "rule": "all 189 TRAIN scenarios, including the 8 P1-12 pilot scenarios, are collected in this namespace "
            "under this plan; pilot records stay in p1-12-train-pilot@1, immutable and diagnostic only; they are "
            "never counted, merged or promoted to training data. The 8 overlapping groups are compared with the "
            "pilot as an extra cross-run regeneration check (per-teacher final transition hashes).",
    "not_a_retry": "this is a new, separately authorized experiment (attempt 1 of p1-17-train-full@1), not "
                   "attempt 2 of the pilot"}


class CollectionBlocked(RuntimeError):
    """The full-TRAIN collection is not authorized, a gate failed or the evidence root refuses this session."""


# ----- host resources -----------------------------------------------------------------------------------------
def host_free_ram_gib() -> float | None:
    try:
        if sys.platform == "win32":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return None
            return m.ullAvailPhys / 2 ** 30
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 2 ** 20
    except (OSError, ValueError, AttributeError):
        return None
    return None


def peak_rss_mb(pid: int) -> float | None:
    """Peak working set (Windows) or VmHWM (Linux) of a live process in MiB, or None if unreadable."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.OpenProcess.restype = wintypes.HANDLE
            k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            k32.K32GetProcessMemoryInfo.argtypes = (wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD)
            k32.CloseHandle.argtypes = (wintypes.HANDLE,)
            h = k32.OpenProcess(0x1000, False, pid)
            if not h:
                return None
            try:
                c = PMC()
                c.cb = ctypes.sizeof(PMC)
                if not k32.K32GetProcessMemoryInfo(h, ctypes.byref(c), c.cb):
                    return None
                return c.PeakWorkingSetSize / 2 ** 20
            finally:
                k32.CloseHandle(h)
        with open(f"/proc/{pid}/status", encoding="ascii") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024
    except (OSError, ValueError, AttributeError):
        return None
    return None


class _MemSampler:
    """Samples the peak RSS of known pids and the host free RAM once per second (resource evidence only)."""

    def __init__(self) -> None:
        self.pids: dict[str, int] = {}
        self.peak: dict[str, float] = {}
        self.host_min: float | None = None
        self._halt = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> _MemSampler:
        self._t.start()
        return self

    def sample(self) -> None:
        for n, pid in list(self.pids.items()):
            v = peak_rss_mb(pid)
            if v is not None:
                self.peak[n] = max(self.peak.get(n, 0.0), v)
        free = host_free_ram_gib()
        if free is not None:
            self.host_min = free if self.host_min is None else min(self.host_min, free)

    def _loop(self) -> None:
        while not self._halt.wait(1.0):
            self.sample()

    def stop(self) -> dict[str, Any]:
        self._halt.set()
        self._t.join(5)
        self.sample()
        return {"peak_rss_mb": {k: round(v, 1) for k, v in sorted(self.peak.items())},
                "host_min_free_gib": None if self.host_min is None else round(self.host_min, 3)}


# ----- write-once files ---------------------------------------------------------------------------------------
write_once = tp.write_once


def _read(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _replace_json(p: Path, doc: Mapping[str, Any]) -> None:
    """Status files only (progress.json): atomic replace, never evidence."""
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, p)


def collection_paths(root: Path) -> dict[str, Path]:
    return {"root": root, "log": root / "log.jsonl", "started": root / "started", "groups": root / "groups",
            "rows": root / "rows", "interrupted": root / "interrupted", "regen_started": root / "regen_started",
            "regen": root / "regen", "regen_rows": root / "regen_rows", "logs": root / "logs",
            "evidence": root / "evidence.json", "progress": root / "progress.json", "stop": root / "STOP"}


def sessions(root: Path) -> list[dict[str, Any]]:
    found = [p for p in root.glob("session-*.json") if p.stem.count("-") == 1 and p.stem.split("-")[1].isdigit()]
    return [_read(p) for p in sorted(found, key=lambda p: int(p.stem.split("-")[1]))]


# ----- Phase 1: pilot result summary (immutable pilot records are only read) ---------------------------------
PILOT_PROVENANCE = {
    "executions": [
        {"execution": 1, "checkout": "D:/Coding/ember_v1-pilot3", "launched_local": "2026-11-10T12:22:32+05:30",
         "result": "interrupted: the operator closed the visible wrapper console at 12:42 IST during group 1 "
                   "(T2 running); 0 of 8 groups finished; postrun: incomplete",
         "archive": "D:/Coding/pilot-evidence-run16/attempt-1-killed/ (SHA256SUMS)",
         "attempt_marker": "attempt-1.json written (attempt 1 consumed)"},
        {"execution": 2, "checkout": "D:/Coding/ember_v1-pilot4 (fresh clone, same commit ee5455f)",
         "launched_local": "2026-11-10T12:46:20+05:30", "ended_local": "2026-11-10T13:47:10+05:30",
         "result": "8/8 groups complete and verified, 2/2 preregistered regenerations identical, accepted",
         "archive": "D:/Coding/pilot-evidence-run16b/ (ATTEMPT_SHA256SUMS)"}],
    "technical_integrity": "PASS for execution 2: every group replays (verify_fixture), leakage gate passed, "
                           "regenerations identical, caps held",
    "protocol_compliance": "DEVIATION: execution 2 ran attempt 1 again in a fresh clone, which the no-retry rule "
                           "(one attempt; attempt 2 needs a new authorization) did not allow. The owner chose it "
                           "after a warning (run 16). The fresh clone had no attempt marker, so the guard did not "
                           "fire: the guard was bypassed, not satisfied.",
    "consequence": "pilot evidence is feasibility and diagnostic evidence only; never promoted to training data; "
                   "P1-17 stores its markers outside every checkout so this bypass cannot recur"}


def pilot_summary(attempt_dir: Path) -> dict[str, Any]:
    """Recompute the pilot totals from the stored group files (read only)."""
    rows, tot, people = [], {t: {"casualties": 0, "evacuated": 0, "unresolved": 0} for t in CANDIDATES}, 0
    for f in sorted((attempt_dir / "groups").glob("*.json")):
        g = _read(f)
        o0 = g["policies"]["T0"]["outcome"]
        people += o0["people"]
        sup = g["runtime"]["supervision"]
        per = {}
        for t, p in g["policies"].items():
            o = p["outcome"]
            for k in tot[t]:
                tot[t][k] += o[k]
            per[t] = {"casualties": o["casualties"], "evacuated": o["evacuated"], "unresolved": o["unresolved"],
                      "steps": o["steps"], "final_transition_hash": p["final_transition_hash"]}
        rows.append({"scenario_id": f.stem, "people": o0["people"], "max_steps": o0["max_steps"],
                     "content_hash": g["content_hash"], "file_sha256": _sha256_file(f),
                     "group_complete": g["group_complete"], "wall_s": sup["wall_s"],
                     "process_cpu_s": sup["process_cpu_s"],
                     "budget_limited_decisions": sup["budget_limited_decisions"], "teachers": per})
    for t in tot.values():
        t["casualties_plus_unresolved"] = t["casualties"] + t["unresolved"]
    ev = _read(attempt_dir / "evidence.json")
    body = {"summary_version": "p1-12-train-pilot-summary@1.0.0", "plan_hash": ev["plan_hash"],
            "evidence_hash": ev["evidence_hash"], "accepted": ev["acceptance"]["pass"],
            "groups": rows, "people_total": people, "totals": tot,
            "t2_cpu_s_total": round(sum(r["process_cpu_s"]["T2"] for r in rows), 3),
            "provenance": PILOT_PROVENANCE, "training_eligible": False,
            "note": "Eight cases support no population-level teacher claim."}
    return {**body, "summary_hash": digest(body)}


# ----- cost ---------------------------------------------------------------------------------------------------
def cost_basis(repo: Path) -> list[dict[str, Any]]:
    """Measured T2 CPU per scenario: P1-09 rows plus the P1-12 pilot groups (committed summary)."""
    out = [{**r, "source": "p1-09"} for r in tp.p1_09_cost_basis(repo)]
    p = repo / PILOT_SUMMARY_FILE
    if p.exists():
        for r in _read(p)["groups"]:
            out.append({"people": r["people"], "t2_cpu_s": r["process_cpu_s"]["T2"],
                        "budget_limited": r["budget_limited_decisions"]["T2"],
                        "others_cpu_s": sum(v for k, v in r["process_cpu_s"].items() if k != "T2"),
                        "source": "p1-12-pilot"})
    return out


def _bin(n: int) -> tuple[int, int]:
    return next(b for b in N_BINS if b[0] <= n < b[1])


def lpt_wall(costs: Sequence[float], workers: int) -> float:
    """Makespan of longest-first list scheduling on `workers` slots (costs in the given order)."""
    slots = [0.0] * max(1, workers)
    for c in costs:
        i = slots.index(min(slots))
        slots[i] += c
    return max(slots)


def cost_estimate(scenarios: Sequence[Mapping[str, Any]], basis: Sequence[Mapping[str, Any]],
                  workers: int) -> dict[str, Any]:
    bins: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for r in basis:
        bins.setdefault(_bin(r["people"]), []).append(r)
    per_bin, central, upper, mean_by = {}, [], [], {}
    for b, rows in sorted(bins.items()):
        cpu = [r["t2_cpu_s"] for r in rows]
        mean_by[b] = (statistics.fmean(cpu), max(cpu))
        per_bin[f"{b[0]}-{b[1]}"] = {"basis_scenarios": len(rows), "t2_cpu_s_mean": round(mean_by[b][0], 1),
                                     "t2_cpu_s_max": round(mean_by[b][1], 1),
                                     "budget_limited_share": round(sum(1 for r in rows if r["budget_limited"])
                                                                   / len(rows), 3),
                                     "train_scenarios": sum(1 for s in scenarios if _bin(s["people"]) == b)}
    for s in scenarios:
        m = mean_by.get(_bin(s["people"]))
        if m is None:
            raise ValueError(f"no measurement in the N bin of {s['scenario_id']}")
        central.append(m[0])
        upper.append(m[1])
    worst = sum(s["max_steps"] * tp.T2_BUDGET_S for s in scenarios)
    return {"basis": "measured T2 CPU per scenario in the same people bin: P1-09 (72) + P1-12 pilot (8)",
            "per_bin": per_bin, "t2_cpu_s_central": round(sum(central), 1), "t2_cpu_s_upper": round(sum(upper), 1),
            "t2_cpu_s_worst_case": worst, "t2_cpu_s_total_cap": round(1.25 * sum(upper), 1),
            "wall_s_central_at_concurrency": round(lpt_wall(sorted(central, reverse=True), workers), 1),
            "wall_s_upper_at_concurrency": round(lpt_wall(sorted(upper, reverse=True), workers), 1),
            "concurrency": workers,
            "caveats": ["T2 CPU varies 100x inside a bin (fire spread, bottlenecks); bin means are coarse",
                        "a T2 budget-limited decision depends on host speed and contention: such groups may not "
                        "regenerate identically", "upper = every scenario at its bin maximum"],
            "modal_usd": 0}


# ----- plan ---------------------------------------------------------------------------------------------------
def order(entries: Sequence[Mapping[str, Any]], *, salt: str = SALT) -> list[dict[str, Any]]:
    """All TRAIN scenarios, longest expected first (people, then max_steps, then hash rank). No outcome read."""
    rows = [{"scenario_id": e["scenario_id"], "scenario_hash": e["scenario_hash"], "family_id": e["family_id"],
             "lineage_id": e["lineage_id"], "core_hash": e.get("core_hash"), "label": e["label"],
             "fire_model": e["diagnostics"]["fire_model"], "people": e["diagnostics"]["people"],
             "max_steps": e["diagnostics"]["max_steps"], "rank": tp._rank(salt, e["scenario_hash"])}
            for e in entries]
    return sorted(rows, key=lambda s: (-s["people"], -s["max_steps"], s["rank"]))


def regeneration_ids(sc: Sequence[Mapping[str, Any]]) -> list[str]:
    """Preregistered regeneration sample: the REGEN_PER_BIN lowest-rank scenarios in each people bin."""
    out = []
    for b in N_BINS:
        inb = sorted((s for s in sc if _bin(s["people"]) == b), key=lambda s: s["rank"])
        out += [s["scenario_id"] for s in inb[:REGEN_PER_BIN]]
    return out


def design(sc: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"scenarios": len(sc), "families": len({s["family_id"] for s in sc}),
            "labels": dict(sorted(Counter(s["label"] for s in sc).items())),
            "fire_models": dict(sorted(Counter(s["fire_model"] for s in sc).items())),
            "people_bins": {f"{b[0]}-{b[1]}": sum(1 for s in sc if _bin(s["people"]) == b) for b in N_BINS},
            "people_max": max(s["people"] for s in sc), "people_total": sum(s["people"] for s in sc),
            "max_steps_total": sum(s["max_steps"] for s in sc), "groups_per_scenario": 1,
            "fire_seeds_per_scenario": 1, "teacher_episodes": len(sc) * len(CANDIDATES)}


def plan_body(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in plan.items() if k not in ("status", "plan_hash")}


def limits_for(concurrency: int, t2_cap: float) -> dict[str, Any]:
    return {**BASE_LIMITS, "max_concurrent_groups": concurrency, "worker_processes_max": concurrency * 5,
            "t2_cpu_s_total_cap": t2_cap}


def build_plan(train: Mapping[str, Any], heldout: Mapping[str, Mapping[str, set]], *,
               pins: Mapping[str, Any] | None = None, code: Mapping[str, Any] | None = None,
               geometry: Mapping[str, Any] | None = None, basis: Sequence[Mapping[str, Any]] = (),
               calibration: Mapping[str, Any] | None = None, storage_root: str = STORAGE_ROOT,
               pilot_ids: Sequence[str] = ()) -> dict[str, Any]:
    audit = sp.scenario_plan(train, heldout)["audit"]
    sc = order(train["scenarios"])
    if len(sc) != SCENARIOS or len({s["scenario_id"] for s in sc}) != SCENARIOS:
        raise ValueError(f"TRAIN must hold exactly {SCENARIOS} distinct scenarios")
    pins = dict(pins if pins is not None else tp.teacher_pins())
    if pins.get("pin_problems") or set(pins.get("teachers", {})) != set(CANDIDATES):
        raise ValueError(f"teacher pins are not exact: {pins.get('pin_problems')}")
    if geometry is not None and not geometry.get("pass"):
        raise ValueError(f"geometry leakage audit failed: {geometry.get('hits')}")
    cal = dict(calibration or {})
    conc = int(cal.get("chosen_concurrency") or 1)
    cost = cost_estimate(sc, basis, conc) if basis else {"not_computed": True, "t2_cpu_s_total_cap": 1e9}
    body = {
        "collection_version": VERSION, "namespace": NAMESPACE, "salt": SALT, "attempt": ATTEMPT,
        "source": {"split": "train", "manifest_hash": train["manifest_hash"], "spec_hash": train["spec_hash"],
                   "plan_hash": train["plan_hash"], "generator_revision": train["generator_revision"]},
        "selection_rule": {"rule": "every TRAIN scenario; no selection", "order": "people desc, max_steps desc, "
                           "sha256(salt|scenario_hash) (longest expected first)", "teacher_outcomes_consulted": False},
        "scenarios": sc, "design": design(sc),
        "disjointness_audit": audit, "geometry_audit": dict(geometry or {"not_computed": True}),
        "identities": dict(code or {}), "teacher_pins": pins,
        "limits": limits_for(conc, cost["t2_cpu_s_total_cap"]), "failure_policy": FAILURE_POLICY,
        "calibration": cal, "cost_estimate": cost,
        "regeneration": {"scenario_ids": regeneration_ids(sc),
                         "rule": f"the {REGEN_PER_BIN} lowest-rank scenarios of each people bin run a second time "
                                 "in the same attempt after every primary group, same caps and concurrency. A "
                                 "group with no budget-limited decision must reproduce content_hash exactly; a "
                                 "group with T2 budget-limited decisions is reported (host-dependent decisions)"},
        "pilot_overlap": {**PILOT_OVERLAP, "scenario_ids": sorted(pilot_ids)},
        "storage": {"root": storage_root, "in_git": False,
                    "layout": {k: str(v.relative_to(Path(storage_root))) for k, v in
                               collection_paths(Path(storage_root)).items() if k != "root"},
                    "expected_gib": cal.get("expected_storage_gib"),
                    "integrity": "SHA256SUMS over every file after verification; group files hold content_hash"},
        "sessions": {"first": "the evidence root must not exist or be empty", "resume": "--resume; re-runs every "
                     "gate; re-verifies finished groups; interrupted groups recorded, never rerun",
                     "max_sessions": BASE_LIMITS["max_sessions"]},
        "acceptance": {"full_train_verified": f"all {SCENARIOS} groups verified complete (verify command)",
                       "coverage": "reported exactly; incomplete or missing groups are never hidden",
                       "regeneration": "non-budget-limited regeneration groups identical",
                       "leakage_failures_max": 0, "training_eligible": False},
        "authorization_policy": tp.APPROVAL_POLICY,
    }
    return {**body, "status": PLAN_STATUS, "plan_hash": digest(body)}


def check_plan(plan: Mapping[str, Any], *, frozen: bool = True) -> list[str]:
    errs = []
    if digest(plan_body(plan)) != plan.get("plan_hash"):
        errs.append("plan hash does not match its content (tampered or stale)")
    if frozen and plan.get("status") != FROZEN:
        errs.append("the collection plan is not FROZEN")
    sc = plan.get("scenarios") or []
    if len(sc) != SCENARIOS or len({s.get("scenario_id") for s in sc}) != SCENARIOS:
        errs.append(f"the plan must hold exactly {SCENARIOS} distinct scenarios")
    if plan.get("source", {}).get("split") != "train":
        errs.append("the plan source is not the TRAIN split")
    if plan.get("attempt") != ATTEMPT or plan.get("namespace") != NAMESPACE:
        errs.append(f"this plan version covers {NAMESPACE} attempt {ATTEMPT} only")
    if not isinstance((plan.get("limits") or {}).get("max_concurrent_groups"), int):
        errs.append("the plan names no concurrency")
    return errs


def is_evaluation_set(name: str) -> bool:
    """Held-out student evaluation / selection maps (p1-02 DEV and test families, P1-09 selection set)."""
    return name.startswith("family:") or name == "p1-09-selection@1"


def classify_geometry(geo: Mapping[str, Any], reps: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Split geometry hits by the role of the matched set.

    A TRAIN map similar to a held-out evaluation map is leakage: the plan fails. A TRAIN map similar to a map of
    a teacher-development pilot (t4-pilot@*, t4-explore@*) is not student leakage, but a teacher tuned on that
    map may look better there: those families are flagged for every teacher-comparison claim."""
    fam_of = {s["scenario_id"]: s["family_id"] for s in reps}
    ev = [h for h in geo["hits"] if is_evaluation_set(h["against"])]
    dev = [h for h in geo["hits"] if not is_evaluation_set(h["against"])]
    return {**dict(geo), "hits": ev, "teacher_dev_similarity_hits": dev,
            "teacher_dev_similar_families": sorted({fam_of[h["scenario_id"]] for h in dev}),
            "within_pilot_hits": [], "within_train_note": "TRAIN-internal similarity is not leakage",
            "rule": "fail on any hit against a held-out evaluation set; flag hits against teacher-development "
                    "pilot sets (teacher-comparison caveat, no exclusion)", "pass": not ev}


def live_plan(repo: Path) -> dict[str, Any]:
    train = _read(repo / TRAIN_FILE)
    held, maps, src = tp.heldout_context(repo)
    sc = order(train["scenarios"])
    reps = list({s["family_id"]: s for s in sc}.values())          # one map per family
    fams = tp.train_families(train, [s["family_id"] for s in reps])
    geo = classify_geometry(tp.geometry_audit(reps, fams, maps), reps)
    geo["sources"] = src
    cal_p = repo / CALIBRATION_FILE
    pilot_p = repo / PILOT_SUMMARY_FILE
    pilot_ids = [g["scenario_id"] for g in _read(pilot_p)["groups"]] if pilot_p.exists() else []
    return build_plan(train, held, code=tp.code_identities() | {"collection": {
        "version": VERSION, "module_fingerprint": sp.module_fingerprint("ember_v1.train_collect")}},
        geometry=geo, basis=cost_basis(repo), calibration=_read(cal_p) if cal_p.exists() else None,
        pilot_ids=pilot_ids)


# ----- authorization ------------------------------------------------------------------------------------------
def _auth_scenarios(plan: Mapping[str, Any]) -> list[dict[str, str]]:
    return [{"scenario_id": s["scenario_id"], "scenario_hash": s["scenario_hash"]} for s in plan["scenarios"]]


def authorization_template(plan: Mapping[str, Any]) -> dict[str, Any]:
    """The PENDING record. Never approved here; no approval is invented."""
    return {"kind": AUTH_KIND, "authorization_version": AUTH_VERSION, "status": tp.AUTH_PENDING,
            "plan_hash": plan["plan_hash"], "plan_version": plan["collection_version"],
            "namespace": NAMESPACE, "manifest_hash": plan["source"]["manifest_hash"],
            "plan_freeze": {"commit": "<40-hex commit that adds the FROZEN plan>", "path": FROZEN_PLAN_FILE},
            "attempt": ATTEMPT, "scenarios": _auth_scenarios(plan),
            "teacher_fingerprints": tp.fingerprints_of(plan["teacher_pins"]),
            "code_identities": plan["identities"], "limits": plan["limits"],
            "failure_policy": plan["failure_policy"], "storage_root": plan["storage"]["root"],
            "expected_storage_gib": plan["storage"]["expected_gib"],
            "execution_commit": "<40-hex reviewed commit on main to check out and run>",
            "reviewer": {"github_login": "<owner login from the plan allowlist; must merge the PR>"},
            "approval": {"kind": tp.APPROVAL_POLICY["kind"], "repository": tp.APPROVAL_POLICY["repository"],
                         "pull_request": None, "owner_instruction": "<the owner's decision, quoted>",
                         "recorded_by": "<owner | agent on the owner's instruction>"},
            "statement": f"I reviewed the frozen plan and authorize attempt {ATTEMPT} of the full {SCENARIOS}-"
                         "scenario five-teacher TRAIN collection on local CPU, in up to max_sessions sessions "
                         "that run only never-started groups. No retry of a started group, no Modal, GPU, "
                         "training or benchmark."}


def check_authorization(auth: Mapping[str, Any] | None, plan: Mapping[str, Any]) -> list[str]:
    if not auth:
        return ["no full-TRAIN collection authorization"]
    errs = []
    if auth.get("kind") != AUTH_KIND or auth.get("authorization_version") != AUTH_VERSION:
        errs.append(f"kind must be {AUTH_KIND} version {AUTH_VERSION} (the pilot record does not extend)")
    if plan.get("status") != FROZEN:
        errs.append("the collection plan is not FROZEN")
    want = {"plan_hash": plan.get("plan_hash"), "namespace": NAMESPACE,
            "manifest_hash": plan["source"]["manifest_hash"], "scenarios": _auth_scenarios(plan),
            "teacher_fingerprints": tp.fingerprints_of(plan["teacher_pins"]),
            "code_identities": plan.get("identities"), "limits": plan.get("limits"),
            "failure_policy": plan.get("failure_policy"), "storage_root": plan["storage"]["root"]}
    for k, v in want.items():
        if auth.get(k) != v:
            errs.append(f"authorization {k} differs from the plan")
    if len(want["scenarios"]) != SCENARIOS:
        errs.append(f"the plan does not hold {SCENARIOS} scenarios")
    if type(auth.get("attempt")) is not int or auth["attempt"] != ATTEMPT:
        errs.append(f"authorization must name attempt {ATTEMPT}")
    if not isinstance(auth.get("execution_commit"), str) or not trust._HEX40.match(auth["execution_commit"]):
        errs.append("execution_commit must be a full 40-hex commit")
    pf = auth.get("plan_freeze") or {}
    if not isinstance(pf.get("commit"), str) or not trust._HEX40.match(pf["commit"]) \
            or pf.get("path") != FROZEN_PLAN_FILE:
        errs.append(f"plan_freeze must name a 40-hex commit and {FROZEN_PLAN_FILE}")
    rv, ap = auth.get("reviewer") or {}, auth.get("approval") or {}
    if auth.get("status") != tp.AUTH_APPROVED:
        errs.append(f"authorization status is {auth.get('status')!r}, not {tp.AUTH_APPROVED}")
    if rv.get("github_login") not in tp.APPROVAL_POLICY["approver_logins"]:
        errs.append("reviewer login is not an allowlisted approver")
    if ap.get("kind") != tp.APPROVAL_POLICY["kind"] or ap.get("repository") != tp.APPROVAL_POLICY["repository"] \
            or type(ap.get("pull_request")) is not int:
        errs.append("approval must name the GitHub pull request that carries the record")
    for k in ("owner_instruction", "recorded_by"):
        if not isinstance(ap.get(k), str) or not ap[k].strip() or ap[k].startswith("<"):
            errs.append(f"approval.{k} must state the owner's decision and who recorded it")
    return errs


# ----- scenarios ----------------------------------------------------------------------------------------------
def collection_scenarios(train: Mapping[str, Any], plan: Mapping[str, Any]) -> list[tuple[dict[str, Any], Any]]:
    """Regenerate all planned TRAIN scenarios from the committed manifest; check every hash."""
    return tp.pilot_scenarios(train, plan)


# ----- sessions and gates -------------------------------------------------------------------------------------
def check_root(root: Path, plan: Mapping[str, Any], *, resume: bool) -> int:
    """The session number this run may open, or CollectionBlocked."""
    if Path(os.path.abspath(root)) != Path(os.path.abspath(plan["storage"]["root"])):
        raise CollectionBlocked(f"evidence root {root} is not the plan's {plan['storage']['root']}")
    paths = collection_paths(root)
    done = sessions(root) if root.exists() else []
    if paths["evidence"].exists():
        raise CollectionBlocked("final evidence exists: the attempt is closed")
    if paths["stop"].exists():
        raise CollectionBlocked(f"{paths['stop']} exists: remove the operator stop request first")
    if not resume:
        if root.exists() and any(root.iterdir()):
            raise CollectionBlocked(f"{root} is not empty: a first session needs an empty root (use --resume)")
        return 1
    if not done:
        raise CollectionBlocked("--resume needs an existing session 1")
    if any(s.get("plan_hash") != plan["plan_hash"] for s in done):
        raise CollectionBlocked("an earlier session ran another plan")
    n = len(done) + 1
    if n > plan["limits"]["max_sessions"]:
        raise CollectionBlocked(f"session {n} exceeds max_sessions {plan['limits']['max_sessions']}")
    if not (root / f"session-{n - 1}.json").exists():
        raise CollectionBlocked("session markers are not contiguous")
    return n


def run_preconditions(repo: Path, plan_ref: Mapping[str, Any], auth_ref: Mapping[str, Any], root: Path, *,
                      resume: bool = False, trusted_ref: str = "origin/main",
                      gh: Callable[[str], Any] = tp.gh_api, live: Callable[[], Mapping[str, Any]] | None = None
                      ) -> dict[str, Any]:
    from .teacher import confirmation as C
    from .teacher import registry as reg
    reasons: list[str] = []
    plan = auth = auth_meta = None
    session = None
    try:
        plan, _ = trust.load_trusted_json(repo, plan_ref, trusted_ref=trusted_ref)
        if plan_ref.get("path") != FROZEN_PLAN_FILE:
            reasons.append(f"plan ref must point at {FROZEN_PLAN_FILE}")
        reasons += [f"plan: {e}" for e in check_plan(plan)]
    except (ValueError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as e:
        reasons.append(f"plan: {e}")
        plan = None
    try:
        auth, auth_meta = trust.load_trusted_json(repo, auth_ref, trusted_ref=trusted_ref)
    except (ValueError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as e:
        reasons.append(f"authorization: {e}")
    if plan is not None:
        reasons += [f"authorization: {e}" for e in check_authorization(auth, plan)]
        if auth and auth_meta and not reasons:
            reasons += [f"approval: {e}" for e in tp.verify_approval(repo, auth, auth_meta,
                                                                       trusted_ref=trusted_ref, gh=gh)]
        if auth and (auth.get("plan_freeze") or {}).get("commit") != plan_ref.get("commit"):
            reasons.append("authorization names another plan freeze commit")
        if reg.check_pins() or reg.check_registry():
            reasons.append("teacher identity drift (registry pins)")
        try:
            fresh = (live or (lambda: live_plan(repo)))()
            if fresh["plan_hash"] != plan["plan_hash"]:
                diff = sorted(k for k in plan_body(plan) if plan_body(plan)[k] != fresh.get(k))
                reasons.append(f"live rebuild differs from the frozen plan: {diff}")
        except (ValueError, OSError, KeyError) as e:
            reasons.append(f"live plan rebuild failed: {e}")
        try:
            session = check_root(root, plan, resume=resume)
        except CollectionBlocked as e:
            reasons.append(str(e))
    try:
        head = C._git(repo, "rev-parse", "HEAD")
    except subprocess.SubprocessError as e:
        head = ""
        reasons.append(f"git: {e}")
    dirty = C.tree_problems(repo)
    if dirty:
        reasons.append(f"working tree not clean: {dirty[:20]}")
    if auth and head != auth.get("execution_commit"):
        reasons.append("HEAD is not the authorized execution commit")
    if head and not trust.is_ancestor(repo, head, trusted_ref):
        reasons.append(f"HEAD is not in the history of {trusted_ref} (unreviewed code)")
    if plan_ref.get("commit") and head and not trust.is_ancestor(repo, plan_ref["commit"], head):
        reasons.append("HEAD does not descend from the plan freeze commit")
    if reasons:
        raise CollectionBlocked("; ".join(reasons))
    return {"plan": plan, "auth": auth, "auth_meta": auth_meta, "head": head, "session": session}


# ----- group runner (child process) ---------------------------------------------------------------------------
def _group_main(job: Mapping[str, Any], stop_event: Any, target: Callable[..., None] | None) -> None:
    """One matched group in its own process: five teacher workers, replay, verify, write group + row."""
    from . import multiteacher as mt
    from . import pilot_supervisor as ps
    from .sim import Scenario
    from .teacher import registry as reg
    from .teacher.confirmation import AttemptLog
    t_start = _utc()
    sid, kind = job["scenario_id"], job["kind"]
    scn = Scenario.from_dict(job["scenario"])
    base = AttemptLog(Path(job["log"]))
    mem = _MemSampler()

    def log(event: str, **f: Any) -> None:
        if event == "worker_start":
            mem.pids[f["teacher"]] = f["pid"]
        base(event, scenario_id=sid, kind=kind, **f)

    parent = mp.parent_process()

    def stop() -> bool:
        return stop_event.is_set() or (parent is not None and not parent.is_alive())

    mem.pids["runner"] = os.getpid()
    mem.start()
    jobs = job["jobs"]
    insts = {}
    for j in jobs:
        cls = reg.teacher_class(j["name"], j["identity"]["version"])
        insts[j["name"]] = cls() if j["seed"] is None else cls(seed=j["seed"])
    pins = {"identities": {n: p.identity.to_dict() for n, p in insts.items()},
            "seeds": {j["name"]: j["seed"] for j in jobs}}
    prov = mt.provenance(scn, insts, {"kind": "train", "manifest_hash": job["manifest_hash"], "scenario_id": sid,
                                      "lineage_id": job["lineage_id"]})
    deadline = time.monotonic() + max(0.0, job["deadline_epoch"] - time.time())
    run = ps.run_group(scn, jobs, decision_cpu_s=job["limits"]["decision_cpu_s"],
                       group_wall_s=job["limits"]["group_wall_s"], deadline=deadline, log=log, target=target,
                       stop_requested=stop)
    # Only run-independent fields enter the sealed group: a regeneration must be able to reproduce content_hash.
    # Session and run kind are in the row and in runtime (outside content_hash).
    extra = {"collection": {"plan_hash": job["plan_hash"], "namespace": NAMESPACE, "scenario_id": sid,
                            "attempt": ATTEMPT}}
    rep, err = None, None
    try:
        rep = ps.build_group(scn, run, pins["identities"], pins["seeds"], prov, extra=extra, run_kind=kind,
                             scope=job["scope"])
    except ps.GroupIntegrityError as e:
        err = f"integrity_error: {e}"
    resources = mem.stop()
    out = Path(job["group_file"])
    if job.get("keep_group", True):
        doc = rep if rep is not None else {"scenario_id": sid, "integrity_error": err, "group_complete": False,
                                           "training_eligible": False, "runtime": {"supervision": run["supervision"]},
                                           "failures": run["failures"], "extra": extra}
        write_once(out, doc)
    row = tp._group_summary(sid, rep, scn, run, "ran", err)
    row.update(kind=kind, session=job["session"], started_utc=t_start, ended_utc=_utc(),
               interrupted=run["interrupted"], process_cpu_s=run["supervision"]["process_cpu_s"],
               peak_live_decision_cpu_ms=run["supervision"]["peak_live_decision_cpu_ms"], **resources,
               file_sha256=_sha256_file(out) if job.get("keep_group", True) else None,
               final_transition_hashes={n: r["final_transition_hash"] for n, r in rep["policies"].items()}
               if rep else None,
               outcomes={n: r["outcome"] for n, r in rep["policies"].items()} if rep else None,
               failures=run["failures"])
    write_once(Path(job["row_file"]), row)


# ----- scheduler ----------------------------------------------------------------------------------------------
def _jobs(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    return tp._jobs(plan)


class _Scheduler:
    """Run group jobs with at most `limit` live runners; RAM admission; stop file; session deadline."""

    def __init__(self, paths: Mapping[str, Path], log: Callable[..., None], *, limit: int, min_free_gib: float,
                 deadline_epoch: float, t2_cap: float, t2_used: float, target: Any, poll_s: float = POLL_S) -> None:
        self.paths, self.log, self.limit, self.min_free = paths, log, limit, min_free_gib
        self.deadline, self.t2_cap, self.t2_used, self.target, self.poll = deadline_epoch, t2_cap, t2_used, target, poll_s
        self.ctx = mp.get_context("spawn")
        self.stop_event = self.ctx.Event()
        self.halt_reason: str | None = None
        self.ram_waits = 0
        self.max_live = 0
        self.host_min: float | None = None
        self._last_progress = 0.0

    def _admit_reason(self) -> str | None:
        if self.paths["stop"].exists():
            return "operator_stop"
        if time.time() > self.deadline:
            return "session_wall_cap"
        if self.t2_used > self.t2_cap:
            return "t2_cpu_total_cap"
        return None

    def run(self, jobs: Sequence[Mapping[str, Any]], *, started_dir: Path, progress: Callable[[dict], None]
            ) -> dict[str, str]:
        """Returns {sid: "row" | "runner_crashed" | "not_started"}."""
        pending, running, result = list(jobs), {}, {}
        waiting_ram = False
        try:
            while pending or running:
                reason = self._admit_reason()
                if reason and not self.halt_reason:
                    self.halt_reason = reason
                    self.log("admission_closed", reason=reason)
                if pending and not self.halt_reason and len(running) < self.limit:
                    free = host_free_ram_gib()
                    if free is not None:
                        self.host_min = free if self.host_min is None else min(self.host_min, free)
                    if free is None or free >= self.min_free:
                        if waiting_ram:
                            self.log("ram_wait_end", free_gib=round(free or -1, 3))
                            waiting_ram = False
                        job = pending.pop(0)
                        write_once(started_dir / f"{job['scenario_id']}.json",
                                   {"scenario_id": job["scenario_id"], "kind": job["kind"],
                                    "session": job["session"], "started_utc": _utc(),
                                    "scenario_hash": job["scenario_hash"], "host_free_gib": free})
                        p = self.ctx.Process(target=_group_main, args=(job, self.stop_event, self.target),
                                             name=f"p1-17-group-{job['scenario_id']}", daemon=False)
                        p.start()
                        running[job["scenario_id"]] = (p, job)
                        self.max_live = max(self.max_live, len(running))
                        self.log("group_runner_start", scenario_id=job["scenario_id"], kind=job["kind"],
                                 pid=p.pid, live=len(running), free_gib=None if free is None else round(free, 3))
                        time.sleep(STAGGER_S)
                        continue
                    if not waiting_ram:
                        waiting_ram = True
                        self.ram_waits += 1
                        self.log("ram_wait", free_gib=round(free, 3), need_gib=self.min_free)
                if self.halt_reason and pending:
                    for j in pending:
                        result[j["scenario_id"]] = "not_started"
                    pending = []
                for sid, (p, job) in list(running.items()):
                    if p.is_alive():
                        continue
                    p.join()
                    del running[sid]
                    row_p = Path(job["row_file"])
                    if row_p.exists():
                        r = _read(row_p)
                        self.t2_used += r.get("t2_cpu_s") or 0.0
                        result[sid] = "row"
                        self.log("group_runner_end", scenario_id=sid, kind=job["kind"], exit_code=p.exitcode,
                                 complete=r.get("group_complete"), verified=r.get("verified"))
                    else:
                        result[sid] = "runner_crashed"
                        self.log("group_runner_crashed", scenario_id=sid, kind=job["kind"], exit_code=p.exitcode)
                        write_once(row_p, {"scenario_id": sid, "kind": job["kind"], "status": "runner_crashed",
                                           "session": job["session"], "exit_code": p.exitcode, "verified": False,
                                           "group_complete": False, "group_file_exists":
                                               Path(job["group_file"]).exists(), "ended_utc": _utc()})
                now = time.monotonic()
                if now - self._last_progress > PROGRESS_EVERY_S:
                    self._last_progress = now
                    progress({"running": sorted(running), "pending": len(pending), "halt_reason": self.halt_reason,
                              "t2_cpu_s_used": round(self.t2_used, 1)})
                time.sleep(self.poll)
        except KeyboardInterrupt:
            self.stop_event.set()
            self.halt_reason = "interrupted"
            self.log("interrupt", running=sorted(running))
            for p, _ in running.values():
                p.join(60)
            raise
        return result


def _scan(plan: Mapping[str, Any], paths: Mapping[str, Path], kind: str) -> dict[str, dict[str, Any]]:
    """Per scenario: started, row, group file and interrupted record state."""
    sd, rd, gd = ((paths["started"], paths["rows"], paths["groups"]) if kind == "primary"
                  else (paths["regen_started"], paths["regen_rows"], paths["regen"]))
    ids = [s["scenario_id"] for s in plan["scenarios"]] if kind == "primary" else plan["regeneration"]["scenario_ids"]
    return {sid: {"started": (sd / f"{sid}.json").exists(), "row": (rd / f"{sid}.json").exists(),
                  "group": (gd / f"{sid}.json").exists(),
                  "interrupted": (paths["interrupted"] / f"{kind}-{sid}.json").exists()} for sid in ids}


def _check_finished(paths: Mapping[str, Path], kind: str, sid: str) -> list[str]:
    """Resume check of a finished group: its row names the stored file's sha256."""
    rd, gd = (paths["rows"], paths["groups"]) if kind == "primary" else (paths["regen_rows"], paths["regen"])
    row = _read(rd / f"{sid}.json")
    g = gd / f"{sid}.json"
    if row.get("status") == "runner_crashed":
        return []
    if not g.exists():
        return [f"{kind} {sid}: row without group file"]
    if row.get("file_sha256") != _sha256_file(g):
        return [f"{kind} {sid}: group file changed after its row was written"]
    return []


def execute(plan: Mapping[str, Any], scenarios: Sequence[tuple[Mapping[str, Any], Any]], root: Path,
            marker_doc: Mapping[str, Any], *, session: int = 1, target: Callable[..., None] | None = None,
            jobs: Sequence[Mapping[str, Any]] | None = None, poll_s: float = POLL_S) -> dict[str, Any]:
    """After every gate: open the session, run every never-started group, then the regeneration sample."""
    from . import multiteacher as mt
    from .teacher.confirmation import AttemptLog
    paths = collection_paths(root)
    check_root(root, plan, resume=session > 1)
    root.mkdir(parents=True, exist_ok=True)
    for k in ("started", "groups", "rows", "interrupted", "regen_started", "regen", "regen_rows", "logs"):
        paths[k].mkdir(exist_ok=True)
    body = {"session_version": SESSION_VERSION, "namespace": NAMESPACE, "attempt": ATTEMPT, "session": session,
            "plan_hash": plan["plan_hash"], **dict(marker_doc), "limits": plan["limits"], "started_utc": _utc(),
            "pid": os.getpid(),
            "platform": {"system": platform.system(), "machine": platform.machine(),
                         "python": platform.python_version(), "cpu_count": os.cpu_count()}}
    marker = {**body, "session_hash": digest(body)}
    write_once(root / f"session-{session}.json", marker)          # BEFORE the first teacher call of this session
    alog = AttemptLog(paths["log"])

    def log(event: str, **f: Any) -> None:
        alog(event, session=session, **f)

    log("session_start", session_hash=marker["session_hash"])
    problems, newly_interrupted = [], []
    for kind in ("primary", "regeneration"):
        for sid, st in _scan(plan, paths, kind).items():
            if st["row"]:
                problems += _check_finished(paths, kind, sid)
            elif st["started"] and not st["interrupted"]:
                write_once(paths["interrupted"] / f"{kind}-{sid}.json",
                           {"scenario_id": sid, "kind": kind, "status": "interrupted", "recorded_session": session,
                            "group_file_exists": st["group"], "recorded_utc": _utc(),
                            "policy": FAILURE_POLICY["interrupted"]})
                newly_interrupted.append(f"{kind}:{sid}")
    if problems:
        log("resume_verification_failed", problems=problems)
        raise CollectionBlocked(f"finished groups fail verification: {problems[:10]}")
    if newly_interrupted:
        log("interrupted_recorded", groups=newly_interrupted)
    by_id = {e["scenario_id"]: (e, s) for e, s in scenarios}
    jobs = list(jobs if jobs is not None else _jobs(plan))
    deadline = time.time() + plan["limits"]["session_wall_s"]
    t2_used = sum((_read(p).get("t2_cpu_s") or 0.0) for d in ("rows", "regen_rows") for p in paths[d].glob("*.json"))

    def job(sid: str, kind: str) -> dict[str, Any]:
        e, scn = by_id[sid]
        prim = kind == "primary"
        return {"scenario_id": sid, "kind": kind, "session": session, "scenario": scn.to_dict(),
                "scenario_hash": scn.content_hash(), "lineage_id": e["lineage_id"],
                "manifest_hash": plan["source"]["manifest_hash"], "plan_hash": plan["plan_hash"],
                "limits": plan["limits"], "jobs": jobs, "scope": mt.SCOPE_TRAIN_FULL, "deadline_epoch": deadline,
                "log": str(paths["logs"] / f"{kind}-{sid}.jsonl"),
                "group_file": str((paths["groups"] if prim else paths["regen"]) / f"{sid}.json"),
                "row_file": str((paths["rows"] if prim else paths["regen_rows"]) / f"{sid}.json")}

    sched = _Scheduler(paths, log, limit=plan["limits"]["max_concurrent_groups"],
                       min_free_gib=plan["limits"]["min_free_ram_gib_to_start"], deadline_epoch=deadline,
                       t2_cap=plan["limits"]["t2_cpu_s_total_cap"], t2_used=t2_used, target=target, poll_s=poll_s)

    def progress(extra: dict) -> None:
        st = _scan(plan, paths, "primary")
        _replace_json(paths["progress"], {"session": session, "updated_utc": _utc(), "planned": len(st),
                                          "finished": sum(v["row"] for v in st.values()),
                                          "interrupted": sum(v["interrupted"] for v in st.values()),
                                          "started": sum(v["started"] for v in st.values()), **extra,
                                          "host_free_gib": host_free_ram_gib()})

    t0 = time.monotonic()
    interrupted = False
    try:
        prim = [job(s["scenario_id"], "primary") for s in plan["scenarios"]
                if not _scan(plan, paths, "primary")[s["scenario_id"]]["started"]]
        sched.run(prim, started_dir=paths["started"], progress=progress)
        prim_state = _scan(plan, paths, "primary")
        regen = []
        if not sched.halt_reason and all(v["row"] or v["interrupted"] for v in prim_state.values()):
            for sid, st in _scan(plan, paths, "regeneration").items():
                if st["started"]:
                    continue
                r = _read(paths["rows"] / f"{sid}.json") if prim_state[sid]["row"] else {}
                if r.get("group_complete") and r.get("verified"):
                    regen.append(job(sid, "regeneration"))
            sched.run(regen, started_dir=paths["regen_started"], progress=progress)
    except KeyboardInterrupt:
        interrupted = True
    progress({"running": [], "pending": 0, "halt_reason": sched.halt_reason, "t2_cpu_s_used": round(sched.t2_used, 1)})
    end = {"session": session, "ended_utc": _utc(), "wall_s": round(time.monotonic() - t0, 3),
           "halt_reason": sched.halt_reason, "interrupted": interrupted, "max_live_groups": sched.max_live,
           "ram_waits": sched.ram_waits, "host_min_free_gib": sched.host_min,
           "supervisor_peak_rss_mb": peak_rss_mb(os.getpid()), "newly_interrupted": newly_interrupted}
    write_once(root / f"session-{session}-end.json", end)
    log("session_end", **{k: v for k, v in end.items() if k != "session"})
    out = {"session": session, **end, "closed": False}
    if not interrupted and not sched.halt_reason and attempt_terminal(plan, paths):
        ev = summarize(plan, root)
        write_once(paths["evidence"], ev)
        log("evidence_written", full_train_verified=ev["coverage"]["full_train_verified"])
        out.update(closed=True, evidence=str(paths["evidence"]), coverage=ev["coverage"])
    if interrupted:
        raise KeyboardInterrupt
    return out


def attempt_terminal(plan: Mapping[str, Any], paths: Mapping[str, Path]) -> bool:
    """Every primary group has a row or an interrupted record, and every eligible regeneration too."""
    prim = _scan(plan, paths, "primary")
    if not all(v["row"] or v["interrupted"] for v in prim.values()):
        return False
    for sid, st in _scan(plan, paths, "regeneration").items():
        r = _read(paths["rows"] / f"{sid}.json") if prim[sid]["row"] else {}
        if r.get("group_complete") and r.get("verified") and not (st["row"] or st["interrupted"]):
            return False
    return True


def summarize(plan: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Session-side evidence: coverage, regeneration, cost (the independent `verify` re-checks everything)."""
    paths = collection_paths(root)
    prim = _scan(plan, paths, "primary")
    rows = {sid: _read(paths["rows"] / f"{sid}.json") for sid, v in prim.items() if v["row"]}
    complete = sorted(s for s, r in rows.items() if r.get("verified") and r.get("group_complete")
                      and not r.get("failed_teachers") and r.get("leakage_ok"))
    status = Counter("verified_complete" if s in complete else
                     ("interrupted" if prim[s]["interrupted"] and s not in rows else
                      (rows[s].get("status") if s in rows and rows[s].get("status") != "ran" else
                       "integrity_error" if s in rows and str(rows[s].get("error") or "").startswith("integrity")
                       else "incomplete" if s in rows else "not_started"))
                     for s in prim)
    regen = {}
    for sid in plan["regeneration"]["scenario_ids"]:
        p = paths["regen_rows"] / f"{sid}.json"
        r0 = rows.get(sid) or {}
        bl = (r0.get("budget_limited_decisions") or {}).get("T2", 0)
        if not p.exists():
            regen[sid] = {"run": False, "budget_limited_t2": bl}
            continue
        r = _read(p)
        regen[sid] = {"run": True, "identical": r.get("content_hash") == r0.get("content_hash")
                      and r.get("content_hash") is not None, "budget_limited_t2": bl,
                      "teachers_identical": {t: (r.get("final_transition_hashes") or {}).get(t)
                                             == (r0.get("final_transition_hashes") or {}).get(t) for t in CANDIDATES}}
    det_ok = all(v.get("identical") for v in regen.values() if v["run"] and not v["budget_limited_t2"])
    body = {"evidence_version": EVIDENCE_VERSION, "plan_hash": plan["plan_hash"], "namespace": NAMESPACE,
            "sessions": sessions(root), "coverage": {"planned": len(prim), **dict(sorted(status.items())),
                                                     "full_train_verified": len(complete) == SCENARIOS},
            "regeneration": regen, "regeneration_deterministic_ok": det_ok,
            "t2_cpu_s_total": round(sum((r.get("t2_cpu_s") or 0) for r in rows.values()), 1),
            "training_eligible": False,
            "note": "Collection evidence. Training eligibility is decided later by the dataset gate."}
    return {**body, "evidence_hash": digest(body)}


# ----- verification (read only; completion-triggered) ---------------------------------------------------------
def _common_prefix_disagreement(g: Mapping[str, Any]) -> dict[str, Any]:
    """Teacher joint-action disagreement at step 0 (identical initial state for all five teachers)."""
    acts = {}
    for t, r in g["policies"].items():
        if r["decisions"]:
            acts[t] = digest(r["decisions"][0]["proposal"].get("actions"))
    distinct = len(set(acts.values()))
    pairs = [(a, b) for i, a in enumerate(CANDIDATES) for b in CANDIDATES[i + 1:] if a in acts and b in acts]
    return {"distinct_joint_actions_step0": distinct,
            "pairwise_differ_step0": {f"{a}-{b}": acts[a] != acts[b] for a, b in pairs}}


from .sim.movement import DELTAS as _DELTA  # noqa: E402


def fire_entry_moves(decision: Mapping[str, Any]) -> list[str]:
    """Person ids whose proposed move targets a cell that is burning in the decision's own observation
    (an unsafe proposal; with enter_fire='blocked' the simulator blocks it, with 'casualty' it kills)."""
    obs = decision["observation"]
    burning = {tuple(c) for c in obs["burning"]}
    pos = {p["id"]: tuple(p["position"]) for p in obs["people"] if p["status"] == "active"}
    out = []
    for pid, a in sorted((decision["proposal"].get("actions") or {}).items()):
        if a != "stay" and pid in pos and a in _DELTA:
            r, c = pos[pid]
            if (r + _DELTA[a][0], c + _DELTA[a][1]) in burning:
                out.append(pid)
    return out


def _fire_entries(g: Mapping[str, Any], t: str) -> int:
    return sum(len(fire_entry_moves(d)) for d in g["policies"][t]["decisions"])


def verify_collection(plan: Mapping[str, Any], root: Path, scenarios: Sequence[tuple[Mapping[str, Any], Any]],
                      *, pilot_dir: Path | None = None) -> dict[str, Any]:
    """Independent full verification of every stored group (replay, outcomes, identities, limits, hashes)."""
    from . import multiteacher as mt
    paths = collection_paths(root)
    by_id = {e["scenario_id"]: (e, s) for e, s in scenarios}
    want_fp = tp.fingerprints_of(plan["teacher_pins"])
    lim = plan["limits"]
    per, totals = [], {t: Counter() for t in CANDIDATES}
    fam: dict[str, dict[str, Counter]] = {}
    for spec in plan["scenarios"]:
        sid = spec["scenario_id"]
        e, scn = by_id[sid]
        gp, rp = paths["groups"] / f"{sid}.json", paths["rows"] / f"{sid}.json"
        v = {"scenario_id": sid, "family_id": spec["family_id"], "people": spec["people"], "problems": []}
        if not gp.exists():
            v["status"] = "interrupted" if (paths["interrupted"] / f"primary-{sid}.json").exists() else (
                "runner_crashed" if rp.exists() else "missing")
            per.append(v)
            continue
        g = _read(gp)
        row = _read(rp) if rp.exists() else {}
        if row.get("file_sha256") != _sha256_file(gp):
            v["problems"].append("group file sha256 differs from its row")
        if "integrity_error" in g:
            v["status"] = "integrity_error"
            per.append(v)
            continue
        try:
            mt.verify_fixture(g, scn)
        except (mt.MultiTeacherError, KeyError, TypeError, ValueError) as exc:
            v["problems"].append(f"replay: {type(exc).__name__}: {exc}"[:300])
            v["status"] = "invalid"
            per.append(v)
            continue
        try:
            for r in g["policies"].values():
                for d in r["decisions"]:
                    mt.check_public_observation(d["observation"])
        except mt.MultiTeacherError as exc:
            v["problems"].append(f"leakage: {exc}")
        if g.get("scope") != mt.SCOPE_TRAIN_FULL or (g.get("collection") or {}).get("plan_hash") != plan["plan_hash"]:
            v["problems"].append("scope or plan hash")
        src = g["provenance"]["scenario"]
        if src["content_hash"] != spec["scenario_hash"] or src["source"]["manifest_hash"] != \
                plan["source"]["manifest_hash"] or src["source"]["lineage_id"] != spec["lineage_id"]:
            v["problems"].append("scenario provenance")
        for t, pv in g["provenance"]["teachers"].items():
            got = {"version": pv["identity"]["version"], "config_hash": pv["identity"]["config_hash"],
                   "code_fingerprint": pv["code_fingerprint"]}
            if got != want_fp[t]:
                v["problems"].append(f"{t} identity/fingerprint")
        sup = g["runtime"]["supervision"]
        if sup["wall_s"] > lim["group_wall_s"]:
            v["problems"].append("group wall cap")
        over = {t: c for t, c in sup["peak_live_decision_cpu_ms"].items() if c > lim["decision_cpu_s"][t] * 1000}
        complete = g["group_complete"] and not g["failed_teachers"]
        v.update(status="verified_complete" if complete and not v["problems"] else
                 ("invalid" if v["problems"] else "incomplete"),
                 failed_teachers=g["failed_teachers"], wall_s=sup["wall_s"], process_cpu_s=sup["process_cpu_s"],
                 budget_limited_decisions=sup["budget_limited_decisions"], decision_cap_overruns=over,
                 peak_rss_mb=row.get("peak_rss_mb"), disagreement=_common_prefix_disagreement(g),
                 common_state_steps=len(g.get("common_state_steps") or []),
                 t2_fire_entries=_fire_entries(g, "T2"), fire_model=spec["fire_model"], label=spec["label"],
                 outcomes={t: {k: r["outcome"][k] for k in ("casualties", "evacuated", "unresolved", "steps")}
                           if r["outcome"] else None for t, r in g["policies"].items()},
                 final_transition_hashes={t: r["final_transition_hash"] for t, r in g["policies"].items()})
        if v["status"] == "verified_complete":
            for t, o in v["outcomes"].items():
                totals[t].update({k: o[k] for k in ("casualties", "evacuated", "unresolved")})
                f = fam.setdefault(spec["family_id"], {x: Counter() for x in CANDIDATES})
                f[t].update({"casualties_plus_unresolved": o["casualties"] + o["unresolved"]})
        per.append(v)
    status = Counter(v["status"] for v in per)
    complete = [v for v in per if v["status"] == "verified_complete"]
    dis = {}
    if complete:
        for k in complete[0]["disagreement"]["pairwise_differ_step0"]:
            dis[k] = round(sum(v["disagreement"]["pairwise_differ_step0"][k] for v in complete) / len(complete), 4)
    overlap = {}
    if pilot_dir is not None and (pilot_dir / "groups").exists():
        for v in complete:
            pg = pilot_dir / "groups" / f"{v['scenario_id']}.json"
            if pg.exists():
                pr = _read(pg)
                overlap[v["scenario_id"]] = {t: pr["policies"][t]["final_transition_hash"]
                                             == v["final_transition_hashes"][t] for t in CANDIDATES}
    fam_best = Counter()
    for f in fam.values():
        vals = {t: f[t]["casualties_plus_unresolved"] for t in CANDIDATES}
        m = min(vals.values())
        for t, x in vals.items():
            if x == m:
                fam_best[t] += 1
    body = {"verification_version": "p1-17-train-full-verify@1.0.0", "plan_hash": plan["plan_hash"],
            "verified_utc": _utc(), "status_counts": dict(sorted(status.items())),
            "full_train_verified": status.get("verified_complete", 0) == SCENARIOS,
            "teacher_episodes_complete": len(complete) * len(CANDIDATES),
            "totals_verified_complete": {t: {**dict(c), "casualties_plus_unresolved":
                                             c["casualties"] + c["unresolved"]} for t, c in totals.items()},
            "people_verified_complete": sum(v["people"] for v in complete),
            "step0_pairwise_disagreement_rate": dis,
            "step0_all_five_identical_share": round(sum(v["disagreement"]["distinct_joint_actions_step0"] == 1
                                                        for v in complete) / max(1, len(complete)), 4),
            "families_best_count": dict(sorted(fam_best.items())), "families": len(fam),
            "t2_budget_limited_decisions": sum((v.get("budget_limited_decisions") or {}).get("T2", 0) for v in per),
            "t2_budget_limited_groups": sum(1 for v in per if (v.get("budget_limited_decisions") or {}).get("T2")),
            "t2_fire_entries": sum(v.get("t2_fire_entries", 0) for v in per),
            "decision_cap_overruns": {v["scenario_id"]: v["decision_cap_overruns"] for v in per
                                      if v.get("decision_cap_overruns")},
            "cpu_s_total": {t: round(sum((v.get("process_cpu_s") or {}).get(t, 0) for v in per), 1)
                            for t in CANDIDATES},
            "pilot_overlap_identical": overlap, "groups": per, "training_eligible": False}
    return {**body, "verification_hash": digest(body)}


def sha256sums(root: Path, out: Path) -> int:
    lines = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p != out and p.name != "progress.json":
            lines.append(f"{_sha256_file(p)[7:]} *{p.relative_to(root).as_posix()}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return len(lines)


# ----- calibration (resource measurement on the 8 pilot scenarios; no evidence) -------------------------------
def calibrate(repo: Path, ids: Sequence[str], root: Path, concurrency: int, *, repeat: int = 1,
              target: Any = None) -> dict[str, Any]:
    """Run pilot scenarios with `concurrency` live groups to measure RAM, CPU and wall. Writes rows only.

    `repeat` > 1 runs each scenario that many times at once (`<sid>~r<k>`): a pure contention measurement."""
    from .teacher.confirmation import AttemptLog
    pilot = _read(repo / PILOT_SUMMARY_FILE)
    allowed = {g["scenario_id"]: g for g in pilot["groups"]}
    if not ids or any(i not in allowed for i in ids):
        raise CollectionBlocked("calibration runs only already-collected P1-12 pilot scenarios")
    plan = _read(repo / tp.FROZEN_PLAN_FILE) if (repo / tp.FROZEN_PLAN_FILE).exists() else None
    train = _read(repo / TRAIN_FILE)
    fake = {"plan_hash": "calibration", "scenarios": [{"scenario_id": i} for i in ids],
            "teacher_pins": tp.teacher_pins()}
    want = {e["scenario_id"]: e for e in train["scenarios"] if e["scenario_id"] in ids}
    fake["scenarios"] = [{"scenario_id": i, "scenario_hash": want[i]["scenario_hash"],
                          "lineage_id": want[i]["lineage_id"], "family_id": want[i]["family_id"]} for i in ids]
    scen = {e["scenario_id"]: s for e, s in tp.pilot_scenarios(train, fake)}
    root.mkdir(parents=True, exist_ok=False)
    for d in ("started", "rows", "logs"):
        (root / d).mkdir()
    jobs = tp._jobs(fake)
    lim = {**BASE_LIMITS, "max_concurrent_groups": concurrency}
    deadline = time.time() + 4 * 3600
    alog = AttemptLog(root / "log.jsonl")
    keys = [(i, f"{i}~r{k}" if repeat > 1 else i) for i in ids for k in range(repeat)]
    js = [{"scenario_id": key, "kind": "calibration", "session": 0, "scenario": scen[i].to_dict(),
           "scenario_hash": scen[i].content_hash(), "lineage_id": want[i]["lineage_id"],
           "manifest_hash": train["manifest_hash"], "plan_hash": "calibration", "limits": lim, "jobs": jobs,
           "scope": "p1-17-train-full@1", "deadline_epoch": deadline, "log": str(root / "logs" / f"{key}.jsonl"),
           "group_file": str(root / f"{key}.unused.json"), "row_file": str(root / "rows" / f"{key}.json"),
           "keep_group": False} for i, key in keys]
    sched = _Scheduler(collection_paths(root), alog, limit=concurrency, min_free_gib=0.0, deadline_epoch=deadline,
                       t2_cap=1e12, t2_used=0.0, target=target)
    t0 = time.monotonic()
    free0 = host_free_ram_gib()
    sched.run(js, started_dir=root / "started", progress=lambda d: None)
    wall = time.monotonic() - t0
    rows = {key: (i, _read(root / "rows" / f"{key}.json")) for i, key in keys}
    out = {"calibration_version": "p1-17-calibration@1.0.0", "concurrency": concurrency, "ids": list(ids),
           "wall_s": round(wall, 3), "host_free_gib_before": free0, "host_min_free_gib": sched.host_min,
           "pilot_plan_hash": (plan or {}).get("plan_hash"), "groups": {}}
    for key, (i, r) in rows.items():
        pg = allowed[i]
        out["groups"][key] = {
            "people": pg["people"], "wall_s": r.get("wall_s"), "process_cpu_s": r.get("process_cpu_s"),
            "pilot_process_cpu_s": pg["process_cpu_s"], "pilot_wall_s": pg["wall_s"],
            "peak_rss_mb": r.get("peak_rss_mb"), "host_min_free_gib": r.get("host_min_free_gib"),
            "budget_limited_decisions": r.get("budget_limited_decisions"), "complete": r.get("group_complete"),
            "verified": r.get("verified"),
            "matches_pilot": {t: (r.get("final_transition_hashes") or {}).get(t)
                              == pg["teachers"][t]["final_transition_hash"] for t in CANDIDATES}}
    _replace_json(root / "calibration.json", out)
    return out


# ----- CLI ----------------------------------------------------------------------------------------------------
def run_collection(repo: Path, plan_ref: Mapping[str, Any], auth_ref: Mapping[str, Any], root: Path, *,
                   resume: bool = False, trusted_ref: str = "origin/main") -> dict[str, Any]:
    pre = run_preconditions(repo, plan_ref, auth_ref, root, resume=resume, trusted_ref=trusted_ref)
    plan = pre["plan"]
    scenarios = collection_scenarios(_read(repo / TRAIN_FILE), plan)
    return execute(plan, scenarios, root, {
        "plan_freeze_commit": plan_ref["commit"], "authorization": {**pre["auth_meta"]},
        "execution_commit": pre["head"], "trusted_ref": trusted_ref, "resume": resume}, session=pre["session"])


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ember_v1.train_collect")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("plan", help="write the PROPOSED plan (runs no teacher)")
    pl.add_argument("--out", required=True)
    tpl = sub.add_parser("template", help="write the PENDING authorization template")
    tpl.add_argument("--plan", required=True)
    tpl.add_argument("--out", required=True)
    vp = sub.add_parser("verify-plan", help="rebuild and compare with a plan file (runs no teacher)")
    vp.add_argument("--plan", required=True)
    ps_ = sub.add_parser("pilot-summary", help="recompute the P1-12 pilot result summary (read only)")
    ps_.add_argument("--attempt-dir", required=True)
    ps_.add_argument("--out", required=True)
    cb = sub.add_parser("calibrate", help="resource calibration on P1-12 pilot scenarios (rows only)")
    cb.add_argument("--ids", required=True)
    cb.add_argument("--root", required=True)
    cb.add_argument("--concurrency", type=int, required=True)
    cb.add_argument("--repeat", type=int, default=1)
    rn = sub.add_parser("run", help="one authorized session; refused unless every gate passes")
    rn.add_argument("--plan-ref", required=True)
    rn.add_argument("--auth-ref", required=True)
    rn.add_argument("--root", default=STORAGE_ROOT)
    rn.add_argument("--resume", action="store_true")
    rn.add_argument("--trusted-ref", default="origin/main")
    gt = sub.add_parser("gates", help="preflight: every run gate, no teacher call, nothing written")
    gt.add_argument("--plan-ref", required=True)
    gt.add_argument("--auth-ref", required=True)
    gt.add_argument("--root", default=STORAGE_ROOT)
    gt.add_argument("--resume", action="store_true")
    gt.add_argument("--trusted-ref", default="origin/main")
    vf = sub.add_parser("verify", help="full read-only verification of the evidence root")
    vf.add_argument("--plan", required=True)
    vf.add_argument("--root", default=STORAGE_ROOT)
    vf.add_argument("--pilot-dir", default=None)
    vf.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    repo = Path(__file__).resolve().parents[2]
    if a.cmd == "run":
        out = run_collection(repo, tp.parse_ref(a.plan_ref), tp.parse_ref(a.auth_ref), Path(a.root),
                             resume=a.resume, trusted_ref=a.trusted_ref)
        sys.stdout.write(json.dumps(out, default=str) + "\n")
        return 0
    if a.cmd == "gates":
        try:
            pre = run_preconditions(repo, tp.parse_ref(a.plan_ref), tp.parse_ref(a.auth_ref), Path(a.root),
                                    resume=a.resume, trusted_ref=a.trusted_ref)
        except CollectionBlocked as e:
            sys.stdout.write(json.dumps({"status": "BLOCKED", "reasons": str(e).split("; ")}) + "\n")
            return 2
        sys.stdout.write(json.dumps({"status": "READY_AUTHORIZED", "session": pre["session"], "head": pre["head"],
                                     "plan_hash": pre["plan"]["plan_hash"],
                                     "host_free_gib": host_free_ram_gib()}) + "\n")
        return 0
    if a.cmd == "pilot-summary":
        tp._write_json(Path(a.out), pilot_summary(Path(a.attempt_dir)))
        return 0
    if a.cmd == "calibrate":
        out = calibrate(repo, a.ids.split(","), Path(a.root), a.concurrency, repeat=a.repeat)
        sys.stdout.write(json.dumps({k: out[k] for k in ("concurrency", "wall_s", "host_min_free_gib")}) + "\n")
        return 0
    if a.cmd == "template":
        plan = _read(Path(a.plan))
        if check_plan(plan, frozen=False):
            raise SystemExit(f"plan problems: {check_plan(plan, frozen=False)}")
        tp._write_json(Path(a.out), authorization_template(plan))
        return 0
    if a.cmd == "verify":
        plan = _read(Path(a.plan))
        scen = collection_scenarios(_read(repo / TRAIN_FILE), plan)
        out = verify_collection(plan, Path(a.root), scen, pilot_dir=Path(a.pilot_dir) if a.pilot_dir else None)
        tp._write_json(Path(a.out), out)
        n = sha256sums(Path(a.root), Path(a.root) / "SHA256SUMS")
        sys.stdout.write(json.dumps({"status_counts": out["status_counts"], "full_train_verified":
                                     out["full_train_verified"], "sha256sums_files": n}) + "\n")
        return 0 if out["full_train_verified"] else 1
    plan = live_plan(repo)
    if a.cmd == "verify-plan":
        old = _read(Path(a.plan))
        diff = sorted(k for k in plan_body(plan) if plan_body(old).get(k) != plan_body(plan)[k])
        sys.stdout.write(json.dumps({"same": old.get("plan_hash") == plan["plan_hash"], "differs": diff}) + "\n")
        return 0 if old.get("plan_hash") == plan["plan_hash"] else 1
    tp._write_json(Path(a.out), plan)
    sys.stdout.write(json.dumps({"plan_hash": plan["plan_hash"], "scenarios": len(plan["scenarios"]),
                                 "concurrency": plan["limits"]["max_concurrent_groups"],
                                 "t2_cpu_s_central": plan["cost_estimate"].get("t2_cpu_s_central"),
                                 "geometry_pass": plan["geometry_audit"].get("pass")}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
