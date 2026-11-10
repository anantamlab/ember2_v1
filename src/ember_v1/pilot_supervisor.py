"""Killable five-teacher group runner for the P1-12 TRAIN pilot (issue #45).

One matched group = one scenario, five teachers. Every teacher runs in its OWN spawned process
(one live process per teacher; a fresh process per group; no retries). The worker streams each
proposal to this supervisor; the supervisor's own simulator replays the proposals (it is the
source of truth for every observation, transition and outcome, see `multiteacher.Trajectory`).

Live limits (checked every poll on OS process CPU, not on the teacher's own report):
- per-decision CPU: CPU of the worker process since the current `decide()` began;
- group wall: from the first worker start to the end of the group;
- pilot deadline: an absolute monotonic time shared by all groups of one attempt.
A breach kills the affected worker (decision cap) or every live worker (wall caps). A crash,
a missing response, a code-fingerprint drift or an interruption ends that teacher's trajectory.
Every such event is a recorded teacher failure: the group is incomplete, kept as diagnostic
evidence, never retried, never replaced and never training-eligible.
"""
from __future__ import annotations

import copy
import multiprocessing as mp
import os
import queue as queue_mod
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from . import multiteacher as mt
from .sim import Scenario, Simulator
from .sim.canonical import digest
from .teacher import proposals
from .teacher import registry as reg
from .teacher.confirmation import BEAT, CPU, DEC, AttemptLog, budget_limited, process_cpu_s
from .teacher.registry import CANDIDATES

SUPERVISOR_VERSION = "p1-12-pilot-supervisor@1.1.0"   # 1.1.0: stop hook, scope argument (P1-17)
POLL_S = 0.1
GRACE_S = 5.0              # terminate -> kill grace per worker
MAX_MSGS_PER_POLL = 256    # per worker per poll
STALL_S = 60.0             # heartbeat age that counts as unresponsive (only without an OS CPU reading)
CLOCK = "worker time.thread_time_ns around decide(); caps on supervisor-read OS process CPU"
FAILURE_KINDS = ("teacher_exception", "invalid_proposal", "decision_cpu_cap", "group_wall_cap", "pilot_wall_cap",
                 "worker_crashed", "worker_unresponsive", "missing_policy_response", "code_drift",
                 "identity_drift", "interrupted", "supervisor_error")


class GroupIntegrityError(RuntimeError):
    """The worker stream does not replay in the supervisor's simulator (collector fault, not a teacher result)."""


def _heartbeat(slot: Any) -> None:
    parent = mp.parent_process()
    while True:
        slot[CPU], slot[BEAT] = time.process_time(), time.time()
        if parent is not None and not parent.is_alive():
            os._exit(4)
        time.sleep(0.2)


def worker_main(name: str, identity: Mapping[str, str], seed: int | None, scenario: Mapping[str, Any],
                q: Any, slot: Any) -> None:
    """One teacher on one scenario. Streams ("start" | "step" | "failure" | "done" | "fatal") messages."""
    from .provenance import code_fingerprint
    threading.Thread(target=_heartbeat, args=(slot,), daemon=True).start()
    code = 0
    try:
        cls = reg.teacher_class(identity["name"], identity["version"])
        pol = cls() if seed is None else cls(seed=seed)
        key = {k: identity[k] for k in ("name", "policy", "version", "config_hash")}
        got = {k: pol.identity.to_dict()[k] for k in key}
        if got != key or pol.seed != seed:
            q.put(("fatal", name, "identity_drift", f"worker identity {got} seed {pol.seed}"))
            raise SystemExit(3)
        q.put(("start", name, os.getpid(), code_fingerprint(cls)))
        sc = Scenario.from_dict(scenario)
        sim = Simulator(sc)
        i = 0
        while not sim.terminated:
            obs = sim.observe()
            w0, c0 = time.perf_counter_ns(), time.thread_time_ns()
            slot[DEC] = time.process_time()
            try:
                d = pol.decide(copy.deepcopy(obs))
            except Exception as exc:  # noqa: BLE001 - a teacher failure, reported and never repaired
                q.put(("failure", name, i, "teacher_exception", f"{type(exc).__name__}: {exc}"[:500]))
                break
            finally:
                slot[DEC] = -1.0
            c1, w1 = time.thread_time_ns(), time.perf_counter_ns()
            try:
                raw = proposals.proposal(d)
            except proposals.ProposalError as exc:
                q.put(("failure", name, i, "invalid_proposal", str(exc)[:500]))
                break
            status = mt._decision_status(d)
            q.put(("step", name, i, {"observation_digest": digest(obs), "proposal": raw, "teacher_status": status,
                                     "cpu_ms": (c1 - c0) / 1e6, "wall_ms": (w1 - w0) / 1e6,
                                     "reported_cpu_ms": mt.reported_cpu_ms(d),
                                     "budget_limited": budget_limited(getattr(d, "diagnostics", None))}))
            try:
                prop = proposals.validate_proposal(obs, raw)
            except proposals.ProposalError:
                break                     # the supervisor's replay records the invalid proposal
            sim.step(prop["actions"])
            i += 1
        q.put(("done", name, i, time.process_time()))
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 3
    except BaseException as exc:          # noqa: BLE001 - reported to the supervisor
        q.put(("fatal", name, "worker_crashed", f"{type(exc).__name__}: {exc}"[:500]))
        code = 3
    q.close()
    q.join_thread()
    sys.exit(code)


def _stop(p: Any) -> None:
    if p.is_alive():
        p.terminate()
        p.join(GRACE_S)
    if p.is_alive():
        p.kill()
        p.join(GRACE_S)


def run_group(scenario: Scenario, jobs: Sequence[Mapping[str, Any]], *, decision_cpu_s: Mapping[str, float],
              group_wall_s: float, deadline: float | None = None, log: AttemptLog | None = None,
              target: Callable[..., None] | None = None, poll_s: float = POLL_S,
              stop_requested: Callable[[], bool] | None = None) -> dict[str, Any]:
    """Run one matched group under live limits.

    jobs: one per teacher in CANDIDATES order: {"name", "identity" (registry key), "seed",
    "code_fingerprint" (expected)}. Returns {"streams", "failures", "supervision", "interrupted"}.
    `stop_requested` (checked every poll): True ends every live teacher as an `interrupted` failure.
    """
    log = log or AttemptLog(None)
    names = [j["name"] for j in jobs]
    if names != list(CANDIDATES):
        raise ValueError("a group needs exactly the five teachers in registry order")
    ctx = mp.get_context("spawn")
    # One queue per worker. A worker killed while its feeder thread holds a SHARED queue's write lock would
    # block every other worker (seen on Linux CI). A killed worker's own queue is never read again: a message
    # cut by the kill could block the reader.
    qs = {n: ctx.Queue() for n in names}
    closed: set[str] = set()
    procs: dict[str, Any] = {}
    slots: dict[str, Any] = {}
    streams: dict[str, dict[int, dict[str, Any]]] = {n: {} for n in names}
    failures: dict[str, dict[str, Any]] = {}
    done: dict[str, int] = {}
    seen_fp: dict[str, str] = {}
    cpu = {n: 0.0 for n in names}
    peak = {n: 0.0 for n in names}
    via = {n: "worker_reported" for n in names}
    want_fp = {j["name"]: j["code_fingerprint"] for j in jobs}
    interrupted = False
    t0 = time.monotonic()

    def fail(n: str, kind: str, message: str, **extra: Any) -> None:
        if n not in failures and n not in done:
            failures[n] = {"kind": kind, "message": str(message)[:500], "decisions_received": len(streams[n]),
                           **extra}
            log("teacher_failure", teacher=n, **failures[n])

    def handle(msg: tuple) -> None:
        kind, n = msg[0], msg[1]
        if kind == "start":
            seen_fp[n] = msg[3]
            log("worker_start", teacher=n, pid=msg[2])
            if msg[3] != want_fp[n]:
                fail(n, "code_drift", f"worker code fingerprint {msg[3]} differs from the plan {want_fp[n]}")
                stop(n)
        elif kind == "step":              # kept even after a failure: a true decision made before it
            streams[n][msg[2]] = msg[3]
        elif kind == "failure":
            fail(n, msg[3], msg[4], step_index=msg[2])
        elif kind == "fatal":
            fail(n, msg[2], msg[3])
        elif kind == "done":
            if n not in failures:
                done[n] = msg[2]
                cpu[n] = max(cpu[n], msg[3])
                log("worker_done", teacher=n, decisions=msg[2], process_cpu_s=msg[3])

    def drain_one(n: str) -> int:
        got = 0
        while n not in closed and got < MAX_MSGS_PER_POLL:     # a flooding worker cannot starve the cap checks
            try:
                msg = qs[n].get_nowait()
            except queue_mod.Empty:
                break
            got += 1
            handle(msg)
        return got

    def drain(timeout: float) -> None:
        if not sum(drain_one(n) for n in names):
            time.sleep(timeout)
            for n in names:
                drain_one(n)

    def stop(n: str) -> None:
        if n in closed:
            return
        alive = procs[n].is_alive()
        if alive:
            drain_one(n)          # the writer is alive, so every message it started is complete
        _stop(procs[n])
        if alive:
            closed.add(n)         # killed by the supervisor: its last message may be cut

    try:
        for j in jobs:
            n = j["name"]
            slots[n] = ctx.Array("d", [0.0, 0.0, -1.0], lock=False)
            procs[n] = ctx.Process(target=target or worker_main, name=f"p1-12-pilot-{n}", daemon=True,
                                   args=(n, dict(j["identity"]), j["seed"], scenario.to_dict(), qs[n], slots[n]))
            procs[n].start()
        log("group_start", scenario_hash=scenario.content_hash(), pids={n: p.pid for n, p in procs.items()})
        while True:
            drain(poll_s)
            for n, p in procs.items():
                if not p.is_alive() or n in failures:
                    continue
                ext = process_cpu_s(p.pid)
                if ext is not None:
                    via[n] = "os"
                now_cpu = ext if ext is not None else slots[n][CPU]
                cpu[n] = max(cpu[n], now_cpu, slots[n][CPU])
                start = slots[n][DEC]
                if start >= 0:
                    d_s = now_cpu - start
                    peak[n] = max(peak[n], d_s * 1000)
                    if d_s > decision_cpu_s[n]:
                        fail(n, "decision_cpu_cap", f"decision used {d_s:.2f} s CPU, cap {decision_cpu_s[n]} s",
                             decision_cpu_s=round(d_s, 3))
                        stop(n)
                        continue
                if ext is None and slots[n][BEAT] > 0 and time.time() - slots[n][BEAT] > STALL_S:
                    fail(n, "worker_unresponsive", f"no heartbeat for {STALL_S} s")
                    stop(n)
            now = time.monotonic()
            live = [n for n, p in procs.items() if p.is_alive()]
            if live and now - t0 > group_wall_s:
                for n in live:
                    fail(n, "group_wall_cap", f"group wall {now - t0:.1f} s over cap {group_wall_s} s")
                    stop(n)
            elif live and deadline is not None and now > deadline:
                for n in live:
                    fail(n, "pilot_wall_cap", "pilot wall deadline reached")
                    stop(n)
            elif live and stop_requested is not None and stop_requested():
                interrupted = True
                for n in live:
                    fail(n, "interrupted", "stop requested (parent supervisor gone or operator stop)")
                    stop(n)
            if all(not p.is_alive() for p in procs.values()):
                drain(0.05)
                while sum(drain_one(n) for n in names):   # every worker has exited: read what is left
                    pass
                for n, p in procs.items():
                    if n not in done and n not in failures:
                        if p.exitcode == 0:
                            fail(n, "missing_policy_response", "worker exited without finishing the episode")
                        else:
                            fail(n, "worker_crashed", f"worker exit code {p.exitcode}", exit_code=p.exitcode)
                break
    except KeyboardInterrupt:
        interrupted = True
        for n in names:
            fail(n, "interrupted", "operator interrupt")
    except Exception as exc:              # noqa: BLE001 - supervisor fault: fail closed, keep evidence
        for n in names:
            fail(n, "supervisor_error", f"{type(exc).__name__}: {exc}")
    finally:
        for p in procs.values():
            _stop(p)
        codes = {n: p.exitcode for n, p in procs.items()}
        for q in qs.values():
            q.close()
            q.cancel_join_thread()
    sup = {"supervisor_version": SUPERVISOR_VERSION, "wall_s": round(time.monotonic() - t0, 3),
           "process_cpu_s": {n: round(v, 3) for n, v in cpu.items()}, "cpu_measurement": via,
           "peak_live_decision_cpu_ms": {n: round(v, 1) for n, v in peak.items()}, "exit_codes": codes,
           "worker_code_fingerprints": seen_fp, "decisions_received": {n: len(streams[n]) for n in names},
           "budget_limited_decisions": {n: sum(1 for s in streams[n].values() if s.get("budget_limited"))
                                        for n in names},
           "limits": {"decision_cpu_s": dict(decision_cpu_s), "group_wall_s": group_wall_s}}
    log("group_end", failures=sorted(failures), wall_s=sup["wall_s"])
    return {"streams": streams, "failures": failures, "done": done, "supervision": sup, "interrupted": interrupted}


def build_group(scenario: Scenario, run: Mapping[str, Any], identities: Mapping[str, Mapping[str, Any]],
                seeds: Mapping[str, Any], prov: Mapping[str, Any], *, extra: Mapping[str, Any],
                run_kind: str = "primary", scope: str = mt.SCOPE_TRAIN_PILOT) -> dict[str, Any]:
    """Replay every streamed proposal in the supervisor's simulator and seal the group report.

    Raise GroupIntegrityError if a stream does not replay (observation digest or step order mismatch).
    """
    result = {}
    for n in CANDIDATES:
        traj = mt.Trajectory(scenario, n, clock=CLOCK)
        steps = run["streams"][n]
        if sorted(steps) != list(range(len(steps))):
            raise GroupIntegrityError(f"{n}: streamed steps are not contiguous")
        for i in range(len(steps)):
            s = steps[i]
            if traj.done:
                raise GroupIntegrityError(f"{n}: step {i} streamed after the episode ended")
            obs = traj.observation()
            if digest(obs) != s["observation_digest"]:
                raise GroupIntegrityError(f"{n} step {i}: the worker saw another observation")
            try:
                traj.apply(obs, s["proposal"], status=s["teacher_status"], cpu_ms=s["cpu_ms"],
                           wall_ms=s["wall_ms"], reported=s["reported_cpu_ms"])
            except mt._TeacherFailure as f:
                traj.fail(f.kind, str(f))
                break
            except mt.MultiTeacherError as e:
                raise GroupIntegrityError(f"{n} step {i}: {e}") from None
        f = run["failures"].get(n)
        if f is not None:
            traj.fail(f["kind"], f["message"])
        elif n not in run["done"]:
            traj.fail("missing_policy_response", "no completion message")
        try:
            result[n] = traj.finish(identities[n], seeds.get(n))
        except mt.MultiTeacherError as e:
            raise GroupIntegrityError(str(e)) from None
    try:
        return mt.assemble_group(scenario, result, prov, scope=scope,
                                 extra={**dict(extra), "runtime": {"run": run_kind,
                                                                    "supervision": run["supervision"]}})
    except mt.MultiTeacherError as e:
        raise GroupIntegrityError(str(e)) from None
