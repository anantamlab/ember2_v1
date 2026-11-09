"""Spawn-importable fake worker targets for the P1-12 pilot supervisor tests (no TRAIN data).

Each target has the `pilot_supervisor.worker_main` signature. Real teachers run unless the fake
targets the named teacher.
"""
from __future__ import annotations

import os
import time

from ember_v1 import pilot_supervisor as ps
from ember_v1.sim.canonical import digest
from ember_v1.teacher.confirmation import DEC


def _burn(seconds):
    t = time.process_time()
    while time.process_time() - t < seconds:
        pass


def crash_t2(name, identity, seed, scenario, q, slot):
    if name == "T2":
        os._exit(7)
    ps.worker_main(name, identity, seed, scenario, q, slot)


def hang_t3_in_decide(name, identity, seed, scenario, q, slot):
    if name == "T3":
        q.put(("start", name, os.getpid(), _fp(name, identity)))
        slot[DEC] = time.process_time()                  # a decide() that never returns
        _burn(60)
    ps.worker_main(name, identity, seed, scenario, q, slot)


def flood_then_hang_t2(name, identity, seed, scenario, q, slot):
    """T2 keeps its queue feeder busy with large messages while it burns CPU in decide(); the decision cap
    kills it mid-write. With one SHARED queue this could leave the write lock held and block the others."""
    if name == "T2":
        q.put(("start", name, os.getpid(), _fp(name, identity)))
        slot[DEC] = time.process_time()
        blob = "x" * (1 << 20)
        end = time.monotonic() + 60
        while time.monotonic() < end:
            q.put(("heartbeat_noise", name, blob))           # unknown kind: ignored by the supervisor
    ps.worker_main(name, identity, seed, scenario, q, slot)


def sleep_t1(name, identity, seed, scenario, q, slot):
    if name == "T1":
        q.put(("start", name, os.getpid(), _fp(name, identity)))
        time.sleep(60)                                   # blocked without CPU: only the wall cap ends it
    ps.worker_main(name, identity, seed, scenario, q, slot)


def silent_exit_t4(name, identity, seed, scenario, q, slot):
    if name == "T4":
        q.put(("start", name, os.getpid(), _fp(name, identity)))
        q.close()
        q.join_thread()
        os._exit(0)                                      # exit 0 with no decisions and no "done"
    ps.worker_main(name, identity, seed, scenario, q, slot)


def invalid_action_t0(name, identity, seed, scenario, q, slot):
    if name == "T0":
        from ember_v1.sim import Scenario, Simulator
        q.put(("start", name, os.getpid(), _fp(name, identity)))
        obs = Simulator(Scenario.from_dict(scenario)).observe()
        from ember_v1.teacher.proposals import PROPOSAL_VERSION
        body = {"proposal_version": PROPOSAL_VERSION,
                "teacher": {k: identity[k] for k in ("name", "policy", "version", "config_hash")},
                "observation_digest": digest(obs), "step": obs["step"],
                "actions": {p["id"]: "fly" for p in obs["people"] if p["status"] == "active"},
                "values": None, "policy_seed": None}
        bad = {**body, "compute": {"cpu_ms": 0.0, "wall_ms": 0.0}, "proposal_hash": digest(body)}
        q.put(("step", name, 0, {"observation_digest": digest(obs), "proposal": bad, "teacher_status": "ok",
                                 "cpu_ms": 0.0, "wall_ms": 0.0, "reported_cpu_ms": None, "budget_limited": False}))
        q.put(("done", name, 1, 0.0))
        q.close()
        q.join_thread()
        os._exit(0)
    ps.worker_main(name, identity, seed, scenario, q, slot)


def forged_observation_t1(name, identity, seed, scenario, q, slot):
    if name == "T1":
        q.put(("start", name, os.getpid(), _fp(name, identity)))
        q.put(("step", name, 0, {"observation_digest": "sha256:" + "0" * 64, "proposal": {}, "teacher_status": "ok",
                                 "cpu_ms": 0.0, "wall_ms": 0.0, "reported_cpu_ms": None, "budget_limited": False}))
        q.put(("done", name, 1, 0.0))
        q.close()
        q.join_thread()
        os._exit(0)
    ps.worker_main(name, identity, seed, scenario, q, slot)


def _fp(name, identity):
    from ember_v1.provenance import code_fingerprint
    from ember_v1.teacher import registry as reg
    return code_fingerprint(reg.teacher_class(identity["name"], identity["version"]))
