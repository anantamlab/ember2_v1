"""S1 storage and CPU estimate from a SYNTHETIC probe plan (issue #10). Labels nothing from p1-02-v1.

Usage: python experiments/s1_storage_estimate.py <out.json>

The probe plan uses the p1-02-v1 generator spec with another master seed and plan id. Its scenarios are checked
disjoint (family, lineage, scenario, core geometry) from every p1-02-v1 split, every pilot namespace and the
p1-09 confirmation set before T1 runs. T1 actions are measured for size only and are not written anywhere.
"""
import json
import sys
import time
import zlib
from pathlib import Path

from ember_v1 import s1_pipeline as sp
from ember_v1.gen.splits import DEFAULT_PLAN, SplitPlan, build_splits, manifest
from ember_v1.sim import Simulator
from ember_v1.sim.canonical import canonical_json
from ember_v1.teacher import confirmation as C
from ember_v1.teacher.t1 import T1CongestionAware

REPO = Path(__file__).resolve().parents[1]
PROBE = {"plan_id": "s1-storage-probe-1", "master_seed": 991,
         "families": {"train": 24, "dev": 1, "calibration": 1, "locked_test": 1}, "lineages_per_family": 2}


def main(out_path: str) -> int:
    d = dict(DEFAULT_PLAN)
    d.update(PROBE)
    build = build_splits(SplitPlan.from_dict(d))
    probe_m = manifest(build, "train")
    historical, _, _ = C.historical_sets()
    conf = json.loads((REPO / "data/selection/p1-09-selection@1/manifest.json").read_text(encoding="utf-8"))
    held = {ns: sp.heldout_from_entries(e) for ns, e in historical.items()}
    held["p1-09-selection@1"] = sp.heldout_from_manifest(conf)
    mine = sp.heldout_from_manifest(probe_m)
    overlap = {ns: {k: len(mine[k] & h[k]) for k in sp.HELD_OUT_KEYS} for ns, h in held.items()}
    if any(v for o in overlap.values() for v in o.values()):
        print(json.dumps(overlap, indent=1))
        return 2
    pol = T1CongestionAware()
    samples, cpu_ms, decisions = [], [], 0
    for g in build.scenarios["train"]:
        s = g.scenario
        sim, recs, traces = Simulator(s), [], []
        while not sim.terminated:
            o = sim.observe()
            t0 = time.process_time()
            dd = pol.decide(o)
            cpu_ms.append((time.process_time() - t0) * 1e3)
            traces.append({"observation": o, "actions": dd.actions, "group_id": "g", "source": "T1@1.0.0",
                           "step": len(recs) + 1, "provenance": {"record_hash": "sha256:" + "0" * 64}})
            recs.append(sim.step(dd.actions))
        decisions += len(recs)
        member = canonical_json({"member_format": "s1-shard-member@1.0.0",
                                 "episode": {"scenario": s.to_dict(), "seed": None, "records": recs},
                                 "traces": traces}).encode()
        # A real provenance record is about 1.3 kB; add the difference to the placeholder so sizes stay upper.
        member_bytes = len(member) + 1300 * len(traces)
        samples.append({"people": len(s.people), "walls": len(s.obstacles), "steps": len(recs), "max_steps": s.max_steps,
                        "member_bytes": member_bytes, "zip_bytes": len(zlib.compress(member, 6)) + 700 * len(traces)})
    train = json.loads((REPO / "data/splits/p1-02-v1/train.json").read_text(encoding="utf-8"))
    est = sp.storage_estimate(samples, train)
    mean_cpu = sum(cpu_ms) / len(cpu_ms)
    upper_decisions = est["decisions"]["upper"]
    report = {"probe_plan": PROBE, "probe_episodes": len(samples), "probe_decisions": decisions,
              "probe_overlap_with_heldout": overlap, "probe_people": [s["people"] for s in samples],
              "estimate": est,
              "t1_cpu": {"mean_decision_ms": round(mean_cpu, 2), "max_decision_ms": round(max(cpu_ms), 2),
                         "train_upper_decisions": upper_decisions,
                         "train_upper_cpu_s": round(upper_decisions * max(cpu_ms) / 1e3, 1),
                         "note": "upper = every train scenario runs to max_steps at the probe max decision CPU"},
              "provenance_record_bytes_assumed": 1300, "note": "synthetic probe only; no p1-02-v1 label made"}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: report[k] for k in ("probe_episodes", "probe_decisions", "estimate", "t1_cpu")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
