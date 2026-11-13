import json, pathlib, statistics, collections
B = pathlib.Path.home()/"ember-ws/runs/r19-base-control@1/run"
V1 = pathlib.Path.home()/"ember-ws/runs/r19-v1-arms@1/attempt-1/final"
AO = ["east","north","south","stay","west"]
NAME = {"A_raw": "raw", "B_static_mask": "mask", "C_joint_coord": "coord"}

def agg(p):
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    pe = sum(r["people"] for r in rows)
    mo = collections.Counter()
    for r in rows:
        for k, v in (r.get("movement_outcomes") or {}).items():
            mo[k] += v
    blocked = sum(v for k, v in mo.items() if k.startswith("blocked_"))
    moved = mo.get("moved", 0)
    return {"scenarios": len(rows), "people": pe,
            "evacuated": sum(r["evacuated"] for r in rows),
            "casualties": sum(r["casualties"] for r in rows),
            "unresolved": sum(r["unresolved"] for r in rows),
            "cu": sum(r["casualties"]+r["unresolved"] for r in rows),
            "blocked": blocked, "attempted": moved+blocked, "collisions": sum(r["collisions"] for r in rows),
            "steps_mean": round(statistics.fmean(r["steps"] for r in rows), 1),
            "failures": sum(1 for r in rows if r.get("failure")),
            "maxsteps": sum(1 for r in rows if r.get("termination_reason") == "max_steps")}

print(f"{'policy':22s}{'scn':>5s}{'people':>7s}{'cas/p':>8s}{'unres/p':>9s}{'cu/p':>8s}{'evac/p':>8s}{'blocked':>9s}{'blk/att':>8s}{'steps':>7s}{'maxstp':>7s}")
for label, root in (("V1", V1), ("BASE", B)):
    for a in ("A_raw","B_static_mask","C_joint_coord"):
        g = agg(root/f"rollouts-{a}.jsonl")
        print(f"{label+'/'+NAME[a]:22s}{g['scenarios']:5d}{g['people']:7d}{g['casualties']/g['people']:8.4f}"
              f"{g['unresolved']/g['people']:9.4f}{g['cu']/g['people']:8.4f}{g['evacuated']/g['people']:8.4f}"
              f"{g['blocked']:9d}{(g['blocked']/g['attempted'] if g['attempted'] else 0):8.4f}{g['steps_mean']:7.1f}"
              f"{g['maxsteps']:7d}{'  FAIL='+str(g['failures']) if g['failures'] else ''}")

print("\naction distribution (raw argmax vs executed):")
for label, root in (("V1", V1), ("BASE", B)):
    for a in ("A_raw","B_static_mask","C_joint_coord"):
        rows = [json.loads(l) for l in (root/f"rollouts-{a}.jsonl").read_text().splitlines() if l.strip()]
        ex, raw = collections.Counter(), collections.Counter()
        for r in rows:
            for key, c in (("trace", ex), ("raw_trace", raw)):
                for s in r[key]:
                    c.update([AO[int(x)] for x in s] if isinstance(s, str) else list(s.values()))
        ne, nr = sum(ex.values()), sum(raw.values())
        f = lambda c, n: " ".join(f"{k}={c[k]/n:.2f}" for k in ("north","south","east","west","stay"))
        print(f"  {label}/{NAME[a]:6s} raw[{f(raw,nr)}]  exec[{f(ex,ne)}]")
