"""Arm A/B/C report from the executed rollouts (raw vs executed actions kept apart)."""
import json, pathlib, statistics, collections, sys

rd = pathlib.Path.home()/"ember-ws/runs/r19-v1-arms@1/attempt-1/final"
ARMS = ["A_raw", "B_static_mask", "C_joint_coord"]
data = {}
for a in ARMS:
    p = rd/f"rollouts-{a}.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    data[a] = rows

def agg(rows):
    pe = sum(r["people"] for r in rows)
    mo = collections.Counter()
    for r in rows:
        for k, v in (r.get("movement_outcomes") or {}).items():
            mo[k] += v
    blocked = sum(v for k, v in mo.items() if k.startswith("blocked_"))
    moved = mo.get("moved", 0)
    lat = [r["decision_wall_ms_p50"] for r in rows if r.get("decision_wall_ms_p50") is not None]
    return {
        "scenarios": len(rows),
        "people": pe,
        "evacuated": sum(r["evacuated"] for r in rows),
        "casualties": sum(r["casualties"] for r in rows),
        "unresolved": sum(r["unresolved"] for r in rows),
        "cu": sum(r["casualties"] + r["unresolved"] for r in rows),
        "moved": moved, "blocked": blocked,
        "moves_attempted": moved + blocked,
        "collisions": sum(r["collisions"] for r in rows),
        "fw": mo.get("blocked_wall", 0), "fb": mo.get("blocked_boundary", 0),
        "ff": mo.get("blocked_fire", 0), "fo": mo.get("blocked_occupied", 0),
        "fc": mo.get("blocked_conflict", 0), "fcy": mo.get("blocked_cycle", 0),
        "decisions": sum(r["decisions"] for r in rows),
        "failures": collections.Counter(r["failure"] for r in rows if r.get("failure")),
        "lat_p50_ms": round(statistics.fmean(lat), 1) if lat else None,
    }

print(f"{'metric':38s}" + "".join(f"{a:>16s}" for a in ARMS))
aggs = {a: agg(data[a]) for a in ARMS}
def line(label, fn, fmt="{:.4f}"):
    print(f"{label:38s}" + "".join(fmt.format(fn(aggs[a])).rjust(16) for a in ARMS))
line("scenarios", lambda x: x["scenarios"], "{:.0f}")
line("people", lambda x: x["people"], "{:.0f}")
line("casualties", lambda x: x["casualties"], "{:.0f}")
line("unresolved", lambda x: x["unresolved"], "{:.0f}")
line("casualties+unresolved", lambda x: x["cu"], "{:.0f}")
line("casualties/person", lambda x: x["casualties"]/x["people"])
line("unresolved/person", lambda x: x["unresolved"]/x["people"])
line("combined failure/person", lambda x: x["cu"]/x["people"])
line("evacuation success (evac/person)", lambda x: x["evacuated"]/x["people"])
line("blocked moves", lambda x: x["blocked"], "{:.0f}")
line("moves attempted", lambda x: x["moves_attempted"], "{:.0f}")
line("blocked rate (/attempted move)", lambda x: x["blocked"]/x["moves_attempted"] if x["moves_attempted"] else 0)
line("  wall", lambda x: x["fw"], "{:.0f}")
line("  boundary", lambda x: x["fb"], "{:.0f}")
line("  fire", lambda x: x["ff"], "{:.0f}")
line("  occupied", lambda x: x["fo"], "{:.0f}")
line("  conflict", lambda x: x["fc"], "{:.0f}")
line("  cycle", lambda x: x["fcy"], "{:.0f}")
line("collisions (occ+conf+cycle)", lambda x: x["collisions"], "{:.0f}")
line("decision p50 latency (ms)", lambda x: x["lat_p50_ms"], "{:.1f}")
for a in ARMS:
    print(a, "failures:", dict(aggs[a]["failures"]) or "none")

print("\n=== raw -> executed action changes (postprocess) ===")
for a in ARMS[1:]:
    tot = collections.Counter()
    per_set = collections.defaultdict(collections.Counter)
    for r in data[a]:
        pc = r.get("postprocess_changes") or {}
        for k in ("mask", "coord", "steps_changed", "coord_iterations"):
            tot[k] += pc.get(k, 0)
            per_set[r["set"]][k] += pc.get(k, 0)
    print(a, dict(tot), "| decisions", aggs[a]["decisions"])
    for s, c in sorted(per_set.items()):
        print("   ", s, dict(c))

print("\n=== per set ===")
for s in sorted({r["set"] for r in data["A_raw"]}):
    print("--", s)
    for a in ARMS:
        rows = [r for r in data[a] if r["set"] == s]
        g = agg(rows)
        print(f"   {a:15s} n={g['scenarios']:3d} people={g['people']:5d} cas/p={g['casualties']/g['people']:.4f}"
              f" unres/p={g['unresolved']/g['people']:.4f} cu/p={g['cu']/g['people']:.4f}"
              f" evac/p={g['evacuated']/g['people']:.3f}"
              f" blocked={g['blocked']}/{g['moves_attempted']} ({g['blocked']/g['moves_attempted'] if g['moves_attempted'] else 0:.3f})"
              f" lat50={g['lat_p50_ms']}")

# identity of raw traces across arms (same model must agree on raw argmax)
def raw_sig(rows):
    return {r["scenario_id"]: json.dumps(r.get("raw_trace")) for r in rows}
a_sig, b_sig, c_sig = raw_sig(data["A_raw"]), raw_sig(data["B_static_mask"]), raw_sig(data["C_joint_coord"])
print("\nraw argmax identical across arms: A==B", a_sig == b_sig, "A==C", a_sig == c_sig)
