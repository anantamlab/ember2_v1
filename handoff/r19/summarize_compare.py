import json, pathlib
d = json.loads(pathlib.Path("/mnt/c/Users/Admin/Documents/Test/ws-tmp/arms_compare.json").read_text())
POL = ["A_raw","B_static_mask","C_joint_coord","T0","T1","T2","T3","T4"]
print("version", d["version"], "unit", d["unit"], "bootstrap", d["bootstrap"])
tot = {p: {k:0 for k in ("people","evacuated","casualties","unresolved","cu","blocked","collisions","failed")} for p in POL}
for st, r in d["sets"].items():
    print(f"\n=== {st}: {r['scenarios']} scenarios, {r['families']} families, missing_pairs={len(r['missing_pairs'])}")
    print(f"{'policy':16s}{'people':>7s}{'cas/p':>8s}{'unres/p':>9s}{'cu/p':>8s}{'evac/p':>8s}{'blocked':>9s}{'blk/move':>10s}{'failed':>7s}")
    for p in POL:
        x = r["policies"][p]
        for k in tot[p]: tot[p][k] += x[k]
        print(f"{p:16s}{x['people']:7d}{x['casualties_rate']:8.4f}{x['unresolved_rate']:9.4f}"
              f"{x['cu_rate']:8.4f}{x['evacuated_rate']:8.4f}{x['blocked']:9d}{x['blocked_rate']:10.4f}{x['failed']:7d}")
print("\n=== ALL 148 matched scenarios pooled ===")
print(f"{'policy':16s}{'people':>7s}{'cas/p':>8s}{'unres/p':>9s}{'cu/p':>8s}{'evac/p':>8s}{'blocked':>9s}{'failed':>7s}")
for p in POL:
    t = tot[p]
    print(f"{p:16s}{t['people']:7d}{t['casualties']/t['people']:8.4f}{t['unresolved']/t['people']:9.4f}"
          f"{t['cu']/t['people']:8.4f}{t['evacuated']/t['people']:8.4f}{t['blocked']:9d}{t['failed']:7d}")

print("\n=== contrasts (A/B/C minus reference), cu rate diff [95% CI over families] ===")
for st, r in d["sets"].items():
    print(f"-- {st}")
    for name, c in r["contrasts"].items():
        if name.split("-")[0] not in ("A_raw","B_static_mask","C_joint_coord"): continue
        x = c["cu"]
        star = "*" if x["excludes_zero"] else " "
        print(f"   {name:34s} d={x['diff']:+.4f} CI[{x['ci95'][0]:+.4f},{x['ci95'][1]:+.4f}] {star}")
