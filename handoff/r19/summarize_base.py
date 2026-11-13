import json, pathlib, collections
d = json.loads(pathlib.Path("/mnt/c/Users/Admin/Documents/Test/ws-tmp/base_compare.json").read_text())
POL = ["V1_A_raw","V1_B_mask","V1_C_coord","BASE_A_raw","BASE_B_mask","BASE_C_coord","T0","T1","T2","T3","T4"]
tot = {p: collections.Counter() for p in POL}
for s, r in d["sets"].items():
    for p in POL:
        tot[p].update(r["policies"][p])
print(f"{'policy':16s}{'people':>7s}{'cas/p':>8s}{'unres/p':>9s}{'cu/p':>8s}{'evac/p':>8s}{'blocked':>9s}{'failed':>7s}")
for p in POL:
    t = tot[p]
    print(f"{p:16s}{t['people']:7d}{t['casualties']/t['people']:8.4f}{t['unresolved']/t['people']:9.4f}"
          f"{t['cu']/t['people']:8.4f}{t['evacuated']/t['people']:8.4f}{t['blocked']:9d}{t['failed']:7d}")

print("\ncontrasts (cu rate, family bootstrap 95% CI):")
want = {("V1_A_raw","BASE_A_raw"),("V1_B_mask","BASE_B_mask"),("V1_C_coord","BASE_C_coord"),
        ("BASE_C_coord","T0"),("V1_C_coord","T0"),("V1_C_coord","T4"),("BASE_A_raw","T0"),("V1_A_raw","BASE_A_raw")}
for s, r in d["sets"].items():
    print("--", s)
    for name, c in r["contrasts"].items():
        a, b = name.split("-", 1)
        if (a, b) in want and a in ("V1_A_raw","V1_B_mask","V1_C_coord","BASE_A_raw","BASE_B_mask","BASE_C_coord"):
            x = c["cu"]
            print(f"   {name:28s} d={x['diff']:+.4f}  CI[{x['ci95'][0]:+.4f},{x['ci95'][1]:+.4f}]  "
                  f"{'*' if x['excludes_zero'] else 'ns'}  (families {x['families']})")
