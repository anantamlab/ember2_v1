"""P1-09 confirmation evaluation (post-run). Verdict only from the frozen `selection.selection_gate`.

Usage: python evaluate_confirmation.py <evidence.json> <out_dir>
Writes <out_dir>/selection_gate.json (gate output, verbatim) and <out_dir>/statistics.json (descriptive
tables). On incomplete evidence it writes only a failure record and no verdict. Never changes the evidence.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

from ember_v1.sim.canonical import digest
from ember_v1.teacher import confirmation as C
from ember_v1.teacher import selection as sel

PC = "64b2d9873953ba750fd6c08fb84761f1bcd8e0bb"
BASE = "T1"


def dump(path: Path, doc) -> None:
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def three_way(deltas):
    return {"better": sum(d < 0 for d in deltas), "equal": sum(d == 0 for d in deltas),
            "worse": sum(d > 0 for d in deltas)}


def main(ev_path: str, out_dir: str) -> int:
    ev_path, out = Path(ev_path), Path(out_dir)
    evd = json.loads(ev_path.read_text(encoding="utf-8"))
    hash_ok = digest({k: v for k, v in evd.items() if k != "evidence_hash"}) == evd["evidence_hash"]
    sup = evd["supervision"]
    head = {"evidence_path": ev_path.as_posix(), "evidence_hash": evd["evidence_hash"], "evidence_hash_ok": hash_ok,
            "complete": evd["complete"], "errors": evd["errors"], "stage": sup.get("stage"), "abort": sup.get("abort"),
            "supervision": {k: v for k, v in sup.items() if k != "limits"}, "limits": sup.get("limits")}
    rows = evd["rows"]
    eligible = list(C.reg.CANDIDATES)
    complete = (evd["complete"] is True and not evd["errors"] and hash_ok and len(rows) == 72
                and all(set(r["results"]) == set(eligible) for r in rows))
    if not complete:
        head["verdict"] = None
        head["note"] = "INCOMPLETE: no selection verdict is computed (protocol stop rule)."
        head["partial_results_received"] = sup.get("results_received")
        dump(out / "failure_record.json", head)
        print(json.dumps(head, indent=1, sort_keys=True))
        return 2

    protocol, _ = C.trust.load_trusted_json(C.ev.REPO, {"kind": "git", "commit": PC,
                                                        "path": "data/selection/p1-09-selection@1/protocol.json"})
    historical, _, _ = C.historical_sets()
    gate = sel.selection_gate(protocol, evd, repo=C.ev.REPO, historical=historical,
                              protocol_ref={"kind": "git", "commit": PC,
                                            "path": "data/selection/p1-09-selection@1/protocol.json"})
    dump(out / "selection_gate.json", gate)

    # Descriptive tables (not the verdict).
    per = {}
    for c in eligible:
        res = [r["results"][c] for r in rows]
        dec = sum(x["decisions"] for x in res)
        cpu = sum(x["policy_cpu_s"] for x in res)
        per[c] = {"casualties": sum(x["casualties"] for x in res), "evacuated": sum(x["evacuated"] for x in res),
                  "unresolved": sum(x["unresolved"] for x in res), "people": sum(x["people"] for x in res),
                  "casualties_plus_unresolved": sum(x["casualties"] + x["unresolved"] for x in res),
                  "policy_cpu_s": round(cpu, 3), "process_cpu_s": sup["process_cpu_s"].get(c),
                  "decisions": dec, "mean_decision_cpu_ms": round(1000 * cpu / dec, 3) if dec else None,
                  "max_decision_cpu_ms": max(x["max_decision_cpu_ms"] for x in res),
                  "peak_live_decision_cpu_ms": sup["peak_live_decision_cpu_ms"].get(c),
                  "budget_limited_decisions": sum(x["budget_limited_decisions"] for x in res),
                  "failures": sum(x["failure"] is not None for x in res)}
    comp = {}
    for c in eligible:
        if c == BASE:
            continue
        sd = [(r["scenario_id"], r["family_id"], r["results"][c]["casualties"] - r["results"][BASE]["casualties"],
               r["results"][c]["people"]) for r in rows]
        fam = defaultdict(int)
        for _, f, d, _ in sd:
            fam[f] += d
        worse = sorted(({"scenario_id": s, "family_id": f, "casualty_delta": d, "people": n,
                         "cand_unresolved_delta": next(r["results"][c]["unresolved"] - r["results"][BASE]["unresolved"]
                                                       for r in rows if r["scenario_id"] == s)}
                        for s, f, d, n in sd if d > 0), key=lambda x: (-x["casualty_delta"], x["scenario_id"]))
        a = gate.get("analyses", {}).get(c, {})
        comp[c] = {"scenarios": three_way([d for *_, d, _ in sd]), "families": three_way(list(fam.values())),
                   "mean_paired_casualty_diff": a.get("mean_paired_casualty_diff"), "ci95": a.get("ci95"),
                   "sign_test_p": a.get("sign_test_p"), "sign_test_p_holm": a.get("sign_test_p_holm"),
                   "superior": a.get("superior"), "safety_pass": a.get("safety_pass"),
                   "compute_feasible": a.get("compute_feasible"), "qualifies": a.get("qualifies"),
                   "worse_scenarios": worse}
    stats = {**head, "verdict": gate["decision"], "selected": gate["selected"], "gate_reasons": gate["reasons"],
             "per_teacher": per, "vs_T1": comp, "multiplicity": gate.get("multiplicity"),
             "note": "Verdict from selection.selection_gate (frozen). 'worse_scenarios' lists every scenario with "
                     "more casualties than T1 (descriptive; the protocol defines no separate severity threshold)."}
    dump(out / "statistics.json", stats)
    print(json.dumps({k: stats[k] for k in ("verdict", "selected", "gate_reasons", "complete")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
