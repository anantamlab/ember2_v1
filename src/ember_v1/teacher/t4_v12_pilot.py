"""T4 v1.2.0 independent synthetic confirmation pilot, preregistered in docs/P1_T4_V12_NODE_BUDGET.md.

    python -m ember_v1.teacher.t4_v12_pilot --exploratory [--out PATH]   # pre-freeze smoke (t4-explore@v12)
    python -m ember_v1.teacher.t4_v12_pilot [--out PATH]                 # the ONE confirmation run (t4-pilot@3)

The confirmation runs ONCE. A failed gate is a result; the protocol forbids changing v1.2.0 and re-running this
sample. One CPU process, no GPU, no network. No T4 decision on DEV and no DEV comparison: a pass only makes T4
ready for the independent selection evaluation (issue #25), it does not select T4.

1 Cells: the v2 cell design (`t4_v2_pilot.pilot_cells_v2`, 74 cells) under namespace `t4-pilot@3`. A cell is
  rejected, recorded and not replaced if its content hash or seed equals a v1 pilot, v2 pilot or exploratory
  (`t4-explore@v12`) scenario, if its layout equals an inspected crowded v1 cell, or if it duplicates a cell.
2 Runs and gate: `t4_v2_pilot.build_report` with the v1.2.0 identities (A0, A1, A2 closed loop; neutral paired
  states on frozen T1; frozen-T1 DEV workload profile for the cost estimate). G1-G5c are `gate_v2` unchanged.
3 Decision: T4_READY_FOR_INDEPENDENT_SELECTION_EVALUATION if no check fails, else T4_V12_BLOCKED.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from ..sim.scenario import Scenario
from . import evaluate as ev
from . import t4_v2_pilot as v2
from .t4_pilot import pilot_cells, pilot_scenario
from .t4_v12 import ABLATIONS_V12

PILOT_VERSION = "t4-v12-confirmation-pilot@1.0.0"
NAMESPACE = "t4-pilot@3"
EXPLORATORY_NAMESPACE = "t4-explore@v12"
SID_PREFIX = {NAMESPACE: "t4-pilot3", EXPLORATORY_NAMESPACE: "t4-explore12"}
DEFAULT_OUT = ev.REPO / "data/baselines/p1-08-t4-v12-pilot.json"
SMOKE_OUT = ev.REPO / "data/exploratory/p1-08-t4-v12-smoke.json"
READY = "T4_READY_FOR_INDEPENDENT_SELECTION_EVALUATION"
BLOCKED = "T4_V12_BLOCKED"


def scenario(cell: dict[str, Any], namespace: str = NAMESPACE) -> Scenario:
    return v2.pilot_scenario_v2(cell, namespace=namespace, sid_prefix=SID_PREFIX[namespace])


def select_cells(namespace: str = NAMESPACE
                 ) -> tuple[list[tuple[dict[str, Any], Scenario]], list[dict[str, Any]], dict[str, Any]]:
    other = EXPLORATORY_NAMESPACE if namespace == NAMESPACE else NAMESPACE
    v1 = [pilot_scenario(c) for c in pilot_cells()]
    v2s = [v2.pilot_scenario_v2(c) for c in v2.pilot_cells_v2()]
    ot = [scenario(c, other) for c in v2.pilot_cells_v2()]
    sets = (("v1_pilot", v1), ("v2_pilot", v2s), (other.replace("@", "_"), ot))
    content = {name: {v2.content_hash(s) for s in ss} for name, ss in sets}
    seeds = {name: {s.seed for s in ss} for name, ss in sets}
    inspected = {v2.layout_hash(pilot_scenario({**c, "mode": "full_episode"})) for c in v2.INSPECTED_V1}
    kept, rejected, seen = [], [], set()
    for cell in v2.pilot_cells_v2():
        s = scenario(cell, namespace)
        ch, lh = v2.content_hash(s), v2.layout_hash(s)
        why = next((f"content_equals_{n}" for n, _ in sets if ch in content[n]), None) or \
            next((f"seed_equals_{n}" for n, _ in sets if s.seed in seeds[n]), None) or \
            ("layout_equals_inspected_v1_crowded_cell" if lh in inspected else
             "duplicate_content" if ch in seen else None)
        if why:
            rejected.append({"cell": v2._cell_json(cell), "reason": why, "content_hash": ch})
            continue
        seen.add(ch)
        kept.append((cell, s))
    return kept, rejected, {f"{n}_scenarios_checked": len(content[n]) for n, _ in sets} | {
        "inspected_v1_layouts": len(inspected)}


def decide(gate: dict[str, Any]) -> dict[str, Any]:
    """`gate_v2` checks unchanged; only the decision label is the v1.2 one."""
    return {**gate, "v2_label": gate["decision"], "decision": BLOCKED if gate["failed"] else READY}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exploratory", action="store_true", help="pre-freeze smoke on t4-explore@v12 (not evidence)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--wall-limit-s", type=float, default=v2.WALL_LIMIT_S)
    args = ap.parse_args(argv)
    namespace = EXPLORATORY_NAMESPACE if args.exploratory else NAMESPACE
    out = args.out or (SMOKE_OUT if args.exploratory else DEFAULT_OUT)
    if out.exists():
        sys.exit(f"{out} exists: each run happens once (docs/P1_T4_V12_NODE_BUDGET.md section 6)")
    t_start = time.perf_counter()
    kept, rejected, excl = select_cells(namespace)
    report = v2.build_report(kept, rejected, excl, t_start=t_start, wall_limit_s=args.wall_limit_s,
                             ablations=ABLATIONS_V12, namespace=namespace, pilot_version=PILOT_VERSION)
    report["gate"] = decide(report["gate"])
    report["evidence_class"] = "exploratory_smoke_not_evidence" if args.exploratory else "prospective_confirmation"
    report["dev_use"] += "; no DEV comparison in this protocol"
    if args.exploratory:                      # the smoke keeps gate inputs, not every decision row
        report.pop("pilot")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "gate": report["gate"], "resources": report["resources"]}, indent=1))


if __name__ == "__main__":
    main()
