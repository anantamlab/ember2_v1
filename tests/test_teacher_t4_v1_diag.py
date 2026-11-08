"""Committed T4 v1.0.0 diagnosis report (overnight P1): consistency with the frozen pilot and the analysis doc."""
from __future__ import annotations

import json
import random

from ember_v1.sim import Simulator
from ember_v1.teacher.evaluate import REPO
from ember_v1.teacher.t4 import ABLATIONS
from ember_v1.teacher.t4_v1_diag import _Traced
from gen_helpers import random_scenario

REPORT = json.loads((REPO / "data/baselines/p1-08-t4-v1-diagnosis.json").read_text(encoding="utf-8"))
PILOT = json.loads((REPO / "data/baselines/p1-08-t4-pilot.json").read_text(encoding="utf-8"))


def test_report_reproduces_the_frozen_pilot_and_partitions_the_238_events():
    assert REPORT["reproduces_frozen_pilot_rows"] is True
    assert REPORT["teachers"] == {v: cls().identity.config_hash for v, cls in ABLATIONS.items()}
    for v in ("A0", "A1", "A2"):
        t = REPORT["variants"][v]["totals"]
        assert t["over_id"] == PILOT["mechanism"][v]["yields_over_id_rule"]
        assert t["converted_to_stay"] == PILOT["mechanism"][v]["converted_to_stay"]
        assert t["over_id"] == sum(t.get(f"over_id/{k}", 0) for k in ("label_only", "priority_cascade",
                                                                        "genuine_inversion"))
        assert sum(REPORT["variants"][v]["converted_to_stay_attribution"].values()) == t["converted_to_stay"]
        cas = sum(n for k, n in t.items() if k.startswith("final_by_initial_stratum/") and k.endswith("/casualty"))
        assert cas == PILOT["mechanism"][v]["episodes"]["casualties"]
    a0 = REPORT["variants"]["A0"]["totals"]
    assert (a0["over_id"], a0["over_id/label_only"], a0["over_id/priority_cascade"],
            a0["over_id/genuine_inversion"]) == (238, 118, 18, 102)


def test_traced_planner_never_changes_a_decision():
    for k in range(30):
        obs = Simulator(random_scenario(random.Random(4000 + k))).observe()
        for v, cls in ABLATIONS.items():
            acts, dg = _Traced(v).decide(obs)
            d = cls().decide(obs)
            assert acts == d.actions and dg == d.diagnostics
