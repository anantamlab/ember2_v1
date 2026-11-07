"""T1 vs T0 DEV harness (issue #8): split gate, observation-only loop, family-cluster statistics, frozen config."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ember_v1.contract import validate_observation
from ember_v1.sim.canonical import digest
from ember_v1.teacher import StayControl
from ember_v1.teacher import t1_eval as te
from ember_v1.teacher.t1 import T1CongestionAware
from test_teacher_t0 import ascii_scenario

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data/splits/p1-02-v1"
COMMITTED = ROOT / "data/baselines/p1-04-t1-dev-v1.json"
DOC = ROOT / "docs/P1_T1_PLANNER.md"
PUBLIC_ROOT = {"observation_version", "step", "map", "rules", "people", "burning", "allowed_actions"}
FROZEN_T1_CONFIG_HASH = "sha256:5047e942db3326706b6184c9f29e0b992078920f03c8a53adc8cc24258345361"


def test_t1_configuration_is_frozen_and_documented():
    assert T1CongestionAware().identity.config_hash == FROZEN_T1_CONFIG_HASH
    assert FROZEN_T1_CONFIG_HASH in DOC.read_text(encoding="utf-8")


@pytest.mark.parametrize("split", ["train", "calibration"])
def test_only_the_dev_split_is_accepted(split):
    with pytest.raises(PermissionError):
        te.load_dev(SPLITS / "index.json", SPLITS / f"{split}.json")


class Spy(StayControl):
    def __init__(self) -> None:
        super().__init__()
        self.seen: list[dict] = []

    def decide(self, observation):
        self.seen.append(observation)
        d = super().decide(observation)
        d.diagnostics.update(people={p: {"predicted_outcome": "stayed", "route_found": True} for p in d.actions},
                             no_route=[])
        return d


def test_harness_passes_only_the_public_observation():
    s = ascii_scenario(["0..a", "b..*"], interval=1, max_steps=6, seed=99, scenario_id="private-lineage")
    spy = Spy()
    rec, lat = te.run_episode(s, spy)
    assert len(spy.seen) == rec["decisions"] == len(lat) == rec["steps"]
    for obs in spy.seen:
        assert set(obs) == PUBLIC_ROOT and validate_observation(obs) is not None
        assert "99" not in json.dumps(obs) and "private-lineage" not in json.dumps(obs)


def test_episode_metrics_on_a_congested_room():
    rows = ["####0####", "#abcdefg#", "#hijklmn#", "#.......#", "####1####"]
    t0, _ = te.run_episode(ascii_scenario(rows), te.policies()[0])
    t1, _ = te.run_episode(ascii_scenario(rows), te.policies()[1])
    assert t0["evacuated"] == t1["evacuated"] == 14 and t1["steps"] < t0["steps"]
    assert t1["blocked"] == 0 < t0["blocked"] and t1["prediction_mismatches"] == t0["prediction_mismatches"] == 0
    assert t1["exit_utilization"] > t0["exit_utilization"] and t1["busiest_exit_share"] < t0["busiest_exit_share"] == 1
    assert t1["t1"]["budget_failure_decisions"] == 0


def test_family_cluster_bootstrap():
    ci = te.cluster_bootstrap({"a": [1.0, 1.0], "b": [1.0]}, 500, "s", 0.95)
    assert ci == (1.0, 1.0)
    assert te.cluster_bootstrap({"a": [1.0], "b": [2.0]}, 500, "s", 0.95) == \
        te.cluster_bootstrap({"b": [2.0], "a": [1.0]}, 500, "s", 0.95)
    wide = te.cluster_bootstrap({"a": [10.0, 10.0, 10.0], **{f"z{i}": [0.0] for i in range(9)}}, 2000, "s", 0.95)
    assert wide[0] == 0.0 and wide[1] > 2.0          # one family with a large effect does not give a narrow CI
    assert te.cluster_bootstrap({"a": []}, 10, "s", 0.95) is None


# ----- committed DEV results (written by `python -m ember_v1.teacher.t1_eval`) ------------
@pytest.fixture(scope="module")
def committed():
    if not COMMITTED.exists():
        pytest.skip("DEV results not committed yet")
    return json.loads(COMMITTED.read_text(encoding="utf-8"))


def test_committed_results_are_complete_and_consistent(committed):
    idx = json.loads((SPLITS / "index.json").read_text(encoding="utf-8"))
    assert committed["dev"]["split"] == "dev" and committed["dev"]["plan_id"] == "p1-02-v1"
    assert committed["dev"]["manifest_hash"] == idx["splits"]["dev"]["manifest_hash"]
    assert committed["scenarios_evaluated"] == len(committed["rows"]) == 48 and committed["families_evaluated"] == 16
    assert committed["environment"]["src_dirty"] is False
    assert committed["policies"]["T1"]["config_hash"] == FROZEN_T1_CONFIG_HASH
    det = [{k: (te.ev.deterministic_view(v) if k in te.POLICY_ORDER else v) for k, v in r.items()}
           for r in committed["rows"]]
    assert digest(det) == committed["deterministic_digest"]
    for name in te.POLICY_ORDER:
        agg = committed["aggregates"][name]
        assert agg["failures"] == 0 and agg["prediction_mismatches"] == 0
        assert sum(agg["casualty_context"].values()) == agg["casualties"]
    t1 = committed["aggregates"]["T1"]
    assert t1["blocked"] == 0 and t1["t1"]["budget_failure_decisions"] == 0
    old = json.loads((ROOT / "data/baselines/p1-03-t0-dev-v1.json").read_text(encoding="utf-8"))
    for new, prev in zip(committed["rows"], old["rows"]):            # T0 is frozen: same episodes as P1-03
        assert new["T0"]["final_transition_hash"] == prev["T0"]["final_transition_hash"]


def test_committed_rows_regenerate_exactly(committed):
    fresh = te.evaluate(limit=2)
    for new, old in zip(fresh["rows"], committed["rows"][:2]):
        for k in new:
            if k in te.POLICY_ORDER:
                assert te.ev.deterministic_view(new[k]) == te.ev.deterministic_view(old[k]), (new["position"], k)
            else:
                assert new[k] == old[k]
