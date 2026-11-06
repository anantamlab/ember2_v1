"""T0 DEV baseline harness (issue #7): split gate, observation-only loop, committed results."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ember_v1.contract import validate_observation
from ember_v1.sim.canonical import digest
from ember_v1.teacher import StayControl
from ember_v1.teacher import evaluate as ev
from test_teacher_t0 import ascii_scenario

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data/splits/p1-02-v1"
COMMITTED = ROOT / "data/baselines/p1-03-t0-dev-v1.json"
PUBLIC_ROOT = {"observation_version", "step", "map", "rules", "people", "burning", "allowed_actions"}


@pytest.mark.parametrize("split", ["train", "calibration"])
def test_only_the_dev_split_is_accepted(split):
    with pytest.raises(PermissionError):
        ev.load_dev(SPLITS / "index.json", SPLITS / f"{split}.json")


def test_teacher_package_never_touches_sealed_or_blind_data():
    for path in (ROOT / "src/ember_v1/teacher").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "LOCKED_PURPOSE" not in text and "final_locked_evaluation" not in text, path.name
        assert "gen.blind" not in text and "gen import blind" not in text, path.name
        assert "locked_test.json" not in text and "data/sealed" not in text, path.name


class Spy(StayControl):
    def __init__(self) -> None:
        super().__init__()
        self.seen: list[dict] = []

    def decide(self, observation):
        self.seen.append(observation)
        return super().decide(observation)


def test_harness_passes_only_the_public_observation():
    s = ascii_scenario(["0..a", "b..*"], interval=1, max_steps=6, seed=99, scenario_id="private-lineage")
    spy = Spy()
    rec, lat = ev.run_episode(s, spy)
    assert len(spy.seen) == rec["decisions"] == len(lat) == rec["steps"]
    for obs in spy.seen:
        assert set(obs) == PUBLIC_ROOT
        assert validate_observation(obs) is not None
        assert "99" not in json.dumps(obs) and "private-lineage" not in json.dumps(obs)


def test_random_control_seed_depends_on_plan_and_position_only():
    assert ev.random_control_seed("p1-02-v1", 3) == ev.random_control_seed("p1-02-v1", 3)
    assert ev.random_control_seed("p1-02-v1", 3) != ev.random_control_seed("p1-02-v1", 4)


def test_sign_test():
    assert ev._sign_test(0, 0) == 1.0
    assert ev._sign_test(5, 0) == pytest.approx(2 / 32)
    assert ev._sign_test(3, 3) == 1.0


# ----- committed DEV results -------------------------------------------------
@pytest.fixture(scope="module")
def committed():
    return json.loads(COMMITTED.read_text(encoding="utf-8"))


def test_committed_results_are_complete_and_consistent(committed):
    idx = json.loads((SPLITS / "index.json").read_text(encoding="utf-8"))
    assert committed["dev"]["split"] == "dev" and committed["dev"]["plan_id"] == "p1-02-v1"
    assert committed["dev"]["manifest_hash"] == idx["splits"]["dev"]["manifest_hash"]
    assert committed["scenarios_evaluated"] == committed["dev"]["scenarios"] == len(committed["rows"]) == 48
    assert committed["environment"]["src_dirty"] is False
    det = [{k: (ev.deterministic_view(v) if k in ev.POLICY_ORDER else v) for k, v in r.items()}
           for r in committed["rows"]]
    assert digest(det) == committed["deterministic_digest"]
    t0 = committed["aggregates"]["T0"]
    assert t0["prediction_mismatches"] == 0
    assert sum(t0["casualty_context"].values()) == t0["casualties"]
    for name in ev.POLICY_ORDER:
        assert committed["aggregates"][name]["failures"] == 0
    for o in ("blocked_cycle", "blocked_wall", "blocked_boundary", "blocked_fire"):
        assert t0["movement_outcomes"][o] == 0
    assert all("entered_fire" not in r["T0"]["casualty_reasons"] for r in committed["rows"])


def test_committed_rows_regenerate_exactly(committed):
    fresh = ev.evaluate(limit=3)
    for new, old in zip(fresh["rows"], committed["rows"][:3]):
        for k in new:
            if k in ev.POLICY_ORDER:
                assert ev.deterministic_view(new[k]) == ev.deterministic_view(old[k]), (new["position"], k)
            else:
                assert new[k] == old[k]
