"""Belief teacher, per-person environment and the DAgger loop (CPU, mock student)."""
import json

import pytest

from ember_v1 import human_dagger as hd
from ember_v1.contract import ACTIONS
from ember_v1.human_env import HumanEvacuationEnv, HumanRewardConfig, run_episode, summarize
from ember_v1.human_teacher import belief_action, belief_policy, labels, random_policy
from ember_v1.human_view import HumanObserver, ViewConfig, admissible
from ember_v1.rl_env import EvacuationEnv, RewardConfig
from ember_v1.rl_online_scenarios import build_scenarios
from ember_v1.sim import Simulator
from ember_v1.student_train import MockDecisionAdapter
from test_human_view import DIVIDER, room


def fresh(count=3, max_steps=12, namespace="human-dagger-test"):
    return [s for s, _ in build_scenarios(namespace, "train", count, max_steps)]


# ----- belief teacher ------------------------------------------------------------------------------
def test_belief_teacher_finds_an_unseen_exit_through_the_door():
    scenario = room(DIVIDER, people=((0, 0), (4, 1)), max_steps=20)
    result = run_episode(scenario, belief_policy, ViewConfig(radius=2))
    assert result["evacuated"] == 2 and result["casualties"] == 0
    assert result["movements"]["blocked_wall"] == result["movements"]["blocked_boundary"] == 0


def test_belief_action_rules():
    view = HumanObserver().observe(Simulator(room()).observe())["P0"]
    base = {**view, "moves": {"north": "wall", "south": "floor", "east": "person", "west": "floor"},
            "steps": {"stay": 3, "north": None, "south": 4, "east": 2, "west": 4},
            "last_action": "east"}
    assert belief_action({**base, "last_outcome": "moved"}) == "east"
    assert belief_action({**base, "last_outcome": "blocked_occupied"}) == "east"        # wait in the queue
    assert belief_action({**base, "last_outcome": "blocked_cycle"}) == "south"          # step aside
    level = {**base, "steps": {"stay": 3, "north": None, "south": 2, "east": 2, "west": 4}}
    assert belief_action({**level, "last_outcome": "moved"}) == "east"                  # keep the direction
    assert belief_action({**level, "last_outcome": "blocked_conflict"}) == "south"      # equally good, free
    no_gain = {**base, "steps": {"stay": 2, "north": None, "south": 3, "east": 2, "west": 3}}
    assert belief_action({**no_gain, "last_outcome": None}) == "stay"
    assert belief_action({**base, "steps": {a: None for a in ACTIONS}, "last_outcome": None}) == "stay"
    blind = {**base, "goal": None, "steps": None, "last_outcome": None}
    with pytest.raises(ValueError):
        belief_action(blind)
    assert belief_action(blind, {"goal": "exit", "steps": base["steps"]}) == "east"


def test_belief_teacher_never_walks_into_a_visible_wall_or_fire_and_beats_random():
    scenarios = fresh(6, 30)
    for scenario in scenarios[:2]:
        env = HumanEvacuationEnv(scenario)
        views, _ = env.reset()
        done = False
        while not done:
            actions = belief_policy(views)
            assert all(actions[pid] in admissible(v) for pid, v in views.items())
            views, _, done, _ = env.step(actions)
    teacher = summarize([run_episode(s, belief_policy) for s in scenarios])
    chance = summarize([run_episode(s, random_policy(1)) for s in scenarios])
    assert teacher["movements"]["blocked_wall"] == teacher["movements"]["blocked_boundary"] == 0
    assert teacher["evacuated_rate"] > chance["evacuated_rate"] + 0.2
    assert teacher["utility"] > chance["utility"]
    hidden = summarize([run_episode(s, belief_policy, ViewConfig(hints=False)) for s in scenarios])
    assert hidden == teacher                                # the teacher does not depend on shown hints


def test_label_modes():
    views = HumanObserver().observe(Simulator(room(people=((2, 1), (0, 0)))).observe())
    own = labels(views)
    assert own == belief_policy(views) and set(own) == {"P0", "P1"}
    forced = {"P0": "west", "P1": "north"}                  # west = away from the route, north = the edge
    assert labels(views, "privileged", privileged=forced) == forced
    assert labels(views, "consistent", privileged=forced) == own
    agree = {"P0": own["P0"], "P1": "stay"}
    assert labels(views, "consistent", privileged=agree)["P0"] == own["P0"]
    for bad in ({"P0": "east"}, {"P0": "east", "P1": "jump"}, None):
        with pytest.raises(ValueError):
            labels(views, "privileged", privileged=bad)
    with pytest.raises(ValueError):
        labels(views, "oracle")


# ----- environment ---------------------------------------------------------------------------------
def test_per_person_rewards_add_up_to_the_team_reward():
    scenario = room(h=3, w=4, exits=((1, 3),), people=((1, 2), (1, 1)), fire=((0, 0),), max_steps=2)
    env, team = HumanEvacuationEnv(scenario, reward=HumanRewardConfig(eta=0.0)), EvacuationEnv(scenario, RewardConfig())
    env.reset()
    team.reset()
    total, joint = 0.0, 0.0
    for actions, expected in (({"P0": "east", "P1": "east"}, {"P0": 1.0, "P1": 0.0}), ({"P1": "east"}, {"P1": 1.0})):
        views, rewards, done, info = env.step(actions)
        assert rewards == expected
        total += sum(rewards.values())
        joint += team.step(actions)[1]
    assert done and views == {} and total / 2 == joint == 1.0
    with pytest.raises(Exception):
        env.step({})


def test_step_cap_shaping_block_cost_and_team_share():
    scenario = room(h=3, w=4, exits=((1, 3),), people=((1, 2), (1, 1)), fire=((0, 0),), max_steps=2)
    env = HumanEvacuationEnv(scenario, reward=HumanRewardConfig(eta=0.0))
    env.reset()
    env.step({"P0": "stay", "P1": "stay"})
    assert env.step({"P0": "stay", "P1": "stay"})[1] == {"P0": -0.75, "P1": -0.75}      # cap = unresolved

    shaped = HumanEvacuationEnv(scenario, reward=HumanRewardConfig(eta=0.5, gamma=1.0))
    shaped.reset()
    first = shaped.step({"P0": "east", "P1": "east"})[1]
    second = shaped.step({"P1": "east"})[1]
    assert first["P1"] + second["P1"] == pytest.approx(1.0 + 0.5 * (2 / 12))            # shaping telescopes
    assert first["P0"] == pytest.approx(1.0 + 0.5 * (1 / 12))

    cost = HumanEvacuationEnv(scenario, reward=HumanRewardConfig(eta=0.0, blocked=-0.1))
    cost.reset()
    assert cost.step({"P0": "north", "P1": "west"})[1] == {"P0": 0.0, "P1": 0.0}        # both moved
    assert cost.step({"P0": "north", "P1": "west"})[1] == {"P0": -0.85, "P1": -0.85}    # both hit the edge

    shared = HumanEvacuationEnv(scenario, reward=HumanRewardConfig(eta=0.0, team=1.0))
    shared.reset()
    assert shared.step({"P0": "east", "P1": "stay"})[1] == {"P0": 0.5, "P1": 0.5}
    for kwargs in ({"gamma": 0.0}, {"eta": -1.0}, {"blocked": 0.1}, {"team": 2.0}, {"casualty": float("nan")}):
        with pytest.raises(ValueError):
            HumanRewardConfig(**kwargs)


def test_episode_statistics():
    scenario = room(people=((2, 1),), max_steps=4)
    result = run_episode(scenario, lambda views, routes: {pid: "north" for pid in views}, ViewConfig(radius=1))
    assert result["movements"]["moved"] == 2 and result["movements"]["blocked_boundary"] == 2
    assert result["attempted"] == 4 and result["blocked"] == 2 and result["repeated_ineffective"] == 1
    assert result["unresolved"] == 1 and result["steps"] == 4
    total = summarize([result, result])
    assert total["people"] == 2 and total["blocked_per_attempt"] == 0.5 and total["utility"] == -0.75
    with pytest.raises(ValueError):
        summarize([])


# ----- DAgger --------------------------------------------------------------------------------------
def test_teacher_only_round_labels_what_it_executes():
    scenarios = fresh()
    rows, stats = hd.collect(scenarios)
    again, _ = hd.collect(scenarios)
    assert rows == again and stats["rows"] == len(rows) > 0
    assert stats["executed_by"] == {"teacher": len(rows)} and stats["label_equals_executed"] == len(rows)
    decisions = stats["episodes"]["attempted"] + stats["episodes"]["stayed"]
    assert decisions == len(rows)
    assert json.dumps(rows[0], sort_keys=True)              # rows are plain JSON


def test_student_rounds_visit_student_states_and_keep_teacher_labels():
    scenarios = fresh()
    student = random_policy(5, masked=False)
    rows, stats = hd.collect(scenarios, beta=0.0, student=student, round_index=1)
    assert stats["executed_by"] == {"student": len(rows)} and stats["label_equals_executed"] < len(rows)
    teacher_rows, _ = hd.collect(scenarios)
    assert {hd.view_row(r["view"], "stay")["state"] for r in rows} - \
           {hd.view_row(r["view"], "stay")["state"] for r in teacher_rows}       # states the teacher never saw
    for r in rows:
        assert r["label"] in admissible(r["view"]) and r["round"] == 1
    mixed, mixed_stats = hd.collect(scenarios, beta=0.5, student=random_policy(5, masked=False), seed=3)
    assert set(mixed_stats["executed_by"]) == {"student", "teacher"}
    for kwargs in ({"beta": 0.5}, {"beta": 1.5}, {"label_mode": "privileged"},
                   {"privileged": lambda: None}, {"label_mode": "oracle"}):
        with pytest.raises(ValueError):
            hd.collect(scenarios, **kwargs)


def test_full_map_teacher_labels_are_available_for_the_ablation():
    from ember_v1.teacher.registry import teacher_class
    scenarios = fresh(1)
    own, _ = hd.collect(scenarios)
    for mode in ("privileged", "consistent"):
        rows, _ = hd.collect(scenarios, label_mode=mode, privileged=teacher_class("T1"))
        assert rows and all(r["label"] in ACTIONS for r in rows)
    assert all(r["label"] in admissible(r["view"]) for r in rows)    # consistent labels stay visible-legal
    assert own


def test_dataset_is_write_once_hashed_and_train_only(tmp_path):
    rows, stats = hd.collect(fresh(1))
    meta = {"split": "train", "view": ViewConfig().to_dict(), "round": 0}
    manifest = hd.write_dataset(tmp_path / "r0", rows, meta)
    back, read = hd.read_dataset(tmp_path / "r0")
    assert back == rows and read == manifest and b"\r" not in (tmp_path / "r0" / "rows.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        hd.write_dataset(tmp_path / "r0", rows, meta)
    with pytest.raises(ValueError):
        hd.write_dataset(tmp_path / "empty", [], meta)
    hd.write_dataset(tmp_path / "dev", rows, {**meta, "split": "select"})
    with pytest.raises(ValueError):
        hd.read_dataset(tmp_path / "dev")
    with open(tmp_path / "r0" / "rows.jsonl", "ab") as f:
        f.write(b"{}\n")
    with pytest.raises(ValueError):
        hd.read_dataset(tmp_path / "r0")


def test_aggregate_drops_duplicates_and_counts_conflicts():
    rows, _ = hd.collect(fresh(1))
    flipped = [{**rows[0], "label": next(a for a in ACTIONS if a != rows[0]["label"])}]
    out, info = hd.aggregate([rows, rows, flipped])
    assert info["input_rows"] == 2 * len(rows) + 1 and info["conflicting_states"] >= 1
    assert info["rows"] == len(out) and info["duplicates_dropped"] >= len(rows)
    assert all(set(r) == {"state", "questions", "gold"} for r in out)


def test_fit_learns_the_labels_and_metrics_report_legal_moves():
    rows, _ = hd.collect(fresh(1))
    train, _ = hd.aggregate([rows])
    adapter = MockDecisionAdapter()
    before = hd.imitation_metrics(adapter, rows)
    cfg = {"epochs": 30, "grad_accum_draws": 4, "lr": 1.0, "head_lr": 1.0, "warmup_ratio": 0.0}
    report = hd.fit(adapter, train, cfg)
    after = hd.imitation_metrics(adapter, rows)
    assert report["last_loss"] < report["first_loss"] and report["optimizer_steps"] == report["schedule"]["optimizer_steps"]
    assert after["accuracy"] > before["accuracy"] and after["accuracy"] > 0.9 and after["legal_rate"] > 0.9
    assert set(after["accuracy_by_label"]) <= set(ACTIONS)
    masked = hd.adapter_policy(MockDecisionAdapter(seed=4), masked=True)
    assert all(masked({"p": r["view"]})["p"] in admissible(r["view"]) for r in rows)

    class Broken(MockDecisionAdapter):
        def loss_and_backward(self, row, scale):
            return float("nan")

        def scores(self, row):
            return {hd.QUESTION_ID: {"stay": 0.0}}
    with pytest.raises(FloatingPointError):
        hd.fit(Broken(), train, cfg)
    with pytest.raises(ValueError):
        hd.adapter_policy(Broken())({"p": rows[0]["view"]})


def test_command_line_runs_two_dagger_rounds_and_an_evaluation(tmp_path, capsys):
    common = ["--kind", "mock", "--max-steps", "8", "--radius", "3"]
    assert hd.main(["collect", "--namespace", "cli-test", "--count", "2", "--out", str(tmp_path / "d0"), *common]) == 0
    assert hd.main(["fit", "--data", str(tmp_path / "d0"), "--out", str(tmp_path / "s0"), "--kind", "mock",
                    "--epochs", "5", "--lr", "1", "--head-lr", "1"]) == 0
    assert hd.main(["collect", "--namespace", "cli-test-r1", "--count", "2", "--out", str(tmp_path / "d1"),
                    "--student", str(tmp_path / "s0"), "--beta", "0.5", "--round", "1", *common]) == 0
    assert hd.main(["fit", "--data", str(tmp_path / "d0"), str(tmp_path / "d1"), "--out", str(tmp_path / "s1"),
                    "--resume", str(tmp_path / "s0"), "--kind", "mock"]) == 0
    assert hd.main(["eval", "--namespace", "cli-test", "--split", "select", "--count", "2",
                    "--out", str(tmp_path / "eval.json"), "--student", str(tmp_path / "s1"),
                    "--train-data", str(tmp_path / "d0"), str(tmp_path / "d1"), *common]) == 0
    capsys.readouterr()
    manifest = json.loads((tmp_path / "d1" / "manifest.json").read_text())
    assert manifest["round"] == 1 and manifest["split"] == "train" and manifest["view"]["radius"] == 3
    assert len(manifest["topology_hashes"]) == 2 and set(manifest["stats"]["executed_by"]) <= {"student", "teacher"}
    report = json.loads((tmp_path / "s1" / "human_fit_report.json").read_text())
    assert report["aggregate"]["input_rows"] >= report["aggregate"]["rows"] and len(report["datasets"]) == 2
    arms = json.loads((tmp_path / "eval.json").read_text())["arms"]
    assert set(arms) == {"belief_teacher", "random_masked", "student_raw", "student_masked"}
    assert arms["student_masked"]["movements"]["blocked_wall"] == 0

    with pytest.raises(SystemExit):                         # write-once outputs
        hd.main(["collect", "--namespace", "cli-test", "--count", "2", "--out", str(tmp_path / "d0"), *common])
    with pytest.raises(SystemExit):                         # different view settings do not mix
        hd.main(["collect", "--namespace", "cli-x", "--count", "1", "--out", str(tmp_path / "d2"), "--kind", "mock",
                 "--max-steps", "8", "--radius", "2"]) or hd.main(
            ["fit", "--data", str(tmp_path / "d0"), str(tmp_path / "d2"), "--out", str(tmp_path / "s2"), "--kind", "mock"])
    leak = [s for s, _ in build_scenarios("cli-test", "select", 2, 8)]
    rows, _ = hd.collect(fresh(1))
    hd.write_dataset(tmp_path / "leak", rows, {"split": "train", "view": ViewConfig(radius=3).to_dict(), "round": 0,
                                               "topology_hashes": [hd.topology(leak[0])]})
    with pytest.raises(SystemExit):                         # an evaluation map that was trained on
        hd.main(["eval", "--namespace", "cli-test", "--split", "select", "--count", "2",
                 "--out", str(tmp_path / "eval2.json"), "--train-data", str(tmp_path / "leak"), *common])
    capsys.readouterr()
