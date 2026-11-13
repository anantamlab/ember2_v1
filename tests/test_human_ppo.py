"""Per-person PPO on human views. Needs torch; the default CPU test environment has none, so this
file is skipped there (reported as skipped). Run it where torch is installed (see the runbook)."""
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ember_v1 import human_ppo as hp  # noqa: E402
from ember_v1.human_env import HumanRewardConfig  # noqa: E402
from ember_v1.human_view import HumanObserver, ViewConfig, admissible  # noqa: E402
from ember_v1.rl_online_scenarios import build_scenarios  # noqa: E402
from ember_v1.sim import Simulator  # noqa: E402
from ember_v1.student_train import ACTION_ORDER  # noqa: E402

DRAFT = Path(__file__).resolve().parents[1] / "data" / "rl" / "human-pov-ppo.draft.json"


def config(**over):
    cfg = json.loads(DRAFT.read_text(encoding="utf-8"))
    cfg.update(smoke_test=True, updates=2, episodes_per_update=2, max_steps=8, eval_every=1, select_count=2,
               minibatch=16, accumulation=2, namespace="human-ppo-test", hidden=16)
    cfg.update(over)
    return cfg


def scenarios(n=2, steps=8):
    return [s for s, _ in build_scenarios("human-ppo-test", "train", n, steps)]


def test_masked_distribution_entropy_and_ratio():
    logits = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [5.0, 1.0, 1.0, 1.0, 1.0]])
    mask = torch.tensor([[True, True, False, False, False], [False, True, True, True, True]])
    lp = hp.masked_log_probs(logits, mask)
    assert torch.allclose(lp.exp().sum(-1), torch.ones(2))
    assert torch.equal(lp.exp() > 0, mask)                              # no mass on a masked move
    assert torch.allclose(hp.entropy(lp), torch.log(torch.tensor([2.0, 4.0])))
    with pytest.raises(ValueError):
        hp.masked_log_probs(logits, torch.zeros(2, 5, dtype=torch.bool))
    with pytest.raises(FloatingPointError):
        hp.masked_log_probs(logits * float("nan"), mask)
    adv = torch.tensor([1.0, -2.0])
    loss, ratio = hp.ppo_loss(torch.zeros(2), torch.zeros(2), adv, 0.2)
    assert torch.allclose(ratio, torch.ones(2)) and float(loss) == pytest.approx(0.5)
    clipped, _ = hp.ppo_loss(torch.tensor([1.0, 1.0]), torch.zeros(2), adv, 0.2)
    assert float(clipped) == pytest.approx(-(1.2 * 1.0 + torch.e * -2.0) / 2)      # gains are clipped, losses not
    with pytest.raises(FloatingPointError):
        hp.ppo_loss(torch.tensor([30.0]), torch.zeros(1), torch.ones(1), 0.2)


def test_actor_policy_respects_the_mask_and_reloads(tmp_path):
    torch.manual_seed(0)
    actor = hp.SmallActor(radius=4, hidden=16)
    views = HumanObserver().observe(Simulator(scenarios(1)[0]).observe())
    for greedy in (True, False):
        actions = hp.actor_policy(actor, greedy=greedy, generator=torch.Generator().manual_seed(1))(views)
        assert all(actions[pid] in admissible(v) for pid, v in views.items())
    actor.save(tmp_path / "a")
    again = hp.SmallActor.load(tmp_path / "a")
    vs = list(views.values())
    assert torch.equal(actor.logits(vs), again.logits(vs))
    with pytest.raises(FileExistsError):
        actor.save(tmp_path / "a")
    with pytest.raises(ValueError):
        hp.SmallActor(radius=2).logits(vs)


def test_collect_gives_one_row_per_person_decision():
    torch.manual_seed(0)
    cfg, actor, critic = config(), hp.SmallActor(4, 16), hp.make_critic()
    gen = torch.Generator().manual_seed(3)
    batch = hp.collect(scenarios(), actor, critic, cfg, ViewConfig(**cfg["view"]), HumanRewardConfig(**cfg["reward"]), gen)
    n = len(batch["views"])
    decisions = batch["episodes"]["attempted"] + batch["episodes"]["stayed"]
    assert n == decisions > 0
    assert all(len(batch[k]) == n for k in ("masks", "actions", "old", "values", "features", "teacher",
                                             "advantages", "returns"))
    assert len(batch["features"][0]) == hp.CRITIC_INPUT
    for view, action, teacher, old in zip(batch["views"], batch["actions"], batch["teacher"], batch["old"]):
        assert ACTION_ORDER[action] in admissible(view) and ACTION_ORDER[teacher] in admissible(view)
        assert old <= 0.0
    again = hp.collect(scenarios(), actor, critic, cfg, ViewConfig(**cfg["view"]), HumanRewardConfig(**cfg["reward"]),
                       torch.Generator().manual_seed(3))
    assert again["actions"] == batch["actions"] and again["advantages"] == batch["advantages"]


def test_optimize_replays_exactly_updates_weights_and_follows_the_teacher():
    torch.manual_seed(0)
    cfg, actor, critic = config(target_kl=10.0, ppo_epochs=8), hp.SmallActor(4, 16), hp.make_critic()
    gen = torch.Generator().manual_seed(3)
    view, reward = ViewConfig(**cfg["view"]), HumanRewardConfig(**cfg["reward"])
    batch = hp.collect(scenarios(), actor, critic, cfg, view, reward, gen)
    optimizer = torch.optim.Adam(actor.parameter_groups({**cfg, "actor_lr": 0.01})
                                 + [{"params": list(critic.parameters()), "lr": 0.01, "name": "critic"}])
    before = [p.detach().clone() for p in actor.parameters()]
    stats = hp.optimize(actor, critic, optimizer, batch, cfg, imitation_coef=5.0, generator=gen)
    assert stats["replay_error"] < 1e-5 and stats["optimizer_steps"] > 0 and not stats["kl_early_stop"]
    assert any(not torch.equal(a, b) for a, b in zip(before, actor.parameters()))
    teacher = torch.tensor(batch["teacher"])
    with torch.no_grad():
        lp = hp.masked_log_probs(actor.logits(batch["views"]), torch.stack(batch["masks"]))
        fresh = hp.masked_log_probs(hp.SmallActor(4, 16).logits(batch["views"]), torch.stack(batch["masks"]))
    assert float(lp.gather(-1, teacher[:, None]).mean()) > float(fresh.gather(-1, teacher[:, None]).mean())

    with pytest.raises(RuntimeError):                       # the policy changed: old probabilities are stale
        hp.optimize(actor, critic, optimizer, batch, cfg, 0.0, gen)
    batch = hp.collect(scenarios(), actor, critic, cfg, view, reward, gen)
    stopped = hp.optimize(actor, critic, optimizer, batch, {**cfg, "target_kl": -1.0}, 0.0, gen)
    assert stopped["kl_early_stop"] and stopped["optimizer_steps"] == 0


def test_train_smoke_writes_checkpoints_and_refuses_drafts(tmp_path):
    torch.manual_seed(0)
    cfg = config()
    actor = hp.SmallActor(cfg["view"]["radius"], cfg["hidden"])
    summary = hp.train(actor, hp.make_critic(), cfg, tmp_path / "run")
    assert summary["selected_update"] in (1, 2) and len(summary["checks"]) == 2
    assert (tmp_path / "run" / "update-0002" / "actor.pt").exists()
    assert len((tmp_path / "run" / "log.jsonl").read_text().splitlines()) == 2
    assert json.loads((tmp_path / "run" / "summary.json").read_text())["baseline"]["update"] == 0
    with pytest.raises(FileExistsError):
        hp.train(actor, hp.make_critic(), cfg, tmp_path / "run")
    draft = json.loads(DRAFT.read_text(encoding="utf-8"))
    with pytest.raises(PermissionError):                    # the tracked config is a draft
        hp.train(actor, hp.make_critic(), draft, tmp_path / "draft")
    assert not (tmp_path / "draft").exists()
    with pytest.raises(ValueError):
        hp.check_config({"status": "authorized"})
