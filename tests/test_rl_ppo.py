"""Torch checks run in the existing GPU environment; CPU contracts live in test_rl_env."""
import itertools


def test_joint_probability_normalizes_and_gradient_is_sum():
    import torch
    from ember_v1.rl_ppo import joint_statistics
    z = torch.tensor([[0.1, -0.2, 0.3, 0.4, -0.1], [-0.3, 0.4, 0.6, 0.2, -0.2]], requires_grad=True)
    total = sum(joint_statistics(z, torch.tensor(a))[0].exp() for a in itertools.product(range(5), repeat=2))
    assert torch.allclose(total, torch.tensor(1.0, dtype=torch.float64), atol=1e-6)
    a = torch.tensor([1, 3])
    lp, entropy, _ = joint_statistics(z, a)
    lp.backward()
    expected = torch.nn.functional.one_hot(a, 5) - z.detach().softmax(-1)
    assert torch.allclose(z.grad, expected, atol=1e-6)
    assert torch.isfinite(entropy)


def test_joint_ppo_ratio_clips_once_for_whole_map():
    import torch
    from ember_v1.rl_ppo import ppo_actor_loss
    # Two persons each change likelihood by 1.15: true joint ratio 1.3225 clips at 1.2.
    new = torch.tensor([2 * __import__("math").log(1.15)], requires_grad=True)
    loss, ratio = ppo_actor_loss(new, torch.zeros(1), torch.ones(1), 0.2)
    assert abs(float(ratio.detach()) - 1.3225) < 1e-6
    assert abs(float(loss.detach()) + 1.2) < 1e-6
    loss.backward()
    assert float(new.grad) == 0


def test_ppo_negative_advantage_and_nonfinite_rejected():
    import torch
    import pytest
    from ember_v1.rl_ppo import ppo_actor_loss
    loss, _ = ppo_actor_loss(torch.tensor([0.5]).log(), torch.zeros(1), -torch.ones(1), 0.2)
    assert float(loss) == pytest.approx(0.8)
    with pytest.raises(FloatingPointError):
        ppo_actor_loss(torch.tensor([float("nan")]), torch.zeros(1), torch.ones(1), 0.2)
