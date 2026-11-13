"""Explicit Torch checks; kept outside the dependency-light CPU test suite."""


def test_counterfactual_pair_gradient_is_signed_joint_map_difference():
    import torch
    from ember_v1.rl_ppo import counterfactual_pair_loss
    z = torch.zeros((2, 5), requires_grad=True)
    chosen, rejected = torch.tensor([1, 2]), torch.tensor([0, 2])
    loss = counterfactual_pair_loss(z, chosen, rejected, 0.0, 0.5)
    loss.backward()
    assert torch.isfinite(z.grad).all()
    assert z.grad[0, 1] < 0 < z.grad[0, 0]
    assert (z.grad[1] == 0).all()  # agreeing person's action cancels from the joint comparison
    changed = (z.detach()-.1*z.grad).requires_grad_()
    assert counterfactual_pair_loss(changed, chosen, rejected, 0.0, .5) < loss.detach()


def test_behavior_margin_has_zero_initial_preference_and_does_not_use_teacher_scores():
    import torch
    import pytest
    from ember_v1.rl_ppo import counterfactual_pair_loss, joint_statistics
    z = torch.tensor([[2., -1., 0., 1., 3.]], requires_grad=True)
    a, b = torch.tensor([1]), torch.tensor([4])
    reference = float((joint_statistics(z, a)[0]-joint_statistics(z, b)[0]).detach())
    loss = counterfactual_pair_loss(z, a, b, reference, .25)
    assert float(loss.detach()) == pytest.approx(.25*__import__('math').log(2))
