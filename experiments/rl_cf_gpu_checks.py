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


def test_counterfactual_probe_and_applied_backward_use_separate_graphs(monkeypatch):
    import torch
    from ember_v1.rl_cf_training import CounterfactualObjective

    class Actor:
        def __init__(self):
            self.model = torch.nn.Module()
            self.model.head = torch.nn.Linear(2, 5, device="cuda")
            self.calls = 0

        def prepare(self, observations):
            assert observations == [{"public": True}]
            return torch.ones(1, 2, device="cuda")

        def logits(self, batch):
            self.calls += 1
            return [self.model.head(batch)]

    # The real compiled backward rejects retaining a donated-buffer graph.
    grad = torch.autograd.grad
    def single_use(*args, **kwargs):
        assert not kwargs.get("retain_graph")
        return grad(*args, **kwargs)
    monkeypatch.setattr(torch.autograd, "grad", single_use)
    actor = Actor()
    objective = CounterfactualObjective([])
    objective.pairs.append({"public": {"public": True}, "positive": [1], "negative": [0],
        "reference_margin": 0., "weight": .5, "study_hash": "fixture", "chosen": "T0"})
    loss, proof = objective(actor)
    assert actor.calls == 2 and proof["independent_auxiliary_head_gradient_norm"] > 0
    assert all(p.grad is None for p in actor.model.parameters())
    (.05*loss).backward()
    assert all(torch.isfinite(p.grad).all() and p.grad.norm() > 0 for p in actor.model.parameters())
    loss, proof = objective(actor)
    assert actor.calls == 3 and "independent_auxiliary_head_gradient_norm" not in proof
    loss.backward()
