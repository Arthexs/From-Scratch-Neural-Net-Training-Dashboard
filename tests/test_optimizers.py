"""
Tests for the Optimizer base class, SGD, and Adam.

Each optimizer is compared against its torch.optim counterpart over
single and multiple steps with various hyperparameter configurations.
"""

import torch
import pytest
from model.optimizers import Optimizer, SGD, Adam


# ── helpers ────────────────────────────────────────────────────────────────────

def make_params(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return two identical leaf tensors (ours, reference)."""
    p_ours = tensor.clone().requires_grad_(False)  # we set .grad manually
    p_ref  = tensor.clone().requires_grad_(True)
    return p_ours, p_ref


def set_grads(p_ours: torch.Tensor, p_ref: torch.Tensor, grad: torch.Tensor) -> None:
    p_ours.grad = grad.clone()
    p_ref.grad  = grad.clone()


# ── shared fixtures ────────────────────────────────────────────────────────────

# (param_init, grad) pairs — shapes must match
STEP_CASES = [
    pytest.param(
        torch.tensor([1.0, -2.0, 0.5]),
        torch.tensor([0.1, -0.3,  0.2]),
        id="1d-mixed",
    ),
    pytest.param(
        torch.tensor([[1.0, 2.0], [-0.5, 3.0]]),
        torch.tensor([[0.5, -0.1], [0.2, -0.8]]),
        id="2d",
    ),
    pytest.param(
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([1.0, 1.0, 1.0]),
        id="zero-params",
    ),
    pytest.param(
        torch.tensor([-5.0, 5.0]),
        torch.tensor([0.01, -0.01]),
        id="large-params-small-grad",
    ),
]

SGD_CONFIG_CASES = [
    pytest.param(dict(lr=0.1),                                          id="vanilla"),
    pytest.param(dict(lr=0.01, momentum=0.9),                           id="momentum"),
    pytest.param(dict(lr=0.1,  weight_decay=0.01),                      id="weight_decay"),
    pytest.param(dict(lr=0.01, momentum=0.9, weight_decay=0.001),       id="full"),
]

ADAM_CONFIG_CASES = [
    pytest.param(dict(lr=0.1),                                          id="default"),
    pytest.param(dict(lr=0.01, betas=(0.9, 0.999), eps=1e-8),          id="standard"),
    pytest.param(dict(lr=0.1,  betas=(0.8, 0.99)),                      id="low-betas"),
    pytest.param(dict(lr=0.001, weight_decay=0.01),                     id="weight_decay"),
    pytest.param(dict(lr=0.01, betas=(0.9, 0.999), weight_decay=0.001), id="full"),
]


# ── base class ─────────────────────────────────────────────────────────────────

def test_base_step_raises():
    assert hasattr(Optimizer, "step")
    with pytest.raises(NotImplementedError):
        Optimizer().step([])


def test_zero_grad_clears_gradients():
    p = torch.tensor([1.0, 2.0])
    p.grad = torch.tensor([0.5, 0.5])
    SGD(lr=0.1).zero_grad([p])
    assert torch.all(p.grad == 0)


def test_zero_grad_skips_none_grad():
    p = torch.tensor([1.0, 2.0])
    assert p.grad is None
    SGD(lr=0.1).zero_grad([p])   # must not raise
    assert p.grad is None


def test_zero_grad_does_not_alter_data():
    p = torch.tensor([3.0, -1.0])
    p.grad = torch.ones_like(p)
    data_before = p.data.clone()
    SGD(lr=0.1).zero_grad([p])
    assert torch.equal(p.data, data_before)


def test_zero_grad_multiple_params():
    params = [torch.tensor([float(i)]) for i in range(4)]
    for p in params:
        p.grad = torch.ones_like(p)
    SGD(lr=0.1).zero_grad(params)
    for p in params:
        assert torch.all(p.grad == 0)


# ── SGD ────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cfg", SGD_CONFIG_CASES)
@pytest.mark.parametrize("init,grad", STEP_CASES)
def test_sgd_single_step(init, grad, cfg):
    p_ours, p_ref = make_params(init)
    set_grads(p_ours, p_ref, grad)

    SGD(**cfg).step([p_ours])
    torch.optim.SGD([p_ref], **cfg).step()

    assert torch.allclose(p_ours.data, p_ref.data, atol=1e-6)


@pytest.mark.parametrize("cfg", SGD_CONFIG_CASES)
@pytest.mark.parametrize("init,grad", STEP_CASES)
def test_sgd_multi_step(init, grad, cfg):
    """Run 5 steps with varying gradients and compare parameter trajectories."""
    p_ours, p_ref = make_params(init)
    our_opt   = SGD(**cfg)
    torch_opt = torch.optim.SGD([p_ref], **cfg)

    for scale in [1.0, 0.8, 1.2, 0.5, 1.5]:
        g = grad * scale
        set_grads(p_ours, p_ref, g)
        our_opt.step([p_ours])
        torch_opt.step()

    assert torch.allclose(p_ours.data, p_ref.data, atol=1e-6)


# ── Adam ───────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cfg", ADAM_CONFIG_CASES)
@pytest.mark.parametrize("init,grad", STEP_CASES)
def test_adam_single_step(init, grad, cfg):
    p_ours, p_ref = make_params(init)
    set_grads(p_ours, p_ref, grad)

    Adam(**cfg).step([p_ours])
    torch.optim.Adam([p_ref], **cfg).step()

    assert torch.allclose(p_ours.data, p_ref.data, atol=1e-6)


@pytest.mark.parametrize("cfg", ADAM_CONFIG_CASES)
@pytest.mark.parametrize("init,grad", STEP_CASES)
def test_adam_multi_step(init, grad, cfg):
    """Run 5 steps with varying gradients and compare parameter trajectories."""
    p_ours, p_ref = make_params(init)
    our_opt   = Adam(**cfg)
    torch_opt = torch.optim.Adam([p_ref], **cfg)

    for scale in [1.0, 0.8, 1.2, 0.5, 1.5]:
        g = grad * scale
        set_grads(p_ours, p_ref, g)
        our_opt.step([p_ours])
        torch_opt.step()

    assert torch.allclose(p_ours.data, p_ref.data, atol=1e-6)
