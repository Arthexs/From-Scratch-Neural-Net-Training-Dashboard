"""
Tests for MSELoss and BCELoss forward and backward passes,
compared against torch.nn reference implementations.
"""

import pytest
import torch

from model.losses import BCELoss, MSELoss

# ── shared fixtures ────────────────────────────────────────────────────────────

REDUCTIONS = ["mean", "sum", "none"]

INPUT_CASES = [
    pytest.param(
        torch.tensor([0.5, 1.5, -0.5, 2.0]),
        torch.tensor([0.0, 1.0, 0.0, 1.0]),
        id="1d-4elem",
    ),
    pytest.param(
        torch.tensor([[0.2, 0.8], [1.5, -1.0]]),
        torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        id="2d-2x2",
    ),
    pytest.param(
        torch.tensor([[2.0, -2.0, 0.0], [-1.0, 1.0, 3.0]]),
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]),
        id="2d-2x3",
    ),
    pytest.param(
        torch.zeros(1),
        torch.zeros(1),
        id="zero-singleton",
    ),
]


# ── MSELoss ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize("x,y", INPUT_CASES)
def test_mse_forward(x, y, reduction):
    ref = torch.nn.MSELoss(reduction=reduction)(x, y)
    out = MSELoss(reduction=reduction).forward(x, y)
    assert torch.allclose(out, ref, atol=1e-6)


@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize("x,y", INPUT_CASES)
def test_mse_backward(x, y, reduction):
    layer = MSELoss(reduction=reduction)
    layer.forward(x, y)
    grad_in = torch.ones(x.shape) if reduction == "none" else torch.tensor(1.0)

    x_ref = x.clone().requires_grad_(True)
    loss_ref = torch.nn.MSELoss(reduction=reduction)(x_ref, y)
    loss_ref.backward(torch.ones_like(loss_ref))

    assert torch.allclose(layer.backward(grad_in), x_ref.grad, atol=1e-6)


# ── BCELoss ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize("x,y", INPUT_CASES)
def test_bce_forward(x, y, reduction):
    ref = torch.nn.BCEWithLogitsLoss(reduction=reduction)(x, y)
    out = BCELoss(reduction=reduction).forward(x, y)
    assert torch.allclose(out, ref, atol=1e-6)


@pytest.mark.parametrize("reduction", REDUCTIONS)
@pytest.mark.parametrize("x,y", INPUT_CASES)
def test_bce_backward(x, y, reduction):
    layer = BCELoss(reduction=reduction)
    layer.forward(x, y)
    grad_in = torch.ones(x.shape) if reduction == "none" else torch.tensor(1.0)

    x_ref = x.clone().requires_grad_(True)
    loss_ref = torch.nn.BCEWithLogitsLoss(reduction=reduction)(x_ref, y)
    loss_ref.backward(torch.ones_like(loss_ref))

    assert torch.allclose(layer.backward(grad_in), x_ref.grad, atol=1e-6)
