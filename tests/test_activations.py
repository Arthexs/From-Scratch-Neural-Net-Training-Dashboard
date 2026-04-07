"""
Tests for activation layer forward and backward passes (ReLU, Sigmoid, Softmax).
"""

import torch
import pytest
from model.activations import ReLU, Sigmoid, Softmax


ACTIVATION_CASES = [
    pytest.param(ReLU,  torch.nn.functional.relu, torch.tensor([-1.0, 0.0, 1.0, 2.0]),  id="relu-mixed"),
    pytest.param(ReLU,  torch.nn.functional.relu, torch.tensor([-3.0, -0.5, 0.0]),       id="relu-negative"),
    pytest.param(Sigmoid, torch.sigmoid,           torch.tensor([-2.0, 0.0, 0.5, 2.0]), id="sigmoid-mixed"),
    pytest.param(Sigmoid, torch.sigmoid,           torch.tensor([0.0]),                  id="sigmoid-zero"),
]

SOFTMAX_CASES = [
    pytest.param(torch.tensor([1.0, 2.0, 3.0]),          -1, id="softmax-1d"),
    pytest.param(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), -1, id="softmax-2d-last-dim"),
    pytest.param(torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  0, id="softmax-2d-first-dim"),
    pytest.param(torch.tensor([1000.0, 1001.0, 1002.0]), -1, id="softmax-large-values"),
]


@pytest.mark.parametrize("layer_cls,torch_fn,x", ACTIVATION_CASES)
def test_forward(layer_cls, torch_fn, x):
    out = layer_cls().forward(x)
    assert torch.allclose(out, torch_fn(x))


@pytest.mark.parametrize("x,dim", SOFTMAX_CASES)
def test_softmax_forward(x, dim):
    out = Softmax(dim=dim).forward(x)
    assert torch.allclose(out, torch.softmax(x, dim=dim))


@pytest.mark.parametrize("layer_cls,torch_fn,x", ACTIVATION_CASES)
def test_backward(layer_cls, torch_fn, x):
    layer = layer_cls()
    layer.forward(x)

    x_ref = x.clone().requires_grad_(True)
    torch_fn(x_ref).backward(torch.ones_like(x_ref))

    assert torch.allclose(layer.backward(torch.ones_like(x)), x_ref.grad)


@pytest.mark.parametrize("x,dim", SOFTMAX_CASES)
def test_softmax_backward(x, dim):
    layer = Softmax(dim=dim)
    layer.forward(x)

    grad = torch.ones_like(x)
    x_ref = x.clone().requires_grad_(True)
    torch.softmax(x_ref, dim=dim).backward(torch.ones_like(x_ref))

    assert torch.allclose(layer.backward(grad), x_ref.grad, atol=1e-6)


@pytest.mark.parametrize("layer_cls,torch_fn,x", ACTIVATION_CASES)
def test_parameters(layer_cls, torch_fn, x):
    assert layer_cls().parameters() == []
