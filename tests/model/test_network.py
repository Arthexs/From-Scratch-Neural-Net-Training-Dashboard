"""
Tests for the Network class: forward/backward shapes, parameters, summary, and preset loading.
"""

import pytest
import torch

from model.network import Network
from model.presets import CNN_SMALL, MLP_BASELINE

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def mlp():
    return Network.from_config(MLP_BASELINE)


@pytest.fixture
def cnn():
    return Network.from_config(CNN_SMALL)


# ── Forward shape ──────────────────────────────────────────────────────────────


def test_mlp_forward_shape(mlp):
    x = torch.randn(4, 784)
    out = mlp.forward(x)
    assert out.shape == (4, 10)


def test_cnn_forward_shape(cnn):
    x = torch.randn(4, 1, 28, 28)
    out = cnn.forward(x)
    assert out.shape == (4, 10)


# ── Backward shape ─────────────────────────────────────────────────────────────


def test_mlp_backward_shape(mlp):
    x = torch.randn(4, 784)
    out = mlp.forward(x)
    grad = torch.ones_like(out)
    dx = mlp.backward(grad)
    assert dx.shape == x.shape


def test_cnn_backward_shape(cnn):
    x = torch.randn(4, 1, 28, 28)
    out = cnn.forward(x)
    grad = torch.ones_like(out)
    dx = cnn.backward(grad)
    assert dx.shape == x.shape


# ── Parameters ─────────────────────────────────────────────────────────────────


def test_mlp_has_parameters(mlp):
    params = mlp.parameters()
    assert len(params) > 0
    assert all(isinstance(p, torch.Tensor) for p in params)


def test_cnn_has_parameters(cnn):
    params = cnn.parameters()
    assert len(params) > 0
    assert all(isinstance(p, torch.Tensor) for p in params)


# ── Summary ────────────────────────────────────────────────────────────────────


def test_mlp_summary_runs(mlp, capsys):
    mlp.summary((784,))
    out = capsys.readouterr().out
    assert "Dense" in out
    assert "Total" in out
    assert "Params" in out


def test_cnn_summary_runs(cnn, capsys):
    cnn.summary((1, 28, 28))
    out = capsys.readouterr().out
    assert "Conv2D" in out
    assert "MaxPool2D" in out
    assert "Total" in out


# ── from_config ────────────────────────────────────────────────────────────────


def test_from_config_unknown_type():
    with pytest.raises(KeyError):
        Network.from_config([{"type": "nonexistent"}])


def test_from_config_preserves_order():
    net = Network.from_config(MLP_BASELINE)
    names = [type(layer).__name__ for layer in net.layers]
    assert names[0] == "Flatten"
    assert names[-1] == "Softmax"
