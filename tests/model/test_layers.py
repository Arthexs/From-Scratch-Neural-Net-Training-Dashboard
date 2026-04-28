"""
Tests for Dense, Flatten, and Conv2D layer forward and backward passes,
compared against torch.nn reference implementations.
"""

import time

import pytest
import torch

from model.layers import Conv2D, Dense, Flatten, MaxPool2D

WARMUP_ITERS = 10
BENCH_ITERS = 50


def bench(fn, iters: int) -> float:
    """Return mean wall-clock time in ms over `iters` calls."""
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - start) / iters * 1000


# ── Dense ──────────────────────────────────────────────────────────────────────

INPUT_CASES = [
    pytest.param(torch.randn(1, 4), 4, 3, id="single-sample"),
    pytest.param(torch.randn(8, 4), 4, 3, id="batch-8"),
    pytest.param(torch.randn(4, 16), 16, 8, id="wide"),
    pytest.param(torch.randn(2, 1), 1, 1, id="scalar-features"),
]

INITIALIZER_CASES = [
    pytest.param("xavier_uniform", id="xavier_uniform"),
    pytest.param("xavier_normal", id="xavier_normal"),
    pytest.param("kaiming_uniform", id="kaiming_uniform"),
    pytest.param("kaiming_normal", id="kaiming_normal"),
]


def make_ref_linear(layer: Dense) -> torch.nn.Linear:
    """Build a torch.nn.Linear with identical weights to our Dense layer."""
    ref = torch.nn.Linear(layer._cfg.input_size, layer._cfg.output_size, bias=layer._cfg.bias)
    with torch.no_grad():
        ref.weight.copy_(layer.W.T)  # nn.Linear stores W as (out, in)
        if layer._cfg.bias:
            ref.bias.copy_(layer.b)
    return ref


@pytest.mark.parametrize("x,in_size,out_size", INPUT_CASES)
def test_dense_forward(x, in_size, out_size):
    layer = Dense(input_size=in_size, output_size=out_size)
    ref = make_ref_linear(layer)

    out = layer.forward(x)
    out_ref = ref(x)

    assert torch.allclose(out, out_ref, atol=1e-6)


@pytest.mark.parametrize("x,in_size,out_size", INPUT_CASES)
def test_dense_forward_no_bias(x, in_size, out_size):
    layer = Dense(input_size=in_size, output_size=out_size, bias=False)
    ref = make_ref_linear(layer)

    out = layer.forward(x)
    out_ref = ref(x)

    assert torch.allclose(out, out_ref, atol=1e-6)


@pytest.mark.parametrize("x,in_size,out_size", INPUT_CASES)
def test_dense_backward_dx(x, in_size, out_size):
    """Gradient w.r.t. input matches torch autograd."""
    layer = Dense(input_size=in_size, output_size=out_size)
    ref = make_ref_linear(layer)

    layer.forward(x)
    grad = torch.ones(x.shape[0], out_size)
    dx = layer.backward(grad)

    x_ref = x.clone().requires_grad_(True)
    ref(x_ref).backward(torch.ones(x.shape[0], out_size))

    assert torch.allclose(dx, x_ref.grad, atol=1e-6)


@pytest.mark.parametrize("x,in_size,out_size", INPUT_CASES)
def test_dense_backward_dW(x, in_size, out_size):
    """Gradient w.r.t. weights matches torch autograd."""
    layer = Dense(input_size=in_size, output_size=out_size)
    ref = make_ref_linear(layer)

    layer.forward(x)
    grad = torch.ones(x.shape[0], out_size)
    layer.backward(grad)

    x_ref = x.clone().requires_grad_(True)
    ref(x_ref).backward(torch.ones(x.shape[0], out_size))

    assert torch.allclose(layer.W.grad, ref.weight.grad.T, atol=1e-6)


@pytest.mark.parametrize("x,in_size,out_size", INPUT_CASES)
def test_dense_backward_db(x, in_size, out_size):
    """Gradient w.r.t. bias matches torch autograd."""
    layer = Dense(input_size=in_size, output_size=out_size)
    ref = make_ref_linear(layer)

    layer.forward(x)
    grad = torch.ones(x.shape[0], out_size)
    layer.backward(grad)

    x_ref = x.clone().requires_grad_(True)
    ref(x_ref).backward(torch.ones(x.shape[0], out_size))

    assert torch.allclose(layer.b.grad, ref.bias.grad, atol=1e-6)


@pytest.mark.parametrize("initializer", INITIALIZER_CASES)
def test_dense_initializer_shapes(initializer):
    layer = Dense(input_size=8, output_size=4, initializer=initializer)
    assert layer.W.shape == (8, 4)
    assert layer.b.shape == (4,)


def test_dense_parameters_with_bias():
    layer = Dense(input_size=4, output_size=2)
    params = layer.parameters()
    assert len(params) == 2
    assert params[0] is layer.W
    assert params[1] is layer.b


def test_dense_parameters_no_bias():
    layer = Dense(input_size=4, output_size=2, bias=False)
    params = layer.parameters()
    assert len(params) == 1
    assert params[0] is layer.W


# ── Flatten ────────────────────────────────────────────────────────────────────

FLATTEN_CASES = [
    pytest.param(torch.randn(2, 3, 4, 5), 1, -1, id="default-full"),
    pytest.param(torch.randn(2, 3, 4, 5), 1, 2, id="partial-mid"),
    pytest.param(torch.randn(2, 3, 4, 5), 2, 3, id="partial-last"),
    pytest.param(torch.randn(4, 8), 1, -1, id="2d-input"),
    pytest.param(torch.randn(1, 1, 1), 1, -1, id="singleton"),
]


@pytest.mark.parametrize("x,start_dim,end_dim", FLATTEN_CASES)
def test_flatten_forward(x, start_dim, end_dim):
    out = Flatten(start_dim=start_dim, end_dim=end_dim).forward(x)
    out_ref = torch.flatten(x, start_dim, end_dim)
    assert torch.equal(out, out_ref)


@pytest.mark.parametrize("x,start_dim,end_dim", FLATTEN_CASES)
def test_flatten_backward_restores_shape(x, start_dim, end_dim):
    layer = Flatten(start_dim=start_dim, end_dim=end_dim)
    out = layer.forward(x)
    dx = layer.backward(torch.ones_like(out))
    assert dx.shape == x.shape


@pytest.mark.parametrize("x,start_dim,end_dim", FLATTEN_CASES)
def test_flatten_backward_values(x, start_dim, end_dim):
    """Gradient through flatten matches torch autograd."""
    layer = Flatten(start_dim=start_dim, end_dim=end_dim)
    out = layer.forward(x)
    grad = torch.ones_like(out)
    dx = layer.backward(grad)

    x_ref = x.clone().requires_grad_(True)
    torch.flatten(x_ref, start_dim, end_dim).backward(torch.ones_like(out))

    assert torch.equal(dx, x_ref.grad)


def test_flatten_no_parameters():
    assert Flatten().parameters() == []


# ── MaxPool2D smoke ────────────────────────────────────────────────────────────


def test_maxpool2d_forward_smoke():
    x = torch.randn(2, 3, 8, 8)
    layer = MaxPool2D(kernel_size=2, stride=2, padding=0)
    ref = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    out = layer.forward(x)

    print(f"input:    {x.shape}")
    print(f"output:   {out.shape}")
    print(f"expected: {ref(x).shape}")

    assert out.shape == ref(x).shape
    assert torch.allclose(out, ref(x), atol=1e-6)


def test_maxpool2d_backward_smoke():
    x = torch.randn(2, 3, 8, 8)
    layer = MaxPool2D(kernel_size=2, stride=2, padding=0)
    ref = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    out = layer.forward(x)
    grad = torch.ones_like(out)
    dx = layer.backward(grad)

    x_ref = x.clone().requires_grad_(True)
    ref(x_ref).backward(torch.ones_like(out))

    print(f"dx:       {dx.shape}")
    print(f"expected: {x_ref.grad.shape}")

    assert dx.shape == x.shape
    assert torch.allclose(dx, x_ref.grad, atol=1e-6)


# ── Conv2D performance ─────────────────────────────────────────────────────────

CONV_PERF_CASES = [
    pytest.param(dict(N=1, Cin=1, H=28, W=28, Cout=8, K=3, S=1, P=1), id="mnist-like"),
    pytest.param(dict(N=8, Cin=3, H=32, W=32, Cout=16, K=3, S=1, P=1), id="cifar-like"),
    pytest.param(dict(N=8, Cin=16, H=32, W=32, Cout=32, K=3, S=1, P=1), id="mid-depth"),
    pytest.param(dict(N=8, Cin=32, H=16, W=16, Cout=64, K=3, S=1, P=1), id="deep-small-spatial"),
    pytest.param(dict(N=4, Cin=3, H=64, W=64, Cout=16, K=5, S=1, P=2), id="larger-input-k5"),
    pytest.param(dict(N=4, Cin=3, H=64, W=64, Cout=16, K=3, S=2, P=1), id="strided"),
]


def make_ref_conv(layer: Conv2D) -> torch.nn.Conv2d:
    """Build a torch.nn.Conv2d with identical weights to our Conv2D layer."""
    Cout, Cin, Kh, _ = layer.W.shape
    ref = torch.nn.Conv2d(
        Cin,
        Cout,
        kernel_size=Kh,
        stride=layer._cfg.stride,
        padding=layer._cfg.padding,
        bias=layer._cfg.bias,
    )
    with torch.no_grad():
        ref.weight.copy_(layer.W)
        if layer._cfg.bias:
            assert ref.bias is not None
            ref.bias.copy_(layer.b)
    return ref


@pytest.mark.parametrize("cfg", CONV_PERF_CASES)
def test_conv2d_forward_perf(cfg, capsys):
    N, Cin, H, W = cfg["N"], cfg["Cin"], cfg["H"], cfg["W"]
    Cout, K, S, P = cfg["Cout"], cfg["K"], cfg["S"], cfg["P"]

    x = torch.randn(N, Cin, H, W)
    layer = Conv2D(in_channels=Cin, out_channels=Cout, kernel_size=K, stride=S, padding=P)
    ref = make_ref_conv(layer)

    # correctness
    with torch.no_grad():
        assert torch.allclose(layer.forward(x), ref(x), atol=1e-5)

    # warm-up
    for _ in range(WARMUP_ITERS):
        layer.forward(x)
        ref(x)

    our_ms = bench(lambda: layer.forward(x), BENCH_ITERS)
    torch_ms = bench(lambda: ref(x), BENCH_ITERS)
    ratio = our_ms / torch_ms

    with capsys.disabled():
        print(
            f"\n  {cfg}\n"
            f"    ours:  {our_ms:.3f} ms/iter\n"
            f"    torch: {torch_ms:.3f} ms/iter\n"
            f"    ratio: {ratio:.1f}x slower"
        )
