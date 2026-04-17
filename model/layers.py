"""
Layer base class and concrete layer implementations (Dense, Conv2D, MaxPool, Flatten).

To be implemented using raw torch.Tensor operations, without torch.nn.
"""

import torch

from model.configs import Conv2DConfig, DenseConfig, FlattenConfig, MaxPool2DConfig
from model.registry import LAYERS


class Layer:
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def parameters(self) -> list[torch.Tensor]:
        return []

    def to(self, device: torch.device | str) -> "Layer":
        for p in self.parameters():
            p.data = p.data.to(device)
        return self


_INITIALIZERS = {
    "xavier_uniform": torch.nn.init.xavier_uniform_,
    "xavier_normal": torch.nn.init.xavier_normal_,
    "kaiming_uniform": torch.nn.init.kaiming_uniform_,
    "kaiming_normal": torch.nn.init.kaiming_normal_,
}


@LAYERS.register("dense", config=DenseConfig)
class Dense(Layer):
    """Fully connected layer: out = x @ W + b.

    Parameters (returned by parameters()) are plain tensors with no
    autograd tracking. Gradients are computed manually in backward()
    and stored in W.grad / b.grad so the optimizer can read them.
    """

    def __init__(self, **kwargs):
        self._cfg = DenseConfig(**kwargs)
        self.W = torch.empty(self._cfg.input_size, self._cfg.output_size)
        _INITIALIZERS[self._cfg.initializer](self.W)
        self.b = torch.zeros(self._cfg.output_size) if self._cfg.bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._x = x
        out = torch.matmul(x, self.W)
        if self._cfg.bias:
            out = torch.add(out, self.b)
        return out

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        self.W.grad = torch.matmul(self._x.T, grad)  # (input_size, output_size)
        if self._cfg.bias:
            self.b.grad = grad.sum(dim=0)  # (output_size,)
        return torch.matmul(grad, self.W.T)  # (batch, input_size)

    def parameters(self) -> list[torch.Tensor]:
        return [self.W, self.b] if self._cfg.bias else [self.W]


@LAYERS.register("conv2d", config=Conv2DConfig)
class Conv2D(Layer):
    """2D convolution via im2col (unfold) + matmul.

    Forward:
        1. Unfold input (N, Cin, H, W)
               → (N, Cin*Kh*Kw, Oh*Ow)
        2. Reshape kernel (Cout, Cin, Kh, Kw)
               → (Cout, Cin*Kh*Kw)
        3. Matmul → (N, Cout, Oh*Ow)
        4. Reshape → (N, Cout, Oh, Ow)

    Parameters are plain tensors; gradients are set manually in backward()
    so the optimizer can read them via W.grad / b.grad.
    """

    def __init__(self, **kwargs):
        self._cfg = Conv2DConfig(**kwargs)
        self.W = torch.empty(
            self._cfg.out_channels,
            self._cfg.in_channels,
            self._cfg.kernel_size,
            self._cfg.kernel_size,
        )
        _INITIALIZERS[self._cfg.initializer](self.W)
        self.b = torch.zeros(self._cfg.out_channels) if self._cfg.bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, Cin, H, W = x.shape
        Cout, _, Kh, Kw = self.W.shape

        Oh = (H + 2 * self._cfg.padding - Kh) // self._cfg.stride + 1
        Ow = (W + 2 * self._cfg.padding - Kw) // self._cfg.stride + 1

        self._input_hw = (H, W)

        # (N, Cin*Kh*Kw, Oh*Ow)
        self._x_unfolded = torch.nn.functional.unfold(
            x, kernel_size=(Kh, Kw), stride=self._cfg.stride, padding=self._cfg.padding
        )

        # (Cout, Cin*Kh*Kw) @ (N, Cin*Kh*Kw, Oh*Ow) → (N, Cout, Oh*Ow)
        W_col = self.W.view(Cout, -1)
        out = torch.matmul(W_col, self._x_unfolded)

        # (N, Cout, Oh, Ow)
        out = out.view(N, Cout, Oh, Ow)

        if self._cfg.bias:
            out = torch.add(out, self.b.view(1, Cout, 1, 1))

        return out

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        N, Cout, Oh, Ow = grad.shape
        _, Cin, Kh, Kw = self.W.shape

        grad_flat = grad.view(N, Cout, -1)  # (N, Cout, Oh*Ow)
        W_col = self.W.view(Cout, -1)  # (Cout, Cin*Kh*Kw)

        # (N, Cout, Oh*Ow) @ (N, Oh*Ow, Cin*Kh*Kw) → (N, Cout, Cin*Kh*Kw) → sum over batch
        dW_col = torch.matmul(grad_flat, self._x_unfolded.transpose(1, 2)).sum(dim=0)
        self.W.grad = dW_col.view_as(self.W)

        if self._cfg.bias:
            self.b.grad = grad.sum(dim=(0, 2, 3))  # (Cout,)

        # (Cin*Kh*Kw, Cout) @ (N, Cout, Oh*Ow) → (N, Cin*Kh*Kw, Oh*Ow),
        # then fold back to (N, Cin, H, W)
        dx_unfolded = torch.matmul(W_col.T, grad_flat)
        return torch.nn.functional.fold(
            dx_unfolded,
            output_size=self._input_hw,
            kernel_size=(Kh, Kw),
            stride=self._cfg.stride,
            padding=self._cfg.padding,
        )

    def parameters(self) -> list[torch.Tensor]:
        return [self.W, self.b] if self._cfg.bias else [self.W]


@LAYERS.register("maxpool2d", config=MaxPool2DConfig)
class MaxPool2D(Layer):
    def __init__(self, **kwargs):
        self._cfg = MaxPool2DConfig(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        K = self._cfg.kernel_size

        Oh = (H + 2 * self._cfg.padding - K) // self._cfg.stride + 1
        Ow = (W + 2 * self._cfg.padding - K) // self._cfg.stride + 1

        # (N, C*K*K, Oh*Ow) → (N, C, K*K, Oh*Ow) → max over window axis
        x_unfolded = torch.nn.functional.unfold(
            x, kernel_size=K, padding=self._cfg.padding, stride=self._cfg.stride
        )
        x_max, x_argmax = x_unfolded.view(N, C, K * K, Oh * Ow).max(dim=2)

        self._x_argmax = x_argmax  # (N, C, Oh*Ow) — needed by backward
        self._input_hw = (H, W)

        return x_max.view(N, C, Oh, Ow)

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        N, C, Oh, Ow = grad.shape
        H, W = self._input_hw
        K = self._cfg.kernel_size

        # Scatter upstream grad to argmax positions: (N, C, K*K, Oh*Ow)
        dx = torch.zeros(N, C, K * K, Oh * Ow, dtype=grad.dtype, device=grad.device)
        dx.scatter_(2, self._x_argmax.unsqueeze(2), grad.reshape(N, C, -1).unsqueeze(2))

        # Fold back to input shape (N, C, H, W)
        return torch.nn.functional.fold(
            dx.view(N, C * K * K, Oh * Ow),
            output_size=(H, W),
            kernel_size=K,
            padding=self._cfg.padding,
            stride=self._cfg.stride,
        )


@LAYERS.register("flatten", config=FlattenConfig)
class Flatten(Layer):
    def __init__(self, **kwargs):
        self._cfg = FlattenConfig(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._input_shape = x.shape
        return torch.flatten(x, self._cfg.start_dim, self._cfg.end_dim)

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        return grad.view(self._input_shape)
