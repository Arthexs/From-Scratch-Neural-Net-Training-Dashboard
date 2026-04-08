"""
Layer base class and concrete layer implementations (Dense, Conv2D, MaxPool, Flatten).

To be implemented using raw torch.Tensor operations, without torch.nn.
"""


import torch
from model.registry import LAYERS
from model.configs import DenseConfig, FlattenConfig


class Layer:
    config_model = None

    def __init__(self, **kwargs):
        if self.config_model is not None:
            config = self.config_model(**kwargs)
            for name, value in config.model_dump().items():
                setattr(self, name, value)

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
    "xavier_uniform":  torch.nn.init.xavier_uniform_,
    "xavier_normal":   torch.nn.init.xavier_normal_,
    "kaiming_uniform": torch.nn.init.kaiming_uniform_,
    "kaiming_normal":  torch.nn.init.kaiming_normal_,
}


@LAYERS.register("dense")
class Dense(Layer):
    """Fully connected layer: out = x @ W + b.

    Parameters (returned by parameters()) are plain tensors with no
    autograd tracking. Gradients are computed manually in backward()
    and stored in W.grad / b.grad so the optimizer can read them.
    """

    config_model = DenseConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.W = torch.empty(self.input_size, self.output_size)
        _INITIALIZERS[self.initializer](self.W)
        self.b = torch.zeros(self.output_size) if self.bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._x = x
        out = torch.matmul(x, self.W)
        if self.bias:
            out = torch.add(out, self.b)
        return out

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        self.W.grad = torch.matmul(self._x.T, grad)          # (input_size, output_size)
        if self.bias:
            self.b.grad = grad.sum(dim=0)                     # (output_size,)
        return torch.matmul(grad, self.W.T)                   # (batch, input_size)

    def parameters(self) -> list[torch.Tensor]:
        return [self.W, self.b] if self.bias else [self.W]


@LAYERS.register("flatten")
class Flatten(Layer):
    config_model = FlattenConfig

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._input_shape = x.shape
        return torch.flatten(x, self.start_dim, self.end_dim)

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        return grad.view(self._input_shape)