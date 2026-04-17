"""
Activation layers (ReLU, Sigmoid, Softmax) implemented as Layer subclasses.

To be implemented using tensor-level math only.
"""

from typing import cast

import torch

from model.configs import ReLUConfig, SigmoidConfig, SoftmaxConfig
from model.layers import Layer
from model.registry import LAYERS


@LAYERS.register("relu", config=ReLUConfig)
class ReLU(Layer):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = torch.maximum(torch.zeros_like(x), x)
        return self.out

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        mask = torch.where(
            torch.gt(self.out, 0), torch.ones_like(self.out), torch.zeros_like(self.out)
        )
        return torch.mul(grad, mask)


@LAYERS.register("sigmoid", config=SigmoidConfig)
class Sigmoid(Layer):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = cast(torch.Tensor, torch.special.expit(x))
        return self.out

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        return torch.mul(grad, (torch.mul(self.out, (torch.sub(1, self.out)))))


@LAYERS.register("softmax", config=SoftmaxConfig)
class Softmax(Layer):
    def __init__(self, **kwargs):
        self._cfg = SoftmaxConfig(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Log-max trick: subtract max for numerical stability before exponentiating
        x_shifted = torch.sub(x, x.max(dim=self._cfg.dim, keepdim=True).values)
        exp_x = torch.exp(x_shifted)
        self.out = torch.div(exp_x, exp_x.sum(dim=self._cfg.dim, keepdim=True))
        return self.out

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        # dx_i = S_i * (grad_i - sum_j(grad_j * S_j))
        # Elementwise grad*S gives per-element products; summing collapses to a scalar per sample
        # then broadcast-subtract before final elementwise mul with S
        dot = torch.sum(torch.mul(grad, self.out), dim=self._cfg.dim, keepdim=True)
        return torch.mul(self.out, torch.sub(grad, dot))
