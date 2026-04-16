"""
Loss functions (e.g. CrossEntropy, MSE) implemented from scratch.

To be implemented without torch.nn loss modules.
"""

import torch

from model.configs import CrossEntropyLossConfig, MSELossConfig
from model.layers import Layer
from model.registry import LOSSES


@LOSSES.register("mse")
class MSELoss(Layer):
    config_model = MSELossConfig

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self._x = x
        self._y = y
        loss = torch.pow(torch.sub(x, y), 2)
        return self._reduce(loss)

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        n = self._x.numel() if self.reduction == "mean" else 1
        dx = torch.mul(2.0 / n, torch.sub(self._x, self._y))
        return torch.mul(grad, dx)

    def _reduce(self, loss: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss  # "none"


@LOSSES.register("bce")
class BCELoss(Layer):
    """Binary cross-entropy loss with sigmoid baked in (BCEWithLogits).

    Accepts raw logits and applies sigmoid internally using the numerically
    stable formulation:
        loss = max(x, 0) - x*y + log(1 + exp(-|x|))

    This avoids log(0) and overflow without needing clamping.
    The backward simplifies to: sigmoid(x) - y.
    """

    config_model = CrossEntropyLossConfig

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self._sig = torch.special.expit(x)
        self._y = y
        loss = torch.clamp(x, min=0) - torch.mul(x, y) + torch.log1p(torch.exp(-torch.abs(x)))
        return self._reduce(loss)

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        n = self._sig.numel() if self.reduction == "mean" else 1
        dx = torch.sub(self._sig, self._y)
        return torch.mul(grad, dx / n)

    def _reduce(self, loss: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss  # "none"
