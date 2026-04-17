"""
Loss functions (e.g. CrossEntropy, MSE) implemented from scratch.

To be implemented without torch.nn loss modules.
"""

import torch

from model.configs import CrossEntropyLossConfig, MSELossConfig
from model.registry import LOSSES


class Loss:
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


@LOSSES.register("mse", config=MSELossConfig)
class MSELoss(Loss):
    def __init__(self, **kwargs):
        self._cfg = MSELossConfig(**kwargs)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self._x = x
        self._y = y
        loss = torch.pow(torch.sub(x, y), 2)
        return self._reduce(loss)

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        n = self._x.numel() if self._cfg.reduction == "mean" else 1
        dx = torch.mul(2.0 / n, torch.sub(self._x, self._y))
        return torch.mul(grad, dx)

    def _reduce(self, loss: torch.Tensor) -> torch.Tensor:
        if self._cfg.reduction == "mean":
            return loss.mean()
        if self._cfg.reduction == "sum":
            return loss.sum()
        return loss  # "none"


@LOSSES.register("bce", config=CrossEntropyLossConfig)
class BCELoss(Loss):
    """Binary cross-entropy loss with sigmoid baked in (BCEWithLogits).

    Accepts raw logits and applies sigmoid internally using the numerically
    stable formulation:
        loss = max(x, 0) - x*y + log(1 + exp(-|x|))

    This avoids log(0) and overflow without needing clamping.
    The backward simplifies to: sigmoid(x) - y.
    """

    def __init__(self, **kwargs):
        self._cfg = CrossEntropyLossConfig(**kwargs)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self._sig = torch.special.expit(x)
        self._y = y
        loss = torch.clamp(x, min=0) - torch.mul(x, y) + torch.log1p(torch.exp(-torch.abs(x)))
        return self._reduce(loss)

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        n = self._sig.numel() if self._cfg.reduction == "mean" else 1
        dx = torch.sub(self._sig, self._y)
        return torch.mul(grad, dx / n)

    def _reduce(self, loss: torch.Tensor) -> torch.Tensor:
        if self._cfg.reduction == "mean":
            return loss.mean()
        if self._cfg.reduction == "sum":
            return loss.sum()
        return loss  # "none"
