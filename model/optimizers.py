"""
Optimizer implementations (SGD, Adam) operating on raw tensors with gradients.

Implemented with manual update rules, no torch.optim.
"""

from typing import Any

import torch

from model.configs import AdamConfig, SGDConfig
from model.registry import OPTIMIZERS


class Optimizer:
    def step(self, parameters: list[torch.Tensor]) -> None:
        raise NotImplementedError

    def zero_grad(self, parameters: list[torch.Tensor]) -> None:
        for p in parameters:
            if p.grad is not None:
                p.grad.zero_()


@OPTIMIZERS.register("sgd", config=SGDConfig)
class SGD(Optimizer):
    """SGD with optional momentum and L2 weight decay.

    Update rule (with momentum):
        v  = momentum * v + (grad + weight_decay * p)
        p -= lr * v
    Without momentum this reduces to vanilla gradient descent.

    Parameters passed to step() must be torch.Tensor objects where:
        - p.data  holds the current parameter values (read and updated in-place)
        - p.grad  holds the gradient dL/dp (must be populated before calling step(),
                  e.g. via loss.backward())
    """

    def __init__(self, **kwargs: Any):
        self._cfg = SGDConfig(**kwargs)
        self._velocity: dict[int, torch.Tensor] = {}

    def step(self, parameters: list[torch.Tensor]) -> None:
        for p in parameters:
            if p.grad is None:
                continue
            grad = p.grad + self._cfg.weight_decay * p.data

            if self._cfg.momentum > 0:
                pid = id(p)
                if pid not in self._velocity:
                    self._velocity[pid] = torch.zeros_like(p)
                self._velocity[pid] = self._cfg.momentum * self._velocity[pid] + grad
                grad = self._velocity[pid]

            p.data -= self._cfg.lr * grad


@OPTIMIZERS.register("adam", config=AdamConfig)
class Adam(Optimizer):
    """Adam optimiser with L2 weight decay.

    Update rule:
        m  = beta1 * m + (1 - beta1) * grad
        v  = beta2 * v + (1 - beta2) * grad²
        m̂  = m  / (1 - beta1^t)
        v̂  = v  / (1 - beta2^t)
        p -= lr * m̂ / (sqrt(v̂) + eps)

    Parameters passed to step() must be torch.Tensor objects where:
        - p.data  holds the current parameter values (read and updated in-place)
        - p.grad  holds the gradient dL/dp (must be populated before calling step(),
                  e.g. via loss.backward())
    """

    def __init__(self, **kwargs: Any):
        self._cfg = AdamConfig(**kwargs)
        self._m: dict[int, torch.Tensor] = {}
        self._v: dict[int, torch.Tensor] = {}
        self._step: dict[int, int] = {}

    def step(self, parameters: list[torch.Tensor]) -> None:
        beta1, beta2 = self._cfg.betas
        for p in parameters:
            if p.grad is None:
                continue
            grad = p.grad + self._cfg.weight_decay * p.data

            pid = id(p)
            if pid not in self._m:
                self._m[pid] = torch.zeros_like(p)
                self._v[pid] = torch.zeros_like(p)
                self._step[pid] = 0

            self._step[pid] += 1
            t = self._step[pid]

            self._m[pid] = beta1 * self._m[pid] + (1 - beta1) * grad
            self._v[pid] = beta2 * self._v[pid] + (1 - beta2) * grad**2

            m_hat = self._m[pid] / (1 - beta1**t)
            v_hat = self._v[pid] / (1 - beta2**t)

            p.data -= self._cfg.lr * m_hat / (torch.sqrt(v_hat) + self._cfg.eps)
