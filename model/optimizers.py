"""
Optimizer implementations (SGD, Adam) operating on raw tensors with gradients.

Implemented with manual update rules, no torch.optim.
"""

import torch

from model.configs import AdamConfig, SGDConfig
from model.registry import OPTIMIZERS


class Optimizer:
    config_model = None

    def __init__(self, **kwargs):
        if self.config_model is not None:
            config = self.config_model(**kwargs)
            for name, value in config.model_dump().items():
                setattr(self, name, value)

    def step(self, parameters: list[torch.Tensor]) -> None:
        raise NotImplementedError

    def zero_grad(self, parameters: list[torch.Tensor]) -> None:
        for p in parameters:
            if p.grad is not None:
                p.grad.zero_()


@OPTIMIZERS.register("sgd")
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

    config_model = SGDConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._velocity: dict[int, torch.Tensor] = {}

    def step(self, parameters: list[torch.Tensor]) -> None:
        for p in parameters:
            if p.grad is None:
                continue
            grad = p.grad + self.weight_decay * p.data

            if self.momentum > 0:
                pid = id(p)
                if pid not in self._velocity:
                    self._velocity[pid] = torch.zeros_like(p)
                self._velocity[pid] = self.momentum * self._velocity[pid] + grad
                grad = self._velocity[pid]

            p.data -= self.lr * grad


@OPTIMIZERS.register("adam")
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

    config_model = AdamConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._m: dict[int, torch.Tensor] = {}
        self._v: dict[int, torch.Tensor] = {}
        self._step: dict[int, int] = {}

    def step(self, parameters: list[torch.Tensor]) -> None:
        beta1, beta2 = self.betas
        for p in parameters:
            if p.grad is None:
                continue
            grad = p.grad + self.weight_decay * p.data

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

            p.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
