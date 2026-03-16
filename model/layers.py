"""
Layer base class and concrete layer implementations (Dense, Conv2D, MaxPool, Flatten).

To be implemented using raw torch.Tensor operations, without torch.nn.
"""


import torch

class Layer:
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def parameters(self) -> list[torch.Tensor]:
        return []