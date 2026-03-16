"""
Activation layers (ReLU, Sigmoid, Softmax) implemented as Layer subclasses.

To be implemented using tensor-level math only.
"""


import torch
from model.layers import Layer
from model.registry import LAYERS
from model.configs import ReLUConfig, SigmoidConfig, SoftmaxConfig


@LAYERS.register("relu")
class ReLU(Layer):
    config_model = ReLUConfig

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = torch.maximum(torch.zeros_like(x), x)
        return self.out

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        return torch.matmul(grad, torch.where(torch.gt(self.out, 0), 1))

@LAYERS.register("sigmoid")
class Sigmoid(Layer):
    config_model = SigmoidConfig

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = torch.special.expit(x)
        return self.out

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        return torch.matmul(grad, (torch.matmul(self.out, (torch.sub(1, self.out)))))