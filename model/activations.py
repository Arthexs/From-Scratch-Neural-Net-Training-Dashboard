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
        mask = torch.where(torch.gt(self.out, 0), torch.ones_like(self.out), torch.zeros_like(self.out))
        return torch.mul(grad, mask)

@LAYERS.register("sigmoid")
class Sigmoid(Layer):
    config_model = SigmoidConfig

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = torch.special.expit(x)
        return self.out

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        return torch.mul(grad, (torch.mul(self.out, (torch.sub(1, self.out)))))
    
@LAYERS.register("softmax")
class Softmax(Layer):
    config_model = SoftmaxConfig

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Log-max trick: subtract max for numerical stability before exponentiating
        x_shifted = torch.sub(x, x.max(dim=self.dim, keepdim=True).values)
        exp_x = torch.exp(x_shifted)
        self.out = torch.div(exp_x, exp_x.sum(dim=self.dim, keepdim=True))
        return self.out

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        # dx_i = S_i * (grad_i - sum_j(grad_j * S_j))
        # Elementwise grad*S gives per-element products; summing collapses to a scalar per sample
        # then broadcast-subtract before final elementwise mul with S
        dot = torch.sum(torch.mul(grad, self.out), dim=self.dim, keepdim=True)
        return torch.mul(self.out, torch.sub(grad, dot))
