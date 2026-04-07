"""
Layer base class and concrete layer implementations (Dense, Conv2D, MaxPool, Flatten).

To be implemented using raw torch.Tensor operations, without torch.nn.
"""


import torch

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