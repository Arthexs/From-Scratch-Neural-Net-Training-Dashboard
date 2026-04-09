"""
Network class that composes a list of Layer instances.
"""

import torch
from model.layers import Layer
from model.registry import LAYERS


class Network:
    def __init__(self, layers: list[Layer]) -> None:
        self.layers = layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self) -> list[torch.Tensor]:
        return [p for layer in self.layers for p in layer.parameters()]

    def to(self, device: torch.device | str) -> "Network":
        for layer in self.layers:
            layer.to(device)
        return self

    @classmethod
    def from_config(cls, config: list[dict]) -> "Network":
        """Build a Network from a list of layer config dicts, e.g. from a frontend preset.

        Each dict must have a "type" key matching a registered layer name,
        plus any keyword arguments for that layer's config.

        Example:
            Network.from_config([
                {"type": "dense", "input_size": 784, "output_size": 128},
                {"type": "relu"},
                {"type": "dense", "input_size": 128, "output_size": 10},
                {"type": "softmax"},
            ])
        """
        layers = []
        for cfg in config:
            cfg        = dict(cfg)
            layer_type = cfg.pop("type")
            layers.append(LAYERS.get(layer_type)(**cfg))
        return cls(layers)

    def summary(self, input_shape: tuple[int, ...]) -> None:
        x = torch.zeros(1, *input_shape)

        rows = []
        for layer in self.layers:
            in_shape  = tuple(x.shape[1:])
            x         = layer.forward(x)
            out_shape = tuple(x.shape[1:])
            n_params  = sum(p.numel() for p in layer.parameters())
            rows.append((type(layer).__name__, in_shape, out_shape, n_params))

        total = sum(r[3] for r in rows)

        col_w = [
            max(max(len(str(r[i])) for r in rows), len(h))
            for i, h in enumerate(("Layer", "Input", "Output", "Params"))
        ]

        header = f"{'Layer':<{col_w[0]}}  {'Input':<{col_w[1]}}  {'Output':<{col_w[2]}}  {'Params':>{col_w[3]}}"
        divider = "─" * len(header)
        print(header)
        print(divider)
        for name, in_shape, out_shape, n_params in rows:
            print(f"{name:<{col_w[0]}}  {str(in_shape):<{col_w[1]}}  {str(out_shape):<{col_w[2]}}  {n_params:>{col_w[3]},}")
        print(divider)
        print(f"{'Total':>{len(header) - len(f'{total:,}') - 2}}  {total:,}")
