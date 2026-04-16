"""
Model package: from-scratch neural network components.

Files in this package are implemented using raw PyTorch tensor operations only
and must not depend on torch.nn or higher-level abstractions.
"""

import model.activations  # noqa: F401 — side-effect: registers relu, sigmoid, softmax
import model.layers  # noqa: F401 — side-effect: registers dense, conv2d, maxpool2d, flatten
import model.losses  # noqa: F401 — side-effect: registers mse, bce
import model.optimizers  # noqa: F401 — side-effect: registers sgd, adam
