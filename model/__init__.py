"""
Model package: from-scratch neural network components.

Files in this package are implemented using raw PyTorch tensor operations only
and must not depend on torch.nn or higher-level abstractions.
"""

import model.layers       # registers: dense, conv2d, maxpool2d, flatten
import model.activations  # registers: relu, sigmoid, softmax
import model.losses       # registers: mse, bce
import model.optimizers   # registers: sgd, adam