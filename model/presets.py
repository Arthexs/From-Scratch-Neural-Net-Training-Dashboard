"""
Named architecture presets for the training dashboard.

Each preset is a list of layer config dicts as the frontend would send them.
Pass to Network.from_config() to build a Network.
"""

# Flat MLP for MNIST (784 → 128 → 64 → 10)
MLP_BASELINE = [
    {"type": "dense", "input_size": 784, "output_size": 128},
    {"type": "relu"},
    {"type": "dense", "input_size": 128, "output_size": 64},
    {"type": "relu"},
    {"type": "dense", "input_size": 64, "output_size": 10},
    {"type": "softmax"},
]

# Small CNN for MNIST (1x28x28 → conv → pool → dense → 10)
CNN_SMALL = [
    {"type": "conv2d", "in_channels": 1, "out_channels": 8, "kernel_size": 3, "padding": 1},
    {"type": "relu"},
    {"type": "maxpool2d", "kernel_size": 2, "stride": 2},
    {"type": "conv2d", "in_channels": 8, "out_channels": 16, "kernel_size": 3, "padding": 1},
    {"type": "relu"},
    {"type": "maxpool2d", "kernel_size": 2, "stride": 2},
    {"type": "flatten"},
    {"type": "dense", "input_size": 784, "output_size": 10},
    {"type": "softmax"},
]
