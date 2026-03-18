import pytest
from pydantic import ValidationError
from model.configs import (
    DenseConfig, Conv2DConfig, MaxPool2DConfig, FlattenConfig,
    ReLUConfig, SigmoidConfig, SoftmaxConfig,
    MSELossConfig, CrossEntropyLossConfig,
    SGDConfig, AdamConfig
)

# --- Positive Tests: (ConfigClass, ValidParams) ---
@pytest.mark.parametrize("config_cls, params", [
    # DenseConfig
    (DenseConfig, {"input_size": 784, "output_size": 10}),
    (DenseConfig, {"input_size": 1, "output_size": 1, "bias": False}),
    (DenseConfig, {"input_size": 1024, "output_size": 1024}),
    # Conv2DConfig
    (Conv2DConfig, {"in_channels": 1, "out_channels": 16, "kernel_size": 3}),
    (Conv2DConfig, {"in_channels": 3, "out_channels": 64, "kernel_size": 5, "stride": 2, "padding": 2}),
    (Conv2DConfig, {"in_channels": 16, "out_channels": 32, "kernel_size": 1, "bias": False}),
    # MaxPool2DConfig
    (MaxPool2DConfig, {"kernel_size": 2, "stride": 2}),
    (MaxPool2DConfig, {"kernel_size": 3, "stride": 1, "padding": 1}),
    (MaxPool2DConfig, {"kernel_size": 5, "stride": 5, "padding": 0}),
    # FlattenConfig
    (FlattenConfig, {"start_dim": 1, "end_dim": -1}),
    (FlattenConfig, {"start_dim": 0, "end_dim": 2}),
    (FlattenConfig, {}),  # Defaults
    # ReLUConfig
    (ReLUConfig, {}),
    # SigmoidConfig
    (SigmoidConfig, {}),
    # SoftmaxConfig
    (SoftmaxConfig, {"dim": -1}),
    (SoftmaxConfig, {"dim": 0}),
    (SoftmaxConfig, {"dim": 1}),
    # Loss Configs
    (MSELossConfig, {"reduction": "mean"}),
    (MSELossConfig, {"reduction": "sum"}),
    (MSELossConfig, {"reduction": "none"}),
    (CrossEntropyLossConfig, {"reduction": "mean"}),
    (CrossEntropyLossConfig, {"reduction": "sum"}),
    (CrossEntropyLossConfig, {"reduction": "none"}),
    # SGDConfig
    (SGDConfig, {"lr": 0.01}),
    (SGDConfig, {"lr": 1e-3, "momentum": 0.9}),
    (SGDConfig, {"lr": 0.1, "momentum": 0.0, "weight_decay": 1e-4}),
    # AdamConfig
    (AdamConfig, {"lr": 0.001}),
    (AdamConfig, {"lr": 1e-4, "betas": (0.9, 0.99)}),
    (AdamConfig, {"lr": 0.01, "eps": 1e-7, "weight_decay": 0.01}),
])
def test_configs_positive(config_cls, params):
    """Ensure valid parameters pass validation and preserve values."""
    config = config_cls(**params)
    for key, value in params.items():
        assert getattr(config, key) == value

# --- Negative Tests: (ConfigClass, InvalidParams) ---
@pytest.mark.parametrize("config_cls, params", [
    # DenseConfig
    (DenseConfig, {"input_size": 0, "output_size": 10}),
    (DenseConfig, {"input_size": 10, "output_size": -1}),
    (DenseConfig, {"input_size": "not-an-int", "output_size": 10}),
    # Conv2DConfig
    (Conv2DConfig, {"in_channels": 1, "out_channels": 16, "kernel_size": 4}),  # Even kernel
    (Conv2DConfig, {"in_channels": -1, "out_channels": 16}),
    (Conv2DConfig, {"in_channels": 1, "out_channels": 16, "stride": 0}),
    # MaxPool2DConfig
    (MaxPool2DConfig, {"kernel_size": 0}),
    (MaxPool2DConfig, {"stride": -1}),
    (MaxPool2DConfig, {"padding": -1}),
    # Loss Configs
    (MSELossConfig, {"reduction": "invalid"}),
    (CrossEntropyLossConfig, {"reduction": 123}),
    (CrossEntropyLossConfig, {"reduction": ""}),
    # SGDConfig
    (SGDConfig, {"lr": 0}),
    (SGDConfig, {"momentum": 1.1}),
    (SGDConfig, {"momentum": -0.1}),
    # AdamConfig
    (AdamConfig, {"betas": (0.9, 1.0)}),  # Beta 2 must be < 1
    (AdamConfig, {"betas": (-0.1, 0.9)}), # Beta 1 must be >= 0
    (AdamConfig, {"eps": 0}),
    # Typing Tests
    (DenseConfig, {"input_size": "not-an-int", "output_size": 10}),
    (Conv2DConfig, {"in_channels": [1], "out_channels": 16}),
    (MaxPool2DConfig, {"kernel_size": {"size": 3}}),
    (SGDConfig, {"lr": None}),
    (AdamConfig, {"betas": "0.9, 0.999"}),
    (ReLUConfig, {"extra_param": 1}),  # ReLU takes no params
])
def test_configs_negative(config_cls, params):
    """Ensure invalid parameters raise ValidationError."""
    with pytest.raises(ValidationError):
        config_cls(**params)

