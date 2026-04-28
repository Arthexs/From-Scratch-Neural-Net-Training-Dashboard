"""
Pydantic configuration models for layers, losses, and optimizers.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from model.registry import INITIALIZERS


class BaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InitializableConfig(BaseConfig):
    initializer: str = Field(default="xavier_uniform", description="Weight initialization method")

    @field_validator("initializer")
    @classmethod
    def valid_initializer(cls, v: str) -> str:
        valid = INITIALIZERS.keys()
        if v not in valid:
            raise ValueError(f"initializer must be one of {valid}, got '{v}'")
        return v


# ----- Layer Configs -----
class DenseConfig(InitializableConfig):
    input_size: int = Field(..., gt=0, description="Number of input features")
    output_size: int = Field(..., gt=0, description="Number of output features")
    bias: bool = Field(default=True, description="Add a learnable bias to the output")


class Conv2DConfig(InitializableConfig):
    in_channels: int = Field(..., gt=0, description="Number of input channels")
    out_channels: int = Field(..., gt=0, description="Number of output channels")
    kernel_size: int = Field(default=3, gt=0, description="Size of the convolutional kernel")
    stride: int = Field(default=1, gt=0, description="Stride of the convolutional operation")
    padding: int = Field(default=0, ge=0, description="Padding of the convolutional operation")
    bias: bool = Field(default=True, description="Add a learnable bias to the output")
    initializer: str = Field(default="kaiming_uniform", description="Weight initialization method")

    @field_validator("kernel_size")
    @classmethod
    def must_be_odd(cls, v: int) -> int:
        if v % 2 == 0:
            raise ValueError("Kernel size must be odd")
        return v


class MaxPool2DConfig(BaseConfig):
    kernel_size: int = Field(default=2, gt=0, description="Size of the pooling kernel")
    stride: int = Field(default=2, gt=0, description="Stride of the pooling operation")
    padding: int = Field(default=0, ge=0, description="Padding of the pooling operation")


class FlattenConfig(BaseConfig):
    start_dim: int = Field(default=1, description="First dimension to flatten")
    end_dim: int = Field(default=-1, description="Last dimension to flatten")


class ReLUConfig(BaseConfig):
    pass


class SigmoidConfig(BaseConfig):
    pass


class SoftmaxConfig(BaseConfig):
    dim: int = Field(default=-1, description="Dimension along which to apply softmax")


# ----- Loss Configs -----
class MSELossConfig(BaseConfig):
    reduction: Literal["mean", "sum", "none"] = Field(
        default="mean", description="How to reduce the loss: mean, sum, or none (per-sample losses)"
    )


class CrossEntropyLossConfig(BaseConfig):
    reduction: Literal["mean", "sum", "none"] = Field(
        default="mean", description="How to reduce the loss: mean, sum, or none (per-sample losses)"
    )


# ----- Optimizer Configs -----
class SGDConfig(BaseConfig):
    lr: float = Field(default=1e-3, gt=0, description="Learning rate")
    momentum: float = Field(default=0, ge=0, le=1, description="Momentum coefficient")
    weight_decay: float = Field(default=0, ge=0, description="Weight decay coefficient")


class AdamConfig(BaseConfig):
    lr: float = Field(default=1e-3, gt=0, description="Learning rate")
    betas: tuple[float, float] = Field(default=(0.9, 0.999), description="Beta coefficients")
    eps: float = Field(default=1e-8, gt=0, description="Epsilon")
    weight_decay: float = Field(default=0, ge=0, description="Weight decay coefficient")

    @field_validator("betas")
    @classmethod
    def must_be_in_range(cls, v: tuple[float, float]) -> tuple[float, float]:
        if not all(0.0 <= b < 1 for b in v):
            raise ValueError("Betas must be between 0 and 1")
        return v
