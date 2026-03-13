"""
Pydantic configuration models for layers, losses, and optimizers.

To be implemented following the project structure plan.
"""


from pydantic import BaseModel, Field, field_validator
import torch


# ----- Layer Configs -----
class DenseConfig(BaseModel):
    input_size: int = Field(..., gt=0, description="Number of input features")
    output_size: int = Field(..., gt=0, description="Number of output features")
    bias: bool = Field(default=True, description="Whether to include a bias term")

class Conv2DConfig(BaseModel):
    in_channels: int = Field(..., gt=0, description="Number of input channels")
    out_channels: int = Field(..., gt=0, description="Number of output channels")
    kernel_size: int = Field(..., gt=0, description="Size of the convolutional kernel")
    stride: int = Field(default=1, gt=0, description="Stride of the convolutional operation")
    padding: int = Field(default=0, gt=0, description="Padding of the convolutional operation")
    bias: bool = Field(default=True, description="Whether to include a bias term")

    @field_validator('kernel_size')
    @classmethod
    def must_be_odd(cls, v: int) -> int:
        if v % 2 == 0:
            raise ValueError("Kernel size must be odd")
        return v

class MaxPool2DConfig(BaseModel):
    kernel_size: int = Field(..., gt=0, description="Size of the pooling kernel")
    stride: int = Field(default=1, gt=0, description="Stride of the pooling operation")
    padding: int = Field(default=0, gt=0, description="Padding of the pooling operation")

    @field_validator('kernel_size')
    @classmethod
    def must_be_odd(cls, v: int) -> int:
        if v % 2 == 0:
            raise ValueError("Kernel size must be odd")
        return v

class ReLUConfig(BaseModel):
    pass

class SigmoidConfig(BaseModel):
    pass

class SoftmaxConfig(BaseModel):
    dim: int = Field(default=-1, description="Dimension along which to apply softmax")


#----- Loss Configs -----
class MSELossConfig(BaseModel):
    reduction: str = Field(default="mean", description="Reduction method")

class CrossEntropyLossConfig(BaseModel):
    reduction: str = Field(default="mean", description="Reduction method")


#----- Optimizer Configs -----
class SGDConfig(BaseModel):
    params: list[torch.Tensor] = Field(..., description="List of parameters to optimize")
    lr: float = Field(default=1e-3, gt=0, description="Learning rate")
    momentum: float = Field(default=0, ge=0, le=1, description="Momentum coefficient")
    weight_decay: float = Field(default=0, ge=0, description="Weight decay coefficient")

class AdamConfig(BaseModel):
    params: list[torch.Tensor] = Field(..., description="List of parameters to optimize")
    lr: float = Field(default=1e-3, gt=0, description="Learning rate")
    betas: tuple[float, float] = Field(default=(0.9, 0.999), ge=0, le=1, description="Beta coefficients")
    eps: float = Field(default=1e-8, gt=0, description="Epsilon")
    weight_decay: float = Field(default=0, ge=0, description="Weight decay coefficient")
    