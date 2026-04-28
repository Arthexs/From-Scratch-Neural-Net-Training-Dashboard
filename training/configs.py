"""
Pydantic configuration models for data loading and training.
"""

from pydantic import Field, field_validator

from model.configs import BaseConfig
from training.registry import METRICS


# ----- Data Config -----
class DataConfig(BaseConfig):
    val_split: float = Field(
        default=0.2, ge=0, lt=1, description="Fraction of training data for validation"
    )


class MNISTConfig(BaseConfig):
    mean: float = Field(
        default=0.1307, description="Channel mean for normalisation (precomputed on training split)"
    )
    std: float = Field(
        default=0.3081, description="Channel std for normalisation (precomputed on training split)"
    )


# ----- Trainer Config -----
class TrainerConfig(BaseConfig):
    epochs: int = Field(default=10, gt=0, description="Number of training epochs")
    batch_size: int = Field(default=32, gt=0, description="Batch size")
    log_per_batch_loss: bool = Field(default=True, description="Emit loss after every batch")
    log_per_epoch_loss: bool = Field(
        default=True, description="Emit aggregated loss after every epoch"
    )
    log_validation: bool = Field(
        default=True, description="Run and emit validation loss each epoch"
    )
    monitor_interval_s: float = Field(
        default=1.0, gt=0, description="How often the monitor thread samples CPU/GPU resources"
    )
    log_grad_norm: bool = Field(
        default=False, description="Emit global L2 gradient norm after every batch"
    )
    metrics: list[str] = Field(
        default_factory=list,
        description="Metric names to compute each epoch (keys from METRICS registry)",
    )
    save_checkpoints: bool = Field(
        default=False, description="Save model weights when training completes naturally"
    )
    checkpoint_dir: str = Field(
        default="runs/", description="Directory for checkpoint files; filename is {run_id}_checkpoint.pt"
    )

    @field_validator("metrics")
    @classmethod
    def valid_metrics(cls, v: list[str]) -> list[str]:
        invalid = [m for m in v if m not in METRICS.keys()]
        if invalid:
            raise ValueError(f"Unknown metrics: {invalid}. Available: {METRICS.keys()}")
        return v


# ----- Logger Config -----
class LoggerConfig(BaseConfig):
    db_path: str = Field(default="runs/runs.db", description="Path to the SQLite database file; directory is created on first use")
    queue_timeout: float = Field(default=0.1, gt=0, description="Seconds to block on queue.get() before looping")
