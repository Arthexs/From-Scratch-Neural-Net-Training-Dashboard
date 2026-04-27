import pytest
from pydantic import ValidationError

from training.configs import DataConfig, MNISTConfig, TrainerConfig

# ── DataConfig ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("val_split", [0.0, 0.01, 0.2, 0.5, 0.99])
def test_data_config_valid_val_split(val_split):
    assert DataConfig(val_split=val_split).val_split == val_split


@pytest.mark.parametrize("val_split", [-0.1, 1.0, 1.5])
def test_data_config_invalid_val_split(val_split):
    with pytest.raises(ValidationError):
        DataConfig(val_split=val_split)


def test_data_config_default():
    assert DataConfig().val_split == 0.2


def test_data_config_extra_forbidden():
    with pytest.raises(ValidationError):
        DataConfig(val_split=0.2, unknown=True)


# ── MNISTConfig ────────────────────────────────────────────────────────────────


def test_mnist_config_defaults():
    cfg = MNISTConfig()
    assert cfg.mean == pytest.approx(0.1307)
    assert cfg.std == pytest.approx(0.3081)


@pytest.mark.parametrize("mean,std", [(0.0, 1.0), (0.5, 0.25), (1.0, 0.1)])
def test_mnist_config_custom(mean, std):
    cfg = MNISTConfig(mean=mean, std=std)
    assert cfg.mean == mean
    assert cfg.std == std


def test_mnist_config_extra_forbidden():
    with pytest.raises(ValidationError):
        MNISTConfig(unknown=1)


# ── TrainerConfig ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("epochs,batch_size", [(1, 1), (10, 32), (100, 128)])
def test_trainer_config_valid(epochs, batch_size):
    cfg = TrainerConfig(epochs=epochs, batch_size=batch_size)
    assert cfg.epochs == epochs
    assert cfg.batch_size == batch_size


@pytest.mark.parametrize(
    "field,value",
    [
        ("epochs", 0),
        ("epochs", -1),
        ("batch_size", 0),
        ("batch_size", -1),
        ("monitor_interval_s", 0.0),
        ("monitor_interval_s", -1.0),
    ],
)
def test_trainer_config_invalid(field, value):
    with pytest.raises(ValidationError):
        TrainerConfig(**{field: value})


def test_trainer_config_defaults():
    cfg = TrainerConfig()
    assert cfg.epochs == 10
    assert cfg.batch_size == 32
    assert cfg.log_per_batch_loss is True
    assert cfg.log_per_epoch_loss is True
    assert cfg.log_validation is True
    assert cfg.monitor_interval_s == 1.0


def test_trainer_config_extra_forbidden():
    with pytest.raises(ValidationError):
        TrainerConfig(unknown=1)
