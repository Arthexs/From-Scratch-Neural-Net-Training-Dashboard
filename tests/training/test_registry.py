import pytest

from model.registry import Registry
from training.registry import DATASETS


def test_datasets_is_registry():
    assert isinstance(DATASETS, Registry)


def test_mnist_registered():
    assert "mnist" in DATASETS.keys()


def test_get_unknown_raises():
    with pytest.raises(KeyError, match="nonexistent"):
        DATASETS.get("nonexistent")


def test_datasets_schemas_include_mnist():
    schemas = DATASETS.schemas()
    assert "mnist" in schemas
    assert isinstance(schemas["mnist"], dict)


def test_datasets_independent_from_model_registries():
    from model.registry import LAYERS, LOSSES, OPTIMIZERS

    assert DATASETS is not LAYERS
    assert DATASETS is not LOSSES
    assert DATASETS is not OPTIMIZERS
