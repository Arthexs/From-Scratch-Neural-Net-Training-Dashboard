"""
Tests for the model registry (lookup, duplicate registration, missing key errors).

To be implemented.
"""

from pydantic import ValidationError

from model.configs import (
    AdamConfig,
    Conv2DConfig,
    CrossEntropyLossConfig,
    DenseConfig,
    FlattenConfig,
    MaxPool2DConfig,
    MSELossConfig,
    ReLUConfig,
    SGDConfig,
    SigmoidConfig,
    SoftmaxConfig,
)


# --- Registry Compatibility Test ---
def test_registry_readiness():
    """Ensure all config models are Pydantic models and return valid JSON schemas."""
    all_configs = [
        DenseConfig,
        Conv2DConfig,
        MaxPool2DConfig,
        FlattenConfig,
        ReLUConfig,
        SigmoidConfig,
        SoftmaxConfig,
        MSELossConfig,
        CrossEntropyLossConfig,
        SGDConfig,
        AdamConfig,
    ]
    for cfg in all_configs:
        schema = cfg.model_json_schema()
        assert isinstance(schema, dict)
        assert "type" in schema
        assert schema["type"] == "object"
