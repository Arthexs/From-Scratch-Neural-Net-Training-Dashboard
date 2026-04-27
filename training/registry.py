"""
Registries for training-side components (datasets, metrics).
"""

from model.registry import FnRegistry, Registry

DATASETS = Registry("Dataset")
METRICS = FnRegistry("Metric")
