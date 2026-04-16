"""
Registry for layers, losses, and optimizers.

Implementation to be added according to project_structure.md.
"""

from typing import Any, Callable, Type


class Registry:
    """
    Registry for neural network components (layers, losses, optimizers).
    Each instance maintains its own independent registry dict.
    """

    def __init__(self, label: str) -> None:
        self._registry: dict[str, Type] = {}
        self.label = label

    def register(self, name: str) -> Callable[[Type], Type]:
        """Decorator to register a component class under a given name."""

        def decorator(component_cls: Type) -> Type:
            if name in self._registry:
                raise ValueError(f"{self.label} '{name}' already registered.")
            if not hasattr(component_cls, "config_model"):
                raise AttributeError(
                    f"{self.label} class '{component_cls.__name__}' must define a 'config_model'."
                )
            self._registry[name] = component_cls
            return component_cls

        return decorator

    def get(self, name: str) -> Type:
        """Retrieve a registered component by name."""
        if name not in self._registry:
            raise KeyError(
                f"Unknown {self.label}: '{name}'. Available: {list(self._registry.keys())}"
            )
        return self._registry[name]

    def schemas(self) -> dict[str, dict[str, Any]]:
        """Return a dict mapping names to their Pydantic config_model JSON schemas."""
        return {
            name: component_cls.config_model.model_json_schema()
            for name, component_cls in self._registry.items()
        }

    def keys(self) -> list[str]:
        """Return all registered names."""
        return list(self._registry.keys())


LAYERS = Registry("Layer")
LOSSES = Registry("Loss")
OPTIMIZERS = Registry("Optimizer")
