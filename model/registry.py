"""
Registry for layers, losses, and optimizers.
"""

from typing import Any, Callable, Type

from pydantic import BaseModel


class Registry:
    """
    Registry for neural network components (layers, losses, optimizers).
    Each instance maintains its own independent registry dict.
    """

    def __init__(self, label: str) -> None:
        self._registry: dict[str, Type] = {}
        self._configs: dict[str, type[BaseModel]] = {}
        self.label = label

    def register(self, name: str, config: type[BaseModel]) -> Callable[[Type], Type]:
        """Decorator to register a component class under a given name."""

        def decorator(component_cls: Type) -> Type:
            if name in self._registry:
                raise ValueError(f"{self.label} '{name}' already registered.")
            self._registry[name] = component_cls
            self._configs[name] = config
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
        """Return a dict mapping names to their Pydantic config JSON schemas."""
        return {name: config_cls.model_json_schema() for name, config_cls in self._configs.items()}

    def keys(self) -> list[str]:
        """Return all registered names."""
        return list(self._registry.keys())


class FnRegistry:
    """Registry for named callables (no config required)."""

    def __init__(self, label: str) -> None:
        self._fns: dict[str, Callable] = {}
        self.label = label

    def register(self, name: str) -> Callable[[Callable], Callable]:
        def decorator(fn: Callable) -> Callable:
            if name in self._fns:
                raise ValueError(f"{self.label} '{name}' already registered.")
            self._fns[name] = fn
            return fn

        return decorator

    def get(self, name: str) -> Callable:
        if name not in self._fns:
            raise KeyError(f"Unknown {self.label}: '{name}'. Available: {list(self._fns.keys())}")
        return self._fns[name]

    def keys(self) -> list[str]:
        return list(self._fns.keys())


LAYERS = Registry("Layer")
LOSSES = Registry("Loss")
OPTIMIZERS = Registry("Optimizer")
INITIALIZERS = FnRegistry("Initializer")
