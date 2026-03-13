"""
Registry for layers, losses, and optimizers.

Implementation to be added according to project_structure.md.
"""

LAYERS = {}
LOSSES = {}
OPTIMIZERS = {}

# ----- Registry functions for layers -----

def register_layer(name):
    """Decorator to register a new layer under a given name."""
    def decorator(cls):
        if name in LAYERS:
            raise ValueError(f"Layer '{name}' already registered.")
        LAYERS[name] = cls
        return cls
    return decorator

def get_layer(name):
    """Retrieve a registered layer by name. Raise clear KeyError if not found."""
    if name not in LAYERS:
        raise KeyError(
            f"Unknown layer: '{name}'. "
            f"Available layers: {list(LAYERS.keys())}"
        )
    return LAYERS[name]

def layer_schemas():
    """
    Return a dict mapping layer names to their Pydantic config_model JSON schemas.
    Assumes each layer class has a 'config_model' attribute.
    """
    return {
        name: cls.config_model.model_json_schema()
        for name, cls in LAYERS.items()
    }

# ----- Registry functions for losses -----

def register_loss(name):
    """Decorator to register a new loss function under a given name."""
    def decorator(cls):
        if name in LOSSES:
            raise ValueError(f"Loss '{name}' already registered.")
        LOSSES[name] = cls
        return cls
    return decorator

def get_loss(name):
    """Retrieve a registered loss by name. Raise clear KeyError if not found."""
    if name not in LOSSES:
        raise KeyError(
            f"Unknown loss: '{name}'. "
            f"Available losses: {list(LOSSES.keys())}"
        )
    return LOSSES[name]

def loss_schemas():
    """
    Return a dict mapping loss names to their Pydantic config_model JSON schemas.
    Assumes each loss class has a 'config_model' attribute.
    """
    return {
        name: cls.config_model.model_json_schema()
        for name, cls in LOSSES.items()
    }

# ----- Registry functions for optimizers -----

def register_optimizer(name):
    """Decorator to register a new optimizer under a given name."""
    def decorator(cls):
        if name in OPTIMIZERS:
            raise ValueError(f"Optimizer '{name}' already registered.")
        OPTIMIZERS[name] = cls
        return cls
    return decorator

def get_optimizer(name):
    """Retrieve a registered optimizer by name. Raise clear KeyError if not found."""
    if name not in OPTIMIZERS:
        raise KeyError(
            f"Unknown optimizer: '{name}'. "
            f"Available optimizers: {list(OPTIMIZERS.keys())}"
        )
    return OPTIMIZERS[name]

def optimizer_schemas():
    """
    Return a dict mapping optimizer names to their Pydantic config_model JSON schemas.
    Assumes each optimizer class has a 'config_model' attribute.
    """
    return {
        name: cls.config_model.model_json_schema()
        for name, cls in OPTIMIZERS.items()
    }

