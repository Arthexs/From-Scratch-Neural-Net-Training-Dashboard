# Claude Code Instructions

## Project overview
A from-scratch neural network training dashboard. The model is implemented using raw PyTorch tensor operations. A frontend (not yet built) will consume training metrics streamed from the backend.

## Hard constraints
- **No `torch.nn` modules** — layers, losses, and activations must be implemented with raw tensor ops only. `torch.nn.functional.unfold` / `fold` are the only `torch.nn` exceptions (used for im2col in Conv2D and MaxPool2D).
- **No `torch.autograd`** — gradients are computed manually in each layer's `backward()`.
- **No `torch.optim`** — optimizers are implemented from scratch.

## Architecture

### `model/`
- `layers.py` — `Dense`, `Conv2D`, `MaxPool2D`, `Flatten`
- `activations.py` — `ReLU`, `Sigmoid`, `Softmax`
- `losses.py` — `MSELoss`, `BCELoss`
- `optimizers.py` — `SGD`, `Adam`
- `network.py` — `Network`: composes layers, exposes `forward`, `backward`, `parameters`, `to`, `summary`, `from_config`
- `configs.py` — Pydantic configs for all components (`extra="forbid"` on base)
- `registry.py` — `LAYERS`, `LOSSES`, `OPTIMIZERS` registries (decorator-based)
- `presets.py` — `MLP_BASELINE`, `CNN_SMALL` as lists of layer config dicts
- `__init__.py` — imports all modules to trigger registry decoration

### `training/` (in progress)
- `device.py` — CUDA if available, else CPU
- `data.py` — MNIST DataLoader (TODO)
- `logger.py` — metric logging (TODO)
- `trainer.py` — training loop (TODO)

## Layer pattern
Every layer:
- Subclasses `Layer`, sets `config_model`
- `forward(x)` caches state needed for backward, returns output
- `backward(grad)` computes and stores `.grad` on parameters, returns `dx`
- `parameters()` returns `[self.W, self.b]` (or `[]` for no-param layers)
- No `__init__` needed unless allocating parameter tensors

GPU support is free — tensors move via `.to(device)`, all ops dispatch to CUDA automatically.

## Adding a new layer
1. Add a `XxxConfig(BaseConfig)` in `configs.py`
2. Implement `@LAYERS.register("xxx") class Xxx(Layer)` in `layers.py`
3. Add the import comment to `model/__init__.py` if in a new file
4. Add tests in `tests/test_layers.py` comparing against a `torch.nn` reference

## Working style — hybrid approach

Arthur is the pilot, Claude is the driver. Follow this protocol on every task:

**Before writing any code:**
1. State which files you will touch and what you plan to change in each, in 3–6 bullet points
2. In the same response, flag any open questions — missing type signatures, ambiguous behaviour, unclear scope, edge cases. Do not ask questions in a separate turn and do not assume and proceed.
3. Wait for explicit approval ("go ahead", "yes", "looks good") before writing anything. Approval also covers the answers to your questions — if Arthur answers them, proceed without asking again.

**While writing code:**
- Make the minimal change that satisfies the request — no unrequested refactors or additions
- Follow existing patterns in the file (naming, docstring style, error handling)
- Respect all hard constraints listed above without exception

**After writing code:**
- Summarise what was changed and why in 2–4 lines
- Flag anything that still needs attention or follow-up

**When to ask questions (not proceed):**
- The request touches multiple files but doesn't specify which one is the source of truth
- A design decision has non-trivial trade-offs (e.g. where to put a new config, whether to change an interface)
- The request contradicts a hard constraint or an established pattern
- Expected behaviour at edge cases is unspecified

## Running tests
```bash
pytest                          # all tests
pytest tests/test_layers.py -v  # layers only
pytest -k maxpool -v -s         # filter by name, show prints
```

## Presets format
Presets are lists of dicts with a `"type"` key matching a registered layer name:
```python
Network.from_config([
    {"type": "dense", "input_size": 784, "output_size": 128},
    {"type": "relu"},
])
```
