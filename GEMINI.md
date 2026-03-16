# 🧠 Project Context: MNIST From-Scratch Dashboard

## Core Mandate: "From Scratch"
- **Strict Rule:** No `torch.nn.Module`, `nn.Linear`, `nn.Conv2d`, or high-level PyTorch APIs.
- **Allowed:** `torch.Tensor` operations, `torch.autograd`, and `torchvision` (for data loading only).
- **Weight Management:** Weights must be raw tensors with `requires_grad=True`. 
- **Update Rule:** Manual updates: `param.data -= lr * param.grad`.

## Architecture & Registry Pattern
- **Registry:** All layers, losses, and optimizers must be registered via `@register_layer`, `@register_loss`, or `@register_optimizer` in `model/registry.py`.
- **Configs:** Every registered class MUST have a corresponding Pydantic `config_model` defined in `model/configs.py`.
- **Initialization:** All submodules are imported in `model/__init__.py` to trigger decorators.
- **Device Awareness:** Always use the `device` object from `training/device.py`.

## Hybrid Workflow Rules
- **Schema-First:** Backend `/api/options` drives the frontend. Do not hardcode UI dropdowns.
- **WebSocket:** Use `backend/stream.py` to push real-time metrics from the `logger.py` to the React `useTrainingStream.js` hook.
- **Validation:** Use Pydantic `Field` constraints (gt, le, validators) for all hyperparameters.

## Project Structure Reference
- `model/`: Raw tensor logic and registry.
- `training/`: Training loop and data loading.
- `backend/`: FastAPI + WebSocket streaming.
- `frontend/`: React + Recharts + Tailwind.