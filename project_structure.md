# 🧠 Neural Net Training Dashboard — Project Structure

## Overview
A weekend project: build a CNN/MLP from scratch using raw PyTorch tensors (no `nn.Module`,
no built-in layers), train it on MNIST with full GPU support, and watch it learn in real
time via a React dashboard. The frontend lets you configure hyperparameters, pick a preset
model, choose optimizer/loss/device — and an advanced tab lets you build a custom architecture
layer by layer.

**"From scratch" rules:** You can use `torch.Tensor` operations, `torch.autograd` for
gradient tracking, and `torchvision` for MNIST loading only. No `nn.Linear`, `nn.Conv2d`,
`nn.Module`, or any higher-level API inside `model/`.

---

## Folder Structure

```
training-dashboard/
│
├── model/
│   ├── __init__.py                  # Imports all submodules to trigger @register decorators
│   ├── registry.py                  # Separate dicts: LAYERS / LOSSES / OPTIMIZERS
│   │                                #   each one has .registry .get .schemas and .keys
│   ├── configs.py                   # All Pydantic config models in one place
│   │                                #   Conv2DConfig, DenseConfig, AdamConfig, SGDConfig, etc.
│   │                                #   Field constraints (gt=0, must_be_odd validators, etc.)
│   ├── layers.py                    # Layer base class + concrete layers
│   │                                #   Base: forward(), backward(), parameters()
│   │                                #   Each class: config_model attr + @register_layer
│   │                                #   Layers: Dense, Conv2D, MaxPool, Flatten
│   │                                #   Weights: torch.randn(..., requires_grad=True)
│   ├── activations.py               # ReLU, Sigmoid, Softmax — Layer subclasses, also registered
│   │                                #   Slot into architecture list like any other layer
│   ├── losses.py                    # CrossEntropy, MSE — each with @register_loss + config model
│   │                                #   Manual: -(targets * torch.log(probs)).sum()
│   ├── optimizers.py                # SGD, Adam — each with @register_optimizer + config model
│   │                                #   Manual update: param.data -= lr * param.grad
│   ├── network.py                   # Network class — takes a list of Layer instances
│   │                                #   forward(), parameters(), .to(device) on init
│   │                                #   Knows nothing about layer names — purely structural
│   └── presets.py                   # Named architectures as plain config dicts
│                                    #   CNN_SMALL, MLP_BASELINE
│                                    #   Same format as custom architecture payload from frontend
│                                    #   → both go through the identical registry build path
│
├── training/
│   ├── __init__.py
│   ├── device.py                    # device = torch.device("cuda" if available else "cpu")
│   ├── data.py                      # MNIST via torchvision — DataLoader, normalize
│   │                                #   pin_memory=True for faster GPU transfers
│   ├── trainer.py                   # Training loop — moves batches to device, emits metrics
│   └── logger.py                    # Metric store — .item() to detach tensors from GPU
│
├── backend/
│   ├── __init__.py
│   ├── main.py                      # FastAPI app entrypoint + mounts routes
│   ├── schemas.py                   # Pydantic models for API request/response
│   │                                #   TrainingConfig: preset name OR architecture list
│   │                                #   + optimizer, loss, hyperparams, device choice
│   ├── routes.py                    # POST /api/start  — validate, build + launch training run
│   │                                # POST /api/stop   — stop current run
│   │                                # GET  /api/status — is a run active?
│   │                                # GET  /api/options — registry keys + Pydantic schemas
│   │                                #   frontend calls this on mount to populate all UI
│   └── stream.py                    # WebSocket /ws/metrics — pushes logger output each batch
│
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── App.jsx                  # Root layout — tab switching: Simple / Advanced / Live
│   │   ├── api/
│   │   │   └── client.js            # fetch() wrappers: getOptions(), startTraining(), stopTraining()
│   │   ├── hooks/
│   │   │   ├── useOptions.js        # Fetches /api/options on mount, exposes registry data
│   │   │   └── useTrainingStream.js # WebSocket → appends incoming metrics to state array
│   │   ├── components/
│   │   │   ├── config/
│   │   │   │   ├── SimpleConfig.jsx     # Preset picker + hyperparameter inputs
│   │   │   │   │                        #   All dropdowns driven by useOptions — never hardcoded
│   │   │   │   ├── AdvancedConfig.jsx   # Layer-by-layer architecture builder
│   │   │   │   │                        #   Renders dynamic fields per layer from Pydantic schema
│   │   │   │   └── LayerCard.jsx        # Single layer row: type picker + its config fields
│   │   │   ├── charts/
│   │   │   │   ├── LossCurve.jsx        # Recharts LineChart — live {batch, loss} points
│   │   │   │   ├── AccuracyCurve.jsx    # Same pattern for accuracy
│   │   │   │   └── LRSchedule.jsx       # Learning rate over time (extension)
│   │   │   └── dashboard/
│   │   │       ├── MetricsBar.jsx       # Epoch, batch, loss, accuracy, device indicator
│   │   │       └── Controls.jsx         # Start / Stop / Reset — calls api/client.js
│   │   └── index.jsx
│   ├── package.json
│   └── vite.config.js
│
├── notebooks/
│   └── scratch.ipynb                # Experiment with layer math before coding it
│                                    #   Key check: can the network overfit a single GPU batch?
│
├── tests/
│   ├── test_registry.py             # Lookup, duplicate registration, missing key errors
│   ├── test_configs.py              # Pydantic validation: wrong types, violated constraints
│   ├── test_layers.py               # Gradient checks: manual grads vs torch.autograd
│   └── test_network.py              # Forward pass shape checks + GPU/CPU parity
│
├── requirements.txt                 # torch, torchvision, fastapi, uvicorn, pydantic, python-multipart
├── .cursorrules                     # Cursor project instructions (see below)
└── README.md
```

---

## Data Flow (how the pieces connect)

```
Frontend (React)
    │
    ├── on mount → GET /api/options
    │               └── backend serializes registry keys + Pydantic JSON schemas
    │               └── frontend populates all dropdowns + layer field renderers dynamically
    │
    ├── user clicks Start → POST /api/start
    │               body: { preset: "cnn_small", optimizer: "adam", loss: "cross_entropy",
    │                        device: "cuda", hyperparams: { lr: 0.001, batch_size: 64 } }
    │               OR:   { architecture: [ {type: "conv2d", ...}, {type: "relu"}, ... ],
    │                        optimizer: "adam", ... }
    │               └── FastAPI validates against TrainingConfig (Pydantic)
    │               └── backend calls get_layer(name)(**config) for each layer in list
    │               └── assembles Network, Optimizer, Loss from registry
    │               └── starts trainer in background thread
    │
    └── WebSocket /ws/metrics (stays open for the full training run)
                    └── trainer calls logger each batch
                    └── stream.py pushes { epoch, batch, loss, accuracy, lr } as JSON
                    └── useTrainingStream appends to state → charts re-render automatically
```

---

## Build Order (Weekend Plan)

### Saturday Morning — The Model
Write this yourself, use Cursor only to check/refine:
1. `training/device.py` — one-liner, sets the GPU mindset from the start
2. `model/registry.py` — three dicts, decorators, safe lookup functions
3. `model/configs.py` — Pydantic models for Dense, Conv2D, SGD, Adam, CrossEntropy
4. `model/activations.py` — simplest tensor ops, register each one
5. `model/layers.py` — Dense first (`torch.matmul`), then Conv2D; attach `config_model`
6. `model/losses.py` + `model/optimizers.py` — manual impls, registered
7. `model/__init__.py` — import all submodules so decorators fire on startup
8. `model/presets.py` — CNN_SMALL + MLP_BASELINE as config dicts
9. `model/network.py` — wire layers, `.to(device)` on init
10. `notebooks/scratch.ipynb` — can it overfit a single tiny GPU batch?

> 💡 **Autograd decision:** Two valid "from scratch" interpretations:
> - **torch.autograd** — `requires_grad=True` on weights, PyTorch computes grads,
>   you write the update rule. Cleaner, still legitimately from scratch.
> - **Manual backprop** — implement `backward()` yourself, `torch.no_grad()` everywhere.
>   Harder, more educational, closer to your uni work.
>
> Recommended: start with autograd, then swap one layer to manual backprop as an exercise.

### Saturday Afternoon — The Backend
Good Cursor snippet territory:
1. `backend/schemas.py` — `TrainingConfig` accepting preset or architecture list
2. `backend/routes.py` — `/api/options` first (just serializes registry), then `/api/start`
3. `training/data.py` — torchvision MNIST, `pin_memory=True`, `.to(device)` in loop
4. `training/logger.py` — metric store, `.item()` to detach scalars from GPU
5. `training/trainer.py` — loop, calls logger each batch
6. `backend/stream.py` — WebSocket that reads logger and pushes JSON
7. `backend/main.py` — mount everything, test all endpoints with curl / Postman

### Sunday — The Frontend
Let Cursor do the heavy lifting; focus on understanding the wiring:
1. Scaffold: `npm create vite@latest frontend -- --template react`
2. Install: `recharts tailwindcss`
3. `api/client.js` — fetch wrappers (Cursor snippet)
4. `hooks/useOptions.js` — calls `/api/options`, returns structured data (Cursor snippet)
5. `hooks/useTrainingStream.js` — WebSocket hook (Cursor snippet)
6. `SimpleConfig.jsx` — dropdowns + inputs, all driven by `useOptions` data
7. `LossCurve.jsx` + `AccuracyCurve.jsx` — Recharts live charts (Cursor snippet)
8. `MetricsBar.jsx` + `Controls.jsx` — wire to hooks + api/client
9. `AdvancedConfig.jsx` + `LayerCard.jsx` — dynamic fields rendered from Pydantic schemas

> 💡 **Before writing any frontend:** spend 2 hours on react.dev/learn.
> Focus only on: components, useState, useEffect. That's all this project needs.

---

## .cursorrules

```
This project is a neural network training dashboard.

- model/ contains a neural network built from scratch using raw PyTorch tensors only.
  Allowed: torch.Tensor ops, torch.matmul, torch.autograd, F.unfold.
  NOT allowed: nn.Module, nn.Linear, nn.Conv2d, nn.CrossEntropyLoss, or any nn.* API.
  All layers, weight inits, and optimizer steps must be implemented manually.

- Weights are plain tensors with requires_grad=True stored in lists — not nn.Parameters.
  Optimizer update rule: param.data -= lr * param.grad

- Every layer/loss/optimizer class must have:
    1. A config_model class attribute (a Pydantic BaseModel from model/configs.py)
    2. A @register_layer / @register_loss / @register_optimizer decorator
  Do NOT suggest adding any of these without both.

- All Pydantic config models live in model/configs.py.
  Use Field() with constraints (gt, ge, le) and @field_validator for custom rules.

- All tensors must be device-aware. Import device from training/device.py.
  Always include .to(device) in suggestions touching model/ or training/.

- Backend is FastAPI. All request bodies use Pydantic schemas from backend/schemas.py.
  GET /api/options serializes registry keys + Pydantic JSON schemas for the frontend.
  Frontend dropdowns and layer config fields are always driven by this endpoint — never hardcoded.

- Frontend is React + Vite + Recharts + Tailwind.
  Prefer small focused snippets over full rewrites.
  In model/, explain the math before writing code.
```

---

## Key Cursor Prompts (Hybrid Style)

| When you're stuck on... | Ask Cursor... |
|---|---|
| Conv2D forward pass | *"What are 2 ways to implement Conv2D forward using only torch.Tensor ops? Pros/cons."* |
| Manual backprop for ReLU | *"Show me just the backward pass for ReLU as a raw tensor snippet, no autograd."* |
| Weight initialisation | *"What init schemes work well for a from-scratch CNN? Show me the torch.Tensor ops."* |
| Pydantic constraint | *"How do I write a Pydantic validator that ensures kernel_size is always odd?"* |
| Registry safe lookup | *"Write get_layer(name) that raises a clear KeyError listing all available options."* |
| /api/options endpoint | *"Write a FastAPI GET endpoint returning all registry keys + their Pydantic JSON schemas."* |
| GPU data loading | *"Show me a torchvision MNIST DataLoader with pin_memory and .to(device) in the loop."* |
| WebSocket stream | *"Write a FastAPI WebSocket endpoint that pushes dicts from an asyncio Queue each batch."* |
| useOptions hook | *"Write a React hook that fetches /api/options on mount and returns layers/optimizers/losses."* |
| useTrainingStream hook | *"Write a React hook that opens a WebSocket to /ws/metrics and appends JSON to state."* |
| Dynamic layer fields | *"Given a Pydantic JSON schema object, render the right input type per field in React."* |
| Recharts live update | *"Give me a Recharts LineChart that appends new {batch, loss} points from a prop array."* |
| Gradient debugging | *"Here's my backward pass for CrossEntropy. Check the math against autograd, don't rewrite it."* |

---

## Extensions (if you finish early)
- **LR schedule** — step decay in the optimizer, visualised as a third live chart
- **Weight histograms** — log weight distributions per layer to catch vanishing/exploding gradients
- **Confusion matrix** — updates after each epoch in the dashboard
- **Export / import config** — download the current architecture as JSON, re-upload to restore it
- **Dataset picker** — add FashionMNIST or CIFAR-10 alongside MNIST
