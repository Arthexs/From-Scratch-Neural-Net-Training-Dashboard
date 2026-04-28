# Session Notes — NN Dashboard

## Current project state

| File | Status |
| ---- | ------ |
| `model/` (all files) | ✅ Complete |
| `model/registry.py` | ✅ Complete — `FnRegistry` added alongside `Registry`; `INITIALIZERS = FnRegistry("Initializer")` |
| `model/configs.py` | ✅ Complete — `InitializableConfig` base with `field_validator` for initializer; `DenseConfig` / `Conv2DConfig` inherit from it |
| `model/layers.py` | ✅ Complete — `_INITIALIZERS` dict replaced with `@INITIALIZERS.register` decorated functions; mypy fixed (call in-place, return `t`) |
| `training/device.py` | ✅ Complete |
| `training/data.py` | ✅ Complete — `**kwargs: Any` fix for mypy |
| `training/analysis.py` | ✅ Complete — compute_mean_std, class_distribution |
| `training/registry.py` | ✅ Complete — `DATASETS` + `METRICS = FnRegistry("Metric")` |
| `training/configs.py` | ✅ Complete — `TrainerConfig` with `metrics: list[str]`, `log_grad_norm: bool`; validator checks against `METRICS.keys()`; `LoggerConfig` not yet added |
| `training/trainer.py` | ✅ Complete — metric functions registered (`classification_accuracy`, `binary_accuracy`, `mae`, `rmse`); accumulated per-batch in train/val epoch loops; grad norm emitted per-batch when `log_grad_norm=True` |
| `training/__init__.py` | ✅ Updated — side-effect `import training.trainer` to ensure METRICS populated before any `TrainerConfig` is instantiated |
| `training/logger.py` | ❌ Empty stub — implement next |
| `training/__main__.py` | ❌ Not started |
| `dashboard/app.py` | ❌ Not started |
| `streamlit_app.py` | ❌ Not started (future — cloud inference) |
| Docker | ❌ Deferred — finish working system first |
| GitHub Actions | ❌ Not started |
| README | ❌ Not started |

---

## Architecture decisions made this session

- **FnRegistry** — lightweight registry for named callables (no config required). `INITIALIZERS` lives in `model/registry.py`; `METRICS` lives in `training/registry.py`. Adding a new initializer or metric = one entry in one place.
- **No FastAPI / TensorBoard / PowerBI** — removed from scope. Streamlit polls SQLite directly for the local dashboard; no streaming backend needed.
- **Docker deferred** — implement everything working locally first, then containerise.
- **Metrics** — per-batch accumulation averaged at epoch end; emitted as `{"type": "metric", "name": ..., "split": "train"/"val", "epoch": ..., "value": ...}`. RMSE per-batch averaging is a standard approximation.
- **Grad norm** — global L2 norm computed after `backward()`, before `zero_grad()`. Guarded by `log_grad_norm: bool = False` in `TrainerConfig`.
- **Initializer mypy fix** — call `torch.nn.init.*_(t)` for side effect, return `t` explicitly. Avoids `Any` return type from PyTorch stubs.
- **Metric mypy fix** — wrap `.item()` in `float()`. `float(Any)` → `float` via constructor signature.

---

## Where to start next session: Logger

The Logger is a dedicated consumer thread that reads from the shared `queue.Queue` (written to by `Trainer` and `monitor_loop`) and persists every payload to SQLite.

### Step 1 — add `LoggerConfig` to `training/configs.py`

- `db_path: str = "runs/runs.db"` — path to SQLite file (directory created on start if missing)
- `queue_timeout: float = 0.1` — seconds to block on `queue.get()` before looping

### Step 2 — implement `training/logger.py`

- `Logger(q: queue.Queue, cfg: LoggerConfig, stop: threading.Event)`
- Generates `run_id = uuid4().hex` at instantiation; stamps every row
- `.start()` spawns a daemon consumer thread; `.join()` waits for it to finish
- Thread loop: `q.get(timeout=cfg.queue_timeout)` → write row; exits cleanly on `{"type": "done"}` or stop event, draining the queue first
- SQLite schema: `(id INTEGER PRIMARY KEY, run_id TEXT, epoch INTEGER, batch INTEGER, type TEXT, metric TEXT, value REAL, ts REAL)`
  - `metric` is NULL for non-metric rows (loss, grad_norm, resource); populated for `type="metric"` rows
  - `value` holds the scalar for all numeric payloads; non-numeric types (e.g. `"done"`) write NULL
- Creates the `runs/` directory and the DB/table on first use

### Step 3 — wire Logger in `training/__main__.py`

```python
# rough sketch
stop = threading.Event()
q: queue.Queue = queue.Queue()
logger = Logger(q, LoggerConfig(), stop)
trainer = Trainer(network, loss, optimizer, trainer_cfg, dataset, q, stop)
monitor_t = threading.Thread(target=monitor_loop, args=(q, stop, trainer_cfg.monitor_interval_s), daemon=True)

logger.start()
monitor_t.start()
trainer.run()          # blocks until done or stop
logger.join()
```

---

## Action plan (remaining work)

### 1. Logger (next)

- `LoggerConfig` in `training/configs.py`
- `Logger` consumer thread in `training/logger.py`
- `training/__main__.py` wiring script (`python -m training`)

### 2. Local dashboard (`dashboard/app.py`)

- Streamlit app; polls SQLite on a configurable interval
- Run selector (by `run_id`), loss curves (batch + epoch + val), gradient norm chart, resource stats, metrics table
- Read-only — no writes to SQLite

### 3. Cloud inference app (`streamlit_app.py`)

- Drawable canvas → preprocess → `Network.forward()` → top-3 predictions
- Loads `.pt` weights committed to repo
- Deploys to Streamlit Cloud

### 4. Docker (deferred)

- `Dockerfile.training` + `Dockerfile.dashboard`
- `docker-compose.yml` — two services, shared `runs/` volume, NVIDIA GPU runtime for training
- Requires WSL2 + NVIDIA WSL2 driver on Windows 10

### 5. GitHub Actions + README (last)