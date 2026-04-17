"""
Training loop and resource monitor.

Trainer.run() is the entry point for the training thread.
monitor_loop() is the entry point for the monitor thread.
Both write metric dicts to a shared queue.Queue and honour a shared threading.Event for stopping.
"""

import queue
import threading
import time
from typing import Any

import psutil
import torch

from model.configs import DataConfig, TrainerConfig
from model.losses import Loss
from model.network import Network
from model.optimizers import Optimizer
from training.data import get_dataloaders
from training.device import device


def monitor_loop(
    metrics: "queue.Queue[dict[str, Any]]",
    stop: threading.Event,
    interval_s: float,
) -> None:
    """Sample CPU + GPU utilisation until stop is set.

    Pushes dicts with type="resource" to the metrics queue.

    GPU stat priority:
        1. pynvml  — util % + OS-level memory used/total (most accurate)
        2. torch.cuda — PyTorch-allocated memory only, util unavailable
        3. None    — no GPU or all imports failed
    """
    nvml_ok = False
    cuda_fallback = False
    handle = None

    if torch.cuda.is_available():
        try:
            import pynvml  # type: ignore[import]
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            nvml_ok = True
        except Exception:
            cuda_fallback = True  # pynvml unavailable but CUDA is — use torch.cuda

    try:
        while not stop.is_set():
            payload: dict[str, Any] = {
                "type": "resource",
                "cpu_percent": psutil.cpu_percent(),
                "gpu_util": None,
                "gpu_mem_used": None,
                "gpu_mem_total": None,
            }

            if nvml_ok and handle is not None:
                import pynvml  # type: ignore[import]
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                payload["gpu_util"] = util.gpu
                payload["gpu_mem_used"] = mem.used
                payload["gpu_mem_total"] = mem.total
            elif cuda_fallback:
                # util % not available via torch.cuda; memory reflects PyTorch allocations only
                payload["gpu_mem_used"] = torch.cuda.memory_allocated()
                payload["gpu_mem_total"] = torch.cuda.get_device_properties(0).total_memory

            try:
                metrics.put_nowait(payload)
            except queue.Full:
                pass

            time.sleep(interval_s)
    finally:
        if nvml_ok:
            import pynvml  # type: ignore[import]
            pynvml.nvmlShutdown()


class Trainer:
    def __init__(
        self,
        network: Network,
        loss: Loss,
        optimizer: Optimizer,
        cfg: TrainerConfig,
        data_cfg: DataConfig,
        metrics: "queue.Queue[dict[str, Any]]",
        stop: threading.Event,
    ) -> None:
        self._network = network
        self._loss = loss
        self._optimizer = optimizer
        self._cfg = cfg
        self._data_cfg = data_cfg
        self._metrics = metrics
        self._stop = stop

    def run(self) -> None:
        """Entry point for the training thread. Runs all epochs then emits done."""
        train_loader, val_loader = get_dataloaders(self._data_cfg, self._cfg.batch_size)
        self._network.to(device)

        try:
            for epoch in range(self._cfg.epochs):
                if self._stop.is_set():
                    break
                self._train_epoch(epoch, train_loader)
                if self._cfg.log_validation and not self._stop.is_set():
                    self._validate_epoch(epoch, val_loader)
        finally:
            self._emit({"type": "done"})

    # ── private helpers ────────────────────────────────────────────────────────

    def _emit(self, payload: dict[str, Any]) -> None:
        """Non-blocking put; silently drops if queue is full."""
        try:
            self._metrics.put_nowait(payload)
        except queue.Full:
            pass

    def _train_epoch(self, epoch: int, loader: Any) -> None:
        total_loss = 0.0
        n_batches = 0

        for batch_idx, (x, y) in enumerate(loader):
            if self._stop.is_set():
                break

            x = x.to(device)
            y = y.to(device)

            out = self._network.forward(x)
            loss_val = self._loss.forward(out, y)

            loss_grad = self._loss.backward(torch.tensor(1.0, device=device))
            self._network.backward(loss_grad)

            params = self._network.parameters()
            self._optimizer.step(params)
            self._optimizer.zero_grad(params)

            batch_loss = loss_val.item()
            total_loss += batch_loss
            n_batches += 1

            if self._cfg.log_per_batch_loss:
                self._emit({
                    "type": "batch_loss",
                    "epoch": epoch,
                    "batch": batch_idx,
                    "loss": batch_loss,
                })

        if self._cfg.log_per_epoch_loss and n_batches > 0:
            self._emit({
                "type": "epoch_loss",
                "epoch": epoch,
                "loss": total_loss / n_batches,
            })

    def _validate_epoch(self, epoch: int, loader: Any) -> None:
        total_loss = 0.0
        n_batches = 0

        for x, y in loader:
            if self._stop.is_set():
                break

            x = x.to(device)
            y = y.to(device)

            out = self._network.forward(x)
            loss_val = self._loss.forward(out, y)

            total_loss += loss_val.item()
            n_batches += 1

        if n_batches > 0:
            self._emit({
                "type": "val_loss",
                "epoch": epoch,
                "loss": total_loss / n_batches,
            })
