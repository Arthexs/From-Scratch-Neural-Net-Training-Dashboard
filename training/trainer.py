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

from model.losses import Loss
from model.network import Network
from model.optimizers import Optimizer
from training.configs import TrainerConfig
from training.data import BaseDataset
from training.device import device
from training.registry import METRICS


@METRICS.register("classification_accuracy")
def _classification_accuracy(out: torch.Tensor, y: torch.Tensor) -> float:
    pred = out.argmax(dim=-1)
    target = y.argmax(dim=-1) if y.dim() > 1 else y
    return float((pred == target).float().mean().item())


@METRICS.register("binary_accuracy")
def _binary_accuracy(out: torch.Tensor, y: torch.Tensor) -> float:
    pred = out.sigmoid() >= 0.5
    return float((pred == y.bool()).float().mean().item())


@METRICS.register("mae")
def _mae(out: torch.Tensor, y: torch.Tensor) -> float:
    return float((out - y).abs().mean().item())


@METRICS.register("rmse")
def _rmse(out: torch.Tensor, y: torch.Tensor) -> float:
    return float(((out - y) ** 2).mean().sqrt().item())


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
        dataset: BaseDataset,
        metrics: "queue.Queue[dict[str, Any]]",
        stop: threading.Event,
    ) -> None:
        self._network = network
        self._loss = loss
        self._optimizer = optimizer
        self._cfg = cfg
        self._dataset = dataset
        self._metrics = metrics
        self._stop = stop

    def run(self) -> None:
        """Entry point for the training thread. Runs all epochs then emits done."""
        train_loader, val_loader = self._dataset.get_loaders(self._cfg.batch_size)
        self._network.to(device)

        try:
            for epoch in range(self._cfg.epochs):
                if self._stop.is_set():
                    break
                self._train_epoch(epoch, train_loader)
                if self._cfg.log_validation and val_loader is not None and not self._stop.is_set():
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
        metric_totals: dict[str, float] = {m: 0.0 for m in self._cfg.metrics}
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

            if self._cfg.log_grad_norm:
                grads = [p.grad for p in params if p.grad is not None]
                if grads:
                    grad_norm = torch.stack([g.norm() for g in grads]).norm().item()
                    self._emit(
                        {
                            "type": "grad_norm",
                            "epoch": epoch,
                            "batch": batch_idx,
                            "value": grad_norm,
                        }
                    )

            self._optimizer.zero_grad(params)

            batch_loss = loss_val.item()
            total_loss += batch_loss
            for m in self._cfg.metrics:
                metric_totals[m] += METRICS.get(m)(out, y)
            n_batches += 1

            if self._cfg.log_per_batch_loss:
                self._emit(
                    {
                        "type": "batch_loss",
                        "epoch": epoch,
                        "batch": batch_idx,
                        "loss": batch_loss,
                    }
                )

        if n_batches > 0:
            if self._cfg.log_per_epoch_loss:
                self._emit(
                    {
                        "type": "epoch_loss",
                        "epoch": epoch,
                        "loss": total_loss / n_batches,
                    }
                )
            for m, total in metric_totals.items():
                self._emit(
                    {
                        "type": "metric",
                        "name": m,
                        "split": "train",
                        "epoch": epoch,
                        "value": total / n_batches,
                    }
                )

    def _validate_epoch(self, epoch: int, loader: Any) -> None:
        total_loss = 0.0
        metric_totals: dict[str, float] = {m: 0.0 for m in self._cfg.metrics}
        n_batches = 0

        for x, y in loader:
            if self._stop.is_set():
                break

            x = x.to(device)
            y = y.to(device)

            out = self._network.forward(x)
            loss_val = self._loss.forward(out, y)

            total_loss += loss_val.item()
            for m in self._cfg.metrics:
                metric_totals[m] += METRICS.get(m)(out, y)
            n_batches += 1

        if n_batches > 0:
            self._emit(
                {
                    "type": "val_loss",
                    "epoch": epoch,
                    "loss": total_loss / n_batches,
                }
            )
            for m, total in metric_totals.items():
                self._emit(
                    {
                        "type": "metric",
                        "name": m,
                        "split": "val",
                        "epoch": epoch,
                        "value": total / n_batches,
                    }
                )
