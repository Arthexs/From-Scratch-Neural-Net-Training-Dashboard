"""Tests for training.logger — LoggerConfig and Logger."""

import queue
import sqlite3
import threading

import pytest
from pydantic import ValidationError

from training.configs import LoggerConfig
from training.logger import Logger

# ── helpers ───────────────────────────────────────────────────────────────────


def _run(cfg: LoggerConfig, payloads: list[dict]) -> tuple[Logger, list[sqlite3.Row]]:
    """Start a logger, push payloads + done sentinel, join, return (logger, rows)."""
    q: queue.Queue = queue.Queue()
    stop = threading.Event()
    logger = Logger(q, cfg, stop)
    logger.start()
    for p in payloads:
        q.put(p)
    q.put({"type": "done"})
    logger.join()

    con = sqlite3.connect(cfg.db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT * FROM runs").fetchall()
    con.close()
    return logger, rows


@pytest.fixture
def cfg(tmp_path):
    return LoggerConfig(db_path=str(tmp_path / "runs.db"))


# ── LoggerConfig ──────────────────────────────────────────────────────────────


def test_logger_config_defaults():
    cfg = LoggerConfig()
    assert cfg.db_path == "runs/runs.db"
    assert cfg.queue_timeout == pytest.approx(0.1)


@pytest.mark.parametrize("queue_timeout", [0.0, -0.1, -10.0])
def test_logger_config_invalid_queue_timeout(queue_timeout):
    with pytest.raises(ValidationError):
        LoggerConfig(queue_timeout=queue_timeout)


def test_logger_config_extra_forbidden():
    with pytest.raises(ValidationError):
        LoggerConfig(unknown=True)


# ── Logger.run_id ─────────────────────────────────────────────────────────────


def test_run_id_is_32_char_hex(cfg):
    logger = Logger(queue.Queue(), cfg, threading.Event())
    assert len(logger.run_id) == 32
    assert all(c in "0123456789abcdef" for c in logger.run_id)


def test_two_loggers_have_different_run_ids(cfg):
    stop = threading.Event()
    a = Logger(queue.Queue(), cfg, stop)
    b = Logger(queue.Queue(), cfg, stop)
    assert a.run_id != b.run_id


# ── DB bootstrap ──────────────────────────────────────────────────────────────


def test_db_directory_created(tmp_path):
    deep_path = tmp_path / "a" / "b" / "c" / "runs.db"
    cfg = LoggerConfig(db_path=str(deep_path))
    _run(cfg, [])
    assert deep_path.exists()


def test_runs_table_exists(cfg):
    _run(cfg, [])
    con = sqlite3.connect(cfg.db_path)
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    con.close()
    assert "runs" in tables


# ── row mapping ───────────────────────────────────────────────────────────────


def test_batch_loss_row(cfg):
    _, rows = _run(cfg, [{"type": "batch_loss", "epoch": 2, "batch": 5, "loss": 0.42}])
    row = next(r for r in rows if r["type"] == "batch_loss")
    assert row["epoch"] == 2
    assert row["batch"] == 5
    assert row["metric"] is None
    assert row["value"] == pytest.approx(0.42)


def test_epoch_loss_row(cfg):
    _, rows = _run(cfg, [{"type": "epoch_loss", "epoch": 1, "loss": 0.3}])
    row = next(r for r in rows if r["type"] == "epoch_loss")
    assert row["epoch"] == 1
    assert row["batch"] is None
    assert row["metric"] is None
    assert row["value"] == pytest.approx(0.3)


def test_val_loss_row(cfg):
    _, rows = _run(cfg, [{"type": "val_loss", "epoch": 3, "loss": 0.25}])
    row = next(r for r in rows if r["type"] == "val_loss")
    assert row["epoch"] == 3
    assert row["batch"] is None
    assert row["metric"] is None
    assert row["value"] == pytest.approx(0.25)


def test_grad_norm_row(cfg):
    _, rows = _run(cfg, [{"type": "grad_norm", "epoch": 0, "batch": 10, "value": 1.23}])
    row = next(r for r in rows if r["type"] == "grad_norm")
    assert row["epoch"] == 0
    assert row["batch"] == 10
    assert row["metric"] is None
    assert row["value"] == pytest.approx(1.23)


@pytest.mark.parametrize("split,name", [("train", "classification_accuracy"), ("val", "mae")])
def test_metric_row(cfg, split, name):
    _, rows = _run(
        cfg, [{"type": "metric", "epoch": 4, "split": split, "name": name, "value": 0.9}]
    )
    row = next(r for r in rows if r["type"] == "metric")
    assert row["epoch"] == 4
    assert row["batch"] is None
    assert row["metric"] == f"{split}/{name}"
    assert row["value"] == pytest.approx(0.9)


def test_resource_not_written_to_db(cfg):
    _, rows = _run(
        cfg,
        [
            {
                "type": "resource",
                "cpu_percent": 50.0,
                "gpu_util": None,
                "gpu_mem_used": None,
                "gpu_mem_total": None,
            },
        ],
    )
    assert not any(r["type"] == "resource" for r in rows)


def test_done_row_written_with_null_value(cfg):
    _, rows = _run(cfg, [])  # _run always appends the done sentinel
    row = next(r for r in rows if r["type"] == "done")
    assert row["value"] is None
    assert row["metric"] is None


def test_checkpoint_row(cfg):
    _, rows = _run(cfg, [{"type": "checkpoint", "epoch": 5, "path": "runs/abc123_checkpoint.pt"}])
    row = next(r for r in rows if r["type"] == "checkpoint")
    assert row["epoch"] == 5
    assert row["path"] == "runs/abc123_checkpoint.pt"
    assert row["value"] is None
    assert row["metric"] is None


# ── metadata ──────────────────────────────────────────────────────────────────


def test_run_id_stamped_on_all_rows(cfg):
    logger, rows = _run(
        cfg,
        [
            {"type": "epoch_loss", "epoch": 0, "loss": 0.5},
            {"type": "epoch_loss", "epoch": 1, "loss": 0.4},
        ],
    )
    assert all(r["run_id"] == logger.run_id for r in rows)


def test_ts_is_positive_float(cfg):
    _, rows = _run(cfg, [{"type": "epoch_loss", "epoch": 0, "loss": 0.5}])
    row = next(r for r in rows if r["type"] == "epoch_loss")
    assert isinstance(row["ts"], float)
    assert row["ts"] > 0


# ── shutdown ──────────────────────────────────────────────────────────────────


def test_done_terminates_thread(cfg):
    logger, _ = _run(cfg, [])  # join() in _run means the thread is already dead here
    assert not logger._thread.is_alive()


def test_stop_event_drains_queue(cfg) -> None:
    q: queue.Queue = queue.Queue()
    stop = threading.Event()
    logger = Logger(q, cfg, stop)
    logger.start()

    # Pre-fill the queue before signalling stop so drain must write them
    q.put({"type": "epoch_loss", "epoch": 0, "loss": 0.5})
    q.put({"type": "epoch_loss", "epoch": 1, "loss": 0.4})
    stop.set()
    logger.join()

    con = sqlite3.connect(cfg.db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT * FROM runs WHERE type = 'epoch_loss'").fetchall()
    con.close()

    assert len(rows) == 2


def test_two_runs_append_to_same_db(tmp_path):
    cfg = LoggerConfig(db_path=str(tmp_path / "runs.db"))

    _run(cfg, [{"type": "epoch_loss", "epoch": 0, "loss": 0.5}])
    _run(cfg, [{"type": "epoch_loss", "epoch": 0, "loss": 0.3}])

    con = sqlite3.connect(cfg.db_path)
    count = con.execute("SELECT COUNT(*) FROM runs WHERE type = 'epoch_loss'").fetchone()[0]
    con.close()
    assert count == 2
