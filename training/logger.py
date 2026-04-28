"""
Logger: consumes metric dicts from a shared queue and persists them to SQLite.

Resource payloads (type="resource") are skipped — they are only useful for
live display and do not need to be stored for post-hoc analysis.
"""

import queue
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from training.configs import LoggerConfig

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS runs (
    id        INTEGER PRIMARY KEY,
    run_id    TEXT,
    epoch     INTEGER,
    batch     INTEGER,
    type      TEXT,
    metric    TEXT,
    value     REAL,
    path      TEXT,
    ts        REAL
)
"""


class Logger:
    def __init__(
        self,
        q: "queue.Queue[dict[str, Any]]",
        cfg: LoggerConfig,
        stop: threading.Event,
    ) -> None:
        self._q = q
        self._cfg = cfg
        self._stop = stop
        self.run_id: str = uuid4().hex
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def join(self) -> None:
        if self._thread is not None:
            self._thread.join()

    # ── thread entry point ────────────────────────────────────────────────────

    def _loop(self) -> None:
        db_path = Path(self._cfg.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        con = sqlite3.connect(db_path)
        try:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute(_CREATE_TABLE)
            con.commit()

            while True:
                if self._stop.is_set():
                    self._drain(con)
                    break
                try:
                    payload = self._q.get(timeout=self._cfg.queue_timeout)
                    self._write_row(con, payload)
                    if payload.get("type") == "done":
                        self._drain(con)
                        break
                except queue.Empty:
                    pass
        finally:
            con.close()

    def _drain(self, con: sqlite3.Connection) -> None:
        """Non-blockingly flush all remaining queue items."""
        while True:
            try:
                self._write_row(con, self._q.get_nowait())
            except queue.Empty:
                break

    # ── SQLite helpers ────────────────────────────────────────────────────────

    def _write_row(self, con: sqlite3.Connection, payload: dict[str, Any]) -> None:
        p_type = payload.get("type")

        if p_type == "resource":
            return

        epoch = payload.get("epoch")
        batch = payload.get("batch")
        path: str | None = None

        if p_type == "checkpoint":
            path = payload.get("path")
            metric: str | None = None
            value: float | None = None
        elif p_type == "metric":
            metric = f"{payload['split']}/{payload['name']}"
            value = payload.get("value")
        elif p_type in ("batch_loss", "epoch_loss", "val_loss"):
            metric = None
            value = payload.get("loss")
        elif p_type == "grad_norm":
            metric = None
            value = payload.get("value")
        else:  # "done" and any unrecognised types
            metric = None
            value = None

        con.execute(
            "INSERT INTO runs (run_id, epoch, batch, type, metric, value, path, ts)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (self.run_id, epoch, batch, p_type, metric, value, path, time.time()),
        )
        con.commit()
