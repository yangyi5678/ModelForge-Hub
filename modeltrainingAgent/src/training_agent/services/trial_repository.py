from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from training_agent.models import ExecutionUpdate, TrialRecord


class SQLiteTrialRepository:
    def __init__(self, uri: str | Path) -> None:
        self.path = Path(uri)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trials (
                    task_id TEXT NOT NULL,
                    trial_number INTEGER NOT NULL,
                    request_id TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    runtime_params_json TEXT NOT NULL,
                    params_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reason TEXT,
                    PRIMARY KEY (task_id, trial_number),
                    UNIQUE (task_id, request_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS attempts (
                    task_id TEXT NOT NULL,
                    trial_number INTEGER NOT NULL,
                    attempt INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    checkpoint_uri TEXT,
                    best_checkpoint_uri TEXT,
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    current_epoch INTEGER NOT NULL DEFAULT 0,
                    duration_seconds REAL NOT NULL DEFAULT 0,
                    gpu_hours REAL NOT NULL DEFAULT 0,
                    error_type TEXT,
                    error_message TEXT,
                    PRIMARY KEY (task_id, trial_number, attempt)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_state (
                    task_id TEXT PRIMARY KEY,
                    config_path TEXT,
                    final_report_uri TEXT
                )
                """
            )

    def register_task(self, task_id: str, config_path: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO task_state(task_id, config_path) VALUES (?, ?)",
                (task_id, config_path),
            )

    def get_task_config_path(self, task_id: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT config_path FROM task_state WHERE task_id=?", (task_id,)
            ).fetchone()
        return None if row is None else str(row["config_path"])

    def set_final_report(self, task_id: str, final_report_uri: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO task_state(task_id, final_report_uri) VALUES (?, ?) "
                "ON CONFLICT(task_id) DO UPDATE SET final_report_uri=excluded.final_report_uri",
                (task_id, final_report_uri),
            )

    def create_or_get_trial(self, record: TrialRecord) -> TrialRecord:
        params_json = json.dumps(record.params, sort_keys=True)
        runtime_json = json.dumps(record.runtime_params, sort_keys=True)
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM trials WHERE task_id=? AND trial_number=?",
                (record.task_id, record.trial_number),
            ).fetchone()
            if existing:
                if existing["params_hash"] != record.params_hash:
                    raise ValueError("attempted to overwrite immutable trial params")
                return self._row_to_trial(existing)
            conn.execute(
                """
                INSERT INTO trials(
                    task_id, trial_number, request_id, params_json,
                    runtime_params_json, params_hash, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.task_id,
                    record.trial_number,
                    record.request_id,
                    params_json,
                    runtime_json,
                    record.params_hash,
                    record.status,
                ),
            )
        return record

    def trial_by_request(self, task_id: str, request_id: str) -> TrialRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM trials WHERE task_id=? AND request_id=?", (task_id, request_id)
            ).fetchone()
        return None if row is None else self._row_to_trial(row)

    def start_attempt(self, task_id: str, trial_number: int, attempt: int) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO attempts(task_id, trial_number, attempt, status)
                VALUES (?, ?, ?, ?)
                """,
                (task_id, trial_number, attempt, "running"),
            )
            conn.execute(
                "UPDATE trials SET status=? WHERE task_id=? AND trial_number=?",
                ("running", task_id, trial_number),
            )

    def update_execution(self, update: ExecutionUpdate) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO attempts(
                    task_id, trial_number, attempt, status, checkpoint_uri, best_checkpoint_uri,
                    metrics_json, current_epoch, duration_seconds, gpu_hours,
                    error_type, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id, trial_number, attempt) DO UPDATE SET
                    status=excluded.status,
                    checkpoint_uri=excluded.checkpoint_uri,
                    best_checkpoint_uri=excluded.best_checkpoint_uri,
                    metrics_json=excluded.metrics_json,
                    current_epoch=excluded.current_epoch,
                    duration_seconds=excluded.duration_seconds,
                    gpu_hours=excluded.gpu_hours,
                    error_type=excluded.error_type,
                    error_message=excluded.error_message
                """,
                (
                    update.task_id,
                    update.trial_number,
                    update.attempt,
                    update.status,
                    update.checkpoint_uri,
                    update.best_checkpoint_uri,
                    json.dumps(update.metrics, sort_keys=True),
                    update.current_epoch,
                    update.duration_seconds,
                    update.gpu_hours,
                    update.error_type,
                    update.error_message,
                ),
            )

    def save_metrics(self, task_id: str, trial_number: int, metrics: dict[str, float]) -> None:
        latest = self.latest_attempt(task_id, trial_number)
        attempt = 1 if latest is None else int(latest["attempt"])
        self.update_execution(
            ExecutionUpdate(
                task_id=task_id,
                trial_number=trial_number,
                attempt=attempt,
                status="completed",
                metrics=metrics,
                current_epoch=int(latest["current_epoch"]) if latest else 0,
            )
        )

    def mark_completed(self, task_id: str, trial_number: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE trials SET status=? WHERE task_id=? AND trial_number=?",
                ("completed", task_id, trial_number),
            )

    def mark_failed(self, task_id: str, trial_number: int, reason: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE trials SET status=?, reason=? WHERE task_id=? AND trial_number=?",
                ("failed", reason, task_id, trial_number),
            )

    def list_trials(self, task_id: str) -> list[dict[str, object]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM trials WHERE task_id=? ORDER BY trial_number", (task_id,)
            ).fetchall()
        return [
            {
                "task_id": row["task_id"],
                "trial_number": row["trial_number"],
                "request_id": row["request_id"],
                "params": json.loads(row["params_json"]),
                "runtime_params": json.loads(row["runtime_params_json"]),
                "params_hash": row["params_hash"],
                "status": row["status"],
                "reason": row["reason"],
            }
            for row in rows
        ]

    def latest_attempt(self, task_id: str, trial_number: int) -> sqlite3.Row | None:
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT * FROM attempts
                WHERE task_id=? AND trial_number=?
                ORDER BY attempt DESC LIMIT 1
                """,
                (task_id, trial_number),
            ).fetchone()

    def attempt_count(self, task_id: str, trial_number: int) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS count FROM attempts WHERE task_id=? AND trial_number=?",
                (task_id, trial_number),
            ).fetchone()
        return int(row["count"])

    def _row_to_trial(self, row: sqlite3.Row) -> TrialRecord:
        return TrialRecord(
            task_id=row["task_id"],
            trial_number=int(row["trial_number"]),
            request_id=row["request_id"],
            params=json.loads(row["params_json"]),
            runtime_params=json.loads(row["runtime_params_json"]),
            params_hash=row["params_hash"],
            status=row["status"],
        )
