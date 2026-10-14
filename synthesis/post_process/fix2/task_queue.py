"""Durable, lease-based task queue for sample-level local worker groups.

The SQLite implementation is intentionally for one coordinator host.  It is
safe for many local worker threads/processes, but a SQLite DB placed on HDFS,
NFS, or another network filesystem is not a cross-machine queue.  A remote
deployment should keep this interface and replace the store with a Redis
Streams/central-service adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Mapping


_PRIVATE_KEYS = {
    "gold_answer",
    "answer",
    "answers",
    "acceptable_answers",
    "ground_truth_list",
    "hop_chain",
    "statement",
    "knowledge_graph",
    "reference_answer",
}


def _now() -> float:
    return time.time()


def _assert_public_payload(value: Any) -> None:
    """Reject supervision-bearing task payloads before they reach a teacher."""

    if isinstance(value, dict):
        exposed = _PRIVATE_KEYS.intersection(str(key).casefold() for key in value)
        if exposed:
            raise ValueError(f"Task payload contains private fields: {sorted(exposed)}")
        for nested in value.values():
            _assert_public_payload(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_public_payload(nested)


@dataclass(frozen=True)
class TaskLease:
    queue_name: str
    task_id: str
    payload: dict[str, Any]
    attempt: int
    lease_owner: str
    lease_until: float


class SqliteTaskQueue:
    """A transactional task store with explicit leases and idempotent enqueue."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS fix2_tasks (
                    queue_name TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 0,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 3,
                    lease_owner TEXT,
                    lease_until REAL,
                    result_path TEXT,
                    error_text TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (queue_name, task_id)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS fix2_tasks_ready ON fix2_tasks(queue_name, status, priority DESC, created_at)"
            )

    def enqueue(
        self,
        queue_name: str,
        task_id: str,
        payload: dict[str, Any],
        *,
        priority: int = 0,
        max_attempts: int = 3,
    ) -> bool:
        _assert_public_payload(payload)
        identifier = str(task_id).strip()
        if not identifier:
            raise ValueError("task_id is required")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        now = _now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO fix2_tasks
                (queue_name, task_id, payload_json, status, priority, max_attempts, created_at, updated_at)
                VALUES (?, ?, ?, 'pending', ?, ?, ?, ?)
                """,
                (queue_name, identifier, json.dumps(payload, ensure_ascii=False, sort_keys=True), priority, max_attempts, now, now),
            )
        return cursor.rowcount == 1

    def reclaim_expired(self, queue_name: str) -> int:
        now = _now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            retried = connection.execute(
                """
                UPDATE fix2_tasks
                SET status = 'pending', lease_owner = NULL, lease_until = NULL,
                    error_text = COALESCE(error_text, 'lease_expired'), updated_at = ?
                WHERE queue_name = ? AND status = 'leased' AND lease_until <= ?
                  AND attempt < max_attempts
                """,
                (now, queue_name, now),
            ).rowcount
            connection.execute(
                """
                UPDATE fix2_tasks
                SET status = 'failed', lease_owner = NULL, lease_until = NULL,
                    error_text = COALESCE(error_text, 'lease_expired_max_attempts'), updated_at = ?
                WHERE queue_name = ? AND status = 'leased' AND lease_until <= ?
                  AND attempt >= max_attempts
                """,
                (now, queue_name, now),
            )
            connection.execute("COMMIT")
        return int(retried)

    def requeue_leased(self, queue_name: str, *, reason: str = "worker_group_restarted") -> int:
        """Return every currently leased task to ``pending``.

        This is deliberately separate from :meth:`reclaim_expired`: callers
        must first prove that the worker group which owns the leases has
        stopped. It is intended for a controlled worker-group restart, so a
        freshly interrupted task can be retried immediately instead of waiting
        for its normal long lease to expire.
        """

        error = str(reason).strip()[:4000] or "worker_group_restarted"
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                """
                UPDATE fix2_tasks
                SET status = 'pending', lease_owner = NULL, lease_until = NULL,
                    error_text = ?, updated_at = ?
                WHERE queue_name = ? AND status = 'leased'
                """,
                (error, _now(), queue_name),
            ).rowcount
            connection.execute("COMMIT")
        return int(updated)

    def lease(self, queue_name: str, *, worker_id: str, lease_s: float) -> TaskLease | None:
        if lease_s <= 0:
            raise ValueError("lease_s must be positive")
        now = _now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE fix2_tasks
                SET status = 'pending', lease_owner = NULL, lease_until = NULL,
                    error_text = COALESCE(error_text, 'lease_expired'), updated_at = ?
                WHERE queue_name = ? AND status = 'leased' AND lease_until <= ?
                  AND attempt < max_attempts
                """,
                (now, queue_name, now),
            )
            row = connection.execute(
                """
                SELECT * FROM fix2_tasks
                WHERE queue_name = ? AND status = 'pending'
                ORDER BY priority DESC, created_at ASC, task_id ASC
                LIMIT 1
                """,
                (queue_name,),
            ).fetchone()
            if row is None:
                connection.execute("COMMIT")
                return None
            lease_until = now + lease_s
            updated = connection.execute(
                """
                UPDATE fix2_tasks
                SET status = 'leased', attempt = attempt + 1, lease_owner = ?, lease_until = ?,
                    error_text = NULL, updated_at = ?
                WHERE queue_name = ? AND task_id = ? AND status = 'pending'
                """,
                (worker_id, lease_until, now, queue_name, row["task_id"]),
            )
            if updated.rowcount != 1:  # pragma: no cover - protected by BEGIN IMMEDIATE
                connection.execute("ROLLBACK")
                return None
            connection.execute("COMMIT")
        return TaskLease(
            queue_name=queue_name,
            task_id=str(row["task_id"]),
            payload=json.loads(str(row["payload_json"])),
            attempt=int(row["attempt"]) + 1,
            lease_owner=worker_id,
            lease_until=lease_until,
        )

    def complete(self, lease: TaskLease, *, result_path: str) -> None:
        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE fix2_tasks
                SET status = 'succeeded', result_path = ?, lease_owner = NULL, lease_until = NULL, updated_at = ?
                WHERE queue_name = ? AND task_id = ? AND status = 'leased' AND lease_owner = ?
                """,
                (result_path, _now(), lease.queue_name, lease.task_id, lease.lease_owner),
            )
        if updated.rowcount != 1:
            raise RuntimeError(f"Cannot complete unowned/expired task {lease.task_id}")

    def fail(self, lease: TaskLease, *, error_text: str, retryable: bool = True) -> str:
        error = str(error_text).strip()[:4000]
        with self._connect() as connection:
            status = "pending" if retryable and lease.attempt < self._max_attempts(connection, lease) else "failed"
            updated = connection.execute(
                """
                UPDATE fix2_tasks
                SET status = ?, error_text = ?, lease_owner = NULL, lease_until = NULL, updated_at = ?
                WHERE queue_name = ? AND task_id = ? AND status = 'leased' AND lease_owner = ?
                """,
                (status, error, _now(), lease.queue_name, lease.task_id, lease.lease_owner),
            )
        if updated.rowcount != 1:
            raise RuntimeError(f"Cannot fail unowned/expired task {lease.task_id}")
        return status

    @staticmethod
    def _max_attempts(connection: sqlite3.Connection, lease: TaskLease) -> int:
        row = connection.execute(
            "SELECT max_attempts FROM fix2_tasks WHERE queue_name = ? AND task_id = ?",
            (lease.queue_name, lease.task_id),
        ).fetchone()
        return int(row["max_attempts"]) if row else 1

    def stats(self, queue_name: str) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM fix2_tasks WHERE queue_name = ? GROUP BY status",
                (queue_name,),
            ).fetchall()
        result = {str(row["status"]): int(row["count"]) for row in rows}
        result["total"] = sum(result.values())
        return result

    def succeeded(self, queue_name: str) -> list[dict[str, str]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT task_id, result_path FROM fix2_tasks
                WHERE queue_name = ? AND status = 'succeeded'
                ORDER BY task_id ASC
                """,
                (queue_name,),
            ).fetchall()
        return [{"task_id": str(row["task_id"]), "result_path": str(row["result_path"] or "")} for row in rows]

    def recover_succeeded_results(self, queue_name: str, result_paths: Mapping[str, str]) -> dict[str, int]:
        """Recover durable successes from immutable atomic result files.

        A local SQLite queue is intentionally not placed on HDFS.  If its host
        loses ``/tmp``, the shared ``teacher_results/cases/*.json`` files are
        the durable source of truth for completed collection work.  This method
        is used only while no worker group is running: it marks queued task IDs
        backed by a validated atomic result file as succeeded, without exposing
        private supervision or re-running a teacher call.
        """

        counts = {"recovered": 0, "already_succeeded": 0, "unknown_queue_task": 0}
        if not result_paths:
            return counts
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for raw_task_id, raw_result_path in result_paths.items():
                task_id = str(raw_task_id).strip()
                result_path = str(raw_result_path).strip()
                if not task_id or not result_path:
                    continue
                row = connection.execute(
                    "SELECT status FROM fix2_tasks WHERE queue_name = ? AND task_id = ?",
                    (queue_name, task_id),
                ).fetchone()
                if row is None:
                    counts["unknown_queue_task"] += 1
                    continue
                if str(row["status"]) == "succeeded":
                    counts["already_succeeded"] += 1
                    continue
                connection.execute(
                    """
                    UPDATE fix2_tasks
                    SET status = 'succeeded', result_path = ?, lease_owner = NULL,
                        lease_until = NULL, error_text = 'recovered_from_atomic_result', updated_at = ?
                    WHERE queue_name = ? AND task_id = ?
                    """,
                    (result_path, _now(), queue_name, task_id),
                )
                counts["recovered"] += 1
            connection.execute("COMMIT")
        return counts
