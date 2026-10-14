"""Independent teacher workers that lease one public sample at a time."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path
import socket
import threading
import time
import traceback
from typing import Any

from .collector import collect_case
from .io import append_jsonl, write_json
from .schema import CollectorConfig, PublicCase
from .task_queue import SqliteTaskQueue, TaskLease


def _safe_filename(value: str) -> str:
    from .collector import _safe_case_dirname

    return _safe_case_dirname(value)


def _result_path(results_dir: Path, task_id: str) -> Path:
    return results_dir / "cases" / f"{_safe_filename(task_id)}.json"


def _run_worker(
    *,
    queue_path: str | Path,
    queue_name: str,
    results_dir: Path,
    collector_config: CollectorConfig,
    worker_id: str,
    lease_s: float,
    poll_s: float,
    idle_exit_s: float,
    max_tasks: int | None,
    debug: bool,
) -> dict[str, int]:
    queue = SqliteTaskQueue(queue_path)
    counts = {"leased": 0, "succeeded": 0, "retryable_error": 0, "failed": 0, "lease_lost": 0, "idle_exit": 0}
    idle_started = time.monotonic()
    while max_tasks is None or counts["leased"] < max_tasks:
        lease = queue.lease(queue_name, worker_id=worker_id, lease_s=lease_s)
        if lease is None:
            if idle_exit_s > 0 and time.monotonic() - idle_started >= idle_exit_s:
                counts["idle_exit"] += 1
                break
            time.sleep(poll_s)
            continue
        idle_started = time.monotonic()
        counts["leased"] += 1
        if debug:
            print(
                f"[fix2-teacher] worker={worker_id} leased case={lease.task_id} attempt={lease.attempt}",
                flush=True,
            )
        try:
            public_case = PublicCase.from_dict(lease.payload)
            record = collect_case(public_case, collector_config)
            output_path = _result_path(results_dir, lease.task_id)
            # The file is published before queue acknowledgement. If a worker
            # crashes after this point, a retry deterministically rewrites the
            # same case file and no shared JSONL corruption is possible.
            write_json(output_path, record)
            queue.complete(lease, result_path=str(output_path.resolve()))
            counts["succeeded"] += 1
            if debug:
                print(f"[fix2-teacher] worker={worker_id} succeeded case={lease.task_id}", flush=True)
        except BaseException as error:
            # A dependency may call ``sys.exit(1)`` from a worker thread.  In
            # a ThreadPoolExecutor that becomes a SystemExit re-raised by
            # ``future.result()`` in the parent, silently terminating the
            # entire 16-worker group.  Treat it as a task-local retryable
            # failure and retain the original stack in the durable queue/log.
            # User interrupts still have their normal process-wide behavior.
            if isinstance(error, (KeyboardInterrupt, GeneratorExit)):
                raise
            error_detail = traceback.format_exc()
            error_text = f"{error.__class__.__name__}: {error}\n{error_detail}"
            try:
                status = queue.fail(
                    lease,
                    error_text=error_text,
                    retryable=True,
                )
                counts["retryable_error" if status == "pending" else "failed"] += 1
                if debug:
                    print(
                        f"[fix2-teacher] worker={worker_id} case={lease.task_id} status={status} "
                        f"error={error.__class__.__name__}: {error}",
                        flush=True,
                    )
                    print(error_detail, flush=True)
            except RuntimeError:
                # A long-running task may lose its lease to another worker.
                # Its per-case output is still atomic; the current owner will
                # publish/ack the final outcome.
                counts["lease_lost"] += 1
                if debug:
                    print(f"[fix2-teacher] worker={worker_id} lost lease for case={lease.task_id}", flush=True)
    return counts


def run_teacher_workers(
    *,
    queue_path: str | Path,
    results_dir: str | Path,
    collector_config: CollectorConfig,
    workers: int,
    queue_name: str = "teacher_collect",
    lease_s: float = 3600.0,
    poll_s: float = 1.0,
    idle_exit_s: float = 30.0,
    max_tasks_per_worker: int | None = None,
    debug: bool = False,
) -> dict[str, int]:
    """Run a local worker group; each worker owns one full sample at a time."""

    if workers < 1:
        raise ValueError("workers must be positive")
    if poll_s <= 0:
        raise ValueError("poll_s must be positive")
    if lease_s <= 0:
        raise ValueError("lease_s must be positive")
    result_root = Path(results_dir)
    result_root.mkdir(parents=True, exist_ok=True)
    host = socket.gethostname()
    merged: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="fix2-teacher") as executor:
        futures = [
            executor.submit(
                _run_worker,
                queue_path=queue_path,
                queue_name=queue_name,
                results_dir=result_root,
                collector_config=collector_config,
                worker_id=f"{host}:{os.getpid()}:{index}",
                lease_s=lease_s,
                poll_s=poll_s,
                idle_exit_s=idle_exit_s,
                max_tasks=max_tasks_per_worker,
                debug=debug,
            )
            for index in range(workers)
        ]
        for future in as_completed(futures):
            for key, value in future.result().items():
                merged[key] = merged.get(key, 0) + value
    merged["workers"] = workers
    merged.update({f"queue_{key}": value for key, value in SqliteTaskQueue(queue_path).stats(queue_name).items()})
    return merged


def merge_teacher_results(
    *,
    queue_path: str | Path,
    output_jsonl: str | Path,
    queue_name: str = "teacher_collect",
) -> dict[str, int]:
    """Deterministically merge atomic per-case outputs after worker completion."""

    output = Path(output_jsonl)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite merged result JSONL: {output}")
    counts = {"succeeded_tasks": 0, "merged": 0, "missing_result": 0}
    for item in SqliteTaskQueue(queue_path).succeeded(queue_name):
        counts["succeeded_tasks"] += 1
        result_path = Path(item["result_path"])
        if not result_path.is_file():
            counts["missing_result"] += 1
            continue
        import json

        append_jsonl(output, json.loads(result_path.read_text(encoding="utf-8")))
        counts["merged"] += 1
    return counts


def merge_atomic_teacher_results(
    *,
    results_dir: str | Path,
    output_jsonl: str | Path,
) -> dict[str, int]:
    """Merge every valid immutable per-case result without consulting SQLite.

    The coordinator queue lives on local storage by design.  It can therefore
    be lost or rebuilt independently of the durable ``teacher_results`` tree.
    Final judging must use the latter as the source of truth: a collector only
    publishes an atomic case file after it has returned a complete record.

    The merged JSONL is written to a sibling temporary path and renamed only
    after the whole scan succeeds, so an interrupted merge is never mistaken
    for a complete judge input on resume.
    """

    output = Path(output_jsonl)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite merged result JSONL: {output}")
    result_root = Path(results_dir) / "cases"
    if not result_root.is_dir():
        raise FileNotFoundError(f"Atomic teacher result directory does not exist: {result_root}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    counts = {"scanned": 0, "merged": 0, "invalid": 0, "duplicate_case_id": 0}
    seen_case_ids: set[str] = set()
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for path in sorted(result_root.glob("*.json")):
                counts["scanned"] += 1
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    counts["invalid"] += 1
                    continue
                case_id = str(record.get("case_id") or "").strip() if isinstance(record, dict) else ""
                if not case_id:
                    counts["invalid"] += 1
                    continue
                if case_id in seen_case_ids:
                    counts["duplicate_case_id"] += 1
                    continue
                seen_case_ids.add(case_id)
                json.dump(record, handle, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
                counts["merged"] += 1
        if counts["invalid"] or counts["duplicate_case_id"]:
            raise ValueError(f"Refusing incomplete atomic merge: {counts}")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return counts


def reconcile_teacher_results(
    *,
    queue_path: str | Path,
    results_dir: str | Path,
    queue_name: str = "teacher_collect",
) -> dict[str, int]:
    """Mark queue tasks completed when their shared atomic result exists.

    This is the local-state-loss recovery path.  It deliberately trusts only a
    parseable JSON object whose ``case_id`` maps back to a task already queued
    from a public manifest.  It does not decide data quality; final judges do
    that later exactly as in a normal run.
    """

    result_root = Path(results_dir) / "cases"
    counts = {"scanned": 0, "valid": 0, "invalid": 0, "duplicate_case_id": 0}
    recovered_paths: dict[str, str] = {}
    if result_root.is_dir():
        for path in sorted(result_root.glob("*.json")):
            counts["scanned"] += 1
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                counts["invalid"] += 1
                continue
            case_id = str(value.get("case_id") or "").strip() if isinstance(value, dict) else ""
            if not case_id:
                counts["invalid"] += 1
                continue
            if case_id in recovered_paths:
                counts["duplicate_case_id"] += 1
                continue
            recovered_paths[case_id] = str(path.resolve())
            counts["valid"] += 1
    queue = SqliteTaskQueue(queue_path)
    counts.update(queue.recover_succeeded_results(queue_name, recovered_paths))
    counts.update({f"queue_{key}": value for key, value in queue.stats(queue_name).items()})
    return counts
