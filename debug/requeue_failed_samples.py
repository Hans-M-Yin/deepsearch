"""Requeue selected failed graph-run tasks by editing ``graph_runner_state.json``.

Examples:
  python debug/requeue_failed_samples.py     --graph-dir synthesis/runs/0716_my_graph

  python debug/requeue_failed_samples.py     --vqa-dir synthesis/runs/0716_my_graph/vqa/0716_123456

  python debug/requeue_failed_samples.py     --graph-dir synthesis/runs/0716_my_graph     --all-failed     --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_STATE_FILE_NAME = "graph_runner_state.json"
DEFAULT_MATCH_ERROR_PATTERNS = (
    "HTTP Error 502",
    "HTTP Error 503",
    "HTTP Error 504",
)
DEFAULT_SHOW_LIMIT = 20


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Requeue selected failed tasks into graph_runner_state.json without touching "
            "existing nodes/edges/assets. The input path may be a graph dir or a nested "
            "VQA output dir; the script searches upward for graph_runner_state.json."
        )
    )
    parser.add_argument(
        "--graph-dir",
        "--vqa-dir",
        dest="input_dir",
        required=True,
        help="Graph run dir or nested VQA dir. The script searches upward for graph_runner_state.json.",
    )
    parser.add_argument(
        "--state-file-name",
        default=DEFAULT_STATE_FILE_NAME,
        help="State file name to edit inside the resolved graph dir.",
    )
    parser.add_argument(
        "--all-failed",
        action="store_true",
        help="Requeue every failed task instead of only matching transient HTTP 5xx reader errors.",
    )
    parser.add_argument(
        "--match-error",
        action="append",
        default=None,
        help=(
            "Case-insensitive substring match over failed task error text. Can be repeated. "
            "Ignored when --all-failed is set. Defaults to HTTP 502/503/504 patterns."
        ),
    )
    parser.add_argument(
        "--keep-failed-records",
        action="store_true",
        help="Keep the selected records inside failed_tasks history instead of removing them.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not write a timestamped backup of graph_runner_state.json before updating it.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the state rewrite without writing any files.",
    )
    parser.add_argument(
        "--show-limit",
        type=int,
        default=DEFAULT_SHOW_LIMIT,
        help="How many example tasks to print per category. <=0 means print none.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit one JSON object instead of human-readable text.",
    )
    return parser.parse_args(argv)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short(value: Any, limit: int = 160) -> str:
    text = " ".join(str(value or "").split())
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_state_path(input_path: Path, *, state_file_name: str) -> Path:
    candidate = input_path.resolve()
    if candidate.is_file():
        if candidate.name == state_file_name:
            return candidate
        candidate = candidate.parent

    for directory in (candidate, *candidate.parents):
        state_path = directory / state_file_name
        if state_path.exists():
            return state_path
    raise FileNotFoundError(
        f"Could not find {state_file_name!r} at or above input path: {input_path}"
    )


def _task_key(task: dict[str, Any]) -> str:
    task_type = str(task.get("task_type") or "text_expand")
    metadata = task.get("metadata") or {}
    if task_type == "image_expand":
        source_node_id = metadata.get("source_text_node_id") if isinstance(metadata, dict) else None
        if source_node_id:
            return f"{task_type}:{source_node_id}"
    return f"{task_type}:{task.get('url')}"


def _error_text(record: dict[str, Any]) -> str:
    parts = []
    for key in ("error", "attribute_error"):
        value = record.get(key)
        if value:
            parts.append(str(value))
    return " | ".join(parts)


def _matches_failed_record(
    record: dict[str, Any],
    *,
    match_patterns: list[str],
    all_failed: bool,
) -> bool:
    task = record.get("task")
    if not isinstance(task, dict):
        return False
    if all_failed:
        return True
    haystack = _error_text(record).lower()
    return any(pattern.lower() in haystack for pattern in match_patterns)


def _task_summary(record: dict[str, Any], *, dedupe_key: str | None = None) -> dict[str, Any]:
    task = record.get("task") if isinstance(record.get("task"), dict) else {}
    return {
        "dedupe_key": dedupe_key,
        "url": task.get("url"),
        "title": task.get("title"),
        "depth": task.get("depth"),
        "task_type": task.get("task_type"),
        "status": task.get("status"),
        "error": _error_text(record),
    }


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def _emit_human(summary: dict[str, Any], *, show_limit: int) -> None:
    print(f"resolved_input_dir: {summary['resolved_input_dir']}")
    print(f"graph_dir: {summary['graph_dir']}")
    print(f"state_path: {summary['state_path']}")
    if summary.get("backup_path"):
        print(f"backup_path: {summary['backup_path']}")
    print(f"dry_run: {summary['dry_run']}")
    print(f"selection_mode: {'all_failed' if summary['all_failed'] else 'pattern_match'}")
    if not summary["all_failed"]:
        print(f"match_error: {summary['match_error']}")
    print(f"queue_count_before: {summary['queue_count_before']}")
    print(f"queue_count_after: {summary['queue_count_after']}")
    print(f"failed_count_before: {summary['failed_count_before']}")
    print(f"failed_count_after: {summary['failed_count_after']}")
    print(f"selected_failed_count: {summary['selected_failed_count']}")
    print(f"requeued_count: {summary['requeued_count']}")
    print(f"already_queued_count: {summary['already_queued_count']}")
    print(f"invalid_selected_count: {summary['invalid_selected_count']}")

    categories = (
        ("requeued_examples", "Requeued"),
        ("already_queued_examples", "Already Queued"),
        ("invalid_selected_examples", "Invalid Selected"),
    )
    for key, title in categories:
        examples = list(summary.get(key) or [])
        if not examples or show_limit == 0:
            continue
        print(f"{title} ({len(examples)} shown):")
        for index, item in enumerate(examples[:show_limit], start=1):
            print(
                f"  {index}. [{item.get('task_type')}] depth={item.get('depth')} "
                f"key={item.get('dedupe_key')} title={item.get('title')!r}"
            )
            print(f"     url={item.get('url')}")
            print(f"     error={_short(item.get('error'), 240)}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input_dir)
    state_path = _find_state_path(input_path, state_file_name=args.state_file_name)
    graph_dir = state_path.parent
    state = _read_json(state_path)

    queue_before = list(state.get("queue") or [])
    failed_before = list(state.get("failed_tasks") or [])
    match_patterns = list(args.match_error or DEFAULT_MATCH_ERROR_PATTERNS)
    queue_after = [dict(item) for item in queue_before if isinstance(item, dict)]
    queue_keys = {_task_key(task) for task in queue_after}
    remaining_failed: list[dict[str, Any]] = []
    requeued: list[dict[str, Any]] = []
    already_queued: list[dict[str, Any]] = []
    invalid_selected: list[dict[str, Any]] = []
    selected_failed_count = 0

    for record in failed_before:
        if not isinstance(record, dict):
            remaining_failed.append(record)
            continue
        if not _matches_failed_record(record, match_patterns=match_patterns, all_failed=args.all_failed):
            remaining_failed.append(record)
            continue

        selected_failed_count += 1
        task = record.get("task")
        if not isinstance(task, dict) or not task.get("url"):
            invalid_selected.append(_task_summary(record))
            remaining_failed.append(record)
            continue

        task_copy = dict(task)
        task_copy["status"] = "pending"
        key = _task_key(task_copy)
        summary = _task_summary(record, dedupe_key=key)
        if key in queue_keys:
            already_queued.append(summary)
            if args.keep_failed_records:
                remaining_failed.append(record)
            continue

        queue_after.append(task_copy)
        queue_keys.add(key)
        requeued.append(summary)
        if args.keep_failed_records:
            remaining_failed.append(record)

    state_after = dict(state)
    state_after["queue"] = queue_after
    state_after["failed_tasks"] = failed_before if args.keep_failed_records else remaining_failed
    if queue_after:
        state_after["status"] = "paused"
    state_after["updated_at"] = _utc_now()
    stats = dict(state_after.get("stats") or {})
    stats["queue_size"] = len(queue_after)
    state_after["stats"] = stats

    backup_path = None
    if not args.dry_run:
        if not args.no_backup:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup_path = state_path.with_name(f"{state_path.name}.bak.{timestamp}")
            backup_path.write_text(state_path.read_text(encoding="utf-8"), encoding="utf-8")
        _write_state(state_path, state_after)

    summary = {
        "resolved_input_dir": str(input_path.resolve()),
        "graph_dir": str(graph_dir),
        "state_path": str(state_path),
        "backup_path": str(backup_path) if backup_path is not None else None,
        "dry_run": bool(args.dry_run),
        "all_failed": bool(args.all_failed),
        "match_error": match_patterns,
        "keep_failed_records": bool(args.keep_failed_records),
        "queue_count_before": len(queue_before),
        "queue_count_after": len(queue_after),
        "failed_count_before": len(failed_before),
        "failed_count_after": len(state_after.get("failed_tasks") or []),
        "selected_failed_count": selected_failed_count,
        "requeued_count": len(requeued),
        "already_queued_count": len(already_queued),
        "invalid_selected_count": len(invalid_selected),
        "requeued_examples": requeued[: max(0, args.show_limit)],
        "already_queued_examples": already_queued[: max(0, args.show_limit)],
        "invalid_selected_examples": invalid_selected[: max(0, args.show_limit)],
    }

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        _emit_human(summary, show_limit=args.show_limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
