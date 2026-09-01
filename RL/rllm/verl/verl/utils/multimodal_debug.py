"""Low-overhead multimodal diagnostics safe to import in Megatron workers.

This module intentionally lives under ``verl`` rather than ``rllm``.  Importing
``rllm.utils`` executes ``rllm/__init__.py`` and eagerly imports optional tools;
some of those tools install signal handlers and therefore cannot be imported
from Megatron's pipeline worker thread.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any


_lock = threading.Lock()
_event_count = 0
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = next((parent for parent in _THIS_FILE.parents if parent.name == "OpenSearch-VL"), _THIS_FILE.parents[5])
_DEFAULT_LOG_FILE = _PROJECT_ROOT / "synthesis/.ignore/rl_test.log"


def enabled() -> bool:
    return os.getenv("RLLM_MM_DEBUG", "0").lower() in {"1", "true", "yes", "on"}


def _rank() -> str:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return str(dist.get_rank())
    except Exception:  # noqa: BLE001
        pass
    return os.getenv("RANK", os.getenv("LOCAL_RANK", "na"))


def _rank_allowed(rank: str) -> bool:
    configured = os.getenv("RLLM_MM_DEBUG_RANK", "all").strip().lower()
    return configured in {"", "all", "*"} or configured == rank


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(dim) for dim in shape]
    except Exception:  # noqa: BLE001
        return [str(dim) for dim in shape]


def _safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > 32:
            return {"length": len(value), "preview": [_safe_value(item) for item in value[:4]]}
        return [_safe_value(item) for item in value]
    shape = _shape(value)
    if shape is not None:
        return {"shape": shape, "type": type(value).__name__}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_log_file(record: dict[str, Any]) -> None:
    log_file = os.getenv("RLLM_MM_DEBUG_LOG_FILE")
    if log_file is None and not os.getenv("RLLM_MM_DEBUG_LOG_DIR"):
        log_file = str(_DEFAULT_LOG_FILE)
    if log_file:
        output_path = Path(log_file)
        if not output_path.is_absolute():
            output_path = _PROJECT_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as stream:
            try:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                stream.flush()
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            except ImportError:
                stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                stream.flush()
        return

    log_dir = os.getenv("RLLM_MM_DEBUG_LOG_DIR")
    if log_dir:
        output_path = Path(log_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / f"multimodal_debug_pid{os.getpid()}_rank{_rank()}.jsonl"
        with output_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def event(stage: str, **fields: Any) -> None:
    global _event_count
    if not enabled():
        return
    rank = _rank()
    if not _rank_allowed(rank):
        return

    max_events = int(os.getenv("RLLM_MM_DEBUG_MAX_EVENTS", "40"))
    with _lock:
        if _event_count >= max_events:
            return
        _event_count += 1

    record = {"stage": stage, "pid": os.getpid(), "rank": rank, **{key: _safe_value(value) for key, value in fields.items()}}
    line = "[RLLM_MM_DEBUG] " + json.dumps(record, ensure_ascii=False, default=str)
    if os.getenv("RLLM_MM_DEBUG_STDOUT", "1").lower() in {"1", "true", "yes", "on"}:
        print(line, flush=True)
    try:
        _write_log_file(record)
    except Exception as exc:  # noqa: BLE001
        print(f"[RLLM_MM_DEBUG] failed to write debug log: {exc}", flush=True)


def count_token(value: Any, token_id: int | None) -> int | None:
    if token_id is None or value is None:
        return None
    try:
        return int((value == token_id).sum().item())
    except AttributeError:
        try:
            return sum(int(item) == token_id for item in value)
        except TypeError:
            return None


def token_ids_text(value: Any, max_tokens: int | None = None) -> str | None:
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            values = value.detach().to("cpu").reshape(-1).tolist()
        elif hasattr(value, "tolist"):
            values = value.tolist()
            while values and isinstance(values[0], list):
                values = values[0]
        else:
            values = list(value)
        values = [int(item) for item in values]
    except (TypeError, ValueError, RuntimeError):
        return None

    if max_tokens is None:
        max_tokens = int(os.getenv("RLLM_MM_DEBUG_TOKEN_IDS_MAX", "32768"))
    shown = values[: max(0, int(max_tokens))]
    text = " ".join(str(item) for item in shown)
    if len(values) > len(shown):
        text += f" ... [truncated; total={len(values)}]"
    return text


def token_positions(value: Any, token_id: int | None) -> list[int] | None:
    if token_id is None or value is None:
        return None
    try:
        if hasattr(value, "detach"):
            values = value.detach().to("cpu").reshape(-1).tolist()
        elif hasattr(value, "tolist"):
            values = value.tolist()
        else:
            values = list(value)
        return [index for index, item in enumerate(values) if int(item) == token_id]
    except (TypeError, ValueError, RuntimeError):
        return None
