"""Low-volume diagnostics for multimodal RL data flow.

The diagnostics are intentionally disabled by default.  Enable them with
``RLLM_MM_DEBUG=1``.  They only print metadata and tensor shapes; image
payloads themselves are never serialized.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any


_lock = threading.Lock()
_event_count = 0
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_LOG_FILE = _PROJECT_ROOT / "synthesis/.ignore/rl_test.log"


def sequence_mode() -> str:
    """Return the multimodal sequence-construction mode.

    The default deliberately preserves the historical behavior.  The two
    opt-in modes are used for the staged multimodal fix:

    * ``stepwise``: use the processor output captured for every model call.
    * ``cumulative``: build one processor-expanded sequence for the complete
      multi-turn trajectory, while keeping the cumulative response mask.

    Aliases are accepted so launch scripts can use ``phase1``/``phase2`` or
    ``fixed`` without changing the implementation.  Invalid values fail fast
    on the worker instead of silently selecting a different data path.
    """
    raw = os.getenv("RLLM_MM_SEQUENCE_MODE", "legacy").strip().lower()
    aliases = {
        "legacy": "legacy",
        "current": "legacy",
        "stepwise": "stepwise",
        "phase1": "stepwise",
        "stage1": "stepwise",
        "cumulative": "cumulative",
        "fixed": "cumulative",
        "phase2": "cumulative",
        "stage2": "cumulative",
    }
    if raw not in aliases:
        raise ValueError(
            "RLLM_MM_SEQUENCE_MODE must be one of legacy, stepwise, cumulative "
            f"(got {raw!r})"
        )
    return aliases[raw]


def enabled() -> bool:
    return os.getenv("RLLM_MM_DEBUG", "0").lower() in {"1", "true", "yes", "on"}


def abort_on_missing_payload() -> bool:
    return enabled() and os.getenv("RLLM_MM_DEBUG_ABORT_ON_MISSING", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


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
            return {"length": len(value), "preview": [_safe_value(x) for x in value[:4]]}
        return [_safe_value(item) for item in value]
    shape = _shape(value)
    if shape is not None:
        return {"shape": shape, "type": type(value).__name__}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_log_file(record: dict[str, Any]) -> None:
    """Append one event to a shared log file when requested.

    ``fcntl`` keeps writes from multiple Ray/Megatron processes from
    interleaving on the same node.  The log contains metadata only; image and
    tensor payloads are never written.
    """

    log_file = os.getenv("RLLM_MM_DEBUG_LOG_FILE")
    if log_file is None and not os.getenv("RLLM_MM_DEBUG_LOG_DIR"):
        log_file = str(_DEFAULT_LOG_FILE)
    if log_file:
        output_path = Path(log_file)
        if not output_path.is_absolute():
            output_path = _PROJECT_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with output_path.open("a", encoding="utf-8") as stream:
            try:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                stream.write(line)
                stream.flush()
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            except ImportError:
                stream.write(line)
                stream.flush()
        return

    # Keep the original per-process directory option for existing launchers.
    log_dir = os.getenv("RLLM_MM_DEBUG_LOG_DIR")
    if log_dir:
        output_dir = Path(log_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"multimodal_debug_pid{os.getpid()}_rank{_rank()}.jsonl"
        with output_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def event(stage: str, **fields: Any) -> None:
    """Emit one bounded multimodal debug event."""

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

    record = {
        "stage": stage,
        "pid": os.getpid(),
        "rank": rank,
        **{key: _safe_value(value) for key, value in fields.items()},
    }
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
    """Return a compact printable representation of a token-id sequence.

    This is intentionally opt-in at the call site because a full sequence can
    be large.  The value is returned as text so ``event`` does not apply its
    normal list preview truncation.
    """
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
    except Exception:  # noqa: BLE001
        return None

    try:
        values = [int(item) for item in values]
    except (TypeError, ValueError):
        return None

    if max_tokens is None:
        max_tokens = int(os.getenv("RLLM_MM_DEBUG_TOKEN_IDS_MAX", "32768"))
    max_tokens = max(0, int(max_tokens))
    shown = values[:max_tokens] if max_tokens else []
    text = " ".join(str(item) for item in shown)
    if len(values) > len(shown):
        text += f" ... [truncated; total={len(values)}]"
    return text


def token_positions(value: Any, token_id: int | None) -> list[int] | None:
    """Return positions of ``token_id`` in a one-dimensional token sequence."""
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
    except (TypeError, ValueError):
        return None
