"""Small streaming/atomic I/O helpers used by every pipeline stage."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator
import uuid


def iter_json_records(
    path: str | Path,
    *,
    allow_incomplete_final_line: bool = False,
) -> Iterator[dict[str, Any]]:
    """Read JSONL or a top-level JSON array without loading the latter at once.

    ``allow_incomplete_final_line`` is for a reader following a JSONL file
    concurrently with its writer.  A writer may have flushed part of its last
    record but not its terminating newline when a router takes a snapshot.  In
    that narrowly defined case, retain the valid prefix and let the next poll
    retry the unfinished record.  Normal readers remain strict by default.
    """

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first != "[":
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    # JSONL records are newline-delimited.  Therefore a
                    # malformed record without a trailing newline can only be
                    # the writer's current final record in this file snapshot.
                    # Ignore it only in explicit follow mode; a later poll
                    # will see the completed record, while a persisted corrupt
                    # record remains an error for normal/offline consumers.
                    if allow_incomplete_final_line and not raw_line.endswith("\n"):
                        return
                    raise
                if not isinstance(value, dict):
                    raise ValueError(f"{source}:{line_number} is not a JSON object")
                yield value
            return

        decoder = json.JSONDecoder()
        buffer = ""
        eof = False
        started = False
        while not eof or buffer.strip():
            # Always request another chunk after retaining an incomplete item.
            # A ShareGPT record can itself exceed one chunk, so a size-based
            # refill condition would otherwise loop forever on that record.
            if not eof:
                chunk = handle.read(64 * 1024)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
            offset = 0
            while True:
                while offset < len(buffer) and buffer[offset].isspace():
                    offset += 1
                if not started:
                    if offset >= len(buffer):
                        break
                    if buffer[offset] != "[":
                        raise ValueError(f"{source} must be a JSON array")
                    started = True
                    offset += 1
                    continue
                while offset < len(buffer) and (buffer[offset].isspace() or buffer[offset] == ","):
                    offset += 1
                if offset >= len(buffer):
                    break
                if buffer[offset] == "]":
                    return
                try:
                    value, end = decoder.raw_decode(buffer, offset)
                except json.JSONDecodeError:
                    break
                if not isinstance(value, dict):
                    raise ValueError(f"{source} contains a non-object array item")
                yield value
                offset = end
            buffer = buffer[offset:]
            if eof:
                if buffer.strip() not in {"", "]"}:
                    raise ValueError(f"Unterminated JSON array in {source}")
                return


def append_jsonl(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def write_json(path: str | Path, value: Any) -> None:
    """Atomically publish small manifests/reports."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(target)


def load_completed_ids(
    path: str | Path,
    *,
    key: str = "case_id",
    require_success: bool = False,
) -> set[str]:
    """Load durable completions; retryable error rows are not completions."""

    completed: set[str] = set()
    source = Path(path)
    if not source.exists():
        return completed
    for record in iter_json_records(source):
        if require_success and (record.get("error") or record.get("collection_error")):
            continue
        value = str(record.get(key) or "").strip()
        if value:
            completed.add(value)
    return completed
