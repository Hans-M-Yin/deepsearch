#!/usr/bin/env python3
"""Repair malformed <thinking>/<answer> wrappers in a ShareGPT JSON dataset.

The repair is deliberately deterministic and only touches malformed assistant
messages. Tool-call blocks and answer text are preserved; the script records
the original values of every changed message in a JSONL backup before replacing
the input file atomically.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


THINKING_OPEN_RE = re.compile(r"<thinking\b[^>]*>", re.IGNORECASE)
THINKING_CLOSE_RE = re.compile(r"</thinking\s*>", re.IGNORECASE)
ANSWER_OPEN_RE = re.compile(r"<answer\b[^>]*>", re.IGNORECASE)
ANSWER_CLOSE_RE = re.compile(r"</answer\s*>", re.IGNORECASE)
ANSWER_BLOCK_RE = re.compile(r"<answer\b[^>]*>\s*(.*?)\s*</answer\s*>", re.IGNORECASE | re.DOTALL)
TOOL_BLOCK_RE = re.compile(r"<tool_call\b[^>]*>.*?</tool_call\s*>", re.IGNORECASE | re.DOTALL)


def signature(value: str) -> tuple[int, int, int, int]:
    return (
        len(THINKING_OPEN_RE.findall(value)),
        len(THINKING_CLOSE_RE.findall(value)),
        len(ANSWER_OPEN_RE.findall(value)),
        len(ANSWER_CLOSE_RE.findall(value)),
    )


def is_malformed(value: str) -> bool:
    thinking_open, thinking_close, answer_open, answer_close = signature(value)
    return (
        thinking_open != thinking_close
        or thinking_open > 1
        or answer_open != answer_close
        or answer_open > 1
    )


def normalized_answer(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _strip_reasoning_markers(value: str) -> str:
    value = THINKING_OPEN_RE.sub("", value)
    value = THINKING_CLOSE_RE.sub("", value)
    return value.strip()


def _tool_blocks(value: str) -> list[str]:
    return [match.group(0) for match in TOOL_BLOCK_RE.finditer(value)]


def _repair_tool_turn(value: str) -> str:
    """Normalize reasoning wrappers while preserving every tool block verbatim."""
    blocks = list(TOOL_BLOCK_RE.finditer(value))
    if not blocks:
        raise ValueError("internal error: expected a tool call")

    reasoning_parts: list[str] = []
    cursor = 0
    for match in blocks:
        reasoning_parts.append(value[cursor : match.start()])
        cursor = match.end()
    reasoning_parts.append(value[cursor:])
    reasoning = _strip_reasoning_markers("".join(reasoning_parts))
    if not reasoning:
        raise ValueError("refusing to repair a tool turn with empty reasoning")

    thinking = f"<thinking>\n{reasoning}\n</thinking>"
    return thinking + "\n" + "\n".join(match.group(0) for match in blocks)


def _repair_final_turn(value: str) -> str:
    answers = [match.group(1).strip() for match in ANSWER_BLOCK_RE.finditer(value)]
    if not answers:
        raise ValueError("internal error: expected an answer block")
    if len({normalized_answer(answer) for answer in answers}) != 1:
        raise ValueError("refusing to merge malformed message with different answer blocks")

    # Remove every answer block first. This also removes an answer accidentally
    # nested inside the outer thinking block, while retaining the final answer
    # text exactly once below.
    reasoning = ANSWER_BLOCK_RE.sub("", value)
    reasoning = _strip_reasoning_markers(reasoning)
    if not reasoning:
        raise ValueError("refusing to repair a final turn with empty reasoning")

    answer = answers[-1]
    return f"<thinking>\n{reasoning}\n</thinking>\n<answer>\n{answer}\n</answer>"


def repair_value(value: str) -> str:
    if not is_malformed(value):
        return value
    if ANSWER_OPEN_RE.search(value) or ANSWER_CLOSE_RE.search(value):
        return _repair_final_turn(value)
    if TOOL_BLOCK_RE.search(value):
        return _repair_tool_turn(value)
    raise ValueError("unsupported malformed assistant message without tool or answer")


def _load(path: Path) -> tuple[Any, bool]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError(f"expected a JSON array at {path}")
    return value, False


def _write_json(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _backup_record(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--backup-jsonl",
        type=Path,
        help="JSONL patch backup; defaults to <input>.tag_cleanup_backup.jsonl",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    records, _ = _load(args.input)
    backup_path = args.backup_jsonl or args.input.with_name(
        f"{args.input.stem}.tag_cleanup_backup.jsonl"
    )
    backup_path.parent.mkdir(parents=True, exist_ok=True)

    changed: list[dict[str, Any]] = []
    repaired_records = copy.deepcopy(records)
    for record_index, record in enumerate(repaired_records):
        for message_index, message in enumerate(record.get("conversations") or []):
            if str(message.get("from", "")).lower() != "gpt":
                continue
            original_value = str(message.get("value", ""))
            if not is_malformed(original_value):
                continue
            repaired_value = repair_value(original_value)
            before_tools = _tool_blocks(original_value)
            after_tools = _tool_blocks(repaired_value)
            if before_tools != after_tools:
                raise ValueError(
                    f"tool block changed at record {record_index}, message {message_index}"
                )
            changed.append(
                {
                    "record_index": record_index,
                    "id": record.get("id"),
                    "message_index": message_index,
                    "signature_before": signature(original_value),
                    "signature_after": signature(repaired_value),
                    "original_value": original_value,
                    "repaired_value": repaired_value,
                }
            )
            message["value"] = repaired_value

    summary = {
        "input": str(args.input),
        "dry_run": args.dry_run,
        "changed_messages": len(changed),
        "changed_records": len({item["record_index"] for item in changed}),
        "backup_jsonl": None if args.dry_run else str(backup_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    if args.dry_run or not changed:
        return 0

    # Append one recoverable patch per changed message before replacing the
    # large JSON file. If interrupted, the original input remains untouched.
    for item in changed:
        _backup_record(backup_path, item)

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{args.input.name}.tagclean-",
        suffix=".tmp",
        dir=args.input.parent,
        text=True,
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        _write_json(temporary_path, repaired_records)
        os.replace(temporary_path, args.input)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
