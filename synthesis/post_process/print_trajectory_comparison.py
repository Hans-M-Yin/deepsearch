#!/usr/bin/env python3
"""Export original/modified trajectory pairs as readable plain text.

The script pairs records by their stable ``id`` (or the usual ShareGPT id
fallback fields), preserves User and Assistant messages, and intentionally
omits every tool observation body.  Tool calls remain inside the Assistant
message so the search sequence is still visible.  Consecutive Assistant
messages are rendered as one Assistant turn; this is a display normalization
only and does not modify the JSON trajectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _record_id(record: dict[str, Any], index: int) -> str:
    for key in ("id", "sample_id", "question_id", "path_id"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return f"record_{index}"


def _load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError(f"{path} must contain a JSON list")
        return [item for item in value if isinstance(item, dict)]
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        records.append(value)
    return records


def _message_value(message: dict[str, Any]) -> str:
    value = message.get("value", message.get("content", ""))
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def _role(message: dict[str, Any]) -> str:
    role = str(message.get("from", message.get("role", ""))).lower()
    if role in {"human", "user"}:
        return "User"
    if role in {"gpt", "assistant"}:
        return "Assistant"
    if role in {"observation", "tool"}:
        return "Tool"
    return role or "Message"


def _render_trajectory(record: dict[str, Any]) -> str:
    lines: list[str] = []
    assistant_turn = -1
    conversations = record.get("conversations", [])
    index = 0
    while index < len(conversations):
        message = conversations[index]
        if not isinstance(message, dict):
            index += 1
            continue
        role = _role(message)
        if role == "Assistant":
            assistant_turn += 1
            assistant_values = [_message_value(message).rstrip()]
            next_index = index + 1
            while next_index < len(conversations):
                next_message = conversations[next_index]
                if not isinstance(next_message, dict) or _role(next_message) != "Assistant":
                    break
                assistant_values.append(_message_value(next_message).rstrip())
                next_index += 1
            lines.extend(
                [
                    f"[Turn {assistant_turn:03d}] Assistant",
                    "\n\n".join(assistant_values),
                    "",
                ]
            )
            index = next_index
            continue
        elif role == "Tool":
            # Tool output can contain very large snippets/images.  It is
            # deliberately omitted, but the turn marker remains visible.
            lines.extend(
                [
                    f"[Turn {max(assistant_turn, 0):03d}] Tool",
                    "[tool response omitted]",
                    "",
                ]
            )
        elif role == "User":
            lines.extend(
                [
                    f"[Turn {max(assistant_turn, 0):03d}] User",
                    _message_value(message).rstrip(),
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    f"[Turn {max(assistant_turn, 0):03d}] {role}",
                    _message_value(message).rstrip(),
                    "",
                ]
            )
        index += 1
    return "\n".join(lines).rstrip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--modified", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    original_records = _load_records(args.original)
    modified_records = _load_records(args.modified)
    original_by_id = {
        _record_id(record, index): record
        for index, record in enumerate(original_records)
    }

    pairs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    missing: list[str] = []
    for index, modified in enumerate(modified_records):
        record_id = _record_id(modified, index)
        original = original_by_id.get(record_id)
        if original is None:
            missing.append(record_id)
            continue
        pairs.append((record_id, original, modified))

    if missing:
        raise ValueError(f"modified records missing from original input: {missing[:10]}")
    if not pairs:
        raise ValueError("no records could be paired")

    output: list[str] = []
    for number, (record_id, original, modified) in enumerate(pairs, 1):
        output.extend(
            [
                "=" * 100,
                f"Sample {number}/{len(pairs)}: {record_id}",
                "=" * 100,
                "",
                "#################### ORIGINAL ####################",
                "",
                _render_trajectory(original),
                "",
                "#################### MODIFIED ####################",
                "",
                _render_trajectory(modified),
                "",
            ]
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(output).rstrip() + "\n", encoding="utf-8")
    print(json.dumps({"paired": len(pairs), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
