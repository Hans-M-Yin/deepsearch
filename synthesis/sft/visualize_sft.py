#!/usr/bin/env python3
"""Print one or more SFT trajectories in a compact, human-readable form.

Examples:
    python -m synthesis.sft.visualize_sft data.jsonl --id sample_path_abc
    python -m synthesis.sft.visualize_sft data.jsonl --index 12
    python -m synthesis.sft.visualize_sft data.jsonl --samples 5 --seed 42

``--index`` is zero-based.  IDs are matched against ``sample_id``,
``question_id``, and ``path_id`` in that order.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable


_ID_FIELDS = ("sample_id", "question_id", "path_id")
_HIDDEN_MESSAGE_ROLES = {"system", "developer"}


def _message_text(content: Any) -> str:
    """Render message content without exposing binary image payloads."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            item_type = str(item.get("type") or "").lower()
            if item_type in {"text", "input_text"}:
                text = str(item.get("text") or "")
                if text:
                    parts.append(text)
            elif item_type in {"image", "image_url", "input_image", "image_path"}:
                parts.append("<image>")
            else:
                # Keep non-image structured content readable while avoiding
                # accidental dumping of opaque binary/data URLs.
                compact = {key: value for key, value in item.items() if key not in {"data", "url"}}
                if compact:
                    parts.append(json.dumps(compact, ensure_ascii=False, indent=2, default=str))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False, indent=2, default=str)
    return str(content)


def _format_tool_calls(tool_calls: Any) -> str:
    """Render assistant tool names and arguments in a readable form."""

    if not isinstance(tool_calls, list):
        return ""

    rendered: list[str] = []
    for index, tool_call in enumerate(tool_calls, start=1):
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if not isinstance(function, dict):
            function = tool_call
        name = str(function.get("name") or tool_call.get("name") or "unknown_tool").strip()
        call_id = str(tool_call.get("id") or tool_call.get("call_id") or "").strip()
        raw_arguments = function.get("arguments", tool_call.get("arguments", {}))
        if isinstance(raw_arguments, str):
            try:
                arguments: Any = json.loads(raw_arguments)
            except json.JSONDecodeError:
                arguments = raw_arguments
        else:
            arguments = raw_arguments

        header = f"  [{index}] {name}"
        if call_id:
            header += f" (id={call_id})"
        rendered.append(
            f"{header}\n"
            f"  arguments: {json.dumps(arguments, ensure_ascii=False, indent=2, default=str)}"
        )
    return "tool_calls:\n" + "\n".join(rendered) if rendered else ""


def _record_identifier(record: dict[str, Any]) -> str:
    for field in _ID_FIELDS:
        value = str(record.get(field) or "").strip()
        if value:
            return value
    return ""


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on physical line {index + 1}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL physical line {index + 1} is not an object")
            yield index, record


def find_record(path: str | Path, *, record_id: str | None = None, index: int | None = None) -> dict[str, Any]:
    """Find one record by ID or zero-based physical JSONL index."""

    if (record_id is None) == (index is None):
        raise ValueError("Provide exactly one of record_id or index")
    if index is not None and index < 0:
        raise ValueError("index must be non-negative")

    for physical_index, record in _iter_jsonl(Path(path)):
        if index is not None and physical_index == index:
            return record
        if record_id is not None and any(str(record.get(field) or "").strip() == record_id for field in _ID_FIELDS):
            return record
    selector = f"index {index}" if index is not None else f"id {record_id!r}"
    raise LookupError(f"No JSONL record found for {selector}")


def sample_records(
    path: str | Path,
    *,
    samples: int,
    seed: int | None = None,
) -> list[tuple[int, dict[str, Any]]]:
    """Randomly select distinct records and retain their physical line indexes."""

    if samples <= 0:
        raise ValueError("samples must be positive")

    records = list(_iter_jsonl(Path(path)))
    if samples > len(records):
        raise ValueError(
            f"Requested {samples} samples, but the JSONL contains only {len(records)} records"
        )
    return random.Random(seed).sample(records, samples)


def format_trajectory(record: dict[str, Any]) -> str:
    """Format the record's raw messages as User/Tool/Assistant turns."""

    messages = record.get("raw_messages")
    if not isinstance(messages, list):
        raise ValueError("The selected record has no raw_messages list")

    rendered: list[str] = []
    turn = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip().lower()
        if role in _HIDDEN_MESSAGE_ROLES:
            # Do not expose system/developer prompts in the visualization.
            continue
        if role not in {"user", "tool", "assistant"}:
            continue
        turn += 1
        label = role.capitalize()
        if role == "tool":
            content = "(Tool results emitted)"
        else:
            content_parts = [_message_text(message.get("content"))]
            if role == "assistant":
                content_parts.append(_format_tool_calls(message.get("tool_calls")))
            content = "\n\n".join(part for part in content_parts if part)
        rendered.append(f"[Turn {turn}]{label}\n{content}")
    return "\n\n\n".join(rendered)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, help="Input trajectory JSONL")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--id", dest="record_id", help="Match sample_id, question_id, or path_id")
    selector.add_argument("--index", type=int, help="Zero-based physical JSONL line index")
    selector.add_argument("--samples", type=int, help="Randomly sample and print this many trajectories")
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional random seed for reproducible --samples output",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.samples is not None:
        selected = sample_records(args.jsonl, samples=args.samples, seed=args.seed)
        for sample_number, (physical_index, record) in enumerate(selected, start=1):
            if sample_number > 1:
                print("\n\n")
            identifier = _record_identifier(record) or "<no-id>"
            print(
                f"===== Sample {sample_number}/{len(selected)} "
                f"| index={physical_index} | id={identifier} ====="
            )
            print(format_trajectory(record))
    else:
        record = find_record(args.jsonl, record_id=args.record_id, index=args.index)
        print(format_trajectory(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
