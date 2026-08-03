#!/usr/bin/env python3
"""List every ``read_url`` call and its returned content from an SFT JSONL.

Example:
    python debug/list_sft_read_urls.py \
        --input-jsonl runs/.../sft_0802_141558.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


SEPARATOR = "=" * 100


def _parse_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    messages = record.get("raw_messages")
    if not isinstance(messages, list):
        messages = record.get("messages")
    return [message for message in (messages or []) if isinstance(message, dict)]


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object on line {line_number} of {path}.")
            yield line_number, record


def _call_arguments(tool_call: dict[str, Any]) -> dict[str, Any] | Any:
    function = tool_call.get("function")
    arguments = function.get("arguments") if isinstance(function, dict) else tool_call.get("arguments")
    return _parse_json(arguments)


def _is_read_url_call(tool_call: Any) -> bool:
    if not isinstance(tool_call, dict):
        return False
    function = tool_call.get("function")
    name = function.get("name") if isinstance(function, dict) else tool_call.get("name")
    return name == "read_url"


def _sample_matches(record: dict[str, Any], sample_ids: set[str]) -> bool:
    return not sample_ids or str(record.get("sample_id") or "") in sample_ids


def _print_read_url_result(
    *,
    ordinal: int,
    record: dict[str, Any],
    source_line: int,
    tool_call_id: str,
    arguments: Any,
    tool_message: dict[str, Any],
) -> None:
    result = _parse_json(tool_message.get("content"))
    result_dict = result if isinstance(result, dict) else {}
    url = result_dict.get("url")
    if not url and isinstance(arguments, dict):
        url = arguments.get("url")
    title = result_dict.get("title")
    content = result_dict.get("content") if result_dict else result

    print("\n" + SEPARATOR)
    print(f"Read URL #{ordinal}")
    print(f"jsonl_line: {source_line}")
    print(f"sample_id: {record.get('sample_id')}")
    if record.get("question_id") is not None:
        print(f"question_id: {record.get('question_id')}")
    print(f"tool_call_id: {tool_call_id or tool_message.get('tool_call_id') or ''}")
    if isinstance(arguments, dict):
        if arguments.get("resource_id"):
            print(f"resource_id: {arguments['resource_id']}")
        if arguments.get("goal"):
            print(f"goal: {arguments['goal']}")
    elif arguments not in (None, ""):
        print(f"arguments: {arguments}")
    print(f"url: {url or '(not returned)'}")
    if title:
        print(f"title: {title}")
    if result_dict.get("ok") is not None:
        print(f"ok: {result_dict['ok']}")
    if result_dict.get("kind"):
        print(f"kind: {result_dict['kind']}")
    if result_dict.get("error"):
        print(f"error: {result_dict['error']}")
    print("--- returned content ---")
    if content in (None, ""):
        print("(empty)")
    elif isinstance(content, str):
        print(content)
    else:
        print(json.dumps(content, ensure_ascii=False, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True, help="SFT trajectory JSONL.")
    parser.add_argument(
        "--sample-id",
        action="append",
        default=[],
        help="Only inspect this sample ID. Repeat to select multiple samples.",
    )
    parser.add_argument("--limit", type=int, help="Maximum number of read_url results to print.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative.")
    if not args.input_jsonl.is_file():
        raise FileNotFoundError(f"JSONL file does not exist: {args.input_jsonl}")

    sample_ids = {str(sample_id).strip() for sample_id in args.sample_id if str(sample_id).strip()}
    printed = 0
    for source_line, record in _read_jsonl(args.input_jsonl):
        if not _sample_matches(record, sample_ids):
            continue
        pending_calls: dict[str, Any] = {}
        for message in _messages(record):
            for tool_call in message.get("tool_calls") or []:
                if _is_read_url_call(tool_call):
                    pending_calls[str(tool_call.get("id") or "")] = _call_arguments(tool_call)
            if message.get("role") != "tool" or message.get("name") != "read_url":
                continue
            if args.limit is not None and printed >= args.limit:
                break
            call_id = str(message.get("tool_call_id") or "")
            printed += 1
            _print_read_url_result(
                ordinal=printed,
                record=record,
                source_line=source_line,
                tool_call_id=call_id,
                arguments=pending_calls.get(call_id),
                tool_message=message,
            )
        if args.limit is not None and printed >= args.limit:
            break

    if not printed:
        suffix = f" for sample IDs {sorted(sample_ids)}" if sample_ids else ""
        print(f"No read_url tool results found{suffix}.", file=sys.stderr)
        return 1
    print("\n" + SEPARATOR)
    print(f"Printed {printed} read_url result(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
