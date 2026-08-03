#!/usr/bin/env python3
"""Print complete SFT agent trajectories saved in a JSONL file.

The script is intentionally read-only: it replays the raw messages written by
``synthesis.sft.debug_vqa_batch`` and does not call a model or any tool.

Example:
    python -m synthesis.vqa.debug.debug_sft \
        --input-jsonl synthesis_sft_runs/trajectories.jsonl \
        --sample-id sample_path_123 --sample-id sample_path_456
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


def _content_text(content: Any) -> str:
    """Render OpenAI-style content while retaining image references."""
    if content in (None, ""):
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                parts.append(str(part))
                continue
            kind = str(part.get("type") or "")
            if kind in {"text", "input_text", "output_text"}:
                text = str(part.get("text") or "")
                if text:
                    parts.append(text)
            elif kind == "image_url":
                image_url = part.get("image_url")
                url = image_url.get("url") if isinstance(image_url, dict) else image_url
                parts.append(f"[image_url] {url or ''}")
            elif kind in {"image", "input_image", "image_path", "image_ref"}:
                source = (
                    part.get("image")
                    or part.get("path")
                    or part.get("url")
                    or part.get("image_url")
                    or part.get("ref")
                    or ""
                )
                parts.append(f"[{kind}] {source}")
            else:
                parts.append(str(part))
        return "\n".join(part for part in parts if part).strip()
    if isinstance(content, (dict, tuple)):
        return str(content)
    return str(content)


def _record_ids(record: dict[str, Any]) -> set[str]:
    values = (record.get("sample_id"), record.get("question_id"), record.get("path_id"))
    return {str(value) for value in values if value is not None and str(value).strip()}


def _matches(record: dict[str, Any], selected_ids: set[str]) -> bool:
    return not selected_ids or bool(_record_ids(record) & selected_ids)


def _messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    # ``raw_messages`` is the persisted SFT format. ``messages`` supports the
    # in-memory/debug-result format from older batch runs.
    raw = record.get("raw_messages")
    if not isinstance(raw, list):
        raw = record.get("messages")
    return [message for message in (raw or []) if isinstance(message, dict)]


def _increment(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def _tool_call_name(tool_call: Any) -> str:
    if not isinstance(tool_call, dict):
        return "unknown_tool"
    function = tool_call.get("function")
    if isinstance(function, dict) and function.get("name"):
        return str(function["name"])
    return str(tool_call.get("name") or "unknown_tool")


def _tool_summary(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Count issued calls and returned tool results separately."""
    issued: dict[str, int] = {}
    returned: dict[str, int] = {}
    succeeded: dict[str, int] = {}
    failed: dict[str, int] = {}
    for message in messages:
        for tool_call in message.get("tool_calls") or []:
            _increment(issued, _tool_call_name(tool_call))
        if str(message.get("role") or "") != "tool":
            continue
        name = str(message.get("name") or "unknown_tool")
        _increment(returned, name)
        content = message.get("content")
        try:
            parsed = json.loads(content) if isinstance(content, str) else content
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict) and parsed.get("ok") is False:
            _increment(failed, name)
        else:
            _increment(succeeded, name)
    return {
        "issued_calls": dict(sorted(issued.items())),
        "returned_results": dict(sorted(returned.items())),
        "successful_results": dict(sorted(succeeded.items())),
        "failed_results": dict(sorted(failed.items())),
        "issued_call_count": sum(issued.values()),
        "returned_result_count": sum(returned.values()),
    }


def _max_turns_reached(summary: dict[str, Any]) -> bool:
    status = str(summary.get("generation_status") or "unknown")
    stop_reason = str(summary.get("stop_reason") or "unknown")
    return status == "max_turns_reached" or stop_reason in {
        "max_react_turns",
        "max_tool_calling_turns",
    }


def _empty_statistics() -> dict[str, Any]:
    return {
        "sample_count": 0,
        "judged_count": 0,
        "correct_count": 0,
        "turn_count_sum": 0,
        "turn_count_samples": 0,
        "max_turns_reached": 0,
        "tool_issued": {},
        "tool_returned": {},
        "tool_success": {},
        "tool_failed": {},
    }


def _merge_counts(destination: dict[str, int], source: dict[str, int]) -> None:
    for name, count in source.items():
        destination[name] = destination.get(name, 0) + count


def _update_statistics(statistics: dict[str, Any], record: dict[str, Any]) -> None:
    statistics["sample_count"] += 1
    judge = record.get("answer_judge")
    if isinstance(judge, dict) and isinstance(judge.get("is_correct"), bool):
        statistics["judged_count"] += 1
        if judge["is_correct"]:
            statistics["correct_count"] += 1
    summary = record.get("generation_summary") or {}
    turn_count = summary.get("turn_count")
    if isinstance(turn_count, (int, float)):
        statistics["turn_count_sum"] += turn_count
        statistics["turn_count_samples"] += 1
    if _max_turns_reached(summary):
        statistics["max_turns_reached"] += 1
    tool_summary = _tool_summary(_messages(record))
    _merge_counts(statistics["tool_issued"], tool_summary["issued_calls"])
    _merge_counts(statistics["tool_returned"], tool_summary["returned_results"])
    _merge_counts(statistics["tool_success"], tool_summary["successful_results"])
    _merge_counts(statistics["tool_failed"], tool_summary["failed_results"])


def _print_count_lines(title: str, counts: dict[str, int]) -> None:
    print(title)
    if not counts:
        print("  (none)")
        return
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count}")


def _print_statistics(statistics: dict[str, Any]) -> None:
    average_turns = (
        statistics["turn_count_sum"] / statistics["turn_count_samples"]
        if statistics["turn_count_samples"]
        else 0.0
    )
    print("\n" + "=" * 100)
    print("All-sample summary")
    print(f"samples: {statistics['sample_count']}")
    print(f"correct: {statistics['correct_count']}/{statistics['judged_count']} judged samples")
    print(f"average model rounds: {average_turns:.2f}")
    print(f"max-turn limit reached: {statistics['max_turns_reached']}")
    _print_count_lines("tool calls issued:", statistics["tool_issued"])
    _print_count_lines("tool results returned:", statistics["tool_returned"])
    _print_count_lines("successful tool results:", statistics["tool_success"])
    _print_count_lines("failed tool results:", statistics["tool_failed"])


def _print_record(record: dict[str, Any], *, ordinal: int) -> None:
    print("\n" + "=" * 100)
    print(f"SFT trajectory #{ordinal}")
    for key in ("question_id", "sample_id", "path_id"):
        if record.get(key) is not None:
            print(f"{key}: {record[key]}")
    if record.get("question") is not None:
        print(f"question: {record['question']}")
    if record.get("gold_answer") is not None:
        print(f"gold_answer: {record['gold_answer']}")

    messages = _messages(record)
    print(f"\n--- Complete trajectory ({len(messages)} messages) ---")
    for index, message in enumerate(messages, start=1):
        role = str(message.get("role") or "unknown").upper()
        print(f"\n[{index}] {role}")
        if message.get("name"):
            print(f"tool_name: {message['name']}")
        if message.get("tool_call_id"):
            print(f"tool_call_id: {message['tool_call_id']}")
        if message.get("arguments") is not None:
            print(f"tool_arguments: {message['arguments']}")
        # Tool outputs frequently contain long retrieved pages or image-search
        # payloads.  The trajectory debug view intentionally keeps the call
        # specification but omits those result bodies.
        if str(message.get("role") or "") == "tool":
            continue
        content = _content_text(message.get("content"))
        if content:
            print(content)
        tool_calls = message.get("tool_calls")
        if tool_calls:
            for tool_call in tool_calls:
                print(f"tool_call: {_tool_call_name(tool_call)}")
                if isinstance(tool_call, dict):
                    function = tool_call.get("function")
                    arguments = function.get("arguments") if isinstance(function, dict) else tool_call.get("arguments")
                    if arguments not in (None, ""):
                        print(f"tool_arguments: {arguments}")
    if not messages:
        print("(This record has no raw_messages/messages field.)")


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object on line {line_number} of {path}.")
            yield line_number, value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, help="Raw SFT trajectory JSONL to replay.")
    parser.add_argument(
        "--sample-id",
        action="append",
        default=[],
        help="Only print this sample/question/path id. Repeat this argument to select several records.",
    )
    parser.add_argument("--offset", type=int, default=0, help="Skip this many matched records before printing.")
    parser.add_argument("--limit", type=int, help="Maximum number of matched records to print.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.offset < 0 or (args.limit is not None and args.limit < 0):
        raise ValueError("--offset and --limit must be non-negative.")
    input_path = Path(args.input_jsonl)
    if not input_path.is_file():
        raise FileNotFoundError(f"JSONL file does not exist: {input_path}")

    selected_ids = {str(value).strip() for value in args.sample_id if str(value).strip()}
    matched = printed = 0
    statistics = _empty_statistics()
    for _, record in _read_jsonl(input_path):
        if not _matches(record, selected_ids):
            continue
        if matched < args.offset:
            matched += 1
            continue
        if args.limit is not None and printed >= args.limit:
            break
        matched += 1
        printed += 1
        _print_record(record, ordinal=printed)
        _update_statistics(statistics, record)

    if not printed:
        selector = f" matching ids {sorted(selected_ids)}" if selected_ids else ""
        print(f"No trajectory records found{selector}.", file=sys.stderr)
        return 1
    _print_statistics(statistics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
