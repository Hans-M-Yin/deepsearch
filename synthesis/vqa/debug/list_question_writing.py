"""Pretty-print the question writing process for VQA samples.

Examples:
    python -m synthesis.vqa.debug.list_question_writing --vqa-dir /path/to/graph/vqa/0713_123456
    python -m synthesis.vqa.debug.list_question_writing --samples-file /path/to/samples.jsonl
    python -m synthesis.vqa.debug.list_question_writing --vqa-dir /path/to/vqa_dir --limit 5
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any, Iterable


SEPARATOR = "=" * 88
SUB_SEPARATOR = "-" * 88


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--vqa-dir", type=Path, help="Directory containing samples.jsonl.")
    group.add_argument("--samples-file", type=Path, help="Path to samples.jsonl.")
    parser.add_argument("--sample-id", action="append", default=[], help="Only print the given sample_id. Repeatable.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of samples to print.")
    parser.add_argument("--width", type=int, default=100, help="Wrap width for long text fields.")
    return parser


def _resolve_samples_path(args: argparse.Namespace) -> Path:
    if args.samples_file is not None:
        path = args.samples_file.resolve()
    else:
        path = (args.vqa_dir / "samples.jsonl").resolve()
    if not path.exists():
        raise FileNotFoundError(f"samples.jsonl does not exist: {path}")
    return path


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _iter_filtered_samples(records: Iterable[dict[str, Any]], *, sample_ids: set[str], limit: int | None) -> list[dict[str, Any]]:
    matched: list[dict[str, Any]] = []
    for record in records:
        sample_id = str(record.get("sample_id") or "")
        if sample_ids and sample_id not in sample_ids:
            continue
        matched.append(record)
        if limit is not None and len(matched) >= limit:
            break
    return matched


def _get_stage_question(sample: dict[str, Any], stage_name: str) -> str:
    stage = ((sample.get("writer_outputs") or {}).get(stage_name) or {})
    return str(stage.get("question") or "").strip()


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _format_wrapped_block(label: str, value: str, *, width: int, indent: int = 2) -> list[str]:
    prefix = " " * indent
    if not value:
        return [f"{prefix}{label}: -"]
    wrapped = textwrap.wrap(
        value,
        width=max(20, width - indent - 2),
        break_long_words=False,
        break_on_hyphens=False,
    )
    if not wrapped:
        return [f"{prefix}{label}: -"]
    lines = [f"{prefix}{label}: {wrapped[0]}"]
    continuation_prefix = " " * (indent + len(label) + 2)
    for segment in wrapped[1:]:
        lines.append(f"{continuation_prefix}{segment}")
    return lines


def _format_hop(hop: dict[str, Any], *, width: int) -> list[str]:
    hop_index = hop.get("hop_index")
    title = f"Hop {hop_index}" if hop_index is not None else "Hop"
    lines = [f"  {title}"]
    lines.extend(_format_wrapped_block("source", _first_non_empty(hop.get("source")), width=width, indent=4))
    lines.extend(_format_wrapped_block("relation", _first_non_empty(hop.get("relation")), width=width, indent=4))
    lines.extend(_format_wrapped_block("target", _first_non_empty(hop.get("target")), width=width, indent=4))
    lines.extend(_format_wrapped_block("statement", _first_non_empty(hop.get("statement")), width=width, indent=4))
    return lines


def _format_sample(sample: dict[str, Any], *, ordinal: int, width: int) -> str:
    sample_id = _first_non_empty(sample.get("sample_id"), f"sample_{ordinal:06d}")
    status = _first_non_empty(sample.get("status"), "unknown")
    path = sample.get("path") or {}
    path_id = _first_non_empty(path.get("path_id"))
    drafted_question = _first_non_empty(
        _get_stage_question(sample, "draft"),
        _get_stage_question(sample, "drafted"),
        sample.get("drafted_question"),
        sample.get("draft_question"),
    )
    enhanced_question = _first_non_empty(
        _get_stage_question(sample, "polished"),
        _get_stage_question(sample, "enhanced"),
        sample.get("enhanced_question"),
        sample.get("polished_question"),
    )
    final_question = _first_non_empty(
        _get_stage_question(sample, "obfuscated"),
        _get_stage_question(sample, "final"),
        sample.get("final_question"),
        enhanced_question,
        drafted_question,
    )
    hop_chain = list(sample.get("hop_chain") or [])

    lines: list[str] = [SEPARATOR]
    header = f"Sample {ordinal} | sample_id={sample_id} | status={status}"
    if path_id:
        header += f" | path_id={path_id}"
    lines.append(header)
    lines.append(SUB_SEPARATOR)
    lines.append("Question Writing Process")
    if hop_chain:
        for hop in hop_chain:
            lines.extend(_format_hop(hop, width=width))
            lines.append("")
        if lines[-1] == "":
            lines.pop()
    else:
        lines.append("  - No hop_chain found.")

    lines.append(SUB_SEPARATOR)
    lines.append("Question Versions")
    lines.extend(_format_wrapped_block("drafted_question", drafted_question, width=width, indent=2))
    lines.extend(_format_wrapped_block("enhanced_question", enhanced_question, width=width, indent=2))
    lines.extend(_format_wrapped_block("final_question", final_question, width=width, indent=2))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    samples_path = _resolve_samples_path(args)
    records = _load_jsonl(samples_path)
    sample_ids = {str(item) for item in args.sample_id if str(item).strip()}
    selected = _iter_filtered_samples(records, sample_ids=sample_ids, limit=args.limit)

    if not selected:
        if sample_ids:
            print(f"No samples matched sample_id filter in {samples_path}")
        else:
            print(f"No samples found in {samples_path}")
        return 1

    rendered = [
        _format_sample(sample, ordinal=index, width=args.width)
        for index, sample in enumerate(selected, start=1)
    ]
    print("\n\n".join(rendered))
    print(SEPARATOR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
