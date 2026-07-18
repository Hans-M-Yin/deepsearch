"""Pretty-print detailed VQA question-writing debug information.

Examples:
  python debug/debug_vqa_question_writing.py --vqa-dir synthesis/runs/my_graph/vqa/0716_123456
  python debug/debug_vqa_question_writing.py --vqa-dir synthesis/runs/my_graph/vqa/0716_123456 --limit 3
  python debug/debug_vqa_question_writing.py --vqa-dir synthesis/runs/my_graph/vqa/0716_123456 --sample-id sample_path_000123
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any, Iterable


SEPARATOR = "=" * 96
SUB_SEPARATOR = "-" * 96


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vqa-dir", type=Path, required=True, help="Directory containing samples.jsonl.")
    parser.add_argument(
        "--sample-id",
        action="append",
        default=[],
        help="Only print the given sample_id. Repeatable.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of samples to print.")
    parser.add_argument("--width", type=int, default=108, help="Wrap width for long text fields.")
    return parser


def _resolve_samples_path(vqa_dir: Path) -> Path:
    path = (vqa_dir / "samples.jsonl").resolve()
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


def _iter_filtered_samples(
    records: Iterable[dict[str, Any]],
    *,
    sample_ids: set[str],
    limit: int | None,
) -> list[dict[str, Any]]:
    matched: list[dict[str, Any]] = []
    for record in records:
        sample_id = str(record.get("sample_id") or "")
        if sample_ids and sample_id not in sample_ids:
            continue
        matched.append(record)
        if limit is not None and len(matched) >= limit:
            break
    return matched


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _wrap_value(value: Any, *, width: int) -> str:
    text = _first_non_empty(value)
    if not text:
        return "-"
    wrapped = textwrap.wrap(
        text,
        width=max(24, width),
        break_long_words=False,
        break_on_hyphens=False,
    )
    return "\n".join(wrapped) if wrapped else "-"


def _format_field(label: str, value: Any, *, width: int, indent: int = 2) -> list[str]:
    prefix = " " * indent
    wrapped = _wrap_value(value, width=width - indent - len(label) - 2)
    if "\n" not in wrapped:
        return [f"{prefix}{label}: {wrapped}"]
    lines = wrapped.splitlines()
    rendered = [f"{prefix}{label}: {lines[0]}"]
    continuation = " " * (indent + len(label) + 2)
    for line in lines[1:]:
        rendered.append(f"{continuation}{line}")
    return rendered


def _hop_index_map(hops: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for hop in hops:
        hop_index = hop.get("hop_index")
        if isinstance(hop_index, int) and hop_index not in result:
            result[hop_index] = hop
    return result


def _format_hop(hop: dict[str, Any], *, width: int, title: str) -> list[str]:
    lines = [f"  {title}"]
    lines.extend(_format_field("source", hop.get("source"), width=width, indent=4))
    lines.extend(_format_field("relation", hop.get("relation"), width=width, indent=4))
    lines.extend(_format_field("target", hop.get("target"), width=width, indent=4))
    lines.extend(_format_field("statement", hop.get("statement"), width=width, indent=4))
    retrieval_query = _first_non_empty(hop.get("retrieval_query"))
    if retrieval_query:
        lines.extend(_format_field("retrieval_query", retrieval_query, width=width, indent=4))
    return lines


def _stage_question(sample: dict[str, Any], stage_name: str) -> str:
    stage = ((sample.get("writer_outputs") or {}).get(stage_name) or {})
    return str(stage.get("question") or "").strip()


def _final_question(sample: dict[str, Any]) -> str:
    return _first_non_empty(
        _stage_question(sample, "obfuscated"),
        sample.get("final_question"),
        _stage_question(sample, "polished"),
        _stage_question(sample, "draft"),
    )


def _format_image_bridge_section(sample: dict[str, Any], *, width: int) -> list[str]:
    raw_hops = list(sample.get("hop_chain") or [])
    question_hops = list(sample.get("question_hop_chain") or [])
    raw_by_index = _hop_index_map(raw_hops)
    question_by_index = _hop_index_map(question_hops)
    diagnostics = list(sample.get("image_bridge_normalization") or [])

    lines = ["Text -> Image -> Text Merge"]
    if not diagnostics:
        lines.append("  - No image bridge normalization records.")
        return lines

    for ordinal, diag in enumerate(diagnostics, start=1):
        lines.append(f"  Merge {ordinal}")
        lines.extend(_format_field("applied", diag.get("applied"), width=width, indent=4))
        lines.extend(_format_field("decision", diag.get("decision"), width=width, indent=4))
        lines.extend(_format_field("reason", diag.get("reason"), width=width, indent=4))
        lines.extend(_format_field("image_node_id", diag.get("image_node_id"), width=width, indent=4))

        incoming_index = diag.get("incoming_hop_index")
        outgoing_index = diag.get("outgoing_hop_index")
        incoming_hop = raw_by_index.get(incoming_index) if isinstance(incoming_index, int) else None
        outgoing_hop = raw_by_index.get(outgoing_index) if isinstance(outgoing_index, int) else None

        if incoming_hop:
            lines.extend(_format_hop(incoming_hop, width=width, title=f"Before Hop {incoming_index}"))
        if outgoing_hop:
            lines.extend(_format_hop(outgoing_hop, width=width, title=f"Before Hop {outgoing_index}"))

        if diag.get("applied"):
            merged_hop = question_by_index.get(incoming_index) if isinstance(incoming_index, int) else None
            if merged_hop:
                lines.extend(_format_hop(merged_hop, width=width, title="After Merged Hop"))
            else:
                lines.append("  After Merged Hop")
                lines.extend(_format_field("statement", diag.get("rewritten_statement"), width=width, indent=4))
        lines.append("")

    if lines[-1] == "":
        lines.pop()
    return lines


def _format_terminal_merge_section(sample: dict[str, Any], *, width: int) -> list[str]:
    target_ask = sample.get("target_ask") or {}
    question_target_ask = sample.get("question_target_ask") or {}
    bridge = sample.get("question_terminal_bridge") or {}
    diag = sample.get("image_target_terminal_normalization") or {}

    lines = ["Text -> Image -> Target Merge"]
    if not diag and not bridge:
        lines.append("  - No terminal image merge record.")
        return lines

    lines.extend(_format_field("applied", diag.get("applied"), width=width, indent=2))
    lines.extend(_format_field("decision", diag.get("decision"), width=width, indent=2))
    lines.extend(_format_field("reason", diag.get("reason"), width=width, indent=2))

    removed_hop = bridge.get("removed_question_hop") or diag.get("removed_question_hop") or {}
    if isinstance(removed_hop, dict) and removed_hop:
        lines.extend(_format_hop(removed_hop, width=width, title="  Before Final Text -> Image Hop"))

    lines.extend(_format_field("before_raw_ask_target", target_ask.get("ask_target"), width=width, indent=2))
    lines.extend(_format_field("after_question_ask_target", question_target_ask.get("ask_target"), width=width, indent=2))
    if bridge:
        lines.extend(_format_field("bridge_replaces_terminal_hop", bridge.get("replaces_terminal_text_to_image_hop"), width=width, indent=2))
        lines.extend(_format_field("bridge_answer", bridge.get("answer"), width=width, indent=2))
    return lines


def _format_target_ask_section(sample: dict[str, Any], *, width: int) -> list[str]:
    raw_target_ask = sample.get("target_ask") or {}
    question_target_ask = sample.get("question_target_ask") or {}
    if not isinstance(raw_target_ask, dict):
        raw_target_ask = {}
    if not isinstance(question_target_ask, dict):
        question_target_ask = {}

    lines = ["Target Ask"]
    if not raw_target_ask and not question_target_ask:
        lines.append("  - No target ask record found.")
        return lines

    def append_target_ask(title: str, payload: dict[str, Any]) -> None:
        lines.append(f"  {title}")
        lines.extend(_format_field("ask_target", payload.get("ask_target"), width=width, indent=4))
        lines.extend(_format_field("answer", payload.get("answer"), width=width, indent=4))
        supporting_facts = payload.get("supporting_facts") or []
        if supporting_facts:
            lines.extend(
                _format_field(
                    "supporting_facts",
                    " | ".join(str(item) for item in supporting_facts),
                    width=width,
                    indent=4,
                )
            )
        if payload.get("reasoning"):
            lines.extend(_format_field("reasoning", payload.get("reasoning"), width=width, indent=4))
        if payload.get("support"):
            lines.extend(_format_field("support", payload.get("support"), width=width, indent=4))
        if payload.get("writer_warning"):
            lines.extend(_format_field("writer_warning", payload.get("writer_warning"), width=width, indent=4))

    if raw_target_ask:
        append_target_ask("Raw target ask", raw_target_ask)
    if question_target_ask and question_target_ask != raw_target_ask:
        append_target_ask("Question-facing target ask", question_target_ask)
    return lines


def _format_entry_hop_section(sample: dict[str, Any], *, width: int) -> list[str]:
    entry_hop = sample.get("entry_hop") or {}
    lines = ["Entry Hop"]
    if not isinstance(entry_hop, dict) or not entry_hop:
        lines.append("  - No entry hop record found.")
        return lines

    lines.extend(_format_field("entry_kind", entry_hop.get("entry_kind"), width=width, indent=2))
    lines.extend(_format_field("source", entry_hop.get("source"), width=width, indent=2))
    lines.extend(_format_field("target", entry_hop.get("target"), width=width, indent=2))
    lines.extend(_format_field("statement", entry_hop.get("statement"), width=width, indent=2))
    supporting_facts = entry_hop.get("supporting_facts") or []
    if supporting_facts:
        lines.extend(
            _format_field(
                "supporting_facts",
                " | ".join(str(item) for item in supporting_facts),
                width=width,
                indent=2,
            )
        )
    lines.extend(_format_field("why_relevant", entry_hop.get("why_relevant"), width=width, indent=2))
    if entry_hop.get("writer_warning"):
        lines.extend(_format_field("writer_warning", entry_hop.get("writer_warning"), width=width, indent=2))
    return lines

def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _format_polish_section(sample: dict[str, Any], *, width: int) -> list[str]:
    polish = sample.get("polish") or {}
    if not isinstance(polish, dict):
        polish = {}
    payload = polish.get("payload") or {}
    subtasks = polish.get("subtasks") or {}
    rewrite_payload = polish.get("rewrite_payload") or {}
    result = polish.get("result") or {}

    lines = ["Polish Process"]
    if not any((payload, subtasks, rewrite_payload, result, polish.get("rewrite_skipped") is not None)):
        lines.append("  - No persisted polish diagnostics. Samples produced before this change do not contain them.")
        return lines

    if isinstance(payload, dict):
        lines.append("  Input")
        lines.extend(_format_field("question", payload.get("question"), width=width, indent=4))
        for hop in payload.get("hops") or []:
            if isinstance(hop, dict):
                hop_index = hop.get("hop_index")
                title = f"Polish Hop {hop_index}" if hop_index is not None else "Polish Hop"
                lines.extend(_format_hop(hop, width=width, title=title))

    if isinstance(subtasks, dict) and subtasks:
        lines.append("  Subtask Diagnostics")
        for task_name, diagnostic in subtasks.items():
            if not isinstance(diagnostic, dict):
                continue
            lines.append(f"    {task_name}")
            lines.extend(_format_field("has_feedback", diagnostic.get("has_feedback"), width=width, indent=6))
            lines.extend(_format_field("issues", diagnostic.get("issues"), width=width, indent=6))
            lines.extend(_format_field("advice", diagnostic.get("advice"), width=width, indent=6))
            lines.extend(_format_field("image_attached", diagnostic.get("image_attached"), width=width, indent=6))
            input_payload = diagnostic.get("input_payload") or {}
            if isinstance(input_payload, dict) and input_payload:
                lines.extend(_format_field("input_payload", _json_text(input_payload), width=width, indent=6))
            raw = diagnostic.get("raw") or {}
            if isinstance(raw, dict) and raw:
                lines.extend(_format_field("raw_response", _json_text(raw), width=width, indent=6))

    lines.append("  Aggregate Rewrite")
    lines.extend(_format_field("rewrite_skipped", polish.get("rewrite_skipped"), width=width, indent=4))
    lines.extend(_format_field("skip_reason", polish.get("rewrite_skip_reason"), width=width, indent=4))
    if isinstance(rewrite_payload, dict) and rewrite_payload:
        lines.extend(_format_field("rewrite_input", _json_text(rewrite_payload), width=width, indent=4))
    if isinstance(result, dict) and result:
        lines.extend(_format_field("rewrite_result", _json_text(result), width=width, indent=4))
    return lines


def _format_questions_section(sample: dict[str, Any], *, width: int) -> list[str]:
    draft_question = _first_non_empty(_stage_question(sample, "draft"), sample.get("draft_question"))
    polished_question = _first_non_empty(_stage_question(sample, "polished"), sample.get("polished_question"))
    final_question = _final_question(sample)

    lines = ["Question Versions"]
    lines.extend(_format_field("drafted_question", draft_question, width=width, indent=2))
    lines.extend(_format_field("polished_question", polished_question, width=width, indent=2))
    lines.extend(_format_field("final_question", final_question, width=width, indent=2))
    return lines


def _format_sample(sample: dict[str, Any], *, ordinal: int, width: int) -> str:
    sample_id = _first_non_empty(sample.get("sample_id"), f"sample_{ordinal:06d}")
    status = _first_non_empty(sample.get("status"), "unknown")
    path = sample.get("path") or {}
    path_id = _first_non_empty(path.get("path_id"))

    lines: list[str] = [SEPARATOR]
    header = f"Sample {ordinal} | sample_id={sample_id} | status={status}"
    if path_id:
        header += f" | path_id={path_id}"
    lines.append(header)

    raw_hops = list(sample.get("hop_chain") or [])
    question_hops = list(sample.get("question_hop_chain") or [])

    lines.append(SUB_SEPARATOR)
    lines.append("Raw Hop Chain")
    if raw_hops:
        for hop in raw_hops:
            hop_index = hop.get("hop_index")
            title = f"Hop {hop_index}" if hop_index is not None else "Hop"
            lines.extend(_format_hop(hop, width=width, title=title))
            lines.append("")
        if lines[-1] == "":
            lines.pop()
    else:
        lines.append("  - No raw hop chain found.")

    lines.append(SUB_SEPARATOR)
    lines.extend(_format_image_bridge_section(sample, width=width))

    lines.append(SUB_SEPARATOR)
    lines.extend(_format_terminal_merge_section(sample, width=width))

    lines.append(SUB_SEPARATOR)
    lines.extend(_format_entry_hop_section(sample, width=width))

    lines.append(SUB_SEPARATOR)
    lines.extend(_format_target_ask_section(sample, width=width))

    lines.append(SUB_SEPARATOR)
    lines.extend(_format_polish_section(sample, width=width))

    lines.append(SUB_SEPARATOR)
    lines.append("Question-Facing Hop Chain")
    if question_hops:
        for hop in question_hops:
            hop_index = hop.get("hop_index")
            title = f"Hop {hop_index}" if hop_index is not None else "Hop"
            lines.extend(_format_hop(hop, width=width, title=title))
            lines.append("")
        if lines[-1] == "":
            lines.pop()
    else:
        lines.append("  - No question-facing hop chain found.")

    lines.append(SUB_SEPARATOR)
    lines.extend(_format_questions_section(sample, width=width))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    samples_path = _resolve_samples_path(args.vqa_dir)
    records = _load_jsonl(samples_path)
    sample_ids = {str(item).strip() for item in args.sample_id if str(item).strip()}
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
