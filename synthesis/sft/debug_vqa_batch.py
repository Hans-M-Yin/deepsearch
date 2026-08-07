"""Debug and inspect SFT trajectories over one question, VQA batch, or filtered list."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import traceback
from typing import Any

from synthesis.firecrawl_client import get_firecrawl_usage_snapshot

from .api_tools import RESPONSES_SYSTEM_PROMPT_V2

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional progress dependency
    tqdm = None

from .pipeline import (
    build_agent_config,
    build_runtime_context,
    check_hop_chain_coverage,
    extract_answer,
    format_messages,
    judge,
    run_agent_session,
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                records.append(parsed)
    return records


class _InitialWorkerStartGate:
    """Stagger only the first wave of worker tasks before they can call an LLM."""

    def __init__(self, *, worker_count: int, interval_s: float) -> None:
        self._initial_slots = max(0, int(worker_count))
        self._interval_s = max(0.0, float(interval_s))
        self._started_at = time.monotonic()
        self._issued_slots = 0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            if self._issued_slots >= self._initial_slots:
                return
            slot = self._issued_slots
            self._issued_slots += 1
            release_at = self._started_at + slot * self._interval_s
        delay_s = release_at - time.monotonic()
        if delay_s > 0:
            time.sleep(delay_s)


def _extract_image_urls_from_vqa_records(
    question_record: dict[str, Any],
    sample_record: dict[str, Any] | None,
) -> list[str]:
    candidates: list[Any] = [
        question_record.get("image_url"),
        question_record.get("input_image_url"),
    ]
    sample_record = sample_record or {}
    candidates.extend(
        [
            sample_record.get("input_image_url"),
            ((sample_record.get("metadata") or {}).get("input_image_url") if isinstance(sample_record.get("metadata"), dict) else None),
        ]
    )

    writer_outputs = sample_record.get("writer_outputs") or {}
    if isinstance(writer_outputs, dict):
        for stage_name in ("obfuscated", "polished", "draft"):
            stage = writer_outputs.get(stage_name) or {}
            stage_metadata = stage.get("metadata") or {}
            if isinstance(stage_metadata, dict):
                candidates.extend(
                    [
                        stage_metadata.get("starting_image_url"),
                        stage_metadata.get("polish_starting_image_url"),
                    ]
                )

    normalized: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        value = str(candidate or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def _load_vqa_records(vqa_dir: Path) -> list[dict[str, Any]]:
    questions_path = vqa_dir / "questions.jsonl"
    samples_path = vqa_dir / "samples.jsonl"
    if not questions_path.exists():
        raise FileNotFoundError(f"questions.jsonl does not exist: {questions_path}")
    if not samples_path.exists():
        raise FileNotFoundError(f"samples.jsonl does not exist: {samples_path}")

    question_records = _load_jsonl(questions_path)
    sample_records = _load_jsonl(samples_path)
    samples_by_id = {
        str(record.get("sample_id")): record
        for record in sample_records
        if record.get("sample_id") is not None
    }

    merged_records: list[dict[str, Any]] = []
    for question_record in question_records:
        sample = samples_by_id.get(str(question_record.get("sample_id") or ""))
        merged_records.append(
            {
                "question_id": question_record.get("question_id"),
                "sample_id": question_record.get("sample_id"),
                "path_id": question_record.get("path_id"),
                "question": question_record.get("final_question") or question_record.get("question") or "",
                "gold_answer": question_record.get("answer") or "",
                "hop_chain": list((sample or {}).get("hop_chain") or []),
                "image_paths": [],
                "image_urls": _extract_image_urls_from_vqa_records(question_record, sample),
                "sample_record": sample or {},
                "question_record": question_record,
            }
        )
    return merged_records


def _filtered_input_images(record: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Extract the original input image paths/URLs from a filtered trajectory."""

    image_paths: list[str] = []
    image_urls: list[str] = []
    seen_paths: set[str] = set()
    seen_urls: set[str] = set()

    def add_path(value: Any) -> None:
        normalized = str(value or "").strip()
        if normalized and normalized not in seen_paths:
            seen_paths.add(normalized)
            image_paths.append(normalized)

    def add_url(value: Any) -> None:
        normalized = str(value or "").strip()
        if normalized and normalized not in seen_urls:
            seen_urls.add(normalized)
            image_urls.append(normalized)

    for item in record.get("input_images") or []:
        if isinstance(item, dict):
            add_path(item.get("image_path") or item.get("local_path"))
            add_url(item.get("image_url") or item.get("source_url"))
        elif isinstance(item, str):
            value = item.strip()
            if value.startswith(("http://", "https://")):
                add_url(value)
            else:
                add_path(value)

    source_metadata = record.get("source_metadata") or {}
    question_record = source_metadata.get("question_record") if isinstance(source_metadata, dict) else {}
    sample_record = source_metadata.get("sample_record") if isinstance(source_metadata, dict) else {}
    if not isinstance(question_record, dict):
        question_record = {}
    if not isinstance(sample_record, dict):
        sample_record = {}
    sample_metadata = sample_record.get("metadata") if isinstance(sample_record.get("metadata"), dict) else {}
    if not image_urls:
        for candidate in (
            record.get("input_image_url"),
            question_record.get("input_image_url"),
            question_record.get("image_url"),
            sample_record.get("input_image_url"),
            sample_metadata.get("input_image_url"),
        ):
            add_url(candidate)
    return image_paths, image_urls


def _load_filtered_trajectory_records(path: Path) -> list[dict[str, Any]]:
    """Load complete records emitted by filter_sft_trajectories.py.

    The filtered JSONL already contains the question, answer, hop chain and
    original input images, so no questions.jsonl/samples.jsonl join is needed.
    """

    if not path.exists():
        raise FileNotFoundError(f"filtered trajectory JSONL does not exist: {path}")
    records: list[dict[str, Any]] = []
    for source_index, source in enumerate(_load_jsonl(path)):
        source_metadata = source.get("source_metadata") or {}
        question_record = source_metadata.get("question_record") if isinstance(source_metadata, dict) else {}
        sample_record = source_metadata.get("sample_record") if isinstance(source_metadata, dict) else {}
        if not isinstance(question_record, dict):
            question_record = {}
        if not isinstance(sample_record, dict):
            sample_record = {}
        question = str(
            source.get("question")
            or question_record.get("final_question")
            or question_record.get("question")
            or ""
        ).strip()
        if not question:
            raise ValueError(f"filtered trajectory at source index {source_index} has no question")
        gold_answer = str(
            source.get("gold_answer")
            or question_record.get("answer")
            or ""
        ).strip()
        image_paths, image_urls = _filtered_input_images(source)
        records.append(
            {
                "question_id": source.get("question_id") or question_record.get("question_id"),
                "sample_id": source.get("sample_id") or question_record.get("sample_id") or sample_record.get("sample_id"),
                "path_id": source.get("path_id") or question_record.get("path_id"),
                "question": question,
                "gold_answer": gold_answer,
                "hop_chain": list(source.get("hop_chain") or sample_record.get("hop_chain") or []),
                "image_paths": image_paths,
                "image_urls": image_urls,
                "question_record": question_record,
                "sample_record": sample_record,
                "source_filter": source.get("sft_trajectory_filter") or {},
            }
        )
    return records


def _slice_batch_records(records: list[dict[str, Any]], *, offset: int, limit: int) -> list[dict[str, Any]]:
    if offset < 0 or limit < 0:
        raise ValueError("offset and limit must be non-negative")
    end = None if limit == 0 else offset + limit
    return records[offset:end]


def _single_question_record(
    *,
    question: str,
    gold_answer: str,
    hop_chain_json: str | None,
    image_paths: list[str] | None = None,
    image_urls: list[str] | None = None,
) -> list[dict[str, Any]]:
    hop_chain = json.loads(hop_chain_json) if hop_chain_json else []
    if not isinstance(hop_chain, list):
        raise ValueError("--hop-chain-json must decode to a JSON list.")
    return [
        {
            "question_id": "single_question",
            "sample_id": None,
            "path_id": None,
            "question": question,
            "gold_answer": gold_answer,
            "hop_chain": hop_chain,
            "image_paths": list(image_paths or []),
            "image_urls": list(image_urls or []),
            "sample_record": {},
            "question_record": {
                "question": question,
                "answer": gold_answer,
            },
        }
    ]


def _print_record_result(result: dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    print(f"question_id: {result.get('question_id')}")
    if result.get("sample_id") is not None:
        print(f"sample_id: {result.get('sample_id')}")
    if result.get("path_id") is not None:
        print(f"path_id: {result.get('path_id')}")
    print(f"question: {result.get('question')}")
    print(f"gold_answer: {result.get('gold_answer')}")
    if result.get("input_images"):
        print("input_images:")
        print(json.dumps(result.get("input_images") or [], ensure_ascii=False, indent=2))
    if result.get("generation_summary"):
        summary = result.get("generation_summary") or {}
        print(
            "generation: "
            f"status={summary.get('generation_status')} "
            f"complete={summary.get('generation_complete')} "
            f"stop_reason={summary.get('stop_reason')} "
            f"turns={summary.get('turn_count')} "
            f"tools={summary.get('tool_call_count')}"
        )
        if summary.get("failure_reasons"):
            print(f"generation_failure_reasons: {summary.get('failure_reasons')}")
        if summary.get("tool_error_counts"):
            print("tool_error_counts:")
            print(json.dumps(summary.get("tool_error_counts") or {}, ensure_ascii=False, indent=2))
    print(f"extracted_answer: {result.get('extracted_answer')}")
    print("answer_judge:")
    print(json.dumps(result.get("answer_judge") or {}, ensure_ascii=False, indent=2))
    if result.get("hop_chain"):
        print("hop_chain_coverage:")
        print(json.dumps(result.get("hop_chain_coverage") or {}, ensure_ascii=False, indent=2))
    print("\n--- Trajectory Text ---")
    print((result.get("formatted_trajectory") or {}).get("text") or "")


_SFT_FSYNC_WARNING_EMITTED = False
_SFT_JSONL_FLUSH_EVERY = 50


def _write_jsonl_records(handle: Any, records: list[dict[str, Any]]) -> None:
    """Append a batch of records and publish it for HDFS-FUSE visibility."""
    if not records:
        return
    if isinstance(handle, (str, Path)):
        with Path(handle).open("a", encoding="utf-8") as opened_handle:
            _write_jsonl_records(opened_handle, records)
        return
    for record in records:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()
    try:
        os.fsync(handle.fileno())
    except OSError as exc:
        global _SFT_FSYNC_WARNING_EMITTED
        if not _SFT_FSYNC_WARNING_EMITTED:
            _SFT_FSYNC_WARNING_EMITTED = True
            print(
                f"[sft-output] fsync is unavailable; relying on file close for publication: {exc}",
                file=sys.stderr,
                flush=True,
            )


def _write_jsonl_record(handle: Any, record: dict[str, Any]) -> None:
    """Append one record while keeping the single-record helper API."""
    _write_jsonl_records(handle, [record])


def _usage_delta(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    return {
        key: max(0, int(after.get(key, 0)) - int(before.get(key, 0)))
        for key in sorted(set(before) | set(after))
    }


def _resume_record_key(record: dict[str, Any]) -> tuple[str, str, str]:
    """Return the stable VQA identity used to avoid duplicate trajectories."""
    return tuple(
        str(record.get(field) or "").strip()
        for field in ("question_id", "sample_id", "path_id")
    )


def _load_completed_resume_keys(path: Path) -> set[tuple[str, str, str]]:
    """Read completed trajectory identities, tolerating a partial final JSONL line."""
    completed: set[tuple[str, str, str]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"[sft-resume] ignoring malformed existing JSONL record "
                    f"at {path}:{line_number}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            if not isinstance(parsed, dict):
                continue
            key = _resume_record_key(parsed)
            if any(key):
                completed.add(key)
    return completed


def _message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, indent=2)


def _message_text_for_transcript(message: dict[str, Any]) -> str:
    role = str(message.get("role") or "")
    content = _message_content_to_text(message.get("content")).strip()
    if role == "tool":
        tool_name = str(message.get("name") or "").strip()
        if tool_name:
            return f"[{tool_name}]\n{content}" if content else f"[{tool_name}]"
    return content


def _try_parse_json_object(text: Any) -> dict[str, Any] | None:
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return None
    candidate = text.strip()
    if not candidate:
        return None
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(candidate[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None


def _normalize_error_reason(text: str) -> str:
    reason = str(text or "").strip()
    if not reason:
        return "unknown_error"
    lowered = reason.lower()
    if "max react turns" in lowered or "max tool-calling turns" in lowered:
        return "max_turns_reached"
    if "query is required" in lowered:
        return "missing_query"
    if "url is required" in lowered:
        return "missing_url"
    if "no image is available" in lowered:
        return "i2i_no_image_available"
    if "cropped region was created" in lowered and "public url" in lowered:
        return "i2i_crop_upload_failed"
    if "requires a publicly reachable image url" in lowered:
        return "image_upload_failed"
    if "http_403" in lowered or "403" in lowered or "forbidden" in lowered:
        return "http_403"
    if "http_404" in lowered or "404" in lowered or "not found" in lowered:
        return "http_404"
    if "http_429" in lowered or "429" in lowered or "too many requests" in lowered:
        return "http_429"
    if "timeout" in lowered:
        return "timeout"
    if "decode_error" in lowered:
        return "image_decode_error"
    if "non_image_content_type" in lowered:
        return "non_image_content_type"
    if "read_url failed" in lowered:
        return "read_url_failed"
    if "i2i_search failed" in lowered:
        return "i2i_search_failed"
    normalized = re.sub(r"https?://\S+", "<url>", reason)
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    normalized = re.sub(r"[^a-z0-9_ .:-]+", "", normalized)
    return normalized[:120] or "unknown_error"


def _summarize_generation(
    *,
    messages: list[dict[str, Any]],
    generation_metadata: dict[str, Any] | None,
    extracted_answer: str,
) -> dict[str, Any]:
    metadata = dict(generation_metadata or {})
    status = str(metadata.get("generation_status") or "unknown")
    stop_reason = str(metadata.get("stop_reason") or "unknown")
    tool_error_counts: Counter[str] = Counter()
    tool_success_counts: Counter[str] = Counter()
    tool_error_reasons: Counter[str] = Counter()
    tool_call_count = 0
    assistant_turn_count = 0
    tool_turn_count = 0
    total_content_chars = 0

    for message in messages:
        role = str(message.get("role") or "")
        content_text = _message_content_to_text(message.get("content"))
        total_content_chars += len(content_text)
        if role == "assistant":
            assistant_turn_count += 1
        if role != "tool":
            continue
        tool_turn_count += 1
        tool_call_count += 1
        tool_name = str(message.get("name") or "unknown_tool").strip() or "unknown_tool"
        parsed = _try_parse_json_object(message.get("content"))
        ok_value = parsed.get("ok") if isinstance(parsed, dict) else None
        if ok_value is False:
            error_text = str((parsed or {}).get("error") or "tool_returned_ok_false")
            reason = _normalize_error_reason(error_text)
            tool_error_counts[tool_name] += 1
            tool_error_reasons[f"{tool_name}:{reason}"] += 1
        else:
            tool_success_counts[tool_name] += 1

    complete = bool(metadata.get("generation_complete"))
    if status == "finished":
        complete = True
    if status == "max_turns_reached":
        complete = False
    failure_reasons: list[str] = []
    if status == "max_turns_reached":
        failure_reasons.append("max_turns_reached")
    if status == "parse_error_finalized":
        failure_reasons.append("manual_react_parse_error")
    if not str(extracted_answer or "").strip():
        failure_reasons.append("empty_extracted_answer")

    summary = {
        "generation_status": status,
        "generation_complete": complete,
        "stop_reason": stop_reason,
        "failure_reasons": failure_reasons,
        "max_turns": metadata.get("max_turns"),
        "turn_count": metadata.get("turn_count"),
        "assistant_turn_count": assistant_turn_count,
        "tool_turn_count": tool_turn_count,
        "tool_call_count": metadata.get("tool_call_count", tool_call_count),
        "total_content_chars": total_content_chars,
        "tool_success_counts": dict(sorted(tool_success_counts.items())),
        "tool_error_counts": dict(sorted(tool_error_counts.items())),
        "tool_error_reasons": dict(sorted(tool_error_reasons.items())),
    }
    # #### START Response 0720 ####
    for key in (
        "responses_public_reasoning_prompted",
        "responses_i2i_wrapper_enabled",
        "responses_rationale_summary",
        "responses_turn_traces",
        "responses_raw_response_count",
    ):
        if key in metadata:
            summary[key] = metadata[key]
    # #### END Response 0720 ####
    return summary


def _merge_generation_stats(stats: dict[str, Counter[str]], generation_summary: dict[str, Any]) -> None:
    status = str(generation_summary.get("generation_status") or "unknown")
    stats["generation_status"][status] += 1
    if not bool(generation_summary.get("generation_complete")):
        stats["incomplete"][status] += 1
    for reason in generation_summary.get("failure_reasons") or []:
        stats["failure_reasons"][str(reason)] += 1
    for tool_name, count in (generation_summary.get("tool_success_counts") or {}).items():
        stats["tool_success_counts"][str(tool_name)] += int(count)
    for tool_name, count in (generation_summary.get("tool_error_counts") or {}).items():
        stats["tool_error_counts"][str(tool_name)] += int(count)
    for reason, count in (generation_summary.get("tool_error_reasons") or {}).items():
        stats["tool_error_reasons"][str(reason)] += int(count)


def _print_generation_stats(stats: dict[str, Counter[str]]) -> None:
    print("\n--- SFT Generation Stats ---")
    for title, key in (
        ("generation_status", "generation_status"),
        ("incomplete", "incomplete"),
        ("failure_reasons", "failure_reasons"),
        ("tool_success_counts", "tool_success_counts"),
        ("tool_error_counts", "tool_error_counts"),
        ("tool_error_reasons", "tool_error_reasons"),
    ):
        counter = stats[key]
        if not counter:
            continue
        print(f"{title}:")
        for name, count in counter.most_common():
            print(f"  {name}: {count}")


def _select_fields(source: dict[str, Any], field_names: tuple[str, ...]) -> dict[str, Any]:
    return {field_name: source.get(field_name) for field_name in field_names if source.get(field_name) is not None}


def _compact_path_record(path: dict[str, Any]) -> dict[str, Any]:
    return _select_fields(
        path,
        (
            "path_id",
            "start_node_id",
            "target_node_id",
            "node_ids",
            "edge_ids",
            "node_types",
            "edge_types",
            "relations",
            "trajectory",
            "exact_signature",
            "skeleton_signature",
            "core_signature",
        ),
    )


def _compact_question_record(question_record: dict[str, Any]) -> dict[str, Any]:
    return _select_fields(
        question_record,
        (
            "question_id",
            "sample_id",
            "path_id",
            "status",
            "question",
            "draft_question",
            "polished_question",
            "final_question",
            "answer",
            "image_url",
            "input_image_url",
        ),
    )


def _compact_sample_record(sample_record: dict[str, Any]) -> dict[str, Any]:
    compact = _select_fields(
        sample_record,
        (
            "sample_id",
            "status",
            "hop_chain",
            "question_hop_chain",
            "entry_hop",
            "target_ask",
            "question_target_ask",
            "question_terminal_bridge",
            "image_bridge_normalization",
            "image_target_terminal_normalization",
            "input_image_url",
            "created_at",
            "updated_at",
        ),
    )
    path = sample_record.get("path") or {}
    if isinstance(path, dict):
        compact["path"] = _compact_path_record(path)
    metadata = sample_record.get("metadata") or {}
    if isinstance(metadata, dict):
        compact_metadata = _select_fields(metadata, ("input_image_url", "writer_warnings", "timings"))
        if compact_metadata:
            compact["metadata"] = compact_metadata
    return compact


def _build_source_metadata(
    record: dict[str, Any],
    *,
    vqa_dir: str | None,
    mode: str,
) -> dict[str, Any]:
    base = {
        "mode": mode,
        "vqa_dir": vqa_dir,
        "question_id": record.get("question_id"),
        "sample_id": record.get("sample_id"),
        "path_id": record.get("path_id"),
    }
    if mode == "ref":
        return base

    question_record = record.get("question_record") or {}
    sample_record = record.get("sample_record") or {}
    if mode == "compact":
        return {
            **base,
            "question_record": _compact_question_record(question_record) if isinstance(question_record, dict) else {},
            "sample_record": _compact_sample_record(sample_record) if isinstance(sample_record, dict) else {},
        }
    return {
        **base,
        "question_record": question_record,
        "sample_record": sample_record,
    }


def _serialize_runtime(context: Any) -> dict[str, Any]:
    """Return the JSON-safe, provenance-relevant part of a tool runtime."""
    resources: list[dict[str, Any]] = []
    for resource_id, resource in sorted((context.resource_registry or {}).items()):
        resources.append(
            {
                "resource_id": resource.resource_id or resource_id,
                "result_id": resource.result_id,
                "kind": resource.kind,
                "primary_url": resource.primary_url,
                "image_url": resource.image_url,
                "thumbnail_url": resource.thumbnail_url,
                "source_page_url": resource.source_page_url,
                "fallback_urls": list(resource.fallback_urls or []),
                "title": resource.title,
                "snippet": resource.snippet,
                "search_tool": resource.search_tool,
                "search_query": resource.search_query,
                "rank": resource.rank,
                "url_keywords": resource.url_keywords,
            }
        )
    return {
        "session_id": context.session_id,
        "case_id": context.case_id,
        "working_dir": context.working_dir,
        "url_resources": resources,
    }


def _build_raw_trajectory_record(
    *,
    record: dict[str, Any],
    input_images: list[dict[str, str]],
    messages: list[dict[str, Any]],
    extracted_answer: str,
    answer_judge: dict[str, Any],
    generation_summary: dict[str, Any],
    hop_chain_coverage: dict[str, Any] | None,
    vqa_dir: str | None,
    source_metadata_mode: str,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    return {
        "question_id": record.get("question_id"),
        "sample_id": record.get("sample_id"),
        "path_id": record.get("path_id"),
        "question": record.get("question"),
        "gold_answer": record.get("gold_answer"),
        "input_images": input_images,
        "source_metadata": _build_source_metadata(record, vqa_dir=vqa_dir, mode=source_metadata_mode),
        "runtime": runtime,
        "raw_messages": messages,
        "extracted_answer": extracted_answer,
        "answer_judge": answer_judge,
        "generation_summary": generation_summary,
        "hop_chain": list(record.get("hop_chain") or []),
        "hop_chain_coverage": hop_chain_coverage,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vqa-dir", help="Directory produced by synthesis.vqa.run_batch.")
    parser.add_argument(
        "--question-list",
        "--filtered-trajectories-jsonl",
        dest="question_list",
        help="Complete filtered_trajectories.jsonl to regenerate only those questions.",
    )
    parser.add_argument("--question", help="Single question to debug.")
    parser.add_argument("--gold-answer", default="", help="Gold answer for single-question mode.")
    parser.add_argument("--hop-chain-json", help="JSON list for single-question hop chain.")
    parser.add_argument("--image", action="append", help="Attach a local image path to the user input.")
    parser.add_argument("--image-url", action="append", help="Attach a remote image URL to the user input.")
    parser.add_argument("--limit", type=int, default=5, help="How many questions to run in batch mode.")
    parser.add_argument("--offset", type=int, default=0, help="Start offset in batch mode.")
    parser.add_argument("--workers", type=int, default=1, help="How many records to process concurrently.")
    parser.add_argument(
        "--worker-start-stagger-s",
        type=float,
        default=10.0,
        help=(
            "Delay only the initial worker wave by this many seconds per task before its first "
            "LLM call. Defaults to 10; set to 0 to disable."
        ),
    )
    parser.add_argument("--workdir", default=os.path.join(os.getcwd(), "synthesis_sft_runs"))
    parser.add_argument("--output-jsonl", help="Optional path to save raw trajectory records.")
    parser.add_argument(
        "--raw-trajectories-jsonl",
        help="Optional path to save raw formatted trajectories.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an existing batch output: skip source records already present in the explicit "
            "--output-jsonl / --raw-trajectories-jsonl file, then append new trajectories."
        ),
    )
    parser.add_argument(
        "--source-metadata-mode",
        choices=("full", "compact", "ref"),
        default=os.environ.get("SFT_SOURCE_METADATA_MODE") or "compact",
        help=(
            "How much original VQA source metadata to store in each SFT JSONL record. "
            "full stores full question/sample records; compact stores only verifier/export-relevant fields; "
            "ref stores only ids and vqa_dir. Defaults to compact."
        ),
    )
    parser.add_argument(
        "--repair-model",
        default=os.environ.get("SFT_REPAIR_MODEL") or "",
        help="Registered model alias for incorrect-trajectory diagnosis and repair with LLM_WORKER.",
    )
    parser.add_argument(
        "--repair-max-tokens",
        type=int,
        default=_optional_env_int("SFT_REPAIR_MAX_TOKENS") or 2048,
        help="Max tokens for the LLM_WORKER-based incorrect-trajectory repair stages.",
    )
    parser.add_argument("--verbose", action="store_true")

    # #### START Response 0720 ####
    parser.add_argument(
        "--model-alias",
        "--model",
        dest="model_alias",
        default=os.environ.get("SFT_OPENAI_MODEL") or os.environ.get("OPENAI_MODEL") or "",
        help="Primary answer model alias from synthesis/models.json.",
    )
    # #### END Response 0720 ####
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument(
        "--api-mode",
        choices=("manual_react", "chat_completions", "responses"),
        default=os.environ.get("SFT_OPENAI_API_MODE") or "manual_react",
        help="Primary trajectory collection mode. Defaults to manual_react.",
    )
    # #### START Response 0720 ####
    parser.add_argument(
        "--client-type",
        choices=("azure_openai", "openai"),
        default=os.environ.get("SFT_OPENAI_CLIENT_TYPE") or "azure_openai",
        help="OpenAI SDK client type. Use openai for standard/base_url Responses endpoints.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("SFT_OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL"),
        help="Optional base URL for --client-type openai.",
    )
    parser.add_argument("--responses-reasoning-effort", default=os.environ.get("SFT_RESPONSES_REASONING_EFFORT"))
    parser.add_argument("--responses-reasoning-summary", default=os.environ.get("SFT_RESPONSES_REASONING_SUMMARY", "auto"))
    parser.add_argument("--responses-reasoning-mode", default=os.environ.get("SFT_RESPONSES_REASONING_MODE"))
    parser.add_argument("--responses-reasoning-context", default=os.environ.get("SFT_RESPONSES_REASONING_CONTEXT", "all_turns"))
    parser.add_argument("--responses-store", choices=("true", "false"), default=os.environ.get("SFT_RESPONSES_STORE"))
    parser.add_argument("--no-responses-public-reasoning", action="store_true", help="Do not append the Responses public-reasoning prompt.")
    parser.add_argument(
        "--responses-system-prompt-v2",
        action="store_true",
        help=(
            "Use api_tools.RESPONSES_SYSTEM_PROMPT_V2 for the primary agent. "
            "Only valid with --api-mode responses and cannot be combined with --system-prompt."
        ),
    )
    parser.add_argument("--responses-parallel-tool-calls", action="store_true", help="Allow parallel Responses tool calls. Defaults to false.")
    parser.add_argument("--responses-i2i-wrapper", action="store_true", help="Enable the legacy i2i wrapper rewrite in Responses mode.")
    # #### END Response 0720 ####
    parser.add_argument(
        "--azure-endpoint",
        default=(
            os.environ.get("SFT_OPENAI_AZURE_ENDPOINT")
            or os.environ.get("SFT_OPENAI_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
        ),
    )
    parser.add_argument("--api-version", default=os.environ.get("SFT_OPENAI_API_VERSION") or "2024-03-01-preview")
    parser.add_argument("--max-tokens", type=int, default=_optional_env_int("SFT_OPENAI_MAX_TOKENS"))
    parser.add_argument(
        "--temperature",
        type=float,
        default=(float(os.environ["SFT_OPENAI_TEMPERATURE"]) if os.environ.get("SFT_OPENAI_TEMPERATURE") else None),
    )
    parser.add_argument("--max-turns", type=int, default=int(os.environ.get("SFT_OPENAI_MAX_TURNS", "8")))
    parser.add_argument("--timeout-s", type=float, default=float(os.environ.get("SFT_OPENAI_TIMEOUT_S", "120")))
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--headers-json", default=os.environ.get("SFT_OPENAI_HEADERS_JSON"))
    parser.add_argument("--extra-body-json", default=os.environ.get("SFT_OPENAI_EXTRA_BODY_JSON"))
    parser.add_argument(
        "--expert-model",
        default=os.environ.get("SFT_JUDGE_MODEL"),
        help="Expert judge model. Prefer a registered alias from synthesis/models.json.",
    )
    parser.add_argument("--expert-api-key", default=os.environ.get("SFT_JUDGE_API_KEY"))
    parser.add_argument("--expert-azure-endpoint", default=os.environ.get("SFT_JUDGE_AZURE_ENDPOINT"))
    parser.add_argument("--expert-api-version", default=os.environ.get("SFT_JUDGE_API_VERSION") or os.environ.get("SFT_OPENAI_API_VERSION") or "2024-03-01-preview")
    parser.add_argument("--expert-max-tokens", type=int, default=_optional_env_int("SFT_JUDGE_MAX_TOKENS"))
    parser.add_argument(
        "--expert-temperature",
        type=float,
        default=(float(os.environ["SFT_JUDGE_TEMPERATURE"]) if os.environ.get("SFT_JUDGE_TEMPERATURE") else None),
    )
    return parser


def _parse_json_flag(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object.")
    return parsed


def _config_from_model_arg(
    *,
    model_arg: str | None,
    api_key: str | None,
    client_type: str,
    api_mode: str,
    azure_endpoint: str | None,
    base_url: str | None,
    api_version: str | None,
    max_tokens: int | None,
    temperature: float | None,
    timeout_s: float,
    system_prompt: str | None,
    headers_json: str | None,
    extra_body_json: str | None,
    max_turns: int,
    print_rounds: bool,
    # #### START Response 0720 ####
    responses_reasoning_effort: str | None = None,
    responses_reasoning_summary: str | None = None,
    responses_reasoning_mode: str | None = None,
    responses_reasoning_context: str | None = None,
    responses_store: str | None = None,
    responses_prompt_public_reasoning: bool = True,
    responses_parallel_tool_calls: bool = False,
    responses_i2i_wrapper_enabled: bool = False,
    # #### END Response 0720 ####
) -> Any:
    return build_agent_config(
        model=model_arg,
        api_key=api_key,
        client_type=client_type,
        azure_endpoint=azure_endpoint,
        base_url=base_url,
        api_version=api_version,
        api_mode=api_mode,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout_s=timeout_s,
        system_prompt=system_prompt,
        headers=_parse_json_flag(headers_json),
        extra_body=_parse_json_flag(extra_body_json),
        max_turns=max_turns,
        print_rounds=print_rounds,
        # #### START Response 0720 ####
        responses_reasoning_effort=responses_reasoning_effort,
        responses_reasoning_summary=responses_reasoning_summary,
        responses_reasoning_mode=responses_reasoning_mode,
        responses_reasoning_context=responses_reasoning_context,
        responses_store=(None if responses_store is None else responses_store == "true"),
        responses_prompt_public_reasoning=responses_prompt_public_reasoning,
        responses_parallel_tool_calls=responses_parallel_tool_calls,
        responses_i2i_wrapper_enabled=responses_i2i_wrapper_enabled,
        # #### END Response 0720 ####
    )


def format_hop_chain_for_user_prompt(question_sample: dict[str, Any]) -> str:
    """Format the sample hop chain block appended to the SFT user prompt.

    This intentionally preserves the existing SFT-generation behavior: use the
    sample's raw ``hop_chain`` field and expose only each hop's ``statement`` as
    intermediate verification facts.  Keeping this as a separate interface makes
    it easier to swap in a question-facing hop chain later without changing the
    rest of prompt construction.
    """

    statements = [
        str(hop.get("statement") or "").strip()
        for hop in (question_sample.get("hop_chain") or [])
        if isinstance(hop, dict) and str(hop.get("statement") or "").strip()
    ]
    if not statements:
        return ""
    statements_lines = "\n".join(statements)
    return (
        "\nPrivate reference facts for verification only:\n"
        "The following facts describe one possible reasoning route used when constructing the question. "
        "They are not necessarily the only or best route. Do not reveal them directly or cite them as evidence. "
        "Use them only to check whether your tool-based solution is on the right track.\n"
        f"{statements_lines}"
    )


def _build_user_prompt_text(record: dict[str, Any]) -> str:
    question_text = str(record.get("question") or "").strip()
    gold_answer = str(record.get("gold_answer") or "").strip()
    statements_block = format_hop_chain_for_user_prompt(record)
    if gold_answer:
        return f"Question: {question_text}\nAnswer: {gold_answer}{statements_block}"
    return f"{question_text}{statements_block}"


def _build_user_messages(record: dict[str, Any]) -> list[dict[str, Any]] | None:
    image_paths = [str(item).strip() for item in (record.get("image_paths") or []) if str(item).strip()]
    image_urls = [str(item).strip() for item in (record.get("image_urls") or []) if str(item).strip()]
    if not image_paths and not image_urls:
        return None

    content: list[dict[str, Any]] = [{"type": "text", "text": _build_user_prompt_text(record)}]
    for path in image_paths:
        content.append({"type": "image_path", "path": path})
    for url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})
    return [{"role": "user", "content": content}]


def _optional_env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return None
    return int(value)


def _process_record(
    *,
    index: int,
    record: dict[str, Any],
    agent_config: Any,
    expert_config: Any,
    workdir: str,
    vqa_dir: str | None,
    source_metadata_mode: str,
) -> dict[str, Any]:
    context = build_runtime_context(
        working_dir=os.path.join(workdir, f"debug_{index:04d}_{record.get('question_id') or 'question'}"),
        case_id=str(record.get("question_id") or f"debug_{index:04d}"),
        metadata={
            "question_id": record.get("question_id"),
            "sample_id": record.get("sample_id"),
            "path_id": record.get("path_id"),
            "question": record.get("question"),
            "gold_answer": record.get("gold_answer"),
            "hop_chain": list(record.get("hop_chain") or []),
        },
    )
    input_images: list[dict[str, str]] = []
    for image_path in record.get("image_paths") or []:
        normalized_path = os.path.abspath(str(image_path))
        context.register_image(normalized_path)
        input_images.append({"image_path": normalized_path})
    for image_url in record.get("image_urls") or []:
        normalized_url = str(image_url).strip()
        if normalized_url:
            context.register_image(normalized_url)
            input_images.append({"image_url": normalized_url})

    input_messages = _build_user_messages(record)
    agent_result = run_agent_session(
        prompt=None if input_messages is not None else _build_user_prompt_text(record),
        messages=input_messages,
        config=agent_config,
        context=context,
    )
    messages = agent_result.messages
    extracted_answer = extract_answer(messages)
    generation_summary = _summarize_generation(
        messages=messages,
        generation_metadata=agent_result.metadata,
        extracted_answer=extracted_answer,
    )
    answer_judge = judge(
        question=str(record.get("question") or ""),
        answer=str(record.get("gold_answer") or ""),
        extracted_answer=extracted_answer,
    )
    formatted_trajectory = format_messages(messages)
    hop_chain = list(record.get("hop_chain") or [])
    hop_chain_coverage = (
        check_hop_chain_coverage(messages, hop_chain, config=expert_config)
        if hop_chain and expert_config is not None
        else None
    )

    result_record = {
        "question_id": record.get("question_id"),
        "sample_id": record.get("sample_id"),
        "path_id": record.get("path_id"),
        "question": record.get("question"),
        "gold_answer": record.get("gold_answer"),
        "input_images": input_images,
        "generation_summary": generation_summary,
        "extracted_answer": extracted_answer,
        "answer_judge": answer_judge,
        "hop_chain": hop_chain,
        "hop_chain_coverage": hop_chain_coverage,
        "formatted_trajectory": formatted_trajectory,
        "messages": messages,
    }
    raw_record = _build_raw_trajectory_record(
        record=record,
        input_images=input_images,
        messages=messages,
        extracted_answer=extracted_answer,
        answer_judge=answer_judge,
        generation_summary=generation_summary,
        hop_chain_coverage=hop_chain_coverage,
        vqa_dir=vqa_dir,
        source_metadata_mode=source_metadata_mode,
        runtime=_serialize_runtime(context),
    )
    is_correct = bool((answer_judge or {}).get("is_correct"))
    return {
        "index": index,
        "result_record": result_record,
        "raw_record": raw_record,
        "is_correct": is_correct,
        "generation_summary": generation_summary,
    }


def _process_record_after_initial_start_gate(
    *,
    start_gate: _InitialWorkerStartGate,
    **kwargs: Any,
) -> dict[str, Any]:
    start_gate.wait()
    return _process_record(**kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    firecrawl_usage_before = get_firecrawl_usage_snapshot()

    input_modes = sum(bool(value) for value in (args.vqa_dir, args.question_list, args.question))
    if input_modes != 1:
        parser.error("Use exactly one of --vqa-dir, --question-list, or --question.")
    if args.question and not args.model_alias:
        parser.error("--model-alias is required in single-question mode unless SFT_OPENAI_MODEL / OPENAI_MODEL is set.")
    if args.responses_system_prompt_v2 and args.api_mode != "responses":
        parser.error("--responses-system-prompt-v2 requires --api-mode responses.")
    if args.responses_system_prompt_v2 and args.system_prompt:
        parser.error("--responses-system-prompt-v2 cannot be combined with --system-prompt.")
    if (args.vqa_dir or args.question_list) and not args.model_alias:
        parser.error("--model-alias is required in batch mode unless SFT_OPENAI_MODEL / OPENAI_MODEL is set.")
    if args.workers <= 0:
        parser.error("--workers must be positive.")
    if args.worker_start_stagger_s < 0:
        parser.error("--worker-start-stagger-s must be non-negative.")
    if args.offset < 0 or args.limit < 0:
        parser.error("--offset and --limit must be non-negative; --limit 0 means all records.")

    if args.vqa_dir:
        all_records = _load_vqa_records(Path(args.vqa_dir))
        records = _slice_batch_records(all_records, offset=args.offset, limit=args.limit)
    elif args.question_list:
        all_records = _load_filtered_trajectory_records(Path(args.question_list))
        records = _slice_batch_records(all_records, offset=args.offset, limit=args.limit)
    else:
        records = _single_question_record(
            question=args.question,
            gold_answer=args.gold_answer,
            hop_chain_json=args.hop_chain_json,
            image_paths=args.image,
            image_urls=args.image_url,
        )

    if (args.vqa_dir or args.question_list) and (args.image or args.image_url):
        for record in records:
            record["image_paths"] = list(args.image or [])
            record["image_urls"] = list(args.image_url or [])

    primary_system_prompt = (
        RESPONSES_SYSTEM_PROMPT_V2 if args.responses_system_prompt_v2 else args.system_prompt
    )
    agent_config = _config_from_model_arg(
        model_arg=args.model_alias,
        api_key=args.api_key,
        # #### START Response 0720 ####
        client_type=args.client_type,
        # #### END Response 0720 ####
        api_mode=args.api_mode,
        azure_endpoint=args.azure_endpoint,
        # #### START Response 0720 ####
        base_url=args.base_url,
        # #### END Response 0720 ####
        api_version=args.api_version,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout_s=args.timeout_s,
        system_prompt=primary_system_prompt,
        headers_json=args.headers_json,
        extra_body_json=args.extra_body_json,
        max_turns=args.max_turns,
        print_rounds=args.verbose,
        # #### START Response 0720 ####
        responses_reasoning_effort=args.responses_reasoning_effort,
        responses_reasoning_summary=args.responses_reasoning_summary,
        responses_reasoning_mode=args.responses_reasoning_mode,
        responses_reasoning_context=args.responses_reasoning_context,
        responses_store=args.responses_store,
        responses_prompt_public_reasoning=not args.no_responses_public_reasoning,
        responses_parallel_tool_calls=args.responses_parallel_tool_calls,
        responses_i2i_wrapper_enabled=args.responses_i2i_wrapper,
        # #### END Response 0720 ####
    )
    expert_config = None
    if args.expert_model:
        expert_config = _config_from_model_arg(
            model_arg=args.expert_model,
            api_key=args.expert_api_key or args.api_key,
            # #### START Response 0720 ####
            client_type=args.client_type,
            # #### END Response 0720 ####
            api_mode="chat_completions",
            azure_endpoint=args.expert_azure_endpoint or args.azure_endpoint,
            # #### START Response 0720 ####
            base_url=args.base_url,
            # #### END Response 0720 ####
            api_version=args.expert_api_version,
            max_tokens=args.expert_max_tokens,
            temperature=args.expert_temperature,
            timeout_s=args.timeout_s,
            system_prompt=(
                "You are a strict trajectory auditor. "
                "You inspect whether an agent trajectory truly covers each intended reasoning hop."
            ),
            headers_json=args.headers_json,
            extra_body_json=None,
            max_turns=args.max_turns,
            print_rounds=False,
            # #### START Response 0720 ####
            responses_prompt_public_reasoning=False,
            # #### END Response 0720 ####
        )

    raw_output_path: Path | None = None
    if args.raw_trajectories_jsonl:
        raw_output_path = Path(args.raw_trajectories_jsonl)
    elif args.output_jsonl:
        raw_output_path = Path(args.output_jsonl)
    elif args.vqa_dir:
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        raw_output_path = Path(args.vqa_dir) / f"sft_{timestamp}.jsonl"

    if args.resume and not (args.raw_trajectories_jsonl or args.output_jsonl):
        parser.error("--resume requires an explicit --output-jsonl or --raw-trajectories-jsonl path.")

    resumed_count = 0
    if args.resume and raw_output_path is not None and raw_output_path.exists():
        completed_keys = _load_completed_resume_keys(raw_output_path)
        original_count = len(records)
        records = [
            record
            for record in records
            if _resume_record_key(record) not in completed_keys
        ]
        resumed_count = original_count - len(records)
        print(
            f"[sft-resume] existing_records={len(completed_keys)} "
            f"skipped={resumed_count} remaining={len(records)}"
        )

    if raw_output_path is not None:
        raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        if not args.resume:
            with raw_output_path.open("w", encoding="utf-8"):
                pass
        print(f"raw_trajectories_jsonl: {raw_output_path}")
    pending_raw_records: list[dict[str, Any]] = []

    def flush_pending_raw_records() -> None:
        if raw_output_path is None or not pending_raw_records:
            return
        _write_jsonl_records(raw_output_path, pending_raw_records)
        pending_raw_records.clear()

    total_count = 0
    correct_count = 0
    incorrect_count = 0
    failed_count = 0
    max_turns_reached_count = 0
    total_turn_count = 0
    turn_count_samples = 0
    generation_stats: dict[str, Counter[str]] = {
        "generation_status": Counter(),
        "incomplete": Counter(),
        "failure_reasons": Counter(),
        "tool_success_counts": Counter(),
        "tool_error_counts": Counter(),
        "tool_error_reasons": Counter(),
        "record_exceptions": Counter(),
    }
    progress = (
        tqdm(
            total=len(records),
            desc="SFT trajectories",
            unit="sample",
            dynamic_ncols=True,
        )
        if tqdm is not None
        else None
    )

    def advance_progress() -> None:
        if progress is None:
            return
        progress.update(1)
        average_turns = total_turn_count / turn_count_samples if turn_count_samples else None
        progress.set_postfix(
            correct=correct_count,
            incorrect=incorrect_count,
            failed=failed_count,
            max_turns=max_turns_reached_count,
            avg_turns=f"{average_turns:.2f}" if average_turns is not None else "n/a",
        )

    try:
        resolved_vqa_dir = str(Path(args.vqa_dir).resolve()) if args.vqa_dir else None
        start_gate = _InitialWorkerStartGate(
            worker_count=min(args.workers, len(records)),
            interval_s=args.worker_start_stagger_s,
        )
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_context = {
                executor.submit(
                    _process_record_after_initial_start_gate,
                    start_gate=start_gate,
                    index=index,
                    record=record,
                    agent_config=agent_config,
                    expert_config=expert_config,
                    workdir=args.workdir,
                    vqa_dir=resolved_vqa_dir,
                    source_metadata_mode=args.source_metadata_mode,
                ): {
                    "index": index,
                    "record": record,
                }
                for index, record in enumerate(records, start=1)
            }
            for future in as_completed(future_to_context):
                task_context = future_to_context[future]
                record = task_context["record"]
                try:
                    payload = future.result()
                except Exception as exc:
                    failed_count += 1
                    total_count += 1
                    generation_stats["record_exceptions"][exc.__class__.__name__] += 1
                    print("\n" + "=" * 100)
                    print(f"question_id: {record.get('question_id')}")
                    if record.get("sample_id") is not None:
                        print(f"sample_id: {record.get('sample_id')}")
                    if record.get("path_id") is not None:
                        print(f"path_id: {record.get('path_id')}")
                    print("status: failed")
                    print(f"error: {exc.__class__.__name__}: {exc}")
                    print("traceback:")
                    print("".join(traceback.format_exception(exc)).rstrip())
                    advance_progress()
                    _print_generation_stats(generation_stats)
                    continue

                result_record = payload["result_record"]
                raw_record = payload["raw_record"]
                is_correct = bool(payload["is_correct"])
                generation_summary = payload.get("generation_summary") or {}
                _merge_generation_stats(generation_stats, generation_summary)
                if str(generation_summary.get("generation_status") or "") == "max_turns_reached":
                    max_turns_reached_count += 1
                try:
                    turn_count = int(generation_summary.get("turn_count"))
                except (TypeError, ValueError):
                    turn_count = None
                if turn_count is not None and turn_count >= 0:
                    total_turn_count += turn_count
                    turn_count_samples += 1
                _print_record_result(result_record)
                if raw_output_path is not None:
                    pending_raw_records.append(raw_record)
                    if len(pending_raw_records) >= _SFT_JSONL_FLUSH_EVERY:
                        flush_pending_raw_records()

                total_count += 1
                if is_correct:
                    correct_count += 1
                else:
                    incorrect_count += 1
                advance_progress()
                _print_generation_stats(generation_stats)
    finally:
        # Publish the final partial batch as well, so a short run or shutdown
        # does not leave completed trajectories only in memory.
        flush_pending_raw_records()
        if progress is not None:
            progress.close()

    print("\n" + "=" * 100)
    print("Trajectory Judge Summary")
    print(f"total: {total_count}")
    print(f"correct: {correct_count}")
    print(f"incorrect: {incorrect_count}")
    print(f"failed: {failed_count}")
    firecrawl_usage = _usage_delta(get_firecrawl_usage_snapshot(), firecrawl_usage_before)
    print(
        "firecrawl: "
        f"requests={firecrawl_usage.get('requests', 0)} "
        f"successful_requests={firecrawl_usage.get('successful_requests', 0)} "
        f"failed_requests={firecrawl_usage.get('failed_requests', 0)} "
        f"credits_used={firecrawl_usage.get('credits_used', 0)}"
    )
    _print_generation_stats(generation_stats)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
