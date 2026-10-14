"""Expert collection through the production SFT tool runtime.

The only model-facing FVQA content is the question and image. This module is
the boundary that enforces that the label never enters the writer prompt.
"""

from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
import re
from typing import Any, Iterable

from synthesis.sft.api_tools import DEFAULT_SYSTEM_PROMPT, RESPONSES_SYSTEM_PROMPT
from synthesis.sft.pipeline import build_agent_config, build_runtime_context, extract_answer, run_agent_session

from .io import append_jsonl, load_completed_ids
from .schema import CollectorConfig, PublicCase, VERSION
from .style import (
    build_collector_system_prompt,
    build_fvqa_collector_base_prompt,
    build_livevqa_collector_base_prompt,
)


def build_collector_user_messages(case: PublicCase) -> list[dict[str, Any]]:
    """Return the exact model-facing prompt; it deliberately omits the label."""

    prompt_view = case.to_dict()
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_view["question"]},
                {"type": "image_path", "path": prompt_view["image_path"]},
            ],
        }
    ]


def _safe_case_dirname(case_id: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id).strip("._-") or "case"
    return readable[:96]


def _summarize(messages: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    failures: Counter[str] = Counter()
    successes: Counter[str] = Counter()
    assistant_turns = 0
    for message in messages:
        if str(message.get("role") or "") == "assistant":
            assistant_turns += 1
        if str(message.get("role") or "") != "tool":
            continue
        name = str(message.get("name") or "unknown")
        try:
            import json

            payload = json.loads(str(message.get("content") or "{}"))
        except Exception:
            payload = {}
        if isinstance(payload, dict) and payload.get("ok") is False:
            failures[name] += 1
        else:
            successes[name] += 1
    status = str(metadata.get("generation_status") or "unknown")
    return {
        "generation_status": status,
        "generation_complete": bool(metadata.get("generation_complete")) or status == "finished",
        "stop_reason": metadata.get("stop_reason"),
        "max_turns": metadata.get("max_turns"),
        "turn_count": metadata.get("turn_count"),
        "assistant_turn_count": assistant_turns,
        "tool_call_count": metadata.get("tool_call_count"),
        "tool_success_counts": dict(successes),
        "tool_error_counts": dict(failures),
    }


def _serialize_runtime(context: Any) -> dict[str, Any]:
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
                "title": resource.title,
                "snippet": resource.snippet,
                "search_tool": resource.search_tool,
                "search_query": resource.search_query,
                "rank": resource.rank,
            }
        )
    return {"session_id": context.session_id, "case_id": context.case_id, "url_resources": resources}


def collect_case(case: PublicCase, config: CollectorConfig) -> dict[str, Any]:
    """Call the expert writer once and return a raw, converter-compatible row."""

    config.validate()
    image = Path(case.image_path)
    if not image.is_file():
        raise FileNotFoundError(f"Missing source image for {case.case_id}: {image}")
    # Responses uses native function calls.  Starting from its dedicated prompt
    # keeps the writer's visible prose/tool protocol compatible with the
    # Responses trace format; the generic default prompt is for manual ReAct.
    shared_base_system_prompt = (
        RESPONSES_SYSTEM_PROMPT if config.api_mode == "responses" else DEFAULT_SYSTEM_PROMPT
    )
    base_prompt_builder = (
        build_livevqa_collector_base_prompt
        if case.source_dataset.casefold().startswith("livevqa")
        else build_fvqa_collector_base_prompt
    )
    system_prompt = build_collector_system_prompt(
        base_prompt_builder(shared_base_system_prompt, api_mode=config.api_mode),
        style_guide_file=config.style_guide_file,
        style_examples_file=config.style_examples_file,
    )
    agent_config = build_agent_config(
        model=config.model,
        client_type=config.client_type,
        api_mode=config.api_mode,
        max_tokens=config.max_tokens,
        max_turns=config.max_turns,
        temperature=config.temperature,
        timeout_s=config.timeout_s,
        system_prompt=system_prompt,
        print_rounds=config.print_rounds,
    )
    context = build_runtime_context(
        working_dir=os.path.join(config.workdir, _safe_case_dirname(case.case_id)),
        filename_prefix="fvqa",
        case_id=case.case_id,
        metadata={
            "case_id": case.case_id,
            "question": case.question,
            "collector_mode": "hidden_reference",
            # Deliberately no gold_answer/hop_chain/source KG fields.
        },
    )
    context.register_image(str(image.resolve()))
    result = run_agent_session(
        messages=build_collector_user_messages(case),
        config=agent_config,
        context=context,
    )
    messages = result.messages
    extracted = str(result.final_text or extract_answer(messages)).strip()
    return {
        "schema_version": VERSION,
        "case_id": case.case_id,
        "question_id": case.case_id,
        "sample_id": case.case_id,
        "question": case.question,
        "input_images": [{"image_path": str(image.resolve())}],
        "source_metadata": {
            "source_dataset": case.source_dataset,
            "split": case.split,
            "collector_reference_mode": "hidden_reference",
        },
        "raw_messages": messages,
        "extracted_answer": extracted,
        "generation_summary": _summarize(messages, result.metadata),
        "runtime": _serialize_runtime(context),
        "answer_judge": {"is_correct": False, "status": "pending_independent_review"},
    }


def collect_cases(
    cases: Iterable[PublicCase],
    *,
    config: CollectorConfig,
    output_jsonl: str | Path,
    resume: bool = False,
    limit: int | None = None,
) -> dict[str, int]:
    """Sequential, append-only collection. Resume is keyed only by case ID."""

    if Path(output_jsonl).exists() and not resume:
        raise FileExistsError(f"Collector output exists; use --resume or choose a new path: {output_jsonl}")
    completed = load_completed_ids(output_jsonl, require_success=True) if resume else set()
    counts = {"seen": 0, "written": 0, "skipped_resume": 0, "errors": 0}
    for case in cases:
        counts["seen"] += 1
        if case.case_id in completed:
            counts["skipped_resume"] += 1
            continue
        try:
            append_jsonl(output_jsonl, collect_case(case, config))
            counts["written"] += 1
        except Exception as error:
            append_jsonl(
                output_jsonl,
                {
                    "schema_version": VERSION,
                    "case_id": case.case_id,
                    "question": case.question,
                    "collection_error": f"{error.__class__.__name__}: {error}",
                },
            )
            counts["errors"] += 1
        if limit is not None and counts["seen"] >= limit:
            break
    return counts
