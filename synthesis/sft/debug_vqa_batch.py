"""Debug and inspect SFT trajectories over one question or a VQA batch directory."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

from synthesis.model_worker import LLM_WORKER
from synthesis.model_worker import ModelMessage
from synthesis.model_worker import ModelRequest
from .pipeline import (
    build_agent_config,
    build_runtime_context,
    check_hop_chain_coverage,
    extract_answer,
    format_messages,
    judge,
    run_agent_loop,
)


PROMPT_DIAGNOSE_INCORRECT_TRAJECTORY = """You are diagnosing an incorrect multi-hop search trajectory for SFT repair.

You will receive:
1. The original question.
2. The gold answer.
3. The intended hop chain, where each hop contains the expected target and statement.
4. The raw trajectory text produced by an agent.
5. The extracted answer from the trajectory and a lightweight answer-judge result.

Your task:
- Identify the first hop where the trajectory clearly deviates from the intended hop chain.
- Decide whether the root cause is:
  - "question_problem": the question or hop wording is genuinely ambiguous or under-specified.
  - "agent_trajectory_problem": the question is still recoverable, but the agent searched, read, inferred, or validated incorrectly.
- If the root cause is "agent_trajectory_problem", further classify it as:
  - "trajectory_execution_error"
  - "insufficient_evidence"
- Provide a concise explanation grounded in the trajectory and hop chain.
- Provide a short reflection text that can be inserted into the trajectory, where the agent notices the branch is logically inconsistent and decides to restart.
- Suggest the hop index from which the search should restart.

Return strict JSON with this schema:
{
  "first_bad_hop_index": 0,
  "error_category": "question_problem",
  "trajectory_problem_type": "not_applicable",
  "expected_target": "...",
  "observed_target_or_branch": "...",
  "reason": "...",
  "evidence_excerpt": "...",
  "should_patch_question": true,
  "restart_from_hop_index": 0,
  "reflection_text": "...",
  "restart_query_hint": "..."
}

Rules:
- If the trajectory is bad because it never gathered enough evidence, classify it as "agent_trajectory_problem" + "insufficient_evidence".
- Use "not_applicable" for trajectory_problem_type when error_category is "question_problem".
- Keep the reflection text natural and concise, as if written by the agent itself.
- Do not output markdown or any text outside the JSON object.
"""


PROMPT_PATCH_QUESTION_MINIMALLY = """You are minimally repairing a multi-hop question.

The original question should still lead to the same intended gold answer and the same hop chain.
However, one hop is too ambiguous or under-specified, so the question must be minimally edited toward the intended answer.

Your task:
- Modify the original question as little as possible.
- Preserve the original answer target and the overall multi-hop structure.
- Add only the smallest extra clue needed to remove the ambiguity.
- Do not adapt the question to the incorrect trajectory. Repair it toward the intended hop chain and gold answer.

Return strict JSON:
{
  "revised_question": "...",
  "edit_summary": "...",
  "changed_span_summary": "...",
  "reason": "..."
}

Rules:
- Keep the wording style close to the original question.
- Do not make the question easier than necessary.
- Do not output markdown or any text outside the JSON object.
"""


PROMPT_POLISH_CORRECT_ASSISTANT_STEP = """You are polishing one assistant analysis turn inside a correct tool-using trajectory.

Rules:
- Rewrite only the current analysis under "====当前分析====".
- Use both the earlier trajectory and the tool shown under "====下轮工具====" to improve continuity.
- Keep all claims faithful to the provided trajectory and tool outputs.
- Do not modify historical analyses, tool results, or the overall search direction.
- Do not add unsupported facts.

Return strict JSON:
{
  "assistant_content": "..."
}
"""


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
    print(f"extracted_answer: {result.get('extracted_answer')}")
    print("answer_judge:")
    print(json.dumps(result.get("answer_judge") or {}, ensure_ascii=False, indent=2))
    if result.get("hop_chain"):
        print("hop_chain_coverage:")
        print(json.dumps(result.get("hop_chain_coverage") or {}, ensure_ascii=False, indent=2))
    print("\n--- Trajectory Text ---")
    print((result.get("formatted_trajectory") or {}).get("text") or "")


def _write_jsonl_record(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def _worker_generate_json(
    *,
    model_alias: str | None,
    system_prompt: str,
    payload: Any,
    max_tokens: int,
    trace_label: str,
) -> dict[str, Any]:
    if not model_alias:
        raise RuntimeError("A registered model alias is required for LLM_WORKER generation.")
    user_content = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, indent=2)
    return LLM_WORKER.generate_json(
        ModelRequest(
            model=model_alias,
            messages=[
                ModelMessage(role="system", content=system_prompt),
                ModelMessage(role="user", content=user_content),
            ],
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
            metadata={"trace_label": trace_label},
        )
    )


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


def _build_correct_polish_prompt(
    *,
    question: str,
    history_messages: list[dict[str, Any]],
    current_assistant_message: str,
    next_tool_message: dict[str, Any] | None,
) -> str:
    history_lines: list[str] = []
    for message in history_messages:
        role = str(message.get("role") or "")
        if role == "assistant":
            history_lines.append(f"agent:{_message_text_for_transcript(message)}")
        elif role == "tool":
            history_lines.append(f"tool:{_message_text_for_transcript(message)}")

    history_block = "\n\n".join(line for line in history_lines if line).strip()
    next_tool_text = ""
    if next_tool_message is not None:
        next_tool_text = f"tool:{_message_text_for_transcript(next_tool_message)}".strip()

    parts = [
        f"问题\n{question.strip()}",
        "",
        "==== History analysis ====",
        history_block,
        "==== Current analysis ====",
        f"agent:{current_assistant_message.strip()}",
        "",
        "==== Next tool ====",
        next_tool_text,
    ]
    return "\n".join(parts).strip()


def _correct_polish_candidate_indices(messages: list[dict[str, Any]]) -> list[int]:
    indices: list[int] = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        previous_role = str(messages[index - 1].get("role") or "") if index > 0 else ""
        if previous_role in {"user", "tool"}:
            indices.append(index)
    return indices


def _fallback_incorrect_diagnosis(raw_record: dict[str, Any]) -> dict[str, Any]:
    hop_chain = list(raw_record.get("hop_chain") or [])
    first_hop = hop_chain[0] if hop_chain else {}
    first_bad_hop_index = int(first_hop.get("hop_index") or 0)
    return {
        "first_bad_hop_index": first_bad_hop_index,
        "error_category": "agent_trajectory_problem",
        "trajectory_problem_type": "trajectory_execution_error",
        "expected_target": str(first_hop.get("target") or raw_record.get("gold_answer") or "").strip(),
        "observed_target_or_branch": str(raw_record.get("extracted_answer") or "").strip(),
        "reason": "Fallback diagnosis used because the repair model was unavailable or failed.",
        "evidence_excerpt": "",
        "should_patch_question": False,
        "restart_from_hop_index": first_bad_hop_index,
        "reflection_text": (
            "The branch I followed does not line up with the intended reasoning chain, "
            "so I should discard it and restart the search from the last reliable point."
        ),
        "restart_query_hint": str(first_hop.get("retrieval_query") or first_hop.get("statement") or "").strip(),
    }


def _build_source_metadata(record: dict[str, Any], *, vqa_dir: str | None) -> dict[str, Any]:
    return {
        "vqa_dir": vqa_dir,
        "question_id": record.get("question_id"),
        "sample_id": record.get("sample_id"),
        "path_id": record.get("path_id"),
        "question_record": record.get("question_record") or {},
        "sample_record": record.get("sample_record") or {},
    }


def _build_raw_trajectory_record(
    *,
    record: dict[str, Any],
    input_images: list[dict[str, str]],
    messages: list[dict[str, Any]],
    formatted_trajectory: dict[str, Any],
    extracted_answer: str,
    answer_judge: dict[str, Any],
    hop_chain_coverage: dict[str, Any] | None,
    vqa_dir: str | None,
) -> dict[str, Any]:
    return {
        "question_id": record.get("question_id"),
        "sample_id": record.get("sample_id"),
        "path_id": record.get("path_id"),
        "question": record.get("question"),
        "gold_answer": record.get("gold_answer"),
        "input_images": input_images,
        "source_metadata": _build_source_metadata(record, vqa_dir=vqa_dir),
        "raw_messages": messages,
        "raw_trajectory": formatted_trajectory,
        "extracted_answer": extracted_answer,
        "answer_judge": answer_judge,
        "hop_chain": list(record.get("hop_chain") or []),
        "hop_chain_coverage": hop_chain_coverage,
    }


def polish_correct_trajectory(
    raw_record: dict[str, Any],
    *,
    polish_model_alias: str | None,
    polish_max_tokens: int,
) -> dict[str, Any]:
    """Polish each assistant turn from left to right while keeping tools/history fixed."""

    original_messages = list(raw_record.get("raw_messages") or [])
    if not original_messages:
        return {
            "polish_mode": "sequential_assistant_polish",
            "polish_status": "skipped_empty_messages",
            "polish_notes": ["No raw messages were available for correct-trajectory polishing."],
            "correct_trajectory_polish": {
                "model_alias": polish_model_alias,
                "steps": [],
            },
            "final_question": raw_record.get("question") or "",
            "repair_diagnosis": None,
            "question_repair": None,
            "final_messages": [],
            "trajectory": {"text": "", "images": []},
        }

    if not polish_model_alias:
        return {
            "polish_mode": "sequential_assistant_polish",
            "polish_status": "skipped_missing_model_alias",
            "polish_notes": ["Correct trajectory polish was skipped because no model alias was provided."],
            "correct_trajectory_polish": {
                "model_alias": polish_model_alias,
                "steps": [],
            },
            "final_question": raw_record.get("question") or "",
            "repair_diagnosis": None,
            "question_repair": None,
            "final_messages": original_messages,
            "trajectory": raw_record.get("raw_trajectory") or {"text": "", "images": []},
        }

    polished_messages = copy.deepcopy(original_messages)
    step_records: list[dict[str, Any]] = []
    polish_notes: list[str] = []

    for assistant_index in _correct_polish_candidate_indices(polished_messages):
        current_message = polished_messages[assistant_index]
        previous_message = polished_messages[assistant_index - 1] if assistant_index > 0 else None
        previous_role = str(previous_message.get("role") or "") if previous_message else ""

        if previous_role == "tool":
            history_messages = polished_messages[: assistant_index - 1]
        elif previous_role == "user":
            history_messages = polished_messages[:assistant_index]
        else:
            step_records.append(
                {
                    "assistant_index": assistant_index,
                    "status": "skipped_unexpected_previous_role",
                    "previous_role": previous_role,
                }
            )
            continue

        current_content = current_message.get("content")
        if not isinstance(current_content, str):
            step_records.append(
                {
                    "assistant_index": assistant_index,
                    "status": "skipped_non_text_content",
                    "previous_role": previous_role,
                }
            )
            continue

        next_message = polished_messages[assistant_index + 1] if assistant_index + 1 < len(polished_messages) else None
        next_tool_message = next_message if isinstance(next_message, dict) and next_message.get("role") == "tool" else None
        prompt_text = _build_correct_polish_prompt(
            question=str(raw_record.get("question") or ""),
            history_messages=history_messages,
            current_assistant_message=current_content,
            next_tool_message=next_tool_message,
        )

        try:
            parsed = _worker_generate_json(
                model_alias=polish_model_alias,
                system_prompt=PROMPT_POLISH_CORRECT_ASSISTANT_STEP,
                payload=prompt_text,
                max_tokens=polish_max_tokens,
                trace_label=f"sft_correct_polish:{raw_record.get('question_id') or 'question'}:{assistant_index}",
            )
            polished_content = str(parsed.get("assistant_content") or "").strip()
            if not polished_content:
                raise ValueError("assistant_content is empty")
        except Exception as exc:
            step_records.append(
                {
                    "assistant_index": assistant_index,
                    "status": "fallback_original",
                    "previous_role": previous_role,
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            )
            polish_notes.append(
                f"Assistant message at index {assistant_index} kept original content because polishing failed."
            )
            continue

        polished_messages[assistant_index] = {
            **current_message,
            "content": polished_content,
        }
        step_records.append(
            {
                "assistant_index": assistant_index,
                "status": "polished",
                "previous_role": previous_role,
            }
        )

    if not polish_notes:
        polish_notes.append("Correct trajectory was polished sequentially from left to right.")

    final_trajectory = format_messages(polished_messages)
    status = "completed"
    if any(step.get("status") != "polished" for step in step_records):
        status = "completed_with_fallbacks"

    return {
        "polish_mode": "sequential_assistant_polish",
        "polish_status": status,
        "polish_notes": polish_notes,
        "correct_trajectory_polish": {
            "model_alias": polish_model_alias,
            "steps": step_records,
        },
        "final_question": raw_record.get("question") or "",
        "repair_diagnosis": None,
        "question_repair": None,
        "final_messages": polished_messages,
        "trajectory": final_trajectory,
    }


def diagnose_incorrect_trajectory(
    raw_record: dict[str, Any],
    *,
    repair_model_alias: str | None,
    repair_max_tokens: int,
) -> dict[str, Any]:
    payload = {
        "question": raw_record.get("question"),
        "gold_answer": raw_record.get("gold_answer"),
        "hop_chain": raw_record.get("hop_chain") or [],
        "raw_trajectory": (raw_record.get("raw_trajectory") or {}).get("text") or "",
        "extracted_answer": raw_record.get("extracted_answer") or "",
        "answer_judge": raw_record.get("answer_judge") or {},
    }
    try:
        parsed = _worker_generate_json(
            model_alias=repair_model_alias,
            system_prompt=PROMPT_DIAGNOSE_INCORRECT_TRAJECTORY,
            payload=payload,
            max_tokens=repair_max_tokens,
            trace_label=f"sft_repair_diagnose:{raw_record.get('question_id') or 'question'}",
        )
    except Exception as exc:
        fallback = _fallback_incorrect_diagnosis(raw_record)
        fallback["model_error"] = f"{exc.__class__.__name__}: {exc}"
        return fallback

    normalized = dict(parsed)
    normalized.setdefault("first_bad_hop_index", 0)
    normalized.setdefault("error_category", "agent_trajectory_problem")
    normalized.setdefault("trajectory_problem_type", "trajectory_execution_error")
    normalized.setdefault("expected_target", "")
    normalized.setdefault("observed_target_or_branch", "")
    normalized.setdefault("reason", "")
    normalized.setdefault("evidence_excerpt", "")
    normalized.setdefault("should_patch_question", False)
    normalized.setdefault("restart_from_hop_index", normalized.get("first_bad_hop_index", 0))
    normalized.setdefault("reflection_text", "")
    normalized.setdefault("restart_query_hint", "")
    return normalized


def repair_question_minimally(
    raw_record: dict[str, Any],
    *,
    diagnosis: dict[str, Any],
    repair_model_alias: str | None,
    repair_max_tokens: int,
) -> dict[str, Any]:
    original_question = str(raw_record.get("question") or "").strip()
    if not original_question:
        return {
            "revised_question": "",
            "edit_summary": "Question repair skipped because the original question is empty.",
            "changed_span_summary": "",
            "reason": "missing_original_question",
        }

    if not diagnosis.get("should_patch_question"):
        return {
            "revised_question": original_question,
            "edit_summary": "Question repair skipped because diagnosis did not require a question patch.",
            "changed_span_summary": "",
            "reason": "not_required",
        }

    payload = {
        "original_question": original_question,
        "gold_answer": raw_record.get("gold_answer"),
        "hop_chain": raw_record.get("hop_chain") or [],
        "diagnosis": diagnosis,
    }
    try:
        parsed = _worker_generate_json(
            model_alias=repair_model_alias,
            system_prompt=PROMPT_PATCH_QUESTION_MINIMALLY,
            payload=payload,
            max_tokens=repair_max_tokens,
            trace_label=f"sft_repair_question:{raw_record.get('question_id') or 'question'}",
        )
    except Exception as exc:
        return {
            "revised_question": original_question,
            "edit_summary": "Question repair failed, so the original question was kept.",
            "changed_span_summary": "",
            "reason": f"repair_model_error: {exc.__class__.__name__}: {exc}",
        }

    revised_question = str(parsed.get("revised_question") or "").strip() or original_question
    return {
        "revised_question": revised_question,
        "edit_summary": str(parsed.get("edit_summary") or "").strip(),
        "changed_span_summary": str(parsed.get("changed_span_summary") or "").strip(),
        "reason": str(parsed.get("reason") or "").strip(),
    }


def build_restart_search_stub(
    raw_record: dict[str, Any],
    *,
    diagnosis: dict[str, Any],
    question_repair: dict[str, Any],
) -> dict[str, Any]:
    restart_question = str(question_repair.get("revised_question") or raw_record.get("question") or "").strip()
    restart_from_hop_index = int(diagnosis.get("restart_from_hop_index") or diagnosis.get("first_bad_hop_index") or 0)
    reflection_text = str(diagnosis.get("reflection_text") or "").strip()
    if not reflection_text:
        reflection_text = (
            "The branch I just followed cannot support the intended target, so I should reject it and search again."
        )

    restart_hint = str(diagnosis.get("restart_query_hint") or "").strip()
    if restart_hint:
        restart_text = (
            f"I should restart the search from hop {restart_from_hop_index} and use a tighter query or clue. "
            f"A good restart hint is: {restart_hint}"
        )
    else:
        restart_text = (
            f"I should restart the search from hop {restart_from_hop_index} and verify the next target more carefully."
        )
    if restart_question and restart_question != str(raw_record.get("question") or "").strip():
        restart_text = (
            f"{restart_text}\n\nI should continue with the minimally repaired question:\n{restart_question}"
        )

    final_messages = list(raw_record.get("raw_messages") or [])
    final_messages.append({"role": "assistant", "content": reflection_text})
    final_messages.append({"role": "assistant", "content": restart_text})
    final_trajectory = format_messages(final_messages)
    return {
        "status": "restart_todo",
        "restart_from_hop_index": restart_from_hop_index,
        "restart_question": restart_question,
        "reflection_text": reflection_text,
        "restart_text": restart_text,
        "final_messages": final_messages,
        "trajectory": final_trajectory,
        "todo": (
            "TODO: re-run search from the restart point, keep the wrong branch as context, "
            "and insert the corrected continuation after the reflection step."
        ),
    }


def polish_incorrect_trajectory(
    raw_record: dict[str, Any],
    *,
    repair_model_alias: str | None,
    repair_max_tokens: int,
) -> dict[str, Any]:
    """Diagnose and scaffold repair for incorrect trajectories."""

    diagnosis = diagnose_incorrect_trajectory(
        raw_record,
        repair_model_alias=repair_model_alias,
        repair_max_tokens=repair_max_tokens,
    )
    question_repair = repair_question_minimally(
        raw_record,
        diagnosis=diagnosis,
        repair_model_alias=repair_model_alias,
        repair_max_tokens=repair_max_tokens,
    )
    restart_stub = build_restart_search_stub(
        raw_record,
        diagnosis=diagnosis,
        question_repair=question_repair,
    )

    return {
        "polish_mode": "repair_after_failure",
        "polish_status": str(restart_stub.get("status") or "restart_todo"),
        "polish_notes": [
            "Incorrect trajectory was diagnosed with LLM_WORKER.",
            "A reflection-and-restart scaffold was inserted.",
            str(restart_stub.get("todo") or ""),
        ],
        "final_question": str(question_repair.get("revised_question") or raw_record.get("question") or "").strip(),
        "repair_diagnosis": diagnosis,
        "question_repair": question_repair,
        "trajectory_repair": restart_stub,
        "final_messages": list(restart_stub.get("final_messages") or raw_record.get("raw_messages") or []),
        "trajectory": restart_stub.get("trajectory") or raw_record.get("raw_trajectory") or {"text": "", "images": []},
    }


def _build_sft_training_record(
    *,
    raw_record: dict[str, Any],
    polished: dict[str, Any],
) -> dict[str, Any]:
    return {
        "question_id": raw_record.get("question_id"),
        "sample_id": raw_record.get("sample_id"),
        "path_id": raw_record.get("path_id"),
        "question": polished.get("final_question") or raw_record.get("question"),
        "original_question": raw_record.get("question"),
        "gold_answer": raw_record.get("gold_answer"),
        "input_images": list(raw_record.get("input_images") or []),
        "source_metadata": dict(raw_record.get("source_metadata") or {}),
        "answer_judge": dict(raw_record.get("answer_judge") or {}),
        "hop_chain": list(raw_record.get("hop_chain") or []),
        "hop_chain_coverage": raw_record.get("hop_chain_coverage"),
        "raw_messages": list(raw_record.get("raw_messages") or []),
        "raw_trajectory": raw_record.get("raw_trajectory") or {"text": "", "images": []},
        "polish_mode": polished.get("polish_mode"),
        "polish_status": polished.get("polish_status"),
        "polish_notes": list(polished.get("polish_notes") or []),
        "correct_trajectory_polish": polished.get("correct_trajectory_polish"),
        "repair_diagnosis": polished.get("repair_diagnosis"),
        "question_repair": polished.get("question_repair"),
        "trajectory_repair": polished.get("trajectory_repair"),
        "final_messages": list(polished.get("final_messages") or []),
        "final_trajectory": polished.get("trajectory") or {"text": "", "images": []},
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vqa-dir", help="Directory produced by synthesis.vqa.run_batch.")
    parser.add_argument("--question", help="Single question to debug.")
    parser.add_argument("--gold-answer", default="", help="Gold answer for single-question mode.")
    parser.add_argument("--hop-chain-json", help="JSON list for single-question hop chain.")
    parser.add_argument("--image", action="append", help="Attach a local image path to the user input.")
    parser.add_argument("--image-url", action="append", help="Attach a remote image URL to the user input.")
    parser.add_argument("--limit", type=int, default=5, help="How many questions to run in batch mode.")
    parser.add_argument("--offset", type=int, default=0, help="Start offset in batch mode.")
    parser.add_argument("--workdir", default=os.path.join(os.getcwd(), "synthesis_sft_runs"))
    parser.add_argument("--output-jsonl", help="Optional path to save raw trajectory records.")
    parser.add_argument(
        "--raw-trajectories-jsonl",
        help="Optional path to save raw formatted trajectories.",
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

    parser.add_argument(
        "--model",
        default=os.environ.get("SFT_OPENAI_MODEL") or os.environ.get("OPENAI_MODEL") or "",
        help="Primary answer model. Prefer a registered alias from synthesis/models.json.",
    )
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument(
        "--api-mode",
        choices=("manual_react", "chat_completions", "responses"),
        default=os.environ.get("SFT_OPENAI_API_MODE") or "manual_react",
        help="Primary trajectory collection mode. Defaults to manual_react.",
    )
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
    api_mode: str,
    azure_endpoint: str | None,
    api_version: str | None,
    max_tokens: int | None,
    temperature: float | None,
    timeout_s: float,
    system_prompt: str | None,
    headers_json: str | None,
    extra_body_json: str | None,
    max_turns: int,
    print_rounds: bool,
) -> Any:
    return build_agent_config(
        model=model_arg,
        api_key=api_key,
        client_type="azure_openai",
        azure_endpoint=azure_endpoint,
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
    )


def _build_user_prompt_text(record: dict[str, Any]) -> str:
    question_text = str(record.get("question") or "").strip()
    gold_answer = str(record.get("gold_answer") or "").strip()
    if gold_answer:
        return f"Question: {question_text}\nAnswer: {gold_answer}"
    return question_text


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


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if bool(args.vqa_dir) == bool(args.question):
        parser.error("Use exactly one of --vqa-dir or --question.")
    if args.question and not args.model:
        parser.error("--model is required in single-question mode unless SFT_OPENAI_MODEL / OPENAI_MODEL is set.")
    if args.vqa_dir and not args.model:
        parser.error("--model is required in batch mode unless SFT_OPENAI_MODEL / OPENAI_MODEL is set.")

    if args.vqa_dir:
        all_records = _load_vqa_records(Path(args.vqa_dir))
        records = all_records[args.offset : args.offset + args.limit]
    else:
        records = _single_question_record(
            question=args.question,
            gold_answer=args.gold_answer,
            hop_chain_json=args.hop_chain_json,
            image_paths=args.image,
            image_urls=args.image_url,
        )

    if args.vqa_dir and (args.image or args.image_url):
        for record in records:
            record["image_paths"] = list(args.image or [])
            record["image_urls"] = list(args.image_url or [])

    agent_config = _config_from_model_arg(
        model_arg=args.model,
        api_key=args.api_key,
        api_mode=args.api_mode,
        azure_endpoint=args.azure_endpoint,
        api_version=args.api_version,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout_s=args.timeout_s,
        system_prompt=args.system_prompt,
        headers_json=args.headers_json,
        extra_body_json=args.extra_body_json,
        max_turns=args.max_turns,
        print_rounds=args.verbose,
    )
    expert_config = None
    if args.expert_model:
        expert_config = _config_from_model_arg(
            model_arg=args.expert_model,
            api_key=args.expert_api_key or args.api_key,
            api_mode="chat_completions",
            azure_endpoint=args.expert_azure_endpoint or args.azure_endpoint,
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
        )

    raw_output_path: Path | None = None
    if args.raw_trajectories_jsonl:
        raw_output_path = Path(args.raw_trajectories_jsonl)
    elif args.output_jsonl:
        raw_output_path = Path(args.output_jsonl)

    raw_output_handle = None
    if raw_output_path is not None:
        raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_output_handle = raw_output_path.open("w", encoding="utf-8")
    total_count = 0
    correct_count = 0
    incorrect_count = 0

    try:
        for index, record in enumerate(records, start=1):
            context = build_runtime_context(
                working_dir=os.path.join(args.workdir, f"debug_{index:04d}_{record.get('question_id') or 'question'}"),
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
            messages = run_agent_loop(
                prompt=None if input_messages is not None else _build_user_prompt_text(record),
                messages=input_messages,
                config=agent_config,
                context=context,
            )
            extracted_answer = extract_answer(messages)
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
                "extracted_answer": extracted_answer,
                "answer_judge": answer_judge,
                "hop_chain": hop_chain,
                "hop_chain_coverage": hop_chain_coverage,
                "formatted_trajectory": formatted_trajectory,
                "messages": messages,
            }
            _print_record_result(result_record)
            raw_record = _build_raw_trajectory_record(
                record=record,
                input_images=input_images,
                messages=messages,
                formatted_trajectory=formatted_trajectory,
                extracted_answer=extracted_answer,
                answer_judge=answer_judge,
                hop_chain_coverage=hop_chain_coverage,
                vqa_dir=str(Path(args.vqa_dir).resolve()) if args.vqa_dir else None,
            )
            if raw_output_handle is not None:
                _write_jsonl_record(raw_output_handle, raw_record)

            is_correct = bool((answer_judge or {}).get("is_correct"))
            total_count += 1
            if is_correct:
                correct_count += 1
            else:
                incorrect_count += 1
    finally:
        if raw_output_handle is not None:
            raw_output_handle.close()

    print("\n" + "=" * 100)
    print("Trajectory Judge Summary")
    print(f"total: {total_count}")
    print(f"correct: {correct_count}")
    print(f"incorrect: {incorrect_count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
