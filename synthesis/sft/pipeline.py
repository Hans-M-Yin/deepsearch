"""Clean SFT data-construction pipeline interfaces built on top of api_tools."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from .api_tools import DEFAULT_SYSTEM_PROMPT
from .api_tools import AgentRunResult
from .api_tools import OpenAIToolAgent
from .api_tools import OpenAIToolAgentConfig
from .api_tools import ToolRuntimeContext
from ..model_worker import LLM_WORKER
from ..model_worker import ModelMessage
from ..model_worker import ModelRequest


Message = dict[str, Any]
_SFT_FIXED_REQUEST_ID = "3200636808"


def _optional_env_float(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _sft_worker_metadata(trace_label: str, *, extra_body: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "trace_label": trace_label,
        "session_id": _SFT_FIXED_REQUEST_ID,
        "prompt_cache_key": _SFT_FIXED_REQUEST_ID,
        "user_id": _SFT_FIXED_REQUEST_ID,
        "x_tt_logid": _SFT_FIXED_REQUEST_ID,
    }
    if extra_body:
        metadata["extra_body"] = extra_body
    return metadata


def _optional_env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return None
    return int(value)


# #### START Response 0720 ####
def _optional_env_bool(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return None
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
# #### END Response 0720 ####


def _message_text(content: Any) -> str:
    if content in (None, ""):
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, dict):
                part_type = str(part.get("type") or "")
                if part_type in {"text", "input_text"}:
                    text = str(part.get("text", "")).strip()
                    if text:
                        chunks.append(text)
            elif part is not None:
                text = str(part).strip()
                if text:
                    chunks.append(text)
        return "\n".join(chunks).strip()
    return str(content).strip()


def _normalize_answer_text(text: str) -> str:
    normalized = text.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[^\w\s]", "", normalized)
    return normalized.strip()


def _try_parse_json_text(text: str) -> Any | None:
    candidate = text.strip()
    if not candidate or candidate[0] not in "[{":
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _resolve_registered_model_alias(alias_or_model: str | None) -> dict[str, Any] | None:
    if not alias_or_model:
        return None
    try:
        return LLM_WORKER.get_model(alias_or_model)
    except Exception:
        return None


_IMAGE_FIELD_HINTS = (
    "image",
    "image_url",
    "thumbnail",
    "local_path",
    "cropped_image",
    "path",
)


def _looks_like_image_reference(text: str, field_name: str | None = None) -> bool:
    lowered = text.lower().strip()
    if not lowered:
        return False
    if lowered.startswith("img_") or lowered.startswith("data:image"):
        return True
    if field_name:
        field_lower = field_name.lower()
        if any(hint in field_lower for hint in _IMAGE_FIELD_HINTS):
            return True
    image_suffixes = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".svg")
    return lowered.startswith("/") and lowered.endswith(image_suffixes) or (
        lowered.startswith(("http://", "https://")) and lowered.endswith(image_suffixes)
    )


def _collect_image_references(
    value: Any,
    sink: list[dict[str, str]],
    seen: set[str],
    *,
    field_name: str | None = None,
) -> None:
    if value is None:
        return
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return
        if _looks_like_image_reference(text, field_name):
            if text not in seen:
                seen.add(text)
                sink.append({"image": text})
        return
    if isinstance(value, dict):
        for nested_key, nested in value.items():
            _collect_image_references(nested, sink, seen, field_name=str(nested_key))
        return
    if isinstance(value, list):
        for item in value:
            _collect_image_references(item, sink, seen, field_name=field_name)


def _format_tool_content(content: Any) -> tuple[str, list[dict[str, str]]]:
    text = _message_text(content)
    images: list[dict[str, str]] = []
    seen: set[str] = set()
    parsed = _try_parse_json_text(text) if isinstance(text, str) else None
    if parsed is not None:
        _collect_image_references(parsed, images, seen)
        return json.dumps(parsed, ensure_ascii=False, indent=2), images
    _collect_image_references(content, images, seen)
    return text, images


def _format_single_message(index: int, message: Message) -> tuple[str, list[dict[str, str]]]:
    role = str(message.get("role") or "unknown")
    sections = [f"[Step {index}] {role.upper()}"]
    images: list[dict[str, str]] = []
    seen: set[str] = set()

    if role == "tool":
        tool_name = str(message.get("name") or "").strip()
        tool_call_id = str(message.get("tool_call_id") or "").strip()
        tool_arguments = message.get("arguments")
        if tool_name:
            sections.append(f"Tool Name: {tool_name}")
        if tool_call_id:
            sections.append(f"Tool Call ID: {tool_call_id}")
        if tool_arguments is not None:
            sections.append("Tool Arguments:")
            sections.append(json.dumps(tool_arguments, ensure_ascii=False, indent=2))
        tool_text, tool_images = _format_tool_content(message.get("content"))
        parsed_tool_output = _try_parse_json_text(tool_text) if tool_text else None
        if isinstance(parsed_tool_output, dict):
            ok_value = parsed_tool_output.get("ok")
            if ok_value is False:
                sections.append("Tool Status: error")
                if parsed_tool_output.get("error"):
                    sections.append(f"Tool Error: {parsed_tool_output['error']}")
            elif ok_value is True:
                sections.append("Tool Status: ok")
        elif tool_text:
            sections.append("Tool Status: non_json_output")
        for image in tool_images:
            image_value = image["image"]
            if image_value not in seen:
                seen.add(image_value)
                images.append(image)
        return "\n".join(sections).strip(), images

    text = _message_text(message.get("content"))
    if text:
        sections.append(text)

    for part in message.get("content") or [] if isinstance(message.get("content"), list) else []:
        if not isinstance(part, dict):
            continue
        part_type = str(part.get("type") or "")
        if part_type == "image_url":
            image_url = part.get("image_url")
            if isinstance(image_url, dict):
                image_value = str(image_url.get("url") or "").strip()
            else:
                image_value = str(image_url or "").strip()
        elif part_type in {"image", "input_image", "image_path", "image_ref"}:
            image_value = str(
                part.get("image")
                or part.get("path")
                or part.get("url")
                or part.get("image_url")
                or part.get("ref")
                or ""
            ).strip()
        else:
            image_value = ""
        if image_value and image_value not in seen:
            seen.add(image_value)
            images.append({"image": image_value})

    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        sections.append("Tool Calls:")
        sections.append(json.dumps(tool_calls, ensure_ascii=False, indent=2))
    return "\n".join(sections).strip(), images


def _build_hop_check_prompt(trajectory_text: str, hop_chain: list[dict[str, Any]]) -> str:
    return (
        "You are checking whether a research trajectory actually covered each reasoning hop.\n\n"
        "You will receive:\n"
        "1. A formatted agent trajectory, including user prompts, assistant reasoning/actions, and tool outputs.\n"
        "2. A hop chain, where each hop describes one intended reasoning step.\n\n"
        "Your task:\n"
        "- For each hop, decide whether the hop clearly appears in the trajectory.\n"
        "- Use the hop statement, source, target, relation, and retrieval_query together.\n"
        "- Mark a hop as covered only if the trajectory contains enough evidence that the agent actually visited or used that step.\n"
        "- If the hop is only weakly implied, mark it as not covered.\n"
        "- Also identify missing hops and a short overall diagnosis.\n\n"
        "Return strict JSON with this schema:\n"
        "{\n"
        '  "overall_complete": true,\n'
        '  "summary": "...",\n'
        '  "missing_hop_indices": [1, 2],\n'
        '  "hop_checks": [\n'
        "    {\n"
        '      "hop_index": 0,\n'
        '      "covered": true,\n'
        '      "confidence": 0.0,\n'
        '      "evidence_excerpt": "...",\n'
        '      "reason": "..."\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Trajectory:\n{trajectory_text}\n\n"
        f"Hop Chain:\n{json.dumps(hop_chain, ensure_ascii=False, indent=2)}\n"
    )


def build_agent_config(
    *,
    model: str | None = None,
    api_key: str | None = None,
    client_type: str = "azure_openai",
    azure_endpoint: str | None = None,
    base_url: str | None = None,
    api_version: str | None = None,
    api_mode: str = "manual_react",
    max_tokens: int | None = None,
    temperature: float | None = None,
    timeout_s: float | None = None,
    system_prompt: str | None = None,
    headers: dict[str, str] | None = None,
    extra_body: dict[str, Any] | None = None,
    max_turns: int | None = None,
    print_rounds: bool = False,
    # #### START Response 0720 ####
    responses_reasoning_effort: str | None = None,
    responses_reasoning_summary: str | None = None,
    responses_reasoning_mode: str | None = None,
    responses_reasoning_context: str | None = None,
    responses_parallel_tool_calls: bool | None = None,
    responses_store: bool | None = None,
    responses_prompt_public_reasoning: bool | None = None,
    responses_i2i_wrapper_enabled: bool | None = None,
    # #### END Response 0720 ####
) -> OpenAIToolAgentConfig:
    """Build a reusable agent config from arguments or environment defaults."""

    resolved_model = model or os.environ.get("SFT_OPENAI_MODEL") or os.environ.get("OPENAI_MODEL")
    if not resolved_model:
        raise ValueError("model is required, or set SFT_OPENAI_MODEL / OPENAI_MODEL.")

    return OpenAIToolAgentConfig(
        model=resolved_model,
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        client_type=client_type,
        azure_endpoint=(
            azure_endpoint
            or os.environ.get("SFT_OPENAI_AZURE_ENDPOINT")
            or os.environ.get("SFT_OPENAI_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
        ),
        base_url=(
            base_url
            or os.environ.get("SFT_OPENAI_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
        ),
        api_version=api_version or os.environ.get("SFT_OPENAI_API_VERSION") or "2024-03-01-preview",
        api_mode=api_mode,
        max_tokens=max_tokens if max_tokens is not None else _optional_env_int("SFT_OPENAI_MAX_TOKENS"),
        temperature=temperature if temperature is not None else _optional_env_float("SFT_OPENAI_TEMPERATURE"),
        timeout_s=timeout_s if timeout_s is not None else float(os.environ.get("SFT_OPENAI_TIMEOUT_S", "120")),
        system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
        default_headers=headers,
        extra_body=extra_body,
        max_turns=max_turns or int(os.environ.get("SFT_OPENAI_MAX_TURNS", "8")),
        print_rounds=print_rounds,
        # #### START Response 0720 ####
        responses_reasoning_effort=(
            responses_reasoning_effort
            if responses_reasoning_effort is not None
            else os.environ.get("SFT_RESPONSES_REASONING_EFFORT")
        ),
        responses_reasoning_summary=(
            responses_reasoning_summary
            if responses_reasoning_summary is not None
            else os.environ.get("SFT_RESPONSES_REASONING_SUMMARY", "auto")
        ),
        responses_reasoning_mode=(
            responses_reasoning_mode
            if responses_reasoning_mode is not None
            else os.environ.get("SFT_RESPONSES_REASONING_MODE")
        ),
        responses_reasoning_context=(
            responses_reasoning_context
            if responses_reasoning_context is not None
            else os.environ.get("SFT_RESPONSES_REASONING_CONTEXT", "all_turns")
        ),
        responses_parallel_tool_calls=(
            responses_parallel_tool_calls
            if responses_parallel_tool_calls is not None
            else os.environ.get("SFT_RESPONSES_PARALLEL_TOOL_CALLS", "0").strip().lower()
            in {"1", "true", "yes", "on"}
        ),
        responses_store=(
            responses_store
            if responses_store is not None
            else _optional_env_bool("SFT_RESPONSES_STORE")
        ),
        responses_prompt_public_reasoning=(
            responses_prompt_public_reasoning
            if responses_prompt_public_reasoning is not None
            else os.environ.get("SFT_RESPONSES_PUBLIC_REASONING", "1").strip().lower()
            not in {"0", "false", "no", "off"}
        ),
        responses_i2i_wrapper_enabled=(
            responses_i2i_wrapper_enabled
            if responses_i2i_wrapper_enabled is not None
            else os.environ.get("SFT_RESPONSES_I2I_WRAPPER", "0").strip().lower()
            in {"1", "true", "yes", "on"}
        ),
        # #### END Response 0720 ####
    )


def build_runtime_context(
    *,
    working_dir: str | None = None,
    filename_prefix: str = "sft",
    case_id: str = "sft_session",
    image_registry: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ToolRuntimeContext:
    """Build the runtime context that tracks local artifacts and image refs."""

    return ToolRuntimeContext(
        working_dir=working_dir or os.path.join(os.getcwd(), "synthesis_sft_runs"),
        filename_prefix=filename_prefix,
        case_id=case_id,
        image_registry=dict(image_registry or {}),
        metadata=dict(metadata or {}),
    )


def run_agent_loop(
    *,
    prompt: str | None = None,
    messages: list[Message] | None = None,
    config: OpenAIToolAgentConfig | None = None,
    context: ToolRuntimeContext | None = None,
    system_prompt: str | None = None,
) -> list[Message]:
    """Run the multi-turn agent loop and return the full conversation history.

    The returned list contains the complete message trajectory:
    - system prompt
    - user prompt
    - every assistant turn
    - every tool output
    """

    return run_agent_session(
        prompt=prompt,
        messages=messages,
        config=config,
        context=context,
        system_prompt=system_prompt,
    ).messages


def run_agent_session(
    *,
    prompt: str | None = None,
    messages: list[Message] | None = None,
    config: OpenAIToolAgentConfig | None = None,
    context: ToolRuntimeContext | None = None,
    system_prompt: str | None = None,
) -> AgentRunResult:
    """Run the multi-turn agent loop and return messages plus generation metadata."""

    agent_config = config or build_agent_config(system_prompt=system_prompt)
    runtime_context = context or build_runtime_context()
    agent = OpenAIToolAgent(agent_config)
    return agent.run(
        prompt=prompt,
        messages=messages,
        context=runtime_context,
        system_prompt=system_prompt,
    )


def format_messages(messages: list[Message]) -> dict[str, Any]:
    """Format a trajectory into a natural-language content dict.

    The returned dict is intended for downstream prompting and inspection.
    It contains:
    - `text`: a readable step-by-step trajectory string
    - `images`: image references collected from the trajectory
    """

    formatted_blocks: list[str] = []
    images: list[dict[str, str]] = []
    seen_images: set[str] = set()

    for index, message in enumerate(messages, start=1):
        block_text, block_images = _format_single_message(index, message)
        if block_text:
            formatted_blocks.append(block_text)
        for image in block_images:
            image_value = image["image"]
            if image_value not in seen_images:
                seen_images.add(image_value)
                images.append(image)

    return {
        "text": "\n\n".join(formatted_blocks).strip(),
        "images": images,
    }


def extract_answer(messages: list[Message]) -> str:
    """Extract the final answer from the full trajectory.

    This uses a simple and stable rule for now:
    scan backward and return the last non-empty assistant text message.
    This is safer than assuming messages[-1] is always the final answer,
    because a trajectory may end with a tool message or an empty assistant turn.

    TODO: replace this heuristic with a more robust final-answer extractor.
    """

    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        text = _message_text(message.get("content"))
        if text:
            return text
    return ""


def check_hop_chain_coverage(
    messages: list[Message],
    hop_chain: list[dict[str, Any]],
    *,
    config: OpenAIToolAgentConfig | None = None,
) -> dict[str, Any]:
    """Ask a model to judge whether each hop appears in the trajectory."""

    formatted = format_messages(messages)
    prompt = _build_hop_check_prompt(formatted["text"], hop_chain)
    agent_config = config or build_agent_config(
        model=os.environ.get("SFT_JUDGE_MODEL") or os.environ.get("SFT_OPENAI_MODEL") or os.environ.get("OPENAI_MODEL"),
        api_key=os.environ.get("SFT_JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        azure_endpoint=(
            os.environ.get("SFT_JUDGE_AZURE_ENDPOINT")
            or os.environ.get("SFT_OPENAI_AZURE_ENDPOINT")
            or os.environ.get("SFT_OPENAI_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
        ),
        api_version=os.environ.get("SFT_JUDGE_API_VERSION") or os.environ.get("SFT_OPENAI_API_VERSION") or "2024-03-01-preview",
        max_tokens=int(os.environ.get("SFT_JUDGE_MAX_TOKENS", "4096")),
        temperature=_optional_env_float("SFT_JUDGE_TEMPERATURE"),
        system_prompt=(
            "You are a strict trajectory auditor. "
            "You inspect whether an agent trajectory truly covers each intended reasoning hop."
        ),
        print_rounds=False,
    )
    if _resolve_registered_model_alias(agent_config.model) is None:
        raise ValueError(
            "check_hop_chain_coverage requires `config.model` to be a registered LLM_WORKER model alias. "
            f"Got: {agent_config.model!r}"
        )
    response = LLM_WORKER.generate(
        ModelRequest(
            model=agent_config.model,
            messages=[
                ModelMessage(role="system", content=agent_config.system_prompt),
                ModelMessage(role="user", content=prompt),
            ],
            temperature=agent_config.temperature,
            max_tokens=agent_config.max_tokens,
            response_format={"type": "json_object"},
            metadata=_sft_worker_metadata(
                "hop_chain_coverage",
                extra_body=agent_config.extra_body,
            ),
        )
    )
    content = response.content or "{}"
    parsed = _try_parse_json_text(content)
    if not isinstance(parsed, dict):
        parsed = {
            "overall_complete": None,
            "summary": "Failed to parse judge output as JSON.",
            "missing_hop_indices": [],
            "hop_checks": [],
            "raw_text": content,
        }
    parsed["formatted_trajectory"] = formatted
    parsed["hop_chain"] = hop_chain
    return parsed


def judge(question: str, answer: str, extracted_answer: str) -> dict[str, Any]:
    """Judge whether the extracted answer matches the target answer.

    This is only a lightweight baseline judge so the pipeline can run end to end.
    It does not yet verify evidence sufficiency or hop-chain correctness.

    TODO: replace this heuristic with an expert-judge stage that checks:
    - answer correctness
    - evidence sufficiency
    - hop-chain consistency
    """

    normalized_gold = _normalize_answer_text(answer)
    normalized_pred = _normalize_answer_text(extracted_answer)
    is_exact_match = bool(normalized_gold) and normalized_gold == normalized_pred
    is_substring_match = bool(normalized_gold) and bool(normalized_pred) and (
        normalized_gold in normalized_pred or normalized_pred in normalized_gold
    )
    is_correct = is_exact_match or is_substring_match

    return {
        "question": question,
        "gold_answer": answer,
        "extracted_answer": extracted_answer,
        "normalized_gold_answer": normalized_gold,
        "normalized_extracted_answer": normalized_pred,
        "is_correct": is_correct,
        "match_type": (
            "exact"
            if is_exact_match
            else "substring"
            if is_substring_match
            else "mismatch"
        ),
        "reason": (
            "Heuristic baseline judge only. TODO: replace with expert answer/evidence/hop-chain judge."
        ),
    }
