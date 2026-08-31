"""Small, dependency-light helpers for the SFT Qwen3-VL conversation format.

The active SFT template is ``qwen3_vl_nothink`` in
``SFT/src/llamafactory/data/template.py``.  This module intentionally does not
import LLaMA-Factory, so the inference and RL packages can use the same
formatting rules without importing the SFT training stack.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Sequence


IMAGE_PLACEHOLDER = "<image>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
SYSTEM_ROLE = "system"
USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"
TOOL_RESPONSE_START = "<tool_response>"
TOOL_RESPONSE_END = "</tool_response>"


_QWEN_TOOL_PROMPT = (
    "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>{tool_text}"
    "\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
    "<tool_call></tool_call> XML tags:\n<tool_call>\n{{\"name\": <function-name>, "
    "\"arguments\": <args-json-object>}}\n</tool_call>"
)


def _tool_json(tool: Any) -> dict[str, Any] | str:
    """Return the JSON representation accepted by SFT's Qwen formatter."""

    if hasattr(tool, "json"):
        tool = tool.json
    if isinstance(tool, str):
        return tool
    if not isinstance(tool, dict):
        return str(tool)
    if tool.get("type") == "function":
        return tool
    return {"type": "function", "function": tool}


def format_sft_qwen_tool_prompt(tools: Iterable[Any]) -> str:
    """Format tools exactly like LLaMA-Factory's ``QwenToolUtils``."""

    tool_text = "".join(
        "\n" + json.dumps(_tool_json(tool), ensure_ascii=False) for tool in tools
    )
    return _QWEN_TOOL_PROMPT.format(tool_text=tool_text)


def add_sft_image_placeholders(content: str, image_count: int) -> str:
    """Put SFT-style image placeholders in a message's text.

    SFT's inference path adds placeholders when the caller supplies images but
    did not put placeholders in the message.  Dataset conversion uses one
    placeholder per line after the user question; that is the convention used
    here for newly constructed messages.
    """

    content = str(content or "")
    if image_count <= 0:
        return content

    existing = content.count(IMAGE_PLACEHOLDER)
    if existing == image_count:
        return content
    if existing:
        raise ValueError(
            f"Expected {image_count} {IMAGE_PLACEHOLDER!r} placeholders, found {existing}."
        )

    suffix = "\n".join([IMAGE_PLACEHOLDER] * image_count)
    return f"{content.rstrip()}\n{suffix}" if content.strip() else suffix


def interleave_sft_image_parts(
    parts: Sequence[tuple[str, Any]],
) -> list[tuple[str, Any]]:
    """Resolve ``<image>`` markers against image parts while preserving order.

    ``parts`` contains ``("text", value)`` and ``("image", value)`` items.
    If the number of markers equals the number of images, image parts are
    removed from their original transport position and inserted at their text
    markers.  This handles both SFT layouts used by the project:

    * ``image, question + <image>`` from the Gemini-style bootstrap path;
    * ``tool_response text + <image>, image`` from ``read_url``.

    When the input is incomplete (for example an image failed to resolve), the
    original parts are returned so the caller can retain its existing fallback
    behaviour instead of silently dropping content.
    """

    original = list(parts)
    images = [value for kind, value in original if kind == "image"]
    marker_count = sum(
        str(value or "").count(IMAGE_PLACEHOLDER)
        for kind, value in original
        if kind == "text"
    )
    if not images or marker_count == 0 or marker_count != len(images):
        return original

    result: list[tuple[str, Any]] = []
    image_index = 0
    for kind, value in original:
        if kind == "image":
            continue
        text = str(value or "")
        chunks = text.split(IMAGE_PLACEHOLDER)
        for index, chunk in enumerate(chunks):
            if chunk:
                result.append(("text", chunk))
            if index < len(chunks) - 1:
                result.append(("image", images[image_index]))
                image_index += 1
    return result


def render_sft_qwen3_vl_text(
    messages: Sequence[dict[str, Any]],
    *,
    tools: Iterable[Any] | None = None,
    add_generation_prompt: bool = False,
) -> str:
    """Render a text-only Qwen3-VL prompt using the active SFT structure.

    This is useful for RL's completion fallback and diagnostics.  Multimodal
    callers should keep image objects separate and use
    :func:`interleave_sft_image_parts` before invoking the model processor.
    """

    messages = list(messages)
    tool_prompt = format_sft_qwen_tool_prompt(tools or []) if tools else ""
    rendered: list[str] = []

    has_system = bool(messages and messages[0].get("role") == SYSTEM_ROLE)
    if not has_system and tool_prompt:
        rendered.append(f"{IM_START}{SYSTEM_ROLE}\n{tool_prompt}{IM_END}\n")

    for message in messages:
        role = str(message.get("role") or "")
        content = str(message.get("content") or "")
        if role == SYSTEM_ROLE:
            if tool_prompt and "<tools>" not in content:
                content += tool_prompt
            rendered.append(f"{IM_START}{SYSTEM_ROLE}\n{content}{IM_END}\n")
        elif role == USER_ROLE:
            rendered.append(f"{IM_START}{USER_ROLE}\n{content}{IM_END}\n")
        elif role == ASSISTANT_ROLE:
            rendered.append(f"{IM_START}{ASSISTANT_ROLE}\n{content}{IM_END}\n")
        elif role in {"tool", "observation"}:
            rendered.append(
                f"{IM_START}{USER_ROLE}\n{TOOL_RESPONSE_START}\n"
                f"{content}\n{TOOL_RESPONSE_END}{IM_END}\n"
            )
        else:
            raise NotImplementedError(f"Unsupported message role: {role}")

    if add_generation_prompt:
        rendered.append(f"{IM_START}{ASSISTANT_ROLE}\n")
    return "".join(rendered)


__all__ = [
    "ASSISTANT_ROLE",
    "IMAGE_PLACEHOLDER",
    "IM_END",
    "IM_START",
    "SYSTEM_ROLE",
    "TOOL_RESPONSE_END",
    "TOOL_RESPONSE_START",
    "USER_ROLE",
    "add_sft_image_placeholders",
    "format_sft_qwen_tool_prompt",
    "interleave_sft_image_parts",
    "render_sft_qwen3_vl_text",
]
