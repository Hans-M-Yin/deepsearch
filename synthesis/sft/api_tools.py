"""Runtime context, tool dispatcher, and OpenAI-based tool-calling agent."""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import requests
from PIL import Image

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "synthesis.sft"

from . import tools


logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = """You are an expert in answering multi-hop knowledge questions. You need to analyze each question carefully and, when necessary, use the appropriate tools to identify images, look for clues, and search the web for additional information.

At each turn, you must think step by step, provide a detailed analysis, and clearly plan the next action so that your reasoning process is reliable and trustworthy. Note that you may use only one tool per turn.

As for the available tools:
- t2t_search allows you to retrieve relevant web pages based on text and returns a list of URLs. You should examine the results, select the useful ones, and then use the read_url tool to access the page content.
- i2i_search allows you to search the web for similar images based on a selected region of an image, which is useful for identifying unfamiliar people or objects in the image. It also returns a list of URLs, which you should review and then inspect further using read_url.
- t2i_search allows you to retrieve relevant images based on a text description. As with the other search tools, you should review the returned URLs and then use read_url to inspect the images.

Once you believe the evidence is sufficient and there are no remaining unclear or uncertain points, provide the final answer and end the output.
"""


@dataclass(slots=True)
class ToolRuntimeContext:
    """Per-session runtime state used by the tool dispatcher."""

    working_dir: str
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    image_registry: dict[str, Any] = field(default_factory=dict)
    filename_prefix: str = "sft"
    case_id: str = "sft_session"
    visual_lookup: Callable[..., object] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _image_counter: int = 0

    def __post_init__(self) -> None:
        self.working_dir = os.path.abspath(self.working_dir)
        os.makedirs(self.working_dir, exist_ok=True)
        if self.image_registry:
            for key in self.image_registry:
                if key.startswith("img_"):
                    try:
                        self._image_counter = max(self._image_counter, int(key.split("_", 1)[1]))
                    except (TypeError, ValueError):
                        continue

    @property
    def intermediate_dir(self) -> str:
        path = os.path.join(self.working_dir, "artifacts")
        os.makedirs(path, exist_ok=True)
        return path

    def next_image_id(self) -> str:
        self._image_counter += 1
        return f"img_{self._image_counter}"

    def register_image(self, payload: Any) -> str:
        image_id = self.next_image_id()
        self.image_registry[image_id] = payload
        return image_id

    def image_summary(self) -> str:
        if not self.image_registry:
            return ""
        lines = ["Available image refs:"]
        for image_id, payload in self.image_registry.items():
            lines.append(f"- {image_id}: {str(payload)[:120]}")
        return "\n".join(lines)


@dataclass(slots=True)
class ToolExecutionResult:
    """Structured result for one tool invocation."""

    name: str
    arguments: dict[str, Any]
    output: dict[str, Any]
    output_text: str
    new_images: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OpenAIToolAgentConfig:
    """Configuration for OpenAI-compatible chat-completions tool calling."""

    model: str
    api_key: str | None = None
    client_type: str = "azure_openai"
    azure_endpoint: str | None = None
    base_url: str | None = None
    api_version: str = "2024-03-01-preview"
    api_mode: str = "chat_completions"
    max_tokens: int = 1024
    temperature: float | None = None
    timeout_s: float = 120.0
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    default_headers: dict[str, str] | None = None
    extra_body: dict[str, Any] | None = None
    max_turns: int = 8
    print_rounds: bool = True


@dataclass(slots=True)
class AgentRunResult:
    """Final answer plus the intermediate tool trace."""

    final_text: str
    messages: list[dict[str, Any]]
    tool_results: list[ToolExecutionResult]
    raw_responses: list[dict[str, Any]]


def _guess_mime_type(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".gif":
        return "image/gif"
    return "image/png"


def _normalize_region_bbox(region: object) -> tuple[tuple[int, int, int, int] | None, str | None]:
    if region in (None, ""):
        return None, None
    if isinstance(region, dict):
        if not all(key in region for key in ("x", "y", "width", "height")):
            return None, "Region dict must contain x, y, width, and height."
        try:
            x = int(region["x"])
            y = int(region["y"])
            width = int(region["width"])
            height = int(region["height"])
        except (TypeError, ValueError):
            return None, "Region dict values must be numeric."
        if width <= 0 or height <= 0:
            return None, "Region width and height must be positive."
        return (x, y, width, height), None
    if isinstance(region, (list, tuple)):
        if len(region) != 4:
            return None, "Region list must contain exactly 4 numbers."
        try:
            x1 = int(region[0])
            y1 = int(region[1])
            x2 = int(region[2])
            y2 = int(region[3])
        except (TypeError, ValueError):
            return None, "Region list values must be numeric."
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            return None, "Region list must be [x1, y1, x2, y2] with x2 > x1 and y2 > y1."
        return (x1, y1, width, height), None
    return None, "Region must be a 4-number list or a dict with x/y/width/height."


def _decode_data_url(data_url: str) -> bytes:
    payload = data_url.split("base64,", 1)[-1]
    return base64.b64decode(payload)


def _encode_data_url(data: bytes, mime_type: str) -> str:
    return f"data:{mime_type};base64,{base64.b64encode(data).decode('utf-8')}"


def _resolve_image_payload(source: Any, context: ToolRuntimeContext) -> Any:
    if isinstance(source, str) and source in context.image_registry:
        return context.image_registry[source]
    return source


def _image_source_to_model_url(source: Any, context: ToolRuntimeContext) -> str:
    payload = _resolve_image_payload(source, context)
    if isinstance(payload, str):
        if payload.startswith(("http://", "https://", "data:image")):
            return payload
        if os.path.exists(payload):
            with open(payload, "rb") as handle:
                return _encode_data_url(handle.read(), _guess_mime_type(payload))
    if isinstance(payload, bytes):
        return _encode_data_url(payload, "image/png")
    if isinstance(payload, Image.Image):
        buffer = io.BytesIO()
        payload.save(buffer, format="PNG")
        return _encode_data_url(buffer.getvalue(), "image/png")
    raise ValueError(f"Unsupported image source for model input: {type(payload)!r}")


def _normalize_content_part(part: Any, context: ToolRuntimeContext) -> dict[str, Any]:
    if not isinstance(part, dict):
        return {"type": "text", "text": str(part)}

    part_type = str(part.get("type") or "").strip()
    if part_type in {"text", "input_text"}:
        return {"type": "text", "text": str(part.get("text", ""))}

    if part_type == "image_url":
        image_url = part.get("image_url")
        detail = part.get("detail")
        if isinstance(image_url, dict):
            source = image_url.get("url", "")
        else:
            source = image_url
        normalized = {"type": "image_url", "image_url": {"url": _image_source_to_model_url(source, context)}}
        if detail:
            normalized["image_url"]["detail"] = detail
        return normalized

    if part_type in {"image", "input_image", "image_path", "image_ref"}:
        source = (
            part.get("image")
            or part.get("path")
            or part.get("url")
            or part.get("image_url")
            or part.get("ref")
        )
        normalized = {"type": "image_url", "image_url": {"url": _image_source_to_model_url(source, context)}}
        if part.get("detail"):
            normalized["image_url"]["detail"] = part["detail"]
        return normalized

    return dict(part)


def _append_system_text(message: dict[str, Any], extra_text: str) -> dict[str, Any]:
    content = message.get("content")
    if isinstance(content, str):
        updated = dict(message)
        updated["content"] = f"{content}\n\n{extra_text}" if content else extra_text
        return updated
    if isinstance(content, list):
        updated = dict(message)
        updated["content"] = list(content) + [{"type": "text", "text": extra_text}]
        return updated
    updated = dict(message)
    updated["content"] = extra_text
    return updated


def _normalize_message(message: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
    normalized = dict(message)
    content = normalized.get("content")
    if isinstance(content, list):
        normalized["content"] = [_normalize_content_part(part, context) for part in content]
    elif isinstance(content, dict):
        normalized["content"] = [_normalize_content_part(content, context)]
    elif content is None:
        normalized["content"] = ""
    else:
        normalized["content"] = content
    return normalized


def _build_initial_messages(
    *,
    prompt: str | None,
    messages: list[dict[str, Any]] | None,
    context: ToolRuntimeContext,
    system_prompt: str | None,
    default_system_prompt: str,
) -> list[dict[str, Any]]:
    if prompt is None and messages is None:
        raise ValueError("Either prompt or messages must be provided.")
    if prompt is not None and messages is not None:
        raise ValueError("Provide either prompt or messages, not both.")

    effective_system_prompt = system_prompt or default_system_prompt
    context_summary = context.image_summary()

    if messages is None:
        initial_messages: list[dict[str, Any]] = [
            {"role": "system", "content": effective_system_prompt},
            {"role": "user", "content": prompt or ""},
        ]
        if context_summary:
            initial_messages[0]["content"] = f"{initial_messages[0]['content']}\n\n{context_summary}"
        return initial_messages

    normalized_messages = [_normalize_message(message, context) for message in messages]
    has_system = any(message.get("role") == "system" for message in normalized_messages)
    if not has_system and effective_system_prompt:
        normalized_messages.insert(0, {"role": "system", "content": effective_system_prompt})
        has_system = True
    if context_summary:
        if has_system:
            for index, message in enumerate(normalized_messages):
                if message.get("role") == "system":
                    normalized_messages[index] = _append_system_text(message, context_summary)
                    break
        else:
            normalized_messages.insert(0, {"role": "system", "content": context_summary})
    return normalized_messages


def _load_pil_image(source: Any, context: ToolRuntimeContext) -> Image.Image:
    payload = _resolve_image_payload(source, context)
    if isinstance(payload, Image.Image):
        return payload.copy()
    if isinstance(payload, bytes):
        return Image.open(io.BytesIO(payload))
    if isinstance(payload, str):
        if payload.startswith("data:image"):
            return Image.open(io.BytesIO(_decode_data_url(payload)))
        if payload.startswith(("http://", "https://")):
            response = requests.get(payload, timeout=60)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content))
        if os.path.exists(payload):
            return Image.open(payload)
    raise ValueError(f"Unsupported image source: {type(payload)!r}")


def _persist_pil_image(
    image: Image.Image,
    context: ToolRuntimeContext,
    tool_name: str,
) -> tuple[str, str]:
    image_id = context.next_image_id()
    filename = f"{context.filename_prefix}_{context.session_id}_{tool_name}_{image_id}.png"
    save_path = os.path.join(context.intermediate_dir, filename)
    image.save(save_path)
    context.image_registry[image_id] = save_path
    return image_id, save_path


def _try_upload_pil_image(
    image: Image.Image,
    context: ToolRuntimeContext,
    tool_name: str,
) -> str | None:
    try:
        from opensearch_vl.opensearch_infer import cos_upload
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.debug("COS uploader import failed: %s", exc)
        return None

    try:
        return cos_upload.upload_pil_image(
            image,
            context.filename_prefix,
            0,
            0,
            tool_name,
        )
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("COS upload failed for %s: %s", tool_name, exc)
        return None


def _materialize_remote_image_url(source: Any, context: ToolRuntimeContext, tool_name: str) -> tuple[str | None, str | None]:
    payload = _resolve_image_payload(source, context)
    if isinstance(payload, str) and payload.startswith(("http://", "https://")):
        return payload, None

    try:
        image = _load_pil_image(payload, context)
    except Exception as exc:
        return None, f"Failed to load image for {tool_name}: {exc}"

    uploaded_url = _try_upload_pil_image(image, context, tool_name)
    if uploaded_url:
        return uploaded_url, None
    return None, (
        f"{tool_name} requires a publicly reachable image URL. "
        "Uploading the local image failed; configure the optional COS uploader first."
    )


def _assistant_message_for_followup(message: Any) -> dict[str, Any]:
    tool_calls = []
    for index, tool_call in enumerate(getattr(message, "tool_calls", None) or []):
        tool_calls.append(
            {
                "index": index,
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
        )
    assistant_message = {
        "role": "assistant",
        "tool_calls": tool_calls,
    }
    if getattr(message, "content", None):
        assistant_message["content"] = message.content
    return assistant_message


def _assistant_message_for_followup_from_dict(
    *,
    content: str,
    tool_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    assistant_message: dict[str, Any] = {
        "role": "assistant",
        "tool_calls": tool_calls,
    }
    if content:
        assistant_message["content"] = content
    return assistant_message


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _format_message_content(content: Any) -> str:
    if content in (None, ""):
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        formatted_parts: list[str] = []
        for index, part in enumerate(content, start=1):
            if not isinstance(part, dict):
                formatted_parts.append(f"[part {index}] {part}")
                continue
            part_type = str(part.get("type") or "")
            if part_type in {"text", "input_text"}:
                formatted_parts.append(str(part.get("text", "")))
            elif part_type == "image_url":
                image_url = part.get("image_url")
                if isinstance(image_url, dict):
                    url = image_url.get("url", "")
                else:
                    url = image_url
                formatted_parts.append(f"[image_url] {url}")
            elif part_type in {"image", "input_image", "image_path", "image_ref"}:
                source = (
                    part.get("image")
                    or part.get("path")
                    or part.get("url")
                    or part.get("image_url")
                    or part.get("ref")
                    or ""
                )
                formatted_parts.append(f"[{part_type}] {source}")
            else:
                formatted_parts.append(_json_text(part))
        return "\n".join(item for item in formatted_parts if item)
    return _json_text(content) if isinstance(content, (dict, tuple)) else str(content)


def _print_conversation_trace(messages: list[dict[str, Any]]) -> None:
    print("\n=== Conversation Trace ===")
    for index, message in enumerate(messages, start=1):
        role = str(message.get("role") or "unknown")
        print(f"\n[{index}] {role}")
        content_text = _format_message_content(message.get("content"))
        if content_text:
            print(content_text)
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            print("tool_calls:")
            print(_json_text(tool_calls))
        if role == "tool":
            tool_name = message.get("name")
            tool_call_id = message.get("tool_call_id")
            if tool_name:
                print(f"name: {tool_name}")
            if tool_call_id:
                print(f"tool_call_id: {tool_call_id}")


def _print_round_output(turn_index: int, assistant_message: Any) -> None:
    print(f"\n=== Model Round {turn_index + 1} ===")
    content = getattr(assistant_message, "content", None)
    if content:
        print(content)
    tool_calls = getattr(assistant_message, "tool_calls", None) or []
    if tool_calls:
        print("tool_calls:")
        for index, tool_call in enumerate(tool_calls):
            print(
                _json_text(
                    {
                        "index": index,
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                )
            )


def _print_round_output_from_responses(
    turn_index: int,
    *,
    content: str,
    tool_calls: list[dict[str, Any]],
) -> None:
    print(f"\n=== Model Round {turn_index + 1} ===")
    if content:
        print(content)
    if tool_calls:
        print("tool_calls:")
        for tool_call in tool_calls:
            print(_json_text(tool_call))


def _message_content_to_responses_content(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    if isinstance(content, list):
        normalized_parts: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                normalized_parts.append({"type": "input_text", "text": str(part)})
                continue
            part_type = part.get("type")
            if part_type == "text":
                normalized_parts.append({"type": "input_text", "text": str(part.get("text", ""))})
            elif part_type == "input_text":
                normalized_parts.append({"type": "input_text", "text": str(part.get("text", ""))})
            elif part_type == "image_url":
                image_url = part.get("image_url")
                if isinstance(image_url, dict):
                    url = image_url.get("url", "")
                    detail = image_url.get("detail")
                else:
                    url = image_url
                    detail = part.get("detail")
                item = {"type": "input_image", "image_url": url}
                if detail:
                    item["detail"] = detail
                normalized_parts.append(item)
            elif part_type == "input_image":
                normalized_parts.append(dict(part))
            else:
                normalized_parts.append(dict(part))
        return normalized_parts
    if content is None:
        return [{"type": "input_text", "text": ""}]
    return [{"type": "input_text", "text": str(content)}]


def _messages_to_responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "user")
        if role == "tool":
            continue
        items.append(
            {
                "role": role,
                "content": _message_content_to_responses_content(message.get("content")),
            }
        )
    return items


def _conversation_messages_to_responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert the local conversation history to Responses API input items."""

    items: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "user")
        if role in {"system", "user", "assistant"}:
            content = message.get("content")
            has_textual_content = bool(content not in (None, "", []))
            if has_textual_content:
                items.append(
                    {
                        "role": role,
                        "content": _message_content_to_responses_content(content),
                    }
                )
            if role == "assistant":
                for tool_call in message.get("tool_calls") or []:
                    if not isinstance(tool_call, dict):
                        continue
                    function = tool_call.get("function") or {}
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": tool_call.get("id", ""),
                            "name": function.get("name", ""),
                            "arguments": function.get("arguments", "{}"),
                        }
                    )
        elif role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id", ""),
                    "output": message.get("content", ""),
                }
            )
    return items


def _extract_responses_content_and_tool_calls(raw_response: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    output_items = raw_response.get("output") or []
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for index, item in enumerate(output_items):
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "message":
            for content_item in item.get("content") or []:
                if not isinstance(content_item, dict):
                    continue
                if content_item.get("type") in {"output_text", "text"}:
                    text = content_item.get("text")
                    if text:
                        text_parts.append(str(text))
        elif item_type == "function_call":
            call_id = item.get("call_id") or item.get("id") or ""
            tool_calls.append(
                {
                    "index": len(tool_calls),
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", "{}"),
                    },
                }
            )

    return "\n".join(part for part in text_parts if part).strip(), tool_calls


def _is_previous_response_not_found_error(exc: Exception) -> bool:
    message = str(exc)
    return "previous_response_not_found" in message or "Previous response with id" in message


def execute_tool_call(
    name: str,
    arguments: dict[str, Any],
    context: ToolRuntimeContext,
) -> ToolExecutionResult:
    """Execute one tool call against the runtime context."""

    params = tools.normalize_tool_arguments(name, arguments)

    if name == "t2t_search":
        query = params.get("q") or params.get("query") or ""
        if not query:
            output = {"ok": False, "error": "q is required for t2t_search"}
        else:
            output = tools.t2t_search(
                query=query,
                lang=params.get("hl", "en"),
                top_k=int(params.get("top_k", 5)),
            )
        return ToolExecutionResult(name=name, arguments=params, output=output, output_text=_json_text(output))

    if name == "t2i_search":
        query = params.get("q") or params.get("query") or ""
        if not query:
            output = {"ok": False, "error": "q is required for t2i_search"}
        else:
            output = tools.t2i_search(
                query=query,
                lang=params.get("hl", "en"),
                top_k=int(params.get("top_k", 5)),
            )
        return ToolExecutionResult(name=name, arguments=params, output=output, output_text=_json_text(output))

    if name == "read_url":
        url = params.get("url") or params.get("URL") or ""
        if not url:
            output = {"ok": False, "error": "url is required for read_url"}
            return ToolExecutionResult(name=name, arguments=params, output=output, output_text=_json_text(output))
        output = tools.read_url(url=url, query=params.get("query", "") or "")
        new_images: dict[str, Any] = {}
        if output.get("ok") and output.get("kind") == "image" and output.get("local_path"):
            image_id = context.register_image(output["local_path"])
            output = dict(output)
            output["image_id"] = image_id
            new_images[image_id] = output["local_path"]
        return ToolExecutionResult(name=name, arguments=params, output=output, output_text=_json_text(output), new_images=new_images)

    if name == "i2i_search":
        image_source = params.get("image") or params.get("url") or ""
        if not image_source:
            output = {"ok": False, "error": "image or url is required for i2i_search"}
            return ToolExecutionResult(name=name, arguments=params, output=output, output_text=_json_text(output))

        region = params.get("region")
        new_images: dict[str, Any] = {}
        if region not in (None, ""):
            bbox, err = _normalize_region_bbox(region)
            if err:
                output = {"ok": False, "error": err}
                return ToolExecutionResult(name=name, arguments=params, output=output, output_text=_json_text(output))
            assert bbox is not None
            image = _load_pil_image(image_source, context)
            x, y, width, height = bbox
            cropped = image.crop((x, y, x + width, y + height))
            cropped_id, cropped_path = _persist_pil_image(cropped, context, "i2i_region")
            new_images[cropped_id] = cropped_path
            uploaded_url = _try_upload_pil_image(cropped, context, "i2i_region")
            if not uploaded_url:
                output = {
                    "ok": False,
                    "error": (
                        "Cropped region was created, but reverse image search needs a public URL. "
                        "The optional uploader is not available."
                    ),
                    "cropped_image_id": cropped_id,
                    "cropped_image_path": cropped_path,
                }
                return ToolExecutionResult(name=name, arguments=params, output=output, output_text=_json_text(output), new_images=new_images)
            output = tools.i2i_search(
                image_url=uploaded_url,
                visual_lookup=context.visual_lookup,
            )
            output = dict(output)
            output["cropped_image_id"] = cropped_id
            output["cropped_image_path"] = cropped_path
            output["cropped_image_url"] = uploaded_url
            return ToolExecutionResult(name=name, arguments=params, output=output, output_text=_json_text(output), new_images=new_images)

        remote_url, err = _materialize_remote_image_url(image_source, context, "i2i_search")
        if err:
            output = {"ok": False, "error": err}
            return ToolExecutionResult(name=name, arguments=params, output=output, output_text=_json_text(output))
        output = tools.i2i_search(
            image_url=remote_url or "",
            visual_lookup=context.visual_lookup,
        )
        return ToolExecutionResult(name=name, arguments=params, output=output, output_text=_json_text(output))

    output = {"ok": False, "error": f"Unknown tool: {name}"}
    return ToolExecutionResult(name=name, arguments=params, output=output, output_text=_json_text(output))


class OpenAIToolAgent:
    """OpenAI/AzureOpenAI-based multi-turn chat-completions agent with tool calling."""

    def __init__(self, config: OpenAIToolAgentConfig) -> None:
        try:
            from openai import AzureOpenAI, OpenAI
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "OpenAIToolAgent requires the `openai` package."
            ) from exc

        self.config = config
        api_key = config.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        client_type = str(config.client_type or "azure_openai").strip().lower()
        if client_type == "openai":
            base_url = (
                config.base_url
                or os.environ.get("SFT_OPENAI_BASE_URL")
                or os.environ.get("OPENAI_BASE_URL")
            )
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=config.timeout_s,
                default_headers=config.default_headers,
            )
        else:
            azure_endpoint = (
                config.azure_endpoint
                or config.base_url
                or os.environ.get("SFT_OPENAI_AZURE_ENDPOINT")
                or os.environ.get("SFT_OPENAI_BASE_URL")
                or os.environ.get("OPENAI_BASE_URL")
            )
            self.client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                api_version=config.api_version,
                timeout=config.timeout_s,
                default_headers=config.default_headers,
            )

    def run(
        self,
        prompt: str | None = None,
        *,
        messages: list[dict[str, Any]] | None = None,
        context: ToolRuntimeContext | None = None,
        system_prompt: str | None = None,
    ) -> AgentRunResult:
        if self.config.api_mode == "responses":
            return self._run_responses(
                prompt=prompt,
                messages=messages,
                context=context,
                system_prompt=system_prompt,
            )
        return self._run_chat_completions(
            prompt=prompt,
            messages=messages,
            context=context,
            system_prompt=system_prompt,
        )

    def _run_chat_completions(
        self,
        *,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        context: ToolRuntimeContext | None = None,
        system_prompt: str | None = None,
    ) -> AgentRunResult:
        context = context or ToolRuntimeContext(working_dir=os.getcwd())
        conversation_messages = _build_initial_messages(
            prompt=prompt,
            messages=messages,
            context=context,
            system_prompt=system_prompt,
            default_system_prompt=self.config.system_prompt,
        )
        tool_results: list[ToolExecutionResult] = []
        raw_responses: list[dict[str, Any]] = []
        final_text = ""

        for turn_index in range(self.config.max_turns):
            kwargs: dict[str, Any] = {
                "model": self.config.model,
                "messages": conversation_messages,
                "tools": tools.get_tool_definitions(),
                "max_tokens": self.config.max_tokens,
                "stream": False,
            }
            if self.config.temperature is not None:
                kwargs["temperature"] = self.config.temperature
            if self.config.extra_body:
                kwargs["extra_body"] = self.config.extra_body

            completion = self.client.chat.completions.create(**kwargs)
            raw_responses.append(
                completion.model_dump() if hasattr(completion, "model_dump") else {"repr": repr(completion)}
            )
            choice = completion.choices[0]
            assistant_message = choice.message
            if self.config.print_rounds:
                _print_round_output(turn_index, assistant_message)
            tool_calls = assistant_message.tool_calls or []

            if not tool_calls:
                final_text = assistant_message.content or ""
                conversation_messages.append({"role": "assistant", "content": final_text})
                break

            conversation_messages.append(_assistant_message_for_followup(assistant_message))
            for tool_call in tool_calls:
                parsed_args = json.loads(tool_call.function.arguments or "{}")
                result = execute_tool_call(tool_call.function.name, parsed_args, context)
                tool_results.append(result)
                conversation_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": result.output_text,
                        "type": "function",
                    }
                )
        else:
            final_text = "Max tool-calling turns reached before the model produced a final answer."

        return AgentRunResult(
            final_text=final_text,
            messages=conversation_messages,
            tool_results=tool_results,
            raw_responses=raw_responses,
        )

    def _run_responses(
        self,
        *,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        context: ToolRuntimeContext | None = None,
        system_prompt: str | None = None,
    ) -> AgentRunResult:
        context = context or ToolRuntimeContext(working_dir=os.getcwd())
        conversation_messages = _build_initial_messages(
            prompt=prompt,
            messages=messages,
            context=context,
            system_prompt=system_prompt,
            default_system_prompt=self.config.system_prompt,
        )
        tool_results: list[ToolExecutionResult] = []
        raw_responses: list[dict[str, Any]] = []
        final_text = ""
        current_input = _messages_to_responses_input(conversation_messages)
        previous_response_id: str | None = None
        use_previous_response_id = True

        for turn_index in range(self.config.max_turns):
            kwargs: dict[str, Any] = {
                "model": self.config.model,
                "input": current_input,
                "tools": tools.get_responses_tool_definitions(),
            }
            if self.config.max_tokens is not None:
                kwargs["max_output_tokens"] = self.config.max_tokens
            if self.config.extra_body:
                kwargs["extra_body"] = self.config.extra_body
            if previous_response_id and use_previous_response_id:
                kwargs["previous_response_id"] = previous_response_id

            try:
                response = self.client.responses.create(**kwargs)
            except Exception as exc:
                if previous_response_id and use_previous_response_id and _is_previous_response_not_found_error(exc):
                    logger.warning(
                        "Responses API previous_response_id is unavailable on this backend; "
                        "falling back to full-context replay."
                    )
                    use_previous_response_id = False
                    kwargs.pop("previous_response_id", None)
                    kwargs["input"] = _conversation_messages_to_responses_input(conversation_messages)
                    response = self.client.responses.create(**kwargs)
                else:
                    raise
            raw_response = response.model_dump() if hasattr(response, "model_dump") else {"repr": repr(response)}
            raw_responses.append(raw_response)
            if use_previous_response_id:
                previous_response_id = raw_response.get("id") or getattr(response, "id", None)
            else:
                previous_response_id = None

            assistant_content, assistant_tool_calls = _extract_responses_content_and_tool_calls(raw_response)
            if self.config.print_rounds:
                _print_round_output_from_responses(
                    turn_index,
                    content=assistant_content,
                    tool_calls=assistant_tool_calls,
                )

            if not assistant_tool_calls:
                final_text = assistant_content
                conversation_messages.append({"role": "assistant", "content": final_text})
                break

            conversation_messages.append(
                _assistant_message_for_followup_from_dict(
                    content=assistant_content,
                    tool_calls=assistant_tool_calls,
                )
            )

            current_input = []
            for tool_call in assistant_tool_calls:
                parsed_args = json.loads(tool_call["function"]["arguments"] or "{}")
                result = execute_tool_call(tool_call["function"]["name"], parsed_args, context)
                tool_results.append(result)
                conversation_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": tool_call["function"]["name"],
                        "content": result.output_text,
                        "type": "function",
                    }
                )
                current_input.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call["id"],
                        "output": result.output_text,
                    }
                )
            if not use_previous_response_id:
                current_input = _conversation_messages_to_responses_input(conversation_messages)
        else:
            final_text = "Max tool-calling turns reached before the model produced a final answer."

        return AgentRunResult(
            final_text=final_text,
            messages=conversation_messages,
            tool_results=tool_results,
            raw_responses=raw_responses,
        )


def _parse_json_flag(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object.")
    return parsed


def _parse_messages_json(value: str | None) -> list[dict[str, Any]] | None:
    if not value:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON array of messages.")
    for item in parsed:
        if not isinstance(item, dict):
            raise ValueError("Each message must be a JSON object.")
    return parsed


def _build_context_from_args(args: argparse.Namespace) -> ToolRuntimeContext:
    context = ToolRuntimeContext(
        working_dir=args.workdir,
        filename_prefix=args.filename_prefix,
        case_id=args.case_id,
    )
    for path in args.image or []:
        image_id = context.register_image(os.path.abspath(path))
        logger.info("Registered local image %s -> %s", image_id, path)
    for url in args.image_url or []:
        image_id = context.register_image(url)
        logger.info("Registered remote image %s -> %s", image_id, url)
    return context


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenAI-based synthesis SFT tool-calling framework.")
    parser.add_argument("--prompt", help="Simple user prompt sent to the model.")
    parser.add_argument("--messages-json", help="Full messages list as a JSON array.")
    parser.add_argument("--messages-file", help="Path to a JSON file containing a full messages list.")
    parser.add_argument("--model", default=os.environ.get("SFT_OPENAI_MODEL") or os.environ.get("OPENAI_MODEL") or "")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument(
        "--api-mode",
        choices=("chat_completions", "responses"),
        default=os.environ.get("SFT_OPENAI_API_MODE") or "chat_completions",
    )
    parser.add_argument(
        "--azure-endpoint",
        default=(
            os.environ.get("SFT_OPENAI_AZURE_ENDPOINT")
            or os.environ.get("SFT_OPENAI_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
        ),
    )
    parser.add_argument(
        "--api-version",
        default=os.environ.get("SFT_OPENAI_API_VERSION") or "2024-03-01-preview",
    )
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("SFT_OPENAI_MAX_TOKENS", "1024")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("SFT_OPENAI_TEMPERATURE", "0.2")))
    parser.add_argument("--max-turns", type=int, default=int(os.environ.get("SFT_OPENAI_MAX_TURNS", "8")))
    parser.add_argument("--timeout-s", type=float, default=float(os.environ.get("SFT_OPENAI_TIMEOUT_S", "120")))
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--headers-json", default=os.environ.get("SFT_OPENAI_HEADERS_JSON"))
    parser.add_argument("--extra-body-json", default=os.environ.get("SFT_OPENAI_EXTRA_BODY_JSON"))
    parser.add_argument("--workdir", default=os.path.join(os.getcwd(), "synthesis_sft_runs"))
    parser.add_argument("--filename-prefix", default="sft")
    parser.add_argument("--case-id", default="sft_session")
    parser.add_argument("--image", action="append", help="Preload a local image path as img_n.")
    parser.add_argument("--image-url", action="append", help="Preload a remote image URL as img_n.")
    parser.add_argument("--gpt54", action="store_true", help="Use the GPT-5.4 chat.completions branch from .sft_env.")
    parser.add_argument(
        "--gemini35-flash",
        action="store_true",
        help="Use the Gemini 3.5 Flash chat.completions branch from .sft_env.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.model:
        parser.error("--model is required (or set SFT_OPENAI_MODEL / OPENAI_MODEL).")
    if not any([args.prompt, args.messages_json, args.messages_file]):
        parser.error("One of --prompt, --messages-json, or --messages-file is required.")
    if sum(1 for item in [args.prompt, args.messages_json, args.messages_file] if item) > 1:
        parser.error("Use only one of --prompt, --messages-json, or --messages-file.")
    if args.gpt54 and args.gemini35_flash:
        parser.error("Use only one model shortcut: --gpt54 or --gemini35-flash.")

    if args.gpt54:
        args.api_mode = "chat_completions"
        args.model = os.environ.get("SFT_GPT54_MODEL") or "gpt-5.4-2026-03-05"
        args.api_key = os.environ.get("SFT_GPT54_API_KEY") or args.api_key
        args.azure_endpoint = os.environ.get("SFT_GPT54_AZURE_ENDPOINT") or args.azure_endpoint
        args.api_version = os.environ.get("SFT_GPT54_API_VERSION") or args.api_version
    elif args.gemini35_flash:
        args.api_mode = "chat_completions"
        args.model = os.environ.get("SFT_GEMINI35_FLASH_MODEL") or "gemini-3.5-flash"
        args.api_key = os.environ.get("SFT_GEMINI35_FLASH_API_KEY") or args.api_key
        args.azure_endpoint = os.environ.get("SFT_GEMINI35_FLASH_AZURE_ENDPOINT") or args.azure_endpoint
        args.api_version = os.environ.get("SFT_GEMINI35_FLASH_API_VERSION") or args.api_version

    config = OpenAIToolAgentConfig(
        model=args.model,
        api_key=args.api_key,
        azure_endpoint=args.azure_endpoint,
        api_version=args.api_version,
        api_mode=args.api_mode,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout_s=args.timeout_s,
        system_prompt=args.system_prompt,
        default_headers=_parse_json_flag(args.headers_json),
        extra_body=_parse_json_flag(args.extra_body_json),
        max_turns=args.max_turns,
    )
    context = _build_context_from_args(args)
    agent = OpenAIToolAgent(config)
    input_messages = _parse_messages_json(args.messages_json)
    if args.messages_file:
        input_messages = _parse_messages_json(Path(args.messages_file).read_text(encoding="utf-8"))
    result = agent.run(prompt=args.prompt, messages=input_messages, context=context)

    _print_conversation_trace(result.messages)

    print("=== Final Answer ===")
    print(result.final_text)
    if result.tool_results:
        print("\n=== Tool Trace ===")
        for idx, item in enumerate(result.tool_results, start=1):
            print(f"[{idx}] {item.name}")
            print("arguments:")
            print(_json_text(item.arguments))
            print("result:")
            print(item.output_text)
            if item.new_images:
                print("new_images:")
                print(_json_text(item.new_images))
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
