"""Runtime context, tool dispatcher, and OpenAI-based tool-calling agent."""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import re
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

from synthesis.model_worker import LLM_WORKER
from synthesis.model_worker import ModelMessage
from synthesis.model_worker import ModelRequest
from . import tools


logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = """
You are writing a standard answer for a multi-hop knowledge question. Specifically, based on the question provided to you, you need to produce a complete solution process that includes scientifically rigorous, logically sound reasoning steps. This solution process should contain analysis and reasoning about the question, tool calls, analysis and reflection on tool results, replanning of the solution steps, multiple search attempts, and a final accurate standard answer.

Requirements:
1. You may think freely during your internal reasoning phase, but the statements ultimately included in the written solution process must also follow rigorous logic, ensuring that the solution remains sound and error-free even if one reads only the written solution process and ignores your private thinking.
2. Since no correct answer is provided, you must also correctly solve the question while drafting the standard answer.
3. In the standard answer you write, the following logic should be explicitly visible: after each tool call and its returned result, you must carefully analyze the new clues in detail, review the existing clues and the question, determine and plan the next step in detail, and then call a new tool as needed with an explanation.
4. Your standard answer should be written from the perspective of someone with strong logical reasoning but no memory of world knowledge or history, so every statement must be evidence-based and no unsupported claims may appear. Every statement in your writing should be detailedly analysed or discussed. 
5. Once you believe the evidence is sufficient and there are no remaining unclear or uncertain points, provide the final answer and end the standard answer.
6. In your standard answer, DO NOT use tools to directly search for pages related to Wikipedia or Wiki Commons, in order to avoid shortcuts. However, you can read related Wikipedia or Wiki Commons pages which are the results of the search tools.

**Examples**
Bad writing:

Based on the text and watermarks visible in the provided image, the stock photography agency is Alamy. The question asks about a specific photograph from a different media repository that Alamy is known to source content from. My first step is to identify this repository.
<action>
{
  "tool_name": "t2t_search",
  "params": {
    "query": "Alamy sources content from Wikimedia Commons"
  },
  "goal": "To determine if Alamy sources content from the freely licensed media repository, Wikimedia Commons, as suggested by the question's description."
}
</action>

Discuss: In this example, thee answer never mentioned Wiki Commons during the analysis stage, yet it directly searched whether Alamy is related to Wiki Commons. At that point, Wiki Commons was an unsupported clue that appeared out of nowhere, which violates Rule 4. A better version of the writing would be:

Based on the text and watermarks visible in the provided image, the stock photography agency is Alamy. The question asks about a specific photograph from a different media repository that Alamy is known to source content from. My first step is to identify this repository. Since the clue given in the question is "a large, freely licensed media repository," Wiki Commons may be a possible answer, but there is no evidence yet. So for now, I should first search which repository Alamy sources content from.
<action>
{
"tool_name": "t2t_search",
"params": {
"query": "The large repository Alamy sources content from"
},
"goal": "Confirm which large freely licensed media repository Alamy sources content from."
}
</action>

Discuss: In this version, the answer is more logically rigorous, the reasoning is more careful, and there are no clues appearing from nowhere. You should learn from this style of writing and avoid bad writing like the earlier version.
****
"""

MANUAL_REACT_PROTOCOL = """
When you writing the standard answer, you can use tools following these useful tips:

1. t2t_search returns a list of URLs for text pages. You should examine the results, select the useful ones, and then use the read_url tool to access the page content. Use this tool when you need to look up world knowledge or content information.
2. i2i_search is very useful for identifying unfamiliar people or objects in the image. Note that the return results might be not related to your original image, so you should first select the search results that are likely to match your current image according to the textual title, then use `read_url` to download those images and inspect their content. Once you determine that the new image and the previous image depict the same object, you can then use `read_url` again to read the linked page of the new image to figure out who or what that object is. You should reflect this logic in the standard answer.
3. t2i_search retrieves relevant images based on a text description. You should review the returned information and then use read_url to inspect those images. Use this tool when the missing clues require you to inspect relevant images, or when the images you find are likely to help you answer the question. Note that after using this tool, the searched images are still not provided to you, and you should use `read_url` to inspect the corresponding images.

You must answer exactly one step at a time. Then end your response with exactly one action block in the following format:

<action>
{
  "tool_name": "tool_name",
  "params": {
    "query": "your query here"
  },
  "goal": "why this tool is the right next step"
}
</action>

Rules:
- Output exactly one <action>...</action> block in each round.
- The content inside <action> must be valid JSON.
- The JSON must contain exactly these top-level keys: tool_name, params, goal.

"""

_MANUAL_REACT_ACTIONS = {"t2t_search", "t2i_search", "i2i_search", "read_url", "finish"}
_MANUAL_REACT_ACTION_RE = re.compile(r"<action>\s*(?P<json>\{.*?\})\s*</action>", re.DOTALL | re.IGNORECASE)
_I2I_WRAPPER_DEFAULT_MODEL_ALIAS = "multimodal_process"
_I2I_WRAPPER_MAX_TOKENS = 2048

PROMPT_I2I_REWRITE_ASSISTANT = """
I will give you an image and a passage containing analysis and tool-call process text for a certain question. This passage is missing context. Your goal is to determine, based only on this single passage, which object in the image the passage is focusing on. Then, summarize that object as a noun phrase (possibly with a descriptive referring expression). Finally, polish the parts of the passage that are related to tool calling so that the logic becomes tighter and more coherent.

Rules:
1. Only polish the text related to tool calling detailedly, such as the purpose of calling the tool, the motivation, what is intended to be searched, and so on. Do not modify other content or the overall logic.
2. When polishing, besides making the logic more rigorous and detailed, you may also appropriately add text describing that the next step is to locate the target object of interest. But make sure the logic remains rigorous. Do not use structured polishment such as "Goal: ...".
3. Any text that is kept unchanged must remain exactly the same as the original in content, format, and even punctuation.
4. Output in the following format:
...Think process first...
<object>The entity in the image that this passage is trying to find</object>
<refined>The polished text</refined>

Example:
Input: "To answer this question, I need to follow a multi-step process. First, I need to identify the celestial body shown in the image to determine the orbiter that discovered its prominent equatorial ridge. Once the orbiter is identified, I can find its launch vehicle program. Then, I will research the three consecutive launch failures of that program between August 1998 and April 1999 and find the distinct root cause for each.

My first step is to use the provided image to identify the celestial body.

<action>
{
  "tool_name": "i2i_search",
  "params": {},
  "goal": "Identify the celestial body in the image, which will help in identifying the orbiter that discovered the equatorial ridge."
}
</action>"
Your output:
...The detailed thinking process is ignored in this example, but you should think step by step in your response...
<object>Celestial body</object>
<refined>To answer this question, I need to follow a multi-step process. First, I need to identify the celestial body shown in the image to determine the orbiter that discovered its prominent equatorial ridge. Once the orbiter is identified, I can find its launch vehicle program. Then, I will research the three consecutive launch failures of that program between August 1998 and April 1999 and find the distinct root cause for each. So based on the above plan, for the input image, I first need to locate the position of this celestial body within the image, crop out the relevant local region, and pass its position to the i2i_search tool so that I can search for this celestial body. Ideally, by using similar images and their descriptions, I can determine exactly which celestial body it is.

<action>
{
  "tool_name": "i2i_search",
  "params": {},
  "goal": "Identify the celestial body in the image, which will help in identifying the orbiter that discovered the equatorial ridge."
}
</action></refined>
"""

PROMPT_I2I_GROUND_OBJECT = """
You need to localize a target object in an image for reverse image search.
Please return one bounding box for the target object using normalized coordinates on a 0-1000 scale:
Format: [x1, y1, x2, y2]
If most of the image consists of the target object, or if the target is not a clearly defined entity (for example, it is a description of a scene), please return the full image:
[0, 0, 1000, 1000]
Please return strict JSON:
{
"label": "...",
"bbox": [x1, y1, x1, y1]
}
"""
_NORMALIZED_COORD_SCALE = 1000.0


def _env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _prepare_region_for_crop(region: object, image_size: tuple[int, int]) -> object:
    """Convert model-style normalized coordinates into absolute crop coordinates."""

    if not isinstance(region, (list, tuple)) or len(region) != 4:
        return region

    try:
        coords = [float(value) for value in region]
    except (TypeError, ValueError):
        return region

    if _env_flag("REVERSE_IMAGE_CROP_COORDS"):
        coords = [coords[1], coords[0], coords[3], coords[2]]

    image_width, image_height = image_size
    x1 = int(round(coords[0] / _NORMALIZED_COORD_SCALE * image_width))
    y1 = int(round(coords[1] / _NORMALIZED_COORD_SCALE * image_height))
    x2 = int(round(coords[2] / _NORMALIZED_COORD_SCALE * image_width))
    y2 = int(round(coords[3] / _NORMALIZED_COORD_SCALE * image_height))

    x1 = min(max(x1, 0), image_width)
    y1 = min(max(y1, 0), image_height)
    x2 = min(max(x2, 0), image_width)
    y2 = min(max(y2, 0), image_height)
    return [x1, y1, x2, y2]


def _truncate_tool_calls(tool_calls: list[Any], *, source: str) -> list[Any]:
    if len(tool_calls) > 1:
        logger.warning(
            "Expected at most one tool call per turn; keeping only the first from %s and dropping %d extra call(s).",
            source,
            len(tool_calls) - 1,
        )
        return tool_calls[:1]
    return tool_calls


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

    def latest_image_reference(self) -> str | None:
        if not self.image_registry:
            return None
        return next(reversed(self.image_registry))

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
    api_mode: str = "manual_react"
    max_tokens: int | None = 1024
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


@dataclass(slots=True)
class ManualReActStep:
    thought: str
    action: str
    action_input: dict[str, Any]
    raw_text: str


@dataclass(slots=True)
class I2IRepairResult:
    assistant_text: str
    display_arguments: dict[str, Any]
    execution_arguments: dict[str, Any]
    target_object: str
    used_full_image: bool = False


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


def _resolve_registered_model_alias(alias_or_model: str | None) -> dict[str, Any] | None:
    if not alias_or_model:
        return None
    try:
        return LLM_WORKER.get_model(alias_or_model)
    except Exception:
        return None


def _i2i_wrapper_model_alias() -> str | None:
    alias = os.environ.get("SFT_I2I_WRAPPER_MODEL") or _I2I_WRAPPER_DEFAULT_MODEL_ALIAS
    return alias if _resolve_registered_model_alias(alias) is not None else None


def _i2i_wrapper_max_tokens() -> int:
    raw_value = os.environ.get("SFT_I2I_WRAPPER_MAX_TOKENS")
    if raw_value is None or str(raw_value).strip() == "":
        return _I2I_WRAPPER_MAX_TOKENS
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return _I2I_WRAPPER_MAX_TOKENS


def _worker_generate_json_message(
    *,
    model_alias: str,
    system_prompt: str,
    user_content: Any,
    max_tokens: int,
    trace_label: str,
) -> dict[str, Any]:
    response = LLM_WORKER.generate(
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
    parsed = _extract_json_object(response.content or "")
    if parsed is None:
        raise ValueError(f"Model response is not valid JSON: {response.content[:500]}")
    return parsed


def _worker_generate_text_message(
    *,
    model_alias: str,
    system_prompt: str,
    user_content: Any,
    max_tokens: int,
    trace_label: str,
) -> str:
    response = LLM_WORKER.generate(
        ModelRequest(
            model=model_alias,
            messages=[
                ModelMessage(role="system", content=system_prompt),
                ModelMessage(role="user", content=user_content),
            ],
            max_tokens=max_tokens,
            metadata={"trace_label": trace_label},
        )
    )
    return response.content or ""


def _extract_xml_tag_content(text: str, tag_name: str) -> str:
    pattern = rf"<{re.escape(tag_name)}>\s*(.*?)\s*</{re.escape(tag_name)}>"
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).strip()


def _strip_action_blocks(text: str) -> str:
    if not text:
        return ""
    cleaned = _MANUAL_REACT_ACTION_RE.sub("", text)
    return cleaned.strip()


def _latest_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            chunks: list[str] = []
            for part in content:
                if isinstance(part, dict) and str(part.get("type") or "") in {"text", "input_text"}:
                    text = str(part.get("text", "")).strip()
                    if text:
                        chunks.append(text)
            if chunks:
                return "\n".join(chunks).strip()
    return ""


def _parse_manual_react_goal(text: str) -> str:
    matches = list(_MANUAL_REACT_ACTION_RE.finditer(text.strip()))
    if not matches:
        return ""
    action_payload = _extract_json_object(matches[-1].group("json"))
    if not isinstance(action_payload, dict):
        return ""
    return str(action_payload.get("goal") or "").strip()


def _render_manual_react_text(
    *,
    thought: str,
    action: str,
    params: dict[str, Any],
    goal: str,
) -> str:
    payload = {
        "tool_name": action,
        "params": params,
        "goal": goal,
    }
    action_text = json.dumps(payload, ensure_ascii=False, indent=2)
    cleaned_thought = thought.strip()
    if cleaned_thought:
        return f"{cleaned_thought}\n\n<action>\n{action_text}\n</action>"
    return f"<action>\n{action_text}\n</action>"


def _clip_box_coord(value: Any) -> int:
    try:
        numeric = int(round(float(value)))
    except (TypeError, ValueError):
        return 0
    return min(max(numeric, 0), int(_NORMALIZED_COORD_SCALE))


def _full_image_box_xyxy() -> list[int]:
    max_coord = int(_NORMALIZED_COORD_SCALE)
    return [0, 0, max_coord, max_coord]


def _normalize_xyxy_box_1000(raw_box: Any) -> tuple[list[int], bool]:
    if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
        return _full_image_box_xyxy(), True
    x1, y1, x2, y2 = [_clip_box_coord(value) for value in raw_box]
    if x2 <= x1 or y2 <= y1:
        return _full_image_box_xyxy(), True
    return [x1, y1, x2, y2], False


def _xyxy_to_yxyx(box_xyxy: list[int]) -> list[int]:
    return [int(box_xyxy[1]), int(box_xyxy[0]), int(box_xyxy[3]), int(box_xyxy[2])]


def _maybe_repair_i2i_tool_call(
    *,
    assistant_text: str,
    tool_name: str,
    tool_arguments: dict[str, Any],
    context: ToolRuntimeContext,
    question_text: str,
) -> I2IRepairResult | None:
    if tool_name != "i2i_search":
        return None

    model_alias = _i2i_wrapper_model_alias()
    if not model_alias:
        return None

    image_source = tool_arguments.get("image") or tool_arguments.get("url") or context.latest_image_reference() or ""
    if not image_source:
        return None

    try:
        image = _load_pil_image(image_source, context)
        image_width, image_height = image.size
    except Exception as exc:
        logger.warning("Failed to load image for i2i wrapper repair: %s", exc)
        return None

    try:
        image_url = _image_source_to_model_url(image_source, context)
    except Exception as exc:
        logger.warning("Failed to materialize image input for i2i wrapper repair: %s", exc)
        return None

    wrapper_max_tokens = _i2i_wrapper_max_tokens()
    question = question_text.strip()
    original_text = assistant_text.strip()
    original_region = tool_arguments.get("region")

    try:
        rewrite_payload = {
            "question": question,
            "assistant_text": original_text,
            "current_tool_name": tool_name,
            "current_tool_arguments": tool_arguments,
            "current_region": original_region,
            "image_size": {"width": image_width, "height": image_height},
            "required_sentence_style": (
                "Make it explicit what object in the image needs to be identified first and why reverse image search helps."
            ),
        }
        rewrite_text = _worker_generate_text_message(
            model_alias=model_alias,
            system_prompt=PROMPT_I2I_REWRITE_ASSISTANT,
            user_content=[
                {"type": "text", "text": json.dumps(rewrite_payload, ensure_ascii=False, indent=2)},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
            max_tokens=wrapper_max_tokens,
            trace_label=f"i2i_wrapper_rewrite:{context.case_id}",
        )
        target_object = _extract_xml_tag_content(rewrite_text, "object")
        revised_assistant_text = _strip_action_blocks(
            _extract_xml_tag_content(rewrite_text, "refined")
        ) or original_text
        if not target_object:
            target_object = "the relevant object in the image"
    except Exception as exc:
        logger.warning("i2i wrapper rewrite failed: %s", exc)
        return None

    try:
        grounding_payload = {
            "question": question,
            "assistant_text": revised_assistant_text,
            "target_object": target_object,
            "current_region": original_region,
            "image_size": {"width": image_width, "height": image_height},
            "coordinate_format": {
                "required_output": "[x1, y1, x2, y2]",
                "normalized_scale": [0, int(_NORMALIZED_COORD_SCALE)],
            },
        }
        grounding_result = _worker_generate_json_message(
            model_alias=model_alias,
            system_prompt=PROMPT_I2I_GROUND_OBJECT,
            user_content=[
                {"type": "text", "text": json.dumps(grounding_payload, ensure_ascii=False, indent=2)},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
            max_tokens=wrapper_max_tokens,
            trace_label=f"i2i_wrapper_ground:{context.case_id}",
        )
    except Exception as exc:
        logger.warning("i2i wrapper grounding failed: %s", exc)
        grounding_result = {"bbox": _full_image_box_xyxy(), "used_full_image": True}

    bbox_xyxy, invalid_box = _normalize_xyxy_box_1000(
        grounding_result.get("bbox")
        or grounding_result.get("bbox_xyxy")
        or grounding_result.get("region")
    )
    used_full_image = bool(grounding_result.get("used_full_image")) or invalid_box

    display_arguments = dict(tool_arguments)
    display_arguments["region"] = _xyxy_to_yxyx(bbox_xyxy)

    execution_arguments = dict(tool_arguments)
    execution_arguments["region"] = (
        list(display_arguments["region"])
        if _env_flag("REVERSE_IMAGE_CROP_COORDS")
        else list(bbox_xyxy)
    )

    return I2IRepairResult(
        assistant_text=revised_assistant_text,
        display_arguments=display_arguments,
        execution_arguments=execution_arguments,
        target_object=target_object,
        used_full_image=used_full_image,
    )


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


def _ensure_inline_image_registered(source: Any, context: ToolRuntimeContext) -> None:
    if not isinstance(source, str) or not source.startswith("data:image"):
        return
    if source in context.image_registry:
        return
    for payload in context.image_registry.values():
        if payload == source:
            return
    context.register_image(source)


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
        _ensure_inline_image_registered(source, context)
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
        _ensure_inline_image_registered(source, context)
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


def _tool_reference_text() -> str:
    lines = ["Available tools and their full definitions:"]
    for item in tools.get_tool_definitions():
        function = item["function"]
        lines.append(f"- {function['name']}:")
        lines.append(json.dumps(function, ensure_ascii=False, indent=2, sort_keys=True))
    lines.append('- finish: End the trajectory. Full definition:')
    lines.append(
        json.dumps(
            {
                "name": "finish",
                "description": "End the trajectory and provide the final answer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "The final answer text.",
                        }
                    },
                    "required": ["answer"],
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return "\n".join(lines)


def _latest_tool_message(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for message in reversed(messages):
        if message.get("role") == "tool":
            return message
    return None


def _contains_image_context(messages: list[dict[str, Any]], context: ToolRuntimeContext) -> bool:
    if context.image_registry:
        return True
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if str(part.get("type") or "") in {"image_url", "image", "input_image", "image_path", "image_ref"}:
                return True
    return False


def _build_state_guidance(messages: list[dict[str, Any]], context: ToolRuntimeContext) -> str:
    latest_tool = _latest_tool_message(messages)
    has_images = _contains_image_context(messages, context)
    guidance: list[str] = ["Recommendation for Tool Use:"]
    if latest_tool is None:
        if has_images:
            guidance.extend(
                [
                    "- There is image context available.",
                    "- You may inspect whether the current image already contains the target evidence.",
                    "- If a specific local object matters, consider i2i_search with a region crop first. Then i2i_search will provide recognization of the object.",
                    "- If the entity is generic or the question mainly asks for background knowledge, a text search may be better than reverse image search.",
                ]
            )
        else:
            guidance.extend(
                [
                    "- Start by identifying the first missing piece of evidence.",
                    "- Prefer t2t_search for textual evidence gathering and read_url for inspecting a specific result.",
                ]
            )
        return "\n".join(guidance)

    tool_name = str(latest_tool.get("name") or "")
    output_text = str(latest_tool.get("content") or "")
    output_obj: Any = None
    try:
        output_obj = json.loads(output_text)
    except Exception:
        output_obj = None

    if isinstance(output_obj, dict) and output_obj.get("ok") is False:
        guidance.extend(
            [
                f"- The previous tool `{tool_name}` failed.",
                "- Fix the parameter problem instead of repeating the same invalid action.",
                "- Explain briefly why the previous call failed before choosing the next action.",
            ]
        )
        return "\n".join(guidance)

    if tool_name == "t2t_search":
        guidance.extend(
            [
                "- You now have a list of text-search results.",
                "- Prefer selecting one promising URL and using read_url, rather than repeating a very similar search immediately.",
                "- Only search again if the returned results are clearly off-target or ambiguous.",
            ]
        )
    elif tool_name == "t2i_search":
        guidance.extend(
            [
                "- You now have image-search results.",
                "- You may inspect the image result pages with read_url or use the new clues to refine the search.",
                "- If the question is really about a specific pictured object, consider whether reverse image search is needed next.",
            ]
        )
    elif tool_name == "i2i_search":
        guidance.extend(
            [
                "- You now have reverse-image matches.",
                "- Use the matched titles and source URLs to decide whether to inspect a specific source with read_url.",
                "- If the reverse-image results are too noisy, fall back to text search with the strongest visual clue.",
            ]
        )
    elif tool_name == "read_url":
        if isinstance(output_obj, dict) and output_obj.get("kind") == "image":
            guidance.extend(
                [
                    "- The last URL resolved to an image.",
                    "- Decide whether this image itself contains the target evidence.",
                    "- If a local object matters, i2i_search on the image or a cropped region may help.",
                ]
            )
        else:
            guidance.extend(
                [
                    "- The last URL provided page content.",
                    "- Decide whether the page already supports the current claim strongly enough.",
                    "- If not, either inspect another URL or run a refined search to fill the remaining gap.",
                ]
            )
    else:
        guidance.append("- Choose the next action based on the strongest remaining evidence gap.")
    return "\n".join(guidance)


def _build_manual_react_system_prompt(
    *,
    base_system_prompt: str,
    messages: list[dict[str, Any]],
    context: ToolRuntimeContext,
) -> str:
    # Dynamic state guidance is disabled for now so the prompt stays stable across turns.
    parts = [base_system_prompt.strip(), _tool_reference_text(), MANUAL_REACT_PROTOCOL.strip()]
    return "\n\n".join(part for part in parts if part).strip()


def _build_manual_react_request_messages(
    conversation_messages: list[dict[str, Any]],
    context: ToolRuntimeContext,
    base_system_prompt: str,
) -> list[dict[str, Any]]:
    system_seed = base_system_prompt
    for message in conversation_messages:
        if message.get("role") == "system" and message.get("content"):
            system_seed = str(message.get("content"))
            break
    full_system_prompt = _build_manual_react_system_prompt(
        base_system_prompt=system_seed,
        messages=conversation_messages,
        context=context,
    )
    request_messages: list[dict[str, Any]] = []
    system_replaced = False
    for message in conversation_messages:
        role = message.get("role")
        if role == "tool":
            tool_name = str(message.get("name") or "tool").strip()
            observation_text = str(message.get("content") or "")
            copied = {
                "role": "user",
                "content": f"Observation from {tool_name}:\n{observation_text}",
            }
        else:
            copied = dict(message)
        if copied.get("role") == "system" and not system_replaced:
            copied["content"] = full_system_prompt
            system_replaced = True
        request_messages.append(copied)
    if not system_replaced:
        request_messages.insert(0, {"role": "system", "content": full_system_prompt})
    return request_messages


def _apply_system_prompt_to_messages(
    messages: list[dict[str, Any]],
    system_prompt: str,
) -> list[dict[str, Any]]:
    updated_messages: list[dict[str, Any]] = []
    system_replaced = False
    for message in messages:
        copied = dict(message)
        if copied.get("role") == "system" and not system_replaced:
            copied["content"] = system_prompt
            system_replaced = True
        updated_messages.append(copied)
    if not system_replaced:
        updated_messages.insert(0, {"role": "system", "content": system_prompt})
    return updated_messages


def _strip_code_fence(text: str) -> str:
    candidate = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", candidate, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return candidate


def _extract_json_object(text: str) -> dict[str, Any] | None:
    candidate = _strip_code_fence(text)
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(candidate[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _complete_trailing_action_block(text: str, finish_reason: str | None) -> str:
    """Close a trailing <action> block if generation stopped at the stop sequence."""

    stripped = text.rstrip()
    lower = stripped.lower()
    last_open = lower.rfind("<action>")
    last_close = lower.rfind("</action>")
    if last_open == -1 or last_close > last_open:
        return stripped
    if finish_reason != "stop":
        return stripped
    if stripped.endswith("}"):
        return f"{stripped}\n</action>"
    return f"{stripped}</action>"


def _parse_manual_react_step(text: str) -> ManualReActStep | None:
    stripped = text.strip()
    matches = list(_MANUAL_REACT_ACTION_RE.finditer(stripped))
    if not matches:
        return None
    match = matches[-1]
    thought = stripped[: match.start()].strip()
    action_payload = _extract_json_object(match.group("json"))
    if not isinstance(action_payload, dict):
        return None
    action = str(action_payload.get("tool_name") or "").strip()
    params = action_payload.get("params")
    goal = str(action_payload.get("goal") or "").strip()
    if action not in _MANUAL_REACT_ACTIONS or not isinstance(params, dict):
        return None
    normalized_text = stripped[: match.end()].strip()
    if goal:
        thought = f"{thought}\n\nGoal: {goal}".strip() if thought else f"Goal: {goal}"
    return ManualReActStep(
        thought=thought,
        action=action,
        action_input=params,
        raw_text=normalized_text,
    )


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

    if messages is None:
        initial_messages: list[dict[str, Any]] = [
            {"role": "system", "content": effective_system_prompt},
            {"role": "user", "content": prompt or ""},
        ]
        return initial_messages

    normalized_messages = [_normalize_message(message, context) for message in messages]
    has_system = any(message.get("role") == "system" for message in normalized_messages)
    if not has_system and effective_system_prompt:
        normalized_messages.insert(0, {"role": "system", "content": effective_system_prompt})
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


def _persist_pil_image_to_cache(
    image: Image.Image,
    context: ToolRuntimeContext,
    tool_name: str,
) -> tuple[str, str]:
    image_id = context.next_image_id()
    cache_dir = Path(__file__).resolve().parents[1] / ".image_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{context.filename_prefix}_{context.session_id}_{tool_name}_{image_id}.png"
    save_path = str(cache_dir / filename)
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
    for tool_call in getattr(message, "tool_calls", None) or []:
        tool_calls.append(
            {
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
        for tool_call in tool_calls:
            print(
                _json_text(
                    {
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
    question_text: str = "",
    assistant_text: str = "",
) -> ToolExecutionResult:
    """Execute one tool call against the runtime context."""

    params = tools.normalize_tool_arguments(name, arguments)

    if name == "t2t_search":
        query = params.get("query") or params.get("q") or ""
        if not query:
            output = {"ok": False, "error": "query is required for t2t_search"}
        else:
            output = tools.t2t_search(
                query=query,
                lang=params.get("lang") or params.get("hl") or "en",
                top_k=int(params.get("top_k", 5)),
            )
        return ToolExecutionResult(name=name, arguments=params, output=output, output_text=_json_text(output))

    if name == "t2i_search":
        query = params.get("query") or params.get("q") or ""
        if not query:
            output = {"ok": False, "error": "query is required for t2i_search"}
        else:
            output = tools.t2i_search(
                query=query,
                lang=params.get("lang") or params.get("hl") or "en",
                top_k=int(params.get("top_k", 5)),
            )
        return ToolExecutionResult(name=name, arguments=params, output=output, output_text=_json_text(output))

    if name == "read_url":
        url = params.get("url") or params.get("URL") or ""
        if not url:
            output = {"ok": False, "error": "url is required for read_url"}
            return ToolExecutionResult(name=name, arguments=params, output=output, output_text=_json_text(output))
        output = tools.read_url(
            url=url,
            query=params.get("query", "") or "",
            question_text=question_text,
            assistant_output=assistant_text,
        )
        new_images: dict[str, Any] = {}
        if output.get("ok") and output.get("local_path"):
            image_id = context.register_image(output["local_path"])
            output = dict(output)
            output["image_id"] = image_id
            new_images[image_id] = output["local_path"]
        return ToolExecutionResult(name=name, arguments=params, output=output, output_text=_json_text(output), new_images=new_images)

    if name == "i2i_search":
        image_source = params.get("image") or params.get("url") or context.latest_image_reference() or ""
        if not image_source:
            output = {
                "ok": False,
                "error": "No image is available in the current context for i2i_search.",
            }
            return ToolExecutionResult(name=name, arguments=params, output=output, output_text=_json_text(output))

        region = params.get("region")
        new_images: dict[str, Any] = {}
        if region not in (None, ""):
            image = _load_pil_image(image_source, context)
            region = _prepare_region_for_crop(region, image.size)
            bbox, err = _normalize_region_bbox(region)
            if err:
                output = {"ok": False, "error": err}
                return ToolExecutionResult(name=name, arguments=params, output=output, output_text=_json_text(output))
            assert bbox is not None
            x, y, width, height = bbox
            cropped = image.crop((x, y, x + width, y + height))
            uploaded_url = _try_upload_pil_image(cropped, context, "i2i_region")
            if not uploaded_url:
                cropped_id, cropped_path = _persist_pil_image_to_cache(cropped, context, "i2i_region")
                new_images[cropped_id] = cropped_path
                output = {
                    "ok": False,
                    "error": (
                        "Cropped region was created, but reverse image search needs a public URL. "
                        "Uploading the cropped image failed, so it was saved to the local image cache instead."
                    ),
                    "cropped_image_id": cropped_id,
                    "cropped_image_path": cropped_path,
                }
                return ToolExecutionResult(name=name, arguments=params, output=output, output_text=_json_text(output), new_images=new_images)
            context.register_image(uploaded_url)
            output = tools.i2i_search(
                image_url=uploaded_url,
                visual_lookup=context.visual_lookup,
                top_k=int(params.get("top_k", 5)),
            )
            output = dict(output)
            output["cropped_image_url"] = uploaded_url
            return ToolExecutionResult(name=name, arguments=params, output=output, output_text=_json_text(output), new_images=new_images)

        remote_url, err = _materialize_remote_image_url(image_source, context, "i2i_search")
        if err:
            output = {"ok": False, "error": err}
            return ToolExecutionResult(name=name, arguments=params, output=output, output_text=_json_text(output))
        output = tools.i2i_search(
            image_url=remote_url or "",
            visual_lookup=context.visual_lookup,
            top_k=int(params.get("top_k", 5)),
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
        if self.config.api_mode == "manual_react":
            return self._run_manual_react(
                prompt=prompt,
                messages=messages,
                context=context,
                system_prompt=system_prompt,
            )
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

    def _run_manual_react(
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
            request_messages = _build_manual_react_request_messages(
                conversation_messages,
                context,
                system_prompt or self.config.system_prompt,
            )
            kwargs: dict[str, Any] = {
                "model": self.config.model,
                "messages": request_messages,
                "stream": False,
                "stop": ["</action>"],
            }
            if self.config.max_tokens is not None:
                kwargs["max_tokens"] = self.config.max_tokens
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
            assistant_text = _complete_trailing_action_block(
                assistant_message.content or "",
                getattr(choice, "finish_reason", None),
            )
            if self.config.print_rounds:
                print(f"\n=== Model Round {turn_index + 1} ===")
                if assistant_text:
                    print(assistant_text)

            step = _parse_manual_react_step(assistant_text)
            if step is None:
                conversation_messages.append({"role": "assistant", "content": assistant_text})
                logger.warning("Failed to parse manual ReAct step; treating the latest assistant text as final output.")
                final_text = assistant_text
                break
            repaired_step_text = step.raw_text
            execution_action_input = step.action_input
            if step.action == "i2i_search":
                repaired = _maybe_repair_i2i_tool_call(
                    assistant_text=step.thought or step.raw_text,
                    tool_name=step.action,
                    tool_arguments=step.action_input,
                    context=context,
                    question_text=_latest_user_text(conversation_messages),
                )
                if repaired is not None:
                    repaired_step_text = _render_manual_react_text(
                        thought=repaired.assistant_text,
                        action=step.action,
                        params=repaired.display_arguments,
                        goal=_parse_manual_react_goal(step.raw_text),
                    )
                    execution_action_input = repaired.execution_arguments
            conversation_messages.append({"role": "assistant", "content": repaired_step_text})
            if step.action == "finish":
                final_text = str(step.action_input.get("answer") or step.raw_text).strip()
                break

            result = execute_tool_call(
                step.action,
                execution_action_input,
                context,
                question_text=_latest_user_text(conversation_messages),
                assistant_text=_strip_action_blocks(repaired_step_text),
            )
            tool_results.append(result)
            conversation_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"manual_{turn_index + 1}_{step.action}",
                    "name": step.action,
                    "content": result.output_text,
                    "type": "manual_react_tool",
                }
            )
        else:
            final_text = "Max ReAct turns reached before the model produced a final answer."

        if request_messages:
            effective_system_prompt = str(request_messages[0].get("content") or "")
            conversation_messages = _apply_system_prompt_to_messages(
                conversation_messages,
                effective_system_prompt,
            )

        return AgentRunResult(
            final_text=final_text,
            messages=conversation_messages,
            tool_results=tool_results,
            raw_responses=raw_responses,
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
            print('##############\n',kwargs,'\n##############')
            completion = self.client.chat.completions.create(**kwargs)
            raw_responses.append(
                completion.model_dump() if hasattr(completion, "model_dump") else {"repr": repr(completion)}
            )
            choice = completion.choices[0]
            assistant_message = choice.message
            if self.config.print_rounds:
                _print_round_output(turn_index, assistant_message)
            tool_calls = _truncate_tool_calls(list(assistant_message.tool_calls or []), source="chat_completions")

            if not tool_calls:
                final_text = assistant_message.content or ""
                conversation_messages.append({"role": "assistant", "content": final_text})
                break

            assistant_content = assistant_message.content or ""
            followup_tool_calls: list[dict[str, Any]] = []
            execution_payloads: list[tuple[str, dict[str, Any], str]] = []
            for tool_call in tool_calls:
                parsed_args = json.loads(tool_call.function.arguments or "{}")
                display_args = parsed_args
                execution_args = parsed_args
                if tool_call.function.name == "i2i_search":
                    repaired = _maybe_repair_i2i_tool_call(
                        assistant_text=assistant_content,
                        tool_name=tool_call.function.name,
                        tool_arguments=parsed_args,
                        context=context,
                        question_text=_latest_user_text(conversation_messages),
                    )
                    if repaired is not None:
                        assistant_content = repaired.assistant_text
                        display_args = repaired.display_arguments
                        execution_args = repaired.execution_arguments
                followup_tool_calls.append(
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": json.dumps(display_args, ensure_ascii=False),
                        },
                    }
                )
                execution_payloads.append((tool_call.function.name, execution_args, tool_call.id))
            conversation_messages.append(
                _assistant_message_for_followup_from_dict(
                    content=assistant_content,
                    tool_calls=followup_tool_calls,
                )
            )
            for tool_name, execution_args, tool_call_id in execution_payloads:
                result = execute_tool_call(
                    tool_name,
                    execution_args,
                    context,
                    question_text=_latest_user_text(conversation_messages),
                    assistant_text=assistant_content,
                )
                tool_results.append(result)
                conversation_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
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
            assistant_tool_calls = _truncate_tool_calls(assistant_tool_calls, source="responses")
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

            followup_tool_calls: list[dict[str, Any]] = []
            execution_payloads: list[tuple[str, dict[str, Any], str]] = []
            current_input = []
            for tool_call in assistant_tool_calls:
                parsed_args = json.loads(tool_call["function"]["arguments"] or "{}")
                display_args = parsed_args
                execution_args = parsed_args
                if tool_call["function"]["name"] == "i2i_search":
                    repaired = _maybe_repair_i2i_tool_call(
                        assistant_text=assistant_content,
                        tool_name=tool_call["function"]["name"],
                        tool_arguments=parsed_args,
                        context=context,
                        question_text=_latest_user_text(conversation_messages),
                    )
                    if repaired is not None:
                        assistant_content = repaired.assistant_text
                        display_args = repaired.display_arguments
                        execution_args = repaired.execution_arguments
                followup_tool_calls.append(
                    {
                        "id": tool_call["id"],
                        "type": "function",
                        "function": {
                            "name": tool_call["function"]["name"],
                            "arguments": json.dumps(display_args, ensure_ascii=False),
                        },
                    }
                )
                execution_payloads.append((tool_call["function"]["name"], execution_args, tool_call["id"]))
            conversation_messages.append(
                _assistant_message_for_followup_from_dict(
                    content=assistant_content,
                    tool_calls=followup_tool_calls,
                )
            )
            for tool_name, execution_args, tool_call_id in execution_payloads:
                result = execute_tool_call(
                    tool_name,
                    execution_args,
                    context,
                    question_text=_latest_user_text(conversation_messages),
                    assistant_text=assistant_content,
                )
                tool_results.append(result)
                conversation_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": result.output_text,
                        "type": "function",
                    }
                )
                current_input.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call_id,
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
        choices=("manual_react", "chat_completions", "responses"),
        default=os.environ.get("SFT_OPENAI_API_MODE") or "manual_react",
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
    parser.add_argument(
        "--temperature",
        type=float,
        default=(float(os.environ["SFT_OPENAI_TEMPERATURE"]) if os.environ.get("SFT_OPENAI_TEMPERATURE") else None),
    )
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
    parser.add_argument("--gpt54", action="store_true", help="Use the GPT-5.4 manual-ReAct branch from .sft_env.")
    parser.add_argument(
        "--gemini35-flash",
        action="store_true",
        help="Use the Gemini 3.5 Flash manual-ReAct branch from .sft_env.",
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
        args.api_mode = "manual_react"
        args.model = os.environ.get("SFT_GPT54_MODEL") or "gpt-5.4-2026-03-05"
        args.api_key = os.environ.get("SFT_GPT54_API_KEY") or args.api_key
        args.azure_endpoint = os.environ.get("SFT_GPT54_AZURE_ENDPOINT") or args.azure_endpoint
        args.api_version = os.environ.get("SFT_GPT54_API_VERSION") or args.api_version
    elif args.gemini35_flash:
        args.api_mode = "manual_react"
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
