"""Parse SFT runtime messages into causal tool/evidence provenance."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
from typing import Any, Iterator


_ACTION_RE = re.compile(r"<action>\s*(\{.*?\})\s*</action>", re.IGNORECASE | re.DOTALL)
_IMAGE_ID_RE = re.compile(r"\bimage_id\s*=\s*([^\s,;]+)", re.IGNORECASE)


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        values: list[str] = []
        for item in content:
            if isinstance(item, dict) and str(item.get("type") or "") in {"text", "input_text"}:
                values.append(str(item.get("text") or ""))
        return "\n".join(values).strip()
    return ""


def parse_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _manual_call(text: str) -> tuple[str, dict[str, Any]] | None:
    matches = _ACTION_RE.findall(text)
    if not matches:
        return None
    payload = parse_json(matches[-1])
    name = str(payload.get("tool_name") or "").strip()
    arguments = payload.get("arguments")
    return (name, dict(arguments)) if name and isinstance(arguments, dict) else None


def _native_calls(message: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any], str]]:
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else call
        name = str(function.get("name") or "").strip()
        arguments = parse_json(function.get("arguments"))
        call_id = str(call.get("id") or "").strip()
        if name:
            yield name, arguments, call_id


def _resource_ids(value: Any) -> set[str]:
    ids: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in {"image_id", "resource_id", "result_id"} and isinstance(nested, (str, int)):
                candidate = str(nested).strip()
                if candidate:
                    ids.add(candidate)
            else:
                ids.update(_resource_ids(nested))
    elif isinstance(value, list):
        for item in value:
            ids.update(_resource_ids(item))
    return ids


def _image_result_ids(value: Any) -> set[str]:
    """Only image IDs count as visual candidates; page IDs never do."""

    ids: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            if key == "image_id" and isinstance(nested, (str, int)):
                candidate = str(nested).strip()
                if candidate:
                    ids.add(candidate)
            else:
                ids.update(_image_result_ids(nested))
    elif isinstance(value, list):
        for item in value:
            ids.update(_image_result_ids(item))
    return ids


def _content_image_urls(content: Any) -> list[str]:
    if not isinstance(content, list):
        return []
    values: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("type") or "").lower()
        if kind not in {"image", "image_url", "image_path", "input_image"}:
            continue
        image = item.get("image_url") if kind == "image_url" else item.get("image") or item.get("path") or item.get("url")
        if isinstance(image, dict):
            image = image.get("url")
        if isinstance(image, str) and image.strip():
            values.append(image.strip())
    return values


@dataclass
class ToolEvent:
    index: int
    name: str
    arguments: dict[str, Any]
    output: dict[str, Any]
    ok: bool | None
    call_id: str = ""
    assistant_text: str = ""

    @property
    def resource_id(self) -> str:
        return str(self.arguments.get("resource_id") or self.arguments.get("image_id") or "").strip()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrajectoryProvenance:
    tool_events: list[ToolEvent] = field(default_factory=list)
    visual_candidate_ids: set[str] = field(default_factory=set)
    opened_candidate_ids: set[str] = field(default_factory=set)
    candidate_image_urls: dict[str, str] = field(default_factory=dict)
    assistant_texts: list[str] = field(default_factory=list)
    tool_errors: int = 0
    assistant_turns: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_events": [event.to_dict() for event in self.tool_events],
            "visual_candidate_ids": sorted(self.visual_candidate_ids),
            "opened_candidate_ids": sorted(self.opened_candidate_ids),
            "candidate_image_urls": dict(sorted(self.candidate_image_urls.items())),
            "assistant_texts": self.assistant_texts,
            "tool_errors": self.tool_errors,
            "assistant_turns": self.assistant_turns,
        }


def parse_trajectory(record: dict[str, Any]) -> TrajectoryProvenance:
    """Recover evidence order from manual-ReAct and native-function traces."""

    messages = record.get("raw_messages") or record.get("messages") or []
    result = TrajectoryProvenance()
    pending: list[tuple[str, dict[str, Any], str, str]] = []
    recent_read_resource = ""
    pending_image_attachment_ids: list[str] = []

    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").lower()
        if role == "assistant":
            text = message_text(message.get("content"))
            result.assistant_turns += 1
            if text:
                result.assistant_texts.append(text)
            pending.extend((name, args, call_id, text) for name, args, call_id in _native_calls(message))
            manual = _manual_call(text)
            if manual is not None and not any(name == manual[0] for name, _args, _call_id, _text in pending[-1:]):
                pending.append((manual[0], manual[1], "", text))
            continue

        if role == "tool":
            tool_name = str(message.get("name") or "").strip()
            call_id = str(message.get("tool_call_id") or "").strip()
            matched_index: int | None = None
            if call_id:
                for index, item in enumerate(pending):
                    if item[2] == call_id:
                        matched_index = index
                        break
            if matched_index is None:
                for index, item in enumerate(pending):
                    if item[0] == tool_name:
                        matched_index = index
                        break
            if matched_index is None:
                name, arguments, assistant_text = tool_name, {}, ""
            else:
                name, arguments, _matched_call_id, assistant_text = pending.pop(matched_index)
            output = parse_json(message.get("content"))
            ok_value = output.get("ok") if output else None
            event = ToolEvent(
                index=message_index,
                name=name or tool_name,
                arguments=arguments,
                output=output,
                ok=ok_value if isinstance(ok_value, bool) else None,
                call_id=call_id,
                assistant_text=assistant_text,
            )
            result.tool_events.append(event)
            if event.ok is False:
                result.tool_errors += 1
            if event.name in {"i2i_search", "t2i_search"} and event.ok is not False:
                result.visual_candidate_ids.update(_image_result_ids(output))
            if event.name == "read_url" and event.ok is not False:
                read_id = event.resource_id
                if read_id in result.visual_candidate_ids:
                    result.opened_candidate_ids.add(read_id)
                    recent_read_resource = read_id
                output_image_id = str(output.get("image_id") or "").strip()
                if output_image_id and read_id in result.visual_candidate_ids:
                    pending_image_attachment_ids.append(output_image_id)
            continue

        if role == "user":
            text = message_text(message.get("content"))
            declared_ids = _IMAGE_ID_RE.findall(text)
            image_urls = _content_image_urls(message.get("content"))
            for image_id, image_url in zip(declared_ids, image_urls):
                if image_id in pending_image_attachment_ids and recent_read_resource:
                    result.candidate_image_urls[recent_read_resource] = image_url
            continue
    return result


def has_rejection_language(assistant_texts: list[str]) -> bool:
    joined = "\n".join(assistant_texts).casefold()
    patterns = (
        "not the same",
        "does not match",
        "different image",
        "reject this",
        "not a match",
        "inconsistent",
        "cannot confirm",
        "doesn't match",
        "does not appear to",
    )
    return any(pattern in joined for pattern in patterns)
