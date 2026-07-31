"""Convert debug_vqa_batch raw trajectories into manual-ReAct SFT parquet data.

Example:
    python -m synthesis.sft.export_manual_react_dataset \
      --input-jsonl /path/to/raw_trajectories.jsonl \
      --output-dir /path/to/exported_manual_react_dataset
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import mimetypes
import re
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from PIL import Image

from synthesis.sft import tools as sft_tools


_ACTION_RE = re.compile(r"<action>\s*(?P<json>\{.*?\})\s*</action>", re.DOTALL | re.IGNORECASE)
_THINKING_RE = re.compile(r"<thinking>\s*(.*?)\s*</thinking>", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_RESPONSE_RE = re.compile(r"<response>\s*(.*?)\s*</response>", re.DOTALL | re.IGNORECASE)
_IMAGE_EXT_BY_MIME = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
}
_DEFAULT_GOAL_IN_ARGS_TOOLS = {"read_url"}


@dataclass
class ConversionError:
    sample_id: str
    message: str


@dataclass
class SampleImageStore:
    image_dir: Path
    sample_key: str
    saved_paths: list[str] = field(default_factory=list)
    _source_to_relpath: dict[str, str] = field(default_factory=dict)
    _counter: int = 0

    def save(self, source: Any, *, preferred_stem: str) -> str:
        canonical_key = _canonical_image_key(source)
        cached = self._source_to_relpath.get(canonical_key)
        if cached:
            self.saved_paths.append(cached)
            return cached

        payload = _materialize_image_bytes(source)
        suffix = _detect_image_suffix(payload["bytes"], payload.get("mime_type"), payload.get("hint"))
        digest = hashlib.sha1(payload["bytes"]).hexdigest()[:12]
        self._counter += 1
        filename = f"{self.sample_key}_{preferred_stem}_{self._counter:03d}_{digest}{suffix}"
        abs_path = self.image_dir / filename
        abs_path.write_bytes(payload["bytes"])
        rel_path = str(Path("images") / filename)
        self._source_to_relpath[canonical_key] = rel_path
        self.saved_paths.append(rel_path)
        return rel_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, help="Raw trajectory jsonl from debug_vqa_batch.")
    parser.add_argument("--output-dir", required=True, help="Output directory containing parquet + images + .metadata.")
    parser.add_argument(
        "--include-incorrect",
        action="store_true",
        help="Keep samples even if answer_judge.is_correct is false.",
    )
    parser.add_argument(
        "--goal-in-args-tools",
        default="read_url",
        help="Comma-separated tool names whose top-level goal should move into arguments.goal.",
    )
    return parser


def _message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, indent=2)


def _strip_known_reasoning_tags(text: str) -> str:
    text = _THINKING_RE.sub(lambda m: m.group(1).strip(), text or "")
    text = _THINK_RE.sub(lambda m: m.group(1).strip(), text)
    return text.strip()


def _strip_answer_like_tags(text: str) -> str:
    text = _ANSWER_RE.sub(lambda m: m.group(1).strip(), text or "")
    text = _RESPONSE_RE.sub(lambda m: m.group(1).strip(), text)
    return text.strip()


def _wrap_thinking(text: str) -> str:
    cleaned = _strip_known_reasoning_tags(text).strip()
    if not cleaned:
        return ""
    return f"<thinking>{cleaned}</thinking>"


def _extract_action_payload(text: str) -> tuple[str, dict[str, Any] | None]:
    match = _ACTION_RE.search(text or "")
    if not match:
        return "", None
    thought = ((text or "")[: match.start()] + "\n" + (text or "")[match.end() :]).strip()
    raw_json = match.group("json")
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError:
        payload = None
    return thought.strip(), payload


def _extract_final_answer(text: str, fallback: str) -> tuple[str, str]:
    cleaned = (text or "").strip()
    for regex in (_ANSWER_RE, _RESPONSE_RE):
        match = regex.search(cleaned)
        if match:
            answer = match.group(1).strip()
            thought = (cleaned[: match.start()] + "\n" + cleaned[match.end() :]).strip()
            return thought, answer

    extracted_answer = (fallback or "").strip()
    if extracted_answer:
        last_pos = cleaned.rfind(extracted_answer)
        if last_pos >= 0:
            thought = (cleaned[:last_pos] + "\n" + cleaned[last_pos + len(extracted_answer) :]).strip()
            return thought, extracted_answer
    return "", cleaned


def _normalize_tool_definitions(goal_in_args_tools: set[str]) -> str:
    definitions = copy.deepcopy(sft_tools.get_tool_definitions())
    for item in definitions:
        function = item.get("function") or {}
        name = str(function.get("name") or "")
        if name not in goal_in_args_tools:
            continue
        parameters = function.setdefault("parameters", {"type": "object", "properties": {}, "required": []})
        properties = parameters.setdefault("properties", {})
        if "goal" not in properties:
            properties["goal"] = {
                "type": "string",
                "description": "Why this tool call is the right next step for the current reasoning goal.",
            }
    return json.dumps(definitions, ensure_ascii=False, indent=2)


def _canonical_image_key(source: Any) -> str:
    if isinstance(source, dict):
        if source.get("bytes") is not None:
            raw = source["bytes"]
            if isinstance(raw, bytes):
                return "bytes:" + hashlib.sha1(raw).hexdigest()
        if source.get("path"):
            return "path:" + str(source["path"])
        if source.get("url"):
            return "url:" + str(source["url"])
        if source.get("data"):
            return "data:" + hashlib.sha1(str(source["data"]).encode("utf-8")).hexdigest()
    if isinstance(source, bytes):
        return "bytes:" + hashlib.sha1(source).hexdigest()
    return f"repr:{repr(source)}"


def _materialize_image_bytes(source: Any) -> dict[str, Any]:
    if isinstance(source, dict):
        if isinstance(source.get("bytes"), bytes):
            return {"bytes": source["bytes"], "mime_type": None, "hint": source.get("path") or source.get("url")}
        if source.get("path"):
            return _materialize_image_bytes(str(source["path"]))
        if source.get("url"):
            return _materialize_image_bytes(str(source["url"]))
        if source.get("data"):
            return _materialize_image_bytes(str(source["data"]))
    if isinstance(source, bytes):
        return {"bytes": source, "mime_type": None, "hint": None}
    if isinstance(source, str):
        if source.startswith("data:image/"):
            header, payload = source.split(",", 1)
            mime_type = header.split(";")[0].split(":", 1)[-1].strip()
            return {
                "bytes": base64.b64decode(payload),
                "mime_type": mime_type,
                "hint": mime_type,
            }
        if source.startswith(("http://", "https://")):
            response = requests.get(source, timeout=60)
            response.raise_for_status()
            return {
                "bytes": response.content,
                "mime_type": response.headers.get("Content-Type", "").split(";")[0].strip(),
                "hint": source,
            }
        path = Path(source)
        if path.exists():
            return {"bytes": path.read_bytes(), "mime_type": None, "hint": str(path)}
    raise ValueError(f"Unsupported image source: {type(source)!r} / {source!r}")


def _detect_image_suffix(data: bytes, mime_type: str | None, hint: str | None) -> str:
    if mime_type and mime_type in _IMAGE_EXT_BY_MIME:
        return _IMAGE_EXT_BY_MIME[mime_type]
    if hint:
        parsed = urlparse(hint)
        suffix = Path(parsed.path).suffix.lower()
        if suffix:
            return suffix
        guessed = mimetypes.guess_extension(mime_type or "")
        if guessed:
            return guessed
    try:
        image = Image.open(BytesIO(data))
        fmt = (image.format or "").lower()
    except Exception:
        fmt = ""
    return {
        "png": ".png",
        "jpeg": ".jpg",
        "jpg": ".jpg",
        "webp": ".webp",
        "gif": ".gif",
        "bmp": ".bmp",
        "tiff": ".tiff",
    }.get(fmt, ".png")


def _iter_content_parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        return [part for part in content if isinstance(part, dict)]
    return []


def _render_multimodal_message_content(
    content: Any,
    *,
    image_store: SampleImageStore,
) -> str:
    if isinstance(content, str):
        return content.strip()

    lines: list[str] = []
    for part in _iter_content_parts(content):
        part_type = str(part.get("type") or "")
        if part_type in {"text", "input_text"}:
            text = str(part.get("text") or "").strip()
            if text:
                lines.append(text)
            continue

        image_source = None
        if part_type == "image_path":
            image_source = part.get("path")
        elif part_type == "image_url":
            image_url = part.get("image_url")
            if isinstance(image_url, dict):
                image_source = image_url.get("url")
            elif isinstance(image_url, str):
                image_source = image_url
        elif part_type in {"input_image", "image"}:
            image_source = part.get("image") or part.get("data") or part.get("url") or part.get("path")

        if image_source:
            image_store.save(image_source, preferred_stem="msgimg")
            lines.append("<image>")

    return "\n".join(line for line in lines if line.strip()).strip()


def _format_generic_tool_response(name: str, payload_text: str) -> str:
    body = payload_text.strip()
    if name:
        body = f"[{name}]\n{body}" if body else f"[{name}]"
    return f"<tool_response>\n{body}\n</tool_response>"


def _format_read_url_tool_response(
    payload: dict[str, Any],
    *,
    image_store: SampleImageStore,
) -> str | None:
    if not payload.get("ok"):
        return None
    if str(payload.get("kind") or "") != "image":
        return None

    source = payload.get("local_path") or payload.get("url") or payload.get("image_url")
    if not source:
        return None

    image_store.save(source, preferred_stem="read_url")
    lines = ["读取图片如下:", "<image>"]
    resource_id = str(payload.get("resource_id") or "").strip()
    if resource_id:
        lines.append(f"资源ID: {resource_id}")
    title = str(payload.get("title") or "").strip()
    if title:
        lines.append(f"标题: {title}")
    return "<tool_response>\n" + "\n".join(lines) + "\n</tool_response>"


def _format_tool_message(
    message: dict[str, Any],
    *,
    image_store: SampleImageStore,
) -> dict[str, str]:
    name = str(message.get("name") or "").strip()
    raw_content = _message_content_to_text(message.get("content")).strip()
    parsed: dict[str, Any] | None = None
    try:
        maybe = json.loads(raw_content) if raw_content else None
        if isinstance(maybe, dict):
            parsed = maybe
    except json.JSONDecodeError:
        parsed = None

    if name == "read_url" and parsed is not None:
        rendered = _format_read_url_tool_response(parsed, image_store=image_store)
        if rendered:
            return {"role": "user", "content": rendered}

    payload_text = json.dumps(parsed, ensure_ascii=False, indent=2) if parsed is not None else raw_content
    return {"role": "user", "content": _format_generic_tool_response(name, payload_text)}


def _normalize_action_payload(
    payload: dict[str, Any],
    *,
    goal_in_args_tools: set[str],
) -> dict[str, Any]:
    tool_name = str(payload.get("tool_name") or payload.get("name") or "").strip()
    params = payload.get("params")
    if not isinstance(params, dict):
        params = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}
    arguments = copy.deepcopy(params)
    goal = str(payload.get("goal") or "").strip()
    if goal and tool_name in goal_in_args_tools:
        arguments["goal"] = goal
    return {
        "name": tool_name,
        "arguments": arguments,
    }


# #### START Response 0720 ####
def _normalize_native_tool_call(
    tool_call: dict[str, Any],
    *,
    goal_in_args_tools: set[str],
) -> dict[str, Any] | None:
    function = tool_call.get("function") or {}
    if not isinstance(function, dict):
        return None
    tool_name = str(function.get("name") or "").strip()
    raw_arguments = function.get("arguments")
    if isinstance(raw_arguments, dict):
        arguments = copy.deepcopy(raw_arguments)
    elif isinstance(raw_arguments, str) and raw_arguments.strip():
        try:
            parsed_arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            parsed_arguments = {}
        arguments = parsed_arguments if isinstance(parsed_arguments, dict) else {}
    else:
        arguments = {}
    goal = str(tool_call.get("goal") or "").strip()
    if goal and tool_name in goal_in_args_tools:
        arguments["goal"] = goal
    if not tool_name:
        return None
    return {"name": tool_name, "arguments": arguments}
# #### END Response 0720 ####


def _convert_assistant_message(
    message: dict[str, Any],
    *,
    extracted_answer: str,
    is_last_assistant: bool,
    goal_in_args_tools: set[str],
) -> dict[str, str]:
    raw_text = _message_content_to_text(message.get("content")).strip()
    thought_text, action_payload = _extract_action_payload(raw_text)
    if action_payload is not None:
        blocks: list[str] = []
        thinking_block = _wrap_thinking(thought_text)
        if thinking_block:
            blocks.append(thinking_block)
        normalized_call = _normalize_action_payload(action_payload, goal_in_args_tools=goal_in_args_tools)
        blocks.append(f"<tool_call>{json.dumps(normalized_call, ensure_ascii=False)}</tool_call>")
        return {"role": "assistant", "content": "\n".join(blocks).strip()}

    # #### START Response 0720 ####
    native_tool_calls = [
        normalized
        for tool_call in (message.get("tool_calls") or [])
        if isinstance(tool_call, dict)
        if (normalized := _normalize_native_tool_call(tool_call, goal_in_args_tools=goal_in_args_tools)) is not None
    ]
    if native_tool_calls:
        blocks: list[str] = []
        thinking_block = _wrap_thinking(raw_text)
        if thinking_block:
            blocks.append(thinking_block)
        for normalized_call in native_tool_calls:
            blocks.append(f"<tool_call>{json.dumps(normalized_call, ensure_ascii=False)}</tool_call>")
        return {"role": "assistant", "content": "\n".join(blocks).strip()}
    # #### END Response 0720 ####

    if is_last_assistant:
        thought, answer = _extract_final_answer(raw_text, extracted_answer)
        blocks = []
        thinking_block = _wrap_thinking(thought)
        if thinking_block:
            blocks.append(thinking_block)
        blocks.append(f"<answer>{answer.strip()}</answer>")
        return {"role": "assistant", "content": "\n".join(blocks).strip()}

    thinking_block = _wrap_thinking(_strip_answer_like_tags(raw_text))
    return {"role": "assistant", "content": thinking_block or raw_text}


def _build_sample_key(record: dict[str, Any], row_index: int) -> str:
    raw = "|".join(
        [
            str(record.get("question_id") or ""),
            str(record.get("sample_id") or ""),
            str(record.get("path_id") or ""),
            str(row_index),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _ensure_user_images_present(
    messages: list[dict[str, str]],
    record: dict[str, Any],
    *,
    image_store: SampleImageStore,
) -> None:
    has_any_placeholder = any("<image>" in str(message.get("content") or "") for message in messages)
    if has_any_placeholder:
        return

    fallback_images = list(record.get("input_images") or [])
    if not fallback_images:
        return

    first_user_index = next((index for index, item in enumerate(messages) if item.get("role") == "user"), None)
    if first_user_index is None:
        return

    extra_placeholders: list[str] = []
    for image_item in fallback_images:
        if not isinstance(image_item, dict):
            continue
        source = image_item.get("image_path") or image_item.get("image_url")
        if not source:
            continue
        image_store.save(source, preferred_stem="fallback")
        extra_placeholders.append("<image>")
    if not extra_placeholders:
        return

    original = str(messages[first_user_index].get("content") or "").strip()
    joined = "\n".join([original, *extra_placeholders]).strip()
    messages[first_user_index]["content"] = joined


def _convert_one_record(
    record: dict[str, Any],
    *,
    row_index: int,
    image_dir: Path,
    goal_in_args_tools: set[str],
    tool_definitions_json: str,
) -> dict[str, Any]:
    sample_key = _build_sample_key(record, row_index)
    image_store = SampleImageStore(image_dir=image_dir, sample_key=sample_key)
    raw_messages = list(record.get("raw_messages") or [])
    assistant_indexes = [idx for idx, item in enumerate(raw_messages) if isinstance(item, dict) and item.get("role") == "assistant"]
    last_assistant_index = assistant_indexes[-1] if assistant_indexes else -1

    converted_messages: list[dict[str, str]] = []
    for idx, message in enumerate(raw_messages):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip()
        if role == "system":
            converted_messages.append({"role": "system", "content": _message_content_to_text(message.get("content")).strip()})
            continue
        if role == "user":
            converted_messages.append(
                {
                    "role": "user",
                    "content": _render_multimodal_message_content(message.get("content"), image_store=image_store),
                }
            )
            continue
        if role == "assistant":
            converted_messages.append(
                _convert_assistant_message(
                    message,
                    extracted_answer=str(record.get("extracted_answer") or ""),
                    is_last_assistant=(idx == last_assistant_index),
                    goal_in_args_tools=goal_in_args_tools,
                )
            )
            continue
        if role == "tool":
            converted_messages.append(_format_tool_message(message, image_store=image_store))

    _ensure_user_images_present(converted_messages, record, image_store=image_store)

    deduped_images: list[str] = []
    seen_paths: set[str] = set()
    for rel_path in image_store.saved_paths:
        if rel_path not in seen_paths:
            seen_paths.add(rel_path)
            deduped_images.append(rel_path)

    return {
        "id": sample_key,
        "question_id": record.get("question_id"),
        "sample_id": record.get("sample_id"),
        "path_id": record.get("path_id"),
        "question": record.get("question"),
        "answer": record.get("gold_answer"),
        "messages": converted_messages,
        "images": deduped_images,
        "tools": tool_definitions_json,
        "source_metadata": record.get("source_metadata") or {},
    }


def _write_parquet(rows: list[dict[str, Any]], parquet_path: Path) -> None:
    if not rows:
        raise ValueError("No rows were produced. Nothing to write to parquet.")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Writing parquet requires `pyarrow`. Please install it in the target environment."
        ) from exc
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, parquet_path)


def _write_metadata(
    *,
    metadata_dir: Path,
    input_jsonl: str,
    parquet_path: Path,
    row_count: int,
    skipped_count: int,
    errors: list[ConversionError],
    tool_definitions_json: str,
) -> None:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "tool_definitions.json").write_text(tool_definitions_json + "\n", encoding="utf-8")
    summary = {
        "input_jsonl": input_jsonl,
        "parquet_path": str(parquet_path),
        "row_count": row_count,
        "skipped_count": skipped_count,
        "error_count": len(errors),
        "columns": ["id", "question_id", "sample_id", "path_id", "question", "answer", "messages", "images", "tools", "source_metadata"],
    }
    (metadata_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if errors:
        with (metadata_dir / "errors.jsonl").open("w", encoding="utf-8") as handle:
            for err in errors:
                handle.write(json.dumps({"sample_id": err.sample_id, "message": err.message}, ensure_ascii=False) + "\n")


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected object on line {line_number}, got {type(parsed)!r}")
            records.append(parsed)
    return records


def _parse_goal_in_args_tools(raw_value: str) -> set[str]:
    names = {item.strip() for item in str(raw_value or "").split(",") if item.strip()}
    return names or set(_DEFAULT_GOAL_IN_ARGS_TOOLS)


def main() -> int:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    image_dir = output_dir / "images"
    metadata_dir = output_dir / ".metadata"
    parquet_path = output_dir / "data.parquet"

    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Output directory is not empty: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    goal_in_args_tools = _parse_goal_in_args_tools(args.goal_in_args_tools)
    tool_definitions_json = _normalize_tool_definitions(goal_in_args_tools)
    input_records = _load_jsonl(args.input_jsonl)

    rows: list[dict[str, Any]] = []
    errors: list[ConversionError] = []
    skipped_count = 0
    for row_index, record in enumerate(input_records):
        if not args.include_incorrect and not bool(((record.get("answer_judge") or {}).get("is_correct"))):
            skipped_count += 1
            continue
        try:
            rows.append(
                _convert_one_record(
                    record,
                    row_index=row_index,
                    image_dir=image_dir,
                    goal_in_args_tools=goal_in_args_tools,
                    tool_definitions_json=tool_definitions_json,
                )
            )
        except Exception as exc:  # noqa: BLE001
            sample_id = _build_sample_key(record, row_index)
            errors.append(ConversionError(sample_id=sample_id, message=str(exc)))
            skipped_count += 1

    _write_parquet(rows, parquet_path)
    _write_metadata(
        metadata_dir=metadata_dir,
        input_jsonl=args.input_jsonl,
        parquet_path=parquet_path,
        row_count=len(rows),
        skipped_count=skipped_count,
        errors=errors,
        tool_definitions_json=tool_definitions_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
