#!/usr/bin/env python3
"""Convert ``debug_vqa_batch.py`` trajectories to LlamaFactory ShareGPT data.

The input is the raw JSONL written by ``synthesis/sft/debug_vqa_batch.py``.
The output directory is self-contained and has this layout::

    output_dir/
      trajectories_sharegpt.json
      dataset_info.json
      images/...
      .metadata/summary.json
      .metadata/rejected.jsonl

The converter deliberately rebuilds the first user message from the top-level
``question`` field.  The original first prompt contains the gold answer and
private hop-chain facts, which must not become training input.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlparse

import requests
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from opensearch_vl.opensearch_infer.prompts import build_system_prompt
from synthesis.sft.tools import get_tool_definitions_json


_ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.IGNORECASE | re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.IGNORECASE | re.DOTALL)
_THINKING_RE = re.compile(r"<thinking>\s*(.*?)\s*</thinking>", re.IGNORECASE | re.DOTALL)
_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_RESPONSE_RE = re.compile(r"<response>\s*(.*?)\s*</response>", re.IGNORECASE | re.DOTALL)
_TOOL_RESPONSE_TAG_RE = re.compile(r"</?tool_response>\s*", re.IGNORECASE)
_IMAGE_EXT_BY_MIME = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
}


class ConversionError(ValueError):
    """An individual input record cannot be represented as a valid trajectory."""


@dataclass
class ImageStore:
    """Materialize images while preserving one top-level entry per ``<image>``."""

    output_dir: Path
    sample_key: str
    source_base_dir: Path
    paths: list[str] = field(default_factory=list)
    _cache: dict[str, str] = field(default_factory=dict)
    _counter: int = 0

    def save(self, source: Any, *, stem: str) -> str:
        payload = _materialize_image(source, base_dir=self.source_base_dir)
        key = _image_source_key(payload["bytes"], source)
        relative = self._cache.get(key)
        if relative is None:
            self._counter += 1
            suffix = _image_suffix(payload["bytes"], payload.get("mime_type"), payload.get("hint"))
            digest = hashlib.sha1(payload["bytes"]).hexdigest()[:12]
            filename = f"{self.sample_key}_{stem}_{self._counter:03d}_{digest}{suffix}"
            absolute = self.output_dir / "images" / filename
            absolute.write_bytes(payload["bytes"])
            relative = str(Path("images") / filename)
            self._cache[key] = relative

        # Do not deduplicate this list: repeated visual observations require
        # repeated ``<image>`` placeholders in exactly the same order.
        self.paths.append(relative)
        return relative


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, help="Raw JSONL from debug_vqa_batch.py.")
    parser.add_argument("--output-dir", required=True, help="New directory for the ShareGPT dataset.")
    parser.add_argument(
        "--include-incorrect",
        action="store_true",
        help="Keep records whose answer_judge.is_correct is false (default: skip them).",
    )
    return parser


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [str(part.get("text") or "").strip() for part in content if isinstance(part, dict)]
        text_parts = [part for part in parts if part]
        if text_parts:
            return "\n".join(text_parts).strip()
    return _json_text(content).strip() if content is not None else ""


def _strip_reasoning_tags(text: str) -> str:
    cleaned = _THINKING_RE.sub(lambda match: match.group(1).strip(), text or "")
    return _THINK_RE.sub(lambda match: match.group(1).strip(), cleaned).strip()


def _thinking_block(text: str) -> str:
    cleaned = _strip_reasoning_tags(text)
    return f"<thinking>\n{cleaned}\n</thinking>" if cleaned else ""


def _parse_json_object(text: str) -> dict[str, Any] | None:
    candidate = (text or "").strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, re.IGNORECASE | re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _normalise_tool_call(payload: dict[str, Any]) -> dict[str, Any]:
    name = str(payload.get("tool_name") or payload.get("name") or "").strip()
    if not name:
        raise ConversionError("tool call has no tool name")
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        arguments = payload.get("params")
    if not isinstance(arguments, dict):
        arguments = {}
    return {"name": name, "arguments": arguments}


def _native_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else tool_call
    name = str(function.get("name") or "").strip()
    if not name:
        raise ConversionError("native tool call has no function name")
    raw_arguments = function.get("arguments", {})
    if isinstance(raw_arguments, dict):
        arguments = raw_arguments
    elif isinstance(raw_arguments, str):
        arguments = _parse_json_object(raw_arguments)
        if arguments is None:
            raise ConversionError(f"invalid JSON arguments for tool {name!r}")
    else:
        arguments = {}
    return {"name": name, "arguments": arguments}


def _extract_final_answer(text: str, fallback: str) -> tuple[str, str]:
    cleaned = (text or "").strip()
    for pattern in (_ANSWER_RE, _RESPONSE_RE):
        match = pattern.search(cleaned)
        if match:
            thought = (cleaned[: match.start()] + "\n" + cleaned[match.end() :]).strip()
            return thought, match.group(1).strip()
    answer = (fallback or "").strip()
    if answer and answer in cleaned:
        position = cleaned.rfind(answer)
        return (cleaned[:position] + "\n" + cleaned[position + len(answer) :]).strip(), answer
    return "", cleaned


def _assistant_conversion(
    message: dict[str, Any],
    *,
    is_last_assistant: bool,
    extracted_answer: str,
) -> tuple[str, bool]:
    """Return assistant content and whether it is a final answer."""

    raw_text = _content_text(message.get("content"))
    action_match = _ACTION_RE.search(raw_text)
    if action_match:
        payload = _parse_json_object(action_match.group(1))
        if payload is None:
            raise ConversionError("could not parse JSON inside <action>")
        thought = (raw_text[: action_match.start()] + "\n" + raw_text[action_match.end() :]).strip()
        action_name = str(payload.get("tool_name") or payload.get("name") or "").strip()
        arguments = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else payload.get("params")
        if not isinstance(arguments, dict):
            arguments = {}
        if action_name == "finish":
            answer = str(arguments.get("answer") or extracted_answer or "").strip()
            if not answer:
                raise ConversionError("finish action has no answer")
            blocks = [block for block in (_thinking_block(thought), f"<answer>\n{answer}\n</answer>") if block]
            return "\n".join(blocks), True
        call = _normalise_tool_call(payload)
        blocks = [block for block in (_thinking_block(thought), _format_tool_call(call)) if block]
        return "\n".join(blocks), False

    native_calls = message.get("tool_calls") or []
    if native_calls:
        if not isinstance(native_calls, list):
            raise ConversionError("tool_calls is not a list")
        calls = [_native_tool_call(item) for item in native_calls if isinstance(item, dict)]
        if not calls:
            raise ConversionError("assistant tool_calls list is empty or malformed")
        blocks = [_thinking_block(raw_text)] + [_format_tool_call(call) for call in calls]
        return "\n".join(block for block in blocks if block), False

    preformatted = list(_TOOL_CALL_RE.finditer(raw_text))
    if preformatted:
        calls = []
        for match in preformatted:
            payload = _parse_json_object(match.group(1))
            if payload is None:
                raise ConversionError("could not parse JSON inside <tool_call>")
            calls.append(_normalise_tool_call(payload))
        thought = raw_text
        for match in reversed(preformatted):
            thought = thought[: match.start()] + "\n" + thought[match.end() :]
        blocks = [_thinking_block(thought)] + [_format_tool_call(call) for call in calls]
        return "\n".join(block for block in blocks if block), False

    if is_last_assistant:
        thought, answer = _extract_final_answer(raw_text, extracted_answer)
        if not answer:
            raise ConversionError("final assistant message has no answer")
        blocks = [block for block in (_thinking_block(thought), f"<answer>\n{answer}\n</answer>") if block]
        return "\n".join(blocks), True

    # A non-final assistant without a tool call cannot be aligned with an
    # observation turn and is almost certainly a malformed raw trajectory.
    raise ConversionError("non-final assistant message has neither a tool call nor a finish action")


def _format_tool_call(call: dict[str, Any]) -> str:
    return f"<tool_call>\n{json.dumps(call, ensure_ascii=False)}\n</tool_call>"


def _image_parts(content: Any) -> list[Any]:
    if not isinstance(content, list):
        return []
    sources: list[Any] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = str(part.get("type") or "").lower()
        if part_type == "image_url":
            value = part.get("image_url")
            sources.append(value.get("url") if isinstance(value, dict) else value)
        elif part_type in {"image_path", "input_image", "image"}:
            value = part.get("image_url")
            sources.append(
                part.get("path")
                or part.get("image")
                or part.get("data")
                or part.get("url")
                or (value.get("url") if isinstance(value, dict) else value)
            )
    return [source for source in sources if source]


def _initial_image_sources(record: dict[str, Any], first_user: dict[str, Any] | None) -> list[Any]:
    sources: list[Any] = []
    input_images = record.get("input_images") or []
    if not input_images:
        input_images = [
            *({"image_path": value} for value in (record.get("image_paths") or [])),
            *({"image_url": value} for value in (record.get("image_urls") or [])),
        ]
    for item in input_images:
        if isinstance(item, dict):
            source = item.get("image_path") or item.get("image_url") or item.get("path") or item.get("url")
        else:
            source = item
        if source:
            sources.append(source)
    if not sources:
        sources.extend(_image_parts(first_user.get("content")) if first_user else [])
    return sources


def _observation_text(content: Any) -> str:
    text = _json_text(content).strip()
    if isinstance(content, str):
        parsed = _parse_json_object(content)
        if parsed is not None:
            text = json.dumps(parsed, ensure_ascii=False, indent=2)
    # Older exporters sometimes included the wrapper. qwen3_vl adds it itself.
    return _TOOL_RESPONSE_TAG_RE.sub("", text).strip()


def _append_attachment_to_observation(
    observation: dict[str, str],
    content: Any,
    *,
    image_store: ImageStore,
    stem: str,
) -> None:
    sources = _image_parts(content)
    if not sources:
        raise ConversionError("user attachment after a tool observation contains no image")
    for source in sources:
        image_store.save(source, stem=stem)
    observation["value"] = observation["value"].rstrip() + "\n读取图片如下：\n" + "\n".join(
        "<image>" for _ in sources
    )


def _sample_key(record: dict[str, Any], row_index: int) -> str:
    raw = "|".join(str(record.get(key) or "") for key in ("question_id", "sample_id", "path_id"))
    raw = raw or f"row-{row_index}"
    digest = hashlib.sha1(f"{raw}|{row_index}".encode("utf-8")).hexdigest()[:10]
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._-")[:64] or "sample"
    return f"{readable}_{digest}"


def _convert_record(
    record: dict[str, Any],
    *,
    row_index: int,
    output_dir: Path,
    source_base_dir: Path,
) -> dict[str, Any]:
    question = str(record.get("question") or "").strip()
    if not question:
        raise ConversionError("record has no top-level question")
    raw_messages = [item for item in (record.get("raw_messages") or []) if isinstance(item, dict)]
    first_user_index = next((idx for idx, item in enumerate(raw_messages) if item.get("role") == "user"), None)
    first_user = raw_messages[first_user_index] if first_user_index is not None else None
    image_store = ImageStore(
        output_dir=output_dir,
        sample_key=_sample_key(record, row_index),
        source_base_dir=source_base_dir,
    )
    initial_sources = _initial_image_sources(record, first_user)
    human_value = question
    if initial_sources:
        for source in initial_sources:
            image_store.save(source, stem="input")
        human_value += "\n" + "\n".join("<image>" for _ in initial_sources)

    conversations: list[dict[str, str]] = [{"from": "human", "value": human_value}]
    assistant_indexes = [idx for idx, item in enumerate(raw_messages) if item.get("role") == "assistant"]
    last_assistant_index = assistant_indexes[-1] if assistant_indexes else None
    extracted_answer = str(record.get("extracted_answer") or "").strip()
    final_seen = False

    for idx, message in enumerate(raw_messages):
        role = str(message.get("role") or "").strip().lower()
        if role == "system" or idx == first_user_index:
            continue
        if role == "assistant":
            if final_seen:
                continue
            content, is_final = _assistant_conversion(
                message,
                is_last_assistant=(idx == last_assistant_index),
                extracted_answer=extracted_answer,
            )
            conversations.append({"from": "gpt", "value": content})
            final_seen = final_seen or is_final
            continue
        if role == "tool":
            if not conversations or conversations[-1]["from"] != "gpt":
                raise ConversionError("tool observation is not preceded by an assistant turn")
            conversations.append({"from": "observation", "value": _observation_text(message.get("content"))})
            continue
        if role == "user":
            if conversations[-1]["from"] != "observation":
                raise ConversionError("unexpected user message after the initial question")
            _append_attachment_to_observation(
                conversations[-1],
                message.get("content"),
                image_store=image_store,
                stem="read_url",
            )
            continue

    if not final_seen:
        if conversations[-1]["from"] != "observation":
            raise ConversionError("trajectory does not end with an observation followed by a final answer")
        if not extracted_answer:
            raise ConversionError("trajectory has no extracted_answer for final answer fallback")
        conversations.append({"from": "gpt", "value": f"<answer>\n{extracted_answer}\n</answer>"})

    if any(message.get("from") == "gpt" and "<tool_call>" in message.get("value", "") for message in conversations):
        if not any(message.get("from") == "observation" for message in conversations):
            raise ConversionError("tool call has no observation")

    _validate_conversations(conversations)
    return {
        "id": image_store.sample_key,
        "question_id": record.get("question_id"),
        "sample_id": record.get("sample_id"),
        "path_id": record.get("path_id"),
        "system": build_system_prompt(get_tool_definitions_json()),
        "conversations": conversations,
        "images": image_store.paths,
    }


def _validate_conversations(conversations: list[dict[str, str]]) -> None:
    if not conversations or conversations[0].get("from") != "human":
        raise ConversionError("conversation must start with human")
    if len(conversations) % 2 != 0 or conversations[-1].get("from") != "gpt":
        raise ConversionError("conversation must have an even number of turns and end with gpt")
    for idx, message in enumerate(conversations):
        expected = {"human", "observation"} if idx % 2 == 0 else {"gpt"}
        if message.get("from") not in expected:
            raise ConversionError(f"invalid role at conversation index {idx}: {message.get('from')!r}")
        if message.get("from") == "observation" and "<tool_response>" in message.get("value", ""):
            raise ConversionError("observation contains nested <tool_response>")


def _materialize_image(source: Any, *, base_dir: Path) -> dict[str, Any]:
    if isinstance(source, dict):
        if isinstance(source.get("bytes"), (bytes, bytearray)):
            return {"bytes": bytes(source["bytes"]), "mime_type": None, "hint": None}
        for key in ("path", "url", "data", "image"):
            if source.get(key):
                return _materialize_image(source[key], base_dir=base_dir)
    if isinstance(source, (bytes, bytearray)):
        return {"bytes": bytes(source), "mime_type": None, "hint": None}
    if not isinstance(source, str):
        raise ConversionError(f"unsupported image source type: {type(source).__name__}")
    value = source.strip()
    if value.startswith("data:image/"):
        header, payload = value.split(",", 1)
        mime_type = header.split(";", 1)[0].split(":", 1)[1].lower()
        try:
            data = base64.b64decode(payload, validate=False)
        except Exception as exc:  # pragma: no cover - defensive
            raise ConversionError("invalid image data URL") from exc
        return {"bytes": data, "mime_type": mime_type, "hint": mime_type}
    if value.startswith("file://"):
        path = Path(unquote(urlparse(value).path))
        return {"bytes": path.read_bytes(), "mime_type": None, "hint": str(path)}
    if value.startswith(("http://", "https://")):
        try:
            response = requests.get(value, timeout=60)
            response.raise_for_status()
        except Exception as exc:  # pragma: no cover - network dependent
            raise ConversionError(f"could not download image {value!r}: {exc}") from exc
        mime_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() or None
        return {"bytes": response.content, "mime_type": mime_type, "hint": value}
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    if not path.is_file():
        raise ConversionError(f"image path does not exist: {path}")
    return {"bytes": path.read_bytes(), "mime_type": None, "hint": str(path)}


def _image_source_key(data: bytes, source: Any) -> str:
    # Content hashing makes a data URL and a local copy of the same image share
    # one file while retaining duplicate references in ``paths``.
    return hashlib.sha1(data).hexdigest()


def _image_suffix(data: bytes, mime_type: str | None, hint: str | None) -> str:
    if mime_type in _IMAGE_EXT_BY_MIME:
        suffix = _IMAGE_EXT_BY_MIME[mime_type]
    else:
        suffix = Path(urlparse(hint or "").path).suffix.lower() if hint else ""
    try:
        with Image.open(BytesIO(data)) as image:
            image.verify()
            fmt = (image.format or "").lower()
    except Exception as exc:
        raise ConversionError(f"materialized bytes are not a valid image: {exc}") from exc
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"}:
        return ".jpg" if suffix == ".jpeg" else suffix
    return {
        "png": ".png",
        "jpeg": ".jpg",
        "jpg": ".jpg",
        "webp": ".webp",
        "gif": ".gif",
        "bmp": ".bmp",
        "tiff": ".tiff",
    }.get(fmt, ".png")


def _load_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ConversionError(f"invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ConversionError(f"line {line_number} is not a JSON object")
            yield line_number, value


def _dataset_info() -> dict[str, Any]:
    return {
        "opensearch_vl_sft": {
            "file_name": "trajectories_sharegpt.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "images": "images", "system": "system"},
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
                "observation_tag": "observation",
                "function_tag": "function_call",
                "system_tag": "system",
            },
        }
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def convert_file(input_jsonl: str | Path, output_dir: str | Path, *, include_incorrect: bool = False) -> dict[str, int]:
    """Convert a raw JSONL file and return counts for the CLI/tests."""

    input_path = Path(input_jsonl).expanduser().resolve()
    final_dir = Path(output_dir).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if final_dir.exists():
        raise FileExistsError(f"output directory already exists; choose a new path: {final_dir}")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(tempfile.mkdtemp(prefix=f".{final_dir.name}.tmp-", dir=str(final_dir.parent)))
    try:
        (stage_dir / "images").mkdir()
        metadata_dir = stage_dir / ".metadata"
        metadata_dir.mkdir()
        rows: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        skipped_incorrect = 0
        for row_index, (line_number, record) in enumerate(_load_jsonl(input_path)):
            if not include_incorrect and not bool((record.get("answer_judge") or {}).get("is_correct")):
                skipped_incorrect += 1
                rejected.append({"line": line_number, "sample_id": record.get("sample_id"), "reason": "answer_judge.is_correct is false"})
                continue
            try:
                rows.append(
                    _convert_record(
                        record,
                        row_index=row_index,
                        output_dir=stage_dir,
                        source_base_dir=input_path.parent,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - isolate bad records
                rejected.append({"line": line_number, "sample_id": record.get("sample_id"), "reason": str(exc)})

        _write_json(stage_dir / "trajectories_sharegpt.json", rows)
        _write_json(stage_dir / "dataset_info.json", _dataset_info())
        _write_json(
            metadata_dir / "summary.json",
            {
                "input_jsonl": str(input_path),
                "output_dir": str(final_dir),
                "written_records": len(rows),
                "rejected_records": len(rejected),
                "skipped_incorrect": skipped_incorrect,
                "image_count": sum(len(row.get("images") or []) for row in rows),
            },
        )
        with (metadata_dir / "rejected.jsonl").open("w", encoding="utf-8") as handle:
            for item in rejected:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        stage_dir.rename(final_dir)
        return {"written_records": len(rows), "rejected_records": len(rejected), "skipped_incorrect": skipped_incorrect}
    except Exception:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        counts = convert_file(args.input_jsonl, args.output_dir, include_incorrect=args.include_incorrect)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(str(exc)) from exc
    print(json.dumps(counts, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
