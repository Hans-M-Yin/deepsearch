"""Conversion helpers between Gemini-style content and runner-specific formats."""

from __future__ import annotations

import base64
import io
import logging
import os
from pathlib import Path
import urllib.parse
from typing import Any, Dict, Iterable, List, Optional

from PIL import Image

from . import config
from . import image_io
from synthesis.sft.qwen3_vl_template import interleave_sft_image_parts


logger = logging.getLogger(__name__)


def _normalize_image_url_reference(url: str) -> str:
    """Normalize local absolute paths to ``file://`` URLs for API backends."""

    if not url:
        return url
    if url.startswith(("http://", "https://", "data:", "file://")):
        return url

    candidate = Path(url).expanduser()
    if candidate.is_absolute():
        return candidate.resolve().as_uri()
    return url


def _local_image_path_from_reference(url: str) -> Optional[Path]:
    """Return a local filesystem path for a local image reference."""

    if not url:
        return None
    if url.startswith("file://"):
        parsed = urllib.parse.urlparse(url)
        if not parsed.path:
            return None
        return Path(parsed.path)

    candidate = Path(url).expanduser()
    if candidate.is_absolute():
        return candidate
    return None


def _maybe_inline_local_image_url(url: str) -> str:
    """Inline local image files as data URLs for OpenAI-compatible backends."""

    local_path = _local_image_path_from_reference(url)
    if local_path is None or not local_path.exists():
        return url

    try:
        data = local_path.read_bytes()
    except OSError as exc:
        logger.warning("Failed to read local image %s: %s", local_path, exc)
        return url

    payload = image_io.image_to_base64(data)
    if not payload:
        return url
    mime = image_io.detect_image_format(payload)
    return f"data:{mime};base64,{payload}"


def to_claude_messages(contents: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Gemini-style ``contents`` to Claude content blocks."""

    messages: List[Dict[str, Any]] = []
    for item in contents:
        role = item.get("role", "user")
        parts = item.get("parts", []) or []
        block: List[Dict[str, Any]] = []
        logical_parts: list[tuple[str, Any]] = []
        for part in parts:
            if "image_url" in part:
                value = part["image_url"]
                url = value.get("url", "") if isinstance(value, dict) else str(value)
                if url:
                    url = _normalize_image_url_reference(url)
                    url = _maybe_inline_local_image_url(url)
                    logical_parts.append(("image", {"type": "image_url", "value": url}))
            elif "inline_data" in part:
                data = part["inline_data"]
                payload = data.get("data", "")
                mime = data.get("mime_type", "") or image_io.detect_image_format(payload)
                logical_parts.append(
                    (
                        "image",
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime,
                                "data": payload,
                            },
                        },
                    )
                )
            elif "text" in part:
                logical_parts.append(("text", {"type": "text", "text": part["text"]}))
        for kind, value in interleave_sft_image_parts(
            [("image", value) if kind == "image" else ("text", value["text"]) for kind, value in logical_parts]
        ):
            block.append(value if kind == "image" else {"type": "text", "text": value})
        if block:
            claude_role = "assistant" if role == "model" else role
            messages.append({"role": claude_role, "content": block})
    return messages


def _resolve_image_url_to_pil(url: str) -> Image.Image | None:
    """Try the network first, then fall back to ``FVQA_IMAGE_DIR``."""

    if url.startswith("file://"):
        try:
            local_path = _local_image_path_from_reference(url)
            if local_path is None:
                return None
            return Image.open(local_path)
        except Exception as exc:
            logger.warning("Failed to read file URL image %s: %s", url, exc)

    if os.path.isabs(url):
        try:
            local_path = _local_image_path_from_reference(url)
            if local_path is None:
                return None
            return Image.open(local_path)
        except Exception as exc:
            logger.warning("Failed to read local image %s: %s", url, exc)

    fetch_url = image_io.cos_url_to_internal(url)
    data = image_io.download_image_bytes(fetch_url)
    if data:
        try:
            return Image.open(io.BytesIO(data))
        except Exception as exc:
            logger.warning("Failed to decode downloaded image: %s", exc)

    if config.FVQA_IMAGE_DIR and os.path.isdir(config.FVQA_IMAGE_DIR):
        parsed = urllib.parse.urlparse(url)
        candidate = os.path.basename(parsed.path)
        if candidate:
            local = os.path.join(config.FVQA_IMAGE_DIR, candidate)
            if os.path.exists(local):
                try:
                    return Image.open(local)
                except Exception as exc:
                    logger.warning("Failed to read local image %s: %s", local, exc)
    return None


def to_qwen3vl_messages(contents: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Gemini-style ``contents`` to Qwen3-VL chat messages."""

    messages: List[Dict[str, Any]] = []
    for item in contents:
        role = item.get("role", "user")
        parts = item.get("parts", []) or []
        block: List[Dict[str, Any]] = []
        logical_parts: list[tuple[str, Any]] = []
        for part in parts:
            if "inline_data" in part:
                data = part["inline_data"]
                payload = data.get("data", "")
                if "base64," in payload:
                    payload = payload.split("base64,", 1)[1]
                try:
                    pil_image = Image.open(io.BytesIO(base64.b64decode(payload)))
                    logical_parts.append(("image", {"type": "image", "image": pil_image}))
                except Exception as exc:
                    logger.warning("Failed to decode inline base64 image: %s", exc)
                    logical_parts.append(("image", {"type": "image", "image": payload}))
            elif "image_url" in part:
                value = part["image_url"]
                url = value.get("url", "") if isinstance(value, dict) else str(value)
                if not url:
                    continue
                url = _normalize_image_url_reference(url)
                pil_image = _resolve_image_url_to_pil(url)
                if pil_image is not None:
                    logical_parts.append(("image", {"type": "image", "image": pil_image}))
            elif "text" in part:
                logical_parts.append(("text", str(part["text"])))

        for kind, value in interleave_sft_image_parts(
            [
                ("image", value["image"]) if kind == "image" else ("text", value)
                for kind, value in logical_parts
            ]
        ):
            block.append({"type": "image", "image": value} if kind == "image" else {"type": "text", "text": value})
        if block:
            qwen_role = "assistant" if role == "model" else role
            messages.append({"role": qwen_role, "content": block})
    return messages


def to_openai_messages(
    contents: Iterable[Dict[str, Any]],
    system_instruction: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Convert Gemini-style ``contents`` to OpenAI chat-completions messages.

    This is intentionally lossless for text and image payloads so the same
    agent loop can target Claude gateways, local HF models, or vLLM/OpenAI-
    compatible servers without changing benchmark data fields.
    """

    openai_messages: List[Dict[str, Any]] = []
    if system_instruction:
        openai_messages.append({"role": "system", "content": system_instruction})

    for item in contents:
        role = item.get("role", "user")
        parts = item.get("parts", []) or []
        block: List[Dict[str, Any]] = []
        logical_parts: list[tuple[str, Any]] = []
        for part in parts:
            if "image_url" in part:
                value = part["image_url"]
                url = value.get("url", "") if isinstance(value, dict) else str(value)
                if url:
                    url = _normalize_image_url_reference(url)
                    url = _maybe_inline_local_image_url(url)
                    logical_parts.append(("image", {"url": url}))
            elif "inline_data" in part:
                data = part["inline_data"]
                payload = data.get("data", "")
                if payload:
                    mime = data.get("mime_type", "") or image_io.detect_image_format(
                        payload
                    )
                    if not payload.startswith("data:"):
                        payload = f"data:{mime};base64,{payload}"
                    logical_parts.append(("image", {"url": payload}))
            elif "text" in part:
                logical_parts.append(("text", str(part["text"])))

        for kind, value in interleave_sft_image_parts(logical_parts):
            if kind == "image":
                block.append({"type": "image_url", "image_url": value})
            else:
                block.append({"type": "text", "text": value})
        if block:
            openai_role = "assistant" if role == "model" else role
            openai_messages.append({"role": openai_role, "content": block})
    return openai_messages
