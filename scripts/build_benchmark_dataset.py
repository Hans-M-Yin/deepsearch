#!/usr/bin/env python3
"""Build a unified OpenSearch-VL dataset from benchmark-specific sources.

Example:
    python scripts/build_benchmark_dataset.py --benchmark mmsearch
    python scripts/build_benchmark_dataset.py --benchmark mmsearch_plus --split train
    python scripts/build_benchmark_dataset.py --benchmark vdr_bench_testmini --output data/benchmarks/vdr/testmini.parquet
"""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import importlib.util
import json
import random
import hashlib
import textwrap
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Callable, Iterable, Optional
from urllib.parse import quote


BROWSECOMP_V3_DEFAULT_KEY = "A_Visual_Vertical_Verifiable_Benchmark_for_Multimodal_Browsing_Agents"
BROWSECOMP_V3_KEY = BROWSECOMP_V3_DEFAULT_KEY


def _normalize_image_reference(image_url: str) -> str:
    """Convert local absolute paths to ``file://`` URLs for API backends."""

    if image_url.startswith(("http://", "https://", "data:", "file://")):
        return image_url

    candidate = Path(image_url).expanduser()
    if candidate.is_absolute():
        return candidate.resolve().as_uri()
    return image_url


def _optional_numpy() -> Any:
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError:
        return None
    return np


def _maybe_decode_base64_image(value: str) -> Optional[bytes]:
    stripped = value.strip()
    if not stripped:
        return None
    if stripped.startswith(("http://", "https://", "file://", "data:")):
        return None
    if "\n" in stripped or "\r" in stripped:
        stripped = "".join(stripped.split())
    try:
        decoded = base64.b64decode(stripped, validate=True)
    except (binascii.Error, ValueError):
        return None

    if decoded.startswith(
        (
            b"\xff\xd8\xff",  # jpeg
            b"\x89PNG\r\n\x1a\n",  # png
            b"GIF87a",
            b"GIF89a",
            b"RIFF",  # webp container, validated below
        )
    ):
        if decoded.startswith(b"RIFF") and decoded[8:12] != b"WEBP":
            return None
        return decoded
    return None


def _numeric_sequence_to_bytes(value: Any) -> Optional[bytes]:
    if not isinstance(value, list) or not value:
        return None
    if not all(isinstance(item, int) and 0 <= item <= 255 for item in value):
        return None
    return bytes(value)


def _numpy_array_to_bytes(value: Any) -> Optional[bytes]:
    np = _optional_numpy()
    if np is None or not isinstance(value, np.ndarray):
        return None

    if value.ndim == 1:
        try:
            return bytes(value.astype("uint8").tolist())
        except Exception as exc:
            raise TypeError(f"Failed to convert 1D numpy image buffer: {exc}") from exc

    if value.ndim in (2, 3):
        try:
            from PIL import Image
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Converting numpy pixel arrays to encoded images requires Pillow."
            ) from exc

        array = value
        if str(array.dtype) != "uint8":
            array = array.astype("uint8")
        image = Image.fromarray(array)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    return None


def _write_parquet(rows: list[dict[str, object]], output_path: Path) -> None:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Writing parquet requires pandas plus a parquet backend such as "
            "pyarrow or fastparquet."
        ) from exc

    df = pd.DataFrame(rows)
    buffer = io.BytesIO()
    try:
        df.to_parquet(buffer, index=False)
    except Exception as exc:
        raise RuntimeError(
            "Failed to write parquet. Make sure pyarrow or fastparquet is installed."
        ) from exc
    buffer.seek(0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(buffer.getvalue())


def _write_json(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _guess_mime_from_bytes(data: bytes) -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def _image_entry_to_svg_href(image_entry: Any) -> str:
    if isinstance(image_entry, dict):
        for key in ("url", "image_url", "cos_url", "path"):
            value = image_entry.get(key)
            if value is not None and str(value).strip():
                return _normalize_image_reference(str(value).strip())
        for key in ("bytes", "data"):
            if key not in image_entry or image_entry[key] is None:
                continue
            payload = _image_bytes_from_object({key: image_entry[key]})
            mime = _guess_mime_from_bytes(payload)
            return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"
    if isinstance(image_entry, (bytes, bytearray)):
        payload = bytes(image_entry)
        mime = _guess_mime_from_bytes(payload)
        return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"
    if isinstance(image_entry, str):
        stripped = image_entry.strip()
        if not stripped:
            return ""
        decoded = _maybe_decode_base64_image(stripped)
        if decoded is not None:
            mime = _guess_mime_from_bytes(decoded)
            return f"data:{mime};base64,{base64.b64encode(decoded).decode('ascii')}"
        return _normalize_image_reference(stripped)
    return ""


def _wrap_svg_text(text: Any, *, width: int = 92, max_lines: int = 7) -> list[str]:
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return ["-"]
    lines = textwrap.wrap(normalized, width=width, break_long_words=False, break_on_hyphens=False)
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + [lines[max_lines - 1][: max(0, width - 3)].rstrip() + "..."]
    return lines or ["-"]


def _write_preview_svg(
    rows: list[dict[str, object]],
    *,
    output_path: Path,
    sample_size: int,
    seed: int,
) -> Path | None:
    if not rows or sample_size <= 0:
        return None
    rng = random.Random(seed)
    count = min(max(1, sample_size), 3, len(rows))
    indices = sorted(rng.sample(range(len(rows)), count)) if len(rows) > count else list(range(len(rows)))
    selected = [rows[index] for index in indices]

    card_w = 1200
    card_h = 300
    margin = 24
    gap = 20
    image_w = 300
    image_h = 220
    total_h = margin * 2 + len(selected) * card_h + max(0, len(selected) - 1) * gap
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{card_w}" height="{total_h}" viewBox="0 0 {card_w} {total_h}">',
        '<style>text{font-family:Arial,Helvetica,sans-serif} .small{font-size:16px;fill:#555} .body{font-size:18px;fill:#111} .title{font-size:20px;font-weight:bold}</style>',
        '<rect width="100%" height="100%" fill="#f7f7f7"/>',
    ]
    y = margin
    for ordinal, row in enumerate(selected, start=1):
        sample_id = str(row.get("sample_id") or row.get("data_id") or row.get("question_id") or f"row_{ordinal}")
        category = str(row.get("category") or "unknown")
        question = row.get("question") or ""
        answer = row.get("answer") or ""
        images = row.get("images") if isinstance(row.get("images"), list) else []
        href = _image_entry_to_svg_href(images[0]) if images else ""
        parts.append(f'<rect x="{margin}" y="{y}" width="{card_w - 2 * margin}" height="{card_h}" rx="14" fill="white" stroke="#ddd"/>')
        parts.append(f'<text x="{margin + 18}" y="{y + 30}" class="title">{escape(sample_id)}</text>')
        parts.append(f'<text x="{margin + 18}" y="{y + 56}" class="small">category={escape(category)} | source={escape(str(row.get("data_source") or ""))}</text>')
        image_x = margin + 18
        image_y = y + 70
        parts.append(f'<rect x="{image_x}" y="{image_y}" width="{image_w}" height="{image_h}" fill="#eee" stroke="#ccc"/>')
        if href:
            parts.append(f'<image href="{escape(href)}" x="{image_x}" y="{image_y}" width="{image_w}" height="{image_h}" preserveAspectRatio="xMidYMid meet"/>')
        else:
            parts.append(f'<text x="{image_x + 20}" y="{image_y + 110}" class="small">no image</text>')
        text_x = image_x + image_w + 24
        text_y = y + 88
        parts.append(f'<text x="{text_x}" y="{text_y}" class="small">Question</text>')
        for line in _wrap_svg_text(question, width=86, max_lines=6):
            text_y += 22
            parts.append(f'<text x="{text_x}" y="{text_y}" class="body">{escape(line)}</text>')
        text_y += 34
        parts.append(f'<text x="{text_x}" y="{text_y}" class="small">Answer</text>')
        for line in _wrap_svg_text(answer, width=86, max_lines=3):
            text_y += 22
            parts.append(f'<text x="{text_x}" y="{text_y}" class="body">{escape(line)}</text>')
        y += card_h + gap
    parts.append("</svg>")

    preview_path = output_path.parent / f"{output_path.stem}_preview.svg"
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return preview_path


def _stringify_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(parts)
    return str(value).strip()


def _question_text(record: dict[str, Any], candidates: Iterable[str]) -> str:
    for key in candidates:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _first_non_empty(record: dict[str, Any], candidates: Iterable[str]) -> str:
    for key in candidates:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _prompt_content(value: Any) -> str:
    value = _maybe_parse_json_string(value)
    if isinstance(value, list):
        pieces: list[str] = []
        for item in value:
            if isinstance(item, dict):
                content = item.get("content")
                if content is not None and str(content).strip():
                    pieces.append(str(content).strip())
            elif item is not None and str(item).strip():
                pieces.append(str(item).strip())
        return "\n".join(pieces).strip()
    if isinstance(value, dict):
        return str(value.get("content") or "").strip()
    return str(value or "").strip()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, str) and len(value) > 5000:
            parsed = _maybe_parse_json_string(value)
            if parsed is not value:
                return _json_safe(parsed)
            if _maybe_decode_base64_image(value) is not None:
                return f"<base64-image-string:{len(value)} chars>"
        return value
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if isinstance(value, list):
        if len(value) > 1000 and all(isinstance(item, int) and 0 <= item <= 255 for item in value[:1000]):
            return f"<byte-list:{len(value)} items>"
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    # Handle PIL images or dataset image wrappers without importing PIL eagerly.
    if hasattr(value, "size") and hasattr(value, "mode"):
        size = getattr(value, "size", None)
        mode = getattr(value, "mode", None)
        return {"_type": "PIL.Image", "size": size, "mode": mode}

    return str(value)


def _maybe_parse_json_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _derive_repeating_xor_key(password: str, length: int) -> bytes:
    digest = hashlib.sha256(password.encode()).digest()
    return digest * (length // len(digest)) + digest[: length % len(digest)]


def _decrypt_visbrowse_text(ciphertext_b64: str, password: str) -> str:
    encrypted = base64.b64decode(ciphertext_b64)
    key = _derive_repeating_xor_key(password, len(encrypted))
    return bytes(a ^ b for a, b in zip(encrypted, key)).decode()


def _decrypt_browsecomp_v3_text(encrypted: Any, key_text: str) -> str:
    encrypted = _maybe_parse_json_string(encrypted)
    if not isinstance(encrypted, dict):
        return ""
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "BrowseComp-V3 decryption requires the cryptography package."
        ) from exc
    key = hashlib.sha256(key_text.encode("utf-8")).digest()
    iv = base64.b64decode(encrypted["iv"])
    ciphertext = base64.b64decode(encrypted["ciphertext"])
    tag = base64.b64decode(encrypted["tag"])
    return AESGCM(key).decrypt(iv, ciphertext + tag, None).decode("utf-8")


def _image_bytes_from_object(value: Any) -> bytes:
    value = _maybe_parse_json_string(value)
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        decoded = _maybe_decode_base64_image(value)
        if decoded is not None:
            return decoded
        return base64.b64decode(value)
    numeric_bytes = _numeric_sequence_to_bytes(value)
    if numeric_bytes is not None:
        return numeric_bytes
    numpy_bytes = _numpy_array_to_bytes(value)
    if numpy_bytes is not None:
        return numpy_bytes
    if isinstance(value, dict):
        if "bytes" in value and value.get("bytes") is not None:
            return _image_bytes_from_object(value["bytes"])
        if "data" in value and value.get("data") is not None:
            return _image_bytes_from_object(value["data"])
    if hasattr(value, "save"):
        buffer = io.BytesIO()
        value.save(buffer, format="PNG")
        return buffer.getvalue()
    raise TypeError(f"Unsupported image payload type: {type(value)!r}")


def _image_entry_from_value(value: Any) -> Optional[dict[str, object]]:
    value = _maybe_parse_json_string(value)
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        decoded = _maybe_decode_base64_image(stripped)
        if decoded is not None:
            return {"bytes": decoded}
        return {"url": _normalize_image_reference(stripped)}
    if isinstance(value, dict):
        # Dataset image structs may contain both an embedded ``bytes`` payload
        # and a repository-relative ``path``.  Prefer the embedded payload so
        # the converted parquet remains self-contained; a relative path is not
        # a URL that the inference runtime can dereference.
        if any(key in value for key in ("bytes", "data")):
            payload = _image_bytes_from_object(value)
            if payload:
                return {"bytes": payload}
        url_value = value.get("url")
        if url_value is not None and str(url_value).strip() and str(url_value).strip().lower() != "none":
            return {"url": _normalize_image_reference(str(url_value).strip())}
        path_value = value.get("path")
        if path_value is not None and str(path_value).strip() and str(path_value).strip().lower() != "none":
            return {"url": _normalize_image_reference(str(path_value).strip())}
    return {"bytes": _image_bytes_from_object(value)}


def _collect_images(record: dict[str, Any], image_keys: Iterable[str]) -> list[dict[str, object]]:
    images: list[dict[str, object]] = []
    seen: set[str] = set()
    for key in image_keys:
        if key not in record:
            continue
        value = _maybe_parse_json_string(record.get(key))
        if value is None:
            continue
        candidates = value if isinstance(value, list) else [value]
        for item in candidates:
            image_entry = _image_entry_from_value(item)
            if not image_entry:
                continue
            dedupe_key = json.dumps(_json_safe(image_entry), ensure_ascii=False, sort_keys=True)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            images.append(image_entry)
    return images


@dataclass(frozen=True)
class BenchmarkAdapter:
    benchmark: str
    default_dataset: str
    default_split: str
    data_source: str
    question_keys: tuple[str, ...]
    answer_keys: tuple[str, ...]
    id_keys: tuple[str, ...]
    sample_id_keys: tuple[str, ...]
    category_keys: tuple[str, ...]
    image_keys: tuple[str, ...]
    row_builder: Optional[Callable[[dict[str, Any], "BenchmarkAdapter"], dict[str, object]]] = None


def _default_row_builder(
    record: dict[str, Any],
    adapter: BenchmarkAdapter,
) -> dict[str, object]:
    question = _question_text(record, adapter.question_keys)
    answer = ""
    for key in adapter.answer_keys:
        if key in record:
            answer = _stringify_answer(record.get(key))
            if answer:
                break

    data_id = _first_non_empty(record, adapter.id_keys) or _first_non_empty(
        record, adapter.sample_id_keys
    )
    sample_id = _first_non_empty(record, adapter.sample_id_keys) or data_id
    question_id = _first_non_empty(record, ("question_id", "qid", "id")) or data_id
    category = _first_non_empty(record, adapter.category_keys) or "unknown"
    images = _collect_images(record, adapter.image_keys)

    source_record = _json_safe(record)
    row: dict[str, object] = {
        "data_id": data_id,
        "question_id": question_id,
        "sample_id": sample_id,
        "category": category,
        "data_source": adapter.data_source,
        "question": question,
        "prompt": [{"content": question}],
        "images": images,
        "answer": answer,
        "source_metadata": json.dumps(source_record, ensure_ascii=False),
    }
    return row


def _infer_vdr_category(sample_id: str) -> str:
    parts = sample_id.split("_")
    return parts[1] if len(parts) > 1 else "unknown"


def _build_vdr_row(record: dict[str, Any], adapter: BenchmarkAdapter) -> dict[str, object]:
    row = _default_row_builder(record, adapter)
    sample_id = str(record.get("id") or row["data_id"])
    row["data_id"] = sample_id
    row["question_id"] = str(record.get("question_id") or sample_id)
    row["sample_id"] = sample_id
    row["category"] = _infer_vdr_category(sample_id)
    return row


def _build_mmsearch_plus_row(
    record: dict[str, Any],
    adapter: BenchmarkAdapter,
) -> dict[str, object]:
    row = _default_row_builder(record, adapter)

    num_images = record.get("num_images")
    extra_images: list[dict[str, object]] = []
    if isinstance(num_images, int) and num_images > 0:
        for index in range(1, num_images + 1):
            image_key = f"img_{index}"
            if image_key not in record:
                continue
            image_entry = _image_entry_from_value(record.get(image_key))
            if image_entry:
                extra_images.append(image_entry)
    else:
        for key, value in record.items():
            if key.startswith("img_"):
                image_entry = _image_entry_from_value(value)
                if image_entry:
                    extra_images.append(image_entry)

    if extra_images:
        seen = {
            json.dumps(_json_safe(item), ensure_ascii=False, sort_keys=True)
            for item in row.get("images", [])
        }
        merged = list(row.get("images", []))
        for item in extra_images:
            token = json.dumps(_json_safe(item), ensure_ascii=False, sort_keys=True)
            if token in seen:
                continue
            seen.add(token)
            merged.append(item)
        row["images"] = merged

    return row


def _build_simplevqa_row(record: dict[str, Any], adapter: BenchmarkAdapter) -> dict[str, object]:
    row = _default_row_builder(record, adapter)
    data_id = str(record.get("data_id") or row.get("data_id") or "").strip()
    if not data_id:
        data_id = str(row.get("sample_id") or row.get("question_id") or "")
    if data_id and not data_id.startswith("simplevqa_"):
        data_id = f"simplevqa_{data_id}"
    row["data_id"] = data_id
    row["question_id"] = data_id
    row["sample_id"] = data_id
    row["category"] = str(record.get("original_category") or record.get("language") or row.get("category") or "unknown")
    row["language"] = record.get("language")
    row["source_url"] = record.get("source")
    row["atomic_question"] = record.get("atomic_question")
    row["atomic_fact"] = record.get("atomic_fact")
    return row


def _build_livevqa_row(record: dict[str, Any], adapter: BenchmarkAdapter) -> dict[str, object]:
    row = _default_row_builder(record, adapter)
    sample_id = str(record.get("id") or row.get("data_id") or "").strip()
    if not sample_id:
        sample_id = str(row.get("question_id") or "")
    if sample_id and not sample_id.startswith("livevqa_"):
        sample_id = f"livevqa_{sample_id}"
    row["data_id"] = sample_id
    row["question_id"] = sample_id
    row["sample_id"] = sample_id
    row["category"] = str(record.get("topic") or record.get("source") or row.get("category") or "unknown")
    row["level"] = record.get("level")
    row["options"] = record.get("options") or []
    row["ground_truth_list"] = record.get("Ground_Truth_List") or []
    image_path = str(record.get("img_path") or "").strip()
    if image_path:
        row["image_path"] = image_path
        row["images"] = [
            {
                "url": f"https://huggingface.co/datasets/ONE-Lab/LiveVQA-new/resolve/main/{quote(image_path)}"
            }
        ]
    return row


def _build_fvqa_row(record: dict[str, Any], adapter: BenchmarkAdapter) -> dict[str, object]:
    row = _default_row_builder(record, adapter)
    reward = record.get("reward_model") or {}
    if not isinstance(reward, dict):
        reward = {}
    answer = _stringify_answer(reward.get("ground_truth")) or _stringify_answer(record.get("answer"))
    data_id = str(record.get("data_id") or row.get("data_id") or "").strip()
    question = _prompt_content(record.get("prompt")) or str(row.get("question") or "").strip()
    row["data_id"] = data_id
    row["question_id"] = data_id
    row["sample_id"] = data_id
    row["question"] = question
    row["prompt"] = record.get("prompt") if isinstance(record.get("prompt"), list) else [{"content": question}]
    row["answer"] = answer
    row["category"] = str(record.get("category") or row.get("category") or "unknown")
    row["reward_model"] = _json_safe(reward)
    return row


def _build_visbrowse_row(record: dict[str, Any], adapter: BenchmarkAdapter) -> dict[str, object]:
    decrypted = dict(record)
    password = str(record.get("canary") or "").strip()
    if password:
        question_cipher = str(record.get("question") or "").strip()
        if question_cipher:
            decrypted["question"] = _decrypt_visbrowse_text(question_cipher, password)
        answers = record.get("answer") or []
        if isinstance(answers, list):
            decrypted["answer"] = [
                _decrypt_visbrowse_text(str(answer), password)
                for answer in answers
                if str(answer or "").strip()
            ]
    row = _default_row_builder(decrypted, adapter)
    sample_id = str(record.get("id") or row.get("data_id") or "").strip()
    if sample_id and not sample_id.startswith("visbrowse_"):
        sample_id = f"visbrowse_{sample_id}"
    row["data_id"] = sample_id
    row["question_id"] = sample_id
    row["sample_id"] = sample_id
    row["category"] = str(record.get("domain") or row.get("category") or "unknown")
    row["sub_category"] = record.get("sub_domain")
    row["source_metadata"] = json.dumps(_json_safe(record), ensure_ascii=False)
    return row


def _browsecomp_v3_images(record: dict[str, Any]) -> list[dict[str, object]]:
    paths: list[str] = []
    for key in ("image_paths", "image"):
        value = _maybe_parse_json_string(record.get(key))
        if isinstance(value, list):
            paths.extend(str(item).strip() for item in value if str(item or "").strip())
        elif value is not None and str(value).strip():
            paths.append(str(value).strip())
    seen: set[str] = set()
    images: list[dict[str, object]] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        images.append(
            {
                "url": f"https://huggingface.co/datasets/Halcyon-Zhang/BrowseComp-V3/resolve/main/{quote(path)}"
            }
        )
    return images


def _build_browsecomp_v3_row(record: dict[str, Any], adapter: BenchmarkAdapter) -> dict[str, object]:
    row = _default_row_builder(record, adapter)
    question = _decrypt_browsecomp_v3_text(record.get("encrypted_question"), BROWSECOMP_V3_KEY)
    answer = _decrypt_browsecomp_v3_text(record.get("encrypted_answer"), BROWSECOMP_V3_KEY)
    sample_id = str(record.get("id") or row.get("data_id") or "").strip()
    row["data_id"] = sample_id
    row["question_id"] = sample_id
    row["sample_id"] = sample_id
    row["question"] = question
    row["prompt"] = [{"content": question}]
    row["answer"] = answer
    row["category"] = str(record.get("category") or row.get("category") or "unknown")
    row["sub_category"] = record.get("sub_category")
    row["images"] = _browsecomp_v3_images(record)
    # Keep nested benchmark annotations as JSON strings so pyarrow does not
    # have to infer mixed list/dict scalar types across rows.
    row["metadata"] = json.dumps(
        _json_safe(_maybe_parse_json_string(record.get("metadata"))),
        ensure_ascii=False,
    )
    row["sub_goals"] = json.dumps(
        _json_safe(_maybe_parse_json_string(record.get("sub_goals"))),
        ensure_ascii=False,
    )
    row["source_metadata"] = json.dumps(_json_safe(record), ensure_ascii=False)
    return row


ADAPTERS: dict[str, BenchmarkAdapter] = {
    "mmsearch": BenchmarkAdapter(
        benchmark="mmsearch",
        default_dataset="CaraJ/MMSearch",
        default_split="end2end",
        data_source="MMSearch",
        question_keys=("question", "query", "prompt"),
        answer_keys=("answer", "answers", "gt_answer"),
        id_keys=("id", "question_id", "sample_id"),
        sample_id_keys=("sample_id", "id", "question_id"),
        category_keys=("category", "subfield", "field", "domain"),
        # The MMSearch source stores the question image in ``query_image``.
        # Older converted parquet files only looked for ``images``/``image``
        # and consequently dropped the visual input entirely.
        image_keys=("query_image", "images", "image"),
    ),
    "mmsearch_plus": BenchmarkAdapter(
        benchmark="mmsearch_plus",
        default_dataset="Cie1/MMSearch-Plus",
        default_split="train",
        data_source="MMSearch-Plus",
        question_keys=("question", "query", "prompt"),
        answer_keys=("answer", "answers", "gt_answer", "acceptable_answers"),
        id_keys=("id", "question_id", "sample_id"),
        sample_id_keys=("sample_id", "id", "question_id"),
        category_keys=("category", "primary_category", "secondary_category", "domain"),
        image_keys=("images", "image"),
        row_builder=_build_mmsearch_plus_row,
    ),
    "vdr_bench": BenchmarkAdapter(
        benchmark="vdr_bench",
        default_dataset="Osilly/VDR-Bench",
        default_split="train",
        data_source="VDR-Bench",
        question_keys=("question", "query", "prompt"),
        answer_keys=("answer", "answers"),
        id_keys=("id", "question_id", "sample_id"),
        sample_id_keys=("sample_id", "id", "question_id"),
        category_keys=("category",),
        image_keys=("image", "images"),
        row_builder=_build_vdr_row,
    ),
    "vdr_bench_testmini": BenchmarkAdapter(
        benchmark="vdr_bench_testmini",
        default_dataset="Osilly/VDR-Bench-testmini",
        default_split="train",
        data_source="VDR-Bench-testmini",
        question_keys=("question", "query", "prompt"),
        answer_keys=("answer", "answers"),
        id_keys=("id", "question_id", "sample_id"),
        sample_id_keys=("sample_id", "id", "question_id"),
        category_keys=("category",),
        image_keys=("image", "images"),
        row_builder=_build_vdr_row,
    ),
    "simplevqa": BenchmarkAdapter(
        benchmark="simplevqa",
        default_dataset="m-a-p/SimpleVQA",
        default_split="test",
        data_source="SimpleVQA",
        question_keys=("question",),
        answer_keys=("answer",),
        id_keys=("data_id",),
        sample_id_keys=("data_id",),
        category_keys=("original_category", "vqa_category", "language"),
        image_keys=("image",),
        row_builder=_build_simplevqa_row,
    ),
    "livevqa": BenchmarkAdapter(
        benchmark="livevqa",
        default_dataset="ONE-Lab/LiveVQA-new",
        default_split="train",
        data_source="LiveVQA-new",
        question_keys=("question",),
        answer_keys=("Ground_Truth_List", "Ground_Truth"),
        id_keys=("id",),
        sample_id_keys=("id",),
        category_keys=("topic", "source", "level"),
        image_keys=("img_path", "image", "images"),
        row_builder=_build_livevqa_row,
    ),
    "fvqa": BenchmarkAdapter(
        benchmark="fvqa",
        default_dataset="lmms-lab/FVQA",
        default_split="test",
        data_source="FVQA",
        question_keys=("question", "query", "prompt"),
        answer_keys=("answer",),
        id_keys=("data_id",),
        sample_id_keys=("data_id",),
        category_keys=("category", "data_source"),
        image_keys=("images", "image_urls", "image"),
        row_builder=_build_fvqa_row,
    ),
    "visbrowse_bench": BenchmarkAdapter(
        benchmark="visbrowse_bench",
        default_dataset="Zhengbo-Zhang/VisBrowse-Bench",
        default_split="train",
        data_source="VisBrowse-Bench",
        question_keys=("question",),
        answer_keys=("answer",),
        id_keys=("id",),
        sample_id_keys=("id",),
        category_keys=("domain", "sub_domain"),
        image_keys=("image",),
        row_builder=_build_visbrowse_row,
    ),
    "browsecomp_v3": BenchmarkAdapter(
        benchmark="browsecomp_v3",
        default_dataset="Halcyon-Zhang/BrowseComp-V3",
        default_split="train",
        data_source="BrowseComp-V3",
        question_keys=("question", "encrypted_question"),
        answer_keys=("answer", "encrypted_answer"),
        id_keys=("id",),
        sample_id_keys=("id",),
        category_keys=("category", "sub_category"),
        image_keys=("image_paths", "image"),
        row_builder=_build_browsecomp_v3_row,
    ),
}


def _load_dataset_records(
    dataset_name: str,
    split: str,
    cache_dir: Optional[str] = None,
    benchmark: Optional[str] = None,
    mmsearch_plus_decrypt_script: Optional[str] = None,
    mmsearch_plus_canary: Optional[str] = None,
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": split}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    if benchmark == "mmsearch":
        # MMSearch requires an explicit config name, and the config should stay
        # aligned with the requested split such as end2end/rerank/summarization.
        dataset = load_dataset(dataset_name, split, **kwargs)
    else:
        dataset = load_dataset(dataset_name, **kwargs)

    if benchmark == "mmsearch_plus":
        decrypt_script = (
            Path(mmsearch_plus_decrypt_script).expanduser().resolve()
            if mmsearch_plus_decrypt_script
            else None
        )
        if decrypt_script:
            dataset = _decrypt_mmsearch_plus_dataset(
                dataset=dataset,
                decrypt_script=decrypt_script,
                canary=mmsearch_plus_canary or "MMSearch-Plus",
            )

    return [dict(item) for item in dataset]


def _decrypt_mmsearch_plus_dataset(
    dataset: Any,
    decrypt_script: Path,
    canary: str,
) -> Any:
    if not decrypt_script.exists():
        raise FileNotFoundError(
            f"MMSearch-Plus decrypt script does not exist: {decrypt_script}"
        )

    spec = importlib.util.spec_from_file_location(
        "mmsearch_plus_decrypt_module",
        decrypt_script,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load decrypt script: {decrypt_script}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    decrypt_dataset = getattr(module, "decrypt_dataset", None)
    if decrypt_dataset is None:
        raise AttributeError(
            f"decrypt_dataset() not found in decrypt script: {decrypt_script}"
        )

    return decrypt_dataset(
        encrypted_dataset=dataset,
        canary=canary,
    )


def _build_rows(
    records: list[dict[str, Any]],
    adapter: BenchmarkAdapter,
    limit: int = 0,
) -> list[dict[str, object]]:
    builder = adapter.row_builder or _default_row_builder
    rows: list[dict[str, object]] = []
    for index, record in enumerate(records):
        if limit > 0 and index >= limit:
            break
        row = builder(record, adapter)
        if not row.get("question"):
            raise ValueError(
                f"Missing question text for benchmark={adapter.benchmark} "
                f"record index={index}"
            )
        rows.append(row)
    return rows


def _default_output_path(
    benchmark: str,
    split: str,
    output_format: str,
) -> Path:
    suffix = "json" if output_format == "json" else "parquet"
    if benchmark == "mmsearch":
        return Path("data") / "benchmarks" / benchmark / split / f"data.{suffix}"
    return Path("data") / "benchmarks" / benchmark / f"{split}.{suffix}"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=sorted(ADAPTERS.keys()),
        help="Benchmark adapter to use.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Override the Hugging Face dataset name/path.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split to load. For MMSearch this is typically end2end/rerank/summarization.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output dataset path. Defaults to "
            "data/benchmarks/<benchmark>/<split>.parquet"
        ),
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "json"],
        default="parquet",
        help="Dataset format to write.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optionally convert only the first N records.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face datasets cache dir.",
    )
    parser.add_argument(
        "--mmsearch-plus-decrypt-script",
        default=None,
        help=(
            "Path to MMSearch-Plus decrypt_after_load.py. "
            "Needed when converting the encrypted HF dataset."
        ),
    )
    parser.add_argument(
        "--mmsearch-plus-canary",
        default="MMSearch-Plus",
        help=(
            "Canary string for MMSearch-Plus decryption. "
            "The dataset card hints this is the repo name without username."
        ),
    )
    parser.add_argument(
        "--preview-size",
        type=int,
        default=3,
        help="Randomly sample 1-3 converted rows and write an SVG preview next to the output. Use 0 to disable.",
    )
    parser.add_argument(
        "--preview-seed",
        type=int,
        default=0,
        help="Random seed used for SVG preview sampling.",
    )
    parser.add_argument(
        "--browsecomp-v3-key",
        default=BROWSECOMP_V3_DEFAULT_KEY,
        help="Passphrase used to decrypt BrowseComp-V3 question/answer fields.",
    )
    return parser


def main() -> None:
    global BROWSECOMP_V3_KEY
    args = _build_arg_parser().parse_args()
    BROWSECOMP_V3_KEY = args.browsecomp_v3_key
    adapter = ADAPTERS[args.benchmark]
    dataset_name = args.dataset or adapter.default_dataset
    split = args.split or adapter.default_split
    output_path = (
        Path(args.output).expanduser()
        if args.output
        else _default_output_path(args.benchmark, split, args.format)
    )

    records = _load_dataset_records(
        dataset_name=dataset_name,
        split=split,
        cache_dir=args.cache_dir,
        benchmark=args.benchmark,
        mmsearch_plus_decrypt_script=args.mmsearch_plus_decrypt_script,
        mmsearch_plus_canary=args.mmsearch_plus_canary,
    )
    rows = _build_rows(records, adapter, limit=args.limit)

    if args.format == "json":
        _write_json(rows, output_path)
    else:
        _write_parquet(rows, output_path)
    preview_path = _write_preview_svg(
        rows,
        output_path=output_path,
        sample_size=args.preview_size,
        seed=args.preview_seed,
    )

    print(
        f"Wrote {len(rows)} rows to {output_path} "
        f"(benchmark={args.benchmark}, dataset={dataset_name}, split={split})"
    )
    if preview_path is not None:
        print(f"Preview SVG: {preview_path}")


if __name__ == "__main__":
    main()
