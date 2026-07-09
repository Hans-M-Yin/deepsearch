#!/usr/bin/env python3
"""Render benchmark parquet rows into a portable SVG contact sheet."""

from __future__ import annotations

import argparse
import ast
import base64
import json
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from html import escape
from pathlib import Path
from typing import Any


CARD_WIDTH = 1200
CARD_HEIGHT = 320
CARD_GAP = 24
PAGE_MARGIN = 24
IMAGE_BOX_WIDTH = 320
IMAGE_BOX_HEIGHT = 240
TEXT_X_OFFSET = 380


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", help="Input parquet path.")
    parser.add_argument("--num", type=int, default=10, help="Number of rows to render.")
    parser.add_argument("--start", type=int, default=0, help="Starting row index.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output SVG path.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle rows before slicing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used with --shuffle.",
    )
    parser.add_argument(
        "--fetch-timeout",
        type=float,
        default=15.0,
        help="Timeout in seconds for downloading remote images.",
    )
    return parser.parse_args()


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    df = pd.read_parquet(path)
    return df.to_dict(orient="records")


def _maybe_json_loads(value: str) -> Any:
    text = value.strip()
    if not text:
        return value
    if text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(text)
        except Exception:
            return value


def _normalize_nested(value: Any) -> Any:
    if isinstance(value, str):
        parsed = _maybe_json_loads(value)
        if parsed is not value:
            return _normalize_nested(parsed)
        return value
    if isinstance(value, list):
        return [_normalize_nested(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _normalize_nested(v) for k, v in value.items()}
    return value


def _extract_question(row: dict[str, Any]) -> str:
    for key in ("question", "query"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()

    prompt = _normalize_nested(row.get("prompt"))
    if isinstance(prompt, list):
        pieces: list[str] = []
        for item in prompt:
            if isinstance(item, dict):
                content = item.get("content")
                if content is not None and str(content).strip():
                    pieces.append(str(content).strip())
            elif item is not None and str(item).strip():
                pieces.append(str(item).strip())
        if pieces:
            return "\n".join(pieces)
    return ""


def _extract_answer(row: dict[str, Any]) -> str:
    value = row.get("answer")
    if isinstance(value, list):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    if value is not None and str(value).strip():
        return str(value).strip()

    for key in ("answers", "gt_answer", "acceptable_answers"):
        candidate = row.get(key)
        if isinstance(candidate, list):
            merged = ", ".join(str(item).strip() for item in candidate if str(item).strip())
            if merged:
                return merged
        elif candidate is not None and str(candidate).strip():
            return str(candidate).strip()
    return ""


def _extract_sample_id(row: dict[str, Any], row_index: int) -> str:
    for key in ("data_id", "sample_id", "question_id", "id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return f"row_{row_index}"


def _extract_images(row: dict[str, Any]) -> list[Any]:
    for key in ("images", "image"):
        value = row.get(key)
        if value is None:
            continue
        value = _normalize_nested(value)
        if isinstance(value, list):
            return value
        return [value]
    return []


def _guess_mime_from_bytes(data: bytes) -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"BM"):
        return "image/bmp"
    if data.lstrip().startswith(b"<svg") or b"<svg" in data[:512]:
        return "image/svg+xml"
    return "application/octet-stream"


def _is_probable_base64(text: str) -> bool:
    stripped = "".join(text.strip().split())
    if len(stripped) < 32:
        return False
    if stripped.startswith(("http://", "https://", "file://", "data:")):
        return False
    try:
        base64.b64decode(stripped, validate=True)
    except Exception:
        return False
    return True


def _fetch_url_bytes(url: str, timeout: float) -> tuple[bytes, str]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        data = response.read()
        mime = response.headers.get_content_type() or _guess_mime_from_bytes(data)
    return data, mime


def _load_file_url(file_url: str) -> tuple[bytes, str]:
    parsed = urllib.parse.urlparse(file_url)
    path = urllib.request.url2pathname(parsed.path)
    data = Path(path).read_bytes()
    return data, _guess_mime_from_bytes(data)


def _image_entry_to_data_uri(image_entry: Any, timeout: float) -> tuple[str | None, str | None]:
    if image_entry is None:
        return None, "missing_image"

    if isinstance(image_entry, dict):
        if "url" in image_entry and image_entry["url"]:
            return _string_image_to_data_uri(str(image_entry["url"]), timeout)
        if "path" in image_entry and image_entry["path"]:
            return _string_image_to_data_uri(str(image_entry["path"]), timeout)
        if "bytes" in image_entry and image_entry["bytes"] is not None:
            return _bytes_to_data_uri(_coerce_bytes(image_entry["bytes"])), None
        if "data" in image_entry and image_entry["data"] is not None:
            return _bytes_to_data_uri(_coerce_bytes(image_entry["data"])), None

    if isinstance(image_entry, (bytes, bytearray)):
        return _bytes_to_data_uri(bytes(image_entry)), None

    if isinstance(image_entry, str):
        return _string_image_to_data_uri(image_entry, timeout)

    return None, f"unsupported_image_type:{type(image_entry).__name__}"


def _coerce_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, list) and all(isinstance(item, int) and 0 <= item <= 255 for item in value):
        return bytes(value)
    if isinstance(value, str):
        if value.startswith("data:"):
            payload = value.split("base64,", 1)[-1]
            return base64.b64decode(payload)
        return base64.b64decode("".join(value.split()))
    raise TypeError(f"Unsupported bytes payload: {type(value)!r}")


def _bytes_to_data_uri(data: bytes) -> str:
    mime = _guess_mime_from_bytes(data)
    payload = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _string_image_to_data_uri(value: str, timeout: float) -> tuple[str | None, str | None]:
    text = value.strip()
    if not text:
        return None, "empty_image_reference"

    if text.startswith("data:"):
        return text, None

    if text.startswith(("http://", "https://")):
        try:
            data, mime = _fetch_url_bytes(text, timeout)
            payload = base64.b64encode(data).decode("ascii")
            return f"data:{mime};base64,{payload}", None
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            return None, f"url_fetch_failed:{exc}"

    if text.startswith("file://"):
        try:
            data, mime = _load_file_url(text)
            payload = base64.b64encode(data).decode("ascii")
            return f"data:{mime};base64,{payload}", None
        except OSError as exc:
            return None, f"file_read_failed:{exc}"

    candidate = Path(text).expanduser()
    if candidate.exists():
        try:
            data = candidate.read_bytes()
            return _bytes_to_data_uri(data), None
        except OSError as exc:
            return None, f"local_read_failed:{exc}"

    if _is_probable_base64(text):
        try:
            return _bytes_to_data_uri(base64.b64decode("".join(text.split()))), None
        except Exception as exc:
            return None, f"base64_decode_failed:{exc}"

    return None, "unrecognized_image_reference"


def _wrap_text(text: str, width: int) -> list[str]:
    normalized = " ".join((text or "").split())
    if not normalized:
        return []
    return textwrap.wrap(
        normalized,
        width=width,
        break_long_words=True,
        break_on_hyphens=False,
    )


def _render_text_block(
    lines: list[str],
    x: int,
    y: int,
    line_height: int,
    font_size: int,
    fill: str,
    font_weight: str = "400",
) -> str:
    if not lines:
        return ""
    tspans = []
    for index, line in enumerate(lines):
        dy = "0" if index == 0 else str(line_height)
        tspans.append(
            f'<tspan x="{x}" dy="{dy}">{escape(line)}</tspan>'
        )
    return (
        f'<text x="{x}" y="{y}" font-size="{font_size}" fill="{fill}" '
        f'font-family="Arial, Helvetica, sans-serif" font-weight="{font_weight}">'
        + "".join(tspans)
        + "</text>"
    )


def _build_card(row: dict[str, Any], row_index: int, y: int, timeout: float) -> str:
    sample_id = _extract_sample_id(row, row_index)
    question = _extract_question(row)
    answer = _extract_answer(row)
    category = str(row.get("category") or "")
    images = _extract_images(row)

    image_uri = None
    image_error = "missing_image"
    if images:
        image_uri, image_error = _image_entry_to_data_uri(images[0], timeout)

    header = f"{sample_id}"
    if category:
        header = f"{sample_id} | {category}"

    question_lines = _wrap_text(question, 64)
    answer_lines = _wrap_text(answer, 64)
    error_lines = _wrap_text(image_error or "", 40)

    parts = [
        f'<g transform="translate({PAGE_MARGIN},{y})">',
        f'<rect x="0" y="0" width="{CARD_WIDTH}" height="{CARD_HEIGHT}" rx="18" fill="#ffffff" stroke="#d0d7de" stroke-width="1.5"/>',
        f'<text x="24" y="32" font-size="18" fill="#1f2328" font-family="Arial, Helvetica, sans-serif" font-weight="700">{escape(header)}</text>',
        f'<rect x="24" y="52" width="{IMAGE_BOX_WIDTH}" height="{IMAGE_BOX_HEIGHT}" rx="12" fill="#f6f8fa" stroke="#d8dee4" stroke-width="1"/>',
    ]

    if image_uri:
        parts.append(
            f'<image x="24" y="52" width="{IMAGE_BOX_WIDTH}" height="{IMAGE_BOX_HEIGHT}" '
            f'preserveAspectRatio="xMidYMid meet" href="{escape(image_uri, quote=True)}"/>'
        )
    else:
        parts.append(
            f'<text x="44" y="160" font-size="18" fill="#57606a" font-family="Arial, Helvetica, sans-serif">image unavailable</text>'
        )
        if error_lines:
            parts.append(
                _render_text_block(
                    error_lines[:4],
                    x=44,
                    y=188,
                    line_height=20,
                    font_size=13,
                    fill="#57606a",
                )
            )

    parts.append(
        _render_text_block(
            ["Question"],
            x=TEXT_X_OFFSET,
            y=82,
            line_height=20,
            font_size=16,
            fill="#0969da",
            font_weight="700",
        )
    )
    parts.append(
        _render_text_block(
            question_lines[:7],
            x=TEXT_X_OFFSET,
            y=108,
            line_height=22,
            font_size=18,
            fill="#1f2328",
        )
    )
    parts.append(
        _render_text_block(
            ["Answer"],
            x=TEXT_X_OFFSET,
            y=236,
            line_height=20,
            font_size=16,
            fill="#1a7f37",
            font_weight="700",
        )
    )
    parts.append(
        _render_text_block(
            answer_lines[:3] or ["<empty>"],
            x=TEXT_X_OFFSET,
            y=262,
            line_height=22,
            font_size=18,
            fill="#1f2328",
        )
    )
    parts.append("</g>")
    return "".join(parts)


def _build_svg(rows: list[dict[str, Any]], timeout: float, source_path: Path, start: int) -> str:
    content_height = len(rows) * CARD_HEIGHT + max(0, len(rows) - 1) * CARD_GAP
    total_height = PAGE_MARGIN * 2 + 60 + content_height
    total_width = CARD_WIDTH + PAGE_MARGIN * 2

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{total_width}" height="{total_height}" '
            f'viewBox="0 0 {total_width} {total_height}">'
        ),
        '<rect width="100%" height="100%" fill="#f3f4f6"/>',
        f'<text x="{PAGE_MARGIN}" y="34" font-size="24" fill="#111827" font-family="Arial, Helvetica, sans-serif" font-weight="700">{escape(source_path.name)}</text>',
        f'<text x="{PAGE_MARGIN}" y="56" font-size="14" fill="#4b5563" font-family="Arial, Helvetica, sans-serif">rows {start}..{start + len(rows) - 1} | rendered {len(rows)} sample(s) | images embedded when readable</text>',
    ]

    cursor_y = PAGE_MARGIN + 60
    for row_index, row in enumerate(rows, start=start):
        parts.append(_build_card(row, row_index, cursor_y, timeout))
        cursor_y += CARD_HEIGHT + CARD_GAP

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    args = _parse_args()
    parquet_path = Path(args.parquet).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    rows = _read_parquet(parquet_path)
    if args.shuffle:
        import random

        rng = random.Random(args.seed)
        rows = list(rows)
        rng.shuffle(rows)

    start = max(0, args.start)
    end = min(len(rows), start + max(0, args.num))
    selected = rows[start:end]
    if not selected:
        raise ValueError(
            f"No rows selected from {parquet_path} with start={args.start} num={args.num}"
        )

    svg = _build_svg(
        selected,
        timeout=args.fetch_timeout,
        source_path=parquet_path,
        start=start,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")
    print(f"Wrote {len(selected)} sample(s) to {output_path}")


if __name__ == "__main__":
    main()
