"""Reader API wrapper that upgrades HTML extraction with ReaderLM-v2.

The service keeps the simple ``r.jina.ai``-style URL shape:

    GET /https://example.com

It first asks a self-hosted Jina Reader endpoint for HTML, then sends the
cleaned HTML to an OpenAI-compatible ReaderLM-v2 endpoint and returns Markdown.
"""

from __future__ import annotations

import asyncio
from html import escape
from html.parser import HTMLParser
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse


RAW_READER_URL = os.environ.get("RAW_READER_URL", "http://127.0.0.1:8002")
READERLM_API_BASE = os.environ.get("READERLM_API_BASE", "http://127.0.0.1:8005/v1")
READERLM_API_BASES_ENV = os.environ.get("READERLM_API_BASES", "")
RAW_MARKDOWN_READER_URL = os.environ.get("RAW_MARKDOWN_READER_URL", "http://127.0.0.1:8003")
READERLM_MODEL_NAME = os.environ.get("READERLM_MODEL_NAME", "jinaai/ReaderLM-v2")
READERLM_API_KEY = os.environ.get("READERLM_API_KEY", "")
READERLM_MAX_HTML_CHARS = int(os.environ.get("READERLM_MAX_HTML_CHARS", "120000"))
READER_TIMEOUT = float(os.environ.get("ENHANCED_READER_TIMEOUT", "180"))
READERLM_MAX_TOKENS = int(os.environ.get("READERLM_MAX_TOKENS", "8192"))
DEBUG_READERLM_URL_LEAK = os.environ.get("ENHANCED_READER_DEBUG_URL_LEAK", "1") != "0"
DEBUG_READERLM_URL_LEAK_DIR = Path(os.environ.get("ENHANCED_READER_DEBUG_URL_LEAK_DIR", "/tmp/enhanced_reader_url_leaks"))
TRUNCATION_MARKER = "\n<!-- enhanced_reader_truncated -->"
ENHANCED_READER_FETCH_STRATEGY = os.environ.get("ENHANCED_READER_FETCH_STRATEGY", "markdown_first").strip().lower()
ENHANCED_READER_CACHE_ENABLED = os.environ.get("ENHANCED_READER_CACHE_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
ENHANCED_READER_CACHE_DIR = Path(os.environ.get("ENHANCED_READER_CACHE_DIR", "/mnt/hdfs/byte_ai_sales/user/user/yinzhihan/tmp"))
ENHANCED_READER_CACHE_TTL_S = float(os.environ.get("ENHANCED_READER_CACHE_TTL_S", str(7 * 24 * 3600)))
ENHANCED_READER_CACHE_NEGATIVE_TTL_S = float(os.environ.get("ENHANCED_READER_CACHE_NEGATIVE_TTL_S", "600"))
ENHANCED_READER_MIN_USABLE_CHARS = int(os.environ.get("ENHANCED_READER_MIN_USABLE_CHARS", "500"))
RAW_MARKDOWN_CACHE_ENABLED = os.environ.get("RAW_MARKDOWN_CACHE_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
RAW_MARKDOWN_CACHE_DIR = Path(os.environ.get("RAW_MARKDOWN_CACHE_DIR", str(ENHANCED_READER_CACHE_DIR / "raw_markdown")))
RAW_MARKDOWN_CACHE_TTL_S = float(os.environ.get("RAW_MARKDOWN_CACHE_TTL_S", str(ENHANCED_READER_CACHE_TTL_S)))
RAW_MARKDOWN_CACHE_NEGATIVE_TTL_S = float(os.environ.get("RAW_MARKDOWN_CACHE_NEGATIVE_TTL_S", str(ENHANCED_READER_CACHE_NEGATIVE_TTL_S)))
ENHANCED_READER_MODE = os.environ.get("ENHANCED_READER_MODE", "full").strip().lower()


app = FastAPI(title="Enhanced Reader API")


def _parse_readerlm_api_bases() -> list[str]:
    configured = [item.strip() for item in READERLM_API_BASES_ENV.split(",") if item.strip()]
    if not configured:
        configured = [READERLM_API_BASE]

    raw_markdown_origin = urlparse(RAW_MARKDOWN_READER_URL).netloc.casefold()
    values: list[str] = []
    rejected: list[str] = []
    for value in configured:
        base = value.rstrip("/")
        if base.endswith("/chat/completions"):
            base = base.removesuffix("/chat/completions")
        parsed = urlparse(base)
        if not parsed.scheme or not parsed.netloc or parsed.netloc.casefold() == raw_markdown_origin:
            rejected.append(value)
            continue
        if base not in values:
            values.append(base)

    if not values:
        raise ValueError(
            "No valid ReaderLM API base is configured. READERLM_API_BASE(S) must "
            "contain OpenAI-compatible vLLM bases such as http://127.0.0.1:8005/v1, "
            f"not the raw markdown reader {RAW_MARKDOWN_READER_URL!r}. "
            f"Rejected values: {rejected!r}"
        )
    if rejected:
        print(
            f"[enhanced_reader] ignoring invalid ReaderLM API bases: {rejected!r}",
            file=sys.stderr,
            flush=True,
        )
    return values


READERLM_API_BASES = _parse_readerlm_api_bases()
_READERLM_API_BASE_CYCLE = itertools.cycle(range(len(READERLM_API_BASES)))
_READERLM_API_BASE_LOCK = asyncio.Lock()


ANTI_BOT_PATTERNS = (
    "performing security verification",
    "security service to protect against malicious bots",
    "verify you are not a bot",
    "verification successful. waiting",
    "checking your browser",
    "just a moment",
    "enable javascript",
    "access denied",
    "captcha",
    "cloudflare",
    "bot detection",
    "are you a human",
)


SCRIPT_PATTERN = r"<[ ]*script.*?/[\s]*script[ ]*>"
STYLE_PATTERN = r"<[ ]*style.*?/[\s]*style[ ]*>"
META_PATTERN = r"<[ ]*meta.*?>"
COMMENT_PATTERN = r"<[ ]*!--.*?--[ ]*>"
LINK_PATTERN = r"<[ ]*link.*?>"
BASE64_IMG_PATTERN = r'<img[^>]+src="data:image/[^;]+;base64,[^"]+"[^>]*>'
IMG_PATTERN = r"<img\b[^>]*>"
ALT_ATTR_PATTERN = r"""\salt=("[^"]*"|'[^']*'|[^\s>]+)"""
SVG_PATTERN = r"(<svg[^>]*>)(.*?)(</svg>)"
A_OPEN_PATTERN = r"<a\b[^>]*>"
A_CLOSE_PATTERN = r"</a\s*>"
URL_ATTR_PATTERN = r"""\s(?:href|src|srcset|data-src|data-original|poster|action)=("[^"]*"|'[^']*'|[^\s>]+)"""
BARE_URL_PATTERN = r"https?://[^\s<>'\"]+"
WIKI_MAIN_CLASSES = ("mw-parser-output",)
WIKI_MAIN_IDS = ("mw-content-text",)
WIKI_DROP_CLASS_TOKENS = (
    "ambox",
    "authority-control",
    "catlinks",
    "metadata",
    "mwe-math-fallback-image-inline",
    "mw-editsection",
    "navbox",
    "navbox-styles",
    "noprint",
    "printfooter",
    "reference",
    "references",
    "refbegin",
    "refend",
    "reflist",
    "mw-references-wrap",
    "sistersitebox",
    "vertical-navbox",
)
WIKI_DROP_IDS = {
    "References",
    "Notes",
    "Footnotes",
    "Citations",
    "Notes_and_references",
    "References_and_notes",
    "External_links",
    "Further_reading",
    "Bibliography",
    "Sources",
    "See_also",
    "Authority_control",
}
WIKI_VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "source",
    "track",
    "wbr",
}


class StageError(Exception):
    """Wrap an internal stage failure with an explicit stage label."""

    def __init__(self, stage: str, cause: Exception) -> None:
        super().__init__(f"{stage} failed: {cause}")
        self.stage = stage
        self.cause = cause


def normalize_url(target_url: str) -> str:
    if target_url.startswith(("http://", "https://")):
        return target_url
    return "https://" + target_url


def _now() -> float:
    return time.time()


def _cache_key(url: str, *, wants_json: bool) -> str:
    payload = {
        "url": url,
        "wants_json": bool(wants_json),
        "strategy": ENHANCED_READER_FETCH_STRATEGY,
        "readerlm_model": READERLM_MODEL_NAME,
        "readerlm_max_html_chars": READERLM_MAX_HTML_CHARS,
        "readerlm_max_tokens": READERLM_MAX_TOKENS,
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _raw_markdown_cache_key(url: str) -> str:
    payload = {"url": url}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _cache_path(url: str, *, wants_json: bool) -> Path:
    return ENHANCED_READER_CACHE_DIR / f"{_cache_key(url, wants_json=wants_json)}.json"


def _raw_markdown_cache_path(url: str) -> Path:
    return RAW_MARKDOWN_CACHE_DIR / f"{_raw_markdown_cache_key(url)}.json"


def _delete_cache_path(path: Path, *, label: str) -> None:
    """Best-effort removal for stale or manually refreshed cache entries."""
    try:
        path.unlink(missing_ok=True)
    except Exception as exc:  # pragma: no cover - best-effort cache cleanup
        print(f"[enhanced_reader][{label}_cache_delete_error] path={path} error={exc!r}", file=sys.stderr, flush=True)


def _is_cacheable_success(payload: dict[str, Any]) -> bool:
    """Only successful, structured responses may be replayed from cache."""
    status = payload.get("status", payload.get("code"))
    try:
        is_success = 200 <= int(status) < 300
    except (TypeError, ValueError):
        is_success = False
    return is_success and isinstance(payload.get("data"), dict)


def _read_cache(url: str, *, wants_json: bool) -> dict[str, Any] | None:
    if not ENHANCED_READER_CACHE_ENABLED:
        return None
    path = _cache_path(url, wants_json=wants_json)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    created_at = float(record.get("created_at") or 0.0)
    ttl = float(record.get("ttl_s") or ENHANCED_READER_CACHE_TTL_S)
    if ttl >= 0 and _now() - created_at > ttl:
        return None
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return None
    payload = dict(payload)
    if not _is_cacheable_success(payload):
        # Old versions cached 502 payloads. Remove them so a recovered
        # upstream is retried immediately instead of being replayed as empty.
        _delete_cache_path(path, label="enhanced_reader")
        return None
    data = payload.get("data")
    if isinstance(data, dict):
        data = dict(data)
        debug_timing = dict(data.get("debug_timing") or {})
        debug_timing["cache_hit"] = True
        debug_timing["cache_path"] = str(path)
        data["debug_timing"] = debug_timing
        payload["data"] = data
    payload["debug_timing"] = dict(payload.get("debug_timing") or {})
    payload["debug_timing"].update({"cache_hit": True, "cache_path": str(path)})
    return payload


def _write_cache(url: str, *, wants_json: bool, payload: dict[str, Any], negative: bool = False) -> None:
    if not ENHANCED_READER_CACHE_ENABLED:
        return
    try:
        ENHANCED_READER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _cache_path(url, wants_json=wants_json)
        record = {
            "created_at": _now(),
            "ttl_s": ENHANCED_READER_CACHE_NEGATIVE_TTL_S if negative else ENHANCED_READER_CACHE_TTL_S,
            "url": url,
            "wants_json": wants_json,
            "payload": payload,
        }
        tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        tmp_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_path, path)
    except Exception as exc:  # pragma: no cover - best-effort cache
        print(f"[enhanced_reader][cache_write_error] url={url} error={exc!r}", file=sys.stderr, flush=True)


def _read_raw_markdown_cache(url: str) -> dict[str, Any] | None:
    if not RAW_MARKDOWN_CACHE_ENABLED:
        return None
    path = _raw_markdown_cache_path(url)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    created_at = float(record.get("created_at") or 0.0)
    ttl = float(record.get("ttl_s") or RAW_MARKDOWN_CACHE_TTL_S)
    if ttl >= 0 and _now() - created_at > ttl:
        return None
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return None
    payload = dict(payload)
    if not _is_cacheable_success(payload):
        _delete_cache_path(path, label="raw_markdown")
        return None
    return payload


def _write_raw_markdown_cache(url: str, *, payload: dict[str, Any], negative: bool = False) -> None:
    if not RAW_MARKDOWN_CACHE_ENABLED:
        return
    try:
        RAW_MARKDOWN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _raw_markdown_cache_path(url)
        record = {
            "created_at": _now(),
            "ttl_s": RAW_MARKDOWN_CACHE_NEGATIVE_TTL_S if negative else RAW_MARKDOWN_CACHE_TTL_S,
            "url": url,
            "payload": payload,
        }
        tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        tmp_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_path, path)
    except Exception as exc:  # pragma: no cover - best-effort cache
        print(f"[enhanced_reader][raw_markdown_cache_write_error] url={url} error={exc!r}", file=sys.stderr, flush=True)


def looks_like_antibot(text: str | None) -> bool:
    normalized = re.sub(r"\s+", " ", str(text or "").strip().lower())
    if not normalized:
        return False
    return any(pattern in normalized for pattern in ANTI_BOT_PATTERNS)


# ``markdown_clean`` deliberately uses a small, deterministic rule set instead
# of another LLM pass.  The rules are intentionally conservative: link labels
# and table cell order are kept, while URLs, citation plumbing, and reference
# sections are removed from the model-facing content.
_MARKDOWN_REFERENCE_HEADINGS = frozenset(
    {
        "references",
        "reference",
        "notes",
        "footnotes",
        "citations",
        "bibliography",
        "works cited",
        "sources",
        "external links",
        "further reading",
        "see also",
    }
)
_MARKDOWN_CITATION_HREF_RE = re.compile(
    r"(?:cite[_-](?:note|ref)|mw-reference|#cite|#ref|footnote)",
    flags=re.IGNORECASE,
)
_MARKDOWN_CITATION_LABEL_RE = re.compile(
    r"^(?:\[?\s*(?:\d{1,4}|[a-z]|[α-ωΑ-Ω])\s*\]?|↑|up)$",
    flags=re.IGNORECASE,
)
_MARKDOWN_GENERIC_IMAGE_ALT_RE = re.compile(
    r"^(?:image|img|photo|picture|figure|thumbnail)(?:\s*[-_:]?\s*\d+(?:\s*:\s*.+)?)?$",
    flags=re.IGNORECASE,
)
_MARKDOWN_LINK_DEFINITION_RE = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*\S+", flags=re.IGNORECASE)
_MARKDOWN_CITATION_MARKER_RE = re.compile(
    r"\[\s*(?:\d{1,4}|[a-z]|[α-ωΑ-Ω])\s*\]",
    flags=re.IGNORECASE,
)


def _markdown_heading_info(line: str) -> tuple[int, str] | None:
    """Return ``(level, normalized_title)`` for a Markdown heading.

    Plain ``References``-style lines are accepted as level ``0`` because some
    Jina Markdown responses omit the heading markers for section titles.
    """

    match = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$", line)
    if match:
        title = match.group(2)
        level = len(match.group(1))
    else:
        title = line.strip()
        level = 0
        if len(title) > 80 or "|" in title or not title:
            return None
    normalized = re.sub(r"[*_`]+", "", title)
    normalized = re.sub(r"\s+", " ", normalized).strip(" :").casefold()
    if normalized not in _MARKDOWN_REFERENCE_HEADINGS:
        return None
    return level, normalized


def _parse_markdown_link_at(text: str, start: int) -> tuple[int, str, str, bool] | None:
    """Parse one Markdown link/image beginning at ``start``.

    A small scanner is used instead of a single regex so URLs containing
    parentheses (common in Wikipedia links) are handled correctly.
    """

    is_image = text.startswith("!", start)
    bracket_start = start + 1 if is_image else start
    if bracket_start >= len(text) or text[bracket_start] != "[":
        return None

    bracket_depth = 0
    close_bracket: int | None = None
    escaped = False
    for index in range(bracket_start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth -= 1
            if bracket_depth == 0:
                close_bracket = index
                break
    if close_bracket is None:
        return None

    open_paren = close_bracket + 1
    while open_paren < len(text) and text[open_paren].isspace():
        open_paren += 1
    if open_paren >= len(text) or text[open_paren] != "(":
        return None

    paren_depth = 0
    escaped = False
    close_paren: int | None = None
    for index in range(open_paren, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth -= 1
            if paren_depth == 0:
                close_paren = index
                break
    if close_paren is None:
        return None

    label = text[bracket_start + 1 : close_bracket]
    destination = text[open_paren + 1 : close_paren].strip()
    return close_paren + 1, label, destination, is_image


def _is_markdown_citation(label: str, destination: str) -> bool:
    label_normalized = re.sub(r"[*_`]+", "", label).strip()
    label_normalized = re.sub(r"\s+", " ", label_normalized)
    return bool(
        _MARKDOWN_CITATION_HREF_RE.search(destination)
        or label_normalized.casefold().startswith("jump up to")
        or _MARKDOWN_CITATION_LABEL_RE.fullmatch(label_normalized)
    )


def _clean_markdown_inline(text: str, stats: dict[str, Any]) -> str:
    """Remove URL-bearing Markdown syntax while preserving visible labels."""

    def replace_malformed_image(label: str) -> str:
        """Keep the ``![alt]`` residue and remove its unmatched ``(``."""

        stats["images_seen"] += 1
        stats["malformed_images_seen"] += 1
        normalized_label = label.strip()
        stats["images_preserved"] += 1
        stats["malformed_images_fixed"] += 1
        # The source is already truncated, so preserving the visible marker
        # is more useful than deleting it. Only the unmatched opening
        # parenthesis (and the truncated destination after it) is discarded.
        return f"![{normalized_label}]"

    # The raw response can contain a truncated image token such as
    # ``![Image 10](``. It has no parseable destination, so fix it before
    # the regular Markdown scanner; otherwise the dangling opener leaks into
    # the model-facing content.
    text = re.sub(
        r"!\[([^\]\n]*)\]\(\s*$",
        lambda match: replace_malformed_image(match.group(1)),
        text,
    )

    # MediaWiki/Jina occasionally emits ``[[label]](url)`` instead of the
    # normal ``[label](url)`` form.  Handle this before the general scanner.
    double_link_pattern = re.compile(r"(!?)\[\[([^\]]+)\]\]\((.*?)\)")

    def replace_double_link(match: re.Match[str]) -> str:
        is_image = bool(match.group(1))
        label = match.group(2)
        destination = match.group(3)
        if is_image:
            stats["images_seen"] += 1
            if not label.strip() or _MARKDOWN_GENERIC_IMAGE_ALT_RE.fullmatch(label.strip()):
                stats["images_removed"] += 1
                return ""
            stats["images_preserved"] += 1
            return f"[Image: {label.strip()}]"
        stats["links_seen"] += 1
        if _is_markdown_citation(label, destination):
            stats["citation_links_removed"] += 1
            return ""
        stats["link_urls_removed"] += 1
        return label

    text = double_link_pattern.sub(replace_double_link, text)

    output: list[str] = []
    index = 0
    while index < len(text):
        start = index
        if text[index] == "!" and index + 1 < len(text) and text[index + 1] == "[":
            start = index
        elif text[index] != "[":
            output.append(text[index])
            index += 1
            continue

        parsed = _parse_markdown_link_at(text, start)
        if parsed is None:
            # A malformed image may have additional truncated text after the
            # opener (for example ``![Image 18]( 19: @user](``).  There is no
            # reliable URL boundary in this case; consume the remainder of the
            # line after preserving the visible ``![alt]`` marker.
            malformed_image = re.match(r"!\[([^\]\n]*)\]\(\s*", text[start:]) if text.startswith("![", start) else None
            if malformed_image and ")" not in text[start + malformed_image.end() :]:
                output.append(replace_malformed_image(malformed_image.group(1)))
                index = len(text)
                continue
            output.append(text[index])
            index += 1
            continue
        end, label, destination, is_image = parsed
        if is_image:
            stats["images_seen"] += 1
            if not label.strip() or _MARKDOWN_GENERIC_IMAGE_ALT_RE.fullmatch(label.strip()):
                stats["images_removed"] += 1
                replacement = ""
            else:
                stats["images_preserved"] += 1
                replacement = f"[Image: {label.strip()}]"
        else:
            stats["links_seen"] += 1
            if _is_markdown_citation(label, destination):
                stats["citation_links_removed"] += 1
                replacement = ""
            else:
                stats["link_urls_removed"] += 1
                replacement = label
        output.append(replacement)
        index = end

    text = "".join(output)

    def remove_bare_url(match: re.Match[str]) -> str:
        stats["bare_urls_removed"] += 1
        return ""

    text = re.sub(BARE_URL_PATTERN, remove_bare_url, text)
    text = re.sub(r"<\s*>", "", text)
    text, marker_count = _MARKDOWN_CITATION_MARKER_RE.subn("", text)
    stats["citation_markers_removed"] += marker_count
    # Citation markers are often glued between a comma and the sentence
    # terminator (``fact,[[12]].``).  Removing the marker should not leave
    # malformed ``,.``/``;,`` punctuation behind.
    text = re.sub(r"([,;:])\s*([.!?])", r"\2", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def clean_raw_markdown(raw_markdown: str | None) -> tuple[str, dict[str, Any]]:
    """Clean raw Jina Markdown with deterministic, model-free rules.

    The original Markdown is never mutated in-place by callers and remains
    available as ``raw_markdown`` in the response.  Reference sections are
    removed, ordinary link URLs are replaced by their visible labels, complete
    generic images are dropped, malformed image residues retain their visible
    ``![alt]`` marker, and tables retain their original row/cell ordering.
    """

    raw_text = str(raw_markdown or "").replace("\r\n", "\n").replace("\r", "\n")
    stats: dict[str, Any] = {
        "cleaner": "rule_v1",
        "cleaning_applied": True,
        "original_chars": len(raw_text),
        "cleaned_chars": 0,
        "removed_chars": 0,
        "reference_sections_removed": 0,
        "reference_lines_removed": 0,
        "link_definitions_removed": 0,
        "links_seen": 0,
        "link_urls_removed": 0,
        "citation_links_removed": 0,
        "citation_markers_removed": 0,
        "bare_urls_removed": 0,
        "images_seen": 0,
        "images_removed": 0,
        "images_preserved": 0,
        "malformed_images_seen": 0,
        "malformed_images_fixed": 0,
        "malformed_images_removed": 0,
        "table_count": 0,
        "table_rows_checked": 0,
        "table_column_mismatch_count": 0,
        "table_warnings": [],
    }

    cleaned_lines: list[str] = []
    reference_level: int | None = None
    in_fence = False
    fence_marker = ""
    table_expected_columns: int | None = None
    table_active = False

    for line_number, line in enumerate(raw_text.split("\n"), start=1):
        fence_match = re.match(r"^\s*(`{3,}|~{3,})", line)
        if fence_match:
            marker = fence_match.group(1)[0]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
            cleaned_lines.append(line.rstrip())
            table_active = False
            table_expected_columns = None
            continue
        if in_fence:
            cleaned_lines.append(line.rstrip())
            continue

        heading_info = _markdown_heading_info(line)
        heading_match = re.match(r"^\s{0,3}(#{1,6})\s+", line)
        heading_level = len(heading_match.group(1)) if heading_match else None
        if reference_level is not None:
            # A Markdown heading ends the removed section.  Plain headings
            # are intentionally not treated as terminators because reference
            # lists themselves often contain short, title-like lines.
            if heading_level is not None and (reference_level == 0 or heading_level <= reference_level):
                reference_level = None
            else:
                stats["reference_lines_removed"] += 1
                continue

        if heading_info is not None:
            reference_level = heading_info[0]
            stats["reference_sections_removed"] += 1
            continue

        if _MARKDOWN_LINK_DEFINITION_RE.match(line):
            stats["link_definitions_removed"] += 1
            continue

        stripped = line.strip()
        is_table_line = stripped.startswith("|") or ("|" in stripped and stripped.count("|") >= 2)
        if is_table_line:
            cells = stripped.strip("|").split("|")
            cell_count = len(cells)
            if not table_active:
                stats["table_count"] += 1
                table_active = True
                table_expected_columns = cell_count
            stats["table_rows_checked"] += 1
            if table_expected_columns is not None and cell_count != table_expected_columns:
                stats["table_column_mismatch_count"] += 1
                stats["table_warnings"].append(
                    {
                        "line": line_number,
                        "expected_columns": table_expected_columns,
                        "actual_columns": cell_count,
                    }
                )
        else:
            table_active = False
            table_expected_columns = None

        cleaned_lines.append(_clean_markdown_inline(line.rstrip(), stats))

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    stats["cleaned_chars"] = len(cleaned)
    stats["removed_chars"] = max(0, len(raw_text) - len(cleaned))
    stats["empty_after_cleaning"] = not bool(cleaned)
    return cleaned, stats


def _usable_text(text: str | None) -> bool:
    value = str(text or "").strip()
    return len(value) >= ENHANCED_READER_MIN_USABLE_CHARS and not looks_like_antibot(value)


def _select_best_content(*, readerlm_markdown: str, raw_markdown: str, debug_timing: dict[str, Any]) -> tuple[str, str, str]:
    readerlm_text = str(readerlm_markdown or "").strip()
    raw_text = str(raw_markdown or "").strip()
    readerlm_antibot = looks_like_antibot(readerlm_text)
    raw_antibot = looks_like_antibot(raw_text)
    debug_timing["content_antibot"] = readerlm_antibot
    debug_timing["raw_markdown_antibot"] = raw_antibot
    debug_timing["content_chars"] = len(readerlm_text)
    debug_timing["raw_markdown_chars"] = len(raw_text)

    if readerlm_antibot and raw_text and not raw_antibot:
        return raw_text, "raw_markdown_fallback_after_antibot_content", "usable"
    if not readerlm_text and raw_text:
        return raw_text, "raw_markdown_fallback_empty_content", "usable" if not raw_antibot else "anti_bot"
    if len(readerlm_text) < ENHANCED_READER_MIN_USABLE_CHARS and _usable_text(raw_text):
        return raw_text, "raw_markdown_fallback_short_content", "usable"
    quality = "anti_bot" if readerlm_antibot else "usable" if readerlm_text else "empty"
    return readerlm_text, "readerlm_content", quality


def is_wikipedia_url(url: str | None) -> bool:
    if not url:
        return False
    return urlparse(url).netloc.endswith("wikipedia.org")


class WikipediaMainHTMLExtractor(HTMLParser):
    """Extract Wikipedia article content without page chrome or reference blocks."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.parts: list[str] = []
        self.capture_depth: int | None = None
        self.drop_depth = 0
        self.found_main = False
        self.stop_capture = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key.lower(): value or "" for key, value in attrs}
        tag_lower = tag.lower()

        if self.drop_depth:
            if tag_lower not in WIKI_VOID_TAGS:
                self.drop_depth += 1
            return

        if self.capture_depth is None:
            if self._is_main_container(attrs_dict):
                self.capture_depth = 1
                self.found_main = True
                self._append_starttag(tag_lower, attrs)
            return

        if self.stop_capture:
            return

        if attrs_dict.get("id") in WIKI_DROP_IDS:
            self.stop_capture = True
            return
        if self._should_stop_at_heading(tag_lower, attrs_dict):
            self.stop_capture = True
            return
        if self._should_drop_element(attrs_dict):
            if tag_lower not in WIKI_VOID_TAGS:
                self.drop_depth = 1
            return

        self._append_starttag(tag_lower, attrs)
        if tag_lower not in WIKI_VOID_TAGS:
            self.capture_depth += 1

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if self.drop_depth:
            if tag_lower not in WIKI_VOID_TAGS:
                self.drop_depth -= 1
            return
        if self.capture_depth is None or self.stop_capture:
            return
        self.parts.append(f"</{tag_lower}>")
        if tag_lower not in WIKI_VOID_TAGS:
            self.capture_depth -= 1
            if self.capture_depth <= 0:
                self.capture_depth = None

    def handle_data(self, data: str) -> None:
        if self.capture_depth is not None and not self.drop_depth and not self.stop_capture:
            self.parts.append(data)

    def handle_entityref(self, name: str) -> None:
        if self.capture_depth is not None and not self.drop_depth and not self.stop_capture:
            self.parts.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        if self.capture_depth is not None and not self.drop_depth and not self.stop_capture:
            self.parts.append(f"&#{name};")

    def result(self) -> str:
        return "".join(self.parts).strip()

    @staticmethod
    def _class_tokens(attrs_dict: dict[str, str]) -> set[str]:
        return set(re.split(r"\s+", attrs_dict.get("class", "").strip())) if attrs_dict.get("class") else set()

    def _is_main_container(self, attrs_dict: dict[str, str]) -> bool:
        if attrs_dict.get("id") in WIKI_MAIN_IDS:
            return True
        classes = self._class_tokens(attrs_dict)
        return any(class_name in classes for class_name in WIKI_MAIN_CLASSES)

    def _should_drop_element(self, attrs_dict: dict[str, str]) -> bool:
        if attrs_dict.get("id") in WIKI_DROP_IDS:
            return True
        classes = self._class_tokens(attrs_dict)
        return any(token in classes for token in WIKI_DROP_CLASS_TOKENS)

    @staticmethod
    def _should_stop_at_heading(tag: str, attrs_dict: dict[str, str]) -> bool:
        if tag not in {"h2", "h3"}:
            return False
        heading_id = attrs_dict.get("id", "")
        return heading_id in WIKI_DROP_IDS or heading_id.replace(" ", "_") in WIKI_DROP_IDS

    @staticmethod
    def _append_attrs(attrs: list[tuple[str, str | None]]) -> str:
        safe_attrs = []
        for key, value in attrs:
            if value is None:
                safe_attrs.append(escape(key, quote=True))
            else:
                safe_attrs.append(f'{escape(key, quote=True)}="{escape(value, quote=True)}"')
        return (" " + " ".join(safe_attrs)) if safe_attrs else ""

    def _append_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in WIKI_VOID_TAGS:
            self.parts.append(f"<{tag}{self._append_attrs(attrs)}>")
        else:
            self.parts.append(f"<{tag}{self._append_attrs(attrs)}>")


def extract_wikipedia_main_html(html: str) -> tuple[str, bool]:
    extractor = WikipediaMainHTMLExtractor()
    try:
        extractor.feed(html)
        extracted = extractor.result()
    except Exception:
        return html, False
    if not extractor.found_main or len(extracted) < 500:
        return html, False
    return extracted, True


def replace_svg(html: str, new_content: str = "this is a placeholder") -> str:
    return re.sub(
        SVG_PATTERN,
        lambda match: f"{match.group(1)}{new_content}{match.group(3)}",
        html,
        flags=re.DOTALL,
    )


def replace_base64_images(html: str, new_image_src: str = "#") -> str:
    return re.sub(BASE64_IMG_PATTERN, f'<img src="{new_image_src}"/>', html)


def replace_images_with_alt_text(html: str) -> str:
    """Remove image tags, keeping alt text when available."""

    def replace(match: re.Match[str]) -> str:
        tag = match.group(0)
        alt_match = re.search(ALT_ATTR_PATTERN, tag, flags=re.IGNORECASE | re.DOTALL)
        if not alt_match:
            return ""
        alt = alt_match.group(1).strip("\"'")
        return f" {alt} " if alt else ""

    return re.sub(IMG_PATTERN, replace, html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)


def strip_anchor_links(html: str) -> str:
    """Remove hyperlink tags while keeping their visible anchor text."""

    html = re.sub(A_OPEN_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    return re.sub(A_CLOSE_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE)


def strip_url_noise(html: str) -> str:
    """Remove URL-bearing attributes and literal URLs before ReaderLM sees text."""

    html = re.sub(URL_ATTR_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    return re.sub(BARE_URL_PATTERN, "", html, flags=re.IGNORECASE)


def clean_html(
    html: str,
    *,
    source_url: str | None = None,
    debug_timing: dict[str, Any] | None = None,
    clean_svg: bool = True,
    clean_base64: bool = True,
) -> str:
    """Pre-clean HTML following the ReaderLM-v2 model-card guidance."""

    if is_wikipedia_url(source_url):
        before_chars = len(html)
        html, extracted = extract_wikipedia_main_html(html)
        if debug_timing is not None:
            debug_timing["wiki_main_extracted"] = extracted
            debug_timing["wiki_html_chars_before_main_extract"] = before_chars
            debug_timing["wiki_html_chars_after_main_extract"] = len(html)
    elif debug_timing is not None:
        debug_timing["wiki_main_extracted"] = False

    html = re.sub(SCRIPT_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    html = re.sub(STYLE_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    html = re.sub(META_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    html = re.sub(COMMENT_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    html = re.sub(LINK_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    html = strip_anchor_links(html)
    if clean_base64:
        html = replace_base64_images(html)
    html = replace_images_with_alt_text(html)
    html = strip_url_noise(html)
    if clean_svg:
        html = replace_svg(html)
    return html


def truncate_safely(text: str, max_chars: int, *, marker: str = TRUNCATION_MARKER) -> str:
    """Truncate near structural boundaries instead of cutting raw HTML mid-token."""

    if max_chars <= 0 or len(text) <= max_chars:
        return text

    preferred_breaks = (
        "</section>",
        "</article>",
        "</p>",
        "</div>",
        "</li>",
        "\n\n",
        "\n",
        ". ",
        " ",
    )
    min_cut = int(max_chars * 0.65)
    cut_at = -1
    for needle in preferred_breaks:
        pos = text.rfind(needle, 0, max_chars)
        if pos >= min_cut:
            cut_at = pos + len(needle)
            break
    if cut_at < min_cut:
        cut_at = max_chars
    return text[:cut_at].rstrip() + marker


def create_prompt(
    html: str,
    instruction: str = "Extract the main content from the given HTML and convert it to Markdown format.",
) -> str:
    return f"{instruction}\n```html\n{html}\n```"


def create_markdown_cleanup_prompt(
    markdown: str,
    instruction: str = (
        "Clean the given raw Markdown reader output. Preserve the main content, "
        "tables, image captions, alt text, titles, dates, names, and source-relevant details. "
        "Remove navigation, repeated menus, boilerplate, login prompts, unrelated recommendations, "
        "and tracking text. Do not add facts that are not present in the input. Return Markdown only."
    ),
) -> str:
    return f"{instruction}\n```markdown\n{markdown}\n```"


def strip_outer_markdown_fence(text: str) -> str:
    match = re.fullmatch(r"\s*```(?:markdown|md)?\s*\n(.*?)\n```\s*", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


def debug_url_leak_after_readerlm(
    *,
    source_url: str | None,
    readerlm_input: str,
    markdown: str,
    debug_timing: dict[str, Any] | None = None,
) -> None:
    if not DEBUG_READERLM_URL_LEAK or "(http" not in markdown:
        return

    DEBUG_READERLM_URL_LEAK_DIR.mkdir(parents=True, exist_ok=True)
    leak_key = hashlib.sha1(f"{source_url or ''}\n{time.time()}".encode("utf-8")).hexdigest()[:12]
    input_path = DEBUG_READERLM_URL_LEAK_DIR / f"{leak_key}.before_readerlm.html"
    output_path = DEBUG_READERLM_URL_LEAK_DIR / f"{leak_key}.after_readerlm.md"
    input_path.write_text(readerlm_input, encoding="utf-8")
    output_path.write_text(markdown, encoding="utf-8")

    message = (
        "[enhanced_reader][url_leak] ReaderLM output contains '(http'. "
        f"url={source_url} input={input_path} output={output_path}"
    )
    print(message, file=sys.stderr, flush=True)
    print("[enhanced_reader][url_leak][before_preview]", readerlm_input[:2000], file=sys.stderr, flush=True)
    print("[enhanced_reader][url_leak][after_preview]", markdown[:2000], file=sys.stderr, flush=True)

    if debug_timing is not None:
        debug_timing["readerlm_url_leak_detected"] = True
        debug_timing["readerlm_url_leak_input_path"] = str(input_path)
        debug_timing["readerlm_url_leak_output_path"] = str(output_path)


async def fetch_html(client: httpx.AsyncClient, url: str) -> str:
    response = await client.get(
        f"{RAW_READER_URL.rstrip('/')}/{url}",
        headers={
            "Accept": "text/plain",
            "x-respond-with": "html",
            "x-engine": "browser",
        },
        timeout=READER_TIMEOUT,
    )
    response.raise_for_status()
    return response.text


async def fetch_markdown(client: httpx.AsyncClient, url: str) -> str:
    response = await client.get(
        f"{RAW_READER_URL.rstrip('/')}/{url}",
        headers={"Accept": "text/plain"},
        timeout=READER_TIMEOUT,
    )
    response.raise_for_status()
    return response.text


async def fetch_raw_markdown_via_cache_layer(client: httpx.AsyncClient, url: str) -> str:
    response = await client.get(
        f"{RAW_MARKDOWN_READER_URL.rstrip('/')}/{url}",
        headers={"Accept": "text/plain"},
        timeout=READER_TIMEOUT,
    )
    response.raise_for_status()
    return response.text


async def convert_html_to_markdown(
    client: httpx.AsyncClient,
    html: str,
    *,
    source_url: str | None = None,
    debug_timing: dict[str, Any] | None = None,
) -> str:
    started = time.perf_counter()
    cleaned_html = clean_html(html, source_url=source_url, debug_timing=debug_timing)
    if debug_timing is not None:
        debug_timing["clean_html_s"] = time.perf_counter() - started
        debug_timing["raw_html_chars"] = len(html)
        debug_timing["cleaned_html_chars"] = len(cleaned_html)
        debug_timing["html_link_tags_removed"] = len(re.findall(A_OPEN_PATTERN, html, flags=re.IGNORECASE))

    started = time.perf_counter()
    readerlm_input = truncate_safely(cleaned_html, READERLM_MAX_HTML_CHARS)
    if debug_timing is not None:
        debug_timing["truncate_html_s"] = time.perf_counter() - started
        debug_timing["readerlm_input_chars"] = len(readerlm_input)
        debug_timing["readerlm_input_truncated"] = readerlm_input != cleaned_html

    headers = {"Content-Type": "application/json"}
    if READERLM_API_KEY:
        headers["Authorization"] = f"Bearer {READERLM_API_KEY}"

    markdown = await _readerlm_completion(
        client=client,
        headers=headers,
        prompt=create_prompt(readerlm_input),
        debug_timing=debug_timing,
    )
    debug_url_leak_after_readerlm(
        source_url=source_url,
        readerlm_input=readerlm_input,
        markdown=markdown,
        debug_timing=debug_timing,
    )
    return markdown


async def convert_raw_markdown_to_markdown(
    client: httpx.AsyncClient,
    raw_markdown: str,
    *,
    source_url: str | None = None,
    debug_timing: dict[str, Any] | None = None,
) -> str:
    """Clean raw reader Markdown with ReaderLM without making a second URL fetch."""

    started = time.perf_counter()
    readerlm_input = truncate_safely(raw_markdown, READERLM_MAX_HTML_CHARS)
    if debug_timing is not None:
        debug_timing["raw_markdown_readerlm_input_chars"] = len(readerlm_input)
        debug_timing["raw_markdown_readerlm_input_truncated"] = readerlm_input != raw_markdown
        debug_timing["truncate_raw_markdown_s"] = time.perf_counter() - started

    headers = {"Content-Type": "application/json"}
    if READERLM_API_KEY:
        headers["Authorization"] = f"Bearer {READERLM_API_KEY}"

    markdown = await _readerlm_completion(
        client=client,
        headers=headers,
        prompt=create_markdown_cleanup_prompt(readerlm_input),
        debug_timing=debug_timing,
    )
    debug_url_leak_after_readerlm(
        source_url=source_url,
        readerlm_input=readerlm_input,
        markdown=markdown,
        debug_timing=debug_timing,
    )
    return markdown


async def _readerlm_api_bases_for_request() -> list[str]:
    if len(READERLM_API_BASES) == 1:
        return list(READERLM_API_BASES)
    async with _READERLM_API_BASE_LOCK:
        index = next(_READERLM_API_BASE_CYCLE)
    return READERLM_API_BASES[index:] + READERLM_API_BASES[:index]


async def _readerlm_completion(
    *,
    client: httpx.AsyncClient,
    headers: dict[str, str],
    prompt: str,
    debug_timing: dict[str, Any] | None,
) -> str:
    attempted: list[str] = []
    failures: list[dict[str, Any]] = []
    last_error: Exception | None = None
    for readerlm_api_base in await _readerlm_api_bases_for_request():
        attempted.append(readerlm_api_base)
        try:
            response = await client.post(
                f"{readerlm_api_base}/chat/completions",
                headers=headers,
                json={
                    "model": READERLM_MODEL_NAME,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": READERLM_MAX_TOKENS,
                    "extra_body": {"repetition_penalty": 1.08},
                },
                timeout=READER_TIMEOUT,
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()
            if debug_timing is not None:
                debug_timing["readerlm_api_base"] = readerlm_api_base
                debug_timing["readerlm_attempted_api_bases"] = attempted
                debug_timing["readerlm_failures"] = failures
            return strip_outer_markdown_fence(data["choices"][0]["message"]["content"])
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            last_error = exc
            failures.append({"api_base": readerlm_api_base, "error": f"{exc.__class__.__name__}: {exc}"})

    if debug_timing is not None:
        debug_timing["readerlm_attempted_api_bases"] = attempted
        debug_timing["readerlm_failures"] = failures
    raise RuntimeError(f"All ReaderLM replicas failed: {failures!r}") from last_error


async def timed_call(label: str, coro, timing: dict[str, Any]):
    started = time.perf_counter()
    try:
        return await coro
    finally:
        timing[f"{label}_s"] = time.perf_counter() - started


async def timed_stage_call(stage: str, coro, timing: dict[str, Any]):
    try:
        return await timed_call(stage, coro, timing)
    except Exception as exc:
        raise StageError(stage, exc) from exc


def _shorten(value: Any, limit: int = 800) -> str | None:
    if value is None:
        return None
    text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def record_failure_debug(
    timing: dict[str, Any],
    *,
    stage: str,
    exc: Exception,
) -> None:
    timing["failure_stage"] = stage
    timing["failure_exception_type"] = exc.__class__.__name__
    timing["failure_exception"] = _shorten(repr(exc), limit=1200)

    if isinstance(exc, httpx.HTTPStatusError):
        response = exc.response
        timing["failure_upstream_status_code"] = response.status_code
        timing["failure_upstream_reason"] = response.reason_phrase
        timing["failure_upstream_url"] = str(response.request.url)
        try:
            timing["failure_response_preview"] = _shorten(response.text, limit=1200)
        except Exception:
            pass
        return

    if isinstance(exc, httpx.TimeoutException):
        request = getattr(exc, "request", None)
        if request is not None:
            timing["failure_upstream_url"] = str(request.url)
        return

    request = getattr(exc, "request", None)
    if request is not None:
        timing["failure_upstream_url"] = str(request.url)


@app.get("/{target_url:path}")
async def read(target_url: str, request: Request):
    if ENHANCED_READER_MODE == "raw_only":
        return await read_raw_markdown(target_url, request)

    total_started = time.perf_counter()
    url = normalize_url(target_url)
    debug_requested = request.query_params.get("debug", "").strip().lower() in {"1", "true", "yes", "on"}
    # ``debug=1`` is convenient with curl: it forces the structured response
    # even when curl sends the default ``Accept: */*`` header.
    wants_json = debug_requested or "application/json" in request.headers.get("accept", "")
    debug_timing: dict[str, Any] = {}
    refresh_cache = request.query_params.get("refresh", "").strip().lower() in {"1", "true", "yes"}

    if refresh_cache:
        _delete_cache_path(_cache_path(url, wants_json=wants_json), label="enhanced_reader")
    cached_payload = None if refresh_cache else _read_cache(url, wants_json=wants_json)
    if cached_payload is not None:
        if wants_json:
            return cached_payload
        cached_data = cached_payload.get("data") or {}
        body = f"URL Source: {cached_data.get('url') or url}\n\nMarkdown Content:\n{cached_data.get('content') or ''}\n"
        return Response(
            body,
            media_type="text/plain; charset=utf-8",
            headers={"X-Enhanced-Reader-Cache": "hit"},
        )

    async with httpx.AsyncClient() as client:
        try:
            debug_timing["fetch_strategy"] = ENHANCED_READER_FETCH_STRATEGY
            if ENHANCED_READER_FETCH_STRATEGY == "parallel":
                fetch_started = time.perf_counter()
                markdown_response, html_response = await asyncio.gather(
                    timed_stage_call("fetch_markdown", fetch_markdown(client, url), debug_timing),
                    timed_stage_call("fetch_html", fetch_html(client, url), debug_timing),
                )
                fetch_done = time.perf_counter()
                debug_timing["fetch_markdown_html_parallel_s"] = fetch_done - fetch_started
                readerlm_started = time.perf_counter()
                readerlm_markdown = await timed_stage_call(
                    "readerlm",
                    convert_html_to_markdown(client, html_response, source_url=url, debug_timing=debug_timing),
                    debug_timing,
                )
                debug_timing["readerlm_s"] = time.perf_counter() - readerlm_started
                markdown, content_source, content_quality = _select_best_content(
                    readerlm_markdown=readerlm_markdown,
                    raw_markdown=markdown_response,
                    debug_timing=debug_timing,
                )
            elif ENHANCED_READER_FETCH_STRATEGY == "markdown_clean":
                markdown_response = await timed_stage_call(
                    "fetch_markdown",
                    fetch_raw_markdown_via_cache_layer(client, url),
                    debug_timing,
                )
                debug_timing["raw_markdown_chars"] = len(markdown_response or "")
                debug_timing["raw_markdown_antibot"] = looks_like_antibot(markdown_response)
                markdown, cleaning = clean_raw_markdown(markdown_response)
                debug_timing["markdown_cleaning"] = cleaning
                debug_timing["cleaning_original_chars"] = cleaning["original_chars"]
                debug_timing["cleaning_cleaned_chars"] = cleaning["cleaned_chars"]
                content_source = "rule_cleaned_markdown"
                if not markdown:
                    content_quality = "empty"
                elif looks_like_antibot(markdown):
                    content_quality = "anti_bot"
                elif len(markdown) >= ENHANCED_READER_MIN_USABLE_CHARS:
                    content_quality = "usable"
                else:
                    content_quality = "short"
            else:
                markdown_response = await timed_stage_call(
                    "fetch_markdown",
                    fetch_raw_markdown_via_cache_layer(client, url),
                    debug_timing,
                )
                debug_timing["raw_markdown_chars"] = len(markdown_response or "")
                debug_timing["raw_markdown_antibot"] = looks_like_antibot(markdown_response)
                readerlm_started = time.perf_counter()
                readerlm_markdown = await timed_stage_call(
                    "raw_markdown_readerlm",
                    convert_raw_markdown_to_markdown(
                        client,
                        markdown_response,
                        source_url=url,
                        debug_timing=debug_timing,
                    ),
                    debug_timing,
                )
                debug_timing["readerlm_s"] = time.perf_counter() - readerlm_started
                markdown, content_source, content_quality = _select_best_content(
                    readerlm_markdown=readerlm_markdown,
                    raw_markdown=markdown_response,
                    debug_timing=debug_timing,
                )
                if content_source == "readerlm_content":
                    content_source = "raw_markdown_readerlm_content"
            debug_timing["content_source"] = content_source
            debug_timing["content_quality"] = content_quality
        except StageError as exc:
            debug_timing["total_s"] = time.perf_counter() - total_started
            record_failure_debug(debug_timing, stage=exc.stage, exc=exc.cause)
            message = f"Enhanced Reader error for {url}: {exc.stage} failed: {exc.cause}"
            error_payload = {
                "data": None,
                "code": 502,
                "status": 502,
                "message": message,
                "debug_timing": debug_timing,
            }
            if wants_json:
                return JSONResponse(status_code=502, content=error_payload)
            return Response(
                message,
                status_code=502,
                media_type="text/plain",
                headers={"X-Debug-Timing-Total-S": f"{debug_timing['total_s']:.6f}"},
            )
        except Exception as exc:
            debug_timing["total_s"] = time.perf_counter() - total_started
            record_failure_debug(debug_timing, stage="unknown", exc=exc)
            message = f"Enhanced Reader error for {url}: {exc}"
            error_payload = {
                "data": None,
                "code": 502,
                "status": 502,
                "message": message,
                "debug_timing": debug_timing,
            }
            if wants_json:
                return JSONResponse(status_code=502, content=error_payload)
            return Response(
                message,
                status_code=502,
                media_type="text/plain",
                headers={"X-Debug-Timing-Total-S": f"{debug_timing['total_s']:.6f}"},
            )

    debug_timing["total_s"] = time.perf_counter() - total_started
    debug_timing["cache_hit"] = False
    cleaning = debug_timing.get("markdown_cleaning")
    payload = {
        "data": {
            "title": "",
            "url": url,
            "content": markdown,
            "raw_markdown": markdown_response,
            "content_source": content_source,
            "content_quality": content_quality,
            "debug_timing": debug_timing,
        },
        "code": 200,
        "status": 200,
        "debug_timing": debug_timing,
    }
    if isinstance(cleaning, dict):
        payload["data"]["cleaning"] = cleaning
    _write_cache(url, wants_json=wants_json, payload=payload, negative=content_quality in {"anti_bot", "empty"})

    if wants_json:
        return payload

    body = f"URL Source: {url}\n\nMarkdown Content:\n{markdown}\n"
    response_headers = {"X-Debug-Timing-Total-S": f"{debug_timing['total_s']:.6f}"}
    if "readerlm_s" in debug_timing:
        response_headers["X-Debug-Timing-Readerlm-S"] = f"{debug_timing['readerlm_s']:.6f}"
    if isinstance(cleaning, dict):
        response_headers["X-Markdown-Clean-Original-Chars"] = str(cleaning.get("original_chars", 0))
        response_headers["X-Markdown-Clean-Cleaned-Chars"] = str(cleaning.get("cleaned_chars", 0))
    parallel_fetch_s = debug_timing.get("fetch_markdown_html_parallel_s")
    if parallel_fetch_s is not None:
        response_headers["X-Debug-Timing-Fetch-Markdown-Html-Parallel-S"] = f"{parallel_fetch_s:.6f}"
    return Response(
        body,
        media_type="text/plain; charset=utf-8",
        headers=response_headers,
    )


@app.get("/raw/{target_url:path}")
async def read_raw_markdown(target_url: str, request: Request):
    total_started = time.perf_counter()
    url = normalize_url(target_url)
    wants_json = "application/json" in request.headers.get("accept", "")
    refresh_cache = request.query_params.get("refresh", "").strip().lower() in {"1", "true", "yes"}

    if refresh_cache:
        _delete_cache_path(_raw_markdown_cache_path(url), label="raw_markdown")
    cached_payload = None if refresh_cache else _read_raw_markdown_cache(url)
    if cached_payload is not None:
        payload = dict(cached_payload)
        payload.setdefault("debug_timing", {})
        payload["debug_timing"].update(
            {
                "cache_hit": True,
                "cache_path": str(_raw_markdown_cache_path(url)),
                "total_s": time.perf_counter() - total_started,
            }
        )
        if wants_json:
            return payload
        cached_data = payload.get("data") or {}
        return Response(
            str(cached_data.get("content") or ""),
            media_type="text/plain; charset=utf-8",
            headers={"X-Raw-Markdown-Cache": "hit"},
        )

    debug_timing: dict[str, Any] = {}
    async with httpx.AsyncClient() as client:
        try:
            markdown = await timed_stage_call("fetch_markdown", fetch_markdown(client, url), debug_timing)
        except StageError as exc:
            debug_timing["total_s"] = time.perf_counter() - total_started
            record_failure_debug(debug_timing, stage=exc.stage, exc=exc.cause)
            message = f"Raw markdown reader error for {url}: {exc.stage} failed: {exc.cause}"
            error_payload = {
                "data": None,
                "code": 502,
                "status": 502,
                "message": message,
                "debug_timing": debug_timing,
            }
            if wants_json:
                return JSONResponse(status_code=502, content=error_payload)
            return Response(message, status_code=502, media_type="text/plain")

    debug_timing["total_s"] = time.perf_counter() - total_started
    debug_timing["cache_hit"] = False
    payload = {
        "data": {
            "title": "",
            "url": url,
            "content": markdown,
            "raw_markdown": markdown,
            "content_source": "raw_markdown_reader",
            "content_quality": "raw",
            "debug_timing": debug_timing,
        },
        "code": 200,
        "status": 200,
        "debug_timing": debug_timing,
    }
    _write_raw_markdown_cache(url, payload=payload)
    if wants_json:
        return payload
    return Response(markdown, media_type="text/plain; charset=utf-8")
