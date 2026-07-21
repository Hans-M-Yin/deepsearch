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
READERLM_API_BASE = os.environ.get("READERLM_API_BASE", "http://127.0.0.1:8003/v1")
READERLM_API_BASES_ENV = os.environ.get("READERLM_API_BASES", "")
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


app = FastAPI(title="Enhanced Reader API")


def _parse_readerlm_api_bases() -> list[str]:
    values = [item.strip().rstrip("/") for item in READERLM_API_BASES_ENV.split(",") if item.strip()]
    if values:
        return values
    return [READERLM_API_BASE.rstrip("/")]


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


def _cache_path(url: str, *, wants_json: bool) -> Path:
    return ENHANCED_READER_CACHE_DIR / f"{_cache_key(url, wants_json=wants_json)}.json"


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


def looks_like_antibot(text: str | None) -> bool:
    normalized = re.sub(r"\s+", " ", str(text or "").strip().lower())
    if not normalized:
        return False
    return any(pattern in normalized for pattern in ANTI_BOT_PATTERNS)


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

    readerlm_api_base = await _select_readerlm_api_base()
    if debug_timing is not None:
        debug_timing["readerlm_api_base"] = readerlm_api_base

    response = await client.post(
        f"{readerlm_api_base}/chat/completions",
        headers=headers,
        json={
            "model": READERLM_MODEL_NAME,
            "messages": [{"role": "user", "content": create_prompt(readerlm_input)}],
            "temperature": 0,
            "max_tokens": READERLM_MAX_TOKENS,
            "extra_body": {"repetition_penalty": 1.08},
        },
        timeout=READER_TIMEOUT,
    )
    response.raise_for_status()
    data: dict[str, Any] = response.json()
    markdown = strip_outer_markdown_fence(data["choices"][0]["message"]["content"])
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

    readerlm_api_base = await _select_readerlm_api_base()
    if debug_timing is not None:
        debug_timing["readerlm_api_base"] = readerlm_api_base

    response = await client.post(
        f"{readerlm_api_base}/chat/completions",
        headers=headers,
        json={
            "model": READERLM_MODEL_NAME,
            "messages": [{"role": "user", "content": create_markdown_cleanup_prompt(readerlm_input)}],
            "temperature": 0,
            "max_tokens": READERLM_MAX_TOKENS,
            "extra_body": {"repetition_penalty": 1.08},
        },
        timeout=READER_TIMEOUT,
    )
    response.raise_for_status()
    data: dict[str, Any] = response.json()
    markdown = strip_outer_markdown_fence(data["choices"][0]["message"]["content"])
    debug_url_leak_after_readerlm(
        source_url=source_url,
        readerlm_input=readerlm_input,
        markdown=markdown,
        debug_timing=debug_timing,
    )
    return markdown


async def _select_readerlm_api_base() -> str:
    if len(READERLM_API_BASES) == 1:
        return READERLM_API_BASES[0]
    async with _READERLM_API_BASE_LOCK:
        index = next(_READERLM_API_BASE_CYCLE)
    return READERLM_API_BASES[index]


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
    total_started = time.perf_counter()
    url = normalize_url(target_url)
    wants_json = "application/json" in request.headers.get("accept", "")
    debug_timing: dict[str, Any] = {}

    cached_payload = _read_cache(url, wants_json=wants_json)
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
            else:
                markdown_response = await timed_stage_call("fetch_markdown", fetch_markdown(client, url), debug_timing)
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
            _write_cache(url, wants_json=wants_json, payload=error_payload, negative=True)
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
            _write_cache(url, wants_json=wants_json, payload=error_payload, negative=True)
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
    _write_cache(url, wants_json=wants_json, payload=payload, negative=content_quality in {"anti_bot", "empty"})

    if wants_json:
        return payload

    body = f"URL Source: {url}\n\nMarkdown Content:\n{markdown}\n"
    return Response(
        body,
        media_type="text/plain; charset=utf-8",
        headers={
            "X-Debug-Timing-Fetch-Markdown-Html-Parallel-S": f"{debug_timing['fetch_markdown_html_parallel_s']:.6f}",
            "X-Debug-Timing-Readerlm-S": f"{debug_timing['readerlm_s']:.6f}",
            "X-Debug-Timing-Total-S": f"{debug_timing['total_s']:.6f}",
        },
    )
