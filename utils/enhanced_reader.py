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
READERLM_MODEL_NAME = os.environ.get("READERLM_MODEL_NAME", "jinaai/ReaderLM-v2")
READERLM_API_KEY = os.environ.get("READERLM_API_KEY", "")
READERLM_MAX_HTML_CHARS = int(os.environ.get("READERLM_MAX_HTML_CHARS", "120000"))
READER_TIMEOUT = float(os.environ.get("ENHANCED_READER_TIMEOUT", "180"))
READERLM_MAX_TOKENS = int(os.environ.get("READERLM_MAX_TOKENS", "8192"))
DEBUG_READERLM_URL_LEAK = os.environ.get("ENHANCED_READER_DEBUG_URL_LEAK", "1") != "0"
DEBUG_READERLM_URL_LEAK_DIR = Path(os.environ.get("ENHANCED_READER_DEBUG_URL_LEAK_DIR", "/tmp/enhanced_reader_url_leaks"))
TRUNCATION_MARKER = "\n<!-- enhanced_reader_truncated -->"


app = FastAPI(title="Enhanced Reader API")


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


def normalize_url(target_url: str) -> str:
    if target_url.startswith(("http://", "https://")):
        return target_url
    return "https://" + target_url


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

    response = await client.post(
        f"{READERLM_API_BASE.rstrip('/')}/chat/completions",
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


async def timed_call(label: str, coro, timing: dict[str, Any]):
    started = time.perf_counter()
    try:
        return await coro
    finally:
        timing[f"{label}_s"] = time.perf_counter() - started


@app.get("/{target_url:path}")
async def read(target_url: str, request: Request):
    total_started = time.perf_counter()
    url = normalize_url(target_url)
    wants_json = "application/json" in request.headers.get("accept", "")
    debug_timing: dict[str, Any] = {}

    async with httpx.AsyncClient() as client:
        try:
            fetch_started = time.perf_counter()
            markdown_response, html_response = await asyncio.gather(
                timed_call("fetch_markdown", fetch_markdown(client, url), debug_timing),
                timed_call("fetch_html", fetch_html(client, url), debug_timing),
            )
            fetch_done = time.perf_counter()
            debug_timing["fetch_markdown_html_parallel_s"] = fetch_done - fetch_started
            html = html_response
            readerlm_started = time.perf_counter()
            markdown = await convert_html_to_markdown(client, html, source_url=url, debug_timing=debug_timing)
            debug_timing["readerlm_s"] = time.perf_counter() - readerlm_started
        except Exception as exc:
            debug_timing["total_s"] = time.perf_counter() - total_started
            message = f"Enhanced Reader error for {url}: {exc}"
            if wants_json:
                return JSONResponse(
                    status_code=502,
                    content={
                        "data": None,
                        "code": 502,
                        "status": 502,
                        "message": message,
                        "debug_timing": debug_timing,
                    },
                )
            return Response(
                message,
                status_code=502,
                media_type="text/plain",
                headers={"X-Debug-Timing-Total-S": f"{debug_timing['total_s']:.6f}"},
            )

    debug_timing["total_s"] = time.perf_counter() - total_started

    if wants_json:
        return {
            "data": {
                "title": "",
                "url": url,
                "content": markdown,
                "raw_markdown": markdown_response,
                "debug_timing": debug_timing,
            },
            "code": 200,
            "status": 200,
            "debug_timing": debug_timing,
        }

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
