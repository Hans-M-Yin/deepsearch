"""Tool schemas and backend implementations for synthesis SFT agents."""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import random
import re
import socket
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, unquote, urlparse, urlsplit, urlunparse

import requests
from PIL import Image, ImageOps

try:
    import fitz
except ImportError:  # pragma: no cover - optional dependency
    fitz = None

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "synthesis.sft"

from synthesis.search_client import SerperSearchClient, acquire_serper_api_key
from synthesis.wiki_text_builder import EnhancedReaderClient
from synthesis.model_worker import LLM_WORKER
from synthesis.model_worker import ModelMessage
from synthesis.model_worker import ModelRequest
logger = logging.getLogger(__name__)
MAX_SEARCH_RESULTS = 5
TOOL_NETWORK_TIMEOUT_S = 120
# A request is attempted once initially, then at most this many additional
# times when the failure is a transport timeout.  Keep this distinction
# explicit: retrying authentication, validation, or quota errors only delays
# the trajectory and cannot make them succeed.
TOOL_TIMEOUT_RETRIES = 5
TOOL_RETRY_SLEEP_S = 5
TOOL_429_RETRY_SLEEP_S = 60
MAX_DOWNLOADED_IMAGE_LONG_EDGE = 1920
MAX_DOWNLOADED_IMAGE_SHORT_EDGE = 1080
RESIZED_IMAGE_LONG_EDGE = 1200
AMBIGUOUS_CONTENT_TYPES = {
    "",
    "application/octet-stream",
    "binary/octet-stream",
    "text/plain",
}
PDF_CONTENT_TYPES = {
    "application/pdf",
    "application/x-pdf",
}
T2T_BLOCKED_SEARCH_DOMAINS: tuple[str, ...] = ()
T2I_BLOCKED_IMAGE_SEARCH_DOMAINS = (
    # TikTok image-search results are often video-thumbnail endpoints or signed
    # CDN URLs. The source pages are video posts and direct image URLs frequently
    # return 403, so they are poor static image evidence for SFT trajectories.
    "tiktok.com",
    "tiktokcdn.com",
    "tiktokcdn-us.com",
)
I2I_BLOCKED_IMAGE_SEARCH_DOMAINS: tuple[str, ...] = ()
_SFT_FIXED_REQUEST_ID = "3200636808"
_DEFAULT_SFT_QWEN_MODEL_ALIAS = "multimodal_process"
_URL_KEYWORD_CACHE: dict[tuple[str, str], str] = {}
_URL_KEYWORD_CACHE_LOCK = threading.Lock()

# Wikimedia's upload CDN has stricter limits than ordinary web pages.  Keep
# this transport-level guard isolated and switchable so it can be removed (or
# disabled) without changing any tool implementation.
_WIKIMEDIA_UPLOAD_HOST = "upload.wikimedia.org"
_WIKIMEDIA_REQUEST_SEMAPHORE = threading.BoundedSemaphore(3)
_WIKIMEDIA_THROTTLE_LOCK = threading.Lock()
_WIKIMEDIA_NEXT_REQUEST_AT = 0.0
_WIKIMEDIA_COOLDOWN_UNTIL = 0.0
_WIKIMEDIA_ACTIVE_REQUESTS = 0
_WIKIMEDIA_QUEUED_REQUESTS = 0

_URL_KEYWORD_NOISE_TOKENS = {
    "image", "images", "img", "upload", "uploads", "static", "media",
    "cdn", "cache", "cached", "thumbnail", "thumbnails", "thumb", "thumbs",
    "resize", "resizer", "width", "height", "quality", "auto", "best",
    "format", "fit", "crop", "newscms", "original", "download", "file",
    "files", "content", "assets", "asset", "public", "private", "vision",
    "deepresearch", "synthesis", "trajectory", "turn", "region", "search",
    "hans", "oss", "aliyuncs", "com", "www", "http", "https",
}

PROMPT_URL_SEMANTIC_KEYWORDS = """Extract retrieval keywords that are explicitly present in URL metadata.

You receive a URL after technical noise, signatures, hashes, resize directives,
and opaque identifiers have been removed. Return only useful semantic keywords
or short phrases made entirely from words that appear in the supplied URL
metadata. Do not infer identities, expand abbreviations, correct spellings,
add context, or make factual claims. An empty result is better than guessing.

Return exactly one JSON object:
{"keywords":"semicolon-separated URL-derived keywords, or an empty string"}
"""


@dataclass(slots=True)
class UrlResource:
    """Search-result provenance used internally by read_url fallbacks."""

    primary_url: str
    resource_id: str = ""
    result_id: str | None = None
    kind: str = "unknown"
    title: str | None = None
    snippet: str | None = None
    image_url: str | None = None
    thumbnail_url: str | None = None
    source_page_url: str | None = None
    search_tool: str | None = None
    search_query: str | None = None
    rank: int | None = None
    url_keywords: str | None = None
    fallback_urls: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _is_wikimedia_upload_url(url: str) -> bool:
    try:
        return (urlparse(str(url or "")).hostname or "").lower() == _WIKIMEDIA_UPLOAD_HOST
    except Exception:
        return False


def _wikimedia_throttle_enabled() -> bool:
    return _env_flag("SFT_WIKIMEDIA_THROTTLE", True)


def _wikimedia_min_interval_s() -> float:
    raw = str(os.environ.get("SFT_WIKIMEDIA_MIN_INTERVAL_S") or "0.75").strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.75


def _wikimedia_429_cooldown_s() -> float:
    raw = str(os.environ.get("SFT_WIKIMEDIA_429_COOLDOWN_S") or "60").strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 60.0


def _wikimedia_throttle_debug_enabled() -> bool:
    return _env_flag("SFT_WIKIMEDIA_THROTTLE_DEBUG", True)


def _wikimedia_throttle_debug(event: str, **kwargs: Any) -> None:
    if not _wikimedia_throttle_debug_enabled():
        return
    details = " ".join(f"{key}={value!r}" for key, value in kwargs.items())
    suffix = f" {details}" if details else ""
    print(f"[wikimedia-throttle] event={event}{suffix}", file=sys.stderr, flush=True)


def _acquire_wikimedia_request_slot(url: str) -> bool:
    """Rate-limit only upload.wikimedia.org requests in this process."""

    if not _wikimedia_throttle_enabled() or not _is_wikimedia_upload_url(url):
        return False

    global _WIKIMEDIA_NEXT_REQUEST_AT, _WIKIMEDIA_QUEUED_REQUESTS, _WIKIMEDIA_ACTIVE_REQUESTS
    with _WIKIMEDIA_THROTTLE_LOCK:
        _WIKIMEDIA_QUEUED_REQUESTS += 1
        queued = _WIKIMEDIA_QUEUED_REQUESTS
        active = _WIKIMEDIA_ACTIVE_REQUESTS
    if active >= 3 or queued > 1:
        _wikimedia_throttle_debug("queued", active=active, queued=queued, url=url)

    semaphore_acquired = False
    try:
        _WIKIMEDIA_REQUEST_SEMAPHORE.acquire()
        semaphore_acquired = True
        with _WIKIMEDIA_THROTTLE_LOCK:
            now = time.monotonic()
            scheduled_at = max(now, _WIKIMEDIA_NEXT_REQUEST_AT, _WIKIMEDIA_COOLDOWN_UNTIL)
            _WIKIMEDIA_NEXT_REQUEST_AT = scheduled_at + _wikimedia_min_interval_s()
        wait_s = scheduled_at - now
        if wait_s > 0:
            time.sleep(wait_s)
        with _WIKIMEDIA_THROTTLE_LOCK:
            _WIKIMEDIA_QUEUED_REQUESTS = max(0, _WIKIMEDIA_QUEUED_REQUESTS - 1)
            _WIKIMEDIA_ACTIVE_REQUESTS += 1
            queued = _WIKIMEDIA_QUEUED_REQUESTS
            active = _WIKIMEDIA_ACTIVE_REQUESTS
        if wait_s > 0 or queued > 0:
            _wikimedia_throttle_debug(
                "started",
                active=active,
                queued=queued,
                wait_s=round(wait_s, 3),
                url=url,
            )
        return True
    except BaseException:
        with _WIKIMEDIA_THROTTLE_LOCK:
            _WIKIMEDIA_QUEUED_REQUESTS = max(0, _WIKIMEDIA_QUEUED_REQUESTS - 1)
        if semaphore_acquired:
            _WIKIMEDIA_REQUEST_SEMAPHORE.release()
        raise


def _release_wikimedia_request_slot(acquired: bool, url: str = "") -> None:
    if not acquired:
        return
    global _WIKIMEDIA_ACTIVE_REQUESTS
    with _WIKIMEDIA_THROTTLE_LOCK:
        _WIKIMEDIA_ACTIVE_REQUESTS = max(0, _WIKIMEDIA_ACTIVE_REQUESTS - 1)
        active = _WIKIMEDIA_ACTIVE_REQUESTS
        queued = _WIKIMEDIA_QUEUED_REQUESTS
    _WIKIMEDIA_REQUEST_SEMAPHORE.release()
    if queued > 0:
        _wikimedia_throttle_debug("finished", active=active, queued=queued, url=url)


def _note_wikimedia_429(url: str, retry_after: str | None) -> None:
    if not _wikimedia_throttle_enabled() or not _is_wikimedia_upload_url(url):
        return
    try:
        cooldown_s = max(float(retry_after or ""), 0.0)
    except (TypeError, ValueError):
        cooldown_s = _wikimedia_429_cooldown_s()
    global _WIKIMEDIA_COOLDOWN_UNTIL
    with _WIKIMEDIA_THROTTLE_LOCK:
        _WIKIMEDIA_COOLDOWN_UNTIL = max(
            _WIKIMEDIA_COOLDOWN_UNTIL,
            time.monotonic() + cooldown_s,
        )


def _read_url_image_max_retries() -> int:
    """Return the bounded retry budget for direct image downloads.

    Image CDNs commonly answer a burst of requests with HTTP 429.  Retrying
    the same URL six times is counterproductive and can also delay trying a
    registered thumbnail or source-page image.  Non-429 failures use the
    normal six-attempt budget by default; 429 responses have a separate,
    smaller budget below.
    """

    return max(0, _env_int("SFT_READ_URL_IMAGE_RETRIES", TOOL_TIMEOUT_RETRIES))


def _read_url_image_retry_sleep_s() -> int:
    return max(0, _env_int("SFT_READ_URL_IMAGE_RETRY_SLEEP_S", TOOL_RETRY_SLEEP_S))


def _read_url_429_max_retries() -> int:
    return max(0, min(1, _env_int("SFT_READ_URL_429_RETRIES", 1)))


def _read_url_429_retry_sleep_s() -> int:
    return max(0, _env_int("SFT_READ_URL_429_RETRY_SLEEP_S", TOOL_429_RETRY_SLEEP_S))


def _read_url_debug(message: str, **kwargs: Any) -> None:
    if not _env_flag("SFT_READ_URL_DEBUG"):
        return
    details = " ".join(f"{key}={value!r}" for key, value in kwargs.items())
    suffix = f" {details}" if details else ""
    print(f"[read_url debug] {message}{suffix}", file=sys.stderr, flush=True)


def _tool_retry_debug(
    tool: str,
    *,
    attempt: int,
    max_attempts: int,
    sleep_seconds: float,
    error: object,
    **kwargs: Any,
) -> None:
    """Emit one stable, always-visible line for every tool-level retry.

    ``attempt`` is the failed attempt that triggered the retry (the first
    request is attempt 1), while ``max_attempts`` includes the initial request.
    Keep this separate from the optional read-url debug stream so retry logs
    are available even when ``SFT_READ_URL_DEBUG`` is disabled.
    """

    details = {
        "tool": tool,
        "retry_attempt": attempt,
        "max_attempts": max_attempts,
        "sleep_seconds": round(float(sleep_seconds), 3),
        "error": str(error)[:1000],
        **kwargs,
    }
    suffix = " ".join(f"{key}={value!r}" for key, value in details.items())
    print(f"[tool-retry] {suffix}", file=sys.stderr, flush=True)


def _normalize_request_url(url: str) -> str:
    """Return an HTTP URL safe for stdlib/reader clients.

    Some search results contain raw non-ASCII characters in the path, e.g.
    ``Künstler`` or curly apostrophes.  urllib-based readers can fail before the
    request is sent with an ASCII encoding error.  Normalize by IDNA-encoding the
    host and percent-encoding path/query/fragment while preserving normal URL
    delimiters and existing percent escapes.
    """

    raw_url = str(url or "").strip()
    if not raw_url:
        return raw_url
    if not raw_url.startswith(("http://", "https://")):
        raw_url = f"https://{raw_url}"
    parsed = urlparse(raw_url)
    scheme = parsed.scheme or "https"
    hostname = parsed.hostname or ""
    try:
        hostname = hostname.encode("idna").decode("ascii")
    except UnicodeError:
        hostname = parsed.hostname or ""
    netloc = hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    if parsed.username:
        userinfo = quote(unquote(parsed.username), safe="")
        if parsed.password:
            userinfo += ":" + quote(unquote(parsed.password), safe="")
        netloc = f"{userinfo}@{netloc}"
    path = quote(unquote(parsed.path or "/"), safe="/%:@!$&'()*+,;=-._~")
    query = quote(unquote(parsed.query or ""), safe="=&?/:@!$'()*+,;%-._~")
    fragment = quote(unquote(parsed.fragment or ""), safe="=&?/:@!$'()*+,;%-._~")
    return urlunparse((scheme, netloc, path, "", query, fragment))


DEFAULT_SEARCH_TOP_K = _env_int("SEARCH_TOP_K", MAX_SEARCH_RESULTS)
TOOL_STATS_PRINT_EVERY = max(1, _env_int("SFT_TOOL_STATS_PRINT_EVERY", 4))

_TOOL_STATS_LOCK = threading.Lock()
_TOOL_STATS: dict[str, Any] = {
    "total_calls": 0,
    "tools": {
        "t2t_search": {"success": 0, "failure": 0, "total_results": 0},
        "t2i_search": {"success": 0, "failure": 0, "total_results": 0},
        "i2i_search": {"success": 0, "failure": 0, "total_results": 0},
        "read_url": {
            "success": 0,
            "failure": 0,
            "text_calls": 0,
            "pdf_calls": 0,
            "image_calls": 0,
            "image_http_errors": 0,
            "image_http_statuses": {},
        },
    },
}


def _tool_stats_snapshot_text() -> str:
    stats = _TOOL_STATS
    lines = [f"[tool-stats] total_calls={stats['total_calls']}"]
    for tool_name in ("t2t_search", "t2i_search", "i2i_search"):
        tool_stats = stats["tools"][tool_name]
        lines.append(
            f"[tool-stats] {tool_name} success={tool_stats['success']} "
            f"failure={tool_stats['failure']} total_results={tool_stats['total_results']}"
        )
    read_stats = stats["tools"]["read_url"]
    lines.append(
        f"[tool-stats] read_url success={read_stats['success']} failure={read_stats['failure']} "
        f"text_calls={read_stats['text_calls']} pdf_calls={read_stats['pdf_calls']} "
        f"image_calls={read_stats['image_calls']} image_http_errors={read_stats['image_http_errors']} "
        f"image_http_statuses={json.dumps(read_stats['image_http_statuses'], ensure_ascii=False, sort_keys=True)}"
    )
    return "\n".join(lines)


def _record_search_tool_call(tool_name: str, *, success: bool, result_count: int = 0) -> None:
    snapshot_text: str | None = None
    with _TOOL_STATS_LOCK:
        tool_stats = _TOOL_STATS["tools"][tool_name]
        key = "success" if success else "failure"
        tool_stats[key] += 1
        tool_stats["total_results"] += max(0, int(result_count))
        _TOOL_STATS["total_calls"] += 1
        if _TOOL_STATS["total_calls"] % TOOL_STATS_PRINT_EVERY == 0:
            snapshot_text = _tool_stats_snapshot_text()
    if snapshot_text:
        print(snapshot_text, file=sys.stderr, flush=True)


def _extract_http_status_code(exc: Exception) -> int | None:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code
    return None


def _record_read_url_call(
    *,
    branch: str,
    success: bool,
    image_http_status: int | None = None,
) -> None:
    snapshot_text: str | None = None
    with _TOOL_STATS_LOCK:
        read_stats = _TOOL_STATS["tools"]["read_url"]
        key = "success" if success else "failure"
        read_stats[key] += 1
        branch_key = f"{branch}_calls"
        if branch_key in read_stats:
            read_stats[branch_key] += 1
        if branch == "image" and image_http_status is not None:
            read_stats["image_http_errors"] += 1
            status_key = str(image_http_status)
            status_counts = read_stats["image_http_statuses"]
            status_counts[status_key] = int(status_counts.get(status_key) or 0) + 1
        _TOOL_STATS["total_calls"] += 1
        if _TOOL_STATS["total_calls"] % TOOL_STATS_PRINT_EVERY == 0:
            snapshot_text = _tool_stats_snapshot_text()
    if snapshot_text:
        print(snapshot_text, file=sys.stderr, flush=True)


def _web_request_headers(*, referer_url: str | None = None) -> dict[str, str]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    if referer_url:
        headers["Referer"] = referer_url
    return headers


def _resource_candidate_urls(resource: UrlResource | None, requested_url: str) -> list[tuple[str, str, str | None]]:
    """Return ordered binary download candidates as (kind, url, referer)."""
    candidates: list[tuple[str, str, str | None]] = []
    seen: set[str] = set()

    def add(kind: str, url: str | None, referer: str | None = None) -> None:
        normalized = str(url or "").strip()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        candidates.append((kind, normalized, str(referer or "").strip() or None))

    if resource is None:
        add("requested", requested_url)
        return candidates
    referer = resource.source_page_url
    add("requested", requested_url, referer)
    add("image_url", resource.image_url, referer)
    add("primary_url", resource.primary_url, referer)
    add("thumbnail_url", resource.thumbnail_url, referer)
    for fallback_url in resource.fallback_urls:
        add("resource_fallback", fallback_url, referer)
    return candidates


def _source_page_image_candidates(
    resource: UrlResource | None,
    *,
    max_retries: int | None = None,
    max_429_retries: int | None = None,
) -> list[str]:
    """Best-effort og:image/twitter:image extraction from a result page."""
    if resource is None or not resource.source_page_url:
        return []
    effective_max_retries = (
        _read_url_image_max_retries() if max_retries is None else max(0, int(max_retries))
    )
    try:
        response = _request_with_retry(
            "GET",
            resource.source_page_url,
            timeout=TOOL_NETWORK_TIMEOUT_S,
            allow_redirects=True,
            headers=_web_request_headers(referer_url=resource.source_page_url),
            max_retries=effective_max_retries,
            retry_sleep_s=_read_url_image_retry_sleep_s(),
            max_429_retries=(
                _read_url_429_max_retries() if max_429_retries is None else max(0, int(max_429_retries))
            ),
            retry_429_sleep_s=_read_url_429_retry_sleep_s(),
            retry_on_429=True,
        )
        html = response.text
        response.close()
    except Exception:
        return []
    candidates: list[str] = []
    patterns = (
        r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']',
        r'<meta[^>]+name=["\']twitter:image(?::src)?["\'][^>]+content=["\']([^"\']+)',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']twitter:image(?::src)?["\']',
    )
    for pattern in patterns:
        for match in re.finditer(pattern, html, flags=re.IGNORECASE):
            resolved = requests.compat.urljoin(resource.source_page_url, match.group(1).strip())
            if resolved and resolved not in candidates:
                candidates.append(resolved)
    return candidates[:4]


def get_tool_definitions() -> list[dict[str, Any]]:
    """Return OpenAI-compatible function tool definitions."""

    return [
        {
            "type": "function",
            "function": {
                "name": "t2t_search",
                "description": (
                    "Search text/web documents on Google from a text query. Returns "
                    "search results such as title, snippet, and a compact source_page_id. "
                    "Full URLs are kept privately by the runtime. If the "
                    "agent wants the full content of a result, it should call "
                    "read_url with that source_page_id. URL keyword fields expose information from the original URL and can help select which resource ID to read; inspect the resource before treating it as evidence."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "A concrete web search query string.",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "t2i_search",
                "description": (
                    "Search images from a text query on Google. Results provide compact "
                    "image_id and source_page_id references instead of raw URLs. Use read_url "
                    "with one of those IDs to inspect the image or source page."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "A concrete image-search query string.",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "i2i_search",
                "description": (
                    "Search for visually similar or matching images from the most recent image in the current context. "
                    "Results provide compact image_id and source_page_id references instead of raw URLs. "
                    "You can locate the bounding box of the entity you want to recognize in the image and then search that region separately as an image."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "region": {
                            "type": "array",
                            "description": (
                                # #### START Response 0720 ####
                                "Optional x-first bounding box on the current image to crop before search. "
                                "Use normalized coordinates from 0 to 1000 in exactly this order: "
                                "[x1, y1, x2, y2], where x increases left-to-right and y increases top-to-bottom. "
                                "Use [0, 0, 1000, 1000] for the full image. If this parameter is omitted, "
                                "the entire image will be searched."
                                # #### END Response 0720 ####
                            ),
                            "items": {"type": "number"},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_url",
                "description": (
                    "Read a URL. If it returns text, fetch content through a "
                    "reader backend and optionally summarize only the part "
                    "relevant to the current tool goal. If it returns an image, the image will be downloaded for you. "
                    "Use resource_id from a prior search result whenever available; url remains supported for direct links. "
                    "NOTICE, only resources or URLs you have got from search tools can be read. Wikipedia and Wiki commons is excluded for safety reasons."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "Legacy direct URL to read. Prefer resource_id from a prior search result.",
                        },
                        "resource_id": {
                            "type": "string",
                            "description": "Compact page_id or image_id returned by a prior search result.",
                        },
                        "goal": {
                            "type": "string",
                            "description": (
                                "Why reading this URL is the right next step, and what "
                                "specific evidence should be extracted from it."
                            ),
                        },
                    },
                    "required": [],
                },
            },
        },
    ]


def get_responses_tool_definitions() -> list[dict[str, Any]]:
    """Return Responses-API-compatible function tool definitions."""

    # #### START Response 0720 ####
    definitions: list[dict[str, Any]] = []
    for item in get_tool_definitions():
        function_block = dict(item["function"])
        definitions.append(
            {
                "type": "function",
                "name": function_block["name"],
                "description": function_block.get("description", ""),
                "parameters": function_block.get("parameters") or {"type": "object", "properties": {}},
                # Keep best-effort mode for now because several existing schemas
                # contain optional properties and are not strict-mode compatible.
                "strict": False,
            }
        )
    return definitions
    # #### END Response 0720 ####


def get_tool_definitions_json() -> str:
    """Return tool definitions as a JSON string."""

    return json.dumps(get_tool_definitions(), ensure_ascii=False, indent=2)


def normalize_tool_arguments(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Normalize alias fields to one canonical representation."""

    params = dict(arguments)
    if name in {"t2t_search", "t2i_search"}:
        if "q" in params and "query" not in params:
            params["query"] = params.pop("q")
        if "hl" in params and "lang" not in params:
            params["lang"] = params.pop("hl")
    if name == "read_url":
        if "URL" in params and "url" not in params:
            params["url"] = params.pop("URL")
        if "query" in params:
            params.pop("query")
    if name == "i2i_search" and isinstance(params.get("region"), str):
        raw_region = params["region"].strip()
        if raw_region.startswith(("[", "{")):
            try:
                params["region"] = json.loads(raw_region)
            except json.JSONDecodeError:
                pass
    return params


def _serper_client() -> SerperSearchClient:
    return SerperSearchClient(
        search_url=os.environ.get("SERPER_SEARCH_URL") or "https://google.serper.dev/search",
        images_url=os.environ.get("SERPER_IMAGES_URL") or "https://google.serper.dev/images",
        timeout_s=60.0,
    )


def _resolve_registered_model_alias(alias_or_model: str | None) -> dict[str, Any] | None:
    if not alias_or_model:
        return None
    try:
        return LLM_WORKER.get_model(alias_or_model)
    except Exception:
        return None


def get_sft_qwen_model_alias() -> str:
    """Return the shared default auxiliary Qwen model alias for SFT tools."""

    return str(
        os.environ.get("SFT_QWEN_MODEL_ALIAS") or _DEFAULT_SFT_QWEN_MODEL_ALIAS
    ).strip() or _DEFAULT_SFT_QWEN_MODEL_ALIAS


def _summarizer_model_alias() -> str | None:
    configured_alias = os.environ.get("SFT_SUMMARIZER_MODEL")
    if configured_alias and _resolve_registered_model_alias(configured_alias) is not None:
        return configured_alias

    default_alias = os.environ.get("SFT_SUMMARIZER_MODEL_ALIAS") or get_sft_qwen_model_alias()
    if _resolve_registered_model_alias(default_alias) is not None:
        return default_alias

    return None


def _sft_worker_metadata(trace_label: str) -> dict[str, Any]:
    return {
        "trace_label": trace_label,
        "session_id": _SFT_FIXED_REQUEST_ID,
        "prompt_cache_key": _SFT_FIXED_REQUEST_ID,
        "user_id": _SFT_FIXED_REQUEST_ID,
        "x_tt_logid": _SFT_FIXED_REQUEST_ID,
    }


def _url_keyword_tokens(url: str) -> list[str]:
    """Return non-technical word tokens explicitly present in a URL."""
    try:
        parsed = urlsplit(str(url or "").strip())
    except ValueError:
        return []
    hostname = (parsed.hostname or "").lower()
    raw_parts = [hostname, unquote(parsed.path or "")]
    tokens: list[str] = []
    for raw_part in raw_parts:
        for token in re.findall(r"[A-Za-z][A-Za-z0-9]{1,}", raw_part):
            normalized = token.lower()
            if normalized in _URL_KEYWORD_NOISE_TOKENS:
                continue
            if normalized.isdigit() or re.fullmatch(r"[a-f0-9]{8,}", normalized):
                continue
            # Wire-service IDs, UUID-like strings, and compact cache keys are
            # usually not meaningful to an agent even when alphabetic.
            if re.fullmatch(r"[a-z]{1,4}\d[a-z0-9]{4,}", normalized):
                continue
            if normalized not in tokens:
                tokens.append(normalized)
    return tokens


def _clean_url_for_keyword_prompt(url: str) -> dict[str, Any]:
    """Expose only URL-derived lexical material, never auth/query noise."""
    try:
        parsed = urlsplit(str(url or "").strip())
    except ValueError:
        return {"hostname": "", "path": "", "tokens": []}
    return {
        "hostname": (parsed.hostname or "").lower(),
        "path": unquote(parsed.path or ""),
        "tokens": _url_keyword_tokens(url),
    }


def _validate_url_keyword_hint(value: Any, *, allowed_tokens: list[str]) -> str:
    """Keep only LLM keywords whose word tokens were present in the URL."""
    allowed = set(allowed_tokens)
    candidates = [part.strip() for part in str(value or "").split(";") if part.strip()]
    accepted: list[str] = []
    for candidate in candidates:
        words = [word.lower() for word in re.findall(r"[A-Za-z][A-Za-z0-9]{1,}", candidate)]
        if words and all(word in allowed for word in words):
            accepted.append(" ".join(words))
    return "; ".join(dict.fromkeys(accepted))


def extract_url_semantic_keywords(
    url: str,
    *,
    model_alias: str | None = None,
) -> str:
    """Return a conservative, URL-derived retrieval hint.

    The returned string contains only words present in the URL host/path. It is
    discovery metadata, not evidence about the underlying webpage or image.
    """
    raw_url = str(url or "").strip()
    if not raw_url:
        return ""
    model_alias = model_alias or get_sft_qwen_model_alias()
    cleaned = _clean_url_for_keyword_prompt(raw_url)
    allowed_tokens = list(cleaned["tokens"])
    if not allowed_tokens:
        return ""
    cache_key = (raw_url, model_alias)
    with _URL_KEYWORD_CACHE_LOCK:
        cached = _URL_KEYWORD_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        response = LLM_WORKER.generate(
            ModelRequest(
                model=model_alias,
                messages=[
                    ModelMessage(role="system", content=PROMPT_URL_SEMANTIC_KEYWORDS),
                    ModelMessage(
                        role="user",
                        content=json.dumps(cleaned, ensure_ascii=False, sort_keys=True),
                    ),
                ],
                response_format={"type": "json_object"},
                max_tokens=128,
                metadata=_sft_worker_metadata("extract_url_semantic_keywords"),
            )
        )
        parsed = json.loads(response.content or "{}")
        hint = _validate_url_keyword_hint(
            parsed.get("keywords") if isinstance(parsed, dict) else "",
            allowed_tokens=allowed_tokens,
        )
    except Exception as exc:  # pragma: no cover - remote model bound
        logger.warning("URL keyword extraction failed: %s", exc)
        hint = ""
    with _URL_KEYWORD_CACHE_LOCK:
        _URL_KEYWORD_CACHE[cache_key] = hint
    return hint


def _canonical_resource_url(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return raw
    if parsed.scheme not in {"http", "https"}:
        return raw
    hostname = (parsed.hostname or "").lower()
    return f"{parsed.scheme.lower()}://{hostname}{parsed.path or '/'}" + (f"?{parsed.query}" if parsed.query else "")


def _resource_id(kind: str, url: str) -> str:
    digest = sha256(_canonical_resource_url(url).encode("utf-8")).hexdigest()[:8]
    return f"{kind}_{digest}"


def _canonical_search_result(tool_name: str, raw: dict[str, Any], rank: int) -> dict[str, Any]:
    image_url = str(raw.get("image_url") or raw.get("imageUrl") or "").strip()
    thumbnail_url = str(raw.get("thumbnail_url") or raw.get("thumbnailUrl") or "").strip()
    page_url = str(raw.get("source_page_url") or raw.get("link") or raw.get("url") or "").strip()
    return {
        "title": str(raw.get("title") or "").strip(),
        "source": str(raw.get("source") or "").strip(),
        "snippet": str(raw.get("snippet") or "").strip(),
        "rank": int(raw.get("rank") or rank),
        "page_url": page_url,
        "image_url": image_url,
        "thumbnail_url": thumbnail_url,
        "kind": "image" if tool_name in {"t2i_search", "i2i_search"} else "page",
    }


def postprocess_search_output(
    *,
    tool_name: str,
    output: dict[str, Any],
    url_keyword_model: str | None = None,
) -> tuple[dict[str, Any], list[UrlResource]]:
    """Convert search results into compact agent output plus private resources.

    Agent-visible records contain IDs, semantic metadata, and conservative
    URL-derived hints. Full URLs, thumbnail fallbacks, and provenance remain in
    ``UrlResource`` objects for the runtime context.
    """
    raw_items = output.get("results") if tool_name in {"t2t_search", "t2i_search"} else output.get("matches")
    if not isinstance(raw_items, list):
        return dict(output), []
    canonical = [
        _canonical_search_result(tool_name, item, index)
        for index, item in enumerate(raw_items, start=1)
        if isinstance(item, dict)
    ]
    unique_urls = {
        url for item in canonical for url in (item["page_url"], item["image_url"])
        if url
    }
    hints: dict[str, str] = {}
    resolved_url_keyword_model = url_keyword_model or get_sft_qwen_model_alias()
    max_workers = len(unique_urls)
    if max_workers:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {
                executor.submit(
                    extract_url_semantic_keywords,
                    url,
                    model_alias=resolved_url_keyword_model,
                ): url
                for url in unique_urls
            }
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    hints[url] = future.result()
                except Exception:
                    hints[url] = ""

    resources: list[UrlResource] = []
    agent_results: list[dict[str, Any]] = []
    for index, item in enumerate(canonical, start=1):
        result_id = f"{tool_name}_r_{index:02d}"
        page_id = _resource_id("page", item["page_url"]) if item["page_url"] else None
        image_id = _resource_id("image", item["image_url"]) if item["image_url"] else None
        if item["page_url"]:
            resources.append(
                UrlResource(
                    primary_url=item["page_url"], resource_id=page_id or "", result_id=result_id,
                    kind="page", title=item["title"] or None, snippet=item["snippet"] or None,
                    source_page_url=item["page_url"], search_tool=tool_name,
                    search_query=str(output.get("query") or "") or None, rank=item["rank"],
                    url_keywords=hints.get(item["page_url"]) or None,
                )
            )
        if item["image_url"]:
            resources.append(
                UrlResource(
                    primary_url=item["image_url"], resource_id=image_id or "", result_id=result_id,
                    kind="image", title=item["title"] or None, snippet=item["snippet"] or None,
                    image_url=item["image_url"], thumbnail_url=item["thumbnail_url"] or None,
                    source_page_url=item["page_url"] or None, search_tool=tool_name,
                    search_query=str(output.get("query") or "") or None, rank=item["rank"],
                    url_keywords=hints.get(item["image_url"]) or None,
                    fallback_urls=[item["thumbnail_url"]] if item["thumbnail_url"] else [],
                )
            )
        agent_item: dict[str, Any] = {
            "title": item["title"],
        }
        if item["source"]:
            agent_item["source"] = item["source"]
        if item["snippet"]:
            agent_item["snippet"] = item["snippet"]
        if image_id:
            agent_item["image_id"] = image_id
            if hints.get(item["image_url"]):
                agent_item["image_url_keywords"] = hints[item["image_url"]]
        if page_id:
            agent_item["source_page_id"] = page_id
            if hints.get(item["page_url"]):
                agent_item["source_page_url_keywords"] = hints[item["page_url"]]
        agent_results.append(agent_item)
    compact: dict[str, Any] = {
        "ok": bool(output.get("ok", True)),
        "results": agent_results,
    }
    if tool_name != "i2i_search":
        compact["query"] = output.get("query")
    return compact, resources


def summarize_with_qwen(
    content: str,
    goal: str,
    assistant_output: str = "",
) -> str:
    """Compress webpage content while preserving information relevant to the current task."""

    prompt = (
f"""
	Below you will receive raw webpage content. Your goal is evidence extraction for the user's current objective, not a general summary. Preserve only content that helps verify or disprove the tool goal and the assistant's current reasoning. The extraction must remain faithful to the original text and must not add anything that is not present in the source.
	Rules:
	1. Based on the user's provided reasoning and goal, analyze which content may be useful and which content is definitely not useful. Preserve all potentially useful evidence in detail and keep enough original context so that the evidence still makes sense on its own. Do not over-compress it into vague paraphrases.
	2. Preserve the most relevant evidence in an extractive way whenever possible. When important, keep names, dates, numbers, rankings, table headers/rows, quotations, titles, roles, locations, formulas, and qualifiers exactly as they appear.
	3. Remove obvious noise: navigation text, menus, repeated headers, footer text, social buttons, login or subscription prompts, unrelated recommendations, boilerplate, tracking text, and raw URL lists.
	4. Preserve image-related evidence when relevant: captions, alt text, file titles, figure labels, surrounding paragraph text, metadata, and positional relationships between an image and nearby captions/headings. Do not detach a caption from the image/title it describes.
	5. Perform content extraction only. Do not add extra content, and do not use or introduce any world knowledge. If the raw content does not contain the requested field or cannot substantiate the goal, explicitly write "Insufficient evidence: <brief reason>" inside <result>.
	6. If the webpage content is only an access-control/interstitial page rather than actual page content (for example verification, CAPTCHA/reCAPTCHA, Forbidden, access denied, bot checks, or a similar block page), output exactly <result>BLOCKED</result>. Do not include an explanation or any other text in <result>.
	7. If multiple parts of the raw text may be related, you may extract them in separate segments.
	8. Output format must be exactly:
	<thinking>your analysis</thinking>
	<result>the complete extracted content</result>
	The final extracted content that will be used downstream is only the content inside the <result> tag. Therefore, all content that should be preserved must appear inside <result>. Do not put meta-comments such as "the content is already minimal" in <result> unless those words are in the source.

Agent's current output:\n{assistant_output or '(empty)'}\n
Tool goal:\n{goal or '(empty)'}\n
Raw webpage content:\n{content[:80000]}\n
"""
    )
    try:
        model_alias = _summarizer_model_alias()
        if not model_alias:
            raise RuntimeError(
                "No registered summarizer model alias is available. "
                "Set SFT_SUMMARIZER_MODEL or SFT_QWEN_MODEL_ALIAS to a registered "
                "synthesis/models.json alias."
            )
        response = LLM_WORKER.generate(
            ModelRequest(
                model=model_alias,
                messages=[ModelMessage(role="user", content=prompt)],
                metadata=_sft_worker_metadata("summarize_with_qwen"),
            )
        )
        content_text = response.content or ""
        if content_text:
            cleaned = _clean_summarizer_output(content_text)
            if cleaned:
                return cleaned
    except Exception as exc:  # pragma: no cover - network bound
        logger.warning("Summarization failed: %s", exc)
    return content[:1000] + ("..." if len(content) > 1000 else "")


def _clean_summarizer_output(text: str) -> str:
    """Extract downstream content from summarizer output robustly."""

    candidate = str(text or "").strip()
    if not candidate:
        return ""
    match = re.search(r"<result>(.*?)</result>", candidate, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    open_match = re.search(r"<result>\s*(.*)$", candidate, flags=re.DOTALL | re.IGNORECASE)
    if open_match:
        return open_match.group(1).strip()
    candidate = re.sub(r"<thinking>.*?</thinking>", "", candidate, flags=re.DOTALL | re.IGNORECASE)
    candidate = re.sub(r"<think>.*?</think>", "", candidate, flags=re.DOTALL | re.IGNORECASE)
    candidate = re.sub(r"</?thinking>", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"</?think>", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"</?result>", "", candidate, flags=re.IGNORECASE)
    return candidate.strip()


def summarize_image_search(result_obj: object) -> object:
    """Normalize reverse-image search results to a compact fixed schema."""

    def _normalize_item(item: object) -> dict[str, str] | None:
        if not isinstance(item, dict):
            return None
        return {
            "title": str(item.get("title", "") or item.get("name", "") or item.get("label", "")),
            "source": str(item.get("source", "") or ""),
            "link": str(item.get("link", "") or item.get("url", "") or ""),
            "imageUrl": str(item.get("imageUrl", "") or item.get("image_url", "") or ""),
            "thumbnailUrl": str(item.get("thumbnailUrl", "") or item.get("thumbnail_url", "") or ""),
        }

    if isinstance(result_obj, list):
        return [normalized for item in result_obj if (normalized := _normalize_item(item)) is not None]

    if isinstance(result_obj, dict):
        organic = result_obj.get("organic")
        if isinstance(organic, list):
            return [normalized for item in organic if (normalized := _normalize_item(item)) is not None]
        normalized = _normalize_item(result_obj)
        return [normalized] if normalized is not None else []

    return []

def _guess_image_content_type(url: str) -> str:
    guessed_type, _ = mimetypes.guess_type(urlparse(url).path)
    if guessed_type and guessed_type.startswith("image/"):
        return guessed_type
    return ""


def _guess_pdf_content_type(url: str) -> str:
    guessed_type, _ = mimetypes.guess_type(urlparse(url).path)
    if guessed_type in PDF_CONTENT_TYPES:
        return guessed_type
    return ""


def _sniff_image_content_type(payload: bytes) -> str:
    if payload.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if payload.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if payload.startswith(b"BM"):
        return "image/bmp"
    if len(payload) >= 12 and payload.startswith(b"RIFF") and payload[8:12] == b"WEBP":
        return "image/webp"
    normalized_payload = payload.lstrip().lower()
    if normalized_payload.startswith(b"<svg") or b"<svg" in normalized_payload[:256]:
        return "image/svg+xml"
    return ""


def _sniff_pdf_content_type(payload: bytes) -> str:
    return "application/pdf" if payload.startswith(b"%PDF-") else ""


def _should_retry_http_status(status_code: int) -> bool:
    return status_code in {429, 500, 502, 503, 504}


def _is_timeout_error(exc: BaseException) -> bool:
    """Return whether an exception represents a transport timeout.

    Serper wraps ``urlopen`` failures in a RuntimeError, while requests and
    urllib expose different exception types.  Cover both forms without
    treating ordinary HTTP/API errors as retryable timeouts.
    """

    if isinstance(exc, (TimeoutError, socket.timeout, requests.Timeout)):
        return True
    reason = getattr(exc, "reason", None)
    if reason is not None and reason is not exc and _is_timeout_error(reason):
        return True
    message = str(exc).lower()
    return "timeout" in message or "timed out" in message


def _request_with_retry(
    method: str,
    url: str,
    *,
    timeout: int,
    max_retries: int = TOOL_TIMEOUT_RETRIES,
    retry_sleep_s: int = TOOL_RETRY_SLEEP_S,
    retry_on_429: bool = True,
    max_429_retries: int | None = 1,
    retry_429_sleep_s: float = TOOL_429_RETRY_SLEEP_S,
    **kwargs: Any,
) -> requests.Response:
    last_error: Exception | None = None
    attempts = 1 + max(0, int(max_retries))
    for attempt in range(1, attempts + 1):
        response: requests.Response | None = None
        wikimedia_slot_acquired = False
        try:
            wikimedia_slot_acquired = _acquire_wikimedia_request_slot(url)
            response = requests.request(method, url, timeout=timeout, **kwargs)
            if response.status_code == 429:
                _note_wikimedia_429(url, response.headers.get("Retry-After"))
            if response.status_code == 429:
                can_retry_status = retry_on_429 and (
                    max_429_retries is None or attempt <= max(0, int(max_429_retries))
                )
            else:
                can_retry_status = True
            should_retry_status = _should_retry_http_status(response.status_code) and can_retry_status
            if should_retry_status and attempt < attempts:
                retry_after = response.headers.get("Retry-After")
                if response.status_code == 429:
                    try:
                        wait_s = max(float(retry_after), retry_429_sleep_s) if retry_after else retry_429_sleep_s
                    except (TypeError, ValueError):
                        wait_s = retry_429_sleep_s
                else:
                    wait_s = retry_sleep_s * attempt
                response.close()
                sleep_s = wait_s + random.uniform(0.0, 1.0)
                _tool_retry_debug(
                    "http_request",
                    attempt=attempt,
                    max_attempts=attempts,
                    sleep_seconds=sleep_s,
                    error=f"HTTP {response.status_code}",
                    method=method,
                    url=url,
                )
                time.sleep(sleep_s)
                continue
            response.raise_for_status()
            return response
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = exc
            if response is not None:
                response.close()
            if attempt < attempts:
                sleep_s = retry_sleep_s * attempt + random.uniform(0.0, 1.0)
                _tool_retry_debug(
                    "http_request",
                    attempt=attempt,
                    max_attempts=attempts,
                    sleep_seconds=sleep_s,
                    error=exc,
                    method=method,
                    url=url,
                )
                time.sleep(sleep_s)
                continue
            raise
        except requests.HTTPError as exc:
            last_error = exc
            if response is not None:
                response.close()
            if (
                exc.response is not None
                and _should_retry_http_status(exc.response.status_code)
                and (
                    exc.response.status_code != 429
                    or (
                        retry_on_429
                        and (
                            max_429_retries is None
                            or attempt <= max(0, int(max_429_retries))
                        )
                    )
                )
                and attempt < attempts
            ):
                retry_after = exc.response.headers.get("Retry-After") if exc.response is not None else None
                if exc.response.status_code == 429:
                    try:
                        wait_s = max(float(retry_after), retry_429_sleep_s) if retry_after else retry_429_sleep_s
                    except (TypeError, ValueError):
                        wait_s = retry_429_sleep_s
                else:
                    wait_s = retry_sleep_s * attempt
                sleep_s = wait_s + random.uniform(0.0, 1.0)
                _tool_retry_debug(
                    "http_request",
                    attempt=attempt,
                    max_attempts=attempts,
                    sleep_seconds=sleep_s,
                    error=exc,
                    method=method,
                    url=url,
                )
                time.sleep(sleep_s)
                continue
            raise
        except Exception as exc:
            last_error = exc
            if response is not None:
                response.close()
            raise
        finally:
            _release_wikimedia_request_slot(wikimedia_slot_acquired, url)
    assert last_error is not None
    raise last_error


def _search_fetch_multiplier(tool_name: str | None = None) -> int:
    """Return over-fetch multiplier for search tools.

    Defaults preserve the existing behavior (2x).  Set
    SFT_SEARCH_FETCH_MULTIPLIER for all search tools, or a tool-specific env var
    such as SFT_I2I_SEARCH_FETCH_MULTIPLIER to override only one backend.
    """

    default_multiplier = max(1, _env_int("SFT_SEARCH_FETCH_MULTIPLIER", 2))
    normalized = str(tool_name or "").strip().lower()
    if normalized == "t2t_search":
        return max(1, _env_int("SFT_T2T_SEARCH_FETCH_MULTIPLIER", default_multiplier))
    if normalized == "t2i_search":
        return max(1, _env_int("SFT_T2I_SEARCH_FETCH_MULTIPLIER", default_multiplier))
    if normalized == "i2i_search":
        return max(1, _env_int("SFT_I2I_SEARCH_FETCH_MULTIPLIER", default_multiplier))
    return default_multiplier


def _search_fetch_limit(top_k: int, *, tool_name: str | None = None) -> int:
    return max(1, int(top_k) * _search_fetch_multiplier(tool_name))


def _url_matches_blocked_domain(url: str, blocked_domains: tuple[str, ...]) -> bool:
    normalized_url = str(url or "").strip()
    if not normalized_url or not blocked_domains:
        return False
    try:
        hostname = (urlparse(normalized_url).hostname or "").lower()
    except Exception:
        return False
    if not hostname:
        return False
    return any(
        hostname == blocked_domain or hostname.endswith(f".{blocked_domain}")
        for blocked_domain in blocked_domains
    )


def _sanitize_search_query(query: str) -> str:
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return normalized_query
    sanitized = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", normalized_query)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    return sanitized or normalized_query


def _probe_content_type(url: str) -> str:
    guessed_image_type = _guess_image_content_type(url)
    if guessed_image_type:
        return guessed_image_type
    guessed_pdf_type = _guess_pdf_content_type(url)
    if guessed_pdf_type:
        return guessed_pdf_type

    try:
        response = _request_with_retry(
            "HEAD",
            url,
            allow_redirects=True,
            timeout=TOOL_NETWORK_TIMEOUT_S,
            headers=_web_request_headers(referer_url=url),
        )
        content_type = response.headers.get("Content-Type", "")
        response.close()
        if content_type:
            normalized = content_type.split(";", 1)[0].strip().lower()
            if normalized not in AMBIGUOUS_CONTENT_TYPES:
                return normalized
    except Exception:
        pass

    try:
        response = _request_with_retry(
            "GET",
            url,
            allow_redirects=True,
            stream=True,
            timeout=TOOL_NETWORK_TIMEOUT_S,
            headers=_web_request_headers(referer_url=url),
        )
        content_type = response.headers.get("Content-Type", "")
        normalized = content_type.split(";", 1)[0].strip().lower() if content_type else ""
        first_bytes = response.raw.read(64, decode_content=True)
        sniffed_content_type = _sniff_image_content_type(first_bytes) or _sniff_pdf_content_type(first_bytes)
        response.close()
        if sniffed_content_type:
            return sniffed_content_type
        if content_type:
            if normalized not in AMBIGUOUS_CONTENT_TYPES:
                return normalized
    except Exception:
        pass

    return guessed_image_type or "text/html"


def _classify_read_url_content_type(
    url: str,
    *,
    resource: UrlResource | None = None,
) -> tuple[str, str, bool]:
    """Classify a ``read_url`` target without an eager network probe.

    Search results carry a reliable resource kind, and explicit file suffixes
    are also sufficient to choose the binary path.  For every other URL we
    deliberately treat the target as a web page and let Enhanced Reader (and
    its Firecrawl fallback) perform the fetch.  Previously these URLs first
    incurred a HEAD request and often a streaming GET; each of those requests
    could independently consume all HTTP retries and turn one read into a
    burst of quota-limited requests.

    The third return value indicates whether the image branch should be used.
    Keeping that decision separate from ``content_type`` is important for
    extensionless image resources, whose URL does not provide a MIME guess.
    """

    resource_kind = str(resource.kind if resource is not None else "").strip().lower()
    guessed_image_type = _guess_image_content_type(url)
    guessed_pdf_type = _guess_pdf_content_type(url)

    if resource_kind == "image":
        return guessed_image_type, "resource_kind_image", True
    if resource_kind == "pdf":
        return guessed_pdf_type or "application/pdf", "resource_kind_pdf", False
    if guessed_image_type:
        return guessed_image_type, "url_suffix_image", True
    if guessed_pdf_type:
        return guessed_pdf_type, "url_suffix_pdf", False

    # Unknown URLs are pages by default.  This is intentionally not a MIME
    # assertion; it only selects the text-reader path and avoids a redundant
    # HEAD/GET classification round trip.
    return "text/html", "default_text_reader", False


def _download_binary(
    url: str,
    *,
    timeout: int = TOOL_NETWORK_TIMEOUT_S,
    referer_url: str | None = None,
    max_retries: int | None = None,
    retry_sleep_s: int | None = None,
    retry_on_429: bool = True,
    max_429_retries: int | None = None,
    retry_429_sleep_s: float | None = None,
) -> tuple[bytes, str]:
    effective_max_retries = TOOL_TIMEOUT_RETRIES if max_retries is None else max(0, int(max_retries))
    effective_retry_sleep_s = TOOL_RETRY_SLEEP_S if retry_sleep_s is None else max(0, int(retry_sleep_s))
    effective_max_429_retries = 1 if max_429_retries is None else max(0, int(max_429_retries))
    effective_retry_429_sleep_s = (
        TOOL_429_RETRY_SLEEP_S if retry_429_sleep_s is None else max(0.0, float(retry_429_sleep_s))
    )
    response = _request_with_retry(
        "GET",
        url,
        timeout=timeout,
        headers=_web_request_headers(referer_url=referer_url or url),
        max_retries=effective_max_retries,
        retry_sleep_s=effective_retry_sleep_s,
        retry_on_429=retry_on_429,
        max_429_retries=effective_max_429_retries,
        retry_429_sleep_s=effective_retry_429_sleep_s,
    )
    content_type = response.headers.get("Content-Type", "")
    normalized = content_type.split(";", 1)[0].strip().lower() if content_type else ""
    payload = response.content
    response.close()
    return payload, normalized


def _extract_pdf_text(pdf_path: str) -> tuple[str, str]:
    if fitz is None:
        raise RuntimeError(
            "PyMuPDF is required for PDF reading in read_url. "
            "Install it with `pip install PyMuPDF`."
        )

    text_parts: list[str] = []
    title = ""
    with fitz.open(pdf_path) as document:
        metadata = document.metadata or {}
        title = str(metadata.get("title") or "").strip()
        for page in document:
            page_text = page.get_text("text") or ""
            cleaned = page_text.strip()
            if cleaned:
                text_parts.append(cleaned)
    return "\n\n".join(text_parts).strip(), title


def _maybe_resize_downloaded_image(
    payload: bytes,
    *,
    content_type: str,
) -> tuple[bytes, str]:
    try:
        with Image.open(BytesIO(payload)) as image:
            normalized = ImageOps.exif_transpose(image)
            width, height = normalized.size
            long_edge = max(width, height)
            short_edge = min(width, height)
            if (
                long_edge <= MAX_DOWNLOADED_IMAGE_LONG_EDGE
                and short_edge <= MAX_DOWNLOADED_IMAGE_SHORT_EDGE
            ):
                return payload, content_type

            resized = normalized.copy()
            resized.thumbnail(
                (RESIZED_IMAGE_LONG_EDGE, RESIZED_IMAGE_LONG_EDGE),
                Image.Resampling.LANCZOS,
            )
            output = BytesIO()
            target_format = (normalized.format or "").upper()
            has_alpha = resized.mode in ("RGBA", "LA") or (
                resized.mode == "P" and "transparency" in resized.info
            )
            if has_alpha:
                resized.save(output, format="PNG", optimize=True)
                return output.getvalue(), "image/png"

            if target_format not in {"JPEG", "JPG", "WEBP"}:
                target_format = "JPEG"
            if resized.mode not in ("RGB", "L"):
                resized = resized.convert("RGB")
            if target_format == "WEBP":
                resized.save(output, format="WEBP", quality=90, method=6)
                return output.getvalue(), "image/webp"

            resized.save(output, format="JPEG", quality=90, optimize=True)
            return output.getvalue(), "image/jpeg"
    except Exception:
        return payload, content_type


def _read_via_jina(url: str) -> str:
    reader_url = os.environ.get("JINA_READER_URL", "https://r.jina.ai/").rstrip("/")
    headers = {"Accept": "application/json"}
    jina_api_key = os.environ.get("JINA_API_KEY")
    if jina_api_key:
        headers["Authorization"] = f"Bearer {jina_api_key}"
    response = requests.get(
        f"{reader_url}/{url}",
        headers=headers,
        timeout=TOOL_NETWORK_TIMEOUT_S,
    )
    if response.status_code != 200:
        return ""
    try:
        data = response.json()
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], dict):
                return data["data"].get("content", "")
            return data.get("content", "")
    except ValueError:
        return response.text
    return ""


def _read_document(url: str) -> dict[str, Any]:
    reader_base = (
        os.environ.get("ENHANCED_READER_URL")
        or os.environ.get("JINA_READER_URL")
        or "http://127.0.0.1:8004"
    ).rstrip("/")
    if reader_base.endswith("r.jina.ai") or "r.jina.ai" in reader_base:
        content = _read_via_jina(url)
        return {
            "url": url,
            "title": "",
            "content": content,
            "raw_markdown": content,
            "raw": {"reader": "jina_direct"},
        }

    reader = EnhancedReaderClient(
        base_url=reader_base,
        timeout_s=TOOL_NETWORK_TIMEOUT_S,
    )
    document = reader.read(url)
    return {
        "url": document.url,
        "title": document.title or "",
        "content": document.content or "",
        "raw_markdown": document.raw_markdown or "",
        "raw": document.raw,
    }


def _read_document_with_timeout_retry(url: str) -> dict[str, Any]:
    """Read via Enhanced Reader, retrying only transport timeouts.

    A blocked-page response is a valid reader response and must proceed to the
    normal Firecrawl fallback; only a timeout benefits from retrying the same
    backend.
    """

    for attempt in range(1, TOOL_TIMEOUT_RETRIES + 2):
        try:
            return _read_document(url)
        except Exception as exc:
            if not _is_timeout_error(exc) or attempt > TOOL_TIMEOUT_RETRIES:
                raise
            _read_url_debug(
                "enhanced_reader_timeout_retry",
                url=url,
                attempt=attempt,
                max_retries=TOOL_TIMEOUT_RETRIES,
                error=str(exc),
            )
            _tool_retry_debug(
                "enhanced_reader",
                attempt=attempt,
                max_attempts=TOOL_TIMEOUT_RETRIES + 1,
                sleep_seconds=TOOL_RETRY_SLEEP_S * attempt,
                error=exc,
                url=url,
            )
            time.sleep(TOOL_RETRY_SLEEP_S * attempt)
    raise AssertionError("unreachable")


def _read_via_firecrawl(url: str) -> dict[str, Any]:
    """Fetch markdown through Firecrawl only after the primary reader is blocked."""
    from synthesis.firecrawl_client import FirecrawlClient

    response = FirecrawlClient().scrape(
        url,
        only_main_content=True,
        max_age=172800000,
        parsers=["pdf"],
        formats=["markdown"],
    )
    if response.get("error"):
        raise RuntimeError(str(response["error"]))
    payload = response.get("data") if isinstance(response.get("data"), dict) else response
    markdown = str(payload.get("markdown") or "").strip()
    if not markdown:
        raise RuntimeError("Firecrawl returned no markdown content.")
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    return {
        "url": str(metadata.get("sourceURL") or metadata.get("source_url") or metadata.get("url") or url),
        "title": str(metadata.get("title") or "").strip(),
        "content": markdown,
        "metadata": metadata,
    }


def _read_via_firecrawl_with_timeout_retry(url: str) -> dict[str, Any]:
    """Use the fallback reader with the same timeout retry budget as read_url."""

    for attempt in range(1, TOOL_TIMEOUT_RETRIES + 2):
        try:
            return _read_via_firecrawl(url)
        except Exception as exc:
            if not _is_timeout_error(exc) or attempt > TOOL_TIMEOUT_RETRIES:
                raise
            _read_url_debug(
                "firecrawl_timeout_retry",
                url=url,
                attempt=attempt,
                max_retries=TOOL_TIMEOUT_RETRIES,
                error=str(exc),
            )
            _tool_retry_debug(
                "firecrawl",
                attempt=attempt,
                max_attempts=TOOL_TIMEOUT_RETRIES + 1,
                sleep_seconds=TOOL_RETRY_SLEEP_S * attempt,
                error=exc,
                url=url,
            )
            time.sleep(TOOL_RETRY_SLEEP_S * attempt)
    raise AssertionError("unreachable")


def _is_blocked_summary(content: str) -> bool:
    return "BLOCKED" in str(content or "").upper()


def _read_url_with_firecrawl_fallback(
    *,
    url: str,
    goal: str,
    assistant_output: str,
    original_title: str = "",
    trigger: str,
) -> dict[str, Any]:
    """Read through Firecrawl and summarize after Enhanced Reader is unavailable/blocked."""
    try:
        firecrawl_document = _read_via_firecrawl_with_timeout_retry(url)
        content = firecrawl_document["content"]
        summarized = summarize_with_qwen(
            content=content,
            goal=goal,
            assistant_output=assistant_output,
        )
    except Exception as exc:  # pragma: no cover - network bound
        _record_read_url_call(branch="text", success=False)
        return {
            "ok": False,
            "error": f"{trigger}; Firecrawl fallback failed for {url}: {exc}",
        }

    _read_url_debug("firecrawl_fallback_done", url=url, content_chars=len(content))
    _record_read_url_call(branch="text", success=True)
    return {
        "ok": True,
        "kind": "text",
        "url": firecrawl_document["url"],
        "title": firecrawl_document["title"] or original_title,
        "content": summarized if goal or assistant_output else content[:500],
        "resolved_via": "firecrawl_fallback",
        "firecrawl_metadata": firecrawl_document["metadata"],
    }


def read_url(
    url: str,
    goal: str = "",
    assistant_output: str = "",
    resource: UrlResource | None = None,
) -> dict[str, Any]:
    """Read a URL as either text content or a downloadable image."""

    original_url = (url or "").strip()
    normalized_url = _normalize_request_url(original_url)
    if not normalized_url:
        return {"ok": False, "error": "URL is required."}
    _read_url_debug("normalized_url", original_url=original_url, normalized_url=normalized_url)

    # Do not make a HEAD/streaming-GET round trip merely to classify the URL.
    # Search resources and explicit suffixes provide enough information for
    # the binary paths; all other URLs are sent directly to Enhanced Reader.
    # This avoids multiplying a single read into several retryable requests
    # (and is especially important for sites returning HTTP 429 to HEAD).
    content_type, content_type_source, is_image_resource = _classify_read_url_content_type(
        normalized_url,
        resource=resource,
    )
    _read_url_debug(
        "classified_content_type",
        url=normalized_url,
        content_type=content_type,
        source=content_type_source,
        probe_skipped=True,
    )
    if is_image_resource or content_type.startswith("image/"):
        image_max_retries = _read_url_image_max_retries()
        image_retry_sleep_s = _read_url_image_retry_sleep_s()
        image_max_429_retries = _read_url_429_max_retries()
        image_429_retry_sleep_s = _read_url_429_retry_sleep_s()
        _read_url_debug(
            "image_download_retry_policy",
            url=normalized_url,
            max_retries=image_max_retries,
            max_attempts=image_max_retries + 1,
            retry_sleep_s=image_retry_sleep_s,
            max_429_retries=image_max_429_retries,
            retry_429_sleep_s=image_429_retry_sleep_s,
        )
        failures: list[str] = []
        candidates = _resource_candidate_urls(resource, normalized_url)
        source_page_checked = False
        candidate_index = 0
        while candidate_index < len(candidates):
            candidate_kind, candidate_url, referer_url = candidates[candidate_index]
            candidate_index += 1
            try:
                temp_dir = tempfile.mkdtemp(prefix="synthesis_sft_read_url_")
                filename = os.path.basename(urlparse(candidate_url).path) or "downloaded_image"
                response_content, downloaded_type = _download_binary(
                    candidate_url,
                    timeout=TOOL_NETWORK_TIMEOUT_S,
                    referer_url=referer_url,
                    max_retries=image_max_retries,
                    retry_sleep_s=image_retry_sleep_s,
                    retry_on_429=True,
                    max_429_retries=image_max_429_retries,
                    retry_429_sleep_s=image_429_retry_sleep_s,
                )
                candidate_content_type = (
                    downloaded_type
                    or _sniff_image_content_type(response_content)
                    or _guess_image_content_type(candidate_url)
                    or (content_type if content_type != "text/html" else "")
                )
                if not candidate_content_type.startswith("image/"):
                    raise ValueError(f"downloaded fallback is not an image: {candidate_content_type or 'unknown'}")
                image_bytes, resolved_content_type = _maybe_resize_downloaded_image(
                    response_content,
                    content_type=candidate_content_type,
                )
                extension = mimetypes.guess_extension(resolved_content_type) or os.path.splitext(filename)[1] or ".png"
                stem = os.path.splitext(filename)[0] or "downloaded_image"
                filename = f"{stem}{extension}"
                save_path = os.path.join(temp_dir, filename)
                with open(save_path, "wb") as handle:
                    handle.write(image_bytes)
                _record_read_url_call(branch="image", success=True)
                return {
                    "ok": True,
                    "url": normalized_url,
                    "resolved_url": candidate_url,
                    "resolved_via": candidate_kind,
                    "content_type": resolved_content_type,
                    "local_path": save_path,
                }
            except Exception as exc:  # pragma: no cover - network bound
                failures.append(f"{candidate_kind}:{candidate_url}: {exc}")
                if not source_page_checked and candidate_index >= len(candidates):
                    source_page_checked = True
                    seen_urls = {url for _, url, _ in candidates}
                    for page_image_url in _source_page_image_candidates(
                        resource,
                        max_retries=image_max_retries,
                        max_429_retries=image_max_429_retries,
                    ):
                        if page_image_url in seen_urls:
                            continue
                        seen_urls.add(page_image_url)
                        candidates.append(
                            (
                                "source_page_image",
                                page_image_url,
                                resource.source_page_url if resource is not None else None,
                            )
                        )
                continue
        final_error = failures[-1] if failures else "no download candidates"
        _record_read_url_call(
            branch="image",
            success=False,
            image_http_status=None,
        )
        return {
            "ok": False,
            "error": f"read_url failed for {normalized_url}: {final_error}",
            "fallback_errors": failures,
        }

    if content_type in PDF_CONTENT_TYPES:
        try:
            _read_url_debug("pdf_branch_start", url=normalized_url, content_type=content_type)
            temp_dir = tempfile.mkdtemp(prefix="synthesis_sft_read_pdf_")
            filename = os.path.basename(urlparse(normalized_url).path) or "downloaded.pdf"
            if not filename.lower().endswith(".pdf"):
                filename = f"{os.path.splitext(filename)[0] or 'downloaded'}.pdf"
            save_path = os.path.join(temp_dir, filename)
            _read_url_debug("pdf_download_start", url=normalized_url, save_path=save_path)
            pdf_bytes, _ = _download_binary(
                normalized_url,
                timeout=TOOL_NETWORK_TIMEOUT_S,
                referer_url=resource.source_page_url if resource is not None else None,
            )
            _read_url_debug(
                "pdf_download_done",
                url=normalized_url,
                byte_count=len(pdf_bytes),
                startswith_pdf=pdf_bytes.startswith(b"%PDF-"),
            )
            with open(save_path, "wb") as handle:
                handle.write(pdf_bytes)
            _read_url_debug("pdf_extract_start", path=save_path)
            content, title = _extract_pdf_text(save_path)
            _read_url_debug(
                "pdf_extract_done",
                path=save_path,
                title=title,
                text_chars=len(content),
                text_preview=content[:200].replace("\n", " "),
            )
            if not content:
                _record_read_url_call(branch="pdf", success=False)
                _read_url_debug("pdf_extract_empty", path=save_path)
                return {
                    "ok": False,
                    "error": f"read_url failed for {normalized_url}: PDF text extraction returned empty content.",
                }
            summarized_content = (
                summarize_with_qwen(
                    content=content,
                    goal=goal,
                    assistant_output=assistant_output,
                )
                if goal or assistant_output
                else content[:500]
            )
            _record_read_url_call(branch="pdf", success=True)
            return {
                "ok": True,
                "kind": "text",
                "url": normalized_url,
                "title": title or filename,
                "content": summarized_content,
                "content_type": "application/pdf",
            }
        except Exception as exc:  # pragma: no cover - network bound
            _read_url_debug(
                "pdf_direct_failed",
                url=normalized_url,
                error_type=exc.__class__.__name__,
                error=str(exc),
            )
            # A browser-visible PDF may still be unreachable from the inference
            # host. Fall back to the configured reader, which may use a
            # different network path and can return extracted text directly.
            try:
                _read_url_debug("pdf_reader_fallback_start", url=normalized_url)
                document = _read_document_with_timeout_retry(normalized_url)
                content = str(document.get("content") or "").strip()
                _read_url_debug(
                    "pdf_reader_fallback_done",
                    url=normalized_url,
                    title=document.get("title"),
                    content_chars=len(content),
                    content_preview=content[:200].replace("\n", " "),
                )
                if content:
                    summarized_content = (
                        summarize_with_qwen(
                            content=content,
                            goal=goal,
                            assistant_output=assistant_output,
                        )
                        if goal or assistant_output
                        else content[:500]
                    )
                    _record_read_url_call(branch="pdf", success=True)
                    return {
                        "ok": True,
                        "kind": "text",
                        "url": normalized_url,
                        "title": str(document.get("title") or os.path.basename(urlparse(normalized_url).path)),
                        "content": summarized_content,
                        "content_type": "application/pdf",
                        "resolved_via": "reader_fallback",
                    }
            except Exception as fallback_exc:
                _read_url_debug(
                    "pdf_reader_fallback_failed",
                    url=normalized_url,
                    error_type=fallback_exc.__class__.__name__,
                    error=str(fallback_exc),
                )
            _record_read_url_call(branch="pdf", success=False)
            return {"ok": False, "error": f"read_url failed for {normalized_url}: {exc}"}

    try:
        # print(f"############ {normalized_url} ##############")
        document = _read_document_with_timeout_retry(normalized_url)
        # print(f"############ {document} ##############")
    except Exception as exc:  # pragma: no cover - network bound
        trigger = f"Enhanced Reader failed for {normalized_url}: {exc}"
        _read_url_debug(
            "enhanced_reader_failed",
            url=normalized_url,
            error_type=exc.__class__.__name__,
            error=str(exc),
        )
        return _read_url_with_firecrawl_fallback(
            url=normalized_url,
            goal=goal,
            assistant_output=assistant_output,
            trigger=trigger,
        )

    content = document.get("content", "") or ""
    title = document.get("title", "") or ""
    should_return_summary = bool(goal or assistant_output)
    reader_summary = summarize_with_qwen(
        content=content,
        goal=goal,
        assistant_output=assistant_output,
    )
    summarized_content = reader_summary if should_return_summary else content[:500]
    if _is_blocked_summary(reader_summary):
        _read_url_debug("enhanced_reader_blocked", url=normalized_url)
        return _read_url_with_firecrawl_fallback(
            url=normalized_url,
            goal=goal,
            assistant_output=assistant_output,
            original_title=title,
            trigger=f"Enhanced Reader returned BLOCKED for {normalized_url}",
        )
    _record_read_url_call(branch="text", success=True)
    result = {
        "ok": True,
        "kind": "text",
        "url": document.get("url") or normalized_url,
        "title": title,
        "content": summarized_content,
        "resolved_via": "enhanced_reader",
    }
    return result


def _run_search_with_timeout_retry(
    tool_name: str,
    request: Callable[[], Any],
) -> Any:
    """Run a search backend request with bounded timeout-only retries."""

    for attempt in range(1, TOOL_TIMEOUT_RETRIES + 2):
        try:
            return request()
        except Exception as exc:
            if not _is_timeout_error(exc) or attempt > TOOL_TIMEOUT_RETRIES:
                raise
            _tool_retry_debug(
                tool_name,
                attempt=attempt,
                max_attempts=TOOL_TIMEOUT_RETRIES + 1,
                sleep_seconds=TOOL_RETRY_SLEEP_S * attempt,
                error=exc,
            )
            time.sleep(TOOL_RETRY_SLEEP_S * attempt)
    raise AssertionError("unreachable")


def t2t_search(query: str, lang: str = "en", top_k: int = DEFAULT_SEARCH_TOP_K) -> dict[str, Any]:
    """Search text pages and return search results only."""
    try:
        top_k = max(1, min(int(top_k), MAX_SEARCH_RESULTS))
        fetch_limit = _search_fetch_limit(top_k, tool_name="t2t_search")
        effective_query = _sanitize_search_query(query)
        response = _run_search_with_timeout_retry(
            "t2t_search",
            lambda: _serper_client().search_text(
                effective_query,
                limit=fetch_limit,
                hl=lang,
            ),
        )
        results: list[dict[str, Any]] = []
        for item in response.results:
            if _url_matches_blocked_domain(item.url or "", T2T_BLOCKED_SEARCH_DOMAINS):
                continue
            results.append(
                {
                    "title": item.title or "",
                    "url": item.url or "",
                    "snippet": item.snippet or "",
                    "rank": item.rank,
                }
            )
            if len(results) >= top_k:
                break
        _record_search_tool_call("t2t_search", success=True, result_count=len(results))
        return {
            "ok": True,
            "query": query,
            "results": results,
        }
    except Exception:
        _record_search_tool_call("t2t_search", success=False, result_count=0)
        raise


def t2i_search(query: str, lang: str = "en", top_k: int = DEFAULT_SEARCH_TOP_K) -> dict[str, Any]:
    """Search images from a text query."""
    try:
        top_k = max(1, min(int(top_k), MAX_SEARCH_RESULTS))
        fetch_limit = _search_fetch_limit(top_k, tool_name="t2i_search")
        effective_query = _sanitize_search_query(query)
        response = _run_search_with_timeout_retry(
            "t2i_search",
            lambda: _serper_client().search_image(
                effective_query,
                limit=fetch_limit,
                hl=lang,
            ),
        )
        results: list[dict[str, Any]] = []
        for item in response.results:
            if _url_matches_blocked_domain(item.image_url or "", T2I_BLOCKED_IMAGE_SEARCH_DOMAINS):
                continue
            if _url_matches_blocked_domain(item.source_page_url or "", T2I_BLOCKED_IMAGE_SEARCH_DOMAINS):
                continue
            results.append(
                {
                    "title": item.title,
                    "image_url": item.image_url,
                    "thumbnail_url": item.thumbnail_url,
                    "source_page_url": item.source_page_url,
                    "snippet": item.snippet,
                    "rank": item.rank,
                }
            )
            if len(results) >= top_k:
                break
        _record_search_tool_call("t2i_search", success=True, result_count=len(results))
        return {
            "ok": True,
            "query": query,
            "results": results,
        }
    except Exception:
        _record_search_tool_call("t2i_search", success=False, result_count=0)
        raise


def _image_search_via_serper(image_url: str, top_k: int = MAX_SEARCH_RESULTS) -> object:
    serper_api_key, _ = acquire_serper_api_key()
    fetch_limit = _search_fetch_limit(max(1, min(int(top_k), MAX_SEARCH_RESULTS)), tool_name="i2i_search")

    lens_headers = {
        "X-API-KEY": serper_api_key,
        "Content-Type": "application/json",
    }
    relay_token = os.environ.get("SERPER_RELAY_TOKEN")
    if relay_token:
        lens_headers["X-Serper-Relay-Token"] = relay_token
    response = requests.post(
        os.environ.get("SERPER_LENS_URL") or "https://google.serper.dev/lens",
        headers=lens_headers,
        json={"url": image_url},
        timeout=TOOL_NETWORK_TIMEOUT_S,
    )
    response.raise_for_status()
    data = response.json()
    organic = data.get("organic", []) or []
    if not organic:
        return data

    results = []
    for item in organic[:fetch_limit]:
        results.append(
            {
                "title": item.get("title", ""),
                "source": item.get("source", "") or item.get("link", ""),
                "link": item.get("link", ""),
                "imageUrl": item.get("imageUrl", ""),
                "thumbnailUrl": item.get("thumbnailUrl", ""),
            }
        )
    return results


def i2i_search(
    image_url: str,
    visual_lookup: Callable[..., object] | None = None,
    top_k: int = DEFAULT_SEARCH_TOP_K,
    max_retries: int = TOOL_TIMEOUT_RETRIES,
    base_delay: int = TOOL_RETRY_SLEEP_S,
) -> dict[str, Any]:
    """Reverse-image search using a provided backend or Serper Lens.

    A successful Lens response with no matches can be transient when the
    remote image URL is temporarily inaccessible to the search backend.  Retry
    that specific case once after a short fixed delay, while retaining the
    timeouts are retried at most ``max_retries`` times after the initial
    request.  Other backend errors are returned immediately.
    """

    visual_lookup = visual_lookup or _image_search_via_serper
    top_k = max(1, min(int(top_k), MAX_SEARCH_RESULTS))
    last_error: Exception | None = None
    retried_empty_result = False
    attempt = 1
    attempts_made = 0
    max_attempts = 1 + max(0, int(max_retries))
    while attempt <= max_attempts:
        attempts_made = attempt
        try:
            result = visual_lookup(image_url=image_url, top_k=top_k)
            if isinstance(result, dict) and "error" in result:
                raise RuntimeError(str(result["error"]))
            matches = summarize_image_search(result)
            if isinstance(matches, list):
                matches = [
                    item
                    for item in matches
                    if not (
                        _url_matches_blocked_domain(item.get("source") or "", I2I_BLOCKED_IMAGE_SEARCH_DOMAINS)
                        or _url_matches_blocked_domain(item.get("link") or "", I2I_BLOCKED_IMAGE_SEARCH_DOMAINS)
                        or _url_matches_blocked_domain(item.get("imageUrl") or "", I2I_BLOCKED_IMAGE_SEARCH_DOMAINS)
                        or _url_matches_blocked_domain(item.get("thumbnailUrl") or "", I2I_BLOCKED_IMAGE_SEARCH_DOMAINS)
                    )
                ]
            result_count = len(matches) if isinstance(matches, list) else (1 if matches else 0)
            if result_count == 0 and not retried_empty_result:
                retried_empty_result = True
                _tool_retry_debug(
                    "i2i_search",
                    attempt=attempt,
                    max_attempts=max_attempts,
                    sleep_seconds=5,
                    error="empty result",
                    retry_reason="empty_result",
                )
                time.sleep(5)
                continue
            _record_search_tool_call("i2i_search", success=True, result_count=result_count)
            return {
                "ok": True,
                "top_k": top_k,
                "matches": matches[:top_k] if isinstance(matches, list) else matches,
            }
        except Exception as exc:  # pragma: no cover - network bound
            last_error = exc
            if _is_timeout_error(exc) and attempt < max_attempts:
                sleep_s = base_delay * (2 ** (attempt - 1))
                _tool_retry_debug(
                    "i2i_search",
                    attempt=attempt,
                    max_attempts=max_attempts,
                    sleep_seconds=sleep_s,
                    error=exc,
                    retry_reason="timeout",
                )
                time.sleep(sleep_s)
            elif not _is_timeout_error(exc):
                break
            attempt += 1
    _record_search_tool_call("i2i_search", success=False, result_count=0)
    return {
        "ok": False,
        "top_k": top_k,
        "error": (
            f"i2i_search failed after {attempts_made} attempt(s) "
            f"with at most {max_retries} timeout retries: {last_error}"
        ),
    }
