"""Tool schemas and backend implementations for synthesis SFT agents."""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import re
import sys
import tempfile
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import requests
from PIL import Image, ImageOps

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "synthesis.sft"

from synthesis.search_client import SerperSearchClient
from synthesis.wiki_text_builder import EnhancedReaderClient


logger = logging.getLogger(__name__)
MAX_SEARCH_RESULTS = 5
MAX_DOWNLOADED_IMAGE_LONG_EDGE = 1920
MAX_DOWNLOADED_IMAGE_SHORT_EDGE = 1080
RESIZED_IMAGE_LONG_EDGE = 1200
AMBIGUOUS_CONTENT_TYPES = {
    "",
    "application/octet-stream",
    "binary/octet-stream",
    "text/plain",
}
T2I_BLOCKED_IMAGE_SEARCH_DOMAINS = (
    "facebook.com",
    "m.facebook.com",
    "lookaside.fbsbx.com",
    "fbsbx.com",
)


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


def get_tool_definitions() -> list[dict[str, Any]]:
    """Return OpenAI-compatible function tool definitions."""

    return [
        {
            "type": "function",
            "function": {
                "name": "t2t_search",
                "description": (
                    "Search text/web documents on Google from a text query. Returns "
                    "search results such as title, url, and snippet. If the "
                    "agent wants the full content of a result, it should call "
                    "read_url separately."
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
                "description": "Search images from a text query on Google.",
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
                    "Search for visually similar or matching images from the most recent image in the current context."
                    "You can locate the bounding box of the entity you want to recognize in the image and then search that region separately as an image."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "region": {
                            "type": "array",
                            "description": (
                                "Optional bounding box on the current image to crop before search. The relative coordinates between 0~1000 of the region containing an unfamiliar/task-related person, logo, object, or other entity are recommended. If this parameter is not provided, the entire image will be searched."
                                "Preferred format: [x1, y1, x2, y2]."
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
                    "relevant to the given query. If it returns an image, the image will be downloaded for you. "
                    "NOTICE, only the URLs you have got from search tools can be read. Wikipedia and Wiki commons is excluded for safety reasons."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to read.",
                        },
                        "query": {
                            "type": "string",
                            "description": "Optional query for focused summarization.",
                            "default": "",
                        },
                    },
                    "required": ["url"],
                },
            },
        },
    ]


def get_responses_tool_definitions() -> list[dict[str, Any]]:
    """Return Responses-API-compatible function tool definitions."""

    definitions: list[dict[str, Any]] = []
    for item in get_tool_definitions():
        function_block = dict(item["function"])
        definitions.append(
            {
                "type": "function",
                "name": function_block["name"],
                "function": function_block,
            }
        )
    return definitions


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
    if name == "read_url" and "URL" in params and "url" not in params:
        params["url"] = params.pop("URL")
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
        api_key=os.environ.get("SERPER_API_KEY"),
        search_url=os.environ.get("SERPER_SEARCH_URL") or "https://google.serper.dev/search",
        images_url=os.environ.get("SERPER_IMAGES_URL") or "https://google.serper.dev/images",
        timeout_s=float(os.environ.get("SFT_SERPER_TIMEOUT_S", "60")),
    )


def _openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "synthesis.sft.tools requires the `openai` package for summarization."
        ) from exc

    base_url = os.environ.get("SFT_SUMMARIZER_API_BASE") or os.environ.get("QWEN_API_BASE")
    api_key = (
        os.environ.get("SFT_SUMMARIZER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("QWEN_API_KEY")
        or "EMPTY"
    )
    timeout_s = float(os.environ.get("SFT_SUMMARIZER_TIMEOUT_S", "60"))
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)


def _summarizer_model() -> str:
    return (
        os.environ.get("SFT_SUMMARIZER_MODEL")
        or os.environ.get("QWEN_MODEL_NAME")
        or "Qwen/Qwen3-32B"
    )


def summarize_with_qwen(
    content: str,
    query: str,
    title: str,
    question_text: str = "",
    assistant_output: str = "",
) -> str:
    """Compress webpage content while preserving information relevant to the current task."""

    prompt = (
        "You are cleaning and compressing raw webpage content for a multi-hop question-answering agent.\n\n"
        "Your job is NOT to write a short abstract. Your job is to produce a cleaned, compressed version of the "
        "page that preserves all information relevant to the current question and the agent's current reasoning step, "
        "while aggressively removing noise.\n\n"
        "Compression rules:\n"
        "1. Keep all facts that may help answer the current question or the agent's current sub-goal.\n"
        "2. Preserve important details exactly when possible, including names, dates, numbers, relationships, quotes, "
        "titles, roles, locations, and qualifiers.\n"
        "3. If a paragraph is relevant, keep its meaning complete. Do not over-compress it into vague statements.\n"
        "4. Remove noise completely when it is not relevant: navigation text, menus, repeated headers, footer text, "
        "social buttons, unrelated recommendations, boilerplate, tracking text, and raw URL lists.\n"
        "5. If a section is only partially relevant, keep the relevant sentences and shorten the rest.\n"
        "6. Do not add outside knowledge. Do not infer missing facts. Do not rewrite the article into bullet points "
        "unless the source itself is structured that way.\n"
        "7. Output only the cleaned article text.\n\n"
        f"Original question:\n{question_text or query}\n\n"
        f"Agent's current output:\n{assistant_output or '(empty)'}\n\n"
        f"Focused query for this URL:\n{query or '(empty)'}\n\n"
        f"Webpage title:\n{title or '(untitled)'}\n\n"
        f"Raw webpage content:\n{content[:10000]}\n"
    )
    try:
        completion = _openai_client().chat.completions.create(
            model=_summarizer_model(),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8192,
            temperature=0.3,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        choice = completion.choices[0]
        content_text = choice.message.content or ""
        if content_text:
            return content_text.strip()
    except Exception as exc:  # pragma: no cover - network bound
        logger.warning("Summarization failed: %s", exc)
    return content[:1000] + ("..." if len(content) > 1000 else "")


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


def _guess_image_from_url(url: str) -> bool:
    guessed_type, _ = mimetypes.guess_type(urlparse(url).path)
    return bool(guessed_type and guessed_type.startswith("image/"))


def _guess_image_content_type(url: str) -> str:
    guessed_type, _ = mimetypes.guess_type(urlparse(url).path)
    if guessed_type and guessed_type.startswith("image/"):
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

def _url_matches_blocked_domain(url: str) -> bool:
    normalized_url = str(url or "").strip()
    if not normalized_url:
        return False
    try:
        hostname = (urlparse(normalized_url).hostname or "").lower()
    except Exception:
        return False
    if not hostname:
        return False
    return any(
        hostname == blocked_domain or hostname.endswith(f".{blocked_domain}")
        for blocked_domain in T2I_BLOCKED_IMAGE_SEARCH_DOMAINS
    )


def _sanitize_t2i_query(query: str) -> str:
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

    try:
        response = requests.head(
            url,
            allow_redirects=True,
            timeout=20,
            headers=_web_request_headers(referer_url=url),
        )
        content_type = response.headers.get("Content-Type", "")
        if content_type:
            normalized = content_type.split(";", 1)[0].strip().lower()
            if normalized not in AMBIGUOUS_CONTENT_TYPES:
                return normalized
    except Exception:
        pass

    try:
        response = requests.get(
            url,
            allow_redirects=True,
            stream=True,
            timeout=20,
            headers=_web_request_headers(referer_url=url),
        )
        content_type = response.headers.get("Content-Type", "")
        normalized = content_type.split(";", 1)[0].strip().lower() if content_type else ""
        sniffed_content_type = _sniff_image_content_type(response.raw.read(64, decode_content=True))
        response.close()
        if sniffed_content_type:
            return sniffed_content_type
        if content_type:
            if normalized not in AMBIGUOUS_CONTENT_TYPES:
                return normalized
    except Exception:
        pass

    return guessed_image_type or "text/html"


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
    response = requests.get(f"{reader_url}/{url}", headers=headers, timeout=30)
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
        timeout_s=float(os.environ.get("ENHANCED_READER_TIMEOUT_S", "180")),
    )
    document = reader.read(url)
    return {
        "url": document.url,
        "title": document.title or "",
        "content": document.content or "",
        "raw_markdown": document.raw_markdown or "",
        "raw": document.raw,
    }


def read_url(
    url: str,
    query: str = "",
    question_text: str = "",
    assistant_output: str = "",
) -> dict[str, Any]:
    """Read a URL as either text content or a downloadable image."""

    normalized_url = (url or "").strip()
    if not normalized_url:
        return {"ok": False, "error": "URL is required."}
    if not normalized_url.startswith(("http://", "https://")):
        normalized_url = f"https://{normalized_url}"

    content_type = _probe_content_type(normalized_url)
    if content_type.startswith("image/"):
        try:
            temp_dir = tempfile.mkdtemp(prefix="synthesis_sft_read_url_")
            filename = os.path.basename(urlparse(normalized_url).path) or "downloaded_image"
            response = requests.get(
                normalized_url,
                timeout=60,
                headers=_web_request_headers(referer_url=normalized_url),
            )
            response.raise_for_status()
            image_bytes, resolved_content_type = _maybe_resize_downloaded_image(
                response.content,
                content_type=content_type,
            )
            extension = mimetypes.guess_extension(resolved_content_type) or os.path.splitext(filename)[1] or ".png"
            stem = os.path.splitext(filename)[0] or "downloaded_image"
            filename = f"{stem}{extension}"
            save_path = os.path.join(temp_dir, filename)
            with open(save_path, "wb") as handle:
                handle.write(image_bytes)
            return {
                "ok": True,
                "url": normalized_url,
                "content_type": resolved_content_type,
                "local_path": save_path,
            }
        except Exception as exc:  # pragma: no cover - network bound
            return {"ok": False, "error": f"read_url failed for {normalized_url}: {exc}"}

    try:
        document = _read_document(normalized_url)
    except Exception as exc:  # pragma: no cover - network bound
        return {"ok": False, "error": f"read_url failed for {normalized_url}: {exc}"}

    content = document.get("content", "") or ""
    title = document.get("title", "") or ""
    summarized_content = (
        summarize_with_qwen(
            content=content,
            query=query,
            title=title,
            question_text=question_text,
            assistant_output=assistant_output,
        )
        if query or question_text or assistant_output
        else content[:500]
    )
    return {
        "ok": True,
        "kind": "text",
        "url": document.get("url") or normalized_url,
        "title": title,
        "content": summarized_content,
    }


def t2t_search(query: str, lang: str = "en", top_k: int = 5) -> dict[str, Any]:
    """Search text pages and return search results only."""

    top_k = max(1, min(int(top_k), MAX_SEARCH_RESULTS))
    response = _serper_client().search_text(query, limit=top_k, hl=lang)
    results: list[dict[str, Any]] = []
    for item in response.results[:top_k]:
        results.append(
            {
                "title": item.title or "",
                "url": item.url or "",
                "snippet": item.snippet or "",
                "rank": item.rank,
            }
        )
    return {
        "ok": True,
        "query": query,
        "results": results,
    }


def t2i_search(query: str, lang: str = "en", top_k: int = 5) -> dict[str, Any]:
    """Search images from a text query."""

    top_k = max(1, min(int(top_k), MAX_SEARCH_RESULTS))
    fetch_limit = min(max(top_k * 3, top_k), 20)
    effective_query = _sanitize_t2i_query(query)
    response = _serper_client().search_image(effective_query, limit=fetch_limit, hl=lang)
    results: list[dict[str, Any]] = []
    for item in response.results:
        if _url_matches_blocked_domain(item.image_url or ""):
            continue
        if _url_matches_blocked_domain(item.source_page_url or ""):
            continue
        results.append(
            {
                "title": item.title,
                "image_url": item.image_url,
                "source_page_url": item.source_page_url,
                "snippet": item.snippet,
                "rank": item.rank,
            }
        )
        if len(results) >= top_k:
            break
    return {
        "ok": True,
        "query": query,
        "results": results,
    }


def _image_search_via_serper(image_url: str, top_k: int = MAX_SEARCH_RESULTS) -> object:
    serper_api_key = os.environ.get("SERPER_API_KEY")
    if not serper_api_key:
        raise RuntimeError("SERPER_API_KEY is required for reverse image search.")

    response = requests.post(
        os.environ.get("SERPER_LENS_URL") or "https://google.serper.dev/lens",
        headers={
            "X-API-KEY": serper_api_key,
            "Content-Type": "application/json",
        },
        json={"url": image_url},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    organic = data.get("organic", []) or []
    if not organic:
        return data

    results = []
    top_k = max(1, min(int(top_k), MAX_SEARCH_RESULTS))
    for item in organic[:top_k]:
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
    top_k: int = MAX_SEARCH_RESULTS,
    max_retries: int = 3,
    base_delay: int = 2,
) -> dict[str, Any]:
    """Reverse-image search using a provided backend or Serper Lens."""

    visual_lookup = visual_lookup or _image_search_via_serper
    top_k = max(1, min(int(top_k), MAX_SEARCH_RESULTS))
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            result = visual_lookup(image_url=image_url, top_k=top_k)
            if isinstance(result, dict) and "error" in result:
                raise RuntimeError(str(result["error"]))
            matches = summarize_image_search(result)
            return {
                "ok": True,
                "image_url": image_url,
                "top_k": top_k,
                "matches": matches[:top_k] if isinstance(matches, list) else matches,
            }
        except Exception as exc:  # pragma: no cover - network bound
            last_error = exc
            if attempt < max_retries:
                time.sleep(base_delay * (2 ** (attempt - 1)))
    return {
        "ok": False,
        "image_url": image_url,
        "top_k": top_k,
        "error": f"i2i_search failed after {max_retries} retries: {last_error}",
    }
