"""Search and document-parsing helpers used by the agent's tools.

Two providers are supported for ``text_search``:

1. **Gateway mode** (``API_HOST`` + ``API_USER`` + ``API_KEY``): a single
   gateway proxies Serper and Jina behind one HMAC credential.
2. **Direct mode** (``SERPER_API_KEY`` + ``JINA_API_KEY``): the public
   Serper / Jina endpoints are called directly.

Per-page summarisation is delegated to an OpenAI-compatible chat
completion endpoint serving ``QWEN_MODEL_NAME`` (typically Qwen3-32B).
``image_search`` requires an external visual lookup function
(historically named ``lens_scan``); when the function is not provided
the tool returns a clear, recoverable error message instead of raising.
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import re
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

import requests

from . import config
from . import image_io
from synthesis.search_client import acquire_serper_api_key


logger = logging.getLogger(__name__)
T2T_BLOCKED_SEARCH_DOMAINS = ("wikipedia.org",)


def _build_serper_client():
    """Create the shared Serper client from synthesis' backend wrapper."""

    from synthesis.search_client import SerperSearchClient

    return SerperSearchClient(
        api_key=config.SERPER_API_KEY or None,
        search_url=config.SERPER_SEARCH_URL,
        images_url=config.SERPER_IMAGES_URL,
        timeout_s=60.0,
    )


def _format_search_results(results: list[dict[str, Any]]) -> str:
    return json.dumps(results, ensure_ascii=False, indent=2)


def _url_matches_blocked_domain(url: str, blocked_domains: tuple[str, ...]) -> bool:
    normalized_url = str(url or "").strip()
    if not normalized_url or not blocked_domains:
        return False
    try:
        hostname = (urlparse(normalized_url).hostname or "").lower()
    except Exception:
        return False
    return any(
        hostname == domain or hostname.endswith(f".{domain}")
        for domain in blocked_domains
    )


def _guess_image_from_url(url: str) -> bool:
    guessed_type, _ = mimetypes.guess_type(urlparse(url).path)
    return bool(guessed_type and guessed_type.startswith("image/"))


def _probe_content_type(url: str) -> str:
    try:
        response = requests.head(url, allow_redirects=True, timeout=20)
        content_type = response.headers.get("Content-Type", "")
        if content_type:
            return content_type.split(";", 1)[0].strip().lower()
    except Exception:
        pass

    try:
        response = requests.get(url, allow_redirects=True, stream=True, timeout=20)
        content_type = response.headers.get("Content-Type", "")
        response.close()
        if content_type:
            return content_type.split(";", 1)[0].strip().lower()
    except Exception:
        pass

    if _guess_image_from_url(url):
        guessed_type, _ = mimetypes.guess_type(urlparse(url).path)
        return guessed_type or "image/*"
    return ""


def _enhanced_reader_read(url: str) -> dict[str, Any]:
    from synthesis.wiki_text_builder import EnhancedReaderClient

    reader = EnhancedReaderClient(
        base_url=config.ENHANCED_READER_URL,
        timeout_s=config.ENHANCED_READER_TIMEOUT_S,
    )
    document = reader.read(url)
    return {
        "url": document.url,
        "title": document.title or "",
        "content": document.content or "",
        "raw_markdown": document.raw_markdown or "",
        "raw": document.raw,
    }


def _read_document(url: str) -> dict[str, Any]:
    reader_base = config.ENHANCED_READER_URL.rstrip("/")
    if reader_base.endswith("r.jina.ai") or "r.jina.ai" in reader_base:
        content = _read_via_jina(url)
        return {
            "url": url,
            "title": "",
            "content": content,
            "raw_markdown": content,
            "raw": {"reader": "jina_direct"},
        }
    print(f"########## DEBUG: {url}")
    return _enhanced_reader_read(url)


# ---------------------------------------------------------------------------
# Layout parsing
# ---------------------------------------------------------------------------


def layout_parsing(
    file_path: str,
    use_chart_recognition: bool = False,
    use_doc_orientation_classify: bool = False,
    use_doc_unwarping: bool = False,
) -> Dict:
    """POST a local image to the configured layout-parsing endpoint."""

    if not config.LAYOUT_PARSING_API_URL:
        return {
            "error": (
                "Layout parsing endpoint is not configured. "
                "Set LAYOUT_PARSING_API_URL (and optionally LAYOUT_PARSING_TOKEN)."
            ),
            "rec_texts": [],
            "formatted_text": "",
            "blocks": [],
        }

    if not os.path.exists(file_path):
        return {
            "error": f"File not found: {file_path}",
            "rec_texts": [],
            "formatted_text": "",
            "blocks": [],
        }

    try:
        with open(file_path, "rb") as fh:
            file_data = base64.b64encode(fh.read()).decode("ascii")

        headers = {"Content-Type": "application/json"}
        if config.LAYOUT_PARSING_TOKEN:
            headers["Authorization"] = f"token {config.LAYOUT_PARSING_TOKEN}"

        payload = {
            "file": file_data,
            "fileType": 1,
            "useDocOrientationClassify": use_doc_orientation_classify,
            "useDocUnwarping": use_doc_unwarping,
            "useChartRecognition": use_chart_recognition,
        }

        response = requests.post(
            config.LAYOUT_PARSING_API_URL,
            json=payload,
            headers=headers,
            timeout=60,
        )
        if response.status_code != 200:
            return {
                "error": (
                    f"API request failed with status {response.status_code}: "
                    f"{response.text[:200]}"
                ),
                "rec_texts": [],
                "formatted_text": "",
                "blocks": [],
            }

        result = response.json().get("result", {})
        text_labels = {"paragraph_title", "text", "vision_footnote"}
        blocks = (
            result.get("layoutParsingResults", [{}])[0]
            .get("prunedResult", {})
            .get("parsing_res_list", [])
        )
        texts = [
            blk["block_content"].strip()
            for blk in blocks
            if blk.get("block_label") in text_labels
            and blk.get("block_content", "").strip()
        ]
        return {
            "rec_texts": texts,
            "formatted_text": "\n".join(texts),
            "blocks": blocks,
            "error": None,
        }
    except Exception as exc:  # pragma: no cover - network-bound
        return {
            "error": f"Layout parsing error: {exc}",
            "rec_texts": [],
            "formatted_text": "",
            "blocks": [],
        }


# ---------------------------------------------------------------------------
# Summarisation backbone
# ---------------------------------------------------------------------------


def summarize_with_qwen(content: str, query: str, title: str) -> str:
    """Generate a short, query-focused summary for a single page."""

    prompt = (
        f"Based on the following webpage content, provide a concise summary "
        f"that is relevant to the query: \"{query}\"\n\n"
        f"Webpage Title: {title}\n"
        f"Content:\n{content[:2000]}\n\n"
        f"Please provide a focused summary (2-4 sentences) that directly "
        f"addresses the query. Focus on the most relevant information."
    )
    payload = {
        "model": config.QWEN_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.3,
        "top_p": 0.95,
        "extra_body": {
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }
    try:
        response = requests.post(
            f"{config.QWEN_API_BASE.rstrip('/')}/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        if response.status_code == 200:
            data = response.json()
            choices = data.get("choices", [])
            if choices:
                summary = choices[0].get("message", {}).get("content", "")
                if summary:
                    return summary.strip()
    except Exception as exc:
        logger.warning("Qwen summarization failed: %s", exc)
    return content[:500] + ("..." if len(content) > 500 else "")


def summarize_image_search(result_obj: object) -> object:
    """Reduce a raw image-search payload to ``{title, source}`` records."""

    try:
        result_str = json.dumps(result_obj, ensure_ascii=False, indent=2)
    except Exception:
        result_str = str(result_obj)

    prompt = (
        "You are processing image search results. Extract and summarize only "
        "the relevant \"title\" and \"source\" information from the following "
        "image search results. Remove all irrelevant information and keep only "
        "the essential identification details.\n\n"
        f"Image Search Results:\n{result_str[:3000]}\n\n"
        "Please extract and return ONLY the relevant information in JSON "
        "format with \"title\" and \"source\" fields. If there are multiple "
        "results, return a list of objects, each with \"title\" and \"source\" "
        "fields. Remove any irrelevant details, descriptions, or metadata that "
        "are not directly related to identifying the object/entity."
    )
    payload = {
        "model": config.QWEN_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.3,
        "top_p": 0.95,
        "extra_body": {
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }
    try:
        response = requests.post(
            f"{config.QWEN_API_BASE.rstrip('/')}/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        if response.status_code == 200:
            choices = response.json().get("choices", [])
            if choices:
                text = choices[0].get("message", {}).get("content", "")
                if text:
                    match = re.search(
                        r"```(?:json)?\s*(\{.*?\}|\[.*?\])", text, re.DOTALL
                    )
                    try:
                        if match:
                            return json.loads(match.group(1))
                        return json.loads(text.strip())
                    except json.JSONDecodeError:
                        return {"summary": text.strip()}
    except Exception as exc:
        logger.warning("Qwen image-search summarization failed: %s", exc)

    if isinstance(result_obj, dict):
        filtered: Dict[str, object] = {}
        for src_key, dst_key in (
            ("title", "title"),
            ("name", "title"),
            ("label", "title"),
            ("entity", "title"),
            ("source", "source"),
            ("url", "source"),
            ("link", "source"),
            ("reference", "source"),
        ):
            if src_key in result_obj and dst_key not in filtered:
                filtered[dst_key] = result_obj[src_key]
        return filtered or result_obj
    return result_obj


# ---------------------------------------------------------------------------
# Search providers
# ---------------------------------------------------------------------------


def _search_via_gateway(query: str, lang: str, top_k: int) -> List[dict]:
    headers = {
        "Authorization": f"Bearer {config.API_USER}:{config.API_KEY}?provider=serper&timeout=60",
        "Content-Type": "application/json",
    }
    body = {
        "q": query,
        "location": "United States",
        "hl": lang,
        "num": min(top_k, 20),
    }
    response = requests.post(
        f"{config.API_HOST.rstrip('/')}/search",
        headers=headers,
        json=body,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict) and data.get("code") not in (None, 0):
        raise RuntimeError(f"Gateway error: {data.get('msg', 'Unknown error')}")
    return data.get("organic", []) or []


def _search_via_serper(query: str, lang: str, top_k: int) -> List[dict]:
    serper_api_key, _ = acquire_serper_api_key()
    headers = {
        "X-API-KEY": serper_api_key,
        "Content-Type": "application/json",
    }
    body = {
        "q": query,
        "hl": lang,
        "num": min(top_k, 20),
    }
    response = requests.post(
        config.SERPER_SEARCH_URL,
        headers=headers,
        json=body,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("organic", []) or []

def _image_search_via_serper(image_url: str) -> object:
    """Run reverse-image search against Serper's Google Lens endpoint."""
    serper_api_key, _ = acquire_serper_api_key()

    headers = {
        "X-API-KEY": serper_api_key,
        "Content-Type": "application/json",
    }
    response = requests.post(
        config.SERPER_LENS_URL,
        headers=headers,
        json={"url": image_url},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    organic = data.get("organic", []) or []
    if not organic:
        return data

    results = []
    for item in organic[:3]:
        result = {
            "title": item.get("title", ""),
            "source": item.get("source", "") or item.get("link", ""),
            "url": item.get("link", ""),
            "image_url": item.get("imageUrl", ""),
            "thumbnail_url": item.get("thumbnailUrl", ""),
            "snippet": item.get("snippet", ""),
        }
        results.append(result)
    return results



def _read_via_gateway(url: str) -> str:
    headers = {
        "Authorization": f"Bearer {config.API_USER}:{config.API_KEY}?provider=jina_ai&timeout=60",
        "Content-Type": "application/json",
    }
    response = requests.post(
        f"{config.API_HOST.rstrip('/')}/images",
        headers=headers,
        json={"url": url},
        timeout=30,
    )
    if response.status_code != 200:
        return ""
    data = response.json()
    if isinstance(data, dict):
        if data.get("code") == 200:
            return data.get("data", {}).get("content", "")
        return data.get("content") or data.get("text") or data.get("markdown") or ""
    return ""


def _read_via_jina(url: str) -> str:
    headers = {"Accept": "application/json"}
    if config.JINA_API_KEY:
        headers["Authorization"] = f"Bearer {config.JINA_API_KEY}"
    response = requests.get(
        config.JINA_READER_URL.rstrip("/") + "/" + url,
        headers=headers,
        timeout=30,
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


def read_url(url: str, query: str = "") -> dict[str, Any]:
    """Read a URL as either text content or a downloadable image."""

    normalized_url = (url or "").strip()
    if not normalized_url:
        return {"error": "URL is required."}
    if not normalized_url.startswith(("http://", "https://")):
        normalized_url = f"https://{normalized_url}"

    content_type = _probe_content_type(normalized_url)
    if content_type.startswith("image/"):
        temp_dir = tempfile.mkdtemp(prefix="opensearch_vl_read_url_")
        filename = os.path.basename(urlparse(normalized_url).path) or "downloaded_image"
        if not os.path.splitext(filename)[1]:
            extension = mimetypes.guess_extension(content_type) or ".png"
            filename = f"{filename}{extension}"
        local_path = image_io.download_to_temp(normalized_url, temp_dir, filename)
        if not local_path:
            return {"error": f"Failed to download image from {normalized_url}"}
        return {
            "kind": "image",
            "url": normalized_url,
            "content_type": content_type,
            "local_path": local_path,
        }

    try:
        document = _read_document(normalized_url)
    except Exception as exc:
        return {"error": f"read_url failed for {normalized_url}: {exc}"}

    content = document.get("content", "") or ""
    title = document.get("title", "") or ""
    summary = summarize_with_qwen(content=content, query=query, title=title) if query else ""
    return {
        "kind": "text",
        "url": document.get("url") or normalized_url,
        "title": title,
        "content": content,
        "summary": summary,
        "raw_markdown": document.get("raw_markdown", "") or "",
    }


def t2t_search(query: str, lang: str = "en", top_k: int = 5) -> str:
    """Search text pages via synthesis' Serper backend, then read/summarize."""

    try:
        fetch_limit = max(1, min(int(top_k) * 3, 100))
        response = _build_serper_client().search_text(query, limit=fetch_limit, hl=lang)
    except Exception as exc:
        return f"Tool execution error:\nText search failed: {exc}"

    if not response.results:
        return "Tool execution result:\nNo relevant web pages found for the query."

    formatted: list[str] = []
    for item in response.results:
        if _url_matches_blocked_domain(item.url or "", T2T_BLOCKED_SEARCH_DOMAINS):
            continue
        title = item.title or ""
        url = item.url or ""
        snippet = item.snippet or ""
        summary = snippet
        if url:
            read_result = read_url(url, query=query)
            if read_result.get("kind") == "text":
                summary = read_result.get("summary") or read_result.get("content") or snippet
            elif read_result.get("error"):
                logger.debug("read_url failed during t2t_search for %s: %s", url, read_result["error"])
        formatted.append(
            f"[Passage {len(formatted) + 1}]\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Summary:\n{summary}"
        )
        if len(formatted) >= top_k:
            break

    if not formatted:
        return "Tool execution result:\nNo relevant non-Wikipedia web pages found for the query."

    body = ("\n\n" + "=" * 60 + "\n\n").join(formatted)
    return f"Tool execution result:\n{body}"


def t2i_search(query: str, lang: str = "en", top_k: int = 5) -> str:
    """Search images from text using synthesis' Serper backend."""

    try:
        response = _build_serper_client().search_image(query, limit=top_k, hl=lang)
    except Exception as exc:
        return f"Tool execution error:\nText-to-image search failed: {exc}"

    results = [
        {
            "title": item.title,
            "image_url": item.image_url,
            "source_page_url": item.source_page_url,
            "thumbnail_url": item.thumbnail_url,
            "source": item.source,
            "snippet": item.snippet,
            "rank": item.rank,
        }
        for item in response.results[:top_k]
    ]
    if not results:
        return "Tool execution result:\nNo relevant images found for the query."
    return f"Tool execution result:\n{_format_search_results(results)}"


def i2i_search(
    image_url: str,
    visual_lookup: Optional[Callable[..., object]] = None,
    max_retries: int = 3,
    base_delay: int = 2,
) -> str:
    """Reverse-image search using Serper Lens or a caller-provided backend."""

    if not visual_lookup:
        visual_lookup = _image_search_via_serper

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            result = visual_lookup(image_url=image_url)
            if isinstance(result, dict) and "error" in result:
                raise RuntimeError(result["error"])
            summarised = summarize_image_search(result)
            payload = (
                json.dumps(summarised, ensure_ascii=False, indent=2)
                if isinstance(summarised, (dict, list))
                else str(summarised)
            )
            return f"Tool execution result:\n{payload}"
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(base_delay * (2 ** (attempt - 1)))
    return f"Tool execution error:\ni2i_search failed after {max_retries} retries: {last_error}"


def text_search(query: str, lang: str = "en", top_k: int = 5) -> str:
    """Backward-compatible alias for ``t2t_search``."""

    return t2t_search(query=query, lang=lang, top_k=top_k)


def image_search(
    image_url: str,
    visual_lookup: Optional[Callable[..., object]] = None,
    max_retries: int = 3,
    base_delay: int = 2,
) -> str:
    """Backward-compatible alias for ``i2i_search``."""

    return i2i_search(
        image_url=image_url,
        visual_lookup=visual_lookup,
        max_retries=max_retries,
        base_delay=base_delay,
    )
