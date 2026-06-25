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
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import requests

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "synthesis.sft"

from synthesis.search_client import SerperSearchClient
from synthesis.wiki_text_builder import EnhancedReaderClient


logger = logging.getLogger(__name__)
MAX_SEARCH_RESULTS = 5


def get_tool_definitions() -> list[dict[str, Any]]:
    """Return OpenAI-compatible function tool definitions."""

    return [
        {
            "type": "function",
            "function": {
                "name": "t2t_search",
                "description": (
                    "Search text/web documents from a text query. Returns "
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
                        "lang": {
                            "type": "string",
                            "description": "Language code for search, such as en.",
                            "default": "en",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Maximum number of search results to return.",
                            "default": MAX_SEARCH_RESULTS,
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
                "description": "Search images from a text query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "A concrete image-search query string.",
                        },
                        "lang": {
                            "type": "string",
                            "description": "Language code for search, such as en.",
                            "default": "en",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Maximum number of image results to return.",
                            "default": MAX_SEARCH_RESULTS,
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
                    "Search for visually similar or matching images from the "
                    "most recent image in the current context. Optionally crop "
                    "a region first before performing the search."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "region": {
                            "type": "array",
                            "description": (
                                "Optional bounding box on the current image to crop before search. "
                                "Preferred format: [x1, y1, x2, y2]."
                            ),
                            "items": {"type": "number"},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Maximum number of reverse-image matches to return.",
                            "default": MAX_SEARCH_RESULTS,
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
                    "relevant to the given query. If it returns an image, "
                    "download the image."
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


def summarize_with_qwen(content: str, query: str, title: str) -> str:
    """Summarize webpage content with an OpenAI-compatible client."""

    prompt = (
        f"Based on the following webpage content, extract and summarize the content that is RELEVANT to the query: \"{query}\"\n "
        f"Webpage Title: {title}\n"
        f"Content:\n{content[:10000]}\n\n"
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
    """Reduce a raw image-search payload to compact identification data."""

    try:
        result_str = json.dumps(result_obj, ensure_ascii=False, indent=2)
    except Exception:
        result_str = str(result_obj)

    prompt = (
        "Extract only the relevant title/source information from the "
        "following image search results. Return compact JSON.\n\n"
        f"Image Search Results:\n{result_str[:3000]}"
    )
    try:
        completion = _openai_client().chat.completions.create(
            model=_summarizer_model(),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.3,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        content_text = completion.choices[0].message.content or ""
        if content_text:
            match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])", content_text, re.DOTALL)
            try:
                if match:
                    return json.loads(match.group(1))
                return json.loads(content_text.strip())
            except json.JSONDecodeError:
                return {"summary": content_text.strip()}
    except Exception as exc:  # pragma: no cover - network bound
        logger.warning("Image search summarization failed: %s", exc)

    if isinstance(result_obj, dict):
        filtered: dict[str, object] = {}
        for src_key, dst_key in (
            ("title", "title"),
            ("name", "title"),
            ("label", "title"),
            ("source", "source"),
            ("url", "source"),
            ("link", "source"),
        ):
            if src_key in result_obj and dst_key not in filtered:
                filtered[dst_key] = result_obj[src_key]
        return filtered or result_obj
    return result_obj


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


def read_url(url: str, query: str = "") -> dict[str, Any]:
    """Read a URL as either text content or a downloadable image."""

    normalized_url = (url or "").strip()
    if not normalized_url:
        return {"ok": False, "error": "URL is required."}
    if not normalized_url.startswith(("http://", "https://")):
        normalized_url = f"https://{normalized_url}"

    content_type = _probe_content_type(normalized_url)
    if content_type.startswith("image/"):
        temp_dir = tempfile.mkdtemp(prefix="synthesis_sft_read_url_")
        filename = os.path.basename(urlparse(normalized_url).path) or "downloaded_image"
        if not os.path.splitext(filename)[1]:
            extension = mimetypes.guess_extension(content_type) or ".png"
            filename = f"{filename}{extension}"
        save_path = os.path.join(temp_dir, filename)
        response = requests.get(normalized_url, timeout=60)
        response.raise_for_status()
        with open(save_path, "wb") as handle:
            handle.write(response.content)
        return {
            "ok": True,
            "kind": "image",
            "url": normalized_url,
            "content_type": content_type,
            "local_path": save_path,
        }

    try:
        document = _read_document(normalized_url)
    except Exception as exc:  # pragma: no cover - network bound
        return {"ok": False, "error": f"read_url failed for {normalized_url}: {exc}"}

    content = document.get("content", "") or ""
    title = document.get("title", "") or ""
    summary = summarize_with_qwen(content=content, query=query, title=title) if query else ""
    return {
        "ok": True,
        "kind": "text",
        "url": document.get("url") or normalized_url,
        "title": title,
        "content": content,
        "summary": summary,
        "raw_markdown": document.get("raw_markdown", "") or "",
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
                "source": item.source,
                "rank": item.rank,
            }
        )
    return {
        "ok": True,
        "query": query,
        "lang": lang,
        "count": len(results),
        "results": results,
    }


def t2i_search(query: str, lang: str = "en", top_k: int = 5) -> dict[str, Any]:
    """Search images from a text query."""

    top_k = max(1, min(int(top_k), MAX_SEARCH_RESULTS))
    response = _serper_client().search_image(query, limit=top_k, hl=lang)
    results = [
        {
            "title": item.title,
            "image_url": item.image_url,
            "source_page_url": item.source_page_url,
            "thumbnail_url": item.thumbnail_url,
            "snippet": item.snippet,
            "source": item.source,
            "rank": item.rank,
        }
        for item in response.results[:top_k]
    ]
    return {
        "ok": True,
        "query": query,
        "lang": lang,
        "count": len(results),
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
                "url": item.get("link", ""),
                "image_url": item.get("imageUrl", ""),
                "thumbnail_url": item.get("thumbnailUrl", ""),
                "snippet": item.get("snippet", ""),
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
            if isinstance(result, list):
                result = result[:top_k]
            summarized = summarize_image_search(result)
            return {
                "ok": True,
                "image_url": image_url,
                "top_k": top_k,
                "matches": summarized,
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
