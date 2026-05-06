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
import os
import re
import time
from typing import Callable, Dict, List, Optional

import requests

from . import config


logger = logging.getLogger(__name__)


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
    if not config.SERPER_API_KEY:
        raise RuntimeError(
            "Serper is not configured. Either set API_HOST/API_USER/API_KEY "
            "for the gateway, or set SERPER_API_KEY for direct access."
        )
    headers = {
        "X-API-KEY": config.SERPER_API_KEY,
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


def text_search(query: str, lang: str = "en", top_k: int = 5) -> str:
    """Run a Serper search, fetch each page through Jina and summarise."""

    use_gateway = config.gateway_enabled()
    try:
        if use_gateway:
            organic = _search_via_gateway(query, lang, top_k)
        else:
            organic = _search_via_serper(query, lang, top_k)
    except Exception as exc:
        return f"Tool execution error:\nText search failed: {exc}"

    if not organic:
        return "Tool execution result:\nNo relevant web pages found for the query."

    pages = []
    for idx, item in enumerate(organic[:top_k], start=1):
        url = item.get("link", "")
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        content = snippet
        if url:
            try:
                if use_gateway:
                    fetched = _read_via_gateway(url)
                else:
                    fetched = _read_via_jina(url)
                if fetched:
                    content = fetched
            except Exception as exc:
                logger.debug("Jina fetch failed for %s: %s", url, exc)
        pages.append({"index": idx, "url": url, "title": title, "content": content})

    formatted = []
    for page in pages:
        summary = summarize_with_qwen(
            content=page["content"], query=query, title=page["title"]
        )
        formatted.append(
            f"[Passage {page['index']}]\n"
            f"Title: {page['title']}\n"
            f"URL: {page['url']}\n"
            f"Summary:\n{summary}"
        )

    body = ("\n\n" + "=" * 60 + "\n\n").join(formatted)
    return f"Tool execution result:\n{body}"


def image_search(
    image_url: str,
    visual_lookup: Optional[Callable[..., object]] = None,
    max_retries: int = 3,
    base_delay: int = 2,
) -> str:
    """Run an external visual lookup against ``image_url`` and summarise it."""

    if not visual_lookup:
        return (
            "Tool execution error:\n"
            "image_search requires a visual lookup callable. Configure one via "
            "the runner (e.g. an external lens / similar-image API) before "
            "invoking image_search."
        )

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
    return f"Tool execution error:\nimage_search failed after {max_retries} retries: {last_error}"
