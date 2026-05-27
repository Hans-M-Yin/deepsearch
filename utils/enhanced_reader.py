"""Reader API wrapper that upgrades HTML extraction with ReaderLM-v2.

The service keeps the simple ``r.jina.ai``-style URL shape:

    GET /https://example.com

It first asks a self-hosted Jina Reader endpoint for HTML, then sends the
cleaned HTML to an OpenAI-compatible ReaderLM-v2 endpoint and returns Markdown.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any

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


app = FastAPI(title="Enhanced Reader API")


SCRIPT_PATTERN = r"<[ ]*script.*?/[\s]*script[ ]*>"
STYLE_PATTERN = r"<[ ]*style.*?/[\s]*style[ ]*>"
META_PATTERN = r"<[ ]*meta.*?>"
COMMENT_PATTERN = r"<[ ]*!--.*?--[ ]*>"
LINK_PATTERN = r"<[ ]*link.*?>"
BASE64_IMG_PATTERN = r'<img[^>]+src="data:image/[^;]+;base64,[^"]+"[^>]*>'
SVG_PATTERN = r"(<svg[^>]*>)(.*?)(</svg>)"


def normalize_url(target_url: str) -> str:
    if target_url.startswith(("http://", "https://")):
        return target_url
    return "https://" + target_url


def replace_svg(html: str, new_content: str = "this is a placeholder") -> str:
    return re.sub(
        SVG_PATTERN,
        lambda match: f"{match.group(1)}{new_content}{match.group(3)}",
        html,
        flags=re.DOTALL,
    )


def replace_base64_images(html: str, new_image_src: str = "#") -> str:
    return re.sub(BASE64_IMG_PATTERN, f'<img src="{new_image_src}"/>', html)


def clean_html(html: str, clean_svg: bool = True, clean_base64: bool = True) -> str:
    """Pre-clean HTML following the ReaderLM-v2 model-card guidance."""

    html = re.sub(SCRIPT_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    html = re.sub(STYLE_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    html = re.sub(META_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    html = re.sub(COMMENT_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    html = re.sub(LINK_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    if clean_svg:
        html = replace_svg(html)
    if clean_base64:
        html = replace_base64_images(html)
    return html


def create_prompt(
    html: str,
    instruction: str = "Extract the main content from the given HTML and convert it to Markdown format.",
) -> str:
    return f"{instruction}\n```html\n{html}\n```"


def strip_outer_markdown_fence(text: str) -> str:
    match = re.fullmatch(r"\s*```(?:markdown|md)?\s*\n(.*?)\n```\s*", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


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


async def convert_html_to_markdown(client: httpx.AsyncClient, html: str) -> str:
    cleaned_html = clean_html(html)[:READERLM_MAX_HTML_CHARS]
    headers = {"Content-Type": "application/json"}
    if READERLM_API_KEY:
        headers["Authorization"] = f"Bearer {READERLM_API_KEY}"

    response = await client.post(
        f"{READERLM_API_BASE.rstrip('/')}/chat/completions",
        headers=headers,
        json={
            "model": READERLM_MODEL_NAME,
            "messages": [{"role": "user", "content": create_prompt(cleaned_html)}],
            "temperature": 0,
            "max_tokens": READERLM_MAX_TOKENS,
            "extra_body": {"repetition_penalty": 1.08},
        },
        timeout=READER_TIMEOUT,
    )
    response.raise_for_status()
    data: dict[str, Any] = response.json()
    return strip_outer_markdown_fence(data["choices"][0]["message"]["content"])


@app.get("/{target_url:path}")
async def read(target_url: str, request: Request):
    url = normalize_url(target_url)
    wants_json = "application/json" in request.headers.get("accept", "")

    async with httpx.AsyncClient() as client:
        try:
            markdown_response, html_response = await asyncio.gather(
                fetch_markdown(client, url),
                fetch_html(client, url),
            )
            html = html_response
            markdown = await convert_html_to_markdown(client, html)
        except Exception as exc:
            message = f"Enhanced Reader error for {url}: {exc}"
            if wants_json:
                return JSONResponse(
                    status_code=502,
                    content={"data": None, "code": 502, "status": 502, "message": message},
                )
            return Response(message, status_code=502, media_type="text/plain")

    if wants_json:
        return {
            "data": {
                "title": "",
                "url": url,
                "content": markdown,
                "raw_markdown": markdown_response,
            },
            "code": 200,
            "status": 200,
        }

    body = f"URL Source: {url}\n\nMarkdown Content:\n{markdown}\n"
    return Response(body, media_type="text/plain; charset=utf-8")
