"""Serper-compatible wrapper for a local OpenSERP service.

The adapter lets existing Serper clients keep calling:

    POST /search
    POST /images

while OpenSERP runs behind it.  OpenSERP supports text search and text-to-image
search; reverse image search (`/lens`) is only forwarded when a Serper fallback
API key is configured.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Request


OPENSERP_BASE = os.environ.get("OPENSERP_BASE", "http://127.0.0.1:7000")
OPENSERP_TEXT_ENGINE = os.environ.get("OPENSERP_TEXT_ENGINE", "google")
OPENSERP_IMAGE_ENGINE = os.environ.get("OPENSERP_IMAGE_ENGINE", "bing")
OPENSERP_USE_MEGA = os.environ.get("OPENSERP_USE_MEGA", "").lower() in {
    "1",
    "true",
    "yes",
}
OPENSERP_MEGA_ENGINES = os.environ.get("OPENSERP_MEGA_ENGINES", "google,bing")
OPENSERP_MAX_REQUEST_LIMIT = int(os.environ.get("OPENSERP_MAX_REQUEST_LIMIT", "5"))
ADAPTER_TIMEOUT = float(os.environ.get("SERPER_ADAPTER_TIMEOUT", "60"))

SERPER_FALLBACK_API_KEY = os.environ.get("SERPER_FALLBACK_API_KEY", "")
SERPER_FALLBACK_SEARCH_URL = os.environ.get(
    "SERPER_FALLBACK_SEARCH_URL", "https://google.serper.dev/search"
)
SERPER_FALLBACK_IMAGES_URL = os.environ.get(
    "SERPER_FALLBACK_IMAGES_URL", "https://google.serper.dev/images"
)
SERPER_FALLBACK_LENS_URL = os.environ.get(
    "SERPER_FALLBACK_LENS_URL", "https://google.serper.dev/lens"
)
SERPER_FALLBACK_ON_ERROR = os.environ.get("SERPER_FALLBACK_ON_ERROR", "").lower() in {
    "1",
    "true",
    "yes",
}


app = FastAPI(title="OpenSERP Serper Compatibility Adapter")


def _limit(value: Any, default: int = 10) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(parsed, 100))


def _start(body: dict[str, Any], limit: int) -> int:
    if body.get("start") is not None:
        try:
            return max(0, int(body["start"]))
        except (TypeError, ValueError):
            return 0

    try:
        page = int(body.get("page") or 1)
    except (TypeError, ValueError):
        page = 1
    return max(0, page - 1) * limit


def _normalize_lang(value: Any) -> str | None:
    if not value:
        return None
    return str(value).upper()


def _normalize_region(body: dict[str, Any]) -> str | None:
    value = body.get("gl") or body.get("region")
    if not value:
        return None
    return str(value).upper()


def _displayed_link(item: dict[str, Any]) -> str:
    return item.get("display_url") or item.get("displayedLink") or item.get("domain") or ""


def _position(item: dict[str, Any], fallback: int) -> int:
    position = item.get("rank")
    if position is None and isinstance(item.get("position"), dict):
        position = item["position"].get("absolute")
    try:
        return int(position)
    except (TypeError, ValueError):
        return fallback


def _organic_item(item: dict[str, Any], fallback_position: int) -> dict[str, Any]:
    link = item.get("url") or item.get("link") or ""
    return {
        "title": item.get("title") or "",
        "link": link,
        "snippet": item.get("snippet") or "",
        "position": _position(item, fallback_position),
        "displayedLink": _displayed_link(item),
        "source": item.get("engine") or item.get("source") or OPENSERP_TEXT_ENGINE,
    }


def _is_search_result(item: dict[str, Any]) -> bool:
    item_type = item.get("type")
    if item_type in {"image", "ad", "advertisement"}:
        return False
    return bool(item.get("url") or item.get("link"))


def _image_url(item: dict[str, Any]) -> str:
    image = item.get("image")
    if isinstance(image, dict):
        return image.get("url") or image.get("imageUrl") or ""
    return item.get("imageUrl") or item.get("image_url") or item.get("url") or ""


def _thumbnail_url(item: dict[str, Any]) -> str:
    image = item.get("image")
    if isinstance(image, dict):
        return image.get("thumbnail") or image.get("thumbnailUrl") or ""
    return item.get("thumbnailUrl") or item.get("thumbnail_url") or ""


def _source_page(item: dict[str, Any]) -> str:
    source = item.get("source")
    if isinstance(source, dict):
        return source.get("page_url") or source.get("url") or ""
    return item.get("link") or item.get("page_url") or ""


def _image_item(item: dict[str, Any], fallback_position: int) -> dict[str, Any]:
    return {
        "title": item.get("title") or "",
        "imageUrl": _image_url(item),
        "thumbnailUrl": _thumbnail_url(item),
        "link": _source_page(item),
        "source": item.get("engine") or OPENSERP_IMAGE_ENGINE,
        "position": _position(item, fallback_position),
    }


def _serper_search_response(body: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    organic = []
    for idx, item in enumerate(data.get("results", []) or [], start=1):
        if not _is_search_result(item):
            continue
        organic.append(_organic_item(item, idx))

    return {
        "searchParameters": {
            "q": body.get("q", ""),
            "type": "search",
            "num": _limit(body.get("num")),
            "hl": body.get("hl"),
            "gl": body.get("gl"),
            "engine": "openserp",
        },
        "organic": organic,
        "peopleAlsoAsk": [],
        "relatedSearches": [],
        "_openserp": {
            "query": data.get("query", {}),
            "meta": data.get("meta", {}),
            "pagination": data.get("pagination", {}),
        },
    }


def _serper_images_response(body: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    images = []
    for idx, item in enumerate(data.get("results", []) or [], start=1):
        if item.get("type") not in (None, "image"):
            continue
        images.append(_image_item(item, idx))

    return {
        "searchParameters": {
            "q": body.get("q", ""),
            "type": "images",
            "num": _limit(body.get("num")),
            "hl": body.get("hl"),
            "gl": body.get("gl"),
            "engine": "openserp",
        },
        "images": images,
        "_openserp": {
            "query": data.get("query", {}),
            "meta": data.get("meta", {}),
            "pagination": data.get("pagination", {}),
        },
    }


def _openserp_endpoint(kind: str) -> str:
    if kind == "search":
        engine = "mega" if OPENSERP_USE_MEGA else OPENSERP_TEXT_ENGINE
    else:
        engine = "mega" if OPENSERP_USE_MEGA else OPENSERP_IMAGE_ENGINE
    return f"{OPENSERP_BASE.rstrip('/')}/{engine}/{kind}"


def _openserp_params(
    body: dict[str, Any],
    *,
    limit_override: int | None = None,
    start_override: int | None = None,
) -> dict[str, Any]:
    limit = limit_override or _limit(body.get("num") or body.get("limit"))
    params: dict[str, Any] = {
        "text": body.get("q") or body.get("query") or "",
        "limit": limit,
        "start": _start(body, limit) if start_override is None else start_override,
    }

    lang = _normalize_lang(body.get("hl") or body.get("lang"))
    region = _normalize_region(body)
    if lang:
        params["lang"] = lang
    if region:
        params["region"] = region
    if body.get("date"):
        params["date"] = body["date"]
    if body.get("site"):
        params["site"] = body["site"]
    if body.get("file"):
        params["file"] = body["file"]
    if OPENSERP_USE_MEGA:
        params["mode"] = body.get("mode") or os.environ.get("OPENSERP_MEGA_MODE", "balanced")
        params["engines"] = body.get("engines") or OPENSERP_MEGA_ENGINES
        params["dedupe"] = str(body.get("dedupe", True)).lower()
        params["merge"] = str(body.get("merge", True)).lower()
    return params


def _extract_error_detail(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        text = response.text.strip()
        return text[:1000] if text else response.reason_phrase

    if isinstance(data, dict):
        return (
            data.get("message")
            or data.get("detail")
            or data.get("error")
            or str(data)
        )
    return str(data)


async def _post_fallback(url: str, body: dict[str, Any]) -> dict[str, Any]:
    if not SERPER_FALLBACK_API_KEY:
        raise HTTPException(
            status_code=502,
            detail="OpenSERP failed and SERPER_FALLBACK_API_KEY is not configured.",
        )
    async with httpx.AsyncClient(timeout=ADAPTER_TIMEOUT) as client:
        response = await client.post(
            url,
            headers={
                "X-API-KEY": SERPER_FALLBACK_API_KEY,
                "Content-Type": "application/json",
            },
            json=body,
        )
    try:
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"Serper fallback error: {_extract_error_detail(exc.response)}",
        ) from exc
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail=f"Serper fallback timed out: {exc}") from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Serper fallback request error: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=f"Serper fallback returned invalid JSON: {exc}") from exc


async def _request_openserp_once(
    client: httpx.AsyncClient,
    kind: str,
    body: dict[str, Any],
    *,
    limit: int,
    start: int,
) -> dict[str, Any]:
    response = await client.get(
        _openserp_endpoint(kind),
        params=_openserp_params(body, limit_override=limit, start_override=start),
    )
    try:
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"OpenSERP upstream error: {_extract_error_detail(exc.response)}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=f"OpenSERP returned invalid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="OpenSERP returned a non-object response.")
    return data


async def _request_openserp(kind: str, body: dict[str, Any]) -> dict[str, Any]:
    requested_limit = _limit(body.get("num") or body.get("limit"))
    max_request_limit = max(1, min(OPENSERP_MAX_REQUEST_LIMIT, requested_limit))
    start = _start(body, requested_limit)

    async with httpx.AsyncClient(timeout=ADAPTER_TIMEOUT) as client:
        collected: list[dict[str, Any]] = []
        envelope: dict[str, Any] | None = None
        offset = start
        remaining = requested_limit

        while remaining > 0:
            chunk_limit = min(max_request_limit, remaining)
            try:
                data = await _request_openserp_once(
                    client,
                    kind,
                    body,
                    limit=chunk_limit,
                    start=offset,
                )
            except HTTPException:
                if collected:
                    break
                raise
            except httpx.TimeoutException as exc:
                if collected:
                    break
                raise HTTPException(status_code=504, detail=f"OpenSERP timed out: {exc}") from exc
            except httpx.RequestError as exc:
                if collected:
                    break
                raise HTTPException(status_code=502, detail=f"OpenSERP request error: {exc}") from exc

            if envelope is None:
                envelope = data
            chunk_results = data.get("results", []) or []
            if not chunk_results:
                break
            collected.extend(item for item in chunk_results if isinstance(item, dict))
            if len(chunk_results) < chunk_limit:
                break
            offset += chunk_limit
            remaining -= chunk_limit

    if envelope is None:
        raise HTTPException(status_code=502, detail="OpenSERP returned no response envelope.")

    seen_urls: set[str] = set()
    deduped = []
    for item in collected:
        key = item.get("url") or item.get("link") or item.get("id") or str(item)
        if key in seen_urls:
            continue
        seen_urls.add(key)
        deduped.append(item)
    envelope["results"] = deduped
    return envelope


async def _json_body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object.")
    if not body.get("q") and not body.get("query"):
        raise HTTPException(status_code=400, detail="Missing search query field: q")
    return body


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "openserp_base": OPENSERP_BASE,
        "text_engine": "mega" if OPENSERP_USE_MEGA else OPENSERP_TEXT_ENGINE,
        "image_engine": "mega" if OPENSERP_USE_MEGA else OPENSERP_IMAGE_ENGINE,
    }


@app.post("/search")
async def search(request: Request, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    del x_api_key
    body = await _json_body(request)
    try:
        return _serper_search_response(body, await _request_openserp("search", body))
    except Exception:
        if SERPER_FALLBACK_ON_ERROR:
            return await _post_fallback(SERPER_FALLBACK_SEARCH_URL, body)
        raise


@app.post("/images")
async def images(request: Request, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    del x_api_key
    body = await _json_body(request)
    try:
        return _serper_images_response(body, await _request_openserp("image", body))
    except Exception:
        if SERPER_FALLBACK_ON_ERROR:
            return await _post_fallback(SERPER_FALLBACK_IMAGES_URL, body)
        raise


@app.post("/lens")
async def lens(request: Request, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    del x_api_key
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object.")
    if not SERPER_FALLBACK_API_KEY:
        raise HTTPException(
            status_code=501,
            detail=(
                "OpenSERP does not support reverse image search. Configure "
                "SERPER_FALLBACK_API_KEY to forward /lens to Serper."
            ),
        )
    return await _post_fallback(SERPER_FALLBACK_LENS_URL, body)
