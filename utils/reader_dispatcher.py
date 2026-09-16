"""Lightweight HTTP dispatcher for a pool of self-hosted Jina Reader workers.

The dispatcher keeps the public Reader endpoint stable while spreading requests
across independent Node/Chrome processes.  The configured HTTP connection pool
is only a transport safeguard.  Per-backend admission control is optional and
disabled by default to preserve existing throughput behavior.
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response


def _parse_backend_urls() -> list[str]:
    raw = os.environ.get("READER_BACKEND_URLS", "")
    backends = [item.strip().rstrip("/") for item in raw.split(",") if item.strip()]
    if not backends:
        raise RuntimeError(
            "READER_BACKEND_URLS must contain at least one HTTP Reader backend, "
            "for example http://127.0.0.1:8102,http://127.0.0.1:8112"
        )
    return backends


BACKEND_URLS = _parse_backend_urls()
MAX_CONNECTIONS = max(1, int(os.environ.get("READER_DISPATCHER_MAX_CONNECTIONS", "4096")))
MAX_KEEPALIVE_CONNECTIONS = max(
    1,
    min(
        MAX_CONNECTIONS,
        int(os.environ.get("READER_DISPATCHER_MAX_KEEPALIVE_CONNECTIONS", "1024")),
    ),
)
READ_TIMEOUT_S = float(os.environ.get("READER_DISPATCHER_READ_TIMEOUT", "300"))
CONNECT_TIMEOUT_S = float(os.environ.get("READER_DISPATCHER_CONNECT_TIMEOUT", "10"))
TRUST_ENV = os.environ.get("READER_DISPATCHER_TRUST_ENV", "0").strip().lower() in {"1", "true", "yes", "on"}
MAX_INFLIGHT_PER_BACKEND = max(0, int(os.environ.get("READER_DISPATCHER_MAX_INFLIGHT_PER_BACKEND", "0")))

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "proxy-connection",
}


def _forward_request_headers(request: Request) -> list[tuple[str, str]]:
    return [
        (key, value)
        for key, value in request.headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS | {"host", "content-length"}
    ]


def _forward_response_headers(response: httpx.Response) -> dict[str, str]:
    # httpx transparently decodes compressed response bodies.  Do not forward
    # the old framing/encoding headers with the decoded body.
    return {
        key: value
        for key, value in response.headers.items()
        if key.lower()
        not in HOP_BY_HOP_HEADERS | {"content-length", "content-encoding"}
    }


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    timeout = httpx.Timeout(READ_TIMEOUT_S, connect=CONNECT_TIMEOUT_S)
    app.state.client = httpx.AsyncClient(
        timeout=timeout,
        limits=httpx.Limits(
            max_connections=MAX_CONNECTIONS,
            max_keepalive_connections=MAX_KEEPALIVE_CONNECTIONS,
        ),
        follow_redirects=False,
        trust_env=TRUST_ENV,
    )
    app.state.round_robin_index = 0
    app.state.round_robin_lock = asyncio.Lock()
    app.state.inflight = [0 for _ in BACKEND_URLS]
    app.state.completed = [0 for _ in BACKEND_URLS]
    app.state.failures = [0 for _ in BACKEND_URLS]
    app.state.backend_semaphores = [
        asyncio.Semaphore(MAX_INFLIGHT_PER_BACKEND) if MAX_INFLIGHT_PER_BACKEND else None
        for _ in BACKEND_URLS
    ]
    print(
        "[reader-dispatcher] backends=%s max_connections=%s "
        "max_keepalive_connections=%s read_timeout_s=%s trust_env=%s max_inflight_per_backend=%s"
        % (
            BACKEND_URLS,
            MAX_CONNECTIONS,
            MAX_KEEPALIVE_CONNECTIONS,
            READ_TIMEOUT_S,
            TRUST_ENV,
            MAX_INFLIGHT_PER_BACKEND,
        ),
        flush=True,
    )
    try:
        yield
    finally:
        await app.state.client.aclose()


app = FastAPI(title="Jina Reader Dispatcher", lifespan=lifespan)


async def _ordered_backend_indices() -> list[int]:
    async with app.state.round_robin_lock:
        start = app.state.round_robin_index % len(BACKEND_URLS)
        app.state.round_robin_index += 1
    indices = [(start + offset) % len(BACKEND_URLS) for offset in range(len(BACKEND_URLS))]
    # Under mixed slow/fast sites, least-inflight routing avoids assigning a
    # new request to a backend that is still busy with many browser renders.
    return sorted(indices, key=lambda index: app.state.inflight[index])


def _target_url(backend: str, target_path: str, query: str) -> str:
    target = f"{backend}/{target_path}"
    return f"{target}?{query}" if query else target


@app.get("/__reader_dispatcher/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "backend_count": len(BACKEND_URLS),
        "backends": BACKEND_URLS,
        "inflight": list(app.state.inflight),
        "max_inflight_per_backend": MAX_INFLIGHT_PER_BACKEND,
    }


@app.get("/__reader_dispatcher/stats")
async def stats() -> dict[str, Any]:
    return {
        "backends": [
            {
                "url": backend,
                "inflight": app.state.inflight[index],
                "completed": app.state.completed[index],
                "failures": app.state.failures[index],
            }
            for index, backend in enumerate(BACKEND_URLS)
        ],
        "max_connections": MAX_CONNECTIONS,
        "max_keepalive_connections": MAX_KEEPALIVE_CONNECTIONS,
        "max_inflight_per_backend": MAX_INFLIGHT_PER_BACKEND,
    }


@app.api_route(
    "/{target_path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
)
async def proxy(target_path: str, request: Request) -> Response:
    body = await request.body()
    headers = _forward_request_headers(request)
    query = request.url.query
    errors: list[str] = []

    for backend_index in await _ordered_backend_indices():
        backend = BACKEND_URLS[backend_index]
        semaphore = app.state.backend_semaphores[backend_index]
        if semaphore is not None:
            await semaphore.acquire()
        app.state.inflight[backend_index] += 1
        started = time.perf_counter()
        try:
            upstream = await app.state.client.request(
                request.method,
                _target_url(backend, target_path, query),
                headers=headers,
                content=body or None,
            )
            app.state.completed[backend_index] += 1
            elapsed_s = time.perf_counter() - started
            print(
                "[reader-dispatcher] backend=%s method=%s path_chars=%s "
                "status=%s elapsed_s=%.3f inflight=%s"
                % (
                    backend,
                    request.method,
                    len(target_path),
                    upstream.status_code,
                    elapsed_s,
                    app.state.inflight[backend_index],
                ),
                flush=True,
            )
            response_headers = _forward_response_headers(upstream)
            response_headers["X-Reader-Dispatcher-Backend"] = backend
            return Response(
                content=upstream.content,
                status_code=upstream.status_code,
                headers=response_headers,
            )
        except httpx.RequestError as exc:
            app.state.failures[backend_index] += 1
            errors.append(f"{backend}: {exc.__class__.__name__}: {exc}")
        finally:
            app.state.inflight[backend_index] -= 1
            if semaphore is not None:
                semaphore.release()

    return JSONResponse(
        status_code=503,
        content={
            "error": "reader_dispatcher_no_backend",
            "message": "All Reader backends failed to accept the request.",
            "backend_errors": errors,
        },
    )
