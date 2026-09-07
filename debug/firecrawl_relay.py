#!/usr/bin/env python3
"""Public Firecrawl relay for workers without reliable public egress.

Run this service on a node that can reach ``api.firecrawl.dev``.  A caller
sends a Firecrawl v2 scrape request to this relay; the relay forwards it to
the official Firecrawl API and returns the upstream JSON response unchanged.

The caller supplies a required compatibility key in ``X-API-KEY`` or
``Authorization: Bearer ...``.  When ``--keys-file`` is configured, that
value is ignored and the Relay selects the real upstream key from its own
pool.  Relay authentication is intentionally disabled; restrict the
listening port with network policy if necessary.

Example on the public-egress node::

    python debug/firecrawl_relay.py --bind 0.0.0.0 --port 18081

Then on the worker::

    export FIRECRAWL_RELAY_URL='http://[fdbd:dccd:cde2:1701:3e21:640a:cbaf:b8a3]:18081'

The relay exposes ``/healthz``, ``/v2/scrape``, and the Firecrawl Browser
session endpoints under ``/v2/browser``.  Browser requests are still kept
opaque to workers: the relay only forwards the JSON response, including the
Base64 payload returned by the image-download Browser code.

The relay deliberately limits in-flight upstream requests because a relay is
also a good place to apply backpressure before Firecrawl or the egress gateway
becomes saturated.  Text scraping and Browser image execution use separate
retry budgets because an image Browser execution is idempotent (the code only
performs a GET), while creating or deleting a Browser session is not a safe
operation to repeat after an ambiguous network timeout.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synthesis.firecrawl_client import FirecrawlApiKeyPool, FirecrawlClient, _is_firecrawl_credit_exhausted


MAX_REQUEST_BODY_BYTES = 16 * 1024 * 1024
MAX_RESPONSE_BODY_BYTES = 64 * 1024 * 1024
DEFAULT_TIMEOUT_S = 180.0
DEFAULT_QUEUE_TIMEOUT_S = 60.0
DEFAULT_MAX_INFLIGHT = 8
DEFAULT_TEXT_UPSTREAM_ATTEMPTS = 2
DEFAULT_BROWSER_TIMEOUT_S = 150.0
DEFAULT_BROWSER_IMAGE_ATTEMPTS = 2
DEFAULT_BROWSER_EXECUTE_ATTEMPTS = 2
DEFAULT_BROWSER_CREATE_ATTEMPTS = 1
DEFAULT_BROWSER_DELETE_ATTEMPTS = 1
DEFAULT_RETRY_DELAY_S = 3.0
DEFAULT_BROWSER_RETRY_DELAY_S = 2.0
DEFAULT_POOL_PERSIST_EVERY = 10000
_TRANSIENT_HTTP_STATUS_CODES = {408, 429, 500, 502, 503, 504}
_SNAKE_TO_CAMEL = {
    "include_tags": "includeTags",
    "exclude_tags": "excludeTags",
    "only_main_content": "onlyMainContent",
    "wait_for": "waitFor",
    "skip_tls_verification": "skipTlsVerification",
    "remove_base64_images": "removeBase64Images",
    "fast_mode": "fastMode",
    "use_mock": "useMock",
    "block_ads": "blockAds",
    "store_in_cache": "storeInCache",
    "max_age": "maxAge",
    "min_age": "minAge",
    "redact_pii": "redactPII",
    "threat_protection": "threatProtection",
    "audit_metadata": "auditMetadata",
    "activity_ttl": "activityTtl",
    "stream_web_view": "streamWebView",
}


def _read_limited(stream, *, limit: int) -> bytes:
    body = stream.read(limit + 1)
    if len(body) > limit:
        raise ValueError(f"response exceeds the {limit} byte limit")
    return body


def _json_body(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


def _install_shutdown_handlers(server: "ThreadingHTTPServer") -> None:
    """Flush deferred pool state when the process receives a graceful signal.

    ``SIGKILL`` cannot be handled by user-space code.  For SIGINT/SIGTERM we
    flush immediately and then raise KeyboardInterrupt so the existing
    ``finally`` block performs the normal server shutdown and a second flush.
    """

    def handle_shutdown(signum: int, _frame: object) -> None:
        try:
            signal_name = signal.Signals(signum).name
        except ValueError:
            signal_name = str(signum)
        print(
            f"Received {signal_name}; flushing Firecrawl pool state before shutdown.",
            flush=True,
        )
        relay_pool = getattr(server, "key_pool", None)
        if relay_pool is not None:
            try:
                relay_pool.flush()
            except Exception as exc:  # pragma: no cover - defensive shutdown path
                print(
                    f"Failed to flush Firecrawl pool state during shutdown: {exc!r}",
                    file=sys.stderr,
                    flush=True,
                )
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)


def _normalize_scrape_payload(payload: dict[str, object]) -> dict[str, object]:
    """Apply the same public camel-case mapping used by the Firecrawl SDK.

    ``FirecrawlClient`` normally gives the SDK snake-case Python arguments.
    The worker-side Relay path bypasses that SDK HTTP layer, so the relay must
    perform this small request normalization before forwarding to ``/v2``.
    """

    normalized = dict(payload)
    for snake_case, camel_case in _SNAKE_TO_CAMEL.items():
        if snake_case in normalized and camel_case not in normalized:
            normalized[camel_case] = normalized.pop(snake_case)
    if isinstance(normalized.get("redactPII"), dict):
        redact = dict(normalized["redactPII"])
        if "replace_style" in redact and "replaceStyle" not in redact:
            redact["replaceStyle"] = redact.pop("replace_style")
        normalized["redactPII"] = redact
    return normalized


def _normalize_browser_create_payload(payload: dict[str, object]) -> dict[str, object]:
    """Normalize the small Browser-create payload before forwarding it."""

    normalized = dict(payload)
    for snake_case, camel_case in {
        "activity_ttl": "activityTtl",
        "stream_web_view": "streamWebView",
    }.items():
        if snake_case in normalized and camel_case not in normalized:
            normalized[camel_case] = normalized.pop(snake_case)
    profile = normalized.get("profile")
    if isinstance(profile, dict):
        profile = dict(profile)
        if "save_changes" in profile and "saveChanges" not in profile:
            profile["saveChanges"] = profile.pop("save_changes")
        normalized["profile"] = profile
    return normalized


def _browser_session_path(path: str) -> bool:
    parts = [unquote(part) for part in urlsplit(path).path.strip("/").split("/")]
    return len(parts) == 3 and parts[:2] == ["v2", "browser"] and bool(parts[2])


def _browser_execute_path(path: str) -> bool:
    parts = [unquote(part) for part in urlsplit(path).path.strip("/").split("/")]
    return (
        len(parts) == 4
        and parts[:2] == ["v2", "browser"]
        and bool(parts[2])
        and parts[3] == "execute"
    )


def _browser_session_id(path: str) -> str | None:
    parts = [unquote(part) for part in urlsplit(path).path.strip("/").split("/")]
    if len(parts) >= 3 and parts[:2] == ["v2", "browser"] and parts[2]:
        return parts[2]
    return None


def _firecrawl_response_details(
    response_body: bytes,
    *,
    http_status_code: int,
    request_type: str,
) -> tuple[bool, int | None, int, str, bool, bool]:
    """Return result/credit details without treating target errors as key errors."""

    try:
        payload = json.loads(response_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False, http_status_code, 0, "non-json Firecrawl response", False, False
    if not isinstance(payload, dict):
        return False, http_status_code, 0, "non-object Firecrawl response", False, False

    root_status = FirecrawlClient._root_status_code(payload)
    metadata = FirecrawlClient._response_metadata(payload)
    target_status = FirecrawlClient._non_negative_int(
        metadata.get("statusCode", metadata.get("status_code"))
    )
    credits_used = FirecrawlClient._non_negative_int(
        metadata.get("creditsUsed", metadata.get("credits_used"))
    ) or 0
    if request_type.startswith("browser"):
        browser_credits = FirecrawlClient._non_negative_int(
            payload.get("creditsBilled", payload.get("credits_billed"))
        )
        if browser_credits is not None:
            credits_used = browser_credits

    root_error = str(payload.get("error") or payload.get("message") or "")
    api_status_code = root_status
    if api_status_code is None and http_status_code >= 400:
        api_status_code = http_status_code
    credit_error = _is_firecrawl_credit_exhausted(
        status_code=api_status_code,
        response_body=root_error,
    )
    auth_error = api_status_code == 401 and target_status is None
    target_failure = target_status is not None and target_status != 200
    success = (
        200 <= http_status_code < 300
        and payload.get("success") is not False
        and not root_error
        and not target_failure
    )
    error = root_error
    if not error and target_failure:
        error = f"target statusCode {target_status}"
    if not error and not success:
        error = f"Firecrawl HTTP {http_status_code}"
    return success, api_status_code or target_status, credits_used, error, auth_error, credit_error


def _browser_image_diagnostics(response_body: bytes) -> dict[str, object]:
    """Extract target-image diagnostics from our Browser result envelope."""

    try:
        outer = json.loads(response_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(outer, dict):
        return {}
    result = outer.get("result") or outer.get("stdout") or outer.get("output")
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}
    if not isinstance(result, dict):
        return {}
    diagnostics: dict[str, object] = {}
    try:
        status_code = int(result.get("status") or 0)
    except (TypeError, ValueError):
        status_code = 0
    if 100 <= status_code <= 599:
        diagnostics["status_code"] = status_code
    for source_key, log_key in (
        ("target_error_type", "target_error_type"),
        ("target_error_message", "target_error"),
        ("target_phase", "target_phase"),
        ("error", "target_error"),
    ):
        value = result.get(source_key)
        if value and log_key not in diagnostics:
            diagnostics[log_key] = str(value)[:500]
    try:
        byte_count = int(result.get("byte_count") or 0)
    except (TypeError, ValueError):
        byte_count = 0
    if byte_count >= 0:
        diagnostics["target_byte_count"] = byte_count
    return diagnostics


def _browser_image_status_code(response_body: bytes) -> int | None:
    """Extract the target-image status from our Browser result envelope."""

    status_code = _browser_image_diagnostics(response_body).get("status_code")
    return int(status_code) if isinstance(status_code, int) else None


def _log_upstream(
    *,
    path: str,
    request_type: str | None = None,
    status_code: int | None = None,
    elapsed_s: float,
    response_bytes: int | None = None,
    attempt: int | None = None,
    target_status_code: int | None = None,
    target_error_type: str | None = None,
    target_phase: str | None = None,
    target_error: str | None = None,
    target_byte_count: int | None = None,
    error: BaseException | None = None,
) -> None:
    fields = [f"path={path!r}"]
    if request_type is not None:
        fields.append(f"request_type={request_type!r}")
    fields.append(f"elapsed_s={elapsed_s:.3f}")
    if attempt is not None:
        fields.append(f"attempt={attempt}")
    if status_code is not None:
        fields.append(f"status_code={status_code}")
    if response_bytes is not None:
        fields.append(f"response_bytes={response_bytes}")
    if target_status_code is not None:
        fields.append(f"target_http_status={target_status_code}")
    if target_error_type is not None:
        fields.append(f"target_error_type={target_error_type!r}")
    if target_phase is not None:
        fields.append(f"target_phase={target_phase!r}")
    if target_error is not None:
        fields.append(f"target_error={target_error!r}")
    if target_byte_count is not None:
        fields.append(f"target_byte_count={target_byte_count}")
    if error is not None:
        fields.append(f"error_type={error.__class__.__name__}")
        fields.append(f"error={str(getattr(error, 'reason', error))!r}")
    print("[firecrawl-relay-upstream] " + " ".join(fields), file=sys.stderr, flush=True)


class IPv6ThreadingHTTPServer(ThreadingHTTPServer):
    address_family = socket.AF_INET6
    daemon_threads = True
    allow_reuse_address = True


class FirecrawlRelayHandler(BaseHTTPRequestHandler):
    server_version = "OpenSearch-VL-FirecrawlRelay/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        # Never log request headers or bodies because they may contain API keys
        # and private research prompts.
        sys.stderr.write(f"[firecrawl-relay] {self.address_string()} {fmt % args}\n")
        sys.stderr.flush()

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if urlsplit(self.path).path == "/healthz":
            self._write_json(200, {"ok": True, "service": "firecrawl-relay"})
            return
        self._write_json(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlsplit(self.path).path
        request_type = self._request_type_for_headers(path, method="POST")
        if request_type is None:
            self._write_json(404, {"error": "unsupported_path"})
            return

        body = self._read_json_body()
        if body is None:
            return

        if request_type == "text_scrape" and not str(body.get("url") or "").strip():
            self._write_json(400, {"error": "request_body_requires_url"})
            return

        client_api_key = self._api_key_from_headers()
        if not client_api_key:
            self._write_json(400, {"error": "missing_firecrawl_api_key"})
            return

        if request_type == "text_scrape":
            body = _normalize_scrape_payload(body)
        elif request_type == "browser_create":
            body = _normalize_browser_create_payload(body)

        self._proxy_with_backpressure(
            path=path,
            method="POST",
            body=_json_body(body),
            client_api_key=client_api_key,
            request_type=request_type,
        )

    def do_DELETE(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlsplit(self.path).path
        request_type = self._request_type(path, method="DELETE")
        if request_type is None:
            self._write_json(404, {"error": "unsupported_path"})
            return

        client_api_key = self._api_key_from_headers()
        if not client_api_key:
            self._write_json(400, {"error": "missing_firecrawl_api_key"})
            return

        self._proxy_with_backpressure(
            path=path,
            method="DELETE",
            body=None,
            client_api_key=client_api_key,
            request_type=request_type,
        )

    @staticmethod
    def _request_type(path: str, *, method: str) -> str | None:
        if method == "POST" and path == "/v2/scrape":
            return "text_scrape"
        if method == "POST" and path == "/v2/browser":
            return "browser_create"
        if method == "DELETE" and _browser_session_path(path):
            return "browser_delete"
        if method == "POST" and _browser_execute_path(path):
            # The static route matcher cannot access instance headers.  The
            # caller refines this to browser_image in _request_type_for_headers.
            return "browser_execute"
        return None

    def _request_type_for_headers(self, path: str, *, method: str) -> str | None:
        request_type = self._request_type(path, method=method)
        if request_type == "browser_execute":
            marker = str(self.headers.get("X-Firecrawl-Relay-Request-Type") or "").strip().lower()
            if marker in {"image", "image_download", "browser_image"}:
                return "browser_image"
        return request_type

    def _read_json_body(self) -> dict[str, object] | None:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        if content_length <= 0 or content_length > MAX_REQUEST_BODY_BYTES:
            self._write_json(400, {"error": "invalid_content_length"})
            return None

        try:
            body = self.rfile.read(content_length)
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._write_json(400, {"error": "request_body_must_be_json"})
            return None
        if not isinstance(payload, dict):
            self._write_json(400, {"error": "request_body_must_be_object"})
            return None
        return payload

    def _proxy_with_backpressure(
        self,
        *,
        path: str,
        method: str,
        body: bytes | None,
        client_api_key: str,
        request_type: str,
    ) -> None:
        slots = self.server.upstream_slots
        if not slots.acquire(timeout=self.server.queue_timeout_s):
            self._write_json(503, {"error": "firecrawl_relay_busy"})
            return
        try:
            relay_pool = getattr(self.server, "key_pool", None)
            key_rotation_allowed = request_type in {"text_scrape", "browser_create"}
            if relay_pool is not None and key_rotation_allowed:
                try:
                    key_attempt_budget = max(1, len(relay_pool.active_key_ids()))
                except RuntimeError as exc:
                    self._write_json(
                        503,
                        {"error": "firecrawl_relay_no_available_key", "detail": str(exc)},
                    )
                    return
            else:
                # Browser execute/delete requests are tied to the key that
                # created the remote session and cannot be transparently
                # retried with a different account.
                key_attempt_budget = 1

            for key_attempt in range(1, key_attempt_budget + 1):
                api_key, key_id = self._select_upstream_key(
                    path=path,
                    request_type=request_type,
                    client_api_key=client_api_key,
                )
                if api_key is None:
                    return
                result = self._proxy_upstream(
                    path=path,
                    method=method,
                    body=body,
                    api_key=api_key,
                    key_id=key_id,
                    request_type=request_type,
                )
                if result is None:
                    return

                status_code, response_body, content_type = result
                self._record_pool_result(
                    key_id=key_id,
                    request_type=request_type,
                    http_status_code=status_code,
                    response_body=response_body,
                )
                _, _, _, _, _, credit_exhausted = _firecrawl_response_details(
                    response_body,
                    http_status_code=status_code,
                    request_type=request_type,
                )
                if (
                    credit_exhausted
                    and relay_pool is not None
                    and key_rotation_allowed
                    and key_attempt < key_attempt_budget
                ):
                    print(
                        f"[firecrawl-relay] upstream key exhausted; rotating "
                        f"key_attempt={key_attempt + 1}/{key_attempt_budget} "
                        f"request_type={request_type!r}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue

                self._update_browser_session_mapping(
                    path=path,
                    request_type=request_type,
                    status_code=status_code,
                    response_body=response_body,
                    key_id=key_id,
                )
                self._write_bytes(status_code, response_body, content_type)
                return
        finally:
            slots.release()

    def _select_upstream_key(
        self,
        *,
        path: str,
        request_type: str,
        client_api_key: str,
    ) -> tuple[str | None, str | None]:
        relay_pool = getattr(self.server, "key_pool", None)
        if relay_pool is None:
            # Backward-compatible mode: without --keys-file, forward the
            # caller-provided key exactly as before.
            return client_api_key, None

        if request_type in {"browser_execute", "browser_image", "browser_delete"}:
            session_id = _browser_session_id(path)
            with self.server.browser_session_lock:
                key_id = self.server.browser_session_keys.get(session_id or "")
            if not key_id:
                self._write_json(
                    502,
                    {"error": "firecrawl_relay_unknown_browser_session"},
                )
                return None, None
            try:
                return relay_pool.key_for_id(key_id), key_id
            except KeyError:
                self._write_json(
                    502,
                    {"error": "firecrawl_relay_browser_session_key_unavailable"},
                )
                return None, None

        try:
            api_key, metadata = relay_pool.acquire_key()
        except RuntimeError as exc:
            self._write_json(
                503,
                {"error": "firecrawl_relay_no_available_key", "detail": str(exc)},
            )
            return None, None
        return api_key, str(metadata.get("key_id") or "")

    def _api_key_from_headers(self) -> str:
        api_key = str(self.headers.get("X-API-KEY", "")).strip()
        if api_key:
            return api_key
        authorization = str(self.headers.get("Authorization", "")).strip()
        prefix = "Bearer "
        if authorization.lower().startswith(prefix.lower()):
            return authorization[len(prefix):].strip()
        return ""

    def _proxy_upstream(
        self,
        *,
        path: str,
        method: str,
        body: bytes | None,
        api_key: str,
        key_id: str | None,
        request_type: str,
    ) -> tuple[int, bytes, str] | None:
        upstream_base = str(self.server.upstream_base).rstrip("/")
        upstream_url = (
            upstream_base
            if upstream_base.endswith(path)
            else f"{upstream_base}{path}"
        )
        attempts, timeout_s, retry_delay_s, retry_status_codes = self._retry_policy(request_type)
        for attempt in range(1, attempts + 1):
            headers = {
                "Accept": "application/json",
                "X-API-KEY": api_key,
                "Authorization": f"Bearer {api_key}",
            }
            if body is not None:
                headers["Content-Type"] = "application/json"
            upstream_request = Request(
                upstream_url,
                data=body,
                headers=headers,
                method=method,
            )
            started_at = time.perf_counter()
            try:
                with urlopen(upstream_request, timeout=timeout_s) as response:
                    response_body = _read_limited(
                        response,
                        limit=MAX_RESPONSE_BODY_BYTES,
                    )
                    status_code = int(response.getcode() or 200)
                    content_type = response.headers.get("Content-Type", "application/json")
                target_diagnostics = (
                    _browser_image_diagnostics(response_body)
                    if request_type == "browser_image"
                    else {}
                )
                target_status_code = target_diagnostics.get("status_code")
                if (
                    target_status_code in _TRANSIENT_HTTP_STATUS_CODES
                    and attempt < attempts
                ):
                    _log_upstream(
                        path=path,
                        request_type=request_type,
                        status_code=target_status_code,
                        elapsed_s=time.perf_counter() - started_at,
                        response_bytes=len(response_body),
                        attempt=attempt,
                        target_status_code=target_status_code,
                        target_error_type=target_diagnostics.get("target_error_type"),
                        target_phase=target_diagnostics.get("target_phase"),
                        target_error=target_diagnostics.get("target_error"),
                        target_byte_count=target_diagnostics.get("target_byte_count"),
                        error=RuntimeError("transient target-image status inside Browser result"),
                    )
                    time.sleep(
                        self._retry_delay(
                            attempt=attempt,
                            base_delay_s=retry_delay_s,
                        )
                    )
                    continue
                _log_upstream(
                    path=path,
                    request_type=request_type,
                    status_code=status_code,
                    elapsed_s=time.perf_counter() - started_at,
                    response_bytes=len(response_body),
                    attempt=attempt,
                    target_status_code=target_status_code,
                    target_error_type=target_diagnostics.get("target_error_type"),
                    target_phase=target_diagnostics.get("target_phase"),
                    target_error=target_diagnostics.get("target_error"),
                    target_byte_count=target_diagnostics.get("target_byte_count"),
                )
                return status_code, response_body, content_type
            except HTTPError as exc:
                try:
                    response_body = _read_limited(exc, limit=MAX_RESPONSE_BODY_BYTES)
                except ValueError as limit_error:
                    self._write_json(502, {"error": str(limit_error)})
                    return
                content_type = exc.headers.get("Content-Type", "application/json")
                _log_upstream(
                    path=path,
                    request_type=request_type,
                    status_code=int(exc.code),
                    elapsed_s=time.perf_counter() - started_at,
                    response_bytes=len(response_body),
                    attempt=attempt,
                )
                if int(exc.code) in retry_status_codes and attempt < attempts:
                    time.sleep(
                        self._retry_delay(
                            attempt=attempt,
                            base_delay_s=retry_delay_s,
                            retry_after=exc.headers.get("Retry-After"),
                        )
                    )
                    continue
                return int(exc.code), response_body, content_type
            except (URLError, TimeoutError, OSError) as exc:
                _log_upstream(
                    path=path,
                    request_type=request_type,
                    elapsed_s=time.perf_counter() - started_at,
                    attempt=attempt,
                    error=exc,
                )
                if attempt < attempts:
                    time.sleep(self._retry_delay(attempt=attempt, base_delay_s=retry_delay_s))
                    continue
                self._record_pool_failure(
                    key_id=key_id,
                    error=str(getattr(exc, "reason", exc)),
                )
                self._write_json(
                    502,
                    {
                        "error": "upstream_firecrawl_request_failed",
                        "error_type": exc.__class__.__name__,
                        "detail": str(getattr(exc, "reason", exc)),
                    },
                )
                return
            except ValueError as exc:
                _log_upstream(
                    path=path,
                    request_type=request_type,
                    elapsed_s=time.perf_counter() - started_at,
                    attempt=attempt,
                    error=exc,
                )
                self._record_pool_failure(key_id=key_id, error=str(exc))
                self._write_json(502, {"error": str(exc)})
                return

    def _record_pool_result(
        self,
        *,
        key_id: str | None,
        request_type: str,
        http_status_code: int,
        response_body: bytes,
    ) -> None:
        relay_pool = getattr(self.server, "key_pool", None)
        if not key_id or relay_pool is None:
            return
        success, status_code, credits_used, error, auth_error, credit_error = _firecrawl_response_details(
            response_body,
            http_status_code=http_status_code,
            request_type=request_type,
        )
        relay_pool.record_result(
            key_id,
            success=success,
            error=None if success else error,
            credits_used=credits_used,
            status_code=status_code,
            key_auth_failed=auth_error,
            key_credits_exhausted=credit_error,
        )

    def _record_pool_failure(self, *, key_id: str | None, error: str) -> None:
        relay_pool = getattr(self.server, "key_pool", None)
        if not key_id or relay_pool is None:
            return
        relay_pool.record_result(
            key_id,
            success=False,
            error=error,
        )

    def _update_browser_session_mapping(
        self,
        *,
        path: str,
        request_type: str,
        status_code: int,
        response_body: bytes,
        key_id: str | None,
    ) -> None:
        if getattr(self.server, "key_pool", None) is None:
            return
        session_id = _browser_session_id(path)
        if request_type == "browser_create":
            if not key_id or not (200 <= status_code < 300):
                return
            try:
                response = json.loads(response_body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return
            if isinstance(response, dict) and response.get("id"):
                with self.server.browser_session_lock:
                    self.server.browser_session_keys[str(response["id"])] = key_id
            return
        if request_type == "browser_delete" and session_id and status_code in {200, 204, 404}:
            with self.server.browser_session_lock:
                self.server.browser_session_keys.pop(session_id, None)

    def _retry_policy(self, request_type: str) -> tuple[int, float, float, set[int]]:
        if request_type == "text_scrape":
            return (
                self.server.text_upstream_attempts,
                self.server.text_timeout_s,
                self.server.text_retry_delay_s,
                _TRANSIENT_HTTP_STATUS_CODES,
            )
        if request_type == "browser_image":
            return (
                self.server.browser_image_attempts,
                self.server.browser_timeout_s,
                self.server.browser_retry_delay_s,
                _TRANSIENT_HTTP_STATUS_CODES,
            )
        if request_type == "browser_execute":
            return (
                self.server.browser_execute_attempts,
                self.server.browser_timeout_s,
                self.server.browser_retry_delay_s,
                _TRANSIENT_HTTP_STATUS_CODES,
            )
        if request_type == "browser_create":
            return (
                self.server.browser_create_attempts,
                self.server.browser_timeout_s,
                self.server.browser_retry_delay_s,
                {408, 429, 500, 502, 503, 504},
            )
        return (
            self.server.browser_delete_attempts,
            self.server.browser_timeout_s,
            self.server.browser_retry_delay_s,
            {408, 429, 500, 502, 503, 504},
        )

    @staticmethod
    def _retry_delay(*, attempt: int, base_delay_s: float, retry_after: str | None = None) -> float:
        delay = max(0.0, float(base_delay_s)) * (2 ** max(0, attempt - 1))
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except (TypeError, ValueError):
                pass
        return delay

    def _write_json(self, status_code: int, payload: object) -> None:
        self._write_bytes(status_code, _json_body(payload), "application/json")

    def _write_bytes(self, status_code: int, body: bytes, content_type: str) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="::", help="Bind address; default is all IPv6 interfaces.")
    parser.add_argument("--port", type=int, default=18081, help="Listen port.")
    parser.add_argument(
        "--keys-file",
        default=os.environ.get("FIRECRAWL_RELAY_KEYS_FILE"),
        help="Relay-owned key file; enabling this makes the relay select real upstream keys.",
    )
    parser.add_argument(
        "--state-file",
        default=os.environ.get("FIRECRAWL_RELAY_POOL_STATE_FILE"),
        help="Relay-owned pool state file; defaults to the normal Firecrawl pool state path.",
    )
    parser.add_argument(
        "--pool-default-credits",
        type=int,
        default=int(os.environ.get("FIRECRAWL_RELAY_POOL_DEFAULT_CREDITS") or 10000),
    )
    parser.add_argument(
        "--pool-persist-every",
        type=int,
        default=int(
            os.environ.get("FIRECRAWL_RELAY_POOL_PERSIST_EVERY")
            or DEFAULT_POOL_PERSIST_EVERY
        ),
        help="Checkpoint the in-memory pool after this many uses (default: 10000).",
    )
    parser.add_argument(
        "--upstream",
        default="https://api.firecrawl.dev",
        help="Firecrawl upstream base URL.",
    )
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S, help="Upstream request timeout.")
    parser.add_argument(
        "--queue-timeout-s",
        type=float,
        default=DEFAULT_QUEUE_TIMEOUT_S,
        help="Maximum time to wait for an upstream concurrency slot.",
    )
    parser.add_argument(
        "--max-inflight",
        type=int,
        default=DEFAULT_MAX_INFLIGHT,
        help="Maximum concurrent upstream Firecrawl requests.",
    )
    parser.add_argument(
        "--upstream-attempts",
        type=int,
        default=DEFAULT_TEXT_UPSTREAM_ATTEMPTS,
        help="Legacy alias for --text-upstream-attempts.",
    )
    parser.add_argument(
        "--retry-delay-s",
        type=float,
        default=DEFAULT_RETRY_DELAY_S,
        help="Legacy alias for --text-retry-delay-s.",
    )
    parser.add_argument("--text-upstream-attempts", type=int, default=None)
    parser.add_argument("--text-timeout-s", type=float, default=None)
    parser.add_argument("--text-retry-delay-s", type=float, default=None)
    parser.add_argument(
        "--browser-timeout-s",
        type=float,
        default=None,
        help=f"Timeout for Browser create/execute/delete requests; defaults to {DEFAULT_BROWSER_TIMEOUT_S:g}s.",
    )
    parser.add_argument(
        "--browser-image-attempts",
        type=int,
        default=DEFAULT_BROWSER_IMAGE_ATTEMPTS,
        help="Attempts for idempotent Browser image execution.",
    )
    parser.add_argument(
        "--browser-execute-attempts",
        type=int,
        default=DEFAULT_BROWSER_EXECUTE_ATTEMPTS,
        help="Attempts for non-image Browser execute requests.",
    )
    parser.add_argument(
        "--browser-create-attempts",
        type=int,
        default=DEFAULT_BROWSER_CREATE_ATTEMPTS,
        help="Attempts for Browser session creation; default 1 avoids duplicate sessions after ambiguous timeouts.",
    )
    parser.add_argument("--browser-delete-attempts", type=int, default=DEFAULT_BROWSER_DELETE_ATTEMPTS)
    parser.add_argument("--browser-retry-delay-s", type=float, default=DEFAULT_BROWSER_RETRY_DELAY_S)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535.")
    if (
        args.timeout_s <= 0
        or args.queue_timeout_s < 0
        or args.retry_delay_s < 0
        or args.pool_persist_every <= 0
    ):
        raise SystemExit(
            "timeouts and retry delay must be non-negative; "
            "timeout and --pool-persist-every must be positive."
        )
    text_attempts = args.text_upstream_attempts if args.text_upstream_attempts is not None else args.upstream_attempts
    text_timeout_s = args.text_timeout_s if args.text_timeout_s is not None else args.timeout_s
    text_retry_delay_s = args.text_retry_delay_s if args.text_retry_delay_s is not None else args.retry_delay_s
    browser_timeout_s = (
        args.browser_timeout_s
        if args.browser_timeout_s is not None
        else DEFAULT_BROWSER_TIMEOUT_S
    )
    attempts = {
        "text": text_attempts,
        "browser_image": args.browser_image_attempts,
        "browser_execute": args.browser_execute_attempts,
        "browser_create": args.browser_create_attempts,
        "browser_delete": args.browser_delete_attempts,
    }
    if args.max_inflight <= 0 or any(int(value) <= 0 for value in attempts.values()):
        raise SystemExit("--max-inflight and all attempt counts must be positive.")
    if any(float(value) <= 0 for value in (text_timeout_s, browser_timeout_s)):
        raise SystemExit("text and Browser timeouts must be positive.")
    if args.browser_retry_delay_s < 0 or text_retry_delay_s < 0:
        raise SystemExit("retry delays must be non-negative.")

    server_class = IPv6ThreadingHTTPServer if ":" in args.bind else ThreadingHTTPServer
    server = server_class((args.bind, args.port), FirecrawlRelayHandler)
    server.upstream_base = args.upstream
    server.timeout_s = args.timeout_s
    server.queue_timeout_s = args.queue_timeout_s
    server.text_timeout_s = text_timeout_s
    server.text_upstream_attempts = int(text_attempts)
    server.text_retry_delay_s = text_retry_delay_s
    server.browser_timeout_s = browser_timeout_s
    server.browser_image_attempts = int(args.browser_image_attempts)
    server.browser_execute_attempts = int(args.browser_execute_attempts)
    server.browser_create_attempts = int(args.browser_create_attempts)
    server.browser_delete_attempts = int(args.browser_delete_attempts)
    server.browser_retry_delay_s = args.browser_retry_delay_s
    server.key_pool = None
    server.browser_session_keys: dict[str, str] = {}
    server.browser_session_lock = threading.RLock()
    if args.keys_file:
        keys = FirecrawlApiKeyPool._load_keys(args.keys_file)
        server.key_pool = FirecrawlApiKeyPool(
            keys=keys,
            state_path=args.state_file,
            default_credits=args.pool_default_credits,
            persist_every=args.pool_persist_every,
        )

    server.upstream_slots = threading.BoundedSemaphore(args.max_inflight)
    print(
        f"Firecrawl relay listening on {args.bind}:{args.port}; "
        f"upstream={args.upstream}; max_inflight={args.max_inflight}; "
        f"text_attempts={server.text_upstream_attempts}; "
        f"browser_image_attempts={server.browser_image_attempts}; "
        f"browser_execute_attempts={server.browser_execute_attempts}; "
        f"browser_create_attempts={server.browser_create_attempts}; "
        "relay_authentication=none; "
        f"pool_managed={server.key_pool is not None}; "
        f"pool_keys={len(server.key_pool.keys) if server.key_pool is not None else 0}; "
        f"pool_persist_every={args.pool_persist_every}; "
        f"pool_state={server.key_pool.state_path if server.key_pool is not None else '-'}",
        flush=True,
    )
    print(
        "WARNING: relay authentication is disabled; restrict access with a firewall or private network.",
        file=sys.stderr,
        flush=True,
    )
    _install_shutdown_handlers(server)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Firecrawl relay.", flush=True)
    finally:
        if server.key_pool is not None:
            server.key_pool.flush()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
