#!/usr/bin/env python3
"""Unauthenticated HTTP fetch relay for image downloads.

The relay is intentionally a small, byte-preserving HTTP proxy for the
worker-side ``read_url`` image path.  A caller sends either::

    GET /fetch?url=https%3A%2F%2Fexample.com%2Fimage.jpg

or::

    POST /fetch
    {"url": "https://example.com/image.jpg"}

The relay downloads the target and returns its raw response body, status code,
content type, and a small amount of provenance in response headers.  It does
not expose a general CONNECT proxy and does not forward arbitrary request
headers.

Run this on a node with the desired public egress::

    python debug/http_relay.py --bind 0.0.0.0 --port 18083

Then on the worker::

    export HTTP_RELAY_URL='http://relay-host:18083/'

The worker-side integration is opt-in.  Without ``HTTP_RELAY_URL`` (or when
it is an empty string) the existing direct HTTP behavior is unchanged.  The
normal worker request form is ``http://relay-host:18083/<encoded-target-url>``.
The relay also accepts the explicit ``/fetch?url=...`` form.  The relay blocks private,
loopback, link-local, multicast, and metadata destinations, and validates
every redirect target to avoid turning the service into an SSRF proxy.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import socket
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, unquote, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


MAX_REQUEST_BODY_BYTES = 64 * 1024
MAX_RESPONSE_BODY_BYTES = 32 * 1024 * 1024
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_QUEUE_TIMEOUT_S = 60.0
DEFAULT_MAX_INFLIGHT = 8
DEFAULT_UPSTREAM_ATTEMPTS = 2
DEFAULT_RETRY_DELAY_S = 2.0
DEFAULT_MAX_REDIRECTS = 5
_TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}
_FORWARDED_REQUEST_HEADERS = {
    "User-Agent": "User-Agent",
    "Accept": "Accept",
    "Accept-Language": "Accept-Language",
    "Referer": "Referer",
}
_BLOCKED_HOSTNAMES = {
    "localhost",
    "localhost.localdomain",
    "metadata.google.internal",
    "metadata",
}


class _NoRedirectHandler(HTTPRedirectHandler):
    """Expose redirect responses so each Location can be validated."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        del req, fp, code, msg, headers, newurl
        return None


# Keep this as a module-level symbol so tests and operators can replace it
# without changing the relay implementation.
urlopen = build_opener(_NoRedirectHandler).open


def _json_body(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


def _read_limited(stream, *, limit: int) -> bytes:
    body = stream.read(limit + 1)
    if len(body) > limit:
        raise ValueError(f"upstream response exceeds the {limit} byte limit")
    return body


def _target_for_log(target_url: str) -> str:
    """Hide query strings because signed URLs may contain credentials."""

    parsed = urlsplit(target_url)
    path = parsed.path or "/"
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def _is_private_address(address: str) -> bool:
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return False
    return bool(
        parsed.is_private
        or parsed.is_loopback
        or parsed.is_link_local
        or parsed.is_multicast
        or parsed.is_unspecified
        or parsed.is_reserved
    )


def _resolve_public_addresses(hostname: str, port: int | None) -> None:
    normalized_hostname = hostname.rstrip(".").lower()
    if normalized_hostname in _BLOCKED_HOSTNAMES:
        raise ValueError(f"target hostname is blocked: {hostname!r}")

    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        literal = None
    if literal is not None:
        if _is_private_address(str(literal)):
            raise ValueError(f"target address is blocked: {hostname!r}")
        return

    try:
        addresses = socket.getaddrinfo(
            hostname,
            port or 443,
            type=socket.SOCK_STREAM,
        )
    except OSError as exc:
        raise ValueError(f"target hostname cannot be resolved: {hostname!r}") from exc
    resolved = {str(item[4][0]) for item in addresses if item[4]}
    if not resolved:
        raise ValueError(f"target hostname has no addresses: {hostname!r}")
    blocked = sorted(address for address in resolved if _is_private_address(address))
    if blocked:
        raise ValueError(
            f"target hostname resolves to a blocked address: {hostname!r} -> {blocked[0]}"
        )


def _validate_target_url(target_url: str) -> str:
    normalized = str(target_url or "").strip()
    parsed = urlsplit(normalized)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("target URL must use http or https")
    if not parsed.hostname:
        raise ValueError("target URL must include a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("target URL credentials are not allowed")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("target URL has an invalid port") from exc
    _resolve_public_addresses(parsed.hostname, port)
    return normalized


def _forwarded_headers(incoming_headers) -> dict[str, str]:  # type: ignore[no-untyped-def]
    forwarded: dict[str, str] = {}
    for source_name, target_name in _FORWARDED_REQUEST_HEADERS.items():
        value = str(incoming_headers.get(source_name) or "").strip()
        if value and "\r" not in value and "\n" not in value:
            forwarded[target_name] = value
    # Avoid compressed upstream responses: the relay returns the bytes as-is
    # and does not perform content decoding.
    forwarded["Accept-Encoding"] = "identity"
    return forwarded


def _fetch_once(
    target_url: str,
    *,
    timeout_s: float,
    max_redirects: int,
    request_headers: dict[str, str],
) -> tuple[bytes, int, str, str]:
    current_url = target_url
    for redirect_count in range(max_redirects + 1):
        current_url = _validate_target_url(current_url)
        request = Request(current_url, headers=request_headers, method="GET")
        try:
            with urlopen(request, timeout=timeout_s) as response:
                status_code = int(response.getcode() or 200)
                if 300 <= status_code < 400:
                    location = str(response.headers.get("Location") or "").strip()
                    response.read(0)
                    if not location:
                        raise ValueError(f"upstream redirect {status_code} has no Location")
                    if redirect_count >= max_redirects:
                        raise ValueError("upstream redirect limit exceeded")
                    current_url = urljoin(current_url, location)
                    continue
                body = _read_limited(response, limit=MAX_RESPONSE_BODY_BYTES)
                content_type = str(response.headers.get("Content-Type") or "application/octet-stream")
                return body, status_code, content_type, current_url
        except HTTPError as exc:
            status_code = int(exc.code)
            if 300 <= status_code < 400:
                location = str(exc.headers.get("Location") or "").strip()
                if not location:
                    raise ValueError(f"upstream redirect {status_code} has no Location") from exc
                if redirect_count >= max_redirects:
                    raise ValueError("upstream redirect limit exceeded") from exc
                current_url = urljoin(current_url, location)
                continue
            body = _read_limited(exc, limit=MAX_RESPONSE_BODY_BYTES)
            content_type = str(exc.headers.get("Content-Type") or "application/octet-stream")
            return body, status_code, content_type, current_url
    raise ValueError("upstream redirect limit exceeded")


def _retry_delay(*, attempt: int, base_delay_s: float, retry_after: str | None = None) -> float:
    delay = max(0.0, float(base_delay_s)) * (2 ** max(0, attempt - 1))
    if retry_after:
        try:
            delay = max(delay, float(retry_after))
        except (TypeError, ValueError):
            pass
    return delay


def _log_fetch(
    *,
    target_url: str,
    elapsed_s: float,
    attempt: int,
    status_code: int | None = None,
    byte_count: int | None = None,
    error: BaseException | None = None,
) -> None:
    fields = [
        f"target={_target_for_log(target_url)!r}",
        f"elapsed_s={elapsed_s:.3f}",
        f"attempt={attempt}",
    ]
    if status_code is not None:
        fields.append(f"status_code={status_code}")
    if byte_count is not None:
        fields.append(f"response_bytes={byte_count}")
    if error is not None:
        fields.append(f"error_type={error.__class__.__name__}")
        fields.append(f"error={str(error)!r}")
    print("[http-relay-fetch] " + " ".join(fields), file=sys.stderr, flush=True)


class IPv6ThreadingHTTPServer(ThreadingHTTPServer):
    address_family = socket.AF_INET6
    daemon_threads = True
    allow_reuse_address = True


class HttpRelayHandler(BaseHTTPRequestHandler):
    server_version = "OpenSearch-VL-HttpRelay/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write(f"[http-relay] {self.address_string()} {fmt % args}\n")
        sys.stderr.flush()

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlsplit(self.path).path
        if path == "/healthz":
            self._write_json(
                200,
                {
                    "ok": True,
                    "service": "http-relay",
                    "max_inflight": self.server.max_inflight,
                },
            )
            return
        if path == "/fetch":
            query = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
            values = query.get("url") or []
            if len(values) != 1:
                self._write_json(400, {"error": "request_query_requires_one_url"})
                return
            self._handle_fetch(str(values[0]))
            return
        # The worker's compact mode is Relay-base + encoded target URL.  It
        # avoids requiring a second switch in the worker downloader while
        # still preserving the target's query string losslessly.
        encoded_target = path.lstrip("/")
        if not encoded_target:
            self._write_json(404, {"error": "not_found"})
            return
        self._handle_fetch(unquote(encoded_target))

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if urlsplit(self.path).path != "/fetch":
            self._write_json(404, {"error": "not_found"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        if content_length <= 0 or content_length > MAX_REQUEST_BODY_BYTES:
            self._write_json(400, {"error": "invalid_content_length"})
            return
        try:
            payload = json.loads(self.rfile.read(content_length))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._write_json(400, {"error": "request_body_must_be_json"})
            return
        if not isinstance(payload, dict):
            self._write_json(400, {"error": "request_body_must_be_object"})
            return
        self._handle_fetch(str(payload.get("url") or ""))

    def _handle_fetch(self, target_url: str) -> None:
        try:
            target_url = _validate_target_url(target_url)
        except ValueError as exc:
            self._write_json(400, {"error": "invalid_target_url", "detail": str(exc)})
            return

        slots = self.server.upstream_slots
        if not slots.acquire(timeout=self.server.queue_timeout_s):
            self._write_json(503, {"error": "http_relay_busy"})
            return
        try:
            self._fetch_and_write(target_url)
        finally:
            slots.release()

    def _fetch_and_write(self, target_url: str) -> None:
        attempts = max(1, int(self.server.upstream_attempts))
        headers = _forwarded_headers(self.headers)
        last_error: BaseException | None = None
        for attempt in range(1, attempts + 1):
            started_at = time.perf_counter()
            try:
                body, status_code, content_type, final_url = _fetch_once(
                    target_url,
                    timeout_s=self.server.timeout_s,
                    max_redirects=self.server.max_redirects,
                    request_headers=headers,
                )
                if status_code in _TRANSIENT_STATUS_CODES and attempt < attempts:
                    _log_fetch(
                        target_url=target_url,
                        elapsed_s=time.perf_counter() - started_at,
                        attempt=attempt,
                        status_code=status_code,
                        byte_count=len(body),
                        error=RuntimeError("transient upstream HTTP status"),
                    )
                    time.sleep(
                        _retry_delay(
                            attempt=attempt,
                            base_delay_s=self.server.retry_delay_s,
                        )
                    )
                    continue
                _log_fetch(
                    target_url=target_url,
                    elapsed_s=time.perf_counter() - started_at,
                    attempt=attempt,
                    status_code=status_code,
                    byte_count=len(body),
                )
                self._write_bytes(
                    status_code,
                    body,
                    content_type,
                    extra_headers={
                        "X-Relay-Upstream-Status": str(status_code),
                        "X-Relay-Final-URL": final_url,
                    },
                )
                return
            except (URLError, TimeoutError, OSError) as exc:
                last_error = exc
                _log_fetch(
                    target_url=target_url,
                    elapsed_s=time.perf_counter() - started_at,
                    attempt=attempt,
                    error=exc,
                )
                if attempt < attempts:
                    time.sleep(_retry_delay(attempt=attempt, base_delay_s=self.server.retry_delay_s))
                    continue
                status_code = 504 if "timed out" in str(exc).lower() or isinstance(exc, TimeoutError) else 502
                self._write_json(
                    status_code,
                    {
                        "ok": False,
                        "error": "upstream_http_fetch_failed",
                        "error_type": exc.__class__.__name__,
                        "detail": str(getattr(exc, "reason", exc)),
                    },
                )
                return
            except ValueError as exc:
                last_error = exc
                _log_fetch(
                    target_url=target_url,
                    elapsed_s=time.perf_counter() - started_at,
                    attempt=attempt,
                    error=exc,
                )
                self._write_json(502, {"ok": False, "error": str(exc)})
                return

        self._write_json(
            502,
            {
                "ok": False,
                "error": "upstream_http_fetch_failed",
                "detail": str(last_error or "no response"),
            },
        )

    def _write_json(self, status_code: int, payload: object) -> None:
        self._write_bytes(status_code, _json_body(payload), "application/json")

    def _write_bytes(
        self,
        status_code: int,
        body: bytes,
        content_type: str,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        for name, value in (extra_headers or {}).items():
            if "\r" not in value and "\n" not in value:
                self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="::", help="Bind address; default is all IPv6 interfaces.")
    parser.add_argument("--port", type=int, default=18083, help="Listen port; default is 18083.")
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--queue-timeout-s", type=float, default=DEFAULT_QUEUE_TIMEOUT_S)
    parser.add_argument("--max-inflight", type=int, default=DEFAULT_MAX_INFLIGHT)
    parser.add_argument("--upstream-attempts", type=int, default=DEFAULT_UPSTREAM_ATTEMPTS)
    parser.add_argument("--retry-delay-s", type=float, default=DEFAULT_RETRY_DELAY_S)
    parser.add_argument("--max-redirects", type=int, default=DEFAULT_MAX_REDIRECTS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535.")
    if args.timeout_s <= 0 or args.queue_timeout_s < 0 or args.retry_delay_s < 0:
        raise SystemExit("timeout must be positive; queue timeout and retry delay must be non-negative.")
    if args.max_inflight <= 0 or args.upstream_attempts <= 0 or args.max_redirects < 0:
        raise SystemExit("max-inflight and upstream-attempts must be positive; max-redirects must be non-negative.")

    server_class = IPv6ThreadingHTTPServer if ":" in args.bind else ThreadingHTTPServer
    server = server_class((args.bind, args.port), HttpRelayHandler)
    server.timeout_s = args.timeout_s
    server.queue_timeout_s = args.queue_timeout_s
    server.max_inflight = args.max_inflight
    server.upstream_attempts = args.upstream_attempts
    server.retry_delay_s = args.retry_delay_s
    server.max_redirects = args.max_redirects
    import threading

    server.upstream_slots = threading.BoundedSemaphore(args.max_inflight)
    print(
        f"HTTP relay listening on {args.bind}:{args.port}; "
        f"max_inflight={args.max_inflight}; timeout_s={args.timeout_s:g}; "
        f"attempts={args.upstream_attempts}; authentication=disabled",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping HTTP relay.", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
