#!/usr/bin/env python3
"""Small authenticated Serper relay for an IPv6-only worker.

Run this on a machine that can reach ``google.serper.dev`` and point the
caller at ``/search``, ``/images`` and ``/lens`` on this relay.  The caller supplies its
Serper API key in ``X-API-KEY``; the relay forwards only the JSON body and that
header to the official Serper endpoint.  A shared relay token can optionally
be enabled when the private network is not sufficient.

Example on server B:
  SERPER_RELAY_TOKEN='replace-with-a-long-random-token' \
    python debug/serper_relay.py --bind '::' --port 18080
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import socket
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MAX_BODY_BYTES = 2 * 1024 * 1024
UPSTREAM_MAX_ATTEMPTS = 2
UPSTREAM_RETRY_DELAY_S = 5.0


def _log_upstream(*, path: str, status_code: int | None = None, elapsed_s: float, response_chars: int | None = None, error: Exception | None = None) -> None:
    fields = [f"path={path!r}", f"elapsed_s={elapsed_s:.3f}"]
    if status_code is not None:
        fields.append(f"status_code={status_code}")
    if response_chars is not None:
        fields.append(f"response_bytes={response_chars}")
    if error is not None:
        fields.append(f"error_type={error.__class__.__name__}")
        fields.append(f"error={str(getattr(error, 'reason', error))!r}")
    print("[serper-relay-upstream] " + " ".join(fields), file=sys.stderr, flush=True)


class IPv6ThreadingHTTPServer(ThreadingHTTPServer):
    address_family = socket.AF_INET6
    daemon_threads = True
    allow_reuse_address = True


class SerperRelayHandler(BaseHTTPRequestHandler):
    server_version = "OpenSearch-VL-SerperRelay/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        # Do not log request headers, bodies, or API keys.
        sys.stderr.write(f"[serper-relay] {self.address_string()} {fmt % args}\n")
        sys.stderr.flush()

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path.split("?", 1)[0] != "/healthz":
            self._write_json(404, {"error": "not_found"})
            return
        self._write_json(200, {"ok": True, "service": "serper-relay"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = self.path.split("?", 1)[0]
        if path not in {"/search", "/images", "/lens"}:
            self._write_json(404, {"error": "unsupported_path"})
            return

        expected_token = str(self.server.relay_token or "")
        if expected_token:
            received_token = self.headers.get("X-Serper-Relay-Token", "")
            if not hmac.compare_digest(received_token, expected_token):
                self._write_json(403, {"error": "invalid_relay_token"})
                return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        if content_length <= 0 or content_length > MAX_BODY_BYTES:
            self._write_json(400, {"error": "invalid_content_length"})
            return

        body = self.rfile.read(content_length)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self._write_json(400, {"error": "request_body_must_be_json"})
            return
        if not isinstance(payload, dict):
            self._write_json(400, {"error": "request_body_must_be_object"})
            return

        api_key = str(self.headers.get("X-API-KEY", "")).strip()
        if not api_key:
            self._write_json(400, {"error": "missing_x_api_key"})
            return

        upstream_base = str(self.server.upstream_base).rstrip("/")
        upstream_url = f"{upstream_base}{path}"
        upstream_request = Request(
            upstream_url,
            data=body,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "X-API-KEY": api_key,
            },
            method="POST",
        )
        response_body: bytes | None = None
        status_code = 502
        content_type = "application/json"
        transport_error: Exception | None = None

        for attempt in range(1, UPSTREAM_MAX_ATTEMPTS + 1):
            upstream_started_at = time.perf_counter()
            try:
                with urlopen(upstream_request, timeout=float(self.server.timeout_s)) as response:
                    response_body = response.read()
                    status_code = int(response.getcode() or 200)
                    content_type = response.headers.get("Content-Type", "application/json")
                _log_upstream(
                    path=path,
                    status_code=status_code,
                    elapsed_s=time.perf_counter() - upstream_started_at,
                    response_chars=len(response_body),
                )
                break
            except HTTPError as exc:
                response_body = exc.read()
                status_code = int(exc.code)
                content_type = exc.headers.get("Content-Type", "application/json")
                _log_upstream(
                    path=path,
                    status_code=status_code,
                    elapsed_s=time.perf_counter() - upstream_started_at,
                    response_chars=len(response_body),
                )
                # Retry transient upstream failures, but forward permanent
                # client/authentication errors immediately.
                if status_code >= 500 and attempt < UPSTREAM_MAX_ATTEMPTS:
                    print(
                        f"[serper-relay-upstream] retrying path={path!r} "
                        f"attempt={attempt + 1}/{UPSTREAM_MAX_ATTEMPTS} "
                        f"after_s={UPSTREAM_RETRY_DELAY_S}",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(UPSTREAM_RETRY_DELAY_S)
                    continue
                break
            except (URLError, TimeoutError, OSError) as exc:
                transport_error = exc
                _log_upstream(
                    path=path,
                    elapsed_s=time.perf_counter() - upstream_started_at,
                    error=exc,
                )
                if attempt < UPSTREAM_MAX_ATTEMPTS:
                    print(
                        f"[serper-relay-upstream] retrying path={path!r} "
                        f"attempt={attempt + 1}/{UPSTREAM_MAX_ATTEMPTS} "
                        f"after_s={UPSTREAM_RETRY_DELAY_S}",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(UPSTREAM_RETRY_DELAY_S)
                    continue
                self._write_json(
                    502,
                    {
                        "error": "upstream_serper_request_failed",
                        "error_type": exc.__class__.__name__,
                        "detail": str(getattr(exc, "reason", exc)),
                    },
                )
                return

        if response_body is None:
            # This is defensive: all normal loop exits either produce a
            # response body or return the 502 transport error above.
            error = transport_error or RuntimeError("upstream request produced no response")
            self._write_json(
                502,
                {
                    "error": "upstream_serper_request_failed",
                    "error_type": error.__class__.__name__,
                    "detail": str(getattr(error, "reason", error)),
                },
            )
            return

        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(response_body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(response_body)

    def _write_json(self, status_code: int, payload: dict[str, object]) -> None:
        body = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="::", help="IPv6 bind address (default: ::).")
    parser.add_argument("--port", type=int, default=18080, help="Listen port (default: 18080).")
    parser.add_argument(
        "--token",
        default=os.environ.get("SERPER_RELAY_TOKEN"),
        help="Optional shared relay token; defaults to SERPER_RELAY_TOKEN.",
    )
    parser.add_argument(
        "--upstream",
        default="https://google.serper.dev",
        help="Official Serper upstream base URL.",
    )
    parser.add_argument("--timeout-s", type=float, default=60.0, help="Upstream timeout.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535.")
    if args.timeout_s <= 0:
        raise SystemExit("--timeout-s must be positive.")

    server = IPv6ThreadingHTTPServer((args.bind, args.port), SerperRelayHandler)
    server.relay_token = args.token
    server.upstream_base = args.upstream
    server.timeout_s = args.timeout_s
    print(
        f"Serper relay listening on [{args.bind}]:{args.port}; "
        f"upstream={args.upstream}; token_required={bool(args.token)}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Serper relay.", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
