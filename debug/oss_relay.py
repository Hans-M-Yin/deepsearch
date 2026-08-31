#!/usr/bin/env python3
"""Authenticated Alibaba OSS upload relay.

Run this service on a node with reliable access to the configured Alibaba OSS
endpoint.  Workers send image bytes to this relay instead of opening a direct
OSS connection.  The OSS access key and secret stay on the relay node.

The public API is intentionally small:

* ``GET /healthz`` -- liveness check;
* ``POST /upload`` -- upload one base64-encoded image.

The upload request is a JSON object with ``image_base64``, ``filename``,
``date_str``, ``mode``, ``user`` and optional ``use_direct_url`` fields.  The
object-key layout and returned public URL match the existing uploader.

Example on the OSS-egress node::

    OSS_ACCESS_KEY_ID='...' \
    OSS_ACCESS_KEY_SECRET='...' \
    OSS_ENDPOINT='oss-us-west-1.aliyuncs.com' \
    OSS_BUCKET_NAME='search-hans-us' \
    OSS_RELAY_TOKEN='replace-with-a-long-random-token' \
    python debug/oss_relay.py --bind 0.0.0.0 --port 18082

Then on the worker::

    export OSS_RELAY_URL='http://relay-host:18082'
    export OSS_RELAY_TOKEN='replace-with-a-long-random-token'

The relay applies bounded concurrency and retries only inside the relay for
transient OSS failures.  The existing worker-side retry policy remains active
as a second, end-to-end retry layer.
"""

from __future__ import annotations

import argparse
import base64
import hmac
import json
import mimetypes
import os
import re
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


MAX_REQUEST_BODY_BYTES = 64 * 1024 * 1024
MAX_DECODED_IMAGE_BYTES = 32 * 1024 * 1024
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_QUEUE_TIMEOUT_S = 60.0
DEFAULT_MAX_INFLIGHT = 8
DEFAULT_UPSTREAM_ATTEMPTS = 2
DEFAULT_RETRY_DELAY_S = 3.0
_SAFE_SEGMENT_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _json_body(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


def _read_limited(stream, *, limit: int) -> bytes:
    body = stream.read(limit + 1)
    if len(body) > limit:
        raise ValueError(f"response exceeds the {limit} byte limit")
    return body


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _normalise_endpoint(endpoint: str) -> tuple[str, str]:
    endpoint = endpoint.strip()
    if not endpoint:
        raise RuntimeError("OSS_ENDPOINT is empty")
    if endpoint.startswith("http://"):
        scheme = "http"
        host = endpoint[len("http://") :]
    elif endpoint.startswith("https://"):
        scheme = "https"
        host = endpoint[len("https://") :]
    else:
        scheme = "https"
        host = endpoint
    return scheme, host.rstrip("/")


def _safe_segment(value: object, *, default: str) -> str:
    segment = _SAFE_SEGMENT_RE.sub("_", str(value or "").strip()).strip("._-")
    return segment or default


def _content_type_for(filename: str) -> str:
    content_type, _ = mimetypes.guess_type(filename)
    return content_type or "application/octet-stream"


def _object_key(*, date_str: object, mode: object, user: object, filename: object) -> str:
    safe_date = _safe_segment(date_str, default="unknown-date")
    safe_mode = _safe_segment(mode, default="default")
    safe_user = _safe_segment(user, default="opensearch-vl")
    safe_filename = os.path.basename(str(filename or "")).strip()
    safe_filename = _safe_segment(safe_filename, default="image.png")
    return f"vision_deepresearch/{safe_date}/{safe_mode}/{safe_user}/{safe_filename}"


def _status_code_from_exception(exc: BaseException) -> int | None:
    for candidate in (exc, getattr(exc, "response", None)):
        if candidate is None:
            continue
        for attribute in ("status_code", "status", "code"):
            value = getattr(candidate, attribute, None)
            try:
                status_code = int(value)
            except (TypeError, ValueError):
                continue
            if 100 <= status_code <= 599:
                return status_code
    return None


def _is_retryable(exc: BaseException) -> bool:
    status_code = _status_code_from_exception(exc)
    if status_code is None:
        # OSS SDK transport errors and socket timeouts generally do not carry
        # an HTTP status.  They are the primary reason for using this relay.
        return True
    return status_code in {408, 429} or status_code >= 500


def _decode_image(payload: dict[str, object]) -> bytes:
    encoded = payload.get("image_base64")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("request_body_requires_image_base64")
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("image_base64_is_invalid") from exc
    if not image_bytes:
        raise ValueError("image_base64_is_empty")
    if len(image_bytes) > MAX_DECODED_IMAGE_BYTES:
        raise ValueError(
            f"decoded image exceeds the {MAX_DECODED_IMAGE_BYTES} byte limit"
        )
    return image_bytes


def _upload_once(payload: dict[str, object], *, timeout_s: float) -> tuple[str, str | None]:
    try:
        import oss2
    except ImportError as exc:  # pragma: no cover - depends on relay environment
        raise RuntimeError("oss2 is required by the OSS relay") from exc

    image_bytes = _decode_image(payload)
    access_key_id = _require_env("OSS_ACCESS_KEY_ID")
    access_key_secret = _require_env("OSS_ACCESS_KEY_SECRET")
    endpoint = _require_env("OSS_ENDPOINT")
    bucket_name = _require_env("OSS_BUCKET_NAME")
    scheme, endpoint_host = _normalise_endpoint(endpoint)
    object_key = _object_key(
        date_str=payload.get("date_str"),
        mode=payload.get("mode"),
        user=payload.get("user"),
        filename=payload.get("filename"),
    )

    auth = oss2.Auth(access_key_id, access_key_secret)
    bucket = oss2.Bucket(
        auth,
        f"{scheme}://{endpoint_host}",
        bucket_name,
        connect_timeout=max(1, int(timeout_s)),
    )
    safe_filename = os.path.basename(str(payload.get("filename") or "image.png"))
    result = bucket.put_object(
        object_key,
        image_bytes,
        headers={
            "Content-Type": _content_type_for(safe_filename),
            "Content-Disposition": f'inline; filename="{safe_filename}"',
        },
    )
    status_code = int(getattr(result, "status", 200) or 200)
    if status_code >= 400:
        raise RuntimeError(f"OSS upload returned HTTP {status_code}")

    public_url = None
    if bool(payload.get("use_direct_url", True)):
        public_url = f"{scheme}://{bucket_name}.{endpoint_host}/{object_key}"
    return object_key, public_url


def _log_upload(
    *,
    elapsed_s: float,
    attempt: int,
    status_code: int | None = None,
    error: BaseException | None = None,
) -> None:
    fields = [f"elapsed_s={elapsed_s:.3f}", f"attempt={attempt}"]
    if status_code is not None:
        fields.append(f"status_code={status_code}")
    if error is not None:
        fields.append(f"error_type={error.__class__.__name__}")
        fields.append(f"error={str(error)!r}")
    print("[oss-relay-upload] " + " ".join(fields), file=sys.stderr, flush=True)


class IPv6ThreadingHTTPServer(ThreadingHTTPServer):
    address_family = socket.AF_INET6
    daemon_threads = True
    allow_reuse_address = True


class OSSRelayHandler(BaseHTTPRequestHandler):
    server_version = "OpenSearch-VL-OSSRelay/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        # Never log request bodies or authentication headers.
        sys.stderr.write(f"[oss-relay] {self.address_string()} {fmt % args}\n")
        sys.stderr.flush()

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path.split("?", 1)[0] == "/healthz":
            self._write_json(
                200,
                {
                    "ok": True,
                    "service": "oss-relay",
                    "max_inflight": self.server.max_inflight,
                },
            )
            return
        self._write_json(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = self.path.split("?", 1)[0]
        if path != "/upload":
            self._write_json(404, {"error": "unsupported_path"})
            return

        expected_token = str(self.server.relay_token or "")
        if expected_token:
            received_token = self.headers.get("X-OSS-Relay-Token", "")
            if not hmac.compare_digest(received_token, expected_token):
                self._write_json(403, {"error": "invalid_relay_token"})
                return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        if content_length <= 0 or content_length > MAX_REQUEST_BODY_BYTES:
            self._write_json(400, {"error": "invalid_content_length"})
            return

        try:
            body = self.rfile.read(content_length)
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._write_json(400, {"error": "request_body_must_be_json"})
            return
        if not isinstance(payload, dict):
            self._write_json(400, {"error": "request_body_must_be_object"})
            return
        if not str(payload.get("filename") or "").strip():
            self._write_json(400, {"error": "request_body_requires_filename"})
            return

        try:
            # Validate before occupying an OSS slot so malformed requests do
            # not consume relay capacity.
            _decode_image(payload)
        except ValueError as exc:
            self._write_json(400, {"error": str(exc)})
            return

        slots = self.server.upstream_slots
        if not slots.acquire(timeout=self.server.queue_timeout_s):
            self._write_json(503, {"error": "oss_relay_busy"})
            return
        try:
            self._proxy_upload(payload)
        finally:
            slots.release()

    def _proxy_upload(self, payload: dict[str, object]) -> None:
        attempts = max(1, int(self.server.upstream_attempts))
        for attempt in range(1, attempts + 1):
            started_at = time.perf_counter()
            try:
                object_key, public_url = _upload_once(
                    payload,
                    timeout_s=self.server.timeout_s,
                )
                _log_upload(
                    elapsed_s=time.perf_counter() - started_at,
                    attempt=attempt,
                    status_code=200,
                )
                self._write_json(
                    200,
                    {"ok": True, "object_key": object_key, "url": public_url},
                )
                return
            except Exception as exc:
                _log_upload(
                    elapsed_s=time.perf_counter() - started_at,
                    attempt=attempt,
                    status_code=_status_code_from_exception(exc),
                    error=exc,
                )
                if attempt < attempts and _is_retryable(exc):
                    time.sleep(self.server.retry_delay_s * attempt)
                    continue
                self._write_json(
                    502 if _is_retryable(exc) else 500,
                    {
                        "ok": False,
                        "error": "oss_upload_failed",
                        "error_type": exc.__class__.__name__,
                        "detail": str(exc),
                    },
                )
                return

    def _write_json(self, status_code: int, payload: object) -> None:
        body = _json_body(payload)
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="::", help="Bind address; default is all IPv6 interfaces.")
    parser.add_argument("--port", type=int, default=18082, help="Listen port; default is 18082.")
    parser.add_argument(
        "--token",
        default=os.environ.get("OSS_RELAY_TOKEN"),
        help="Optional shared relay token; defaults to OSS_RELAY_TOKEN.",
    )
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S, help="OSS connection timeout.")
    parser.add_argument(
        "--queue-timeout-s",
        type=float,
        default=DEFAULT_QUEUE_TIMEOUT_S,
        help="Maximum time to wait for an OSS concurrency slot.",
    )
    parser.add_argument(
        "--max-inflight",
        type=int,
        default=DEFAULT_MAX_INFLIGHT,
        help="Maximum concurrent OSS uploads.",
    )
    parser.add_argument(
        "--upstream-attempts",
        type=int,
        default=DEFAULT_UPSTREAM_ATTEMPTS,
        help="Total attempts for transient OSS failures.",
    )
    parser.add_argument(
        "--retry-delay-s",
        type=float,
        default=DEFAULT_RETRY_DELAY_S,
        help="Base delay between relay-side retries.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535.")
    if args.timeout_s <= 0 or args.queue_timeout_s < 0 or args.retry_delay_s < 0:
        raise SystemExit("timeouts and retry delay must be non-negative; timeout must be positive.")
    if args.max_inflight <= 0 or args.upstream_attempts <= 0:
        raise SystemExit("--max-inflight and --upstream-attempts must be positive.")

    server_class = IPv6ThreadingHTTPServer if ":" in args.bind else ThreadingHTTPServer
    server = server_class((args.bind, args.port), OSSRelayHandler)
    server.relay_token = args.token
    server.timeout_s = args.timeout_s
    server.queue_timeout_s = args.queue_timeout_s
    server.max_inflight = args.max_inflight
    server.upstream_attempts = args.upstream_attempts
    server.retry_delay_s = args.retry_delay_s
    server.upstream_slots = threading.BoundedSemaphore(args.max_inflight)
    print(
        f"OSS relay listening on {args.bind}:{args.port}; "
        f"max_inflight={args.max_inflight}; token_required={bool(args.token)}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping OSS relay.", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
