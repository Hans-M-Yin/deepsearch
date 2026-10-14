#!/usr/bin/env python3
"""Small authenticated Serper relay for an IPv6-only worker.

Run this on a machine that can reach ``google.serper.dev`` and point the
caller at ``/search``, ``/images`` and ``/lens`` on this relay.  The caller
supplies a compatibility key in ``X-API-KEY``.  When a relay pool
is configured, that value is ignored and the relay selects the real upstream
key from its own pool.  A shared relay token can optionally be enabled when
the private network is not sufficient.

Example on server B:
  SERPER_RELAY_TOKEN='replace-with-a-long-random-token' \
    python debug/serper_relay.py --bind '::' --port 18080
"""

from __future__ import annotations

import argparse
import hmac
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
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synthesis.search_client import (
    SerperApiKeyPool,
    _is_serper_auth_failure,
    _is_serper_credit_exhausted,
    _serper_timing_debug,
)


MAX_BODY_BYTES = 2 * 1024 * 1024
UPSTREAM_MAX_ATTEMPTS = 2
UPSTREAM_RETRY_DELAY_S = 5.0
DEFAULT_POOL_PERSIST_EVERY = 10000
DEFAULT_REQUEST_QUEUE_SIZE = 512
DEFAULT_MAX_INFLIGHT = 128


def _log_upstream(*, path: str, request_id: str = "-", attempt: int | None = None, status_code: int | None = None, elapsed_s: float, response_chars: int | None = None, error: Exception | None = None) -> None:
    fields = [f"request_id={request_id!r}", f"path={path!r}", f"elapsed_s={elapsed_s:.3f}"]
    if attempt is not None:
        fields.append(f"attempt={attempt}")
    if status_code is not None:
        fields.append(f"status_code={status_code}")
    if response_chars is not None:
        fields.append(f"response_bytes={response_chars}")
    if error is not None:
        fields.append(f"error_type={error.__class__.__name__}")
        fields.append(f"error={str(getattr(error, 'reason', error))!r}")
    print("[serper-relay-upstream] " + " ".join(fields), file=sys.stderr, flush=True)


def _install_shutdown_handlers(server: "IPv6ThreadingHTTPServer") -> None:
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
            f"Received {signal_name}; flushing Serper pool state before shutdown.",
            flush=True,
        )
        relay_pool = getattr(server, "key_pool", None)
        if relay_pool is not None:
            try:
                relay_pool.flush()
            except Exception as exc:  # pragma: no cover - defensive shutdown path
                print(
                    f"Failed to flush Serper pool state during shutdown: {exc!r}",
                    file=sys.stderr,
                    flush=True,
                )
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)


class IPv6ThreadingHTTPServer(ThreadingHTTPServer):
    address_family = socket.AF_INET6
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        request_handler_class: type[BaseHTTPRequestHandler],
        *,
        request_queue_size: int = DEFAULT_REQUEST_QUEUE_SIZE,
        max_inflight: int = DEFAULT_MAX_INFLIGHT,
    ) -> None:
        # socketserver.TCPServer defaults this to 5.  At high fan-in that
        # causes connections to be reset before BaseHTTPRequestHandler sees
        # them, so no relay application log is emitted.  Set it before the
        # parent constructor calls listen().
        if request_queue_size <= 0:
            raise ValueError("request_queue_size must be positive")
        if max_inflight <= 0:
            raise ValueError("max_inflight must be positive")
        self.request_queue_size = int(request_queue_size)
        self.max_inflight = int(max_inflight)
        # The stdlib ThreadingHTTPServer creates one handler thread per
        # accepted connection.  This semaphore limits only the expensive
        # upstream work, so excess requests wait instead of overwhelming the
        # relay's egress/NAT gateway or the Serper upstream.
        self.inflight_semaphore = threading.BoundedSemaphore(self.max_inflight)
        super().__init__(server_address, request_handler_class)


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
        request_started_at = time.perf_counter()
        request_id = self.headers.get("X-Request-ID", "-")
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

        body_read_started_at = time.perf_counter()
        body = self.rfile.read(content_length)
        body_read_elapsed_s = time.perf_counter() - body_read_started_at
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self._write_json(400, {"error": "request_body_must_be_json"})
            return
        if not isinstance(payload, dict):
            self._write_json(400, {"error": "request_body_must_be_object"})
            return

        client_api_key = str(self.headers.get("X-API-KEY", "")).strip()
        if not client_api_key:
            self._write_json(400, {"error": "missing_x_api_key"})
            return

        relay_pool = getattr(self.server, "key_pool", None)
        upstream_base = str(self.server.upstream_base).rstrip("/")
        upstream_url = f"{upstream_base}{path}"
        pool_status_elapsed_s = 0.0
        if relay_pool is not None:
            try:
                pool_status_started_at = time.perf_counter()
                pool_status = relay_pool.status()
                pool_status_elapsed_s = time.perf_counter() - pool_status_started_at
                key_attempt_budget = max(1, int(pool_status.get("available_key_count") or 0))
            except RuntimeError as exc:
                self._write_json(
                    503,
                    {"error": "serper_relay_no_available_key", "detail": str(exc)},
                )
                return
        else:
            # Backward-compatible mode: without --keys-file, forward the
            # caller-provided key as the relay did before pool management.
            key_attempt_budget = 1

        key_acquire_total_s = 0.0
        first_upstream_started_at: float | None = None
        for key_attempt in range(1, key_attempt_budget + 1):
            key_id: str | None = None
            if relay_pool is not None:
                try:
                    key_acquire_started_at = time.perf_counter()
                    try:
                        api_key, pool_metadata = relay_pool.acquire_key()
                    finally:
                        key_acquire_total_s += time.perf_counter() - key_acquire_started_at
                    key_id = str(pool_metadata.get("key_id") or "")
                except RuntimeError as exc:
                    self._write_json(
                        503,
                        {"error": "serper_relay_no_available_key", "detail": str(exc)},
                    )
                    return
            else:
                api_key = client_api_key

            upstream_started_at = time.perf_counter()
            if first_upstream_started_at is None:
                first_upstream_started_at = upstream_started_at
            try:
                response_body, status_code, content_type = self._fetch_upstream_limited(
                    path=path,
                    upstream_url=upstream_url,
                    body=body,
                    api_key=api_key,
                )
            except (URLError, TimeoutError, OSError) as exc:
                _serper_timing_debug(
                    "relay_upstream_error",
                    elapsed_s=time.perf_counter() - request_started_at,
                    force=True,
                    request_id=request_id,
                    path=path,
                    queue_wait_s=round(float(getattr(self, "_last_queue_wait_s", 0.0)), 3),
                    error_type=type(exc).__name__,
                    error=str(getattr(exc, "reason", exc))[:500],
                )
                self._write_json(
                    502,
                    {
                        "error": "upstream_serper_request_failed",
                        "error_type": exc.__class__.__name__,
                        "detail": str(getattr(exc, "reason", exc)),
                    },
                )
                return
            upstream_elapsed_s = time.perf_counter() - upstream_started_at
            queue_wait_s = float(getattr(self, "_last_queue_wait_s", 0.0))

            self._record_pool_result(
                key_id=key_id,
                status_code=status_code,
                response_body=response_body,
            )
            response_text = response_body.decode("utf-8", errors="replace")
            credit_exhausted = relay_pool is not None and _is_serper_credit_exhausted(
                status_code=status_code,
                response_body=response_text,
            )
            auth_failure = relay_pool is not None and _is_serper_auth_failure(
                status_code=status_code,
                response_body=response_text,
            )
            if (credit_exhausted or auth_failure) and key_attempt < key_attempt_budget:
                failure_kind = (
                    "credits exhausted" if credit_exhausted else "authentication failed"
                )
                print(
                    f"[serper-relay] upstream key unusable ({failure_kind}); rotating "
                    f"key_attempt={key_attempt + 1}/{key_attempt_budget} path={path!r}",
                    file=sys.stderr,
                    flush=True,
                )
                continue

            self.send_response(status_code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(response_body)))
            self.send_header("Connection", "close")
            self.end_headers()
            write_started_at = time.perf_counter()
            write_error: Exception | None = None
            try:
                self.wfile.write(response_body)
            except (BrokenPipeError, ConnectionResetError) as exc:
                # The caller may have timed out while this request was waiting
                # for max-inflight.  Keep the relay alive and retain timing
                # information without emitting a full handler traceback.
                write_error = exc
                self.close_connection = True
            write_elapsed_s = time.perf_counter() - write_started_at
            total_elapsed_s = time.perf_counter() - request_started_at
            _serper_timing_debug(
                "relay_request",
                elapsed_s=total_elapsed_s,
                force=write_error is not None,
                request_id=request_id,
                path=path,
                status_code=status_code,
                queue_wait_s=round(queue_wait_s, 3),
                upstream_s=round(upstream_elapsed_s, 3),
                response_write_s=round(write_elapsed_s, 3),
                body_read_s=round(body_read_elapsed_s, 3),
                pool_status_s=round(pool_status_elapsed_s, 3),
                key_acquire_s=round(key_acquire_total_s, 3),
                pre_upstream_s=round(
                    (first_upstream_started_at - request_started_at)
                    if first_upstream_started_at is not None
                    else 0.0,
                    3,
                ),
                response_bytes=len(response_body),
                client_disconnected=write_error is not None,
                error_type=type(write_error).__name__ if write_error else None,
            )
            return

    def _fetch_upstream_limited(
        self,
        *,
        path: str,
        upstream_url: str,
        body: bytes,
        api_key: str,
    ) -> tuple[bytes, int, str]:
        """Run upstream I/O under the relay-wide concurrency limit.

        Some unit tests construct the stdlib ``ThreadingHTTPServer`` directly;
        those servers do not have a semaphore, so they retain the historical
        unlimited behavior.  The production IPv6 relay always installs one.
        """
        gate = getattr(self.server, "inflight_semaphore", None)
        if gate is None:
            self._last_queue_wait_s = 0.0
            return self._fetch_upstream(
                path=path,
                upstream_url=upstream_url,
                body=body,
                api_key=api_key,
            )

        wait_started_at = time.perf_counter()
        gate.acquire()
        wait_s = time.perf_counter() - wait_started_at
        self._last_queue_wait_s = wait_s
        if wait_s >= 0.01:
            print(
                f"[serper-relay] max-inflight wait_s={wait_s:.3f} "
                f"max_inflight={self.server.max_inflight} path={path!r} "
                f"request_id={self.headers.get('X-Request-ID', '-')!r}",
                file=sys.stderr,
                flush=True,
            )
        try:
            return self._fetch_upstream(
                path=path,
                upstream_url=upstream_url,
                body=body,
                api_key=api_key,
            )
        finally:
            gate.release()

    def _fetch_upstream(
        self,
        *,
        path: str,
        upstream_url: str,
        body: bytes,
        api_key: str,
    ) -> tuple[bytes, int, str]:
        """Fetch one request with transient retries, using one selected key."""

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
        for attempt in range(1, UPSTREAM_MAX_ATTEMPTS + 1):
            upstream_started_at = time.perf_counter()
            try:
                with urlopen(upstream_request, timeout=float(self.server.timeout_s)) as response:
                    response_body = response.read()
                    status_code = int(response.getcode() or 200)
                    content_type = response.headers.get("Content-Type", "application/json")
                _log_upstream(
                    path=path,
                    request_id=self.headers.get("X-Request-ID", "-"),
                    attempt=attempt,
                    status_code=status_code,
                    elapsed_s=time.perf_counter() - upstream_started_at,
                    response_chars=len(response_body),
                )
                return response_body, status_code, content_type
            except HTTPError as exc:
                response_body = exc.read()
                status_code = int(exc.code)
                content_type = (exc.headers or {}).get("Content-Type", "application/json")
                _log_upstream(
                    path=path,
                    request_id=self.headers.get("X-Request-ID", "-"),
                    attempt=attempt,
                    status_code=status_code,
                    elapsed_s=time.perf_counter() - upstream_started_at,
                    response_chars=len(response_body),
                )
                # Retry transient upstream failures, but let the caller
                # rotate keys when the final response reports exhausted credit.
                if status_code >= 500 and attempt < UPSTREAM_MAX_ATTEMPTS:
                    print(
                        f"[serper-relay-upstream] retrying "
                        f"request_id={self.headers.get('X-Request-ID', '-')!r} path={path!r} "
                        f"attempt={attempt + 1}/{UPSTREAM_MAX_ATTEMPTS} "
                        f"after_s={UPSTREAM_RETRY_DELAY_S}",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(UPSTREAM_RETRY_DELAY_S)
                    continue
                return response_body, status_code, content_type
            except (URLError, TimeoutError, OSError) as exc:
                _log_upstream(
                    path=path,
                    request_id=self.headers.get("X-Request-ID", "-"),
                    attempt=attempt,
                    elapsed_s=time.perf_counter() - upstream_started_at,
                    error=exc,
                )
                if attempt < UPSTREAM_MAX_ATTEMPTS:
                    print(
                        f"[serper-relay-upstream] retrying "
                        f"request_id={self.headers.get('X-Request-ID', '-')!r} path={path!r} "
                        f"attempt={attempt + 1}/{UPSTREAM_MAX_ATTEMPTS} "
                        f"after_s={UPSTREAM_RETRY_DELAY_S}",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(UPSTREAM_RETRY_DELAY_S)
                    continue
                raise

        raise RuntimeError("upstream Serper request produced no response")

    def _record_pool_result(
        self,
        *,
        key_id: str | None,
        status_code: int,
        response_body: bytes,
    ) -> None:
        relay_pool = getattr(self.server, "key_pool", None)
        if not key_id or relay_pool is None:
            return
        body_text = response_body.decode("utf-8", errors="replace")
        if _is_serper_credit_exhausted(
            status_code=status_code,
            response_body=body_text,
        ):
            relay_pool.mark_credits_exhausted(
                key_id,
                reason=f"serper_relay_http_{status_code}_credits_exhausted",
            )
        elif _is_serper_auth_failure(
            status_code=status_code,
            response_body=body_text,
        ):
            relay_pool.mark_credits_exhausted(
                key_id,
                reason=f"serper_relay_http_{status_code}_unauthorized",
            )

    def _write_json(self, status_code: int, payload: dict[str, object]) -> None:
        body = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="::", help="IPv6 bind address (default: ::).")
    parser.add_argument("--port", type=int, default=18080, help="Listen port (default: 18080).")
    parser.add_argument(
        "--request-queue-size",
        type=int,
        default=int(
            os.environ.get("SERPER_RELAY_REQUEST_QUEUE_SIZE")
            or DEFAULT_REQUEST_QUEUE_SIZE
        ),
        help=(
            "TCP listen backlog before requests enter Python handlers "
            f"(default: {DEFAULT_REQUEST_QUEUE_SIZE}; Python's default is 5)."
        ),
    )
    parser.add_argument(
        "--max-inflight",
        type=int,
        default=int(
            os.environ.get("SERPER_RELAY_MAX_INFLIGHT") or DEFAULT_MAX_INFLIGHT
        ),
        help=(
            "Maximum number of simultaneous upstream Serper requests. "
            f"Excess requests wait in handler threads (default: {DEFAULT_MAX_INFLIGHT})."
        ),
    )
    parser.add_argument(
        "--keys-file",
        default=os.environ.get("SERPER_RELAY_KEYS_FILE"),
        help="Relay-owned key file; enabling this makes the relay select real upstream keys.",
    )
    parser.add_argument(
        "--state-file",
        default=os.environ.get("SERPER_RELAY_POOL_STATE_FILE"),
        help="Relay-owned pool state file; defaults to the normal Serper pool state path.",
    )
    parser.add_argument(
        "--pool-default-credits",
        type=int,
        default=int(os.environ.get("SERPER_RELAY_POOL_DEFAULT_CREDITS") or 2500),
    )
    parser.add_argument(
        "--pool-min-remaining",
        type=int,
        default=int(os.environ.get("SERPER_RELAY_POOL_MIN_REMAINING") or 100),
    )
    parser.add_argument(
        "--pool-persist-every",
        type=int,
        default=int(
            os.environ.get("SERPER_RELAY_POOL_PERSIST_EVERY")
            or DEFAULT_POOL_PERSIST_EVERY
        ),
        help="Checkpoint the in-memory pool after this many uses (default: 10000).",
    )
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
    if (
        args.timeout_s <= 0
        or args.pool_persist_every <= 0
        or args.request_queue_size <= 0
        or args.max_inflight <= 0
    ):
        raise SystemExit(
            "--timeout-s, --pool-persist-every, --request-queue-size, and "
            "--max-inflight must be positive."
        )

    server = IPv6ThreadingHTTPServer(
        (args.bind, args.port),
        SerperRelayHandler,
        request_queue_size=args.request_queue_size,
        max_inflight=args.max_inflight,
    )
    server.relay_token = args.token
    server.upstream_base = args.upstream
    server.timeout_s = args.timeout_s
    server.key_pool = None
    if args.keys_file:
        keys = SerperApiKeyPool._load_keys_from_file(args.keys_file)
        server.key_pool = SerperApiKeyPool(
            keys=keys,
            state_path=args.state_file,
            default_credits=args.pool_default_credits,
            min_remaining=args.pool_min_remaining,
            persist_every=args.pool_persist_every,
        )
    print(
        f"Serper relay listening on [{args.bind}]:{args.port}; "
        f"upstream={args.upstream}; token_required={bool(args.token)}; "
        f"pool_managed={server.key_pool is not None}; "
        f"request_queue_size={args.request_queue_size}; "
        f"max_inflight={args.max_inflight}; "
        f"pool_keys={len(server.key_pool.keys) if server.key_pool is not None else 0}; "
        f"pool_persist_every={args.pool_persist_every}; "
        f"pool_state={server.key_pool.state_path if server.key_pool is not None else '-'}",
        flush=True,
    )
    _install_shutdown_handlers(server)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Serper relay.", flush=True)
    finally:
        if server.key_pool is not None:
            server.key_pool.flush()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
