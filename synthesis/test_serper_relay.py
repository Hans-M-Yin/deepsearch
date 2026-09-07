"""Unit tests for Relay-owned Serper key selection."""

from __future__ import annotations

import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest import mock

from debug import serper_relay
from synthesis.search_client import SerperApiKeyPool


class _Response:
    headers = {"Content-Type": "application/json"}

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def getcode(self) -> int:
        return 200

    def read(self) -> bytes:
        return b'{"organic":[]}'


class SerperRelayTest(unittest.TestCase):
    def test_managed_pool_rotates_after_unauthorized_key(self) -> None:
        with TemporaryDirectory() as directory:
            state_path = Path(directory) / "pool.json"
            pool = SerperApiKeyPool(
                keys=["unauthorized-key", "good-key"],
                state_path=state_path,
                default_credits=100,
                min_remaining=0,
            )
            server = serper_relay.ThreadingHTTPServer(
                ("127.0.0.1", 0), serper_relay.SerperRelayHandler
            )
            server.relay_token = ""
            server.upstream_base = "https://google.serper.dev"
            server.timeout_s = 10.0
            server.key_pool = pool
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                host, port = server.server_address[:2]
                upstream_keys: list[str] = []

                def fake_upstream(request: Request, *, timeout: float) -> _Response:
                    del timeout
                    upstream_keys.append(request.get_header("X-api-key") or "")
                    if upstream_keys[-1] == "unauthorized-key":
                        raise HTTPError(
                            request.full_url,
                            403,
                            "Forbidden",
                            {},
                            io.BytesIO(
                                b'{"message":"Unauthorized.","statusCode":403}'
                            ),
                        )
                    return _Response()

                request = Request(
                    f"http://{host}:{port}/search",
                    data=b'{"q":"auth rotation"}',
                    headers={
                        "Content-Type": "application/json",
                        "X-API-KEY": "relay-placeholder",
                    },
                    method="POST",
                )
                with mock.patch.object(serper_relay, "urlopen", side_effect=fake_upstream):
                    with urlopen(request, timeout=3) as response:
                        self.assertEqual(json.loads(response.read())["organic"], [])

                self.assertEqual(upstream_keys, ["unauthorized-key", "good-key"])
                state = json.loads(state_path.read_text(encoding="utf-8"))
                unauthorized = state["keys"][SerperApiKeyPool.key_id("unauthorized-key")]
                self.assertTrue(unauthorized["disabled"])
                self.assertEqual(
                    unauthorized["disabled_reason"],
                    "serper_relay_http_403_unauthorized",
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_managed_pool_ignores_client_key_and_rotates_after_credit_error(self) -> None:
        with TemporaryDirectory() as directory:
            state_path = Path(directory) / "pool.json"
            pool = SerperApiKeyPool(
                keys=["empty-key", "good-key"],
                state_path=state_path,
                default_credits=100,
                min_remaining=0,
            )
            server = serper_relay.ThreadingHTTPServer(
                ("127.0.0.1", 0), serper_relay.SerperRelayHandler
            )
            server.relay_token = ""
            server.upstream_base = "https://google.serper.dev"
            server.timeout_s = 10.0
            server.key_pool = pool
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                host, port = server.server_address[:2]
                upstream_keys: list[str] = []

                def fake_upstream(request: Request, *, timeout: float) -> _Response:
                    del timeout
                    upstream_keys.append(request.get_header("X-api-key") or "")
                    if upstream_keys[-1] == "empty-key":
                        raise HTTPError(
                            request.full_url,
                            400,
                            "Bad Request",
                            {},
                            io.BytesIO(b'{"message":"Not enough credits"}'),
                        )
                    return _Response()

                def request_once() -> Request:
                    return Request(
                        f"http://{host}:{port}/search",
                        data=json.dumps({"q": "relay test"}).encode("utf-8"),
                        headers={
                            "Content-Type": "application/json",
                            "X-API-KEY": "relay-placeholder",
                        },
                        method="POST",
                    )

                with mock.patch.object(serper_relay, "urlopen", side_effect=fake_upstream):
                    with urlopen(request_once(), timeout=3) as response:
                        self.assertEqual(json.loads(response.read())["organic"], [])

                self.assertEqual(upstream_keys, ["empty-key", "good-key"])
                state = json.loads(state_path.read_text(encoding="utf-8"))
                exhausted = state["keys"][SerperApiKeyPool.key_id("empty-key")]
                self.assertTrue(exhausted["disabled"])
                self.assertEqual(exhausted["remaining_credits"], 0)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_credit_exhaustion_stops_after_all_candidate_keys(self) -> None:
        with TemporaryDirectory() as directory:
            state_path = Path(directory) / "pool.json"
            pool = SerperApiKeyPool(
                keys=["empty-one", "empty-two"],
                state_path=state_path,
                default_credits=100,
                min_remaining=0,
            )
            server = serper_relay.ThreadingHTTPServer(
                ("127.0.0.1", 0), serper_relay.SerperRelayHandler
            )
            server.relay_token = ""
            server.upstream_base = "https://google.serper.dev"
            server.timeout_s = 10.0
            server.key_pool = pool
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                host, port = server.server_address[:2]
                upstream_keys: list[str] = []

                def fake_upstream(request: Request, *, timeout: float) -> _Response:
                    del timeout
                    upstream_keys.append(request.get_header("X-api-key") or "")
                    raise HTTPError(
                        request.full_url,
                        400,
                        "Bad Request",
                        {},
                        io.BytesIO(b'{"message":"Not enough credits"}'),
                    )

                request = Request(
                    f"http://{host}:{port}/search",
                    data=b'{"q":"all exhausted"}',
                    headers={
                        "Content-Type": "application/json",
                        "X-API-KEY": "relay-placeholder",
                    },
                    method="POST",
                )
                with mock.patch.object(serper_relay, "urlopen", side_effect=fake_upstream):
                    with self.assertRaises(HTTPError) as error:
                        with urlopen(request, timeout=3) as response:
                            response.read()
                self.assertEqual(error.exception.code, 400)
                self.assertEqual(upstream_keys, ["empty-one", "empty-two"])
                state = json.loads(state_path.read_text(encoding="utf-8"))
                self.assertTrue(
                    state["keys"][SerperApiKeyPool.key_id("empty-one")]["disabled"]
                )
                self.assertTrue(
                    state["keys"][SerperApiKeyPool.key_id("empty-two")]["disabled"]
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
