"""Unit tests for the standalone HTTP image-download relay."""

from __future__ import annotations

import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen as client_urlopen
from unittest import mock

from debug import http_relay
from synthesis.sft import tools


class _FakeResponse:
    def __init__(self, body: bytes, *, status_code: int, content_type: str = "image/jpeg") -> None:
        self.body = body
        self.status_code = status_code
        self.headers = {
            "Content-Type": content_type,
            "Location": "",
        }

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def getcode(self) -> int:
        return self.status_code

    def read(self, limit: int = -1) -> bytes:
        if limit == 0:
            return b""
        return self.body if limit < 0 else self.body[:limit]


class HttpRelayTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server = http_relay.ThreadingHTTPServer(
            ("127.0.0.1", 0), http_relay.HttpRelayHandler
        )
        self.server.timeout_s = 10.0
        self.server.queue_timeout_s = 1.0
        self.server.max_inflight = 2
        self.server.upstream_attempts = 2
        self.server.retry_delay_s = 0.0
        self.server.max_redirects = 3
        self.server.upstream_slots = threading.BoundedSemaphore(2)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _url(self, path: str) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}{path}"

    def _request(self, target_url: str) -> Request:
        from urllib.parse import quote

        return Request(
            self._url(f"/fetch?url={quote(target_url, safe='')}"),
            headers={
                "Referer": "https://example.com/page",
                "User-Agent": "relay-test",
            },
        )

    def test_get_fetch_decodes_target_and_preserves_binary_response(self) -> None:
        calls: list[tuple[str, str, dict[str, str]]] = []
        target_url = "https://example.com/image.jpg?x=1&signed=a%2Bb"

        def fake_urlopen(request: Request, *, timeout: float) -> _FakeResponse:
            calls.append((request.method, request.full_url, dict(request.header_items())))
            del timeout
            return _FakeResponse(b"\xff\xd8relay-image", status_code=200)

        with mock.patch.object(http_relay, "urlopen", side_effect=fake_urlopen), mock.patch.object(
            http_relay, "_resolve_public_addresses"
        ):
            with client_urlopen(self._request(target_url), timeout=3) as response:
                body = response.read()

        self.assertEqual(body, b"\xff\xd8relay-image")
        self.assertEqual(response.headers["Content-Type"], "image/jpeg")
        self.assertEqual(response.headers["X-Relay-Upstream-Status"], "200")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0:2], ("GET", target_url))
        self.assertEqual(calls[0][2]["Referer"], "https://example.com/page")
        self.assertEqual(calls[0][2]["User-agent"], "relay-test")

    def test_base_prefix_path_form_decodes_target(self) -> None:
        from urllib.parse import quote

        target_url = "https://example.com/image.jpg?x=1&y=2"
        calls: list[str] = []

        def fake_urlopen(request: Request, *, timeout: float) -> _FakeResponse:
            del timeout
            calls.append(request.full_url)
            return _FakeResponse(b"path-image", status_code=200)

        request = Request(
            self._url(f"/{quote(target_url, safe='')}"),
            headers={},
        )
        with mock.patch.object(http_relay, "urlopen", side_effect=fake_urlopen), mock.patch.object(
            http_relay, "_resolve_public_addresses"
        ):
            with client_urlopen(request, timeout=3) as response:
                self.assertEqual(response.read(), b"path-image")
        self.assertEqual(calls, [target_url])

    def test_transient_upstream_status_is_retried(self) -> None:
        calls: list[str] = []

        def fake_urlopen(request: Request, *, timeout: float) -> _FakeResponse:
            del timeout
            calls.append(request.full_url)
            if len(calls) == 1:
                return _FakeResponse(b"temporary", status_code=502, content_type="text/plain")
            return _FakeResponse(b"ok", status_code=200)

        with mock.patch.object(http_relay, "urlopen", side_effect=fake_urlopen), mock.patch.object(
            http_relay, "_resolve_public_addresses"
        ), mock.patch.object(http_relay.time, "sleep") as sleep:
            with client_urlopen(self._request("https://example.com/retry.jpg"), timeout=3) as response:
                self.assertEqual(response.read(), b"ok")

        self.assertEqual(len(calls), 2)
        sleep.assert_called_once_with(0.0)

    def test_private_target_is_rejected_before_upstream_request(self) -> None:
        request = self._request("http://127.0.0.1/private.jpg")
        with mock.patch.object(http_relay, "urlopen") as upstream:
            with self.assertRaises(HTTPError) as context:
                client_urlopen(request, timeout=3)
        self.assertEqual(context.exception.code, 400)
        self.assertIn("blocked", json.loads(context.exception.read())["detail"])
        upstream.assert_not_called()

    def test_worker_uses_empty_prefix_as_direct_url_and_prefixes_other_images(self) -> None:
        target_url = "https://example.com/a.jpg?x=1&y=2"
        with mock.patch.dict("os.environ", {"HTTP_RELAY_URL": ""}, clear=False):
            self.assertEqual(tools._http_relay_request_url(target_url), target_url)
        with mock.patch.dict(
            "os.environ",
            {"HTTP_RELAY_URL": "http://relay:18083/"},
            clear=False,
        ):
            self.assertEqual(
                tools._http_relay_request_url(target_url),
                "http://relay:18083/https%3A%2F%2Fexample.com%2Fa.jpg%3Fx%3D1%26y%3D2",
            )
        with mock.patch.dict(
            "os.environ",
            {"HTTP_RELAY_URL": "http://relay:18083/fetch?url="},
            clear=False,
        ):
            self.assertEqual(
                tools._http_relay_request_url(target_url),
                "http://relay:18083/fetch?url=https%3A%2F%2Fexample.com%2Fa.jpg%3Fx%3D1%26y%3D2",
            )


if __name__ == "__main__":
    unittest.main()
