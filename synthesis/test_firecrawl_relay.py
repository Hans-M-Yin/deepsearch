"""Unit tests for the stdlib Firecrawl Relay endpoints."""

from __future__ import annotations

import io
import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen as client_urlopen
from unittest import mock

from debug import firecrawl_relay


class _FakeUpstreamResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.headers = {"Content-Type": "application/json"}

    def __enter__(self) -> "_FakeUpstreamResponse":
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def getcode(self) -> int:
        return 200

    def read(self, limit: int = -1) -> bytes:
        del limit
        return json.dumps(self.payload).encode("utf-8")


class FirecrawlRelayTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server = firecrawl_relay.ThreadingHTTPServer(
            ("127.0.0.1", 0), firecrawl_relay.FirecrawlRelayHandler
        )
        self.server.upstream_base = "https://api.firecrawl.dev"
        self.server.queue_timeout_s = 1.0
        self.server.text_timeout_s = 10.0
        self.server.text_upstream_attempts = 2
        self.server.text_retry_delay_s = 0.0
        self.server.browser_timeout_s = 10.0
        self.server.browser_image_attempts = 3
        self.server.browser_execute_attempts = 2
        self.server.browser_create_attempts = 1
        self.server.browser_delete_attempts = 1
        self.server.browser_retry_delay_s = 0.0
        import threading as _threading

        self.server.upstream_slots = _threading.BoundedSemaphore(2)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _url(self, path: str) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}{path}"

    def test_browser_execute_image_retries_transient_upstream_error(self) -> None:
        calls: list[tuple[str, str, float]] = []

        def fake_upstream(request: Request, *, timeout: float) -> _FakeUpstreamResponse:
            calls.append((request.method, request.full_url, timeout))
            if len(calls) == 1:
                raise HTTPError(
                    request.full_url,
                    502,
                    "temporary upstream failure",
                    {},
                    io.BytesIO(b'{"error":"temporary"}'),
                )
            return _FakeUpstreamResponse({"success": True, "result": "image-result"})

        payload = {"code": "return 'image';", "language": "node", "timeout": 10}
        request = Request(
            self._url("/v2/browser/browser-1/execute"),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-API-KEY": "relay-test-key",
                "X-Firecrawl-Relay-Request-Type": "browser_image",
            },
            method="POST",
        )
        with mock.patch.object(firecrawl_relay, "urlopen", side_effect=fake_upstream), mock.patch.object(
            firecrawl_relay.time, "sleep"
        ) as sleep:
            with client_urlopen(request, timeout=3) as response:
                result = json.loads(response.read().decode("utf-8"))

        self.assertEqual(result["success"], True)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0], "POST")
        self.assertEqual(calls[0][1], "https://api.firecrawl.dev/v2/browser/browser-1/execute")
        self.assertEqual(calls[0][2], 10.0)
        sleep.assert_called_once_with(0.0)

    def test_browser_image_retries_transient_target_status_inside_result(self) -> None:
        calls: list[str] = []

        def fake_upstream(request: Request, *, timeout: float) -> _FakeUpstreamResponse:
            del timeout
            calls.append(request.full_url)
            if len(calls) == 1:
                return _FakeUpstreamResponse(
                    {
                        "success": True,
                        "result": json.dumps(
                            {
                                "status": 429,
                                "content_type": "text/html",
                                "error": "http_status_429",
                            }
                        ),
                    }
                )
            return _FakeUpstreamResponse(
                {
                    "success": True,
                    "result": json.dumps(
                        {
                            "status": 200,
                            "content_type": "image/jpeg",
                            "body_base64": "aW1hZ2U=",
                        }
                    ),
                }
            )

        request = Request(
            self._url("/v2/browser/browser-1/execute"),
            data=b'{"code":"download","language":"node"}',
            headers={
                "Content-Type": "application/json",
                "X-API-KEY": "relay-test-key",
                "X-Firecrawl-Relay-Request-Type": "browser_image",
            },
            method="POST",
        )
        with mock.patch.object(firecrawl_relay, "urlopen", side_effect=fake_upstream), mock.patch.object(
            firecrawl_relay.time, "sleep"
        ):
            with client_urlopen(request, timeout=3) as response:
                result = json.loads(response.read().decode("utf-8"))

        self.assertEqual(result["success"], True)
        self.assertEqual(len(calls), 2)

    def test_browser_create_and_delete_routes_are_supported(self) -> None:
        calls: list[tuple[str, str, bytes | None]] = []

        def fake_upstream(request: Request, *, timeout: float) -> _FakeUpstreamResponse:
            del timeout
            calls.append((request.method, request.full_url, request.data))
            if request.method == "POST":
                return _FakeUpstreamResponse({"success": True, "id": "browser-2"})
            return _FakeUpstreamResponse({"success": True, "creditsBilled": 1})

        create = Request(
            self._url("/v2/browser"),
            data=json.dumps({"ttl": 300, "activity_ttl": 120}).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-API-KEY": "relay-test-key"},
            method="POST",
        )
        delete = Request(
            self._url("/v2/browser/browser-2"),
            headers={"X-API-KEY": "relay-test-key"},
            method="DELETE",
        )
        with mock.patch.object(firecrawl_relay, "urlopen", side_effect=fake_upstream):
            with client_urlopen(create, timeout=3) as response:
                self.assertEqual(json.loads(response.read())["id"], "browser-2")
            with client_urlopen(delete, timeout=3) as response:
                self.assertEqual(json.loads(response.read())["creditsBilled"], 1)

        self.assertEqual(calls[0][0:2], ("POST", "https://api.firecrawl.dev/v2/browser"))
        self.assertEqual(calls[1][0:2], ("DELETE", "https://api.firecrawl.dev/v2/browser/browser-2"))
        forwarded_create = json.loads(calls[0][2].decode("utf-8"))
        self.assertEqual(forwarded_create, {"ttl": 300, "activityTtl": 120})
        self.assertIsNone(calls[1][2])


if __name__ == "__main__":
    unittest.main()
