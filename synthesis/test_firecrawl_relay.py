"""Unit tests for the stdlib Firecrawl Relay endpoints."""

from __future__ import annotations

import io
import json
import threading
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen as client_urlopen
from unittest import mock

from debug import firecrawl_relay
from synthesis.firecrawl_client import FirecrawlApiKeyPool


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

    def test_managed_pool_ignores_client_key_and_rotates_after_credit_error(self) -> None:
        with TemporaryDirectory() as directory:
            state_path = Path(directory) / "pool.json"
            self.server.key_pool = FirecrawlApiKeyPool(
                keys=["empty-key", "good-key"],
                state_path=state_path,
            )
            self.server.browser_session_keys = {}
            self.server.browser_session_lock = threading.RLock()
            upstream_keys: list[str] = []

            def fake_upstream(request: Request, *, timeout: float) -> _FakeUpstreamResponse:
                del timeout
                api_key = request.get_header("X-api-key") or ""
                upstream_keys.append(api_key)
                if api_key == "empty-key":
                    raise HTTPError(
                        request.full_url,
                        402,
                        "Payment Required",
                        {},
                        io.BytesIO(b'{"error":"Insufficient credits"}'),
                    )
                return _FakeUpstreamResponse(
                    {
                        "success": True,
                        "data": {
                            "markdown": "page",
                            "metadata": {"creditsUsed": 1, "statusCode": 200},
                        },
                    }
                )

            def request_once() -> object:
                return Request(
                    self._url("/v2/scrape"),
                    data=b'{"url":"https://example.com"}',
                    headers={
                        "Content-Type": "application/json",
                        "X-API-KEY": "relay-placeholder",
                    },
                    method="POST",
                )

            with mock.patch.object(firecrawl_relay, "urlopen", side_effect=fake_upstream):
                with client_urlopen(request_once(), timeout=3) as response:
                    self.assertEqual(json.loads(response.read())["success"], True)

            self.assertEqual(upstream_keys, ["empty-key", "good-key"])
            state = json.loads(state_path.read_text(encoding="utf-8"))
            exhausted = state["keys"][FirecrawlApiKeyPool.key_id("empty-key")]
            self.assertTrue(exhausted["disabled"])
            self.assertEqual(exhausted["disabled_reason"], "firecrawl_credits_exhausted")
            self.assertEqual(
                state["keys"][FirecrawlApiKeyPool.key_id("good-key")]["remaining_credits"],
                9999,
            )

    def test_account_banned_error_disables_key_even_without_http_401(self) -> None:
        with TemporaryDirectory() as directory:
            state_path = Path(directory) / "pool.json"
            self.server.key_pool = FirecrawlApiKeyPool(
                keys=["banned-key"],
                state_path=state_path,
            )
            self.server.browser_session_keys = {}
            self.server.browser_session_lock = threading.RLock()

            def fake_upstream(request: Request, *, timeout: float) -> _FakeUpstreamResponse:
                del request, timeout
                return _FakeUpstreamResponse(
                    {
                        "error": "Unauthorized: This account has been banned. "
                        "Contact support@firecrawl.com if you believe this is a mistake."
                    }
                )

            request = Request(
                self._url("/v2/scrape"),
                data=b'{"url":"https://example.com"}',
                headers={
                    "Content-Type": "application/json",
                    "X-API-KEY": "relay-placeholder",
                },
                method="POST",
            )
            with mock.patch.object(firecrawl_relay, "urlopen", side_effect=fake_upstream):
                with client_urlopen(request, timeout=3) as response:
                    self.assertEqual(json.loads(response.read())["error"].split(":", 1)[0], "Unauthorized")

            state = json.loads(state_path.read_text(encoding="utf-8"))
            banned = state["keys"][FirecrawlApiKeyPool.key_id("banned-key")]
            self.assertTrue(banned["disabled"])
            self.assertEqual(banned["disabled_reason"], "firecrawl_api_auth_failed")
            self.assertEqual(self.server.key_pool.status()["available_key_count"], 0)

    def test_target_page_401_does_not_disable_key(self) -> None:
        with TemporaryDirectory() as directory:
            state_path = Path(directory) / "pool.json"
            self.server.key_pool = FirecrawlApiKeyPool(
                keys=["page-key"],
                state_path=state_path,
            )
            self.server.browser_session_keys = {}
            self.server.browser_session_lock = threading.RLock()

            def fake_upstream(request: Request, *, timeout: float) -> _FakeUpstreamResponse:
                del request, timeout
                return _FakeUpstreamResponse(
                    {
                        "success": True,
                        "data": {
                            "markdown": "login page",
                            "metadata": {"statusCode": 401, "creditsUsed": 1},
                        },
                    }
                )

            request = Request(
                self._url("/v2/scrape"),
                data=b'{"url":"https://example.com/login"}',
                headers={
                    "Content-Type": "application/json",
                    "X-API-KEY": "relay-placeholder",
                },
                method="POST",
            )
            with mock.patch.object(firecrawl_relay, "urlopen", side_effect=fake_upstream):
                with client_urlopen(request, timeout=3) as response:
                    self.assertTrue(json.loads(response.read())["success"])

            state = json.loads(state_path.read_text(encoding="utf-8"))
            page_key = state["keys"][FirecrawlApiKeyPool.key_id("page-key")]
            self.assertFalse(page_key["disabled"])
            self.assertEqual(page_key["remaining_credits"], 9999)

    def test_credit_exhaustion_stops_after_all_candidate_keys(self) -> None:
        with TemporaryDirectory() as directory:
            state_path = Path(directory) / "pool.json"
            self.server.key_pool = FirecrawlApiKeyPool(
                keys=["empty-one", "empty-two"],
                state_path=state_path,
            )
            self.server.browser_session_keys = {}
            self.server.browser_session_lock = threading.RLock()
            upstream_keys: list[str] = []

            def fake_upstream(request: Request, *, timeout: float) -> _FakeUpstreamResponse:
                del timeout
                upstream_keys.append(request.get_header("X-api-key") or "")
                raise HTTPError(
                    request.full_url,
                    402,
                    "Payment Required",
                    {},
                    io.BytesIO(b'{"error":"Insufficient credits"}'),
                )

            request = Request(
                self._url("/v2/scrape"),
                data=b'{"url":"https://example.com"}',
                headers={
                    "Content-Type": "application/json",
                    "X-API-KEY": "relay-placeholder",
                },
                method="POST",
            )
            with mock.patch.object(firecrawl_relay, "urlopen", side_effect=fake_upstream):
                with self.assertRaises(HTTPError) as error:
                    with client_urlopen(request, timeout=3) as response:
                        response.read()
            self.assertEqual(error.exception.code, 402)
            self.assertEqual(upstream_keys, ["empty-one", "empty-two"])
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertTrue(
                state["keys"][FirecrawlApiKeyPool.key_id("empty-one")]["disabled"]
            )
            self.assertTrue(
                state["keys"][FirecrawlApiKeyPool.key_id("empty-two")]["disabled"]
            )

    def test_browser_create_credit_exhaustion_rotates_key(self) -> None:
        with TemporaryDirectory() as directory:
            state_path = Path(directory) / "pool.json"
            self.server.key_pool = FirecrawlApiKeyPool(
                keys=["empty-key", "browser-key"],
                state_path=state_path,
            )
            self.server.browser_session_keys = {}
            self.server.browser_session_lock = threading.RLock()
            upstream_keys: list[str] = []

            def fake_upstream(request: Request, *, timeout: float) -> _FakeUpstreamResponse:
                del timeout
                upstream_keys.append(request.get_header("X-api-key") or "")
                if upstream_keys[-1] == "empty-key":
                    raise HTTPError(
                        request.full_url,
                        402,
                        "Payment Required",
                        {},
                        io.BytesIO(b'{"error":"Insufficient credits"}'),
                    )
                return _FakeUpstreamResponse({"success": True, "id": "browser-new"})

            request = Request(
                self._url("/v2/browser"),
                data=b'{"ttl":300}',
                headers={
                    "Content-Type": "application/json",
                    "X-API-KEY": "relay-placeholder",
                },
                method="POST",
            )
            with mock.patch.object(firecrawl_relay, "urlopen", side_effect=fake_upstream):
                with client_urlopen(request, timeout=3) as response:
                    self.assertEqual(json.loads(response.read())["id"], "browser-new")

            self.assertEqual(upstream_keys, ["empty-key", "browser-key"])
            self.assertEqual(
                self.server.browser_session_keys["browser-new"],
                FirecrawlApiKeyPool.key_id("browser-key"),
            )

    def _url(self, path: str) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}{path}"

    def test_browser_image_diagnostics_extract_target_request_timeout(self) -> None:
        body = json.dumps(
            {
                "success": True,
                "result": json.dumps(
                    {
                        "status": None,
                        "error": "target_request_timeout",
                        "target_error_type": "target_request_timeout",
                        "target_error_message": "Timeout 120000ms exceeded",
                        "target_phase": "request",
                        "byte_count": 0,
                    }
                ),
            }
        ).encode("utf-8")

        self.assertEqual(
            firecrawl_relay._browser_image_diagnostics(body),
            {
                "target_error_type": "target_request_timeout",
                "target_error": "Timeout 120000ms exceeded",
                "target_phase": "request",
                "target_byte_count": 0,
            },
        )

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
