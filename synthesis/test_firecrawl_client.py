"""Unit tests for the standalone Firecrawl backend (no network or SDK needed)."""

from __future__ import annotations

import base64
import json
import time
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from synthesis.firecrawl_client import (
    FirecrawlApiKeyPool,
    FirecrawlBrowserImageDownloader,
    FirecrawlBrowserNonImageError,
    FirecrawlBrowserSessionManager,
    FirecrawlClient,
)


class _FakeFirecrawl:
    calls: list[tuple[str, str, dict]] = []

    class UnauthorizedError(RuntimeError):
        status_code = 401

    def __init__(self, *, api_key: str) -> None:
        self.api_key = api_key

    def scrape(self, url: str, **kwargs: object) -> dict:
        self.calls.append((self.api_key, url, kwargs))
        if self.api_key == "bad-key":
            raise self.UnauthorizedError("Unauthorized")
        return {
            "success": True,
            "data": {"markdown": "page", "metadata": {"creditsUsed": 14, "statusCode": 200}},
        }


class _FakeBrowserFirecrawl:
    browser_calls: list[tuple[str, dict]] = []
    execute_calls: list[tuple[str, str, str, dict]] = []
    delete_calls: list[tuple[str, str]] = []
    response_payload: dict = {}
    next_session_id = 0

    def __init__(self, *, api_key: str) -> None:
        self.api_key = api_key

    @classmethod
    def reset(cls, *, payload: dict) -> None:
        cls.browser_calls = []
        cls.execute_calls = []
        cls.delete_calls = []
        cls.response_payload = payload
        cls.next_session_id = 0

    def browser(self, **kwargs: object) -> dict:
        type(self).next_session_id += 1
        session_id = f"browser-{type(self).next_session_id}"
        type(self).browser_calls.append((self.api_key, dict(kwargs)))
        return {"success": True, "id": session_id}

    def browser_execute(self, session_id: str, code: str, **kwargs: object) -> dict:
        type(self).execute_calls.append((self.api_key, session_id, code, dict(kwargs)))
        return {"success": True, "result": json.dumps(type(self).response_payload)}

    def delete_browser(self, session_id: str) -> dict:
        type(self).delete_calls.append((self.api_key, session_id))
        return {"success": True, "creditsBilled": 7}


class FirecrawlClientTest(unittest.TestCase):
    def test_round_robin_and_result_state(self) -> None:
        _FakeFirecrawl.calls = []
        with TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            client = FirecrawlClient(
                api_keys=["good-key", "bad-key"],
                pool_state_path=state_path,
                app_factory=_FakeFirecrawl,
            )

            with mock.patch("sys.stderr", new_callable=StringIO) as stderr:
                self.assertTrue(client.scrape("https://example.com/first", formats=["markdown"])["success"])
            self.assertIn("url=https://example.com/first", stderr.getvalue())
            self.assertIn("markdown_chars=4", stderr.getvalue())
            self.assertIn("credits_used=14", stderr.getvalue())
            with self.assertRaisesRegex(RuntimeError, "Unauthorized"):
                client.scrape("https://example.com/second")
            self.assertTrue(client.scrape("https://example.com/third")["success"])

            self.assertEqual([call[0] for call in _FakeFirecrawl.calls], ["good-key", "bad-key", "good-key"])
            self.assertEqual(_FakeFirecrawl.calls[0][2]["formats"], ["markdown"])
            state = json.loads(state_path.read_text(encoding="utf-8"))
            bad_record = state["keys"][client.key_pool.key_id("bad-key")]
            good_record = state["keys"][client.key_pool.key_id("good-key")]
            self.assertTrue(bad_record["disabled"])
            self.assertEqual(bad_record["state"], "disabled")
            self.assertEqual(good_record["initial_credits"], 10000)
            self.assertEqual(good_record["remaining_credits"], 9972)
            self.assertEqual(good_record["credits_consumed"], 28)
            self.assertEqual(good_record["last_credits_used"], 14)
            self.assertEqual(good_record["last_status_code"], 200)
            self.assertEqual(client.key_pool.status()["available_key_count"], 1)
            self.assertTrue(client.key_pool.lock_path.exists())

    def test_page_401_does_not_disable_api_key(self) -> None:
        class PageUnauthorizedFirecrawl:
            def __init__(self, *, api_key: str) -> None:
                del api_key

            def scrape(self, url: str, **kwargs: object) -> dict:
                del url, kwargs
                return {
                    "success": True,
                    "data": {
                        "markdown": "login page",
                        "metadata": {
                            "statusCode": 401,
                            "creditsUsed": 1,
                            "error": "Unauthorized",
                        },
                    },
                }

        with TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            client = FirecrawlClient(
                api_keys=["one-key"],
                pool_state_path=state_path,
                app_factory=PageUnauthorizedFirecrawl,
            )

            self.assertEqual(
                client.scrape("https://example.com/login"),
                {"error": "Firecrawl scrape returned statusCode 401: Unauthorized"},
            )
            record = json.loads(state_path.read_text(encoding="utf-8"))["keys"][
                client.key_pool.key_id("one-key")
            ]
            self.assertFalse(record["disabled"])
            self.assertEqual(record["remaining_credits"], 9999)
            self.assertEqual(client.key_pool.status()["available_key_count"], 1)

    def test_untyped_invalid_key_text_does_not_disable_api_key(self) -> None:
        class UntypedErrorFirecrawl:
            def __init__(self, *, api_key: str) -> None:
                del api_key

            def scrape(self, url: str, **kwargs: object) -> dict:
                del url, kwargs
                return {"error": "Invalid API key"}

        with TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            client = FirecrawlClient(
                api_keys=["one-key"],
                pool_state_path=state_path,
                app_factory=UntypedErrorFirecrawl,
            )

            self.assertEqual(client.scrape("https://example.com"), {"error": "Invalid API key"})
            record = json.loads(state_path.read_text(encoding="utf-8"))["keys"][
                client.key_pool.key_id("one-key")
            ]
            self.assertFalse(record["disabled"])
            self.assertEqual(client.key_pool.status()["available_key_count"], 1)

    def test_root_api_401_disables_api_key(self) -> None:
        class RootUnauthorizedFirecrawl:
            def __init__(self, *, api_key: str) -> None:
                del api_key

            def scrape(self, url: str, **kwargs: object) -> dict:
                del url, kwargs
                return {"statusCode": 401, "error": "Unauthorized"}

        with TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            client = FirecrawlClient(
                api_keys=["one-key"],
                pool_state_path=state_path,
                app_factory=RootUnauthorizedFirecrawl,
            )

            self.assertEqual(
                client.scrape("https://example.com"),
                {"error": "Unauthorized"},
            )
            record = json.loads(state_path.read_text(encoding="utf-8"))["keys"][
                client.key_pool.key_id("one-key")
            ]
            self.assertTrue(record["disabled"])
            self.assertEqual(record["disabled_reason"], "firecrawl_api_auth_failed")

    def test_recovers_legacy_page_401_disable(self) -> None:
        with TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            key_id = FirecrawlClient(api_keys=["one-key"], pool_state_path=state_path).key_pool.key_id(
                "one-key"
            )
            state_path.write_text(
                json.dumps(
                    {
                        "keys": {
                            key_id: {
                                "disabled": True,
                                "disabled_reason": "firecrawl_key_rejected_or_exhausted",
                                "last_status_code": 401,
                                "last_error": "Firecrawl scrape returned statusCode 401: Unauthorized",
                                "remaining_credits": 9000,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            client = FirecrawlClient(api_keys=["one-key"], pool_state_path=state_path)

            self.assertEqual(client.key_pool.status()["available_key_count"], 1)
            record = json.loads(state_path.read_text(encoding="utf-8"))["keys"][key_id]
            self.assertFalse(record["disabled"])
            self.assertEqual(record["state"], "active")

    def test_non_200_status_returns_an_error_text_and_charges_actual_credits(self) -> None:
        class Non200Firecrawl:
            def __init__(self, *, api_key: str) -> None:
                del api_key

            def scrape(self, url: str, **kwargs: object) -> dict:
                del url, kwargs
                return {
                    "success": True,
                    "data": {
                        "markdown": "partial page",
                        "metadata": {"statusCode": 403, "creditsUsed": 3, "error": "Access denied"},
                    },
                }

        with TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            client = FirecrawlClient(
                api_keys=["one-key"], pool_state_path=state_path, app_factory=Non200Firecrawl,
            )
            self.assertEqual(
                client.scrape("https://example.com/protected"),
                {"error": "Firecrawl scrape returned statusCode 403: Access denied"},
            )
            record = json.loads(state_path.read_text(encoding="utf-8"))["keys"][client.key_pool.key_id("one-key")]
            self.assertEqual(record["last_result"], "failure")
            self.assertEqual(record["last_status_code"], 403)
            self.assertEqual(record["remaining_credits"], 9997)

    def test_direct_snake_case_sdk_response_tracks_credit_and_status(self) -> None:
        class DirectResponseFirecrawl:
            def __init__(self, *, api_key: str) -> None:
                del api_key

            def scrape(self, url: str, **kwargs: object) -> dict:
                del url, kwargs
                return {"markdown": "page", "metadata": {"status_code": 200, "credits_used": 14}}

        with TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            client = FirecrawlClient(
                api_keys=["one-key"], pool_state_path=state_path, app_factory=DirectResponseFirecrawl,
            )
            response = client.scrape("https://example.com")
            self.assertEqual(response["metadata"]["credits_used"], 14)
            record = json.loads(state_path.read_text(encoding="utf-8"))["keys"][client.key_pool.key_id("one-key")]
            self.assertEqual(record["remaining_credits"], 9986)
            self.assertEqual(record["last_status_code"], 200)

    def test_browser_image_download_reuses_one_session_and_returns_raw_bytes(self) -> None:
        jpeg = b"\xff\xd8\xff" + b"image-data"
        _FakeBrowserFirecrawl.reset(
            payload={
                "status": 200,
                "resolved_url": "https://upload.wikimedia.org/resolved.jpg",
                "content_type": "image/jpeg",
                "body_base64": base64.b64encode(jpeg).decode("ascii"),
                "byte_count": len(jpeg),
                "error": None,
            }
        )
        with TemporaryDirectory() as directory:
            pool = FirecrawlApiKeyPool(
                keys=["first-key", "second-key"],
                state_path=Path(directory) / "pool.json",
            )
            manager = FirecrawlBrowserSessionManager(
                key_pool=pool,
                app_factory=_FakeBrowserFirecrawl,
                state_path=Path(directory) / "sessions.json",
                session_ttl_s=300,
                activity_ttl_s=120,
                max_sessions=2,
            )
            downloader = FirecrawlBrowserImageDownloader(
                session_manager=manager,
                retries=0,
            )

            first = downloader.download("https://upload.wikimedia.org/example.jpg")
            second = downloader.download("https://upload.wikimedia.org/other.jpg")

            self.assertEqual(first.payload, jpeg)
            self.assertEqual(first.content_type, "image/jpeg")
            self.assertEqual(second.payload, jpeg)
            self.assertEqual(len(_FakeBrowserFirecrawl.browser_calls), 1)
            self.assertEqual(len(_FakeBrowserFirecrawl.execute_calls), 2)
            self.assertEqual(_FakeBrowserFirecrawl.browser_calls[0][0], "first-key")
            self.assertEqual(
                _FakeBrowserFirecrawl.execute_calls[0][1],
                _FakeBrowserFirecrawl.execute_calls[1][1],
            )
            self.assertIn("page.request.get", _FakeBrowserFirecrawl.execute_calls[0][2])
            self.assertIn("body.toString(\"base64\")", _FakeBrowserFirecrawl.execute_calls[0][2])
            self.assertIn("await (async () => {", _FakeBrowserFirecrawl.execute_calls[0][2])
            self.assertIn("return JSON.stringify(output);", _FakeBrowserFirecrawl.execute_calls[0][2])
            self.assertNotIn("screenshot", _FakeBrowserFirecrawl.execute_calls[0][2].lower())
            state = json.loads((Path(directory) / "sessions.json").read_text(encoding="utf-8"))
            self.assertEqual(state["sessions"][0]["state"], "idle")

    def test_browser_image_download_rejects_non_image_response(self) -> None:
        _FakeBrowserFirecrawl.reset(
            payload={
                "status": 200,
                "resolved_url": "https://example.com/login",
                "content_type": "text/html",
                "body_base64": None,
                "byte_count": 0,
                "error": "non_image_content_type",
            }
        )
        with TemporaryDirectory() as directory:
            pool = FirecrawlApiKeyPool(keys=["one-key"], state_path=Path(directory) / "pool.json")
            manager = FirecrawlBrowserSessionManager(
                key_pool=pool,
                app_factory=_FakeBrowserFirecrawl,
                state_path=Path(directory) / "sessions.json",
            )
            downloader = FirecrawlBrowserImageDownloader(session_manager=manager, retries=0)
            with self.assertRaises(FirecrawlBrowserNonImageError):
                downloader.download("https://example.com/not-an-image")

    def test_browser_pool_rotates_keys_only_when_new_sessions_are_created(self) -> None:
        _FakeBrowserFirecrawl.reset(payload={})
        with TemporaryDirectory() as directory:
            pool = FirecrawlApiKeyPool(
                keys=["first-key", "second-key"],
                state_path=Path(directory) / "pool.json",
            )
            manager = FirecrawlBrowserSessionManager(
                key_pool=pool,
                app_factory=_FakeBrowserFirecrawl,
                state_path=Path(directory) / "sessions.json",
                max_sessions=2,
            )

            first = manager.acquire(acquire_timeout_s=1, lease_timeout_s=60)
            second = manager.acquire(acquire_timeout_s=1, lease_timeout_s=60)
            first.release()
            second.release()

            self.assertEqual(
                [key for key, _ in _FakeBrowserFirecrawl.browser_calls],
                ["first-key", "second-key"],
            )

    def test_browser_pool_reports_waiting_and_retired_session_credits(self) -> None:
        _FakeBrowserFirecrawl.reset(payload={})
        with TemporaryDirectory() as directory:
            pool = FirecrawlApiKeyPool(keys=["one-key"], state_path=Path(directory) / "pool.json")
            manager = FirecrawlBrowserSessionManager(
                key_pool=pool,
                app_factory=_FakeBrowserFirecrawl,
                state_path=Path(directory) / "sessions.json",
            )
            lease = manager.acquire(acquire_timeout_s=1, lease_timeout_s=60)
            manager._register_waiter("test-waiter", expires_at=time.time() + 60)

            queued_snapshot = manager.pool_snapshot()
            self.assertEqual(queued_snapshot["in_use"], 1)
            self.assertEqual(queued_snapshot["waiting"], 1)

            manager._remove_waiter("test-waiter")
            lease.invalidate("test close")
            retired_snapshot = manager.pool_snapshot()

            self.assertEqual(retired_snapshot["in_use"], 0)
            self.assertEqual(retired_snapshot["metrics"]["retired"], 1)
            self.assertEqual(retired_snapshot["metrics"]["credits_billed"], 7)
            self.assertEqual(retired_snapshot["key_pool"]["credits_consumed"], 7)


if __name__ == "__main__":
    unittest.main()
