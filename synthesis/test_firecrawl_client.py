"""Unit tests for the standalone Firecrawl backend (no network or SDK needed)."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from synthesis.firecrawl_client import FirecrawlClient


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


if __name__ == "__main__":
    unittest.main()
