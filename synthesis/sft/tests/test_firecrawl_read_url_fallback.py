"""Tests for Firecrawl fallback after Enhanced Reader block detection."""

from __future__ import annotations

from io import BytesIO
import json
import unittest
from urllib.error import HTTPError
from unittest import mock

from synthesis.sft import tools


class FirecrawlReadUrlFallbackTests(unittest.TestCase):
    def test_reader_retry_diagnostic_preserves_nested_urllib_error_body(self) -> None:
        error = HTTPError(
            "http://127.0.0.1:8004/https://example.com",
            502,
            "Bad Gateway",
            {"X-Reader-Dispatcher-Backend": "http://127.0.0.1:8172"},
            BytesIO(
                json.dumps(
                    {
                        "status": 502,
                        "message": "Enhanced Reader error for https://example.com",
                        "upstream_error": {
                            "status_code": 422,
                            "body": {
                                "message": (
                                    "Failed to goto https://example.com: "
                                    "TimeoutError: Navigation timeout of 30000 ms exceeded"
                                )
                            },
                        },
                        # The production body includes timing; retry logs must
                        # extract the cause rather than dump that payload.
                        "debug_timing": {"total_s": 31.2},
                    }
                ).encode("utf-8")
            ),
        )

        details = tools._http_retry_debug_details(error)

        self.assertEqual(details["http_status"], 502)
        self.assertEqual(details["reader_upstream_status"], 422)
        self.assertEqual(details["reader_backend"], "http://127.0.0.1:8172")
        self.assertIn("Navigation timeout", str(details["reader_error"]))
        self.assertNotIn("http_response_body", details)

    def test_firecrawl_403_application_error_is_not_retried(self) -> None:
        error = RuntimeError(
            'Firecrawl response error: site is unsupported; '
            'full_response: {"statusCode": 403, "error": "unsupported site"}'
        )
        self.assertFalse(tools._is_retryable_firecrawl_error(error))

    def test_blocked_enhanced_reader_summary_uses_firecrawl_and_resummarizes(self) -> None:
        enhanced_document = {
            "url": "https://example.com/page",
            "title": "Enhanced title",
            "content": "The page body content before semantic blocking.",
        }
        firecrawl_document = {
            "url": "https://example.com/page",
            "title": "Firecrawl title",
            "content": "The actual semantic page content.",
            "metadata": {"status_code": 200, "credits_used": 1},
        }
        with (
            mock.patch.object(tools, "_probe_content_type", return_value="text/html"),
            mock.patch.object(tools, "_read_document", return_value=enhanced_document),
            mock.patch.object(tools, "_read_via_firecrawl", return_value=firecrawl_document) as firecrawl_read,
            mock.patch.object(tools, "summarize_with_qwen", side_effect=["BLOCKED", "usable evidence"] ) as summarize,
        ):
            result = tools.read_url("https://example.com/page", goal="Find the claim.")

        self.assertTrue(result["ok"])
        self.assertEqual(result["content"], "usable evidence")
        self.assertEqual(result["title"], "Firecrawl title")
        self.assertEqual(result["resolved_via"], "firecrawl_fallback")
        self.assertEqual(result["firecrawl_metadata"]["credits_used"], 1)
        firecrawl_read.assert_called_once_with("https://example.com/page")
        self.assertEqual(summarize.call_count, 2)

    def test_semantic_enhanced_reader_summary_does_not_call_firecrawl(self) -> None:
        with (
            mock.patch.object(tools, "_probe_content_type", return_value="text/html"),
            mock.patch.object(
                tools,
                "_read_document",
                return_value={"url": "https://example.com/page", "title": "Page", "content": "Actual content"},
            ),
            mock.patch.object(tools, "summarize_with_qwen", return_value="usable evidence"),
            mock.patch.object(tools, "_read_via_firecrawl") as firecrawl_read,
        ):
            result = tools.read_url("https://example.com/page", goal="Find the claim.")

        self.assertTrue(result["ok"])
        self.assertEqual(result["resolved_via"], "enhanced_reader")
        firecrawl_read.assert_not_called()

    def test_empty_enhanced_reader_content_uses_firecrawl(self) -> None:
        firecrawl_document = {
            "url": "https://example.com/page",
            "title": "Firecrawl title",
            "content": "Actual content recovered by Firecrawl.",
            "metadata": {"status_code": 200, "credits_used": 1},
        }
        with (
            mock.patch.object(tools, "_probe_content_type", return_value="text/html"),
            mock.patch.object(
                tools,
                "_read_document",
                return_value={"url": "https://example.com/page", "title": "Page", "content": ""},
            ),
            mock.patch.object(tools, "_read_via_firecrawl", return_value=firecrawl_document) as firecrawl_read,
            mock.patch.object(tools, "summarize_with_qwen", return_value="usable evidence") as summarize,
        ):
            result = tools.read_url("https://example.com/page", goal="Find the claim.")

        self.assertTrue(result["ok"])
        self.assertEqual(result["resolved_via"], "firecrawl_fallback")
        firecrawl_read.assert_called_once_with("https://example.com/page")
        summarize.assert_called_once()

    def test_forbidden_interstitial_uses_firecrawl(self) -> None:
        firecrawl_document = {
            "url": "https://example.com/page",
            "title": "Firecrawl title",
            "content": "Actual content recovered by Firecrawl.",
            "metadata": {"status_code": 200, "credits_used": 1},
        }
        with (
            mock.patch.object(tools, "_probe_content_type", return_value="text/html"),
            mock.patch.object(
                tools,
                "_read_document",
                return_value={
                    "url": "https://example.com/page",
                    "title": "Page",
                    "content": "403 Forbidden",
                },
            ),
            mock.patch.object(tools, "_read_via_firecrawl", return_value=firecrawl_document) as firecrawl_read,
            mock.patch.object(tools, "summarize_with_qwen", return_value="usable evidence"),
        ):
            result = tools.read_url("https://example.com/page", goal="Find the claim.")

        self.assertTrue(result["ok"])
        self.assertEqual(result["resolved_via"], "firecrawl_fallback")
        firecrawl_read.assert_called_once_with("https://example.com/page")

    def test_raw_document_fetcher_reuses_page_but_resummarizes_each_goal(self) -> None:
        document = {
            "url": "https://example.com/page",
            "title": "Page",
            "content": "The complete page content.",
        }
        raw_cache: dict[tuple[str, str], dict] = {}

        def fetch_raw(backend: str, url: str, compute):
            key = (backend, url)
            if key not in raw_cache:
                raw_cache[key] = compute()
            return raw_cache[key]

        with (
            mock.patch.object(tools, "_probe_content_type", return_value="text/html"),
            mock.patch.object(tools, "_read_document", return_value=document) as reader,
            mock.patch.object(
                tools,
                "summarize_with_qwen",
                side_effect=["evidence for goal one", "evidence for goal two"],
            ) as summarize,
        ):
            first = tools.read_url(
                "https://example.com/page",
                goal="goal one",
                raw_document_fetcher=fetch_raw,
            )
            second = tools.read_url(
                "https://example.com/page",
                goal="goal two",
                raw_document_fetcher=fetch_raw,
            )

        self.assertEqual(first["content"], "evidence for goal one")
        self.assertEqual(second["content"], "evidence for goal two")
        reader.assert_called_once()
        self.assertEqual(summarize.call_count, 2)

    def test_enhanced_reader_error_uses_firecrawl_fallback(self) -> None:
        firecrawl_document = {
            "url": "https://example.com/page",
            "title": "Firecrawl title",
            "content": "The actual semantic page content.",
            "metadata": {"status_code": 200, "credits_used": 1},
        }
        with (
            mock.patch.object(tools, "_probe_content_type", return_value="text/html"),
            mock.patch.object(tools, "_read_document", side_effect=RuntimeError("HTTP Error 429: Too Many Requests")),
            mock.patch.object(tools, "_read_via_firecrawl", return_value=firecrawl_document) as firecrawl_read,
            mock.patch.object(tools, "summarize_with_qwen", return_value="usable evidence") as summarize,
        ):
            result = tools.read_url("https://example.com/page", goal="Find the claim.")

        self.assertTrue(result["ok"])
        self.assertEqual(result["content"], "usable evidence")
        self.assertEqual(result["resolved_via"], "firecrawl_fallback")
        firecrawl_read.assert_called_once_with("https://example.com/page")
        summarize.assert_called_once()

    def test_firecrawl_application_error_retries_once(self) -> None:
        firecrawl_document = {
            "url": "https://example.com/page",
            "title": "Firecrawl title",
            "content": "The actual semantic page content.",
            "metadata": {"status_code": 200, "credits_used": 1},
        }
        tunnel_error = (
            "The URL failed to load in the browser with error code "
            '"ERR_TUNNEL_CONNECTION_FAILED". Firecrawl encountered an internal proxy error.'
        )
        with (
            mock.patch.object(tools, "_read_document", side_effect=RuntimeError("HTTP Error 502")),
            mock.patch.object(
                tools,
                "_read_via_firecrawl",
                side_effect=[RuntimeError(tunnel_error), firecrawl_document],
            ) as firecrawl_read,
            mock.patch.object(tools, "summarize_with_qwen", return_value="usable evidence"),
            mock.patch.object(tools.time, "sleep"),
        ):
            result = tools.read_url("https://example.com/page", goal="Find the claim.")

        self.assertTrue(result["ok"])
        self.assertEqual(result["resolved_via"], "firecrawl_fallback")
        self.assertEqual(firecrawl_read.call_count, 2)


if __name__ == "__main__":
    unittest.main()
