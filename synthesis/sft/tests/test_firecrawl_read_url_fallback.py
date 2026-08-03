"""Tests for Firecrawl fallback after Enhanced Reader block detection."""

from __future__ import annotations

import unittest
from unittest import mock

from synthesis.sft import tools


class FirecrawlReadUrlFallbackTests(unittest.TestCase):
    def test_blocked_enhanced_reader_summary_uses_firecrawl_and_resummarizes(self) -> None:
        enhanced_document = {
            "url": "https://example.com/page",
            "title": "Enhanced title",
            "content": "verification required",
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


if __name__ == "__main__":
    unittest.main()
