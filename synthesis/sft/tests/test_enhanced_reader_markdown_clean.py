import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx

import utils.enhanced_reader as enhanced_reader
from utils.enhanced_reader import (
    UpstreamHTTPError,
    _stage_failure_payload,
    clean_raw_markdown,
    fetch_raw_markdown_via_cache_layer,
)


class EnhancedReaderMarkdownCleanTests(unittest.TestCase):
    def test_malformed_generic_image_does_not_leave_opener(self) -> None:
        cleaned, stats = clean_raw_markdown("before ![Image 10](")

        self.assertEqual(cleaned, "before ![Image 10]")
        self.assertEqual(stats["malformed_images_seen"], 1)
        self.assertEqual(stats["malformed_images_fixed"], 1)

    def test_truncated_image_chain_is_removed(self) -> None:
        source = "![Image 18: @davidjgraph]( 19: @antheas]( 20: @shijianjian]("

        cleaned, stats = clean_raw_markdown(source)

        self.assertEqual(cleaned, "![Image 18: @davidjgraph]")
        self.assertEqual(stats["malformed_images_seen"], 1)
        self.assertEqual(stats["malformed_images_fixed"], 1)

    def test_cleaning_stats_and_reference_section(self) -> None:
        source = (
            "正文中的 [Argentina](https://example.com/arg).\n\n"
            "## References\n\n"
            "1. [Source](https://example.com/source)\n\n"
            "## Next section\n\n"
            "继续正文。"
        )

        cleaned, stats = clean_raw_markdown(source)

        self.assertIn("正文中的 Argentina", cleaned)
        self.assertIn("## Next section", cleaned)
        self.assertNotIn("References", cleaned)
        self.assertEqual(stats["original_chars"], len(source))
        self.assertEqual(stats["cleaned_chars"], len(cleaned))
        self.assertEqual(stats["reference_sections_removed"], 1)

    def test_upstream_http_error_keeps_jina_browser_failure(self) -> None:
        response = httpx.Response(
            422,
            headers={"X-Reader-Dispatcher-Backend": "http://127.0.0.1:8172"},
            json={
                "code": 422,
                "name": "AssertionFailureError",
                "message": "Failed to goto http://example.test: net::ERR_BLOCKED_BY_CLIENT",
                "readableMessage": "Failed to goto http://example.test: net::ERR_BLOCKED_BY_CLIENT",
            },
            request=httpx.Request("GET", "http://127.0.0.1:8002/http://example.test"),
        )

        error = UpstreamHTTPError(response)
        status, message, payload, headers = _stage_failure_payload(
            endpoint_name="Raw markdown reader",
            url="http://example.test",
            stage="fetch_markdown",
            exc=error,
            timing={"total_s": 1.25},
        )

        self.assertEqual(status, 422)
        self.assertEqual(payload["code"], 422)
        self.assertIn("ERR_BLOCKED_BY_CLIENT", message)
        self.assertEqual(payload["upstream_error"]["status_code"], 422)
        self.assertEqual(
            payload["upstream_error"]["headers"]["x-reader-dispatcher-backend"],
            "http://127.0.0.1:8172",
        )
        self.assertEqual(headers["X-Reader-Dispatcher-Backend"], "http://127.0.0.1:8172")

    def test_cache_layer_internal_json_is_unwrapped_on_success(self) -> None:
        async def run() -> str:
            def handler(request: httpx.Request) -> httpx.Response:
                self.assertEqual(request.headers["accept"], "application/json")
                return httpx.Response(
                    200,
                    json={"data": {"content": "raw markdown"}, "code": 200, "status": 200},
                    request=request,
                )

            transport = httpx.MockTransport(handler)
            async with httpx.AsyncClient(transport=transport) as client:
                return await fetch_raw_markdown_via_cache_layer(client, "https://example.test")

        self.assertEqual(asyncio.run(run()), "raw markdown")

    def test_raw_and_enhanced_endpoints_preserve_nested_http_error(self) -> None:
        async def run() -> tuple[dict, dict]:
            def jina_handler(request: httpx.Request) -> httpx.Response:
                return httpx.Response(
                    422,
                    headers={"X-Reader-Dispatcher-Backend": "http://127.0.0.1:8172"},
                    json={
                        "code": 422,
                        "status": 422,
                        "message": "Failed to goto http://example.test: net::ERR_BLOCKED_BY_CLIENT",
                    },
                    request=request,
                )

            def cache_handler(request: httpx.Request) -> httpx.Response:
                return httpx.Response(
                    422,
                    headers={"X-Reader-Dispatcher-Backend": "http://127.0.0.1:8172"},
                    json={
                        "code": 422,
                        "status": 422,
                        "message": "Raw markdown reader error",
                        "upstream_error": {
                            "status_code": 422,
                            "url": "http://127.0.0.1:8002/http://example.test",
                            "body": {
                                "message": "Failed to goto http://example.test: net::ERR_BLOCKED_BY_CLIENT"
                            },
                        },
                    },
                    request=request,
                )

            async with httpx.AsyncClient(transport=httpx.MockTransport(jina_handler)) as raw_client:
                raw_request = SimpleNamespace(
                    headers={"accept": "application/json"},
                    query_params={},
                    app=SimpleNamespace(state=SimpleNamespace(internal_reader_client=raw_client)),
                )
                with patch.object(enhanced_reader, "RAW_MARKDOWN_CACHE_ENABLED", False), patch.object(
                    enhanced_reader, "ENHANCED_READER_TIMING_ENABLED", False
                ):
                    raw_response = await enhanced_reader.read_raw_markdown("http://example.test", raw_request)

            async with httpx.AsyncClient(transport=httpx.MockTransport(cache_handler)) as enhanced_client:
                enhanced_request = SimpleNamespace(
                    headers={"accept": "application/json"},
                    query_params={},
                    app=SimpleNamespace(state=SimpleNamespace(internal_reader_client=enhanced_client)),
                )
                with patch.object(enhanced_reader, "ENHANCED_READER_CACHE_ENABLED", False), patch.object(
                    enhanced_reader, "ENHANCED_READER_FETCH_STRATEGY", "markdown_clean"
                ), patch.object(enhanced_reader, "ENHANCED_READER_TIMING_ENABLED", False):
                    enhanced_response = await enhanced_reader.read("http://example.test", enhanced_request)

            return json.loads(raw_response.body), json.loads(enhanced_response.body)

        raw_payload, enhanced_payload = asyncio.run(run())
        self.assertEqual(raw_payload["status"], 422)
        self.assertIn("ERR_BLOCKED_BY_CLIENT", raw_payload["message"])
        self.assertEqual(raw_payload["upstream_error"]["status_code"], 422)
        self.assertEqual(enhanced_payload["status"], 422)
        self.assertIn("ERR_BLOCKED_BY_CLIENT", enhanced_payload["message"])
        nested = enhanced_payload["upstream_error"]["body"]["upstream_error"]
        self.assertIn("ERR_BLOCKED_BY_CLIENT", nested["body"]["message"])


if __name__ == "__main__":
    unittest.main()
