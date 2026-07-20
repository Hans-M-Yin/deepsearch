"""Tests for search-result URL registration and transparent read_url fallback."""

from __future__ import annotations

import tempfile
import unittest
from unittest import mock

from synthesis.sft import tools
from synthesis.sft.api_tools import ToolRuntimeContext


class UrlResourceFallbackTests(unittest.TestCase):
    def test_search_result_registers_all_resource_urls(self) -> None:
        context = ToolRuntimeContext(working_dir=tempfile.mkdtemp())
        context.register_search_output(
            "t2i_search",
            {
                "ok": True,
                "query": "example image",
                "results": [
                    {
                        "title": "Example",
                        "image_url": "https://cdn.example.com/original.jpg",
                        "thumbnail_url": "https://thumb.example.com/thumb.jpg",
                        "source_page_url": "https://example.com/page",
                        "rank": 1,
                    }
                ],
            },
        )

        resource = context.resolve_url_resource("https://cdn.example.com/original.jpg")
        self.assertIsNotNone(resource)
        assert resource is not None
        self.assertEqual(resource.thumbnail_url, "https://thumb.example.com/thumb.jpg")
        self.assertIs(context.resolve_url_resource(resource.thumbnail_url or ""), resource)
        self.assertIs(context.resolve_url_resource(resource.source_page_url or ""), resource)

    def test_image_read_falls_back_to_registered_thumbnail(self) -> None:
        resource = tools.UrlResource(
            primary_url="https://cdn.example.com/original.jpg",
            kind="image",
            image_url="https://cdn.example.com/original.jpg",
            thumbnail_url="https://thumb.example.com/thumb.jpg",
            source_page_url="https://example.com/page",
        )
        jpeg = b"\xff\xd8\xff" + b"x" * 20
        calls: list[tuple[str, str | None]] = []

        def fake_download(url: str, **kwargs):
            calls.append((url, kwargs.get("referer_url")))
            if "original" in url:
                raise RuntimeError("403")
            return jpeg, "image/jpeg"

        with (
            mock.patch.object(tools, "_probe_content_type", return_value="image/jpeg"),
            mock.patch.object(tools, "_download_binary", side_effect=fake_download),
            mock.patch.object(tools, "_maybe_resize_downloaded_image", return_value=(jpeg, "image/jpeg")),
        ):
            result = tools.read_url(resource.primary_url, resource=resource)

        self.assertTrue(result["ok"])
        self.assertEqual(result["resolved_via"], "thumbnail_url")
        self.assertEqual(result["resolved_url"], resource.thumbnail_url)
        self.assertEqual(calls[0][1], resource.source_page_url)


if __name__ == "__main__":
    unittest.main()
