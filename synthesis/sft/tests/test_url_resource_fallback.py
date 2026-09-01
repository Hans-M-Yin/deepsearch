"""Tests for search-result URL registration and transparent read_url fallback."""

from __future__ import annotations

import os
import tempfile
import unittest
from io import BytesIO
from unittest import mock

from PIL import Image

from synthesis.firecrawl_client import FirecrawlBrowserImageDownload
from synthesis.sft import tools
from synthesis.sft.api_tools import ToolRuntimeContext


class UrlResourceFallbackTests(unittest.TestCase):
    def test_svg_is_rasterized_and_validated_before_return(self) -> None:
        svg = b'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" width="2" height="3"><rect width="2" height="3" fill="red"/></svg>'
        png_output = BytesIO()
        Image.new("RGBA", (2, 3), (255, 0, 0, 255)).save(png_output, format="PNG")

        class FakeCairoSVG:
            @staticmethod
            def svg2png(*, bytestring: bytes, unsafe: bool) -> bytes:
                self.assertEqual(bytestring, svg)
                self.assertFalse(unsafe)
                return png_output.getvalue()

        with mock.patch.object(tools, "cairosvg", FakeCairoSVG):
            converted, content_type = tools._maybe_resize_downloaded_image(
                svg,
                content_type="image/jpeg",
            )

        self.assertEqual(content_type, "image/png")
        with Image.open(BytesIO(converted)) as image:
            image.load()
            self.assertEqual(image.size, (2, 3))

    def test_search_result_registers_all_resource_urls(self) -> None:
        context = ToolRuntimeContext(working_dir=tempfile.mkdtemp())
        with mock.patch("synthesis.sft.tools.extract_url_semantic_keywords", return_value=""):
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

    def test_extensionless_image_resource_never_uses_text_reader_or_firecrawl(self) -> None:
        resource = tools.UrlResource(
            primary_url="https://lookaside.instagram.example/crawler/?media_id=123",
            kind="image",
            image_url="https://lookaside.instagram.example/crawler/?media_id=123",
        )
        jpeg = b"\xff\xd8\xff" + b"x" * 20
        with (
            mock.patch.object(tools, "_probe_content_type", return_value="text/html"),
            mock.patch.object(tools, "_download_binary", return_value=(jpeg, "image/jpeg")) as download,
            mock.patch.object(tools, "_maybe_resize_downloaded_image", return_value=(jpeg, "image/jpeg")),
            mock.patch.object(tools, "_read_document") as reader,
            mock.patch.object(tools, "_read_via_firecrawl") as firecrawl,
        ):
            result = tools.read_url(resource.primary_url, resource=resource)

        self.assertTrue(result["ok"])
        download.assert_called_once()
        reader.assert_not_called()
        firecrawl.assert_not_called()

    def test_image_read_uses_firecrawl_browser_backend_when_configured(self) -> None:
        resource = tools.UrlResource(
            primary_url="https://upload.wikimedia.org/wikipedia/commons/example.jpg",
            kind="image",
            image_url="https://upload.wikimedia.org/wikipedia/commons/example.jpg",
        )
        jpeg = b"\xff\xd8\xff" + b"firecrawl-image"

        class FakeBrowserDownloader:
            calls: list[tuple[str, str | None, float]] = []

            def download(self, url: str, *, referer_url: str | None, timeout_s: float):
                self.calls.append((url, referer_url, timeout_s))
                return FirecrawlBrowserImageDownload(
                    payload=jpeg,
                    content_type="image/jpeg",
                    requested_url=url,
                    resolved_url=url,
                    status_code=200,
                    session_id="session-1",
                    key_id="key-1",
                )

        fake_downloader = FakeBrowserDownloader()
        with (
            tempfile.TemporaryDirectory() as cache_dir,
            mock.patch.dict(
                os.environ,
                {
                    "SFT_WIKIMEDIA_IMAGE_DOWNLOAD_BACKEND": "firecrawl_browser",
                    "SFT_WIKIMEDIA_CACHE_DIR": cache_dir,
                },
                clear=False,
            ),
            mock.patch(
                "synthesis.firecrawl_client.FirecrawlBrowserImageDownloader.from_environment",
                return_value=fake_downloader,
            ),
            mock.patch.object(tools, "_download_binary") as direct_download,
            mock.patch.object(tools, "_maybe_resize_downloaded_image", return_value=(jpeg, "image/jpeg")),
        ):
            result = tools.read_url(resource.primary_url, resource=resource)

        self.assertTrue(result["ok"])
        self.assertEqual(result["resolved_via"], "firecrawl_browser:requested")
        self.assertEqual(len(fake_downloader.calls), 1)
        direct_download.assert_not_called()

    def test_wikimedia_cache_requires_exact_full_url(self) -> None:
        url = "https://upload.wikimedia.org/wikipedia/commons/example.jpg?width=640"
        other_url = "https://upload.wikimedia.org/wikipedia/commons/example.jpg?width=960"
        jpeg = b"\xff\xd8\xff" + b"x" * 20
        with tempfile.TemporaryDirectory() as cache_dir, mock.patch.dict(
            os.environ,
            {"SFT_WIKIMEDIA_CACHE_DIR": cache_dir},
            clear=False,
        ), mock.patch.object(
            tools,
            "_download_binary",
            return_value=(jpeg, "image/jpeg"),
        ) as download, mock.patch.object(
            tools,
            "_maybe_resize_downloaded_image",
            return_value=(jpeg, "image/jpeg"),
        ):
            first = tools.read_url(url)
            second = tools.read_url(url)
            third = tools.read_url(other_url)

        self.assertTrue(first["ok"])
        self.assertFalse(first["from_cache"])
        self.assertTrue(second["ok"])
        self.assertTrue(second["from_cache"])
        self.assertEqual(second["resolved_via"], "wikimedia_cache")
        self.assertTrue(third["ok"])
        self.assertFalse(third["from_cache"])
        self.assertEqual(download.call_count, 2)


if __name__ == "__main__":
    unittest.main()
