"""Tests for URL-resource fallback wiring in the run_infer dispatcher."""

from __future__ import annotations

import json
import tempfile
import unittest
from unittest import mock

from opensearch_vl.opensearch_infer import tools
from synthesis.sft import tools as sft_tools


class RunInferUrlResourceTests(unittest.TestCase):
    def test_search_then_read_resolves_registered_resource(self) -> None:
        registry: dict[str, sft_tools.UrlResource] = {}
        search_output = {
            "ok": True,
            "query": "example",
            "results": [
                {
                    "title": "Example",
                    "image_url": "https://cdn.example.com/original.jpg",
                    "thumbnail_url": "https://thumb.example.com/thumb.jpg",
                    "source_page_url": "https://example.com/page",
                    "rank": 1,
                }
            ],
        }
        with mock.patch.object(sft_tools, "t2i_search", return_value=search_output):
            message, images = tools.execute_tool(
                {"name": "t2i_search", "parameters": {"query": "example"}},
                {},
                "case",
                0,
                0,
                tempfile.mkdtemp(),
                url_registry=registry,
            )
        self.assertFalse(images)
        self.assertTrue(json.loads(message)["ok"])

        captured: dict[str, object] = {}

        def fake_read_url(*, url: str, goal: str = "", resource=None, **kwargs):
            captured["url"] = url
            captured["resource"] = resource
            return {"ok": True, "kind": "text", "url": url, "title": "x", "content": "x"}

        with mock.patch.object(sft_tools, "read_url", side_effect=fake_read_url):
            tools.execute_tool(
                {"name": "read_url", "parameters": {"url": "https://cdn.example.com/original.jpg"}},
                {},
                "case",
                0,
                1,
                tempfile.mkdtemp(),
                url_registry=registry,
            )
        resource = captured["resource"]
        self.assertIsInstance(resource, sft_tools.UrlResource)
        assert isinstance(resource, sft_tools.UrlResource)
        self.assertEqual(resource.thumbnail_url, "https://thumb.example.com/thumb.jpg")


if __name__ == "__main__":
    unittest.main()
