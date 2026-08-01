import unittest
from io import StringIO
from unittest.mock import patch

from synthesis.sft.api_tools import ToolRuntimeContext, execute_tool_call
from synthesis.sft.tools import postprocess_search_output


class SearchResultPostprocessTests(unittest.TestCase):
    def _fake_keywords(self, url: str, *, model_alias: str = "multimodal_process") -> str:
        del model_alias
        return "hint" if "example" in url else ""

    def test_t2i_hides_urls_and_registers_page_and_image_resources(self) -> None:
        raw = {
            "ok": True,
            "query": "example query",
            "results": [
                {
                    "title": "Example photo",
                    "image_url": "https://img.example/photo.jpg",
                    "thumbnail_url": "https://thumb.example/photo.jpg",
                    "source_page_url": "https://page.example/story",
                    "snippet": "Example snippet",
                    "rank": 1,
                }
            ],
        }
        with patch("synthesis.sft.tools.extract_url_semantic_keywords", self._fake_keywords):
            compact, resources = postprocess_search_output(tool_name="t2i_search", output=raw)

        item = compact["results"][0]
        self.assertNotIn("result_id", item)
        self.assertNotIn("rank", item)
        self.assertNotIn("image_url", item)
        self.assertNotIn("thumbnail_url", item)
        self.assertNotIn("source_page_url", item)
        self.assertTrue(item["image_id"].startswith("image_"))
        self.assertTrue(item["source_page_id"].startswith("page_"))
        image = next(resource for resource in resources if resource.kind == "image")
        self.assertEqual(image.fallback_urls, ["https://thumb.example/photo.jpg"])

    def test_resource_id_read_uses_registered_primary_url(self) -> None:
        context = ToolRuntimeContext(working_dir="/tmp/sft_resource_test")
        raw = {
            "ok": True,
            "query": "example",
            "results": [{"title": "Example", "url": "https://page.example/story", "snippet": "x", "rank": 1}],
        }
        with patch("synthesis.sft.tools.extract_url_semantic_keywords", self._fake_keywords):
            compact = context.postprocess_search_output("t2t_search", raw)
        resource_id = compact["results"][0]["source_page_id"]

        with (
            patch("synthesis.sft.api_tools.tools.read_url", return_value={"ok": True, "kind": "text"}) as read,
            patch("sys.stderr", new_callable=StringIO) as stderr,
        ):
            result = execute_tool_call(
                "read_url",
                {"resource_id": resource_id, "goal": "inspect"},
                context,
            )
        self.assertTrue(result.output["ok"])
        self.assertEqual(result.output["resource_id"], resource_id)
        self.assertEqual(read.call_args.kwargs["url"], "https://page.example/story")
        self.assertIn(f"resource_id={resource_id}", stderr.getvalue())
        self.assertIn("url=https://page.example/story", stderr.getvalue())

    def test_i2i_compact_output_omits_query_fields(self) -> None:
        raw = {
            "ok": True,
            "top_k": 1,
            "matches": [{"title": "Example", "source": "Example", "link": "https://page.example/x", "imageUrl": "https://img.example/x.jpg"}],
        }
        with patch("synthesis.sft.tools.extract_url_semantic_keywords", self._fake_keywords):
            compact, _resources = postprocess_search_output(tool_name="i2i_search", output=raw)
        self.assertNotIn("query", compact)
        self.assertNotIn("query_image", compact)
        item = compact["results"][0]
        self.assertIn("image_id", item)
        self.assertIn("source_page_id", item)
        self.assertLess(list(item).index("image_id"), list(item).index("source_page_id"))

    def test_keyword_extraction_submits_every_unique_page_and_image_url(self) -> None:
        raw = {
            "ok": True,
            "query": "example",
            "results": [
                {"title": "One", "url": "https://page.example/one"},
                {"title": "Two", "url": "https://page.example/two"},
            ],
        }
        calls: list[str] = []

        def keyword_extractor(url: str, *, model_alias: str = "multimodal_process") -> str:
            del model_alias
            calls.append(url)
            return ""

        with patch("synthesis.sft.tools.extract_url_semantic_keywords", keyword_extractor):
            postprocess_search_output(tool_name="t2t_search", output=raw)

        self.assertCountEqual(calls, ["https://page.example/one", "https://page.example/two"])

    def test_unknown_resource_id_does_not_attempt_direct_read(self) -> None:
        context = ToolRuntimeContext(working_dir="/tmp/sft_resource_test")
        result = execute_tool_call("read_url", {"resource_id": "page_deadbeef"}, context)
        self.assertFalse(result.output["ok"])
        self.assertIn("Unknown resource_id", result.output["error"])

    def test_legacy_url_still_reads_directly_when_not_registered(self) -> None:
        context = ToolRuntimeContext(working_dir="/tmp/sft_resource_test")
        with patch("synthesis.sft.api_tools.tools.read_url", return_value={"ok": True, "kind": "text"}) as read:
            result = execute_tool_call("read_url", {"url": "https://unknown.example/page"}, context)
        self.assertTrue(result.output["ok"])
        self.assertEqual(read.call_args.kwargs["url"], "https://unknown.example/page")


if __name__ == "__main__":
    unittest.main()
