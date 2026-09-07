import contextlib
import io
import unittest
from unittest.mock import patch

from synthesis.sft.api_tools import ToolRuntimeContext, execute_tool_call
from synthesis.sft import tools


class I2ISearchTimingTests(unittest.TestCase):
    def test_does_not_emit_temporary_timing_debug(self) -> None:
        def lookup(*, image_url: str, top_k: int):
            self.assertEqual(image_url, "https://example.com/query.jpg")
            self.assertEqual(top_k, 1)
            return [{"title": "match"}]

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            result = execute_tool_call(
                "i2i_search",
                {"url": "https://example.com/query.jpg", "top_k": 1},
                ToolRuntimeContext(
                    working_dir="/tmp/sft_i2i_timing_test",
                    visual_lookup=lookup,
                ),
            )

        self.assertTrue(result.output["ok"])
        output = stderr.getvalue()
        self.assertNotIn("[i2i_search OSS timing]", output)
        self.assertNotIn("backend_search_start", output)
        self.assertNotIn("backend_search_done", output)

    def test_empty_result_retry_debug_includes_image_url(self) -> None:
        image_url = "https://cdn.example.com/images/query-image.png?token=abc123"
        calls = 0

        def lookup(*, image_url: str, top_k: int):
            nonlocal calls
            calls += 1
            if calls == 1:
                return {
                    "searchParameters": {"url": image_url},
                    "organic": [],
                    "diagnostic": "upstream-empty",
                }
            return [{"title": "match"}]

        stderr = io.StringIO()
        with patch.object(tools, "tracked_sleep"), contextlib.redirect_stderr(stderr):
            result = tools.i2i_search(
                image_url=image_url,
                visual_lookup=lookup,
                top_k=1,
            )

        self.assertTrue(result["ok"])
        self.assertIn("retry_reason='empty_result'", stderr.getvalue())
        self.assertIn(f"url={image_url!r}", stderr.getvalue())
        self.assertIn('"diagnostic": "upstream-empty"', stderr.getvalue())

    def test_empty_result_retries_at_most_three_times(self) -> None:
        calls = 0

        def lookup(*, image_url: str, top_k: int):
            del image_url, top_k
            nonlocal calls
            calls += 1
            return []

        with patch.object(tools, "tracked_sleep"):
            result = tools.i2i_search(
                image_url="https://cdn.example.com/always-empty.jpg",
                visual_lookup=lookup,
                top_k=1,
            )

        self.assertFalse(result["ok"])
        self.assertEqual(result["matches"], [])
        self.assertIn("returned no results", result["error"])
        self.assertEqual(calls, 1 + tools.I2I_EMPTY_RESULT_RETRIES)


if __name__ == "__main__":
    unittest.main()
