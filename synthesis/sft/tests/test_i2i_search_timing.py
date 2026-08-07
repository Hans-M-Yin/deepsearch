import contextlib
import io
import unittest

from synthesis.sft.api_tools import ToolRuntimeContext, execute_tool_call


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


if __name__ == "__main__":
    unittest.main()
