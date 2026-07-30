import contextlib
import io
import unittest

from synthesis.sft.tools import i2i_search


class I2ISearchTimingTests(unittest.TestCase):
    def test_emits_backend_start_and_elapsed_time(self) -> None:
        def lookup(*, image_url: str, top_k: int):
            self.assertEqual(image_url, "https://example.com/query.jpg")
            self.assertEqual(top_k, 1)
            return []

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            result = i2i_search(
                "https://example.com/query.jpg",
                visual_lookup=lookup,
                top_k=1,
                max_retries=1,
            )

        self.assertTrue(result["ok"])
        output = stderr.getvalue()
        self.assertIn("backend_search_start", output)
        self.assertIn("backend_search_done", output)
        self.assertIn("elapsed_s=", output)


if __name__ == "__main__":
    unittest.main()
