"""Tests for transient Serper failures in t2i_search."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from synthesis.sft import tools


class T2ISearchRetryTests(unittest.TestCase):
    def test_retries_once_after_transient_serper_failure(self) -> None:
        response = SimpleNamespace(
            results=[
                SimpleNamespace(
                    title="Example image",
                    image_url="https://images.example/item.jpg",
                    thumbnail_url="",
                    source_page_url="https://example.com/page",
                    snippet="Example",
                    rank=1,
                )
            ]
        )
        client = mock.Mock()
        client.search_image.side_effect = [TimeoutError("timed out"), response]

        with (
            mock.patch.object(tools, "_serper_client", return_value=client),
            mock.patch.object(tools.time, "sleep") as sleep,
        ):
            result = tools.t2i_search("example query")

        self.assertTrue(result["ok"])
        self.assertEqual(client.search_image.call_count, 2)
        sleep.assert_called_once_with(5)

    def test_allows_five_timeout_retries_after_initial_attempt(self) -> None:
        client = mock.Mock()
        client.search_image.side_effect = TimeoutError("connect timed out")

        with (
            mock.patch.object(tools, "_serper_client", return_value=client),
            mock.patch.object(tools.time, "sleep") as sleep,
        ):
            with self.assertRaises(TimeoutError):
                tools.t2i_search("example query")

        self.assertEqual(client.search_image.call_count, 6)
        self.assertEqual(sleep.call_args_list, [mock.call(5), mock.call(10), mock.call(15), mock.call(20), mock.call(25)])


if __name__ == "__main__":
    unittest.main()
