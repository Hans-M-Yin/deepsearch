"""Tests for Serper Lens key invalidation in i2i_search."""

from __future__ import annotations

import unittest
from unittest import mock

import requests

from synthesis.sft import tools


class I2ISerperKeyInvalidationTests(unittest.TestCase):
    def test_lens_not_enough_credits_disables_acquired_key(self) -> None:
        response = mock.Mock()
        response.status_code = 400
        response.text = '{"message":"Not enough credits","statusCode":400}'
        response.raise_for_status.side_effect = requests.HTTPError(
            "400 Client Error: Bad Request"
        )
        pool = mock.Mock()

        with (
            mock.patch.object(
                tools,
                "acquire_serper_api_key",
                return_value=("bad-key", {"key_id": "bad-key-id"}),
            ),
            mock.patch.object(tools.SerperApiKeyPool, "from_fixed_pool", return_value=pool),
            mock.patch.object(tools.requests, "post", return_value=response),
        ):
            with self.assertRaises(requests.HTTPError):
                tools._image_search_via_serper("https://example.com/image.jpg")

        pool.mark_credits_exhausted.assert_called_once_with(
            "bad-key-id",
            reason="serper_lens_http_400_credits_exhausted",
        )


if __name__ == "__main__":
    unittest.main()
