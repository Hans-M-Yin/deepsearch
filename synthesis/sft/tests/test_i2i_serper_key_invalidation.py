"""Tests for fixed-key Serper Lens requests in i2i_search."""

from __future__ import annotations

import unittest
from unittest import mock

import requests

from synthesis.sft import tools


class I2ISerperKeyInvalidationTests(unittest.TestCase):
    def test_lens_credit_error_does_not_update_worker_pool_state(self) -> None:
        response = mock.Mock()
        response.status_code = 400
        response.text = '{"message":"Not enough credits","statusCode":400}'
        response.raise_for_status.side_effect = requests.HTTPError(
            "400 Client Error: Bad Request"
        )
        with (
            mock.patch.object(
                tools,
                "acquire_serper_api_key",
                return_value=("fixed-key", {"key_id": "fixed-client"}),
            ),
            mock.patch.object(tools.requests, "post", return_value=response),
        ):
            with self.assertRaises(requests.HTTPError):
                tools._image_search_via_serper("https://example.com/image.jpg")



if __name__ == "__main__":
    unittest.main()
