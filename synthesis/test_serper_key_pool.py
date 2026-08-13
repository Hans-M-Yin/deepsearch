"""Regression tests for Serper key invalidation and rotation."""

from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import unittest
from urllib.error import HTTPError

from synthesis.search_client import SerperApiKeyPool, SerperSearchClient, _is_serper_credit_exhausted


class _Response:
    def __init__(self, payload: bytes, status_code: int = 200) -> None:
        self._payload = payload
        self._status_code = status_code

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback

    def read(self) -> bytes:
        return self._payload

    def getcode(self) -> int:
        return self._status_code


class _FakeOpener:
    def __init__(self) -> None:
        self.keys: list[str] = []

    def open(self, request, *, timeout: float):
        del timeout
        key = request.get_header("X-api-key")
        self.keys.append(key)
        if key == "bad-key":
            raise HTTPError(
                request.full_url,
                400,
                "Bad Request",
                hdrs=None,
                fp=io.BytesIO(b'{"message":"Not enough credits","statusCode":400}'),
            )
        return _Response(b'{"images":[]}')


class SerperKeyPoolTests(unittest.TestCase):
    def test_not_enough_credits_is_credit_exhaustion_but_rate_limit_is_not(self) -> None:
        self.assertTrue(
            _is_serper_credit_exhausted(
                status_code=400,
                response_body='{"message":"Not enough credits","statusCode":400}',
            )
        )
        self.assertFalse(
            _is_serper_credit_exhausted(
                status_code=429,
                response_body='{"message":"Not enough credits"}',
            )
        )

    def test_credit_error_disables_key_and_next_request_skips_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "serper_pool_state.json"
            client = SerperSearchClient(
                api_keys=["bad-key", "good-key"],
                images_url="https://google.serper.dev/images",
                pool_state_path=state_path,
                pool_default_credits=100,
                pool_min_remaining=0,
            )
            opener = _FakeOpener()
            client._url_opener = opener

            with self.assertRaisesRegex(RuntimeError, "Not enough credits"):
                client.search_image("Indian motorcycle first prototype", limit=10, hl="en")

            client.search_image("Indian motorcycle first prototype", limit=10, hl="en")

            self.assertEqual(opener.keys, ["bad-key", "good-key"])
            state = json.loads(state_path.read_text(encoding="utf-8"))
            bad_record = state["keys"][SerperApiKeyPool.key_id("bad-key")]
            self.assertTrue(bad_record["disabled"])
            self.assertEqual(bad_record["remaining_credits"], 0)
            self.assertEqual(bad_record["disabled_reason"], "serper_http_400_credits_exhausted")
            self.assertEqual(client.key_pool.status()["available_key_count"], 1)


if __name__ == "__main__":
    unittest.main()
