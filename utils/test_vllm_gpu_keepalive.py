import argparse
import json
import unittest

from utils.vllm_gpu_keepalive import build_payload, parse_duration, random_prompt


class KeepaliveTests(unittest.TestCase):
    def test_prompt_starts_with_unique_nonce(self) -> None:
        first = random_prompt(4, 1)
        second = random_prompt(4, 1)
        self.assertNotEqual(first, second)
        self.assertIn("request_nonce_", first)

    def test_build_payload_merges_extra_body(self) -> None:
        args = argparse.Namespace(
            model="test-model", prompt_tokens=2, max_tokens=8, temperature=0.7,
            seed=11, extra_body=json.dumps({"top_p": 0.8}),
        )
        payload = build_payload(args, 3)
        self.assertEqual(payload["seed"], 14)
        self.assertEqual(payload["top_p"], 0.8)

    def test_parse_duration(self) -> None:
        self.assertEqual(parse_duration("2m"), 120)
        self.assertEqual(parse_duration("2.5"), 2.5)


if __name__ == "__main__":
    unittest.main()
