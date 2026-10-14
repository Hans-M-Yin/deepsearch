"""JSON format retries stay local to the model request."""

import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from synthesis.post_process.fix.llm import JSON_REQUIREMENT, JsonModel
from synthesis.post_process.fix.storage import Cache


class JsonRetryTests(unittest.TestCase):
    def test_retry_and_cache_only_success(self):
        with tempfile.TemporaryDirectory() as root:
            model = JsonModel(Cache(root))
            with patch("synthesis.model_worker.LLM_WORKER.generate") as generate:
                generate.side_effect = [
                    SimpleNamespace(content='{"key" "bad"}'),
                    SimpleNamespace(content='{"key": "good"}'),
                ]
                result = model.ask("test", "Task", {"question": "q"})
                self.assertEqual(result, {"key": "good"})
                self.assertEqual(model.ask("test", "Task", {"question": "q"}), result)
                self.assertEqual(generate.call_count, 2)
                first, second = [call.args[0] for call in generate.call_args_list]
                self.assertIn(JSON_REQUIREMENT, first.messages[0].content)
                self.assertEqual(first.messages[1].content, second.messages[1].content)
                self.assertIn("could not be parsed", second.messages[-1].content)

    def test_exhaustion_is_bounded_and_not_cached(self):
        with tempfile.TemporaryDirectory() as root:
            model = JsonModel(Cache(root))
            with patch("synthesis.model_worker.LLM_WORKER.generate") as generate:
                generate.return_value = SimpleNamespace(content="bad json")
                with self.assertRaises(json.JSONDecodeError):
                    model.ask("test", "Task", {})
                self.assertEqual(generate.call_count, 3)
                generate.return_value = SimpleNamespace(content="{}")
                self.assertEqual(model.ask("test", "Task", {}), {})
                self.assertEqual(generate.call_count, 4)

    def test_non_object_is_retried(self):
        with tempfile.TemporaryDirectory() as root:
            with patch("synthesis.model_worker.LLM_WORKER.generate") as generate:
                generate.side_effect = [
                    SimpleNamespace(content="[]"),
                    SimpleNamespace(content="{}"),
                ]
                self.assertEqual(JsonModel(Cache(root)).ask("test", "Task", {}), {})
                self.assertEqual(generate.call_count, 2)

    def test_transport_errors_are_not_format_retries(self):
        with tempfile.TemporaryDirectory() as root:
            with patch(
                "synthesis.model_worker.LLM_WORKER.generate", side_effect=TimeoutError
            ) as generate:
                with self.assertRaises(TimeoutError):
                    JsonModel(Cache(root)).ask("test", "Task", {})
                self.assertEqual(generate.call_count, 1)
