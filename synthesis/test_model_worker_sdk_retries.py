"""Offline checks that SDK retries cannot multiply ModelWorker retries."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from synthesis.model_worker import (
    AzureOpenAIModelWorkerClient,
    OpenAIModelWorkerClient,
)


class ModelWorkerSDKRetriesTest(unittest.TestCase):
    def test_openai_sdk_retries_disabled(self):
        constructor = Mock()
        with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=constructor)}):
            OpenAIModelWorkerClient(
                model="test", api_key="test", base_url="https://example.invalid/v1",
                timeout_s=120,
            )
        self.assertEqual(constructor.call_args.kwargs["max_retries"], 0)
        self.assertEqual(constructor.call_args.kwargs["timeout"], 120)

    def test_azure_sdk_retries_disabled_including_rebuilt_clients(self):
        constructor = Mock()
        with patch.dict(sys.modules, {"openai": SimpleNamespace(AzureOpenAI=constructor)}):
            client = AzureOpenAIModelWorkerClient(
                model="test", api_key="test", azure_endpoint="https://example.invalid",
                api_version="test", timeout_s=120,
            )
            client._build_client()
        self.assertEqual(constructor.call_count, 2)
        for call in constructor.call_args_list:
            self.assertEqual(call.kwargs["max_retries"], 0)
            self.assertEqual(call.kwargs["timeout"], 120)


if __name__ == "__main__":
    unittest.main()
