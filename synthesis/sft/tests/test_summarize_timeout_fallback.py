import os
import unittest
from unittest import mock

from synthesis.model_worker import ModelResponse
from synthesis.sft import tools


class SummarizeTimeoutFallbackTest(unittest.TestCase):
    def test_timeout_retries_same_request_with_fallback_alias(self):
        primary_request = mock.Mock()
        fallback_request = mock.Mock()

        with (
            mock.patch.dict(
                os.environ,
                {"SUMMARIZE_FALLBACK_MODEL_ALIAS": "summarizer_fallback"},
                clear=False,
            ),
            mock.patch.object(tools, "_summarizer_model_alias", return_value="summarizer_primary"),
            mock.patch.object(
                tools,
                "_resolve_registered_model_alias",
                return_value={"served_model": "fallback-model"},
            ),
            mock.patch.object(
                tools.LLM_WORKER,
                "generate",
                side_effect=[
                    TimeoutError("summarizer request timed out"),
                    ModelResponse(content="<result>fallback evidence</result>"),
                ],
            ) as generate,
        ):
            result = tools.summarize_with_qwen(
                content="page content",
                goal="find the relevant fact",
                assistant_output="previous reasoning",
            )

        self.assertEqual(result, "fallback evidence")
        self.assertEqual(generate.call_count, 2)
        first_request = generate.call_args_list[0].args[0]
        second_request = generate.call_args_list[1].args[0]
        self.assertEqual(first_request.model, "summarizer_primary")
        self.assertEqual(second_request.model, "summarizer_fallback")
        self.assertEqual(first_request.messages, second_request.messages)
        self.assertEqual(
            first_request.metadata["prompt_cache_key"],
            second_request.metadata["prompt_cache_key"],
        )
        self.assertEqual(first_request.metadata["llm_retry_count"], 0)
        self.assertNotIn("llm_retry_count", second_request.metadata)

    def test_non_timeout_does_not_use_fallback(self):
        with (
            mock.patch.dict(
                os.environ,
                {"SUMMARIZE_FALLBACK_MODEL_ALIAS": "summarizer_fallback"},
                clear=False,
            ),
            mock.patch.object(tools, "_summarizer_model_alias", return_value="summarizer_primary"),
            mock.patch.object(
                tools.LLM_WORKER,
                "generate",
                side_effect=RuntimeError("invalid request"),
            ) as generate,
        ):
            result = tools.summarize_with_qwen(
                content="page content",
                goal="find the relevant fact",
            )

        self.assertEqual(result, "page content")
        self.assertEqual(generate.call_count, 1)


if __name__ == "__main__":
    unittest.main()
