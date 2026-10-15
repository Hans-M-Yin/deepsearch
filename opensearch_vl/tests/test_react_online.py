"""Isolated online ReAct tests; no external requests."""

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import run_infer
import run_infer_react_online as online
from opensearch_infer import config, pipeline, react_online, runners


def http_response(text="<answer>Alice</answer>"):
    result = mock.Mock()
    result.status_code = 200
    result.json.return_value = {
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    return result


class OnlineReActTests(unittest.TestCase):
    def test_format_error_final_answer_normalization(self):
        cases = [
            ('Alice', 'Alice'),
            ('<thinking>Evidence</thinking>\nAlice', 'Alice'),
            ('<thinking>Evidence</thinking><answer>Alice', 'Alice'),
            ('<answer>Alice</answer>', 'Alice'),
            ('', ''),
            ('\n', ''),
            ('<thinking>Alice might be right, let me check', ''),
            ('thinking>Alice</thinking><thinking>Still checking', ''),
            ('<thought>Alice might be right</thought>', ''),
            ('<tool_call>broken</tool_call>', ''),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(pipeline._format_error_answer_text(text), expected)

    def test_batch_keeps_format_errors_for_evaluation_and_resumes(self):
        import pandas as pd
        # Earlier malformed answers must not win the evaluator's first-match
        # regex after the model eventually returns a plaintext final response.
        outputs = iter([
            http_response('<answer>earlier guess<tool_call>broken</tool_call>'),
            http_response('<thinking>Done checking</thinking>\nAlice'),
            http_response('<answer>Bob'),
            http_response(''),
            http_response('<thinking>Unfinished reasoning about Carol'),
        ])
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / 'data.parquet'
            ids = ['plain', 'missing_close', 'empty', 'thinking_only']
            pd.DataFrame([{'id': id_, 'question': 'Who?'} for id_ in ids]).to_parquet(data)
            output = Path(directory) / 'out'
            argv = ['--model', 'gemini-test', '--base-url', 'https://gateway.example/v1',
                    '--data-path', str(data), '--output-dir', str(output),
                    '--parallel-workers', '1', '--retry-monitor-interval', '0',
                    '--format-error-policy', 'retain']
            with mock.patch.object(react_online.requests, 'post', side_effect=lambda *a, **k: next(outputs)) as post, \
                    mock.patch.object(online, '_git_output', return_value=''), \
                    mock.patch.object(pipeline.tools, 'execute_tool') as execute:
                self.assertEqual(online.main(argv), 0)
                self.assertEqual(online.main(argv), 0)
            self.assertEqual(post.call_count, 5)
            execute.assert_not_called()
            self.assertFalse((output / 'failures').exists())
            for id_, answer in zip(ids, ['Alice', 'Bob', '', '']):
                saved = json.loads((output / (id_ + '_trajectory.json')).read_text())
                self.assertEqual(saved['status'], 'completed')
                self.assertFalse(saved['format_valid'])
                self.assertEqual(saved['answer_missing'], not bool(answer))
                self.assertEqual(saved['final_response_text'], '<answer>' + answer + '</answer>')
                self.assertEqual(saved['format_error']['kind'], 'invalid_response')
            plain = json.loads((output / 'plain_trajectory.json').read_text())
            self.assertIn('earlier guess', plain['turns'][0]['response_text'])
            self.assertNotIn('earlier guess', plain['final_response_text'])
            snapshot = json.loads((output / 'run_config.json').read_text())
            self.assertEqual(snapshot['format_error_policy'], 'retain_for_evaluation')

    def test_format_error_policy_leaves_legacy_strict(self):
        response = {'candidates': [{'content': {'parts': [{'text': 'Alice'}]}}]}
        fake = SimpleNamespace(display_name='fake', infer=mock.Mock(return_value=response))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'without a complete <answer>'):
                pipeline.process_single_case({'id': 'strict', 'question': 'Who?'}, fake, directory, 0)
            saved = json.loads((Path(directory) / 'failures/strict_failure.json').read_text())
            self.assertEqual(saved['failure_kind'], 'invalid_response')
            self.assertFalse((Path(directory) / 'strict_trajectory.json').exists())

    def test_exhausted_malformed_tools_are_retained_but_valid_tool_loop_still_fails(self):
        for malformed in [True, False]:
            with self.subTest(malformed=malformed), tempfile.TemporaryDirectory() as directory:
                text = ('<tool_call>broken</tool_call>' if malformed else
                        '<tool_call>{"name":"t2t_search","arguments":{"query":"x"}}</tool_call>')
                fake = SimpleNamespace(display_name='fake', infer=mock.Mock(return_value={
                    'candidates': [{'content': {'parts': [{'text': text}]}}],
                }))
                with mock.patch.object(config, 'MAX_TURNS', 1), \
                        mock.patch.object(pipeline.tools, 'execute_tool', return_value=('evidence', {})):
                    if malformed:
                        saved = pipeline.process_single_case(
                            {'id': 'sample', 'question': 'Who?'}, fake, directory, 0,
                            retain_format_errors=True,
                        )
                        self.assertEqual(saved['format_error']['kind'], 'invalid_tool_call')
                        self.assertEqual(saved['final_response_text'], '<answer></answer>')
                        self.assertFalse((Path(directory) / 'failures').exists())
                    else:
                        with self.assertRaisesRegex(RuntimeError, '1-turn limit'):
                            pipeline.process_single_case(
                                {'id': 'sample', 'question': 'Who?'}, fake, directory, 0,
                                retain_format_errors=True,
                            )
                        saved = json.loads((Path(directory) / 'failures/sample_failure.json').read_text())
                        self.assertEqual(saved['failure_kind'], 'max_turns')

    def test_inference_error_is_not_retained_as_answer(self):
        response = {'inference_error': {'message': 'HTTP 400'},
                    'candidates': [{'content': {'parts': [{'text': '<answer>ERROR</answer>'}]}}]}
        fake = SimpleNamespace(display_name='fake', infer=mock.Mock(return_value=response))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                pipeline.process_single_case({'id': 'error', 'question': 'Who?'}, fake, directory, 0,
                                             retain_format_errors=True)
            saved = json.loads((Path(directory) / 'failures/error_failure.json').read_text())
            self.assertEqual(saved['failure_kind'], 'inference_error')
            self.assertFalse((Path(directory) / 'error_trajectory.json').exists())

    def args(self, *extra):
        return online._build_arg_parser().parse_args([
            "--data-path", "fake.parquet", "--output-dir", "fake-out", "--model", "gemini-test", *extra,
        ])

    def infer(self, cfg=None):
        instance = react_online.build_runner("gemini-test", base_url="https://gateway.example/v1", api_key="test-key")
        with mock.patch.object(react_online.requests, "post", return_value=http_response()) as post:
            result = instance.infer([{"role": "user", "parts": [{"text": "Who?"}]}], cfg=cfg)
        self.assertEqual(result["candidates"][0]["content"]["parts"][0]["text"], "<answer>Alice</answer>")
        self.assertEqual(post.call_args.args[0], "https://gateway.example/v1/chat/completions")
        return post.call_args.kwargs["json"]

    def test_defaults_are_optional_and_api_only(self):
        args = self.args()
        self.assertEqual(args.backend, "api")
        self.assertEqual(args.model, "gemini-test")
        for name in react_online.SAMPLING_PARAMETERS:
            self.assertIsNone(getattr(args, name))
        payload = self.infer()
        self.assertEqual(set(payload), {"model", "messages"})

    def test_explicit_zero_and_one_values_are_sent(self):
        args = self.args("--temperature", "0", "--max-tokens", "4096", "--top-p", "1",
                         "--top-k", "0", "--repetition-penalty", "1", "--presence-penalty", "0")
        cfg = react_online.OnlineInferenceConfig(**{name: getattr(args, name) for name in react_online.SAMPLING_PARAMETERS})
        payload = self.infer(cfg)
        for name in react_online.SAMPLING_PARAMETERS:
            self.assertEqual(payload[name], getattr(args, name))
        self.assertNotIn("chat_template_kwargs", payload)

    def test_only_selected_parameter_is_sent(self):
        payload = self.infer(react_online.OnlineInferenceConfig(top_k=20))
        self.assertEqual(set(payload), {"model", "messages", "top_k"})
        self.assertEqual(payload["top_k"], 20)

    def test_registry_placeholder_never_sends_qwen_template_kwargs(self):
        instance = react_online.build_runner("8b", served_model_name="gemini-test", base_url="https://gateway.example/v1")
        with mock.patch.object(react_online.requests, "post", return_value=http_response()) as post:
            instance.infer([])
        self.assertEqual(instance.model, "gemini-test")
        self.assertNotIn("chat_template_kwargs", post.call_args.kwargs["json"])

    def test_legacy_parser_and_runner_defaults_are_unchanged(self):
        args = run_infer._build_arg_parser().parse_args(["--data-path", "fake", "--output-dir", "fake-out"])
        self.assertEqual(args.backend, "local")
        self.assertEqual(args.temperature, 0.7)
        self.assertEqual(args.top_k, 20)
        self.assertEqual(args.repetition_penalty, 1.0)
        self.assertEqual(args.presence_penalty, 1.5)
        instance = runners.OpenAICompatibleRunner(config.MODEL_REGISTRY["8b"], "qwen", "https://gateway.example/v1")
        with mock.patch.object(runners.requests, "post", return_value=http_response()) as post:
            instance.infer([])
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["top_k"], 20)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})

    def test_builder_uses_environment_without_exposing_key(self):
        with mock.patch.dict(os.environ, {"OPENAI_BASE_URL": "https://gateway.example/v1", "OPENAI_API_KEY": "test-secret"}, clear=True):
            instance = react_online.build_runner("gemini-test")
        self.assertEqual(instance.base_url, "https://gateway.example/v1")
        self.assertEqual(instance.api_key, "test-secret")
        with self.assertRaises(ValueError):
            react_online.build_runner("gemini-test", backend="local")

    def test_azure_alias_preserves_thinking_but_not_alias_sampling_defaults(self):
        from synthesis.model_worker import LLM_WORKER
        alias = {"served_model": "gemini-2.5-pro", "client_type": "azure_openai",
                 "azure_endpoint": "https://internal.example/api/modelhub/online/v2/crawl",
                 "api_version": "2024-03-01-preview", "api_key": "test-secret",
                 "sampling_params": {"max_tokens": 12000, "temperature": 0.7, "top_k": 20,
                                     "thinking": {"include_thoughts": True, "budget_tokens": 8192}}}
        with mock.patch.object(LLM_WORKER, "get_model", return_value=alias):
            instance = react_online.build_runner(None, model_alias="gemini25pro_internal_azure")
        endpoint = mock.Mock()
        endpoint.create.return_value = http_response().json()
        instance.client = SimpleNamespace(chat=SimpleNamespace(completions=endpoint))
        instance.infer([], cfg=react_online.OnlineInferenceConfig(max_tokens=32768, temperature=0))
        kwargs = endpoint.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "gemini-2.5-pro")
        self.assertEqual(kwargs["max_tokens"], 32768)
        self.assertEqual(kwargs["temperature"], 0)
        self.assertEqual(kwargs["extra_body"], {"thinking": alias["sampling_params"]["thinking"]})
        self.assertNotIn("top_k", kwargs)
        self.assertNotIn("chat_template_kwargs", kwargs["extra_body"])
        instance.infer([])
        self.assertNotIn("max_tokens", endpoint.create.call_args.kwargs)
        self.assertNotIn("temperature", endpoint.create.call_args.kwargs)
        self.assertNotIn("test-secret", json.dumps(instance.configuration()))

    def test_azure_sdk_route_auth_version_and_extra_body_are_serialized_offline(self):
        import httpx
        from openai import AzureOpenAI
        captured = []

        def handle(request):
            captured.append(request)
            body = {"id": "chatcmpl-test", "object": "chat.completion", "created": 0,
                    "model": "gemini_3f", **http_response().json()}
            return httpx.Response(200, json=body)

        with httpx.Client(transport=httpx.MockTransport(handle), trust_env=False) as http_client:
            client = AzureOpenAI(azure_endpoint="https://internal.example/api/modelhub/online/v2/crawl",
                                 api_version="2024-03-01-preview", api_key="test-secret",
                                 http_client=http_client, max_retries=0)
            instance = react_online.build_runner("gemini_3f", client_type="azure_openai",
                                                 azure_endpoint="https://internal.example/api/modelhub/online/v2/crawl",
                                                 api_version="2024-03-01-preview", api_key="test-secret")
            instance.client = client
            instance.extra_body = {"thinking": {"include_thoughts": True, "thinking_level": "high"}}
            result = instance.infer([], cfg=react_online.OnlineInferenceConfig(max_tokens=32768, top_k=0, repetition_penalty=1))
        self.assertEqual(result["candidates"][0]["content"]["parts"][0]["text"], "<answer>Alice</answer>")
        self.assertEqual(len(captured), 1)
        request = captured[0]
        self.assertEqual(request.url.path, "/api/modelhub/online/v2/crawl/openai/deployments/gemini_3f/chat/completions")
        self.assertEqual(request.url.params["api-version"], "2024-03-01-preview")
        self.assertEqual(request.headers["api-key"], "test-secret")
        body = json.loads(request.content)
        self.assertEqual(body["top_k"], 0)
        self.assertEqual(body["repetition_penalty"], 1)
        self.assertEqual(body["thinking"]["thinking_level"], "high")
        self.assertNotIn("extra_body", body)
        self.assertNotIn("temperature", body)
        self.assertNotIn("chat_template_kwargs", body)

    def test_azure_sdk_initialization_uses_configured_headers(self):
        instance = react_online.build_runner("gemini-test", client_type="azure_openai",
                                             azure_endpoint="https://internal.example", api_version="test-version",
                                             api_key="test-secret")
        instance.default_headers = {"X-TT-LOGID": "test-id"}
        with mock.patch("openai.AzureOpenAI") as constructor:
            instance.load()
            instance.load()
        constructor.assert_called_once_with(azure_endpoint="https://internal.example", api_version="test-version",
                                            api_key="test-secret", timeout=600, max_retries=0,
                                            default_headers={"X-TT-LOGID": "test-id"})

    def test_alias_batch_command_saves_route_and_resumes(self):
        import pandas as pd
        from synthesis.model_worker import LLM_WORKER
        endpoint = mock.Mock()
        endpoint.create.return_value = http_response().json()
        client = SimpleNamespace(chat=SimpleNamespace(completions=endpoint))
        alias = {"served_model": "gemini_3f", "client_type": "azure_openai",
                 "azure_endpoint": "https://internal.example", "api_version": "2024-03-01-preview", "api_key": "test-secret",
                 "sampling_params": {"max_tokens": 32768, "thinking": {"thinking_level": "high", "include_thoughts": True}}}
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data.parquet"
            pd.DataFrame([{"id": "example", "question": "Who?"}]).to_parquet(data)
            output = Path(directory) / "out"
            argv = ["--model-alias", "gemini3flash_internal_azure", "--data-path", str(data),
                    "--output-dir", str(output), "--dataset", "test", "--limit", "300",
                    "--parallel-workers", "16", "--max-tokens", "32768", "--retry-monitor-interval", "0"]
            with mock.patch.object(LLM_WORKER, "get_model", return_value=alias), mock.patch("openai.AzureOpenAI", return_value=client), mock.patch.object(online, "_git_output", return_value=""):
                self.assertEqual(online.main(argv), 0)
                self.assertEqual(online.main(argv), 0)
            endpoint.create.assert_called_once()
            snapshot = json.loads((output / "run_config.json").read_text())
            self.assertEqual(snapshot["online_api"]["model_alias"], "gemini3flash_internal_azure")
            self.assertEqual(snapshot["online_api"]["client_type"], "azure_openai")
            self.assertEqual(snapshot["online_api"]["extra_body"], {"thinking": alias["sampling_params"]["thinking"]})
            self.assertNotIn("test-secret", json.dumps(snapshot))

    def test_alias_cannot_be_combined_with_direct_model(self):
        with self.assertRaises(ValueError):
            react_online.build_runner("other-model", model_alias="registered")

    def test_batch_text_tool_loop_config_and_resume(self):
        import pandas as pd
        requests_sent = []
        outputs = iter([
            http_response('<tool_call>{"name":"t2t_search","arguments":{"query":"Alice evidence"}}</tool_call>'),
            http_response(),
        ])

        def post(*args, **kwargs):
            requests_sent.append(deepcopy(kwargs["json"]))
            return next(outputs)

        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "data.parquet"
            pd.DataFrame([{"id": "example", "question": "Who?"}]).to_parquet(data_path)
            output_dir = Path(directory) / "out"
            argv = ["--model", "gemini-test", "--base-url", "https://gateway.example/v1", "--api-key", "test-secret",
                    "--data-path", str(data_path), "--output-dir", str(output_dir),
                    "--max-tokens", "4096", "--temperature", "0", "--retry-monitor-interval", "0"]
            with mock.patch.object(react_online.requests, "post", side_effect=post), mock.patch.object(
                pipeline.tools, "execute_tool", return_value=("Found Alice", {})
            ) as execute, mock.patch.object(online, "_git_output", return_value=""):
                self.assertEqual(online.main(argv), 0)
                self.assertEqual(online.main(argv), 0)
            execute.assert_called_once()
            self.assertEqual(len(requests_sent), 2)
            self.assertEqual(set(requests_sent[0]), {"model", "messages", "temperature", "max_tokens"})
            self.assertIn("Found Alice", json.dumps(requests_sent[1]["messages"]))
            saved = json.loads((output_dir / "example_trajectory.json").read_text())
            self.assertEqual(saved["final_response_text"], "<answer>Alice</answer>")
            self.assertEqual(saved["final_response_text_scope"], "final_turn_only")
            snapshot = json.loads((output_dir / "run_config.json").read_text())
            self.assertEqual(snapshot["inference"], {"temperature": 0.0, "max_tokens": 4096})
            self.assertEqual(snapshot["react_mode"], "text_tool_call")
            self.assertNotIn("test-secret", json.dumps(snapshot))
            self.assertTrue(snapshot["command"]["entrypoint"].endswith("run_infer_react_online.py"))


if __name__ == "__main__":
    unittest.main()
