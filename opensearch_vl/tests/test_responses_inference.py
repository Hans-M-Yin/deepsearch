"""Responses protocol regression tests; no live LLM/search requests."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import run_infer
import run_infer_responses
from opensearch_infer import responses


def text_item(text, phase=None):
    item = {"type": "message", "id": "msg-1", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}
    if phase:
        item["phase"] = phase
    return item


def call_item(call_id="call-1", arguments='{"query":"evidence"}', name="t2t_search"):
    return {"type": "function_call", "id": "item-" + call_id, "call_id": call_id,
            "name": name, "arguments": arguments, "status": "completed"}


def response(*items, response_id="resp-1", status="completed", **extra):
    return {"id": response_id, "status": status, "output": list(items), **extra}


def encrypted_error():
    error = RuntimeError("Gateway bad request")
    error.status_code = 400
    # The gateway wraps the useful upstream code inside the message.
    error.code = "-4003"
    error.body = {"error": {"code": "-4003", "message":
        "code: invalid_encrypted_content; encrypted content could not be decrypted or parsed"}}
    return error


class FakeEndpoint:
    def __init__(self, *items):
        self.items = list(items)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        result = self.items.pop(0)
        if isinstance(result, Exception):
            raise result
        return deepcopy(result)


def runner(*items, **kwargs):
    instance = responses.ResponsesRunner(model="fake-model", base_url="http://test/v1", api_key="secret",
                                         request_options={"max_output_tokens": 4096}, **kwargs)
    endpoint = FakeEndpoint(*items)
    instance.client = SimpleNamespace(responses=endpoint)
    return instance, endpoint


class ResponsesInferenceTests(unittest.TestCase):
    def run_case(self, instance, directory, **kwargs):
        return responses.process_single_case({"id": "example", "question": "Who is it?", "images": []},
                                               instance, directory, 0, **kwargs)

    def test_native_replay_preserves_reasoning_phase_and_real_tool_arguments(self):
        reasoning = {"type": "reasoning", "id": "rs-1", "summary": [], "encrypted_content": "opaque-state"}
        commentary = text_item("I will search.", "commentary")
        instance, endpoint = runner(response(reasoning, commentary, call_item()),
                                    response(text_item("The answer is Alice.", "final_answer"), response_id="resp-2"))
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            responses.tools.sft_tools, "t2t_search", return_value={"ok": False, "error": "mock observation"}
        ) as search:
            result = self.run_case(instance, directory)
            saved = json.loads((Path(directory) / "example_trajectory.json").read_text())
            self.assertIn("example", run_infer._completed_case_ids(directory))
        search.assert_called_once()
        self.assertEqual(search.call_args.kwargs["query"], "evidence")
        replay = endpoint.requests[1]["input"]
        self.assertEqual(replay[1:4], [reasoning, commentary, call_item()])
        self.assertEqual(replay[4]["call_id"], "call-1")
        self.assertEqual(replay[4]["type"], "function_call_output")
        self.assertNotIn("previous_response_id", endpoint.requests[1])
        self.assertEqual(result["final_response_text"], "<answer>The answer is Alice.</answer>")
        self.assertNotIn("I will search", result["final_response_text"])
        self.assertNotIn("opaque-state", json.dumps(saved))
        self.assertEqual(saved["turns"][0]["response"]["output"][1]["phase"], "commentary")

    def test_previous_response_sends_only_outputs_and_isolates_cases(self):
        instance, endpoint = runner(response(call_item()), response(text_item("Alice"), response_id="resp-2"),
                                    response(text_item("Bob"), response_id="resp-3"), state_mode="previous-response-id")
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(responses.tools, "execute_tool", return_value=("found", {})):
            self.run_case(instance, directory)
            responses.process_single_case({"id": "second", "question": "Second?"}, instance, directory, 1)
        self.assertEqual(endpoint.requests[1]["previous_response_id"], "resp-1")
        self.assertEqual(endpoint.requests[1]["input"], [{"type": "function_call_output", "call_id": "call-1", "output": "found"}])
        self.assertNotIn("previous_response_id", endpoint.requests[2])
        self.assertEqual(len(endpoint.requests[2]["input"]), 1)

    def test_previous_id_not_found_falls_back_with_full_raw_state(self):
        error = RuntimeError("previous_response_id not found")
        error.status_code = 404
        reasoning = {"type": "reasoning", "id": "rs-1", "summary": [], "encrypted_content": "state"}
        instance, endpoint = runner(response(reasoning, call_item()), error, response(text_item("Alice")), state_mode="previous-response-id")
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(responses.tools, "execute_tool", return_value=("found", {})):
            result = self.run_case(instance, directory)
        self.assertEqual(result["state_fallback"], "full_native_output_replay")
        self.assertIn(reasoning, endpoint.requests[2]["input"])
        self.assertNotIn("previous_response_id", endpoint.requests[2])

    def test_encrypted_fallback_is_case_local_and_preserves_tool_pairs_phase_images(self):
        reasoning = {"type": "reasoning", "id": "rs-broken", "summary": [], "encrypted_content": "broken"}
        fresh = {"type": "reasoning", "id": "rs-fresh", "summary": [], "encrypted_content": "fresh"}
        commentary = text_item("I will inspect the evidence.", "commentary")
        instance, endpoint = runner(
            response(reasoning, commentary, call_item()), encrypted_error(),
            response(fresh, call_item("call-2")), response(text_item("Alice", "final_answer")),
            response(text_item("Bob", "final_answer")),
        )
        instance.request_options.update({"include": ["reasoning.encrypted_content"], "reasoning": {"effort": "high"}})
        original_options = deepcopy(instance.request_options)
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(responses.tools, "execute_tool", return_value=("found", {"img-1": "https://example.com/image.png"})) as execute:
            result = self.run_case(instance, directory)
            saved = json.loads((Path(directory) / "example_trajectory.json").read_text())
            second = responses.process_single_case({"id": "second", "question": "Second?"}, instance, directory, 1)
        self.assertEqual(execute.call_count, 2)  # Failed API request never re-executes the first tool.
        self.assertEqual(len(result["turns"]), 3)  # Recovery retries the same agent turn.
        self.assertEqual(len(endpoint.requests), 5)
        self.assertIn(reasoning, endpoint.requests[1]["input"])
        cleaned = endpoint.requests[2]["input"]
        self.assertNotIn("broken", json.dumps(cleaned))
        self.assertFalse(any(item.get("type") == "reasoning" for item in cleaned))
        copied_message = next(item for item in cleaned if item.get("type") == "message")
        self.assertEqual(copied_message["phase"], "commentary")
        self.assertNotIn("id", copied_message)
        calls = [item for item in cleaned if item.get("type") == "function_call"]
        outputs = [item for item in cleaned if item.get("type") == "function_call_output"]
        self.assertEqual(calls[0]["call_id"], outputs[0]["call_id"])
        self.assertEqual(calls[0]["arguments"], call_item()["arguments"])
        self.assertNotIn("id", calls[0])
        self.assertIn("https://example.com/image.png", json.dumps(cleaned))
        self.assertNotIn("include", endpoint.requests[2])
        self.assertNotIn("fresh", json.dumps(endpoint.requests[3]["input"]))
        self.assertEqual(endpoint.requests[3]["reasoning"], {"effort": "high"})
        self.assertEqual(endpoint.requests[4]["include"], ["reasoning.encrypted_content"])
        self.assertEqual(instance.request_options, original_options)
        self.assertTrue(saved["reasoning_state_dropped"])
        self.assertEqual(saved["reasoning_items_dropped"], 2)
        self.assertEqual(saved["state_mode_effective"], "replay_without_reasoning")
        self.assertTrue(saved["fallback_events"][0]["request_succeeded"])
        self.assertFalse(second["reasoning_replay_disabled"])
        self.assertEqual(second["fallback_events"], [])

    def test_explicit_fail_policy_does_not_drop_reasoning(self):
        reasoning = {"type": "reasoning", "id": "rs-1", "encrypted_content": "broken", "summary": []}
        instance, endpoint = runner(response(reasoning, call_item()), encrypted_error(), encrypted_reasoning_fallback="fail")
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(responses.tools, "execute_tool", return_value=("found", {})):
            with self.assertRaises(RuntimeError):
                self.run_case(instance, directory)
            saved = json.loads((Path(directory) / "failures/example_failure.json").read_text())
        self.assertEqual(len(endpoint.requests), 2)
        self.assertEqual(saved["failure_kind"], "invalid_encrypted_content")
        self.assertEqual(saved["fallback_events"], [])
        self.assertFalse(saved["reasoning_state_dropped"])

    def test_repeated_encrypted_error_is_terminal_after_one_clean_retry(self):
        reasoning = {"type": "reasoning", "id": "rs-1", "encrypted_content": "broken", "summary": []}
        instance, endpoint = runner(response(reasoning, call_item()), encrypted_error(), encrypted_error())
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(responses.tools, "execute_tool", return_value=("found", {})) as execute:
            with self.assertRaises(RuntimeError):
                self.run_case(instance, directory)
            saved = json.loads((Path(directory) / "failures/example_failure.json").read_text())
            self.assertEqual(run_infer._completed_case_ids(directory), set())
        execute.assert_called_once()
        self.assertEqual(len(endpoint.requests), 3)
        self.assertEqual(saved["failure_kind"], "invalid_encrypted_content")
        self.assertFalse(saved["fallback_events"][0]["request_succeeded"])
        self.assertEqual(saved["state_mode_effective"], "replay_without_reasoning")

    def test_unrelated_bad_request_does_not_trigger_lossy_fallback(self):
        error = RuntimeError("invalid function parameters")
        error.status_code = 400
        reasoning = {"type": "reasoning", "id": "rs-1", "encrypted_content": "state", "summary": []}
        instance, endpoint = runner(response(reasoning, call_item()), error)
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(responses.tools, "execute_tool", return_value=("found", {})):
            with self.assertRaises(RuntimeError):
                self.run_case(instance, directory)
            saved = json.loads((Path(directory) / "failures/example_failure.json").read_text())
        self.assertEqual(len(endpoint.requests), 2)
        self.assertEqual(saved["failure_kind"], "inference_error")
        self.assertEqual(saved["fallback_events"], [])

    def test_missing_stored_id_then_invalid_encryption_chains_bounded_recovery(self):
        missing = RuntimeError("previous_response_id not found")
        missing.status_code = 404
        reasoning = {"type": "reasoning", "id": "rs-1", "encrypted_content": "state", "summary": []}
        instance, endpoint = runner(response(reasoning, call_item()), missing, encrypted_error(), response(text_item("Alice")), state_mode="previous-response-id")
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(responses.tools, "execute_tool", return_value=("found", {})) as execute:
            result = self.run_case(instance, directory)
        execute.assert_called_once()
        self.assertEqual(len(endpoint.requests), 4)
        self.assertEqual(endpoint.requests[1]["previous_response_id"], "resp-1")
        self.assertIn(reasoning, endpoint.requests[2]["input"])
        self.assertNotIn("previous_response_id", endpoint.requests[3])
        self.assertFalse(any(item.get("type") == "reasoning" for item in endpoint.requests[3]["input"]))
        self.assertEqual(len(result["fallback_events"]), 2)
        self.assertTrue(all(event["request_succeeded"] for event in result["fallback_events"]))

    def test_invalid_encryption_in_stored_mode_recovers_without_previous_id(self):
        instance, endpoint = runner(response(call_item()), encrypted_error(), response(text_item("Alice")),
                                    state_mode="previous-response-id", replay_reasoning=False)
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(responses.tools, "execute_tool", return_value=("found", {})):
            result = self.run_case(instance, directory)
        self.assertEqual(endpoint.requests[1]["previous_response_id"], "resp-1")
        self.assertNotIn("previous_response_id", endpoint.requests[2])
        self.assertEqual(len(endpoint.requests[2]["input"]), 3)  # user + call + output
        self.assertTrue(result["fallback_events"][0]["used_stored_state"])

    def test_proactive_reasoning_disable_filters_even_unsolicited_encrypted_items(self):
        reasoning = {"type": "reasoning", "id": "rs-1", "summary": [], "encrypted_content": "unsolicited"}
        instance, endpoint = runner(response(reasoning, call_item()), response(text_item("Alice")), replay_reasoning=False)
        instance.request_options["include"] = ["reasoning.encrypted_content", "message.output_text.logprobs"]
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(responses.tools, "execute_tool", return_value=("found", {})):
            result = self.run_case(instance, directory)
        self.assertFalse(any(item.get("type") == "reasoning" for item in endpoint.requests[1]["input"]))
        self.assertEqual(endpoint.requests[0]["include"], ["message.output_text.logprobs"])
        self.assertTrue(result["reasoning_state_dropped"])
        self.assertEqual(result["fallback_events"], [])

    def test_invalid_encryption_without_any_state_is_not_pointlessly_retried(self):
        instance, endpoint = runner(encrypted_error())
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                self.run_case(instance, directory)
        self.assertEqual(len(endpoint.requests), 1)

    def test_error_classifier_is_specific_to_encryption_bad_requests(self):
        self.assertTrue(responses._invalid_encrypted_content(encrypted_error()))
        error = encrypted_error()
        error.status_code = 500
        self.assertFalse(responses._invalid_encrypted_content(error))
        error.status_code = 403
        self.assertFalse(responses._invalid_encrypted_content(error))

    def test_all_parallel_call_ids_get_results_even_with_budget_skips(self):
        instance, endpoint = runner(response(call_item("first"), call_item("second")),
                                    response(text_item("Alice")), max_tool_calls=1)
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(responses.tools, "execute_tool", return_value=("found", {})) as execute:
            result = self.run_case(instance, directory)
        self.assertEqual(execute.call_count, 1)
        outputs = [item for item in endpoint.requests[1]["input"] if item.get("type") == "function_call_output"]
        self.assertEqual([item["call_id"] for item in outputs], ["first", "second"])
        self.assertEqual(endpoint.requests[1]["tool_choice"], "none")
        self.assertFalse(endpoint.requests[0]["parallel_tool_calls"])
        self.assertEqual(result["tool_call_count"], 1)

    def test_invalid_arguments_are_replied_to_with_original_call_id(self):
        instance, endpoint = runner(response(call_item(arguments="not-json")), response(text_item("Alice")))
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(responses.tools, "execute_tool") as execute:
            self.run_case(instance, directory)
        execute.assert_not_called()
        output = endpoint.requests[1]["input"][-1]
        self.assertEqual(output["call_id"], "call-1")
        self.assertIn("Tool argument error", output["output"])

    def test_image_tool_output_is_multimodal_next_turn(self):
        instance, endpoint = runner(response(call_item(name="read_url", arguments='{"resource_id":"image_12345678"}')),
                                    response(text_item("Alice")))
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(responses.tools, "execute_tool", return_value=("image loaded", {"img_1": "https://example.com/image.png"})):
            self.run_case(instance, directory)
        image_msg = endpoint.requests[1]["input"][-1]
        self.assertEqual(image_msg["content"][1]["type"], "input_image")
        self.assertEqual(image_msg["content"][1]["image_url"], "https://example.com/image.png")

    def test_final_turn_disables_tools(self):
        instance, endpoint = runner(response(text_item("Alice")), max_turns=1)
        with tempfile.TemporaryDirectory() as directory:
            self.run_case(instance, directory)
        self.assertEqual(endpoint.requests[0]["tool_choice"], "none")
        self.assertIn(responses.FINALIZATION_PROMPT, endpoint.requests[0]["instructions"])

    def test_zero_tool_budget_disables_tools_from_first_request(self):
        instance, endpoint = runner(response(text_item("Alice")), max_tool_calls=0)
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(responses.tools, "execute_tool") as execute:
            result = self.run_case(instance, directory)
        execute.assert_not_called()
        self.assertEqual(endpoint.requests[0]["tool_choice"], "none")
        self.assertEqual(result["tool_call_count"], 0)

    def test_commentary_and_textual_xml_do_not_finish_the_case(self):
        for intermediate in (text_item("I will investigate", "commentary"),
                             text_item('<thinking>plan</thinking><tool_call>{}</tool_call>')):
            instance, endpoint = runner(response(intermediate), response(text_item("Alice", "final_answer")))
            with tempfile.TemporaryDirectory() as directory:
                result = self.run_case(instance, directory)
            self.assertEqual(result["final_answer"], "Alice")
            self.assertEqual(len(endpoint.requests), 2)

    def test_failures_are_not_marked_completed(self):
        examples = [
            (response(text_item("partial Alice"), status="incomplete", incomplete_details={"reason": "max_output_tokens"}), "incomplete_response"),
            (response({"type": "reasoning", "summary": [{"type": "summary_text", "text": "Alice"}]}), "empty_response"),
            (response({"type": "message", "content": [{"type": "refusal", "refusal": "cannot answer"}]}), "refusal"),
            (response({"type": "function_call", "id": "not-call-id", "name": "t2t_search", "arguments": "{}"}), "invalid_tool_call_id"),
        ]
        for raw, kind in examples:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                instance, _ = runner(raw)
                with self.assertRaises(RuntimeError):
                    self.run_case(instance, directory)
                saved = json.loads((Path(directory) / "failures/example_failure.json").read_text())
                self.assertEqual(saved["failure_kind"], kind)
                self.assertEqual(run_infer._completed_case_ids(directory), set())

    def test_tool_exception_returns_a_matched_output_instead_of_abandoning_call(self):
        instance, endpoint = runner(response(call_item()), response(text_item("Alice")))
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(responses.tools, "execute_tool", side_effect=RuntimeError("reader failed")):
            self.run_case(instance, directory)
        self.assertIn("reader failed", endpoint.requests[1]["input"][-1]["output"])

    def test_tools_use_flat_responses_schema_and_honor_ablation(self):
        definitions = responses.get_tool_definitions(("t2i_search",))
        self.assertNotIn("t2i_search", [item["name"] for item in definitions])
        self.assertTrue(all(item["strict"] is False for item in definitions))
        self.assertTrue(all("function" not in item for item in definitions))
        reader = next(item for item in definitions if item["name"] == "read_url")
        self.assertIn("disabled", reader["description"])

    def test_redacts_inline_images_on_disk(self):
        serialized = responses._serialise({"type": "input_image", "image_url": "data:image/png;base64,abc"}, {})
        self.assertNotIn("base64,abc", json.dumps(serialized))


class ResponsesEntrypointTests(unittest.TestCase):
    def parse(self, *extra):
        parser = run_infer._build_arg_parser(responses_api=True)
        responses.add_responses_arguments(parser)
        args = parser.parse_args(["--data-path", "fake.parquet", "--output-dir", "fake-out", *extra])
        responses.validate_responses_arguments(parser, args)
        return args

    def test_defaults_do_not_inject_qwen_sampling_parameters(self):
        instance = responses.ResponsesRunner.from_args(self.parse("--model", "gpt-test"))
        self.assertNotIn("temperature", instance.request_options)
        self.assertNotIn("top_p", instance.request_options)
        self.assertEqual(instance.request_options["include"], ["reasoning.encrypted_content"])
        self.assertFalse(instance.request_options["store"])
        self.assertEqual(instance.state_mode, "replay")
        self.assertEqual(instance.encrypted_reasoning_fallback, "drop")

    def test_cli_can_disable_reasoning_replay_or_lossy_fallback(self):
        instance = responses.ResponsesRunner.from_args(self.parse("--model", "gpt-test", "--no-encrypted-reasoning"))
        self.assertFalse(instance.replay_reasoning)
        self.assertNotIn("include", instance.request_options)
        instance = responses.ResponsesRunner.from_args(self.parse("--model", "gpt-test", "--encrypted-reasoning-fallback", "fail"))
        self.assertEqual(instance.encrypted_reasoning_fallback, "fail")

    def test_alias_uses_azure_config_and_explicit_cli_token_budget(self):
        from synthesis.model_worker import LLM_WORKER
        with mock.patch.object(LLM_WORKER, "get_model", return_value={
            "served_model": "served-gpt", "client_type": "azure_openai", "api_key": "secret-key",
            "azure_endpoint": "https://internal.example", "api_version": "test-version",
            "sampling_params": {"max_output_tokens": 9000, "reasoning_effort": "high"},
        }):
            args = self.parse("--model-alias", "registered", "--max-tokens", "4096")
            instance = responses.ResponsesRunner.from_args(args)
        self.assertEqual(instance.client_type, "azure_openai")
        self.assertEqual(instance.request_options["max_output_tokens"], 4096)
        self.assertEqual(instance.request_options["reasoning"], {"effort": "high"})
        with mock.patch.object(run_infer, "_git_output", return_value=""):
            snapshot = run_infer._run_configuration_snapshot(args, [], instance)
        self.assertEqual(snapshot["api_mode"], "responses")
        self.assertEqual(snapshot["inference"]["request_options"]["max_output_tokens"], 4096)
        self.assertNotIn("secret-key", json.dumps(snapshot))

    def test_invalid_state_or_qwen_parameter_rejected(self):
        for extra in (("--top-k", "20"), ("--responses-state-mode", "previous-response-id", "--responses-store", "false")):
            with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
                self.parse("--model", "gpt-test", *extra)

    def test_new_entrypoint_selects_responses(self):
        with mock.patch.object(run_infer_responses, "_shared_main", return_value=0) as shared:
            self.assertEqual(run_infer_responses.main(["--help"]), 0)
        self.assertTrue(shared.call_args.kwargs["responses_api"])

    def test_batch_entrypoint_saves_config_and_resumes_native_results(self):
        import pandas as pd
        instance, endpoint = runner(response(text_item("Alice")), max_turns=3)
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data.parquet"
            pd.DataFrame([{"id": "example", "question": "Who is it?"}]).to_parquet(data)
            output = Path(directory) / "out"
            argv = ["--model", "gpt-test", "--data-path", str(data), "--output-dir", str(output),
                    "--max-turns", "3", "--max-tokens", "4096", "--retry-monitor-interval", "0"]
            with mock.patch.object(responses.ResponsesRunner, "from_args", return_value=instance), mock.patch.object(instance, "load"), mock.patch.object(run_infer, "_git_output", return_value=""):
                self.assertEqual(run_infer_responses.main(argv), 0)
                self.assertEqual(run_infer_responses.main(argv), 0)
            self.assertEqual(len(endpoint.requests), 1)
            snapshot = json.loads((output / "run_config.json").read_text())
            self.assertEqual(snapshot["api_mode"], "responses")
            self.assertIn("--max-tokens 4096", snapshot["command"]["replay_command"])
            self.assertTrue(snapshot["command"]["entrypoint"].endswith("run_infer_responses.py"))
            self.assertEqual(len((output / "run_invocations.jsonl").read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
