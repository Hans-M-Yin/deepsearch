"""Offline coverage for malformed tool calls and model correction feedback."""

import json
import tempfile
import unittest
from unittest import mock

from opensearch_vl.opensearch_infer import config, pipeline, tools
from opensearch_vl.tests.test_pipeline_max_turns import _FakeRunner


def response(text):
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


class ToolFormatFallbackTests(unittest.TestCase):
    def test_known_errors_preserve_query(self):
        cases = [
            ('{"name", "t2t_search", "arguments": {"query": "x"}}', "name_comma_to_colon"),
            ('{"name": "t2t_search", "arguments: {"query": "x"}}', "close_arguments_key_quote"),
            ('{"name": "t2t_search", "arguments": "x"}}', "search_string_to_query"),
        ]
        for raw, repair in cases:
            with self.subTest(raw=raw):
                diagnostics = {}
                parsed = tools.extract_tool_call(f"<tool_call>{raw}</tool_call>", diagnostics)
                self.assertEqual(json.loads(parsed), {"name": "t2t_search", "parameters": {"query": "x"}})
                self.assertIn(repair, diagnostics["repairs"])

    def test_valid_query_is_not_rewritten(self):
        query = 'Find {"name", "x"} and "arguments: in a document'
        raw = json.dumps({"name": "t2t_search", "arguments": {"query": query}})
        diagnostics = {}
        parsed = tools.extract_tool_call(f"<tool_call>{raw}</tool_call>", diagnostics)
        self.assertEqual(json.loads(parsed)["parameters"]["query"], query)
        self.assertEqual(diagnostics, {})

    def test_ambiguous_errors_are_not_guessed(self):
        for raw in ('{"name":"read_url","arguments": "x"}',
                    '{"name", "arguments": {"query": "x"}}',
                    '{"name", "arguments", {"query": "x"}}',
                    '{"name": "t2t_search", "arguments": {"query": }}',
                    '{"name": "t2t_search", "arguments": {}} malicious',
                    '{"name": "t2t_search", "arguments": {"query":"x"}'):
            with self.subTest(raw=raw):
                diagnostics = {}
                self.assertIsNone(tools.extract_tool_call(f"<tool_call>{raw}</tool_call>", diagnostics))
                self.assertIn("error", diagnostics)

    def test_unparseable_call_returns_error_and_model_can_answer(self):
        runner = _FakeRunner()
        runner.responses = [response('<tool_call>{"name":"t2t_search","arguments":{"query":}}</tool_call>'),
                            response('<answer>corrected answer</answer>')]
        with tempfile.TemporaryDirectory() as output_dir, mock.patch.object(config, "MAX_TURNS", 4), \
                mock.patch.object(pipeline.tools, "execute_tool") as execute:
            trajectory = pipeline.process_single_case(
                {"id": "case", "question": "test", "images": []}, runner, output_dir, 0,
            )
            execute.assert_not_called()
            feedback = runner.contents_seen[1][-1]["parts"][0]["text"]
            self.assertIn("<tool_response>\nError format\n</tool_response>", feedback)
            self.assertIn('"name": "t2t_search"', feedback)
            self.assertEqual(trajectory["turns"][0]["tool_output_type"], "error_format")
            self.assertEqual(trajectory["status"], "completed")
            self.assertEqual(trajectory["tool_call_count"], 0)

    def test_repaired_call_is_executed_and_recorded(self):
        runner = _FakeRunner()
        runner.responses[0] = response('<tool_call>{"name", "t2t_search", "arguments":{"query":"x"}}</tool_call>')
        with tempfile.TemporaryDirectory() as output_dir, mock.patch.object(config, "MAX_TURNS", 4), \
                mock.patch.object(pipeline.tools, "execute_tool", return_value=("evidence", {})) as execute:
            trajectory = pipeline.process_single_case(
                {"id": "case", "question": "test", "images": []}, runner, output_dir, 0,
            )
            execute.assert_called_once()
            self.assertEqual(trajectory["turns"][0]["tool_call_format"]["repairs"], ["name_comma_to_colon"])
            self.assertEqual(trajectory["tool_call_count"], 1)

    def test_repeated_format_errors_still_reach_turn_limit(self):
        runner = _FakeRunner()
        runner.responses = [response('<tool_call>broken</tool_call>')] * 2
        with tempfile.TemporaryDirectory() as output_dir, mock.patch.object(config, "MAX_TURNS", 2), \
                mock.patch.object(pipeline.tools, "execute_tool") as execute:
            with self.assertRaisesRegex(RuntimeError, "2-turn limit"):
                pipeline.process_single_case(
                    {"id": "case", "question": "test", "images": []}, runner, output_dir, 0,
                )
            execute.assert_not_called()
            self.assertEqual(len(runner.contents_seen), 2)


if __name__ == "__main__":
    unittest.main()
