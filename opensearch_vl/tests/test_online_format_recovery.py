"""Recover online tag errors without executing ambiguous calls or guessing answers."""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import run_infer_react_online as online
from opensearch_infer import config, pipeline, tools


def response(text, finish="stop"):
    return {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": finish}]}


class OnlineFormatRecoveryTests(unittest.TestCase):
    def run_case(self, fake, directory, **kwargs):
        return pipeline.process_single_case(
            {"id": "sample", "question": "Who?"}, fake, directory, 0,
            retain_format_errors=True, recover_format_errors=True, **kwargs)

    def test_complete_json_with_bad_closing_tag_executes_once(self):
        text = '<thinking>Search</thinking><tool_call>{"name":"t2t_search","arguments":{"query":"x"}}</tool_all>'
        self.assertIsNone(tools.extract_tool_call(text))
        fake = SimpleNamespace(display_name="fake", infer=mock.Mock(side_effect=[
            response(text), response('<thinking>Done</thinking><answer>Alice')]))
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(tools, "execute_tool", return_value=("evidence", {})) as execute:
            saved = self.run_case(fake, directory)
        execute.assert_called_once()
        self.assertEqual(json.loads(execute.call_args.args[0])["name"], "t2t_search")
        self.assertEqual(saved["final_response_text"], "<answer>Alice</answer>")
        self.assertTrue(saved["format_recovered"])
        self.assertFalse(saved["raw_format_valid"])
        self.assertEqual(saved["turns"][0]["response_text"], text)
        self.assertEqual(saved["format_recovery_events"][0]["kind"], "repair_tool_call")

    def test_unfinished_thinking_requests_a_new_answer(self):
        raw = '<thinking>I believe Alice, but I need to search first.'
        fake = SimpleNamespace(display_name="fake", infer=mock.Mock(side_effect=[response(raw), response('<answer>Bob</answer>')]))
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(tools, "execute_tool") as execute:
            saved = self.run_case(fake, directory)
        execute.assert_not_called()
        self.assertEqual(fake.infer.call_count, 2)
        self.assertEqual(saved["final_response_text"], '<answer>Bob</answer>')
        self.assertNotIn('Alice', saved["final_response_text"])
        self.assertIn('Close every tag', saved['turns'][0]['tool_output'])

    def test_repeated_unparseable_calls_stop_after_bounded_corrections(self):
        fake = SimpleNamespace(display_name="fake", infer=mock.Mock(return_value=response('<tool_call>{"name":')))
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(config, "MAX_TURNS", 10), mock.patch.object(tools, "execute_tool") as execute:
            saved = self.run_case(fake, directory, max_format_recovery_attempts=2)
            self.assertEqual(online._completed_case_ids(directory, retry_missing_answers=True), set())
        execute.assert_not_called()
        self.assertEqual(fake.infer.call_count, 3)
        self.assertEqual(saved['format_error']['kind'], 'format_recovery_exhausted')
        self.assertEqual(saved['status'], 'completed')
        self.assertEqual(saved['final_response_text'], '<answer></answer>')
        self.assertTrue(saved['answer_missing'])

    def test_known_tool_tags_are_not_treated_as_plain_answers(self):
        text = '<t2t_search>{"query":"x"}</t2t_search>'
        fake = SimpleNamespace(display_name="fake", infer=mock.Mock(side_effect=[response(text), response('<answer>Alice</answer>')]))
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(tools, "execute_tool", return_value=("evidence", {})) as execute:
            saved = self.run_case(fake, directory)
        execute.assert_called_once()
        self.assertEqual(saved['final_response_text'], '<answer>Alice</answer>')

    def test_truncated_answer_is_not_silently_promoted(self):
        fake = SimpleNamespace(display_name="fake", infer=mock.Mock(side_effect=[
            response('<answer>Alice might be', 'length'), response('<answer>Bob</answer>')]))
        with tempfile.TemporaryDirectory() as directory:
            saved = self.run_case(fake, directory)
        self.assertEqual(fake.infer.call_count, 2)
        self.assertEqual(saved['final_response_text'], '<answer>Bob</answer>')

    def test_plain_answer_is_normalized_and_earlier_guesses_excluded(self):
        fake = SimpleNamespace(display_name="fake", infer=mock.Mock(side_effect=[
            response('<thinking>Earlier guess Alice'), response('Bob')]))
        with tempfile.TemporaryDirectory() as directory:
            saved = self.run_case(fake, directory)
        self.assertEqual(saved['final_response_text'], '<answer>Bob</answer>')

    def test_api_failure_remains_a_failure(self):
        fake = SimpleNamespace(display_name="fake", infer=mock.Mock(return_value={
            'inference_error': {'message': 'HTTP 400'}, **response('<answer>Error</answer>')}))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                self.run_case(fake, directory)
            failed = json.loads((Path(directory) / 'failures/sample_failure.json').read_text())
        self.assertEqual(failed['failure_kind'], 'inference_error')


if __name__ == '__main__':
    unittest.main()
