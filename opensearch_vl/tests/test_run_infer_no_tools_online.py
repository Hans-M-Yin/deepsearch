"""Alias-backed no-tools CLI coverage; all model/tool calls remain offline."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import pandas as pd
from PIL import Image

import run_infer_no_tools_online as no_tools
import run_infer_react_online as online
from opensearch_infer import no_tools_pipeline, no_tools_prompts, pipeline
from synthesis.model_worker import LLM_WORKER
from synthesis.retry_monitor import get_monitor


def chat_result(text='Alice'):
    return {'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}


def fake_client(endpoint):
    return SimpleNamespace(chat=SimpleNamespace(completions=endpoint))


class OnlineNoToolsTests(unittest.TestCase):
    def alias(self, azure=False):
        result = {'served_model': 'test-model', 'api_key': 'alias-private-key',
                  'client_type': 'azure_openai' if azure else 'openai',
                  'sampling_params': {'max_tokens': 999, 'temperature': 0.7,
                                      'thinking': {'include_thoughts': True}}}
        if azure:
            result.update(azure_endpoint='https://azure.example', api_version='test-version')
        else:
            result['base_url'] = 'https://gateway.example/v1'
        return result

    def argv(self, source, output, *extra):
        return ['--model-alias', 'test-alias', '--data-path', str(source),
                '--output-dir', str(output), '--dataset', 'test',
                '--retry-monitor-interval', '0', *extra]

    def test_both_alias_routes_prompt_config_sampling_and_resume(self):
        for azure in (False, True):
            with self.subTest(azure=azure), tempfile.TemporaryDirectory() as directory:
                source, output = Path(directory) / 'data.parquet', Path(directory) / 'out'
                pd.DataFrame([{'id': 'plain', 'question': 'Who?'}]).to_parquet(source)
                endpoint = mock.Mock()
                endpoint.create.return_value = chat_result()
                constructor = 'openai.AzureOpenAI' if azure else 'openai.OpenAI'
                argv = self.argv(source, output, '--parallel-workers', '3',
                                 '--temperature', '0', '--max-tokens', '123', '--top-p', '1',
                                 '--top-k', '0', '--repetition-penalty', '1', '--presence-penalty', '0')
                with mock.patch.object(LLM_WORKER, 'get_model', return_value=self.alias(azure)), \
                        mock.patch(constructor, return_value=fake_client(endpoint)) as build, \
                        mock.patch.object(online, '_git_output', return_value=''), \
                        mock.patch.object(pipeline.tools, 'execute_tool', side_effect=AssertionError('no tools')) as execute:
                    self.assertEqual(no_tools.main(argv), 0)
                    self.assertEqual(no_tools.main(argv), 0)
                execute.assert_not_called()
                endpoint.create.assert_called_once()
                payload = endpoint.create.call_args.kwargs
                self.assertEqual(payload['model'], 'test-model')
                self.assertEqual(payload['messages'][0], {'role': 'system', 'content': no_tools_prompts.SYSTEM_PROMPT})
                self.assertEqual(no_tools_pipeline.SYSTEM_PROMPT, no_tools_prompts.SYSTEM_PROMPT)
                self.assertEqual(payload['max_tokens'], 123)
                self.assertEqual(payload['temperature'], 0)
                self.assertEqual(payload['top_p'], 1)
                self.assertEqual(payload['presence_penalty'], 0)
                self.assertEqual(payload['extra_body'], {'thinking': {'include_thoughts': True},
                                                        'top_k': 0, 'repetition_penalty': 1})
                self.assertNotIn('tools', payload)
                self.assertNotIn('tool_choice', payload)
                self.assertNotIn('chat_template_kwargs', payload['extra_body'])
                route = build.call_args.kwargs
                self.assertEqual(route['api_key'], 'alias-private-key')
                self.assertEqual(route['timeout'], 600)
                saved = json.loads((output / 'plain_trajectory.json').read_text())
                self.assertEqual(saved['mode'], 'no_tools')
                self.assertEqual(saved['status'], 'completed')
                self.assertEqual(saved['final_response_text'], 'Alice')
                self.assertEqual(len(saved['turns']), 1)
                self.assertEqual(online._completed_case_ids(str(output)), {'plain'})
                snapshot = json.loads((output / 'run_config.json').read_text())
                self.assertEqual(snapshot['mode'], 'no_tools')
                self.assertEqual(snapshot['react_mode'], 'no_tools')
                self.assertEqual(snapshot['agent_limits']['max_turns'], 1)
                self.assertEqual(snapshot['agent_limits']['max_tool_calls'], 0)
                self.assertEqual(snapshot['online_api']['model_alias'], 'test-alias')
                self.assertEqual(Path(snapshot['command']['entrypoint']).name, 'run_infer_no_tools_online.py')
                self.assertNotIn('alias-private-key', json.dumps(snapshot))
                self.assertEqual(len((output / 'run_invocations.jsonl').read_text().splitlines()), 2)

    def test_category_and_start_end_selection_and_optional_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'data.parquet', Path(directory) / 'out'
            pd.DataFrame([{'id': 'a', 'category': 'target', 'question': 'first'},
                          {'id': 'excluded', 'category': 'other', 'question': 'excluded'},
                          {'id': 'b', 'category': 'target', 'question': 'second'},
                          {'id': 'c', 'category': 'target', 'question': 'third'}]).to_parquet(source)
            endpoint = mock.Mock()
            endpoint.create.return_value = chat_result()
            with mock.patch.object(LLM_WORKER, 'get_model', return_value=self.alias()), \
                    mock.patch('openai.OpenAI', return_value=fake_client(endpoint)), \
                    mock.patch.object(online, '_git_output', return_value=''):
                self.assertEqual(no_tools.main(self.argv(source, output, '--category', 'target',
                                                        '--start', '1', '--end', '2')), 0)
            endpoint.create.assert_called_once()
            payload = endpoint.create.call_args.kwargs
            self.assertIn('second', json.dumps(payload['messages']))
            for name in ('max_tokens', 'temperature', 'top_p', 'top_k', 'presence_penalty', 'repetition_penalty'):
                self.assertNotIn(name, payload)
                self.assertNotIn(name, payload['extra_body'])
            self.assertEqual([p.name for p in output.glob('*_trajectory.json')], ['b_trajectory.json'])

    def test_two_images_dynamic_resolution_in_one_call(self):
        image = io.BytesIO()
        Image.new('RGB', (3200, 2000), (20, 40, 60)).save(image, format='PNG')
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'data.parquet', Path(directory) / 'out'
            pd.DataFrame([{'id': 'images', 'question': 'Compare the pictures.',
                           'images': [{'bytes': image.getvalue()}, {'bytes': image.getvalue()}]}]).to_parquet(source)
            endpoint = mock.Mock()
            endpoint.create.return_value = chat_result('<answer>Alice</answer>')
            with mock.patch.object(LLM_WORKER, 'get_model', return_value=self.alias()), \
                    mock.patch('openai.OpenAI', return_value=fake_client(endpoint)), \
                    mock.patch.object(online, '_git_output', return_value=''), \
                    mock.patch.object(pipeline.tools, 'execute_tool', side_effect=AssertionError('no tools')):
                self.assertEqual(no_tools.main(self.argv(source, output, '--dynamic-resolution', '--limit', '1')), 0)
            endpoint.create.assert_called_once()
            parts = endpoint.create.call_args.kwargs['messages'][1]['content']
            images = [p['image_url']['url'] for p in parts if p.get('type') == 'image_url']
            self.assertEqual(len(images), 2)
            for encoded in images:
                decoded = base64.b64decode(encoded.split(',', 1)[1])
                with Image.open(io.BytesIO(decoded)) as loaded:
                    self.assertLessEqual(max(loaded.size), 1600)
            saved = (output / 'images_trajectory.json').read_text()
            self.assertNotIn(encoded, saved)

    def test_request_failure_is_saved_and_retryable_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'data.parquet', Path(directory) / 'out'
            pd.DataFrame([{'id': 'retry', 'question': 'Who?'}]).to_parquet(source)
            endpoint = mock.Mock()
            endpoint.create.side_effect = [RuntimeError('backend unavailable'), chat_result()]
            argv = self.argv(source, output, '--api-max-retries', '1', '--api-timeout', '12',
                             '--retry-monitor-interval', '0.01')
            with mock.patch.object(LLM_WORKER, 'get_model', return_value=self.alias()), \
                    mock.patch('openai.OpenAI', return_value=fake_client(endpoint)) as build, \
                    mock.patch.object(online, '_git_output', return_value=''):
                self.assertEqual(no_tools.main(argv), 0)
                saved = json.loads((output / 'failures/retry_failure.json').read_text())
                self.assertEqual(saved['failure_kind'], 'inference_error')
                self.assertEqual(saved['max_turns'], 1)
                self.assertEqual(saved['max_tool_calls'], 0)
                self.assertEqual(online._completed_case_ids(str(output)), set())
                self.assertIsNone(get_monitor())
                self.assertEqual(no_tools.main(argv), 0)
            self.assertEqual(build.call_args.kwargs['timeout'], 12)
            self.assertEqual(endpoint.create.call_count, 2)
            self.assertEqual(online._completed_case_ids(str(output)), {'retry'})
            self.assertIsNone(get_monitor())

    def test_tool_and_no_tools_output_directories_cannot_be_mixed(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'data.parquet', Path(directory) / 'out'
            pd.DataFrame([{'id': 'sample', 'question': 'Who?'}]).to_parquet(source)
            endpoint = mock.Mock()
            endpoint.create.return_value = chat_result()
            argv = self.argv(source, output)
            with mock.patch.object(LLM_WORKER, 'get_model', return_value=self.alias()), \
                    mock.patch('openai.OpenAI', return_value=fake_client(endpoint)), \
                    mock.patch.object(online, '_git_output', return_value=''):
                self.assertEqual(no_tools.main(argv), 0)
                with self.assertRaises(SystemExit) as failure:
                    online.main(argv)
                self.assertEqual(failure.exception.code, 2)
                snapshot = json.loads((output / 'run_config.json').read_text())
                snapshot.update(mode='tools', react_mode='text_tool_call')
                (output / 'run_config.json').write_text(json.dumps(snapshot))
                with self.assertRaises(SystemExit) as failure:
                    no_tools.main(argv)
                self.assertEqual(failure.exception.code, 2)
            endpoint.create.assert_called_once()


if __name__ == '__main__':
    unittest.main()
