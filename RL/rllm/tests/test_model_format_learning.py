"""Exercise the real agent/workflow/batch code with CPU tensors and fake services."""

import ast
import asyncio
from collections import defaultdict
from enum import Enum
import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
from typing import Any, Optional
import unittest
from unittest.mock import patch
import uuid

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "vision_deepresearch_async_workflow/deepresearch_workflow.py"
AGENT = ROOT / "vision_deepresearch_async_workflow/deepresearch_agent.py"
ENGINE = ROOT / "rllm/engine/agent_workflow_engine.py"


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


protocol = load_module("format_learning_protocol", "rllm/utils/model_format.py")
agents = load_module("format_learning_dataclasses", "rllm/agents/agent.py")


def definition(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


termination = definition(ROOT / "rllm/workflows/workflow.py", "TerminationReason", {"Enum": Enum})


class ModelOutput:
    pass


COMMON = dict(
    asyncio=asyncio, os=os, time=time, re=re, json=json,
    Any=Any, Optional=Optional, List=list, ModelOutput=ModelOutput,
    Step=agents.Step, Action=agents.Action, Episode=agents.Episode,
    Trajectory=agents.Trajectory, TerminationReason=termination,
    MODEL_ERROR_TYPES=protocol.MODEL_ERROR_TYPES,
    MODEL_FAILURE_REWARD=protocol.MODEL_FAILURE_REWARD,
    step_error_type=protocol.step_error_type, validate_response=protocol.validate_response,
    format_feedback=protocol.format_feedback, classify_tool_failure=protocol.classify_tool_failure,
    _CONSECUTIVE_ERROR_THRESHOLD=3, _ACCURACY_REWARD_WEIGHT=0.9,
    _FORMAT_REWARD_WEIGHT=0.1, _MAX_TURNS_TERMINAL_REWARD=0.0,
)
for name in ("_is_valid_format", "_format_reward_for_step", "_has_tool_error_observation",
             "_is_step_error", "_is_max_turns_termination", "_find_fatal_step_index",
             "_base_outcome_reward", "_task_system_prompt", "_extract_action_from_response",
             "_get_next_observation", "_map_termination_reason", "_build_rollout_markers"):
    definition(WORKFLOW, name, COMMON)


QUERY_SCHEMA = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
GOOD_TOOL = '<thinking>search</thinking><tool_call>{"name":"t2t_search","arguments":{"query":"q"}}</tool_call>'
GOOD_ANSWER = '<thinking>done</thinking><answer>correct</answer>'
BAD_COMMA = '<thinking>search</thinking><tool_call>{"name", "t2t_search","arguments":{"query":"q"}}</tool_call>'
MISSING_NAME = '<thinking>search</thinking><tool_call>{"name", "arguments":{"query":"q"}}</tool_call>'


class ProtocolTests(unittest.TestCase):
    def test_strict_structure_and_json(self):
        bad = [
            '<thinking>x</thinking>' + GOOD_ANSWER,
            '<thinking>x' + GOOD_ANSWER,
            '<answer>correct</answer>',
            GOOD_ANSWER + '</answer>',
            GOOD_ANSWER + '<answer>other</answer>',
            GOOD_TOOL + '<tool_response>invented</tool_response>',
            BAD_COMMA, MISSING_NAME,
            '<thinking>x</thinking><tool_call>{"arguments":{}}</tool_call>',
            '<thinking>x</thinking><tool_call>{"name":"","arguments":{}}</tool_call>',
            '<thinking>x</thinking><tool_call>{"name":"t2t_search","name":"read_url","arguments":{}}</tool_call>',
            '<thinking>x</thinking><tool_call>{"name":"t2t_search","arguments":{"query":NaN}}</tool_call>',
        ]
        for response in bad:
            with self.subTest(response=response):
                self.assertFalse(protocol.validate_response(response).valid)
        self.assertTrue(protocol.validate_response(GOOD_TOOL).valid)
        self.assertTrue(protocol.validate_response(GOOD_ANSWER).valid)

    def test_registry_and_argument_schema(self):
        schemas = {"t2t_search": QUERY_SCHEMA}
        for payload in ({"name": "unknown", "arguments": {}},
                        {"name": "t2t_search", "arguments": {}},
                        {"name": "t2t_search", "arguments": {"query": 123}}):
            response = '<thinking>x</thinking><tool_call>' + json.dumps(payload) + '</tool_call>'
            result = protocol.validate_response(response, schemas)
            self.assertEqual(result.error_type, "tool_argument_error")
        self.assertIn('colon after "name"', protocol.validate_response(BAD_COMMA).detail)
        response = '<thinking>x</thinking><tool_call>{"arguments":{}}</tool_call>'
        self.assertIn('"name"', protocol.validate_response(response).detail)

    def test_external_failure_does_not_reduce_format(self):
        step = agents.Step(model_response=GOOD_TOOL, observation="Tool execution error: TimeoutError",
                           info={"step_error": True, "error_type": "external_tool_error"})
        self.assertEqual(COMMON['_format_reward_for_step'](step), 1.0)
        step.info["error_type"] = "tool_argument_error"
        self.assertEqual(COMMON['_format_reward_for_step'](step), 0.0)


agent_namespace = dict(COMMON, sequence_mode=lambda: "legacy",
                       DEEPRESEARCH_SYSTEM_PROMPT="test", analyze_repetition_ngram=lambda _: False,
                       count_words=lambda content: len(content.split()),
                       ToolRuntimeContext=lambda **_: SimpleNamespace(image_registry={}),
                       add_sft_image_placeholders=lambda q, _: q,
                       _parse_tool_call_payload=json.loads)


class FakeAgent:
    run = definition(AGENT, "_run", agent_namespace)

    def __init__(self, responses, tool_error=""):
        self.responses = iter(responses)
        self.max_llm_calls = len(responses)
        self.rollout_engine = SimpleNamespace(continuous_tito=False)
        self.system_prompt = "test"
        self.tools = {"t2t_search": SimpleNamespace(json={"function": {"parameters": QUERY_SCHEMA}})}
        self.tool_error = tool_error
        self.dispatches = 0

    async def call_server(self, messages, **kwargs):
        return SimpleNamespace(text=next(self.responses))

    async def custom_call_tool(self, *args, **kwargs):
        self.dispatches += 1
        self._last_tool_error_detail = self.tool_error
        return "Tool execution error: " + self.tool_error if self.tool_error else "evidence"

    def _build_result(self, **kwargs):
        return kwargs

    def record_token_usage(self, *args, **kwargs): pass
    def _record_llm_latency(self, *args, **kwargs): pass
    def _log_step_error(self, *args, **kwargs): pass
    def _record_tool_latency(self, *args, **kwargs): pass
    def _print_tool_timing(self, *args, **kwargs): pass


async def no_credit(**kwargs):
    return {"bonus": 0.0}


async def no_redundancy(*args):
    return {"penalty": 0.0}


workflow_namespace = dict(COMMON, score_statement_partial_credit=no_credit,
                          score_search_redundancy=no_redundancy,
                          partial_credit_metrics=lambda _: {}, _search_redundancy_metrics=lambda _: {},
                          collect_credit_turns=lambda *args: (),
                          _evaluate_answer=lambda prediction, answer: prediction == answer)


class FakeWorkflow:
    run = definition(WORKFLOW, "run", workflow_namespace)
    _convert_result_to_episode = definition(WORKFLOW, "_convert_result_to_episode", workflow_namespace)

    def __init__(self, agent):
        self.agent = agent
        self.reward_function = None

    def reset(self, **kwargs): pass


class AgentAndRewardTests(unittest.TestCase):
    def episode(self, responses, tool_error=""):
        agent = FakeAgent(responses, tool_error)
        episode = asyncio.run(FakeWorkflow(agent).run({"question": "q", "answer": "correct"}, "case"))
        return episode, agent

    def test_first_turn_cascade_is_trainable_with_negative_reward(self):
        episode, agent = self.episode([BAD_COMMA] * 3)
        self.assertEqual(agent.dispatches, 0)
        self.assertEqual(episode.info["raw_termination"], "consecutive_step_errors")
        self.assertTrue(episode.info["is_model_failure"])
        self.assertEqual(episode.info["fatal_step_index"], 0)
        self.assertEqual(episode.metrics["masked"], 0)
        self.assertAlmostEqual(episode.trajectories[0].reward, -0.1)
        self.assertEqual(episode.info["format_error_rounds"], [1, 2, 3])
        feedback = episode.trajectories[0].steps[0].observation
        self.assertIn('colon after "name"', feedback)

    def test_invalid_thinking_is_retried_instead_of_accepted_as_answer(self):
        episode, _ = self.episode(['<answer>correct</answer>'] * 3)
        self.assertTrue(episode.info["is_model_failure"])
        self.assertFalse(episode.is_correct)
        self.assertEqual(episode.info["raw_termination"], "consecutive_step_errors")
        self.assertNotEqual(episode.info["prediction"], "your answer")

    def test_error_counter_resets_after_success_and_recovery_is_preserved(self):
        episode, agent = self.episode([BAD_COMMA, BAD_COMMA, GOOD_TOOL, BAD_COMMA, GOOD_ANSWER])
        self.assertEqual(agent.dispatches, 1)
        self.assertFalse(episode.info["is_model_failure"])
        self.assertTrue(episode.is_correct)
        self.assertAlmostEqual(episode.trajectories[0].reward, 0.9)
        self.assertEqual(episode.metrics["reward/format"], 0)

    def test_external_cascade_remains_protected(self):
        episode, agent = self.episode([GOOD_TOOL] * 3, "TimeoutError: remote service timed out")
        self.assertEqual(agent.dispatches, 3)
        self.assertTrue(episode.info["is_fatal"])
        self.assertFalse(episode.info["is_model_failure"])
        self.assertEqual(episode.metrics["masked"], 1)
        self.assertEqual(episode.metrics["errors/external_tool_error_rounds"], 3)
        self.assertEqual(episode.metrics["reward/format"], 1)
        self.assertTrue(all(COMMON['_format_reward_for_step'](step) == 1 for step in episode.trajectories[0].steps))

    def test_external_failure_followed_by_answer_keeps_format_reward(self):
        episode, _ = self.episode([GOOD_TOOL, GOOD_ANSWER], "TimeoutError")
        self.assertEqual(episode.metrics["reward/format"], 1)
        self.assertAlmostEqual(episode.trajectories[0].reward, 1)

    def test_mixed_error_cascade_retains_external_protection(self):
        episode, _ = self.episode([GOOD_TOOL, BAD_COMMA, BAD_COMMA], "TimeoutError")
        self.assertTrue(episode.info["is_fatal"])
        self.assertFalse(episode.info["is_model_failure"])
        self.assertEqual(episode.metrics["masked"], 1)

    def test_runtime_argument_failure_is_not_classified_as_service_outage(self):
        episode, _ = self.episode([GOOD_TOOL] * 3, "query is required for t2t_search")
        self.assertTrue(episode.info["is_model_failure"])
        self.assertEqual(episode.metrics["errors/tool_argument_error_rounds"], 3)
        self.assertEqual(episode.metrics["reward/format"], 0)

    def test_missing_name_never_dispatches_a_guessed_tool(self):
        response = '<thinking>search</thinking><tool_call>{"arguments":{"query":"q"}}</tool_call>'
        episode, agent = self.episode([response] * 3)
        self.assertTrue(episode.info["is_model_failure"])
        self.assertEqual(agent.dispatches, 0)
        self.assertIn('nonempty "name"', episode.trajectories[0].steps[0].observation)

    def test_single_invalid_answer_at_budget_cannot_extract_feedback_example(self):
        episode, _ = self.episode(['<answer>correct</answer>'])
        self.assertFalse(episode.is_correct)
        self.assertNotEqual(episode.info["raw_termination"], "answer")
        self.assertNotEqual(episode.info["prediction"], "your answer")


class SFTFailureClassificationTests(unittest.TestCase):
    def test_public_reader_failures_are_classified_without_backend_metadata(self):
        async def run_case(arguments, resource_exists):
            output = {'page_id': 'page_1', 'title': '', 'goal': 'g',
                      'content': 'Unable to read the requested page.'}
            namespace = dict(agent_namespace, SFTToolAdapter=type('Adapter', (), {}),
                             sft_api=SimpleNamespace(execute_tool_call=lambda **kwargs: SimpleNamespace(
                                 output=output, output_text=json.dumps(output), timing={}, new_images={})))
            dispatch = definition(AGENT, 'custom_call_tool', namespace)
            agent = SimpleNamespace(
                tools={'read_url': SimpleNamespace(uses_sft_dispatcher=True)},
                _sft_context=SimpleNamespace(resolve_resource_id=lambda _: object() if resource_exists else None),
                _last_tool_error_detail='', _current_round=1,
                _tool_dispatch_latency_samples_s=defaultdict(list), executor=None,
                _log_step_error=lambda **kwargs: None,
                _record_read_url_stage_latency=lambda *args: None,
                _record_t2t_search_trace=lambda *args: None,
            )
            # Executor scheduling itself is unchanged. Run the fake service
            # inline to keep this classification test independent of threads.
            loop = asyncio.get_running_loop()

            def inline_executor(executor, function):
                future = loop.create_future()
                future.set_result(function())
                return future

            with patch.object(loop, 'run_in_executor', inline_executor):
                await dispatch(agent, 'read_url', arguments)
            return protocol.classify_tool_failure(agent._last_tool_error_detail)

        self.assertEqual(asyncio.run(run_case({'resource_id': 'page_1'}, True)), 'external_tool_error')
        self.assertEqual(asyncio.run(run_case({'resource_id': 'invented'}, False)), 'tool_argument_error')
        self.assertEqual(asyncio.run(run_case({}, False)), 'tool_argument_error')


class FakeDataProto:
    @classmethod
    def from_dict(cls, tensors, non_tensors, meta_info=None):
        return SimpleNamespace(batch=tensors, non_tensor_batch=non_tensors, meta_info=meta_info or {})


def pad_sequence(tensor, length, value, left_pad=False):
    difference = max(0, length - tensor.shape[-1])
    return torch.nn.functional.pad(tensor, (difference, 0) if left_pad else (0, difference), value=value)


engine_namespace = dict(COMMON, torch=torch, np=np, uuid=uuid, defaultdict=defaultdict,
                        logger=logging.getLogger(__name__), sequence_mode=lambda: "legacy",
                        fatal_requires_protection=protocol.fatal_requires_protection,
                        event=lambda *args, **kwargs: None, count_token=lambda *args: 0,
                        abort_on_missing_payload=lambda: False,
                        _expected_image_token_count=lambda *args: 0,
                        _validate_multimodal_alignment=lambda **kwargs: None,
                        build_credit_turn_token_mask=lambda *args: (None, True),
                        align_rollout_logprobs=lambda response, *args, **kwargs: (
                            torch.zeros_like(response, dtype=torch.float32), torch.zeros_like(response, dtype=torch.bool)))
definition(ENGINE, '_apply_fatal_step_mask', engine_namespace)


class FakeEngine:
    transform = definition(ENGINE, 'transform_results_for_verl', engine_namespace)

    def __init__(self, stepwise=False):
        filters = dict(enable=True, mask_max_prompt_length_exceeded=True,
                       mask_max_response_length_exceeded=True, mask_env_done=False,
                       mask_max_turns_exceeded=False, mask_timeout=True, mask_unknown=True, mask_error=True)
        self.config = SimpleNamespace(data=SimpleNamespace(max_prompt_length=8, max_response_length=32),
                                      rllm=SimpleNamespace(stepwise_advantage=SimpleNamespace(enable=stepwise),
                                                           compact_filtering=SimpleNamespace(**filters)))
        self.rollout_engine = SimpleNamespace(processor=None, tokenizer=SimpleNamespace(pad_token_id=0),
                                             continuous_tito=False, chat_parser=self)

    def tokenize_and_mask_cumulative(self, messages):
        count = sum(message['role'] == 'assistant' for message in messages)
        mask = torch.tensor(([1, 1, 0] * count)[:-1])
        return torch.tensor([10, 11]), torch.arange(20, 20 + len(mask)), mask

    def tokenize_and_mask_cumulative_multimodal(self, messages, **kwargs):
        return (*self.tokenize_and_mask_cumulative(messages), {})

    def tokenize_and_mask(self, messages):
        return torch.tensor([10, 11]), torch.tensor([20, 21]), torch.tensor([1, 1])


class BatchLearningTests(unittest.TestCase):
    def transform(self, episodes, mode="legacy", stepwise=False):
        fake_verl = ModuleType('verl')
        fake_verl.DataProto = FakeDataProto
        fake_functional = ModuleType('verl.utils.torch_functional')
        fake_functional.pad_sequence_to_length = pad_sequence
        with patch.dict(sys.modules, {'verl': fake_verl, 'verl.utils.torch_functional': fake_functional}), \
             patch.dict(engine_namespace, sequence_mode=lambda: mode):
            return FakeEngine(stepwise).transform(episodes, np.array(['group'] * len(episodes)))

    def test_model_failure_tokens_survive_all_sequence_modes_and_filtering(self):
        for mode, stepwise in [('legacy', False), ('cumulative', False), ('stepwise', True)]:
            with self.subTest(mode=mode):
                episode = asyncio.run(FakeWorkflow(FakeAgent([BAD_COMMA] * 3)).run({'question': 'q'}, 'bad'))
                batch = self.transform([episode], mode, stepwise)
                self.assertTrue(batch.non_tensor_batch['is_valid'].all())
                self.assertTrue(batch.non_tensor_batch['is_model_failure'].all())
                self.assertEqual(batch.batch['response_mask'].sum().item(), 6)

    def test_external_failure_mask_and_filtering_unchanged(self):
        episode = asyncio.run(FakeWorkflow(FakeAgent([GOOD_TOOL] * 3, 'TimeoutError')).run({'question': 'q'}, 'external'))
        batch = self.transform([episode], 'cumulative')
        self.assertFalse(batch.non_tensor_batch['is_valid'].any())
        self.assertEqual(batch.batch['response_mask'].sum().item(), 0)

    def test_rloo_and_grpo_keep_negative_model_failure_advantages(self):
        failure = asyncio.run(FakeWorkflow(FakeAgent([BAD_COMMA] * 3)).run({'question': 'q'}, 'bad'))
        success = asyncio.run(FakeWorkflow(FakeAgent([GOOD_ANSWER])).run({'question': 'q', 'answer': 'correct'}, 'good'))
        batch = self.transform([failure, success], 'cumulative')
        for name in ('compute_rloo_outcome_advantage', 'compute_grpo_outcome_advantage'):
            with self.subTest(estimator=name):
                compute = definition(ROOT / 'verl/verl/trainer/ppo/core_algos.py', name,
                                     dict(torch=torch, np=np, defaultdict=defaultdict, Optional=Optional, AlgoConfig=Any))
                advantages, returns = compute(batch.batch['traj_rewards'], batch.batch['response_mask'], np.array(['q', 'q']))
                batch.batch.update(advantages=advantages, returns=returns)
                protocol.clamp_external_fatal_advantages(batch)
                self.assertTrue((advantages[0][batch.batch['response_mask'][0].bool()] < 0).all())
                self.assertTrue((advantages[1][batch.batch['response_mask'][1].bool()] > 0).all())

    def test_external_clamp_still_protects_prefix_and_legacy_batches(self):
        for metadata in ({'is_fatal': np.array([True, True]), 'is_model_failure': np.array([True, False])},
                         {'is_fatal': np.array([False, True])}):
            batch = SimpleNamespace(non_tensor_batch=metadata,
                                    batch={'advantages': torch.tensor([[-1.], [-1.]]), 'returns': torch.tensor([[-1.], [-1.]])})
            protocol.clamp_external_fatal_advantages(batch)
            self.assertEqual(batch.batch['advantages'].tolist(), [[-1.], [0.]])

    def test_advantage_diagnostics_and_saved_error_metadata(self):
        episode = asyncio.run(FakeWorkflow(FakeAgent([BAD_COMMA] * 3)).run({'question': 'q'}, 'bad'))
        batch = self.transform([episode], 'cumulative')
        batch.batch['advantages'] = -batch.batch['response_mask'].float()
        metrics = protocol.model_failure_advantage_metrics(batch)
        self.assertEqual(metrics['format_learning/model_failure_tokens'], 6)
        self.assertEqual(metrics['format_learning/model_failure_negative_token_fraction'], 1)
        logger_class = definition(ROOT / 'rllm/utils/episode_logger.py', 'EpisodeLogger',
                                  dict(Path=Path, json=json, Any=Any, Episode=agents.Episode,
                                       hashlib=__import__('hashlib'), random=__import__('random')))
        with tempfile.TemporaryDirectory() as directory:
            logger = logger_class(directory, subdirectory='')
            logger.log_episode(episode, 1)
            saved = json.loads((logger.get_step_dir(1) / logger.get_episode_filename(episode, 1)).read_text())
            self.assertTrue(saved['is_model_failure'])
            self.assertEqual(saved['fatal_step_index'], 0)
            self.assertEqual(saved['format_error_rounds'], [1, 2, 3])
            self.assertEqual(saved['trajectories'][0]['step_errors'][0]['error_type'], 'model_format_error')


if __name__ == '__main__':
    unittest.main()
