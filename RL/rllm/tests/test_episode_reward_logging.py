"""CPU-only regression tests without importing the GPU training stack."""
import ast
import hashlib
import json
from pathlib import Path
import random
import re
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_definition(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if getattr(n, "name", None) == name)
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


class RewardLoggingTests(unittest.TestCase):
    def test_prompt_example_is_valid_json(self):
        build = load_definition(
            ROOT / "vision_deepresearch_async_workflow/deepresearch_tools_async_executor.py",
            "build_sft_react_system_prompt",
            {"get_tool_definitions": lambda: [], "format_sft_qwen_tool_prompt": lambda _: ""},
        )
        prompt = build()
        self.assertNotIn("tool use here", prompt)
        examples = re.findall(r"<tool_call>(.*?)</tool_call>", prompt, re.S)
        self.assertEqual(len(examples), 2)
        for example in examples:
            payload = json.loads(example)
            self.assertEqual(payload["name"], "t2t_search")
            self.assertIsInstance(payload["arguments"]["query"], str)

    def test_serialized_reward_audit_and_allowlist(self):
        logger_class = load_definition(
            ROOT / "rllm/utils/episode_logger.py", "EpisodeLogger",
            dict(Path=Path, json=json, hashlib=hashlib, random=random,
                 Any=object, Episode=object),
        )
        with tempfile.TemporaryDirectory() as directory:
            logger = logger_class(directory, subdirectory="")
            for index, metadata in enumerate((
                {"judge_decided": True, "judgment": "yes", "exact_match": False},
                {"judge_decided": False, "fallback_reason": "judge_undecided_use_exact_match"},
                None,
            )):
                info = {"prediction": "alias", "answer": ["canonical", "alias"]}
                if metadata is not None:
                    info["reward_metadata"] = dict(metadata, api_key="DO_NOT_SAVE")
                episode = SimpleNamespace(id=str(index), task={"question": "q"},
                                          info=info, is_correct=True,
                                          termination_reason=None, trajectories=[])
                logger.log_episode(episode, 1)
                path = logger.get_step_dir(1) / logger.get_episode_filename(episode, 1)
                saved = json.loads(path.read_text())
                self.assertEqual(saved["prediction"], "alias")
                self.assertEqual(saved["answer"], ["canonical", "alias"])
                self.assertEqual(saved["reward_metadata"], metadata or {})
                self.assertNotIn("DO_NOT_SAVE", path.read_text())
                self.assertIsNone(saved["data_id"])
                self.assertIsNone(saved["reward_components"]["accuracy"])

    def test_component_scores_and_dataset_identity(self):
        logger_class = load_definition(
            ROOT / "rllm/utils/episode_logger.py", "EpisodeLogger",
            dict(Path=Path, json=json, hashlib=hashlib, random=random,
                 Any=object, Episode=object),
        )
        with tempfile.TemporaryDirectory() as directory:
            logger = logger_class(directory, subdirectory="")
            for mode, masked in (("shadow", 0), ("reward", 0), ("reward", 1)):
                applied = 0.1 if mode == "reward" and not masked else 0.0
                episode = SimpleNamespace(
                    id="rollout-random-id", task={"id": "dataset-42", "extra_info": {
                        "source_question_id": "question-42", "api_key": "DO_NOT_SAVE"}},
                    info={"statement_partial_credit": {"mode": mode},
                          "search_redundancy": {"mode": mode}},
                    metrics={"reward/total": 0.0 if masked else 0.9 + applied,
                             "reward/accuracy": 1.0, "reward/format": 0.0,
                             "reward/search_redundancy_penalty": 0.04 if mode == "reward" else 0.0,
                             "reward/search_redundancy_raw_penalty": 0.04,
                             "reward/statement_partial_credit_applied_bonus": applied,
                             "reward/statement_partial_credit_raw_bonus": 0.1,
                             "masked": masked},
                    is_correct=True, termination_reason=None, trajectories=[],
                )
                logger.log_episode(episode, 1)
                path = logger.get_step_dir(1) / logger.get_episode_filename(episode, 1)
                saved = json.loads(path.read_text())
                self.assertEqual(saved["data_id"], "dataset-42")
                self.assertEqual(saved["data_provenance"], {"source_question_id": "question-42"})
                scores = saved["reward_components"]
                self.assertEqual(scores["partial_reward"], applied)
                self.assertEqual(scores["partial_raw_bonus"], 0.1)
                self.assertEqual(scores["partial_mode"], mode)
                self.assertEqual(scores["redundancy_raw_penalty"], 0.04)
                if masked or mode == "shadow":
                    self.assertEqual(scores["redundancy_penalty"], 0.0)
                self.assertNotIn("DO_NOT_SAVE", path.read_text())


if __name__ == "__main__":
    unittest.main()
