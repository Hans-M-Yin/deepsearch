"""Tests for the reserved final answer turn in the agent pipeline."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from unittest import mock

from opensearch_vl.opensearch_infer import config, pipeline


class _FakeRunner:
    display_name = "fake"

    def __init__(self) -> None:
        self.responses = [
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": (
                                        "<thinking>Need one search.</thinking>\n"
                                        "<tool_call>"
                                        '{"name":"text_search","parameters":{"query":"x"}}'
                                        "</tool_call>"
                                    )
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": (
                                        "<thinking>Enough evidence.</thinking>\n"
                                        "<answer>final answer</answer>"
                                    )
                                }
                            ]
                        }
                    }
                ]
            },
        ]
        self.contents_seen: list[list[dict[str, object]]] = []

    def infer(self, *, contents, system_instruction, cfg):
        self.contents_seen.append(copy.deepcopy(contents))
        return self.responses.pop(0)


class PipelineMaxTurnsTests(unittest.TestCase):
    def test_penultimate_tool_call_gets_finalization_prompt(self) -> None:
        runner = _FakeRunner()
        with tempfile.TemporaryDirectory() as output_dir:
            with (
                mock.patch.object(config, "MAX_TURNS", 2),
                mock.patch.object(
                    pipeline.tools,
                    "execute_tool",
                    side_effect=AssertionError("reserved turn must not execute tools"),
                ) as execute_tool,
            ):
                trajectory = pipeline.process_single_case(
                    row={"id": "case-1", "question": "What is the answer?", "images": []},
                    runner=runner,
                    output_dir=output_dir,
                    case_idx=0,
                )

            self.assertEqual(trajectory["status"], "completed")
            self.assertEqual(len(runner.contents_seen), 2)
            self.assertFalse(execute_tool.called)
            finalization_text = runner.contents_seen[1][-1]["parts"][0]["text"]
            self.assertIn("已达到搜索上限，不能再调用任何工具", finalization_text)
            self.assertIn("直接给出原问题的最终答案", finalization_text)
            self.assertIn("请直接根据历史搜索记录以及输入问题进行回答", finalization_text)
            self.assertIn("<thinking>...</thinking><answer>...</answer>", finalization_text)
            self.assertIn("完整回复必须且只能是", finalization_text)
            self.assertIn("严禁输出<tool_call>", finalization_text)
            self.assertIn("除这两个标签外，不得输出任何文本", finalization_text)
            self.assertEqual(
                trajectory["turns"][0]["tool_output_type"],
                "max_turns_finalization",
            )
            self.assertEqual(trajectory["final_response_text"].count("<answer>"), 1)

            with open(f"{output_dir}/case-1_trajectory.json", encoding="utf-8") as fh:
                saved = json.load(fh)
            self.assertEqual(saved["status"], "completed")


if __name__ == "__main__":
    unittest.main()
