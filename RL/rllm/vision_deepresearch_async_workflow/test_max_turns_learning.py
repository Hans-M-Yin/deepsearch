"""Regression tests for learnable MAX_TURNS trajectories."""

from __future__ import annotations

from pathlib import Path
import unittest

from rllm.agents.agent import Step
from rllm.workflows.workflow import TerminationReason
from vision_deepresearch_async_workflow.deepresearch_workflow import (
    _MAX_TURNS_TERMINAL_REWARD,
    _base_outcome_reward,
    _find_fatal_step_index,
    _map_termination_reason,
)


class MaxTurnsLearningTests(unittest.TestCase):
    def test_max_turns_is_not_fatal_or_penalized(self) -> None:
        steps = [Step(model_response="<thinking>x</thinking><tool_call>{}</tool_call>")]

        fatal_index, fatal_reason = _find_fatal_step_index(
            steps, "answer not found after 40 rounds"
        )

        self.assertIsNone(fatal_index)
        self.assertEqual(fatal_reason, "")
        self.assertEqual(
            _map_termination_reason("answer not found after 40 rounds"),
            TerminationReason.MAX_TURNS_EXCEEDED,
        )
        self.assertEqual(_MAX_TURNS_TERMINAL_REWARD, 0.0)

    def test_base_reward_uses_accuracy_and_format_weights(self) -> None:
        self.assertEqual(_base_outcome_reward(1.0, 1.0), 1.0)
        self.assertEqual(_base_outcome_reward(1.0, 0.0), 0.9)
        self.assertEqual(_base_outcome_reward(0.0, 1.0), 0.1)
        self.assertEqual(_base_outcome_reward(0.0, 0.0), 0.0)

    def test_real_error_cascade_remains_fatal(self) -> None:
        steps = [
            Step(info={"step_error": True}),
            Step(info={"step_error": True}),
            Step(info={"step_error": True}),
        ]

        self.assertEqual(
            _find_fatal_step_index(steps, "answer not found after 40 rounds"),
            (0, "consecutive_errors"),
        )

    def test_formal_launcher_keeps_max_turns_rows(self) -> None:
        launcher = Path(__file__).with_name("run") / "qwen3-vl-8b-2node.sh"

        self.assertIn(
            "rllm.compact_filtering.mask_max_turns_exceeded=False",
            launcher.read_text(encoding="utf-8"),
        )


if __name__ == "__main__":
    unittest.main()
