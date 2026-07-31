import contextlib
import io
import unittest
from unittest.mock import patch

from synthesis.sft.api_tools import DEFAULT_SYSTEM_PROMPT, OpenAIToolAgent, OpenAIToolAgentConfig


class ResponsesPromptDebugTests(unittest.TestCase):
    def test_debug_line_reports_prompt_selection_conditions(self) -> None:
        config = OpenAIToolAgentConfig(
            model="test-model",
            api_mode="responses",
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            responses_prompt_public_reasoning=True,
            max_turns=1,
        )
        agent = OpenAIToolAgent(config)
        stderr = io.StringIO()
        with (
            contextlib.redirect_stderr(stderr),
            patch("synthesis.sft.api_tools.LLM_WORKER.responses_generate", side_effect=RuntimeError("stop")),
        ):
            with self.assertRaises(RuntimeError):
                agent._run_responses(prompt="test")

        output = stderr.getvalue()
        self.assertIn("responses_prompt_public_reasoning=True", output)
        self.assertIn("uses_default_system_prompt=True", output)
        self.assertIn("uses_responses_tool_use_tips_directly=True", output)
        self.assertIn("instructions_source=responses_system_prompt_plus_tool_use_tips", output)


if __name__ == "__main__":
    unittest.main()

class ResponsesInstructionsTests(unittest.TestCase):
    def test_default_responses_instructions_append_tool_use_tips(self) -> None:
        from synthesis.sft.api_tools import RESPONSES_TOOL_USE_TIPS, _build_responses_instructions

        instructions = _build_responses_instructions(DEFAULT_SYSTEM_PROMPT)
        self.assertIn(RESPONSES_TOOL_USE_TIPS, instructions)
