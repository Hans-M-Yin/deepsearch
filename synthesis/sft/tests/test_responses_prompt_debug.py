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
        self.assertIn("uses_responses_tool_use_tips_directly=False", output)
        self.assertIn("instructions_source=responses_system_prompt", output)


if __name__ == "__main__":
    unittest.main()
