import unittest

from synthesis.sft.api_tools import (
    _conversation_messages_to_responses_input,
    _message_content_to_responses_content,
    _messages_to_responses_input,
)


class ResponsesHistoryContentTests(unittest.TestCase):
    def test_assistant_text_uses_output_text(self) -> None:
        content = _message_content_to_responses_content("Prior assistant reply", role="assistant")
        self.assertEqual(content, [{"type": "output_text", "text": "Prior assistant reply"}])

    def test_user_text_uses_input_text(self) -> None:
        content = _message_content_to_responses_content("User question", role="user")
        self.assertEqual(content, [{"type": "input_text", "text": "User question"}])

    def test_full_conversation_replay_uses_role_valid_text_types(self) -> None:
        replay = _conversation_messages_to_responses_input(
            [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "I will search."},
                {"role": "tool", "tool_call_id": "call-1", "content": "Tool result"},
            ]
        )
        self.assertEqual(replay[0]["content"][0]["type"], "input_text")
        self.assertEqual(replay[1]["content"][0]["type"], "output_text")
        self.assertEqual(replay[2]["type"], "function_call_output")

    def test_initial_responses_input_uses_role_valid_text_types(self) -> None:
        items = _messages_to_responses_input(
            [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
            ]
        )
        self.assertEqual(items[0]["content"][0]["type"], "input_text")
        self.assertEqual(items[1]["content"][0]["type"], "output_text")


if __name__ == "__main__":
    unittest.main()
