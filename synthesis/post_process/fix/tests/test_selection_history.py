import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from synthesis.post_process.fix.selection import selection_history, select_visible


class SelectionHistoryTests(unittest.TestCase):
    def test_prefix_includes_search_intent_but_not_results_or_future(self):
        record = {
            "conversations": [
                {"from": "human", "value": "question"},
                {"from": "observation", "value": "earlier evidence"},
                {"from": "gpt", "value": "search intent and call"},
                {"from": "observation", "value": "old results SECRET"},
                {"from": "gpt", "value": "future answer SECRET"},
            ]
        }
        window = SimpleNamespace(call_index=2, question="question")
        history = selection_history(record, window)
        self.assertEqual(len(history), 3)
        self.assertEqual(history[-1]["content"], "search intent and call")
        self.assertNotIn("SECRET", str(history))
        candidate = SimpleNamespace(key="a", public={"image_id": "a"})
        model = Mock()
        model.ask.return_value = {"image_id": "a", "reasoning": "Check this lead."}
        select_visible(
            [candidate],
            set(),
            [],
            window,
            "source",
            model,
            SimpleNamespace(model="test"),
            trajectory_history=history,
        )
        self.assertEqual(model.ask.call_args.args[2]["trajectory_history"], history)

    def test_messages_schema_and_missing_optional_record(self):
        window = SimpleNamespace(call_index=0)
        self.assertEqual(selection_history(None, window), [])
        self.assertEqual(
            selection_history(
                {
                    "messages": [
                        {"role": "assistant", "content": "intent"},
                        {"role": "tool", "content": "future"},
                    ]
                },
                window,
            ),
            [{"role": "assistant", "content": "intent"}],
        )
