"""Insertion-only editing contracts replace the former continuation-patch tests."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from synthesis.post_process.fix.context_patch import patch_context
from synthesis.post_process.fix.schema import Config, Plan
from synthesis.post_process.fix.trajectory_index import index_record, messages
from synthesis.post_process.fix.tests.test_pipeline import fixture


class ContextIdTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.record, _, _ = fixture(Path(self.tmp.name), True)
        self.window = index_record(self.record)[0]
        self.steps = [
            {
                "kind": "read",
                "candidate_key": "test",
                "before": "Inspect this lead.",
                "after": "The roof matches.",
            }
        ]
        self.plan = Plan("direct", [], [])

    def response(self, before="Inspect this lead."):
        return {
            "trace_patches": [
                {
                    "step_id": "inserted_step_000000",
                    "before": before,
                    "after": "The roof matches; check the caption next.",
                }
            ]
        }

    def run_editor(self, model):
        return patch_context(
            self.record, self.window, self.plan, self.steps, model, Config()
        )

    def test_full_read_only_context_and_position(self):
        original = copy.deepcopy(self.record)
        model = Mock()
        model.ask.return_value = self.response()
        self.assertEqual(self.run_editor(model), {})
        self.assertEqual(self.record, original)
        payload = model.ask.call_args.args[2]
        self.assertEqual(len(payload["original_trajectory"]), len(messages(original)))
        self.assertTrue(all(not m["editable"] for m in payload["original_trajectory"]))
        self.assertEqual(
            payload["insertion_position"]["before_message_id"],
            f"original_message_{self.window.end:06d}",
        )
        self.assertFalse(self.plan.metadata["transition_includes_final_check"])

    def test_complete_steps_required_and_original_patches_rejected(self):
        for bad in (
            {"trace_patches": []},
            {"patches": []},
            {
                "trace_patches": [
                    {"step_id": "inserted_step_000000", "before": "Incomplete"}
                ]
            },
        ):
            model = Mock()
            model.ask.side_effect = [bad, self.response()]
            self.assertEqual(self.run_editor(model), {})
            self.assertEqual(model.ask.call_count, 2)

    def test_process_voice_repaired_locally(self):
        model = Mock()
        model.ask.side_effect = [
            self.response("The trajectory investigation linked the painter."),
            self.response(),
        ]
        self.run_editor(model)
        self.assertEqual(model.ask.call_count, 2)
        self.assertNotIn("trajectory", self.steps[0]["before"])

    def test_persistent_failure_does_not_mutate_steps(self):
        before = copy.deepcopy(self.steps)
        model = Mock()
        model.ask.return_value = self.response("Return to the original trajectory.")
        with self.assertRaises(ValueError):
            self.run_editor(model)
        self.assertEqual(self.steps, before)
        self.assertEqual(model.ask.call_count, 2)

    def test_unsafe_not_retried(self):
        model = Mock()
        model.ask.return_value = {
            "unsafe": True,
            "reason": "Original continuation contradicts known identity",
        }
        with self.assertRaisesRegex(ValueError, "Unsafe insertion"):
            self.run_editor(model)
        self.assertEqual(model.ask.call_count, 1)

    def test_original_leakage_deferred_not_edited(self):
        messages(self.record)[-1]["value"] = (
            "<thinking>In this shortened record, identification was not verified.</thinking><answer>City</answer>"
        )
        model = Mock()
        model.ask.return_value = self.response()
        self.assertEqual(self.run_editor(model), {})
        self.assertIn(
            "shortened record", str(self.plan.metadata["deferred_context_issues"])
        )
        self.assertIn("shortened record", messages(self.record)[-1]["value"])

    def test_zero_insertions_still_checks_context(self):
        self.steps = []
        model = Mock()
        model.ask.return_value = {"trace_patches": []}
        self.assertEqual(self.run_editor(model), {})
        self.assertEqual(model.ask.call_count, 1)
