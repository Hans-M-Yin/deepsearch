import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from synthesis.post_process.fix.global_refine import refine_trajectory
from synthesis.post_process.fix.schema import Config
from synthesis.post_process.fix.trajectory_index import messages, content, call
from synthesis.post_process.fix.tests.test_pipeline import fixture
from synthesis.post_process.fix.tests.test_integration import FakeModel, FakeBackend
from synthesis.post_process.fix.pipeline import process_record
from synthesis.post_process.fix.storage import Cache


class GlobalRefineTests(unittest.TestCase):
    def test_search_marker_and_deferred_issue_remapping(self):
        model = Mock()

        def ask(alias, prompt, payload, paths):
            self.assertTrue(payload["trajectory"][2]["updated_search_results"])
            self.assertTrue(payload["affected_assistant_ids"])
            self.assertEqual(
                payload["inspection_state"],
                [{"assessment": {"established_facts": ["County identified"]}}],
            )
            self.assertEqual(
                payload["deferred_context_issues"],
                ["assistant_000002: stale Getty comparison"],
            )
            return {
                "severe_errors": [],
                "turns": [
                    {
                        "assistant_id": row["assistant_id"],
                        "issues": [],
                        "reasoning": None,
                    }
                    for row in payload["trajectory"]
                    if "assistant_id" in row
                ],
            }

        model.ask.side_effect = ask
        _, audit = refine_trajectory(
            self.record,
            self.source,
            self.root,
            model,
            Config(),
            search_context={
                "observation_index": 2,
                "continuation_map": {"8": 3},
                "deferred_issues": ["original_message_000008: stale Getty comparison"],
                "inspection_state": [
                    {"assessment": {"established_facts": ["County identified"]}}
                ],
            },
        )
        self.assertTrue(audit["accepted"])

    def test_global_editor_cannot_leave_process_leakage_unchanged(self):
        messages(self.record)[-1]["value"] = (
            "<thinking>In this abbreviated history, evidence is omitted.</thinking><answer>City</answer>"
        )
        model = Mock()

        def ask(alias, prompt, payload, paths):
            rows = [r for r in payload["trajectory"] if "assistant_id" in r]
            return {
                "severe_errors": [],
                "turns": [
                    {"assistant_id": r["assistant_id"], "issues": [], "reasoning": None}
                    for r in rows
                ],
            }

        model.ask.side_effect = ask
        with self.assertRaisesRegex(ValueError, "Editing-process leakage"):
            refine_trajectory(self.record, self.source, self.root, model, Config())
        self.assertEqual(model.ask.call_count, 2)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.record, self.source, self.target = fixture(self.root, True)

    def test_full_context_images_assistant_only_ids_and_protected_tools(self):
        original = copy.deepcopy(self.record)
        model = Mock()

        def ask(alias, prompt, payload, paths):
            self.assertGreaterEqual(len(paths), 2)
            rows = payload["trajectory"]
            self.assertEqual(len(rows), len(messages(original)))
            for row in rows:
                self.assertEqual("assistant_id" in row, row["role"] == "gpt")
            self.assertTrue(any("image_indices" in row for row in rows))
            return {
                "severe_errors": [],
                "turns": [
                    {
                        "assistant_id": row["assistant_id"],
                        "issues": ["wording"],
                        "reasoning": "I will check the source to resolve the remaining question.",
                    }
                    for row in rows
                    if "assistant_id" in row
                ],
            }

        model.ask.side_effect = ask
        updated, audit = refine_trajectory(
            self.record, self.source, self.root, model, Config()
        )
        self.assertTrue(audit["accepted"])
        self.assertEqual(self.record, original)
        for old, new in zip(messages(original), messages(updated)):
            if old["from"] != "gpt":
                self.assertEqual(old, new)
            else:
                self.assertEqual(call(content(old)), call(content(new)))
        self.assertEqual(
            content(messages(original)[-1]).split("<answer>")[-1],
            content(messages(updated)[-1]).split("<answer>")[-1],
        )

    def test_severe_error_marks_discard_and_skips_final_review(self):
        from synthesis.post_process.fix import global_refine, validate

        model = FakeModel()
        base = model.ask
        seen = []

        def ask(alias, prompt, payload, *args, **kwargs):
            seen.append(prompt)
            if prompt == global_refine.PROMPT:
                return {
                    "severe_errors": ["assistant_000002 substitutes the target entity"],
                    "turns": [],
                }
            return base(alias, prompt, payload, *args, **kwargs)

        model.ask = ask
        output, audit = process_record(
            self.record,
            self.root,
            self.root,
            Config(),
            FakeBackend(self.target),
            model,
            Cache(self.root),
        )
        self.assertTrue(audit["discard"])
        self.assertEqual(audit["status"], "quarantined")
        self.assertEqual(output, self.record)
        self.assertNotIn(validate.JUDGE, seen)
        self.assertTrue(audit["windows"][0]["global_refinement"]["severe_errors"])

    def test_missing_turn_assessments_fail_bounded(self):
        model = Mock()
        model.ask.return_value = {"severe_errors": [], "turns": []}
        with self.assertRaisesRegex(ValueError, "Every assistant"):
            refine_trajectory(self.record, self.source, self.root, model, Config())
        self.assertEqual(model.ask.call_count, 2)
