import tempfile
import unittest
from pathlib import Path
from random import Random

from synthesis.post_process.fix.tests.test_pipeline import fixture, candidates
from synthesis.post_process.fix.tests import test_pipeline
from synthesis.post_process.fix.trajectory_index import index_record
from synthesis.post_process.fix.schema import Config
from synthesis.post_process.fix.candidate_sets import build_plan
from synthesis.post_process.fix.selection import choose_checks
from synthesis.post_process.fix.eligibility import select_anchor
from synthesis.post_process.fix.validate import validate_structure


class Replies:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.payloads = []

    def ask(self, alias, prompt, payload, images):
        self.payloads.append(payload)
        return next(self.responses)


class DualReferenceTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.record, self.source, target = fixture(Path(tmp.name))
        self.window = index_record(self.record)[0]
        self.historical, self.pool = candidates(self.window, target)
        second = self.historical[1]
        second.verdict, second.category = "YES", "CONSISTENT"
        second.image_paths = [str(target)]
        second.evidence = ["Matching roof and window arrangement"]
        self.anchor = {
            "anchor_image_id": "image_0",
            "anchor_image_ids": ["image_0", "image_1"],
        }
        self.config = Config(replacement_counts=[3], blind_visual_review=False)

    def plan(self):
        plan = build_plan(
            self.window,
            self.historical,
            self.pool,
            self.anchor,
            {"eligible": False},
            self.config,
            Random(4),
        )
        plan.metadata["anchor_image_id"] = "image_0"
        return plan

    def test_preserve_two_replace_three(self):
        plan = self.plan()
        self.assertEqual(len(plan.first), 5)
        self.assertTrue(
            set(self.anchor["anchor_image_ids"])
            <= {c.public["image_id"] for c in plan.first}
        )
        self.assertEqual(plan.metadata["replacement_count"], 3)

    def test_later_original_page_id_is_not_renamed_by_new_card(self):
        from synthesis.post_process.fix.trajectory_index import messages

        plan = self.plan()
        page = next(c.public["source_page_id"] for c in plan.first if not c.historical)
        for message in messages(self.record)[self.window.end :]:
            message["value"] = message["value"].replace("page_0", page)
        plan.attempts = self.historical[:2]
        plan.positive_key = plan.attempts[-1].key
        # Reindex: this later page is not one of the old i2i cards.
        window = self.window
        updated, audit = test_pipeline.PipelineTests.assemble(
            self, self.record, window, plan
        )
        self.assertNotIn(page, audit["id_mapping"])
        self.assertEqual(
            validate_structure(self.record, updated, window, plan, audit), []
        )

    def test_selector_retries_internal_field_leakage(self):
        plan = self.plan()
        model = Replies(
            [
                {"image_id": "image_0", "reasoning": "Choose references_to_check."},
                {"image_id": "image_0", "reasoning": "Inspect the captioned painting."},
                {
                    "image_id": "image_1",
                    "reasoning": "Compare the other captioned reproduction.",
                },
            ]
        )
        choose_checks(plan, self.window, self.source, model, self.config, self.record)
        self.assertEqual(len(model.payloads), 3)
        self.assertEqual(plan.metadata["wrong_count"], 0)

    def test_missing_second_image_rejected(self):
        self.historical[1].image_paths = []
        with self.assertRaises(ValueError):
            self.plan()

    def test_both_matches_before_continuation(self):
        plan = self.plan()
        model = Replies(
            [
                {"image_id": rid, "reasoning": "Compare this captioned lead."}
                for rid in ["image_1", "image_0"]
            ]
        )
        choose_checks(plan, self.window, self.source, model, self.config, self.record)
        self.assertEqual(
            [c.public["image_id"] for c in plan.attempts], ["image_1", "image_0"]
        )
        self.assertEqual(plan.branch, "direct")
        self.assertEqual(plan.metadata["wrong_count"], 0)
        self.assertEqual(len(model.payloads[1]["previous_observations"]), 1)
        updated, audit = test_pipeline.PipelineTests.assemble(
            self, self.record, self.window, plan
        )
        self.assertEqual(
            validate_structure(self.record, updated, self.window, plan, audit), []
        )
        plan.attempts = plan.attempts[-1:]
        self.assertIn(
            "Both historical references require visual confirmation",
            validate_structure(self.record, updated, self.window, plan, audit),
        )

    def test_anchor_plural_validation(self):
        for ids in [
            ["image_0", "image_1"],
            ["image_0", "image_0"],
            ["image_0", "image_1", "image_2"],
        ]:
            model = Replies(
                [
                    {
                        "safe": True,
                        "target": "identify building",
                        "entity": "building",
                        "anchor_image_ids": ids,
                    }
                ]
            )
            if len(ids) == 2 and len(set(ids)) == 2:
                self.assertEqual(
                    select_anchor(
                        self.record, self.window, model, self.config, self.source
                    )["anchor_image_ids"],
                    ids,
                )
            else:
                with self.assertRaises(ValueError):
                    select_anchor(
                        self.record, self.window, model, self.config, self.source
                    )

    def test_negative_then_two_matches(self):
        plan = self.plan()
        negative = next(c for c in plan.first if c.verdict == "NO")
        model = Replies(
            [
                {"image_id": rid, "reasoning": "Inspect this plausible lead."}
                for rid in [negative.public["image_id"], "image_0", "image_1"]
            ]
        )
        choose_checks(plan, self.window, self.source, model, self.config, self.record)
        self.assertEqual(plan.branch, "contrast")
        self.assertEqual(plan.metadata["wrong_count"], 1)
        self.assertEqual([c.verdict for c in plan.attempts], ["NO", "YES", "YES"])
        updated, audit = test_pipeline.PipelineTests.assemble(
            self, self.record, self.window, plan
        )
        self.assertEqual(
            validate_structure(self.record, updated, self.window, plan, audit), []
        )
