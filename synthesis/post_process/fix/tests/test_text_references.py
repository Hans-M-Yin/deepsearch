import unittest
import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from synthesis.post_process.fix.errors import SkipAugmentation
from synthesis.post_process.fix.schema import Candidate, Config
from synthesis.post_process.fix.search_cards import build_search_plan, is_convergent
from synthesis.post_process.fix.rollout import inspection_reasons, SELECT, REASONS
from synthesis.post_process.fix.rewrite import AFTER
from synthesis.post_process.fix.evidence_attribution import EVIDENCE_ATTRIBUTION
from synthesis.post_process.fix.pipeline import process_record
from synthesis.post_process.fix.storage import Cache
from synthesis.post_process.fix.validate import validate_structure
from synthesis.post_process.fix.tests.test_pipeline import fixture
from synthesis.post_process.fix.tests.test_integration import FakeModel, FakeBackend


class TextReferenceTests(unittest.TestCase):
    def test_pipeline_retains_unavailable_text_reference_and_rejects_misuse(self):
        from synthesis.post_process.fix import progress

        class Model(FakeModel):
            def ask(self, alias, system, payload, images=()):
                result = super().ask(alias, system, payload, images)
                if system == progress.PROMPT:
                    result["anchor_image_ids"] = ["image_0", "image_1"]
                return result

        class Backend(FakeBackend):
            def read(self, candidate):
                if candidate.key == "old_image_1":
                    raise ValueError("Historical image unavailable")
                return super().read(candidate)

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            original, _, target = fixture(root)
            with (
                patch(
                    "synthesis.post_process.fix.title_recovery.recover_anchor",
                    return_value={"status": "unresolved"},
                ),
                patch(
                    "synthesis.post_process.fix.pipeline.validate_structure",
                    wraps=validate_structure,
                ) as check,
            ):
                _, audit = process_record(
                    original,
                    root,
                    root / "work",
                    Config(),
                    Backend(target),
                    Model(),
                    Cache(root / "cache"),
                )
            self.assertEqual(audit["status"], "rewritten", audit)
            self.assertTrue(audit["windows"][0]["review"]["pass"])
            args = check.call_args_list[0].args
            plan = args[3]
            self.assertEqual(plan.metadata["text_only_reference_keys"], ["old_image_1"])
            self.assertNotIn("old_image_1", [c.key for c in plan.attempts])
            self.assertEqual(validate_structure(*args), [])
            for mutation in [
                "select",
                "positive",
                "new_card",
                "change_title",
                "viewed",
                "missing_flag",
                "metadata",
                "step",
            ]:
                with self.subTest(mutation=mutation):
                    mutated = copy.deepcopy(args)
                    mp = mutated[3]
                    c = next(c for c in mp.first if c.key == "old_image_1")
                    if mutation == "select":
                        mp.attempts.insert(0, c)
                    elif mutation == "positive":
                        c.category, c.verdict = "CONSISTENT", "YES"
                    elif mutation == "new_card":
                        c.historical = False
                    elif mutation == "change_title":
                        c.public["title"] = "Fabricated title"
                    elif mutation == "viewed":
                        mutated[2].reads["image_1"] = {"paths": [str(target)]}
                    elif mutation == "missing_flag":
                        c.text_reference_only = False
                    elif mutation == "metadata":
                        mp.metadata["text_only_reference_keys"] = []
                    elif mutation == "step":
                        mp.metadata["steps"].append({"candidate_key": c.key})
                    self.assertTrue(validate_structure(*mutated))

    def setup_plan(self, viewed=False, positive=True):
        old = Candidate(
            "old_missing",
            {
                "image_id": "missing",
                "source_page_id": "old_page",
                "title": "Original title",
            },
            historical=True,
        )
        found = Candidate(
            "old_found",
            {
                "image_id": "found",
                "source_page_id": "found_page",
                "title": "Found title",
            },
            image_paths=["real.jpg"],
            category="CONSISTENT",
            verdict="YES",
        )
        anchor = {
            "anchor_image_ids": ["missing", "found"] if positive else ["missing"],
            "viewed_before_conclusion_ids": ["missing"] if viewed else [],
            "target": "target",
        }
        plan = build_search_plan(
            [old, found] if positive else [old],
            [],
            anchor,
            SimpleNamespace(question="q"),
            Mock(),
            Config(),
        )
        return plan, old

    def test_preserves_text_without_fabricating_image(self):
        plan, old = self.setup_plan()
        self.assertIn(old, plan.first)
        self.assertTrue(old.text_reference_only)
        self.assertEqual(old.public["source_page_id"], "old_page")
        self.assertEqual(old.image_paths, [])
        self.assertEqual(plan.metadata["text_only_reference_keys"], ["old_missing"])
        self.assertFalse(is_convergent(plan))
        model = Mock()
        model.ask.return_value = {
            "cards": [{"image_id": "found", "reasoning": "Check this source."}]
        }
        inspection_reasons(
            plan.first, SimpleNamespace(question="q"), [], model, Config()
        )
        self.assertEqual(
            [c["image_id"] for c in model.ask.call_args.args[2]["search_results"]],
            ["found"],
        )

    def test_viewed_reference_still_requires_image(self):
        with self.assertRaises(SkipAugmentation) as caught:
            self.setup_plan(viewed=True)
        self.assertEqual(caught.exception.reason, "unresolved_required_reference")

    def test_text_reference_cannot_replace_verified_positive(self):
        with self.assertRaises(SkipAugmentation) as caught:
            self.setup_plan(positive=False)
        self.assertEqual(caught.exception.reason, "no_verified_positive")

    def test_reasoning_prompts_share_evidence_policy(self):
        for prompt in [SELECT, REASONS, AFTER]:
            self.assertIn(EVIDENCE_ATTRIBUTION, prompt)
