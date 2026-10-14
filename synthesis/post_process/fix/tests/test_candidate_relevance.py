import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from synthesis.post_process.fix.candidate_relevance import (
    assess_competition,
)
from synthesis.post_process.fix.candidate_sets import inspectable_negative
from synthesis.post_process.fix.schema import Candidate, Config
from synthesis.post_process.fix.selection import select_visible


class CompetitionTests(unittest.TestCase):
    def test_third_check_is_guided_after_two_failures(self):
        from synthesis.post_process.fix.selection import select_sequence

        cs = [
            Candidate(
                str(i),
                {"image_id": f"image_{i}", "title": f"Lead {i}"},
                verdict="YES" if i == 2 else "NO",
                category="CONSISTENT" if i == 2 else "SIMILAR",
            )
            for i in range(3)
        ]
        model = Mock()
        model.ask.side_effect = [
            {"image_id": f"image_{i}", "reasoning": "Inspect the captioned lead."}
            for i in range(3)
        ]
        audit = []
        attempts, _ = select_sequence(
            cs,
            SimpleNamespace(question="Q"),
            "source",
            model,
            Config(blind_visual_review=False),
            limit=3,
            require_match=True,
            selection_audit=audit,
        )
        self.assertEqual(len(attempts), 3)
        calls = model.ask.call_args_list
        self.assertNotIn("designated_image_id", calls[0].args[2])
        self.assertNotIn("designated_image_id", calls[1].args[2])
        self.assertEqual(calls[2].args[2]["designated_image_id"], "image_2")
        self.assertEqual(
            [x["mode"] for x in audit], ["autonomous", "autonomous", "guided"]
        )

    def test_reference_ids_restored_without_private_labels(self):
        c = Candidate(
            "old_secret",
            {
                "title": "A",
                "source": "site",
                "image_id": "image_a",
                "source_page_id": "page_a",
                "image_url_keywords": "public archival clue",
            },
            verdict="YES",
            category="CONSISTENT",
        )
        model = Mock()
        model.ask.return_value = {
            "image_id": "image_a",
            "reasoning": "Inspect this caption.",
        }
        select_visible(
            [c],
            set(),
            [],
            SimpleNamespace(question="Q"),
            "source",
            model,
            Config(),
            reference_ids=["image_a"],
        )
        payload = model.ask.call_args.args[2]
        self.assertEqual(payload["references_to_check"], ["image_a"])
        self.assertEqual(
            set(payload["search_results"][0]),
            {"title", "source", "image_id", "source_page_id", "image_url_keywords"},
        )
        self.assertNotIn("secret", str(payload))
        self.assertEqual(
            payload["search_results"][0]["image_url_keywords"], "public archival clue"
        )

    def test_weak_negative_is_not_augmentation_candidate(self):
        c = Candidate(
            "n",
            {"image_id": "image_n", "title": "Generic"},
            verdict="NO",
            category="SIMILAR",
        )
        model = Mock()
        model.ask.return_value = {
            "cards": [
                {
                    "image_id": "image_n",
                    "competitive": False,
                    "reason": "No specific clue",
                }
            ]
        }
        assess_competition(
            [c], SimpleNamespace(question="Q"), "source", [], model, Config()
        )
        self.assertFalse(inspectable_negative(c))
        self.assertNotIn("SIMILAR", str(model.ask.call_args.args[2]))
        self.assertEqual(model.ask.call_args.args[3], ["source"])

    def test_incomplete_assessment_is_rejected(self):
        model = Mock()
        model.ask.return_value = {"cards": []}
        with self.assertRaises(ValueError):
            assess_competition(
                [Candidate("n", {"image_id": "image_n"})],
                SimpleNamespace(question="Q"),
                "source",
                [],
                model,
                Config(),
            )
