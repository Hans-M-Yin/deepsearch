import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from synthesis.post_process.fix.candidate_relevance import visible_card
from synthesis.post_process.fix.eligibility import select_anchor
from synthesis.post_process.fix.errors import ReviewRejected
from synthesis.post_process.fix.replay_i2i import recover_historical
from synthesis.post_process.fix.rewrite import validate_observation, write_after
from synthesis.post_process.fix.schema import Candidate, Config
from synthesis.post_process.fix.tests.test_pipeline import fixture, action
from synthesis.post_process.fix.tests.writer_fixture import observation
from synthesis.post_process.fix.trajectory_index import index_record, messages
from synthesis.post_process.fix.search_cards import build_search_plan


class EvidenceGuardrailsTests(unittest.TestCase):
    def test_historical_keywords_survive_recovery(self):
        old = {
            "image_id": "old",
            "source_page_id": "page",
            "title": "Title",
            "source": "site",
            "image_url_keywords": "essential query terms",
        }
        replay = Candidate("new", {**old, "image_id": "new", "image_url_keywords": ""})
        result = recover_historical(
            SimpleNamespace(candidates=[old], reads={}), [replay], Path(".")
        )[0]
        self.assertEqual(result.public["image_id"], "new")
        self.assertEqual(
            visible_card(result)["image_url_keywords"], "essential query terms"
        )
        self.assertEqual(old["image_id"], "old")

    def test_explicit_and_semantic_dependencies_are_retained(self):
        with tempfile.TemporaryDirectory() as root:
            record, source, _ = fixture(Path(root))
            window = index_record(record)[0]
            messages(record).append(action("read_url", {"resource_id": "page_3"}))
            model = Mock()
            model.ask.return_value = {
                "safe": True,
                "anchor_image_ids": ["image_0"],
                "dependency_image_ids": ["image_2"],
                "entity": "Target",
                "target": "building",
            }
            result = select_anchor(record, window, model, Config(), source)
            self.assertEqual(
                set(result["required_reference_ids"]), {"image_0", "image_2", "image_3"}
            )

    def test_writer_cannot_rationalize_negative_label(self):
        candidate = Candidate("n", {}, category="SIMILAR", verdict="NO")
        for verdict in ["YES", "CONTINUE"]:
            with self.assertRaises(ReviewRejected):
                validate_observation(
                    observation("Poster wording is readable", verdict), candidate
                )

    def test_public_text_preserved_but_private_labels_hidden(self):
        c = Candidate(
            "secret",
            {
                "title": "T",
                "snippet": "historical caption",
                "source_page_url_keywords": "archive date",
                "category": "SIMILAR",
            },
        )
        self.assertEqual(visible_card(c)["snippet"], "historical caption")
        self.assertEqual(visible_card(c)["source_page_url_keywords"], "archive date")
        self.assertNotIn("category", visible_card(c))

    def test_composition_keeps_non_identity_dependency(self):
        p = Candidate(
            "old_a",
            {"image_id": "a"},
            category="CONSISTENT",
            verdict="YES",
            image_paths=["image"],
        )
        d = Candidate("old_b", {"image_id": "b", "snippet": "essential fact"})
        model = Mock()
        model.ask.return_value = {"candidate_keys": ["old_a", "old_b"]}
        plan = build_search_plan(
            [p, d],
            [],
            {
                "anchor_image_ids": ["a"],
                "required_reference_ids": ["a", "b"],
                "target": "t",
            },
            SimpleNamespace(question="q"),
            model,
            Config(),
        )
        self.assertEqual({c.key for c in plan.first}, {"old_a", "old_b"})
        self.assertEqual(
            next(c for c in plan.first if c.key == "old_b").public["snippet"],
            "essential fact",
        )

    def test_positive_still_requires_local_goal_evidence(self):
        candidate = Candidate("p", {}, category="SAME_ENTITY", verdict="YES")
        with self.assertRaises(ReviewRejected):
            validate_observation(
                observation("Same person, but requested hand is occluded", "CONTINUE"),
                candidate,
            )
        self.assertEqual(
            validate_observation(observation("Distinctive tower matches"), candidate),
            "Distinctive tower matches",
        )

    def test_format_retry_but_no_retry_for_evidence_conflict(self):
        candidate = Candidate("p", {}, category="CONSISTENT", verdict="YES")
        model = Mock()
        model.ask.side_effect = [{"reasoning": "match"}, observation("match")]
        assessment = {}
        write_after(
            candidate,
            SimpleNamespace(question="q"),
            {"target": "t"},
            "source",
            model,
            Config(),
            assessment=assessment,
        )
        self.assertEqual(model.ask.call_count, 2)
        self.assertEqual(assessment["decision"], "ACCEPT")
        model.reset_mock()
        model.ask.side_effect = [observation("unresolved", "CONTINUE")]
        with self.assertRaises(ReviewRejected):
            write_after(
                candidate,
                SimpleNamespace(question="q"),
                {"target": "t"},
                "source",
                model,
                Config(),
                assessment={},
            )
        self.assertEqual(model.ask.call_count, 1)
