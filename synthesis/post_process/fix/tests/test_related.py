import tempfile
import unittest
from pathlib import Path
from synthesis.post_process.fix.schema import Config
from synthesis.post_process.fix.candidate_sets import (
    build_plan,
    rng_for,
    sample_negatives,
)
from synthesis.post_process.fix.tests.test_pipeline import fixture, candidates
from synthesis.post_process.fix.trajectory_index import index_record


class RelatedTests(unittest.TestCase):
    def test_unrelated_never_sampled_or_selectable(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from synthesis.post_process.fix.selection import select_visible

        bad = SimpleNamespace(
            key="noise",
            public={"image_id": "noise"},
            category="UNRELATED",
            verdict="NO",
        )
        good = SimpleNamespace(
            key="good",
            public={"image_id": "good"},
            category="CONSISTENT",
            verdict="YES",
        )
        with self.assertRaises(ValueError):
            sample_negatives([bad], 1, rng_for(42, "noise", 0), Config())
        model = Mock()
        model.ask.return_value = {
            "image_id": "good",
            "reasoning": "Inspect plausible result.",
        }
        select_visible(
            [bad, good],
            set(),
            [],
            SimpleNamespace(question="q"),
            "source",
            model,
            Config(),
        )
        payload = model.ask.call_args.args[2]
        self.assertEqual(payload["uninspected_image_ids"], ["good"])
        self.assertEqual(len(payload["search_results"]), 2)
        model.ask.return_value = {"image_id": "noise", "reasoning": "Invalid choice"}
        with self.assertRaises(ValueError):
            select_visible(
                [bad, good],
                set(),
                [],
                SimpleNamespace(question="q"),
                "source",
                model,
                Config(),
            )

    def test_related_enables_contrast_and_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            record, source, target = fixture(Path(directory))
            window = index_record(record)[0]
            old, pool = candidates(window, target)
            for candidate in pool:
                candidate.category = "RELATED"
            for branch in ("contrast", "recovery"):
                config = Config(
                    branch_weights={
                        name: float(name == branch)
                        for name in ("direct", "contrast", "recovery", "convergent")
                    }
                )
                plan = build_plan(
                    window,
                    old,
                    pool,
                    {"anchor_image_id": old[0].public["image_id"]},
                    {"eligible": False},
                    config,
                    rng_for(42, "related", 0),
                )
                self.assertEqual(plan.branch, branch)
                self.assertTrue(any(c.category == "RELATED" for c in plan.first))
            selected = sample_negatives(pool, 3, rng_for(42, "sample", 0), Config())
            self.assertEqual(len(selected), 3)
            self.assertTrue(all(c.category == "RELATED" for c in selected))
