import unittest
from unittest.mock import Mock

from synthesis.post_process.fix.rollout import select_with_repair
from synthesis.post_process.fix.schema import Config


class SelectorRepairTests(unittest.TestCase):
    def run_selection(self, responses, audit):
        model = Mock()
        model.ask.side_effect = responses
        payload = {
            "question": "A red train",
            "trajectory_history": [],
            "uninspected_image_ids": ["A", "B"],
        }
        result = select_with_repair(
            model,
            Config(),
            payload,
            {"A": object(), "B": object()},
            {"A": "Inspect this train.", "B": "Inspect the other train."},
            [],
            "A red train",
            0,
            audit,
        )
        return model, result

    def test_tie_repair_preserves_sampling_and_hides_private_state(self):
        bad = {
            "image_id": "A",
            "reasoning": "Inspect this train.",
            "equally_plausible_ids": ["A", "B"],
            "preference_evidence": "A is better",
        }
        good = {**bad, "preference_evidence": ""}
        audit = {}
        model, (_, result) = self.run_selection([bad, good], audit)
        self.assertEqual(model.ask.call_count, 2)
        self.assertEqual(result[2], "plausible_tie_sample")
        feedback = model.ask.call_args.args[2]["validation_feedback"]
        self.assertIn("Tied candidates", feedback["error"])
        self.assertNotIn("category", str(model.ask.call_args_list))
        self.assertEqual(
            [a["valid"] for a in audit["selector_validation_attempts"]], [False, True]
        )

    def test_bad_quote_repaired(self):
        bad = {
            "image_id": "A",
            "reasoning": "Inspect this train.",
            "exclusions": [
                {
                    "image_id": "B",
                    "evidence_quote": "unknown country",
                    "reason": "contradiction",
                }
            ],
        }
        good = {**bad, "exclusions": []}
        audit = {}
        self.run_selection([bad, good], audit)
        self.assertIn(
            "exclusions[0].evidence_quote",
            audit["selector_validation_attempts"][0]["error"],
        )

    def test_bounded_failure_and_valid_first_response(self):
        bad = {"image_id": "missing"}
        audit = {}
        with self.assertRaisesRegex(ValueError, "after 3 attempts"):
            self.run_selection([bad, bad, bad], audit)
        self.assertEqual(len(audit["selector_validation_attempts"]), 3)
        model, _ = self.run_selection(
            [{"image_id": "A", "reasoning": "Inspect this train."}], {}
        )
        self.assertEqual(model.ask.call_count, 1)
