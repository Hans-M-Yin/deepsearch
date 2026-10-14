import unittest
from synthesis.post_process.fix.rollout import choose_from_assessment


class SelectorExplorationTests(unittest.TestCase):
    def test_tie_sampling_is_reproducible_and_uses_selected_reason(self):
        response = {
            "image_id": "a",
            "reasoning": "Inspect A.",
            "equally_plausible_ids": ["a", "b"],
            "preference_evidence": "",
            "exclusions": [],
        }
        remaining = {"a": object(), "b": object()}
        reasons = {"a": "Inspect A.", "b": "Inspect B."}
        chosen = set()
        for seed in range(20):
            args = (response, remaining, reasons, [], "Which train?", seed, 0)
            result = choose_from_assessment(*args)
            self.assertEqual(result, choose_from_assessment(*args))
            rid, text, mode = result
            self.assertEqual(text, reasons[rid])
            self.assertEqual(mode, "plausible_tie_sample")
            chosen.add(rid)
        self.assertEqual(chosen, {"a", "b"})

    def test_exclusion_requires_observed_quote(self):
        response = {
            "image_id": "a",
            "reasoning": "Inspect A.",
            "exclusions": [
                {
                    "image_id": "b",
                    "evidence_quote": "The country is France",
                    "reason": "different country",
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "evidence_quote.*verbatim"):
            choose_from_assessment(
                response, {"a": None, "b": None}, {}, [], "Which country?", 42, 0
            )

    def test_invalid_tie_sets_rejected(self):
        for ties in (["a", "a"], ["b"], ["a", "unavailable"]):
            response = {
                "image_id": "a",
                "reasoning": "Inspect A.",
                "equally_plausible_ids": ties,
            }
            with self.assertRaises(ValueError):
                choose_from_assessment(
                    response, {"a": None, "b": None}, {}, [], "question", 42, 0
                )
