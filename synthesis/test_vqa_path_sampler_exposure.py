"""Targeted tests for trajectory-history exposure matching."""

from __future__ import annotations

import unittest

from synthesis.vqa.path_sampler import RandomPathSampler, SamplerConfiguration


class HistoryExposureMatchTests(unittest.TestCase):
    def setUp(self) -> None:
        # The helpers under test do not access sampler instance state.
        self.sampler = object.__new__(RandomPathSampler)

    def match(self, title: str, *history: str) -> dict:
        labels = self.sampler._candidate_text_exposure_labels(
            {"title": title, "aliases": []}
        )
        return self.sampler._hard_history_exposure_match(
            labels=labels,
            exposure_parts=[
                {"kind": "edge_relation", "text": text}
                for text in history
            ],
        )

    def test_parenthetical_title_is_blocked_by_exposed_base_name(self) -> None:
        result = self.match(
            "Astor Court (Metropolitan Museum of Art)",
            "venue where a 2012 production was performed in the Astor Court",
        )

        self.assertFalse(result["allow"])
        self.assertEqual(result["matched_label"], "Astor Court")

    def test_national_team_is_blocked_when_country_anchor_was_exposed(self) -> None:
        result = self.match(
            "New Zealand national rugby union team",
            "the photo of New Zealand captain David Kirk lifting the inaugural trophy",
        )

        self.assertFalse(result["allow"])
        self.assertEqual(result["matched_label"], "New Zealand")

    def test_generic_topical_overlap_does_not_block(self) -> None:
        result = self.match(
            "Metropolitan Museum of Art",
            "a play was performed in a museum courtyard",
        )

        self.assertTrue(result["allow"])

    def test_different_country_team_is_not_blocked(self) -> None:
        result = self.match(
            "South Africa national rugby union team",
            "New Zealand captain David Kirk lifting a trophy in Auckland",
        )

        self.assertTrue(result["allow"])


class GenericCategoryScoreCapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sampler = object.__new__(RandomPathSampler)
        self.sampler.config = SamplerConfiguration(llm_generic_category_score_cap=0.15)

    def test_caps_generic_category_score(self) -> None:
        raw, effective, classified = self.sampler._capped_llm_candidate_score(
            {"score": 0.92, "is_generic_category_target": True}
        )
        self.assertEqual(raw, 0.92)
        self.assertEqual(effective, 0.15)
        self.assertTrue(classified)

    def test_does_not_cap_specific_entity_score(self) -> None:
        raw, effective, classified = self.sampler._capped_llm_candidate_score(
            {"score": 0.92, "is_generic_category_target": False}
        )
        self.assertEqual(raw, 0.92)
        self.assertEqual(effective, 0.92)
        self.assertFalse(classified)


if __name__ == "__main__":
    unittest.main()
