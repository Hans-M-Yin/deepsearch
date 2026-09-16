from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from synthesis.post_process.audit_hard_rl_questions import (
    _aggregate_decision,
    _deterministic_checks,
    _load_audit_cache,
    _matches_ambiguous_repair_review_filter,
    _parse_judge_result,
    _parse_uniqueness_result,
)


class ParseJudgeResultTest(unittest.TestCase):
    def test_parses_fenced_json(self) -> None:
        result = _parse_judge_result(
            """```json
            {"decision":"drop","objective_failure":true,
             "issue_categories":["wrong_gold"],"reason":"contradiction"}
            ```"""
        )
        self.assertTrue(result["parse_ok"])
        self.assertEqual(result["decision"], "drop")
        self.assertEqual(result["issue_categories"], ["wrong_gold"])

    def test_invalid_output_is_review(self) -> None:
        result = _parse_judge_result("not JSON")
        self.assertFalse(result["parse_ok"])
        self.assertEqual(result["decision"], "review")


class ParseUniquenessResultTest(unittest.TestCase):
    def test_valid_ambiguity_requires_two_distinct_answers(self) -> None:
        result = _parse_uniqueness_result(
            """{
              "decision":"ambiguous",
              "ambiguity_types":["unspecified_temporal_scope"],
              "explicit_constraints":["according to a census"],
              "candidate_interpretations":[
                {"candidate":"2011 census","why_it_satisfies_all_constraints":"No year is stated","resulting_answer":"530,000"},
                {"candidate":"2021 census","why_it_satisfies_all_constraints":"No year is stated","resulting_answer":"467,000"}
              ],
              "missing_disambiguator":"census year",
              "reason":"The year changes the answer",
              "repair_suggestion":"Name the census year"
            }"""
        )
        self.assertTrue(result["parse_ok"])
        self.assertEqual(result["decision"], "ambiguous")

    def test_unsupported_ambiguity_is_downgraded_to_uncertain(self) -> None:
        result = _parse_uniqueness_result(
            """{
              "decision":"ambiguous",
              "ambiguity_types":["entity_collision"],
              "candidate_interpretations":[
                {"candidate":"one","why_it_satisfies_all_constraints":"claimed","resulting_answer":"same"},
                {"candidate":"two","why_it_satisfies_all_constraints":"claimed","resulting_answer":"same"}
              ],
              "missing_disambiguator":"identity",
              "reason":"claimed ambiguity"
            }"""
        )
        self.assertEqual(result["decision"], "uncertain")
        self.assertEqual(
            result["validation_error"],
            "ambiguous_requires_two_concrete_distinct_answers",
        )


class AggregateDecisionTest(unittest.TestCase):
    @staticmethod
    def _result(decision: str, category: str = "wrong_gold") -> dict:
        return {
            "parse_ok": True,
            "decision": decision,
            "objective_failure": decision == "drop",
            "issue_categories": [category] if decision == "drop" else [],
            "evidence": ["specific contradiction"] if decision == "drop" else [],
            "alternative_answers": [],
        }

    def test_two_matching_drop_votes_can_drop(self) -> None:
        decision, metadata = _aggregate_decision(
            [self._result("drop"), self._result("drop")],
            fatal_categories=[],
            min_drop_votes=2,
        )
        self.assertEqual(decision, "drop")
        self.assertEqual(metadata["shared_drop_categories"], ["wrong_gold"])

    def test_disagreement_is_review(self) -> None:
        decision, _ = _aggregate_decision(
            [self._result("drop"), self._result("keep_hard")],
            fatal_categories=[],
            min_drop_votes=2,
        )
        self.assertEqual(decision, "review")

    def test_different_drop_reasons_do_not_auto_drop(self) -> None:
        decision, _ = _aggregate_decision(
            [
                self._result("drop", "wrong_gold"),
                self._result("drop", "visual_ambiguity"),
            ],
            fatal_categories=[],
            min_drop_votes=2,
        )
        self.assertEqual(decision, "review")

    def test_uniqueness_disagreement_prevents_keep(self) -> None:
        decision, metadata = _aggregate_decision(
            [self._result("keep_hard"), self._result("keep_hard")],
            fatal_categories=[],
            min_drop_votes=2,
            uniqueness_result={"parse_ok": True, "decision": "ambiguous"},
        )
        self.assertEqual(decision, "review")
        self.assertEqual(metadata["basis"], "uniqueness_quality_disagreement")


class DeterministicChecksTest(unittest.TestCase):
    def test_visual_question_without_initial_image_is_not_fatal(self) -> None:
        checks = _deterministic_checks(
            {
                "question": "Which person is shown in the image?",
                "answer": "Example Person",
                "images": [],
            },
            duplicate_answer_count=1,
        )
        self.assertNotIn("missing_required_image", checks["issues"])
        self.assertEqual(checks["fatal_categories"], [])

    def test_broken_supplied_image_reference_is_fatal(self) -> None:
        checks = _deterministic_checks(
            {
                "question": "Which person is shown in the image?",
                "answer": "Example Person",
                "images": ["/definitely/missing/image.png"],
            },
            duplicate_answer_count=1,
        )
        self.assertIn("missing_required_image", checks["fatal_categories"])

    def test_readable_image_is_not_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "image.png"
            Image.new("RGB", (8, 8), color="red").save(path)
            checks = _deterministic_checks(
                {
                    "question": "Which person is shown in the image?",
                    "answer": "Example Person",
                    "images": [str(path)],
                },
                duplicate_answer_count=1,
            )
        self.assertEqual(checks["fatal_categories"], [])
        self.assertEqual(checks["image_inspections"][0]["status"], "readable")


class ResumeCacheTest(unittest.TestCase):
    def test_resume_skips_only_complete_records_and_ignores_malformed_lines(self) -> None:
        signature = "test-signature"
        complete = {
            "case_id": "complete",
            "audit_signature": signature,
            "deterministic_checks": {"fatal_categories": []},
            "uniqueness_result": {"parse_ok": True, "decision": "unique"},
            "judge_results": [{"parse_ok": True}, {"parse_ok": True}],
        }
        failed = {
            "case_id": "failed",
            "audit_signature": signature,
            "deterministic_checks": {"fatal_categories": []},
            "uniqueness_result": {"parse_ok": False, "decision": "uncertain"},
            "judge_results": [],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "audit.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(complete) + "\n")
                handle.write(json.dumps(failed) + "\n")
                handle.write('{"case_id":"interrupted"')
            cached, stats = _load_audit_cache(
                path,
                signature,
                expected_quality_judge_calls=2,
            )

        self.assertEqual(set(cached), {"complete"})
        self.assertEqual(stats["reusable_cases"], 1)
        self.assertEqual(stats["retryable_cases"], 1)
        self.assertEqual(stats["malformed_records"], 1)


class AmbiguousRepairReviewFilterTest(unittest.TestCase):
    def test_removes_only_validated_ambiguous_repair_or_review(self) -> None:
        for final_decision in ("repair", "review"):
            self.assertTrue(
                _matches_ambiguous_repair_review_filter(
                    {
                        "final_decision": final_decision,
                        "uniqueness_result": {
                            "parse_ok": True,
                            "decision": "ambiguous",
                            "validation_error": "",
                        },
                    }
                )
            )

    def test_does_not_remove_other_or_failed_uniqueness_results(self) -> None:
        records = [
            {
                "final_decision": "keep_hard",
                "uniqueness_result": {"parse_ok": True, "decision": "ambiguous"},
            },
            {
                "final_decision": "drop",
                "uniqueness_result": {"parse_ok": True, "decision": "ambiguous"},
            },
            {
                "final_decision": "repair",
                "uniqueness_result": {"parse_ok": True, "decision": "unique"},
            },
            {
                "final_decision": "review",
                "uniqueness_result": {"parse_ok": False, "decision": "uncertain"},
            },
            {
                "final_decision": "review",
                "uniqueness_result": {
                    "parse_ok": True,
                    "decision": "ambiguous",
                    "validation_error": "ambiguous_requires_two_concrete_distinct_answers",
                },
            },
        ]
        self.assertTrue(
            all(not _matches_ambiguous_repair_review_filter(record) for record in records)
        )


if __name__ == "__main__":
    unittest.main()
