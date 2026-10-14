"""SearchWeave category reporting for fresh and resumed evaluations."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from opensearch_vl import eval_infer_with_llm as evaluator


class SearchWeaveCategoryTests(unittest.TestCase):
    def test_parquet_categories_enrich_resumed_results_without_judging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            answer_file = root / "searchweave.parquet"
            pd.DataFrame([
                {"question_id": "q1", "answer": "a", "category": "image-start"},
                {"question_id": "q2", "answer": "a", "category": "image-start"},
                {"question_id": "q3", "answer": "a", "category": "image-end"},
                {"question_id": "q4", "answer": "a", "category": "image-interleave"},
            ]).to_parquet(answer_file)
            cached = []
            for i, acc in enumerate([1, 0, 1, 0], start=1):
                case_id = f"case{i}"
                (root / f"{case_id}_trajectory.json").write_text(json.dumps({
                    "case_id": case_id, "original_data": {"question_id": f"q{i}"},
                }))
                cached.append({"case_id": case_id, "acc": acc})
            details = root / "llm_eval_report_details.jsonl"
            details.write_text("".join(json.dumps(item) + "\n" for item in cached))
            with mock.patch.object(evaluator, "judge_answer") as judge:
                report = evaluator.run_eval(str(root), answer_file=str(answer_file))
            judge.assert_not_called()
            self.assertEqual(report["accuracy"], "50.00%")
            self.assertEqual(report["category_accuracy"], {
                "image-start": {"total": 2, "correct": 1, "accuracy": "50.00%"},
                "image-end": {"total": 1, "correct": 1, "accuracy": "100.00%"},
                "image-interleave": {"total": 1, "correct": 0, "accuracy": "0.00%"},
            })
            saved = [json.loads(line) for line in details.read_text().splitlines()]
            self.assertEqual([item["category"] for item in saved], [
                "image-start", "image-start", "image-end", "image-interleave",
            ])
            saved_report = json.loads((root / "llm_eval_report.json").read_text())
            self.assertEqual(saved_report["category_accuracy"], report["category_accuracy"])

    def test_trajectory_categories_include_judge_failures_and_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for case_id, category in [("a", "image-start"), ("b", "image-end"), ("c", "")]:
                (root / f"{case_id}_trajectory.json").write_text(json.dumps({
                    "case_id": case_id, "category": category,
                    "original_data": {"answer": "a"}, "final_response_text": "a",
                }))

            def judge(question, correct_answer, response, **kwargs):
                raise RuntimeError("judge unavailable")

            with mock.patch.object(evaluator, "judge_answer", side_effect=judge):
                report = evaluator.run_eval(str(root))
            self.assertEqual(report["error_count"], 3)
            self.assertEqual(report["accuracy"], "0.00%")
            self.assertEqual(set(report["category_accuracy"]), {"image-start", "image-end", "unknown"})
            self.assertTrue(all(item == {"total": 1, "correct": 0, "accuracy": "0.00%"}
                                for item in report["category_accuracy"].values()))

    def test_other_benchmark_does_not_get_searchweave_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            answer_file = root / "other.parquet"
            pd.DataFrame([{"id": "q1", "answer": "a", "category": "science"}]).to_parquet(answer_file)
            (root / "q1_trajectory.json").write_text(json.dumps({
                "case_id": "q1", "category": "science", "original_data": {},
            }))
            with mock.patch.object(evaluator, "judge_answer", return_value={"acc": 1}):
                report = evaluator.run_eval(str(root), answer_file=str(answer_file))
            self.assertNotIn("category_accuracy", report)
            self.assertEqual(report["accuracy"], "100.00%")


if __name__ == "__main__":
    unittest.main()
