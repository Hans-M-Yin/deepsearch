from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from synthesis.model_worker import ModelResponse
from synthesis.sft.filter_sft_trajectories import filter_jsonl


class FakeJudge:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.requests: list[Any] = []

    def generate(self, request: Any) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(content=json.dumps(self.responses.pop(0), ensure_ascii=False))


def _record(sample_id: str, *, max_turns: bool = False) -> dict[str, Any]:
    return {
        "question_id": sample_id.replace("sample", "q"),
        "sample_id": sample_id,
        "path_id": "path_" + sample_id,
        "question": "Which answer?",
        "gold_answer": "correct answer",
        "extracted_answer": "correct answer",
        "answer_judge": {"is_correct": True},
        "generation_summary": {
            "generation_status": "max_turns_reached" if max_turns else "finished",
            "generation_complete": not max_turns,
            "failure_reasons": ["max_turns_reached"] if max_turns else [],
        },
        "raw_messages": [
            {"role": "user", "content": "Question: Which answer? Answer: correct answer"},
            {"role": "assistant", "content": "I will inspect evidence."},
        ],
        "hop_chain": [{"statement": "private answer hint"}],
    }


class FilterSftTrajectoriesTest(unittest.TestCase):
    def test_categories_disagreements_and_accepted_jsonl(self) -> None:
        wrong = _record("sample_wrong")
        records = [_record("sample_good"), _record("sample_max", max_turns=True), _record("sample_low"), wrong]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "raw.jsonl"
            output_dir = root / "filtered"
            input_path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
            fake = FakeJudge(
                [
                    {"predict": "pass", "confidence": 0.99, "reason": "equivalent"},
                    {"logic_coherence": 8, "answer_exposure": 9, "tool_use": 8},
                    {"predict": "pass", "confidence": 0.95, "reason": "equivalent"},
                    {"logic_coherence": 6, "answer_exposure": 8, "tool_use": 8},
                    {"predict": "reject", "confidence": 0.99, "reason": "different answer"},
                ]
            )
            report = filter_jsonl(
                input_path,
                output_dir,
                quality_model_alias="quality",
                simple_model_alias="simple",
                model_client=fake,
            )
            self.assertEqual(report["decision_counts"], {"keep": 1, "reject": 3})
            self.assertEqual(report["filtered_ids"]["max_turn_reached_ids"], ["sample_max"])
            self.assertEqual(report["filtered_ids"]["wrong_answer_ids"], ["sample_wrong"])
            self.assertEqual(report["filtered_ids"]["logic_coherence_low_score_ids"], ["sample_low"])
            self.assertEqual(report["correctness_disagreements"][0]["record_id"], "sample_wrong")
            self.assertEqual(report["stage_survivors"], {
                "after_max_turn": 3,
                "after_answer_correctness": 2,
                "after_quality_dimensions": 1,
                "after_quality_average": 1,
            })
            accepted = (output_dir / "accepted_trajectories.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(accepted), 1)
            self.assertEqual(json.loads(accepted[0])["sample_id"], "sample_good")
            # max-turn is removed before any model judge: good(simple+quality),
            # low(simple+quality), wrong(simple) = five calls.
            self.assertEqual(len(fake.requests), 5)

    def test_average_score_rejects_without_dimension_category(self) -> None:
        record = _record("sample_average")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "raw.jsonl"
            output_dir = root / "filtered"
            input_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            fake = FakeJudge(
                [
                    {"predict": "pass", "confidence": 0.90, "reason": "equivalent"},
                    {"logic_coherence": 6.5, "answer_exposure": 6.5, "tool_use": 6.8},
                ]
            )
            report = filter_jsonl(
                input_path,
                output_dir,
                quality_model_alias="quality",
                simple_model_alias="simple",
                model_client=fake,
            )
            self.assertEqual(report["filtered_ids"]["average_score_low_ids"], ["sample_average"])
            self.assertEqual(report["filtered_ids"]["logic_coherence_low_score_ids"], [])

    def test_prompt_field_names_are_used_without_legacy_verdict_fallback(self) -> None:
        record = _record("sample_legacy")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "raw.jsonl"
            output_dir = root / "filtered"
            input_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            fake = FakeJudge([{"verdict": "pass"}])

            report = filter_jsonl(
                input_path,
                output_dir,
                quality_model_alias="quality",
                simple_model_alias="simple",
                model_client=fake,
            )

            self.assertEqual(report["decision_counts"], {"reject": 1})
            self.assertEqual(report["additional_review_ids"]["simple_judge_error_ids"], ["sample_legacy"])

    def test_quality_scores_accept_prompt_numeric_values_and_reject_invalid_range(self) -> None:
        good = _record("sample_float")
        bad = _record("sample_invalid")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "raw.jsonl"
            output_dir = root / "filtered"
            input_path.write_text(
                "".join(json.dumps(item) + "\n" for item in (good, bad)),
                encoding="utf-8",
            )
            fake = FakeJudge(
                [
                    {"predict": "pass", "confidence": 0.8, "reason": "equivalent"},
                    {
                        "logic_coherence": 7.5,
                        "answer_exposure": 8,
                        "tool_use": 7.5,
                        "logic_reason": "coherent",
                        "answer_exposure_reason": "no leak",
                        "tool_use_reason": "appropriate",
                        "overall_reason": "usable",
                    },
                    {"predict": "pass", "confidence": 0.8, "reason": "equivalent"},
                    {"logic_coherence": "1~10", "answer_exposure": 8, "tool_use": 8},
                ]
            )
            report = filter_jsonl(
                input_path,
                output_dir,
                quality_model_alias="quality",
                simple_model_alias="simple",
                model_client=fake,
            )

            self.assertEqual(report["decision_counts"], {"keep": 1, "reject": 1})
            self.assertEqual(report["additional_review_ids"]["quality_judge_error_ids"], ["sample_invalid"])
            accepted = [json.loads(line) for line in (output_dir / "accepted_trajectories.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(accepted[0]["sft_trajectory_filter"]["simple_judge"]["predict"], "pass")
            self.assertEqual(accepted[0]["sft_trajectory_filter"]["simple_judge"]["verdict"], "pass")


if __name__ == "__main__":
    unittest.main()
