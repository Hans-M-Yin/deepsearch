from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from synthesis.model_worker import ModelResponse
from synthesis.sft.filter_sft_trajectories import (
    FilterConfig,
    SftTrajectoryFilter,
    _extract_json_object,
    _extract_quality_scores_from_malformed_json,
    _format_cli_report,
    filter_jsonl,
)


class FakeJudge:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.requests: list[Any] = []

    def generate(self, request: Any) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(content=json.dumps(self.responses.pop(0), ensure_ascii=False))


class RoutingFakeJudge:
    def __init__(self) -> None:
        self.requests: list[Any] = []

    def generate(self, request: Any) -> ModelResponse:
        self.requests.append(request)
        label = str(request.metadata.get("trace_label") or "")
        if "simple_correctness" in label:
            response = {"predict": "pass"}
        else:
            response = {"logic_coherence": 8, "answer_exposure": 8, "tool_use": 8}
        return ModelResponse(content=json.dumps(response))


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
    def test_json_parser_ignores_trailing_model_text(self) -> None:
        self.assertEqual(
            _extract_json_object('{"predict":"pass"}\n}'),
            {"predict": "pass"},
        )

    def test_quality_parser_recovers_scores_from_malformed_reason_suffix(self) -> None:
        raw = (
            '{"logic_coherence": 9, "answer_exposure": 8.5, "tool_use": 9, '
            '"overall_reason": "Good trajectory." repeated prose that is not JSON}'
        )
        self.assertEqual(
            _extract_quality_scores_from_malformed_json(raw),
            {"logic_coherence": 9.0, "answer_exposure": 8.5, "tool_use": 9.0},
        )

    def test_quality_judge_marks_recovered_scores_as_scored(self) -> None:
        class MalformedQualityJudge:
            def generate(self, request: Any) -> ModelResponse:
                del request
                return ModelResponse(
                    content=(
                        '{"logic_coherence": 9, "answer_exposure": 8, "tool_use": 9, '
                        '"overall_reason": "Good." repeated prose}'
                    )
                )

        result = SftTrajectoryFilter(
            config=FilterConfig(quality_model_alias="quality", simple_model_alias="simple"),
            model_client=MalformedQualityJudge(),
        )._quality_judge(_record("sample_repaired"), "sample_repaired")
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["parser_repaired"])
        self.assertEqual(result["average_score"], 8.6667)

    def test_tool_use_filter_counts_hidden_read_url_failures(self) -> None:
        record = _record("sample_tool_errors")
        record["raw_messages"].extend(
            [
                {
                    "role": "tool",
                    "name": "read_url",
                    "content": json.dumps(
                        {
                            "page_id": "page_a",
                            "title": "",
                            "goal": "inspect",
                            "content": "Unable to read the requested page.",
                        }
                    ),
                },
                {
                    "role": "tool",
                    "name": "read_url",
                    "content": json.dumps(
                        {
                            "page_id": "page_b",
                            "title": "",
                            "goal": "inspect",
                            "content": "Unable to read the requested page.",
                        }
                    ),
                },
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "raw.jsonl"
            output_dir = root / "filtered"
            input_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            fake = RoutingFakeJudge()
            report = filter_jsonl(
                input_path,
                output_dir,
                quality_model_alias="quality",
                simple_model_alias="simple",
                model_client=fake,
            )

        self.assertEqual(report["decision_counts"], {"reject": 1})
        self.assertEqual(report["filtered_ids"]["tool_use_error_ids"], ["sample_tool_errors"])
        self.assertEqual(report["stage_statistics"]["tool_use_errors"], {
            "input_count": 1,
            "filtered_count": 1,
            "survivor_count": 0,
        })
        self.assertEqual(fake.requests, [])

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
                "after_tool_use_errors": 3,
                "after_answer_correctness": 2,
                "after_quality_dimensions": 1,
                "after_quality_average": 1,
            })
            self.assertEqual(report["stage_statistics"], {
                "max_turn": {"input_count": 4, "filtered_count": 1, "survivor_count": 3},
                "tool_use_errors": {"input_count": 3, "filtered_count": 0, "survivor_count": 3},
                "answer_correctness": {"input_count": 3, "filtered_count": 1, "survivor_count": 2},
                "quality_dimensions": {
                    "input_count": 2,
                    "quality_scored_count": 2,
                    "quality_judge_error_count": 0,
                    "dimension_low_score_count": 1,
                    "filtered_count": 1,
                    "survivor_count": 1,
                },
                "quality_average": {"input_count": 1, "filtered_count": 0, "survivor_count": 1},
            })
            self.assertEqual(
                report["quality_score_averages_after_answer_correctness"],
                {
                    "candidate_count_after_answer_correctness": 2,
                    "scored_count": 2,
                    "quality_judge_error_count": 0,
                    "mean_scores": {
                        "logic_coherence": 7.0,
                        "answer_exposure": 8.5,
                        "tool_use": 8.0,
                    },
                },
            )
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
            self.assertEqual(
                report["quality_score_averages_after_answer_correctness"]["mean_scores"],
                {"logic_coherence": 6.5, "answer_exposure": 6.5, "tool_use": 6.8},
            )

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

    def test_cli_report_is_human_readable_and_not_json(self) -> None:
        report = {
            "output_dir": "/tmp/filtered",
            "processed_records": 4,
            "total_available_records": 4,
            "decision_counts": {"keep": 1, "reject": 3},
            "stage_statistics": {
                "max_turn": {"input_count": 4, "filtered_count": 1, "survivor_count": 3},
                "tool_use_errors": {"input_count": 3, "filtered_count": 0, "survivor_count": 3},
                "answer_correctness": {"input_count": 3, "filtered_count": 1, "survivor_count": 2},
                "quality_dimensions": {"input_count": 2, "filtered_count": 1, "survivor_count": 1},
                "quality_average": {"input_count": 1, "filtered_count": 0, "survivor_count": 1},
            },
            "quality_score_averages_after_answer_correctness": {
                "candidate_count_after_answer_correctness": 2,
                "scored_count": 2,
                "quality_judge_error_count": 0,
                "mean_scores": {
                    "logic_coherence": 7.0,
                    "answer_exposure": 8.5,
                    "tool_use": 8.0,
                },
            },
        }
        rendered = _format_cli_report(report)
        self.assertIn("SFT trajectory filtering completed", rendered)
        self.assertIn("Filtering stages", rendered)
        self.assertIn("Max-turn", rendered)
        self.assertIn("Logic coherence mean       : 7.0000", rendered)
        self.assertNotIn('"stage_statistics"', rendered)
        self.assertFalse(rendered.lstrip().startswith("{"))

    def test_workers_parallelize_samples_and_preserve_output_order(self) -> None:
        records = [_record("sample_1"), _record("sample_2"), _record("sample_3")]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "raw.jsonl"
            output_dir = root / "filtered"
            input_path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
            fake = RoutingFakeJudge()
            report = filter_jsonl(
                input_path,
                output_dir,
                quality_model_alias="quality",
                simple_model_alias="simple",
                workers=2,
                model_client=fake,
            )

            self.assertEqual(report["workers"], 2)
            self.assertEqual(report["decision_counts"], {"keep": 3})
            result_lines = (output_dir / "judge_results.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual([json.loads(line)["source_index"] for line in result_lines], [0, 1, 2])
            accepted_lines = (output_dir / "accepted_trajectories.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual([json.loads(line)["sample_id"] for line in accepted_lines], ["sample_1", "sample_2", "sample_3"])

    def test_workers_must_be_positive(self) -> None:
        record = _record("sample_workers")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "raw.jsonl"
            input_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "workers"):
                filter_jsonl(
                    input_path,
                    root / "filtered",
                    quality_model_alias="quality",
                    simple_model_alias="simple",
                    workers=0,
                    model_client=FakeJudge([]),
                )


if __name__ == "__main__":
    unittest.main()
