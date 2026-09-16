from __future__ import annotations

import tempfile
import unittest
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys

MODULE_PATH = Path(__file__).with_name("build_eval_difficulty_augmented_rl_dataset.py")
SPEC = importlib.util.spec_from_file_location("build_eval_difficulty_augmented_rl_dataset", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

EvalVote = MODULE.EvalVote
_allocate_strata = MODULE._allocate_strata
_decode_bytes = MODULE._decode_bytes
build_difficulty_records = MODULE.build_difficulty_records
build_benchmark_row = MODULE.build_benchmark_row
select_medium_hard = MODULE.select_medium_hard


def _vote(version: str, benchmark: str, case_id: str, acc: int) -> EvalVote:
    return EvalVote(
        version=version,
        benchmark=benchmark,
        case_id=case_id,
        acc=acc,
        question=f"question {case_id}",
        answer=f"answer {case_id}",
        report_dir=Path(tempfile.gettempdir()),
    )


class DifficultySelectionTest(unittest.TestCase):
    def test_classifies_distinct_version_votes(self) -> None:
        votes = {
            ("a", "medium"): {
                "SFT3": _vote("SFT3", "a", "medium", 0),
                "SFT7": _vote("SFT7", "a", "medium", 1),
            },
            ("a", "hard"): {
                "SFT3": _vote("SFT3", "a", "hard", 0),
                "SFT7": _vote("SFT7", "a", "hard", 0),
            },
            ("a", "easy"): {
                "SFT3": _vote("SFT3", "a", "easy", 1),
            },
        }
        records, conflicts = build_difficulty_records(votes)
        labels = {record["case_id"]: record["difficulty"] for record in records}
        self.assertEqual(labels, {"easy": "easy", "hard": "hard", "medium": "medium"})
        self.assertEqual(conflicts, {})

    def test_selects_exact_three_to_two_ratio(self) -> None:
        records = []
        for difficulty, count in (("medium", 10), ("hard", 10)):
            for index in range(count):
                records.append(
                    {
                        "benchmark": "a" if index % 2 else "b",
                        "case_id": f"{difficulty}-{index}",
                        "difficulty": difficulty,
                        "observed_count": 2,
                        "conflicts": [],
                    }
                )
        selected = select_medium_hard(records, target_count=10, seed=7)
        self.assertEqual(Counter(record["difficulty"] for record in selected), {"medium": 6, "hard": 4})

    def test_stratified_allocation_sums_to_target(self) -> None:
        records = [
            {"benchmark": "large"} for _ in range(8)
        ] + [{"benchmark": "small"} for _ in range(2)]
        self.assertEqual(_allocate_strata(records, 5), {"large": 4, "small": 1})

    def test_decodes_python_bytes_representation(self) -> None:
        self.assertEqual(_decode_bytes("b'abc\\x00'"), b"abc\x00")

    def test_builds_valid_text_only_benchmark_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case_id = "text-only"
            (root / f"{case_id}_trajectory.json").write_text(
                json.dumps(
                    {
                        "case_idx": 1,
                        "original_data": {
                            "question_id": case_id,
                            "question": "When was it released?",
                            "answer": "2024-12",
                            "images": [],
                            "source_metadata": {"query_image": None},
                        },
                    }
                ),
                encoding="utf-8",
            )
            record = {
                "benchmark": "mmsearch",
                "case_id": case_id,
                "question": "When was it released?",
                "answer": "2024-12",
                "difficulty": "hard",
                "observed_count": 1,
                "correct_count": 0,
                "correct_rate": 0.0,
                "coverage_confidence": "single_observation",
                "version_results": {
                    "SFT12": {"report_dir": str(root), "acc": 0}
                },
            }
            row = build_benchmark_row(record, images_dir=root / "images")

        self.assertEqual(row["images"], [])
        self.assertEqual(row["extra_info"]["source_image_url"], "")
        self.assertEqual(row["extra_info"]["sample_metadata"]["input_image_url"], "")


if __name__ == "__main__":
    unittest.main()
