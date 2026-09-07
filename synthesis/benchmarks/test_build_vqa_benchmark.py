"""Unit tests for the benchmark builder's deterministic structural logic."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from synthesis.benchmarks import build_vqa_benchmark as builder


class BuildVqaBenchmarkTest(unittest.TestCase):
    def test_classification(self) -> None:
        self.assertEqual(
            builder.classify_image_path(["image", "text", "text"]),
            "image-start",
        )
        self.assertEqual(
            builder.classify_image_path(["text", "text", "image"]),
            "image-end",
        )
        self.assertEqual(
            builder.classify_image_path(["image", "text", "image"]),
            "image-interleave",
        )
        self.assertEqual(
            builder.classify_image_path(["text", "image", "text"]),
            "image-interleave",
        )
        self.assertIsNone(builder.classify_image_path(["text", "text"]))
        self.assertEqual(
            builder._target_counts(300, (2, 3, 4)),
            {2: 100, 3: 100, 4: 100},
        )

    def test_builds_source_like_vqa_dir_with_balanced_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            output = root / "benchmark"
            source.mkdir()
            questions = []
            samples = []
            structures = {
                "image-start": ["image", "text", "text"],
                "image-end": ["text", "text", "image"],
                "image-interleave": ["image", "text", "image"],
            }
            index = 0
            for node_types in structures.values():
                for hop_count in (2, 3):
                    for duplicate in range(2):
                        index += 1
                        sample_id = f"s_{index:03d}"
                        path_id = f"p_{index:03d}"
                        questions.append(
                            {
                                "question_id": f"q_{index:03d}",
                                "sample_id": sample_id,
                                "path_id": path_id,
                                "status": "verified",
                                "question": f"Question {index} {duplicate}",
                                "answer": f"Answer {index}",
                                "image_url": f"https://example.invalid/{index}.png",
                            }
                        )
                        samples.append(
                            {
                                "sample_id": sample_id,
                                "status": "verified",
                                "path": {
                                    "path_id": path_id,
                                    "edge_ids": [f"e_{index}_{n}" for n in range(hop_count)],
                                    "node_types": node_types,
                                    "trajectory": {"hop_count": hop_count},
                                },
                            }
                        )

            for path, records in (
                (source / "questions.jsonl", questions),
                (source / "samples.jsonl", samples),
            ):
                path.write_text(
                    "".join(json.dumps(record) + "\n" for record in records),
                    encoding="utf-8",
                )
            emit_index = root / "emit_index.txt"
            emit_index.write_text("q_001\n\n", encoding="utf-8")

            def fake_quality_judge(
                records,
                _samples_by_id,
                output_path,
                **_kwargs,
            ):
                results = {
                    record["id"]: {
                        "case_id": record["id"],
                        "source_index": record["source_index"],
                        "parse_ok": True,
                        "decision": "accept",
                        "reject_categories": [],
                        "reasons": [],
                    }
                    for record in records
                }
                builder._jsonl_write(output_path, results.values())
                return results

            with patch.object(builder, "_run_quality_judge", fake_quality_judge):
                exit_code = builder.main(
                    [
                        "--vqa-dir",
                        str(source),
                        "--output-dir",
                        str(output),
                        "--emit-index",
                        str(emit_index),
                        "--category-count",
                        "2",
                        "--hop-counts",
                        "2",
                        "3",
                        "--quality-judge-model-alias",
                        "test-model",
                        "--quality-judge-batch-size",
                        "1",
                    ]
                )

            self.assertEqual(exit_code, 0)
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["selected_total"], 6)
            self.assertEqual(report["emit_index"]["question_id_count"], 1)
            self.assertEqual(report["emit_index"]["skipped_question_count"], 1)
            self.assertEqual(report["input_counts"]["quality_judged"], 6)
            self.assertEqual(report["input_counts"]["quality_not_judged_after_quota"], 5)
            self.assertEqual(
                report["selected_counts"],
                {
                    "image-start/2hop": 1,
                    "image-start/3hop": 1,
                    "image-end/2hop": 1,
                    "image-end/3hop": 1,
                    "image-interleave/2hop": 1,
                    "image-interleave/3hop": 1,
                },
            )
            self.assertEqual(
                len((output / "questions.jsonl").read_text(encoding="utf-8").splitlines()),
                6,
            )
            self.assertEqual(
                len((output / "samples.jsonl").read_text(encoding="utf-8").splitlines()),
                6,
            )


if __name__ == "__main__":
    unittest.main()
