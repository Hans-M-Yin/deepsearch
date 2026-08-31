from __future__ import annotations

import tempfile
import unittest
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from synthesis.post_process import build_refined_training_dataset as pipeline
from synthesis.post_process.build_refined_training_dataset import (
    _materialize_record_images,
    _repair_jsonl_append_tail,
)
from synthesis.post_process.expand_trajectory_reasoning import _verification_messages_to_trajectory


class RefinedTrainingDatasetTests(unittest.TestCase):
    def test_new_verification_image_is_isolated_and_original_is_referenced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_dataset = root / "source_dataset"
            (source_dataset / "images").mkdir(parents=True)
            original_image = source_dataset / "images" / "original.jpg"
            original_image.write_bytes(b"source-image")
            downloaded_image = root / "runtime" / "verification.png"
            downloaded_image.parent.mkdir(parents=True)
            downloaded_image.write_bytes(b"new-verification-image")
            output_dir = root / "refined_dataset"

            original_record = {
                "images": ["images/original.jpg"],
                "conversations": [{"from": "human", "value": "Q <image>"}],
            }
            rewritten_record = {
                "images": ["images/original.jpg", str(downloaded_image)],
                "conversations": [
                    {"from": "human", "value": "Q <image>"},
                    {"from": "observation", "value": "downloaded <image>"},
                ],
            }

            materialized, audit = _materialize_record_images(
                rewritten_record,
                original_record,
                source_dataset_dir=source_dataset,
                output_dir=output_dir,
                verification_dir=output_dir / "images" / "verification",
                copy_new_images=True,
            )

            self.assertEqual(materialized["images"][0], "../source_dataset/images/original.jpg")
            self.assertTrue(materialized["images"][1].startswith("images/verification/"))
            self.assertTrue((output_dir / materialized["images"][1]).is_file())
            self.assertEqual(audit["new_verification_image_count"], 1)
            self.assertTrue(audit["image_alignment_ok"])

    def test_verification_image_placeholders_match_all_downloaded_assets(self) -> None:
        result = SimpleNamespace(
            messages=[
                {"role": "user", "content": "verification task"},
                {"role": "tool", "content": "tool image output"},
            ],
            tool_results=[
                SimpleNamespace(new_images={"img_1": "/tmp/one.jpg", "img_2": "/tmp/two.jpg"})
            ],
        )

        messages, image_paths = _verification_messages_to_trajectory(result)

        self.assertEqual(image_paths, ["/tmp/one.jpg", "/tmp/two.jpg"])
        self.assertEqual(messages[0]["value"].count("<image>"), 2)

    def test_resume_repairs_only_an_interrupted_jsonl_tail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "audit.jsonl"
            path.write_bytes(b'{"id":"valid"}\n{"id":"interrupted"')

            self.assertEqual(_repair_jsonl_append_tail(path), "truncated_invalid_final_line")
            self.assertEqual(path.read_text(encoding="utf-8"), '{"id":"valid"}\n')

    def test_pipeline_writes_new_jsonl_without_modifying_source_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_dir = root / "source_dataset"
            (source_dir / "images").mkdir(parents=True)
            (source_dir / "images" / "original.jpg").write_bytes(b"source-image")
            source = source_dir / "trajectories.jsonl"
            records = [
                {
                    "id": f"sample_{index}",
                    "system": "system",
                    "images": ["images/original.jpg"],
                    "conversations": [
                        {"from": "human", "value": "Question <image>"},
                        {"from": "gpt", "value": "<thinking>Reasoning.</thinking><answer>Answer.</answer>"},
                    ],
                }
                for index in range(1, 4)
            ]
            source_text = "".join(json.dumps(record) + "\n" for record in records)
            source.write_text(source_text, encoding="utf-8")
            output_dir = root / "refined_dataset"

            def fake_filter(item, model_alias, max_tokens):
                index, record = item
                verdict = {
                    "record_index": index,
                    "record_id": record["id"],
                    "decision": "keep",
                    "leakage": False,
                    "confidence": 1.0,
                    "leakage_types": [],
                    "turns": [],
                    "evidence": [],
                    "reason": "clean",
                }
                return verdict, {"ok": True}

            def fake_rewrite(item, **kwargs):
                manifest, source_record = item
                # Let later records finish first to exercise the writer's
                # source-order reassembly buffer.
                if manifest["record_index"] == 0:
                    time.sleep(0.03)
                rewrite_audit = {
                    "record_index": manifest["record_index"],
                    "record_id": manifest["record_id"],
                    "status": "ok",
                    "assistant_turns": 1,
                    "changed_turns": 0,
                    "verification_requests": 0,
                    "integrated_verifications": 0,
                    "changes": [],
                }
                return manifest, source_record, rewrite_audit

            argv = [
                "build_refined_training_dataset.py",
                "--input",
                str(source),
                "--output-dir",
                str(output_dir),
                "--filter-model-alias",
                "filter",
                "--rewrite-model-alias",
                "writer",
                "--filter-workers",
                "2",
                "--rewrite-workers",
                "2",
                "--queue-size",
                "1",
            ]
            with (
                mock.patch.object(pipeline, "_filter_one", side_effect=fake_filter),
                mock.patch.object(pipeline, "_rewrite_one", side_effect=fake_rewrite),
                mock.patch("sys.argv", argv),
            ):
                self.assertEqual(pipeline.main(), 0)

            source_after = source.read_text(encoding="utf-8")
            self.assertEqual(source_after, source_text)
            final_records = [
                json.loads(line)
                for line in (output_dir / "trajectories_train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([record["id"] for record in final_records], ["sample_1", "sample_2", "sample_3"])
            self.assertEqual(final_records[0]["images"], ["../source_dataset/images/original.jpg"])
            self.assertTrue((output_dir / "dataset_info.json").is_file())
            self.assertTrue((output_dir / "filter_audit.jsonl").is_file())
            self.assertTrue((output_dir / "rewrite_audit.jsonl").is_file())
            self.assertTrue((output_dir / "pipeline_audit.jsonl").is_file())

    def test_resume_skips_already_committed_trajectories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_dir = root / "source_dataset"
            source_dir.mkdir()
            source = source_dir / "trajectories.jsonl"
            records = [
                {
                    "id": f"sample_{index}",
                    "conversations": [
                        {"from": "human", "value": "Question"},
                        {"from": "gpt", "value": "<thinking>Reasoning.</thinking><answer>Answer.</answer>"},
                    ],
                }
                for index in range(1, 3)
            ]
            source.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            output_dir = root / "refined_dataset"
            calls: list[tuple[str, int]] = []

            def fake_filter(item, model_alias, max_tokens):
                index, record = item
                calls.append(("filter", index))
                return {
                    "record_index": index,
                    "record_id": record["id"],
                    "decision": "keep",
                    "leakage": False,
                    "confidence": 1.0,
                    "leakage_types": [],
                    "turns": [],
                    "evidence": [],
                    "reason": "clean",
                }, {"ok": True}

            def fake_rewrite(item, **kwargs):
                manifest, record = item
                index = manifest["record_index"]
                calls.append(("rewrite", index))
                return manifest, record, {
                    "record_index": index,
                    "record_id": manifest["record_id"],
                    "status": "ok",
                    "assistant_turns": 1,
                    "changed_turns": 0,
                    "verification_requests": 0,
                    "integrated_verifications": 0,
                    "changes": [],
                }

            common = [
                "build_refined_training_dataset.py",
                "--input",
                str(source),
                "--output-dir",
                str(output_dir),
                "--filter-model-alias",
                "filter",
                "--rewrite-model-alias",
                "writer",
            ]
            with (
                mock.patch.object(pipeline, "_filter_one", side_effect=fake_filter),
                mock.patch.object(pipeline, "_rewrite_one", side_effect=fake_rewrite),
                mock.patch("sys.argv", [*common, "--limit", "1"]),
            ):
                self.assertEqual(pipeline.main(), 0)
            self.assertEqual(calls, [("filter", 0), ("rewrite", 0)])

            calls.clear()
            with (
                mock.patch.object(pipeline, "_filter_one", side_effect=fake_filter),
                mock.patch.object(pipeline, "_rewrite_one", side_effect=fake_rewrite),
                mock.patch("sys.argv", [*common, "--resume"]),
            ):
                self.assertEqual(pipeline.main(), 0)

            self.assertEqual(calls, [("filter", 1), ("rewrite", 1)])
            final_records = [
                json.loads(line)
                for line in (output_dir / "trajectories_train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([record["id"] for record in final_records], ["sample_1", "sample_2"])
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["resumed_terminal_records"], 1)
            self.assertEqual(summary["training_records"], 2)


if __name__ == "__main__":
    unittest.main()
