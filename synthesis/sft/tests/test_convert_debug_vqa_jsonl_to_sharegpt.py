from __future__ import annotations

import base64
import importlib.util
import json
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest import mock

from PIL import Image


SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "convert_debug_vqa_jsonl_to_sharegpt.py"
SPEC = importlib.util.spec_from_file_location("debug_vqa_sharegpt_converter", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
CONVERTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CONVERTER
SPEC.loader.exec_module(CONVERTER)


def _data_url(color: tuple[int, int, int]) -> str:
    buffer = BytesIO()
    Image.new("RGB", (4, 3), color=color).save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


class ConvertDebugVqaSharegptTest(unittest.TestCase):
    def test_remote_image_download_retries_timeout(self) -> None:
        class FakeResponse:
            status_code = 200
            headers = {"Content-Type": "image/png"}
            content = base64.b64decode(_data_url((1, 2, 3)).split(",", 1)[1])

            def raise_for_status(self) -> None:
                return None

            def close(self) -> None:
                return None

        with (
            mock.patch.object(
                CONVERTER.requests,
                "get",
                side_effect=[CONVERTER.requests.Timeout("timed out"), FakeResponse()],
            ) as get,
            mock.patch.object(CONVERTER.time, "sleep") as sleep,
        ):
            materialized = CONVERTER._materialize_image(
                "https://images.example/item.png",
                base_dir=Path("."),
            )

        self.assertEqual(get.call_count, 2)
        sleep.assert_called_once_with(5)
        self.assertEqual(materialized["mime_type"], "image/png")

    def test_rebuilds_question_normalizes_react_and_materializes_images(self) -> None:
        input_image = _data_url((10, 20, 30))
        read_url_image = _data_url((40, 50, 60))
        record = {
            "question_id": "q1",
            "sample_id": "sample1",
            "path_id": "path1",
            "question": "What is in the picture?",
            "gold_answer": "a blue square",
            "extracted_answer": "a blue square",
            "answer_judge": {"is_correct": True},
            "input_images": [{"image_url": input_image}],
            "raw_messages": [
                {"role": "system", "content": "internal system"},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Question: What is in the picture?\nAnswer: a blue square\nPrivate reference facts for verification only: leak",
                        },
                        {"type": "image_url", "image_url": {"url": input_image}},
                    ],
                },
                {
                    "role": "assistant",
                    "content": "I need visual evidence.\n<action>\n{\"tool_name\": \"read_url\", \"arguments\": {\"resource_id\": \"img_1\", \"goal\": \"inspect color\"}}\n</action>",
                },
                {"role": "tool", "name": "read_url", "content": '{"image_id":"img_1"}'},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "image_id=img_1"},
                        {"type": "image_url", "image_url": {"url": read_url_image}},
                    ],
                },
                {
                    "role": "assistant",
                    "content": "The image is clear.\n<action>\n{\"tool_name\": \"finish\", \"arguments\": {\"answer\": \"a blue square\"}}\n</action>",
                },
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "raw.jsonl"
            output_path = temp_path / "my_agent_sft"
            input_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

            counts = CONVERTER.convert_file(input_path, output_path)
            self.assertEqual(counts["written_records"], 1)
            rows = json.loads((output_path / "trajectories_sharegpt.json").read_text(encoding="utf-8"))
            row = rows[0]
            self.assertEqual(row["conversations"][0]["value"], "What is in the picture?\n<image>")
            self.assertNotIn("gold_answer", json.dumps(row, ensure_ascii=False))
            self.assertNotIn("Private reference facts", row["conversations"][0]["value"])
            assistant_values = [item["value"] for item in row["conversations"] if item["from"] == "gpt"]
            self.assertIn("<tool_call>", assistant_values[0])
            self.assertIn('"name": "read_url"', assistant_values[0])
            self.assertIn("<answer>\na blue square\n</answer>", assistant_values[-1])
            observations = [item["value"] for item in row["conversations"] if item["from"] == "observation"]
            self.assertEqual(observations[0], '{\n  "image_id": "img_1"\n}\nThe image is shown below:\n<image>')
            self.assertNotIn("<tool_response>", observations[0])
            self.assertEqual(row["conversations"][0]["value"].count("<image>") + 1, len(row["images"]))
            self.assertEqual(len(row["images"]), 2)
            for relative_path in row["images"]:
                self.assertTrue((output_path / relative_path).is_file())
            dataset_info = json.loads((output_path / "dataset_info.json").read_text(encoding="utf-8"))
            self.assertEqual(dataset_info["opensearch_vl_sft"]["formatting"], "sharegpt")

    def test_incorrect_records_are_rejected_by_default(self) -> None:
        record = {
            "question": "Question",
            "answer_judge": {"is_correct": False},
            "raw_messages": [],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "raw.jsonl"
            output_path = temp_path / "my_agent_sft"
            input_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            counts = CONVERTER.convert_file(input_path, output_path)
            self.assertEqual(counts["written_records"], 0)
            self.assertEqual(counts["skipped_incorrect"], 1)
            rejected = (output_path / ".metadata" / "rejected.jsonl").read_text(encoding="utf-8")
            self.assertIn("is_correct is false", rejected)

    def test_native_tool_calls_are_normalized(self) -> None:
        record = {
            "question": "Question",
            "extracted_answer": "Answer",
            "answer_judge": {"is_correct": True},
            "raw_messages": [
                {"role": "user", "content": "Question"},
                {
                    "role": "assistant",
                    "content": "Need evidence",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "t2t_search", "arguments": '{"query":"query"}'},
                        }
                    ],
                },
                {"role": "tool", "name": "t2t_search", "content": {"ok": True, "results": []}},
                {"role": "assistant", "content": "<thinking>done</thinking><answer>Answer</answer>"},
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "raw.jsonl"
            output_path = temp_path / "my_agent_sft"
            input_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            CONVERTER.convert_file(input_path, output_path)
            row = json.loads((output_path / "trajectories_sharegpt.json").read_text(encoding="utf-8"))[0]
            self.assertIn('"name": "t2t_search"', row["conversations"][1]["value"])
            self.assertEqual(row["conversations"][2]["from"], "observation")
            self.assertIn('"ok": true', row["conversations"][2]["value"])


if __name__ == "__main__":
    unittest.main()
