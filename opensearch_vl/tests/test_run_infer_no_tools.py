"""Regression coverage for no-tools CLI, retry diagnostics and resume."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import pandas as pd
from PIL import Image

import run_infer
import run_infer_no_tools
from opensearch_infer import no_tools_pipeline, pipeline, runners
from synthesis.retry_monitor import get_monitor, tracked_sleep


def _response(text: str = "<thinking>reason</thinking><answer>answer</answer>") -> dict:
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


class NoToolsTests(unittest.TestCase):
    def test_sampling_monitor_config_and_resume_through_cli(self) -> None:
        frame = pd.DataFrame([
            {"id": "done", "question": "done", "images": []},
            {"id": "retry", "question": "retry", "images": []},
            {"id": "new", "question": "new", "images": []},
        ])
        runner = mock.Mock(display_name="test-model", model="test-model")
        runner.base_url = "http://localhost:8000/v1"
        should_fail = True

        def infer(*, contents, system_instruction, cfg):
            text = contents[0]["parts"][-1]["text"]
            if should_fail and text == "retry":
                tracked_sleep(0.001, reason="http_500", tool="llm", error="backend unavailable")
                result = _response("")
                result["inference_error"] = {"status_code": 500, "message": "backend unavailable"}
                return result
            return _response("An answer without tags.")

        runner.infer.side_effect = infer
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data.parquet"
            source.write_bytes(b"mocked dataset")
            output = root / "output"
            output.mkdir()
            (output / "done_trajectory.json").write_text(json.dumps({
                "final_response_text": "<answer>already done</answer>",
            }))
            # Legacy no-tools used to save API failures as normal trajectories.
            (output / "retry_trajectory.json").write_text(json.dumps({
                "turns": [{"error": "previous failure"}], "final_response_text": "",
            }))
            error_log = root / "custom_retry_errors.txt"
            argv = [
                "--data-path", str(source), "--output-dir", str(output),
                "--backend", "api", "--parallel-workers", "2",
                "--temperature", "0.4", "--max-tokens", "321",
                "--top-p", "0.6", "--top-k", "9",
                "--repetition-penalty", "1.2", "--presence-penalty", "0.3",
                "--dynamic-resolution", "--api-key", "private-key",
                "--retry-error-log", str(error_log),
            ]
            with (
                mock.patch.object(runners, "build_runner", return_value=runner),
                mock.patch.object(pd, "read_parquet", return_value=frame),
                mock.patch.object(run_infer, "_git_output", return_value=None),
                mock.patch.object(pipeline.tools, "execute_tool", side_effect=AssertionError("no tools")),
                mock.patch.object(no_tools_pipeline, "_bootstrap_images", wraps=pipeline._bootstrap_images) as bootstrap,
            ):
                with self.assertLogs("run_infer_no_tools", level="INFO") as logs:
                    self.assertEqual(run_infer_no_tools.main(argv), 0)
                self.assertIn("success=1 failure=1 skipped=1", "\n".join(logs.output))
                self.assertEqual(runner.infer.call_count, 2)
                for call in runner.infer.call_args_list:
                    cfg = call.kwargs["cfg"]
                    self.assertEqual(
                        (cfg.temperature, cfg.max_tokens, cfg.top_p, cfg.top_k, cfg.repetition_penalty, cfg.presence_penalty),
                        (0.4, 321, 0.6, 9, 1.2, 0.3),
                    )
                self.assertTrue(all(c.kwargs["dynamic_resolution"] for c in bootstrap.call_args_list))
                self.assertIsNone(get_monitor())
                retry_text = error_log.read_text()
                self.assertIn("retry", retry_text)
                self.assertIn("backend unavailable", retry_text)
                failure = json.loads((output / "failures/retry_failure.json").read_text())
                self.assertEqual(failure["failure_kind"], "inference_error")
                self.assertEqual(run_infer._completed_case_ids(str(output)), {"done", "new"})

                initial = json.loads((output / "run_config.json").read_text())
                self.assertEqual(initial["mode"], "no_tools")
                self.assertEqual(Path(initial["command"]["entrypoint"]).name, "run_infer_no_tools.py")
                self.assertEqual(initial["agent_limits"]["max_turns"], 1)
                self.assertEqual(initial["agent_limits"]["max_tool_calls"], 0)
                self.assertEqual(initial["inference"]["top_k"], 9)
                self.assertNotIn("private-key", (output / "run_config.json").read_text())

                should_fail = False
                with self.assertLogs("run_infer_no_tools", level="INFO") as logs:
                    self.assertEqual(run_infer_no_tools.main(argv), 0)
                self.assertIn("success=1 failure=0 skipped=2", "\n".join(logs.output))
                self.assertEqual(runner.infer.call_count, 3)
                self.assertEqual(run_infer._completed_case_ids(str(output)), {"done", "retry", "new"})
                self.assertEqual(initial, json.loads((output / "run_config.json").read_text()))
                self.assertEqual(len((output / "run_invocations.jsonl").read_text().splitlines()), 2)

    def test_resume_retries_corrupt_failed_and_incomplete_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payloads = {
                "ok": {"status": "completed", "final_response_text": "<answer>a</answer>"},
                "legacy": {"final_response_text": "<answer>b</answer>"},
                "failed": {"status": "failed", "final_response_text": "<answer>a</answer>"},
                "empty": {"status": "completed", "final_response_text": ""},
                "truncated": {"final_response_text": "<answer>partial"},
                "no_tools_plain": {"mode": "no_tools", "status": "completed", "final_response_text": "plain answer"},
                "no_tools_legacy": {"mode": "no_tools", "final_response_text": "plain answer"},
                "no_tools_empty": {"mode": "no_tools", "status": "completed", "final_response_text": "  \n"},
                "api_error": {"final_response_text": "<answer>a</answer>", "turns": [{"response": {"inference_error": {"status_code": 500}}}]},
            }
            for case, payload in payloads.items():
                (root / f"{case}_trajectory.json").write_text(json.dumps(payload))
            (root / "corrupt_trajectory.json").write_text('{"case_id":')
            (root / "failures").mkdir()
            (root / "failures/uncompleted_failure.json").write_text("{}")
            self.assertEqual(
                run_infer._completed_case_ids(directory),
                {"ok", "legacy", "no_tools_plain", "no_tools_legacy"},
            )

    def test_resume_skips_only_root_archived_max_turns_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "failures").mkdir()
            terminal = {"status": "failed", "failure_kind": "max_turns"}
            (root / "archived_failure.json").write_text(json.dumps(terminal))
            (root / "failures/retry_failure.json").write_text(json.dumps(terminal))
            (root / "connection_failure.json").write_text(json.dumps(
                {"status": "failed", "failure_kind": "inference_error"}
            ))
            (root / "corrupt_failure.json").write_text("{")
            self.assertEqual(run_infer._completed_case_ids(directory), {"archived"})
            self.assertEqual(list(root.glob("*_trajectory.json")), [])

    def test_pipeline_exception_and_empty_response_are_retryable(self) -> None:
        for result, expected_kind in [
            (RuntimeError("connection lost"), "inference_error"),
            (_response(""), "invalid_response"),
            (_response("  \n"), "invalid_response"),
            ({"candidates": []}, "invalid_response"),
        ]:
            with self.subTest(kind=expected_kind), tempfile.TemporaryDirectory() as directory:
                runner = mock.Mock(display_name="test-model")
                if isinstance(result, Exception):
                    runner.infer.side_effect = result
                else:
                    runner.infer.return_value = result
                timing = pipeline.InferenceTiming()
                with self.assertRaises(RuntimeError):
                    no_tools_pipeline.process_single_case(
                        {"id": "failure", "question": "question", "images": []},
                        runner, directory, 0, timing=timing,
                    )
                self.assertFalse(Path(directory, "failure_trajectory.json").exists())
                saved = json.loads(Path(directory, "failures/failure_failure.json").read_text())
                self.assertEqual(saved["status"], "failed")
                self.assertEqual(saved["failure_kind"], expected_kind)
                self.assertEqual(saved["max_turns"], 1)
                self.assertEqual(saved["max_tool_calls"], 0)
                self.assertEqual(timing._modules["llm"]["calls"], 1)

    def test_nonempty_response_without_answer_tag_is_saved_and_resumable(self) -> None:
        for text in ("plain answer", "<thinking>unfinished</thinking>", "<answer>partial"):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as directory:
                runner = mock.Mock(display_name="test-model")
                runner.infer.return_value = _response(text)
                trajectory = no_tools_pipeline.process_single_case(
                    {"id": "plain", "question": "question", "images": []},
                    runner, directory, 0,
                )
                saved = json.loads(Path(directory, "plain_trajectory.json").read_text())
                self.assertEqual(trajectory["status"], "completed")
                self.assertEqual(saved["final_response_text"], text)
                self.assertEqual(saved["turns"][0]["response_text"], text)
                self.assertFalse(Path(directory, "failures/plain_failure.json").exists())
                self.assertEqual(run_infer._completed_case_ids(directory), {"plain"})

    def test_multi_image_dynamic_resolution_in_single_turn(self) -> None:
        raw = io.BytesIO()
        Image.new("RGB", (3200, 2000), (20, 40, 60)).save(raw, format="PNG")
        runner = mock.Mock(display_name="test-model")
        runner.infer.return_value = _response()
        with tempfile.TemporaryDirectory() as directory:
            trajectory = no_tools_pipeline.process_single_case(
                {"id": "images", "question": "question", "images": [{"bytes": raw.getvalue()}] * 2},
                runner, directory, 0, dynamic_resolution=True,
            )
            parts = runner.infer.call_args.kwargs["contents"][0]["parts"]
            self.assertEqual(len(parts), 3)
            self.assertEqual(parts[-1]["text"].count("<image>"), 2)
            for part in parts[:-1]:
                with Image.open(io.BytesIO(base64.b64decode(part["inline_data"]["data"]))) as image:
                    self.assertEqual(image.size, (1600, 1000))
            runner.infer.assert_called_once()
            self.assertEqual(trajectory["status"], "completed")


if __name__ == "__main__":
    unittest.main()
