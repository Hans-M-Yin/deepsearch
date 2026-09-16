"""Tests for reproducibility metadata written by ``run_infer``."""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
import unittest
from pathlib import Path

import run_infer


class _FakeRunner:
    display_name = "served-model"
    model = "served-model"
    base_url = "http://localhost:8000/v1"


def _args(output_dir: str, data_path: str, temperature: float = 0.7) -> argparse.Namespace:
    return argparse.Namespace(
        output_dir=output_dir,
        data_path=data_path,
        api_key="very-secret-key",
        temperature=temperature,
        max_tokens=4096,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.0,
        presence_penalty=1.5,
    )


class RunInferConfigTests(unittest.TestCase):
    def test_saves_redacted_initial_config_and_invocation_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_path = root / "dataset.parquet"
            data_path.write_bytes(b"placeholder")
            output_dir = root / "outputs"
            logger = logging.getLogger("test_run_infer_config")
            argv = [
                "--data-path",
                str(data_path),
                "--output-dir",
                str(output_dir),
                "--api-key",
                "very-secret-key",
                "--temperature=0.7",
            ]

            run_infer._persist_run_configuration(
                _args(str(output_dir), str(data_path)), argv, _FakeRunner(), logger
            )
            run_infer._persist_run_configuration(
                _args(str(output_dir), str(data_path), temperature=0.0),
                argv,
                _FakeRunner(),
                logger,
            )

            initial = json.loads((output_dir / "run_config.json").read_text())
            history = [
                json.loads(line)
                for line in (output_dir / "run_invocations.jsonl").read_text().splitlines()
            ]

            self.assertEqual(initial["arguments"]["api_key"], "<redacted>")
            self.assertEqual(initial["command"]["argv"][5], "<redacted>")
            self.assertEqual(initial["inference"]["temperature"], 0.7)
            self.assertTrue(initial["dataset"]["exists"])
            self.assertEqual(initial["runner"]["model"], "served-model")
            self.assertEqual(len(history), 2)
            self.assertEqual(history[1]["inference"]["temperature"], 0.0)
            self.assertNotIn("very-secret-key", (output_dir / "run_config.json").read_text())


if __name__ == "__main__":
    unittest.main()
