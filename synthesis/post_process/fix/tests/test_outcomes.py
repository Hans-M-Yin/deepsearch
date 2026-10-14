"""Safe export, terminal quality decisions, and resumable execution failures."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from synthesis.post_process.fix.outcomes import disposition, exportable, should_process
from synthesis.post_process.fix.run import main
from synthesis.post_process.fix.storage import read_json, write_json


class OutcomeTests(unittest.TestCase):
    def test_policy_including_partial_multiwindow_failure(self):
        for kind, expected in [
            ("review_rejected", "quality_rejected"),
            ("global_logic_error", "quality_rejected"),
            ("structure_rejected", "retry_pending"),
            ("selection_exhausted", "retry_pending"),
            ("model_or_tool_error", "retry_pending"),
        ]:
            audit = {
                "status": "rewritten",
                "windows": [
                    {"status": "rewritten"},
                    {"status": "quarantined", "failure_kind": kind},
                ],
            }
            self.assertEqual(disposition(audit), expected)
            self.assertFalse(exportable(audit))
            self.assertEqual(
                should_process({"audit": audit}, resume=True),
                expected == "retry_pending",
            )
        self.assertTrue(should_process(None))
        for status in ["rewritten", "unchanged"]:
            audit = {"status": status, "windows": []}
            self.assertTrue(exportable(audit))
            self.assertFalse(should_process({"audit": audit}, resume=True))

    def test_export_resume_and_new_llm_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [
                {"id": str(i), "conversations": [], "images": []} for i in range(5)
            ]
            source, work, output = root / "input.json", root / "work", root / "out.json"
            write_json(source, records)
            argv = [
                "--input",
                str(source),
                "--workdir",
                str(work),
                "--output",
                str(output),
                "--mode",
                "augment",
                "--record-workers",
                "1",
                "--limit",
                "4",
            ]
            calls, caches = [], []

            def process(
                record, dataset_root, record_root, config, backend, model, cache
            ):
                i = int(record["id"])
                calls.append(i)
                if i == 3:
                    caches.append(model.cache.root)
                if i == 2 or (i == 3 and calls.count(3) == 1):
                    kind = "review_rejected" if i == 2 else "model_or_tool_error"
                    return record, {
                        "status": "quarantined",
                        "windows": [
                            {
                                "status": "quarantined",
                                "failure_kind": kind,
                                "error": "test",
                            }
                        ],
                    }
                return record, {
                    "status": "rewritten" if i else "unchanged",
                    "windows": [],
                }

            with (
                patch(
                    "synthesis.post_process.fix.pipeline.process_record",
                    side_effect=process,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                main(argv)
                self.assertEqual([r["id"] for r in read_json(output)], ["0", "1"])
                self.assertEqual(
                    read_json(work / "summary.json")["unprocessed_indices"], [4]
                )
                self.assertEqual(len(read_json(work / "quality_rejected.json")), 1)
                self.assertEqual(len(read_json(work / "retry_pending.json")), 1)
                main(argv + ["--resume"])
                self.assertEqual(calls, [0, 1, 2, 3, 3])
                self.assertNotEqual(caches[0], caches[1])
                self.assertEqual(read_json(work / "records/3.json")["attempt"], 2)
                self.assertEqual([r["id"] for r in read_json(output)], ["0", "1", "3"])
                self.assertEqual(read_json(work / "retry_pending.json"), [])
                # Expand the selection: completed and rejected records stay cached.
                main(argv[:-2] + ["--resume"])
                self.assertEqual(calls, [0, 1, 2, 3, 3, 4])
                self.assertEqual(
                    [r["id"] for r in read_json(output)], ["0", "1", "3", "4"]
                )
                main(argv[:-2] + ["--resume"])
                self.assertEqual(calls, [0, 1, 2, 3, 3, 4])
