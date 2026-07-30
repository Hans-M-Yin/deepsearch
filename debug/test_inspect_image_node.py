from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from debug.inspect_image_node import build_report, print_report
from synthesis.store import JsonlGraphStore


class InspectImageNodeUniqueStateTest(unittest.TestCase):
    def test_report_and_text_output_include_unique_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_dir = Path(tmpdir)
            store = JsonlGraphStore(graph_dir)
            store.upsert_node(
                {
                    "node_id": "image_1",
                    "node_type": "image",
                    "title": "Example image",
                    "unique_state": "semi-unique",
                    "metadata": {"search_query": "Example image query"},
                }
            )
            store.flush()

            report = build_report(graph_dir, "image_1")
            self.assertEqual(report["image_node"]["unique_state"], "semi-unique")

            output = io.StringIO()
            with redirect_stdout(output):
                print_report(report)
            self.assertIn("Unique state: semi-unique", output.getvalue())

    def test_missing_unique_state_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_dir = Path(tmpdir)
            store = JsonlGraphStore(graph_dir)
            store.upsert_node({"node_id": "image_1", "node_type": "image"})
            store.flush()

            report = build_report(graph_dir, "image_1")
            self.assertIsNone(report["image_node"]["unique_state"])

            output = io.StringIO()
            with redirect_stdout(output):
                print_report(report)
            self.assertIn("Unique state: <missing>", output.getvalue())


if __name__ == "__main__":
    unittest.main()
