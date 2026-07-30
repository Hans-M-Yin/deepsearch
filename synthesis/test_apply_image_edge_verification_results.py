import json
import tempfile
import unittest
from pathlib import Path

from synthesis.post_process.apply_image_edge_verification_results import apply_results
from synthesis.store import JsonlGraphStore


class ApplyImageEdgeVerificationResultsTests(unittest.TestCase):
    def test_applies_latest_checkpoint_record_and_soft_deletes_contradict(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            graph_dir = root / "graph"
            results_path = root / "results.jsonl"
            store = JsonlGraphStore(graph_dir)
            store.upsert_edge(
                {
                    "edge_id": "edge-1",
                    "src_node_id": "image-1",
                    "dst_node_id": "text-1",
                    "edge_type": "image_depicts",
                    "relation": "person in photo",
                    "status": "active",
                    "metadata": {},
                }
            )
            store.flush()
            results_path.write_text(
                "\n".join(
                    [
                        json.dumps({"edge_id": "edge-1", "decision": "support", "reason": "old"}),
                        json.dumps(
                            {
                                "edge_id": "edge-1",
                                "decision": "contradict",
                                "error_type": "wrong_identity",
                                "reason": "new",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            payload = apply_results(
                graph_dir=graph_dir,
                results_jsonl=results_path,
                drop_on="contradict",
                hard_delete=False,
                dry_run=False,
            )
            edge = JsonlGraphStore(graph_dir).get_edge("edge-1")

        self.assertEqual(payload["input_stats"]["superseded_records"], 1)
        self.assertEqual(payload["apply_counts"]["soft_deleted"], 1)
        self.assertEqual(edge["status"], "rejected")
        self.assertEqual(edge["metadata"]["post_verify_image_text"]["reason"], "new")

    def test_skips_an_already_inactive_edge(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            graph_dir = root / "graph"
            results_path = root / "results.jsonl"
            store = JsonlGraphStore(graph_dir)
            store.upsert_edge(
                {
                    "edge_id": "edge-rejected",
                    "src_node_id": "image-1",
                    "dst_node_id": "text-1",
                    "edge_type": "image_depicts",
                    "relation": "person in photo",
                    "status": "rejected",
                    "metadata": {"existing": "preserve"},
                }
            )
            store.flush()
            results_path.write_text(
                json.dumps({"edge_id": "edge-rejected", "decision": "support", "reason": "new"}) + "\n",
                encoding="utf-8",
            )

            payload = apply_results(
                graph_dir=graph_dir,
                results_jsonl=results_path,
                drop_on="contradict",
                hard_delete=False,
                dry_run=False,
            )
            edge = JsonlGraphStore(graph_dir).get_edge("edge-rejected")

        self.assertEqual(payload["apply_counts"]["skipped_inactive_graph_edge"], 1)
        self.assertEqual(edge["status"], "rejected")
        self.assertNotIn("post_verify_image_text", edge["metadata"])


if __name__ == "__main__":
    unittest.main()
