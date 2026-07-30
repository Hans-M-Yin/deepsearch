from __future__ import annotations

import unittest
from pathlib import Path

from debug.debug_image_state_full import _unique_state_stats, build_report


class UniqueStateStatsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.nodes = [
            {"node_id": "image_unique_1", "node_type": "image", "unique_state": "unique"},
            {"node_id": "image_unique_2", "node_type": "image", "unique_state": "unique"},
            {"node_id": "image_semi", "node_type": "image", "unique_state": "semi-unique"},
            {"node_id": "image_non", "node_type": "image", "unique_state": "no-unique"},
            {"node_id": "image_wiki", "node_type": "image", "unique_state": "wiki_inline"},
            {"node_id": "image_missing", "node_type": "image"},
            {"node_id": "text_a", "node_type": "text"},
            {"node_id": "text_b", "node_type": "text"},
        ]
        self.edges = [
            {
                "edge_id": "edge_1",
                "edge_type": "image_depicts",
                "src_node_id": "image_unique_1",
                "dst_node_id": "text_a",
                "status": "active",
            },
            {
                "edge_id": "edge_2",
                "edge_type": "image_depicts",
                "src_node_id": "image_unique_1",
                "dst_node_id": "text_b",
                "status": "soft_deleted",
            },
            {
                "edge_id": "edge_3",
                "edge_type": "image_depicts",
                "src_node_id": "image_semi",
                "dst_node_id": "text_a",
            },
            {
                "edge_id": "edge_4",
                "edge_type": "image_depicts",
                "src_node_id": "image_missing",
                "dst_node_id": "text_a",
            },
            # Not image -> text, so it must not be included.
            {
                "edge_id": "edge_5",
                "edge_type": "image_depicts",
                "src_node_id": "image_non",
                "dst_node_id": "image_wiki",
            },
            {
                "edge_id": "edge_6",
                "edge_type": "search_retrieved",
                "src_node_id": "text_a",
                "dst_node_id": "image_unique_2",
            },
        ]

    def test_counts_nodes_and_image_to_text_edges_by_unique_state(self) -> None:
        images = [node for node in self.nodes if node["node_type"] == "image"]
        nodes_by_id = {node["node_id"]: node for node in self.nodes}
        summary = _unique_state_stats(images, self.edges, nodes_by_id)
        by_state = summary["by_unique_state"]

        self.assertEqual(by_state["unique"]["image_node_count"], 2)
        self.assertEqual(by_state["unique"]["active_image_to_text_edge_count"], 1)
        self.assertEqual(by_state["unique"]["inactive_image_to_text_edge_count"], 1)
        self.assertEqual(by_state["unique"]["all_status_image_to_text_edge_count"], 2)
        self.assertEqual(by_state["semi-unique"]["image_node_count"], 1)
        self.assertEqual(by_state["semi-unique"]["active_image_to_text_edge_count"], 1)
        self.assertEqual(by_state["no-unique"]["all_status_image_to_text_edge_count"], 0)
        self.assertEqual(by_state["wiki_inline"]["image_node_count"], 1)
        self.assertEqual(by_state["missing"]["image_node_count"], 1)
        self.assertEqual(by_state["missing"]["active_image_to_text_edge_count"], 1)
        self.assertEqual(summary["totals"]["image_node_count"], 6)
        self.assertEqual(summary["totals"]["all_status_image_to_text_edge_count"], 4)

    def test_report_exposes_unique_state_summary(self) -> None:
        report = build_report(
            graph_dir=Path("/tmp/example"),
            nodes=self.nodes,
            edges=self.edges,
            plans=[],
            state=None,
        )
        self.assertIn("unique_state_summary", report)
        self.assertEqual(
            report["unique_state_summary"]["by_unique_state"]["unique"]["image_node_count"],
            2,
        )


if __name__ == "__main__":
    unittest.main()
