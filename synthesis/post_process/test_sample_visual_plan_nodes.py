import unittest

from synthesis.post_process.sample_visual_plan_nodes import build_failure_rows


class VisualPlanFailureRowsTests(unittest.TestCase):
    def test_keeps_only_plans_without_a_materialized_text_to_image_edge(self) -> None:
        nodes = [
            {"node_id": "text-1", "node_type": "text"},
            {"node_id": "image-1", "node_type": "image"},
        ]
        edges = [
            {
                "edge_id": "edge-1",
                "edge_type": "search_retrieved",
                "src_node_id": "text-1",
                "dst_node_id": "image-1",
                "evidence_refs": [{"evidence_id": "evidence-materialized"}],
            }
        ]
        plans = [
            {
                "plan_id": "plan-materialized",
                "target_evidence_id": "evidence-materialized",
                "target_description": "Materialized plan",
            },
            {
                "plan_id": "plan-failed",
                "target_evidence_id": "evidence-failed",
                "target_description": "Failed plan",
                "node_id": "text-1",
                "node_title": "Source",
                "queries": ["failed plan search query"],
                "reason": "Unique scene",
            },
        ]

        rows = build_failure_rows(plans, nodes, edges)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["plan_id"], "plan-failed")
        self.assertEqual(rows[0]["queries"], ["failed plan search query"])


if __name__ == "__main__":
    unittest.main()
