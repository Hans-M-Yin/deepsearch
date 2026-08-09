"""Regression tests for JsonlGraphStore's derived indexes."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from synthesis.store import JsonlGraphStore


class JsonlGraphStoreIndexTests(unittest.TestCase):
    def test_indexes_are_rebuilt_from_existing_jsonl_and_preserve_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = JsonlGraphStore(root)
            store.upsert_node(
                {
                    "node_id": "text-a",
                    "node_type": "text",
                    "title": "A",
                    "source": {"url": "https://example.test/a"},
                    "created_at": "2024-01-01T00:00:00+00:00",
                }
            )
            store.upsert_node(
                {
                    "node_id": "text-b",
                    "node_type": "text",
                    "title": "B",
                    "source": {"url": "https://example.test/b"},
                    "created_at": "2024-01-02T00:00:00+00:00",
                }
            )
            store.upsert_node(
                {
                    "node_id": "image-a",
                    "node_type": "image",
                    "source": {"url": "https://example.test/a"},
                    "created_at": "2024-01-03T00:00:00+00:00",
                }
            )
            store.upsert_evidence(
                {
                    "evidence_id": "evidence-a",
                    "evidence_type": "web_text",
                    "node_ids": ["text-a"],
                    "url": "https://example.test/a",
                }
            )
            store.upsert_evidence(
                {
                    "evidence_id": "evidence-b",
                    "evidence_type": "web_text",
                    "node_ids": ["text-b"],
                    "url": "https://example.test/b",
                }
            )
            store.upsert_edge({
                "edge_id": "edge-a",
                "src_node_id": "text-a",
                "dst_node_id": "text-b",
            })
            store.upsert_edge({
                "edge_id": "edge-b",
                "src_node_id": "text-a",
                "dst_node_id": "image-a",
            })
            store.flush()

            reloaded = JsonlGraphStore(root)
            self.assertEqual(reloaded.count_nodes("text"), 2)
            self.assertEqual(reloaded.count_nodes("image"), 1)
            self.assertEqual(
                [item["node_id"] for item in reloaded.find_nodes_by_source_url(
                    "https://example.test/a"
                )],
                ["text-a", "image-a"],
            )
            self.assertEqual(
                [item["edge_id"] for item in reloaded.edges_from("text-a")],
                ["edge-a", "edge-b"],
            )
            self.assertEqual(
                [item["evidence_id"] for item in reloaded.find_evidence(
                    node_id="text-a",
                    url="https://example.test/a",
                    evidence_type="web_text",
                )],
                ["evidence-a"],
            )
            self.assertEqual(reloaded.latest_node()["node_id"], "image-a")

    def test_flush_appends_delta_and_compact_rebuilds_canonical_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = JsonlGraphStore(root)
            store.upsert_node(
                {
                    "node_id": "text-a",
                    "node_type": "text",
                    "title": "Version 1",
                }
            )
            self.assertTrue(store.flush())
            self.assertFalse((root / "nodes.jsonl").exists())
            self.assertTrue((root / "nodes.delta.jsonl").exists())
            self.assertEqual(store.delta_stats()["nodes"]["records"], 1)

            reloaded = JsonlGraphStore(root)
            self.assertEqual(reloaded.get_node("text-a")["title"], "Version 1")
            reloaded.upsert_node(
                {
                    "node_id": "text-a",
                    "node_type": "text",
                    "title": "Version 2",
                }
            )
            reloaded.flush()
            self.assertEqual(reloaded.get_node("text-a")["title"], "Version 2")
            self.assertTrue(reloaded.compact())
            self.assertTrue((root / "nodes.jsonl").exists())
            self.assertFalse((root / "nodes.delta.jsonl").exists())
            self.assertEqual(JsonlGraphStore(root).get_node("text-a")["title"], "Version 2")

    def test_upsert_updates_and_removes_derived_index_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = JsonlGraphStore(Path(temp_dir))
            store.upsert_node(
                {
                    "node_id": "text-a",
                    "node_type": "text",
                    "source": {"url": "https://example.test/old"},
                }
            )
            store.upsert_edge({
                "edge_id": "edge-a",
                "src_node_id": "text-a",
                "dst_node_id": "text-b",
            })
            store.upsert_evidence({
                "evidence_id": "evidence-a",
                "evidence_type": "web_text",
                "node_ids": ["text-a"],
                "url": "https://example.test/old",
            })

            store.upsert_node(
                {
                    "node_id": "text-a",
                    "node_type": "image",
                    "source": {"url": "https://example.test/new"},
                }
            )
            store.upsert_edge({
                "edge_id": "edge-a",
                "src_node_id": "text-c",
                "dst_node_id": "text-d",
            })
            store.upsert_evidence({
                "evidence_id": "evidence-a",
                "evidence_type": "caption",
                "node_ids": ["text-c"],
                "url": "https://example.test/new",
            })

            self.assertEqual(store.find_nodes_by_source_url("https://example.test/old"), [])
            self.assertEqual(
                store.find_node_by_source_url(
                    "https://example.test/new", node_type="image"
                )["node_id"],
                "text-a",
            )
            self.assertEqual(store.count_nodes("text"), 0)
            self.assertEqual(store.count_nodes("image"), 1)
            self.assertEqual(store.edges_from("text-a"), [])
            self.assertEqual([item["edge_id"] for item in store.edges_from("text-c")], ["edge-a"])
            self.assertEqual(store.find_evidence(node_id="text-a"), [])
            self.assertEqual(
                [item["evidence_id"] for item in store.find_evidence(node_id="text-c")],
                ["evidence-a"],
            )

    def test_reload_repairs_incomplete_delta_tail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = JsonlGraphStore(root)
            store.upsert_node({"node_id": "text-a", "node_type": "text"})
            store.flush()
            delta_path = root / "nodes.delta.jsonl"
            with delta_path.open("ab") as handle:
                handle.write(b'{"node_id":"partial"')

            reloaded = JsonlGraphStore(root)
            self.assertIsNotNone(reloaded.get_node("text-a"))
            self.assertIsNone(reloaded.get_node("partial"))
            repaired_delta = delta_path.read_bytes()
            self.assertTrue(repaired_delta.endswith(b"\n"))
            self.assertNotIn(b"partial", repaired_delta)


if __name__ == "__main__":
    unittest.main()
