import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

from synthesis.post_process.verify_image_text_edges import (
    _grounding_entity_counts,
    _append_checkpoint_result,
    _edge_has_post_verification,
    _emit_image_node_complete_status,
    _load_checkpoint_results,
    _sample_image_node_ids,
    _soft_delete_verified_edge,
    _restore_verified_edge,
    _restore_post_verify_edges,
    _verify_worker_metadata,
)
from synthesis.store import JsonlGraphStore
from synthesis.vqa.graph_view import GraphView


class ImageTextEdgeVerificationSamplingTests(unittest.TestCase):
    def test_samples_image_nodes_reproducibly_and_counts_grounded_entities(self) -> None:
        edges = [
            {"src_node_id": "image-c", "edge_id": "edge-1"},
            {"src_node_id": "image-a", "edge_id": "edge-2"},
            {"src_node_id": "image-b", "edge_id": "edge-3"},
            {"src_node_id": "image-a", "edge_id": "edge-4"},
        ]
        selected = _sample_image_node_ids(edges, max_image_nodes=2, random_seed=7)

        self.assertEqual(selected, _sample_image_node_ids(edges, max_image_nodes=2, random_seed=7))
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected, sorted(selected))

        with tempfile.TemporaryDirectory() as temp_dir:
            store = JsonlGraphStore(Path(temp_dir))
            store.upsert_node(
                {
                    "node_id": selected[0],
                    "node_type": "image",
                    "metadata": {"grounded_entities": [{"name": "A"}, {"name": "B"}]},
                }
            )
            store.upsert_node(
                {
                    "node_id": selected[1],
                    "node_type": "image",
                    "metadata": {"grounded_entities": [{"name": "C"}]},
                }
            )

            self.assertEqual(
                _grounding_entity_counts(store, selected),
                {"grounded_entity_count": 3, "image_node_count_with_grounded_entities": 2},
            )

    def test_verifier_metadata_uses_fixed_cache_routing_values(self) -> None:
        metadata = _verify_worker_metadata("image_edge_verify_prepare:text-1")

        self.assertEqual(metadata["trace_label"], "image_edge_verify_prepare:text-1")
        for key in ("session_id", "prompt_cache_key", "user_id", "x_tt_logid"):
            self.assertEqual(metadata[key], "3200636808")

    def test_checkpoint_keeps_latest_record_for_each_edge(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "results.jsonl"
            _append_checkpoint_result(checkpoint_path, {"edge_id": "edge-1", "decision": "insufficient"})
            _append_checkpoint_result(checkpoint_path, {"edge_id": "edge-2", "decision": "support"})
            _append_checkpoint_result(checkpoint_path, {"edge_id": "edge-1", "decision": "contradict"})

            records = _load_checkpoint_results(checkpoint_path)

        self.assertEqual(set(records), {"edge-1", "edge-2"})
        self.assertEqual(records["edge-1"]["decision"], "contradict")

    def test_detects_persisted_graph_post_process_record(self) -> None:
        self.assertFalse(_edge_has_post_verification({"metadata": {}}))
        self.assertTrue(
            _edge_has_post_verification(
                {"metadata": {"post_verify_image_text": {"decision": "support"}}}
            )
        )

    def test_soft_delete_is_reversible_and_keeps_verifier_provenance(self) -> None:
        edge = {
            "edge_id": "edge-verify",
            "status": "active",
            "metadata": {"post_verify_image_text": {"decision": "contradict"}},
        }

        hidden = _soft_delete_verified_edge(dict(edge), decision="contradict")
        lifecycle = hidden["metadata"]["edge_lifecycle"]
        self.assertEqual(hidden["status"], "rejected")
        self.assertEqual(lifecycle["actor"], "verify_image_text_edges")
        self.assertEqual(lifecycle["reason_code"], "post_verify_contradict")
        self.assertEqual(lifecycle["previous_status"], "active")
        self.assertEqual(len(hidden["metadata"]["edge_lifecycle_history"]), 1)

        restored = _restore_verified_edge(hidden)
        self.assertEqual(restored["status"], "active")
        self.assertEqual(restored["metadata"]["post_verify_image_text"]["decision"], "contradict")
        self.assertEqual(restored["metadata"]["edge_lifecycle"]["current_action"], "restored")
        self.assertEqual(len(restored["metadata"]["edge_lifecycle_history"]), 2)

    def test_restore_only_touches_edges_soft_deleted_by_this_verifier(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = JsonlGraphStore(Path(temp_dir))
            verifier_edge = _soft_delete_verified_edge(
                {"edge_id": "edge-verifier", "status": "active", "metadata": {}},
                decision="contradict",
            )
            unrelated_edge = {
                "edge_id": "edge-other", "status": "rejected",
                "metadata": {"edge_lifecycle": {"current_action": "soft_deleted", "actor": "deduplicator", "reason_code": "duplicate"}},
            }
            store.upsert_edge(verifier_edge)
            store.upsert_edge(unrelated_edge)
            store.flush()

            payload = _restore_post_verify_edges(
                store,
                edge_ids=[],
                restore_all=True,
                dry_run=False,
            )
            reloaded = JsonlGraphStore(Path(temp_dir))

        self.assertEqual(payload["restored_edge_ids"], ["edge-verifier"])
        self.assertEqual(reloaded.get_edge("edge-verifier")["status"], "active")
        self.assertEqual(reloaded.get_edge("edge-other")["status"], "rejected")

    def test_graph_view_hides_soft_deleted_edges_but_can_expose_them_for_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = JsonlGraphStore(Path(temp_dir))
            store.upsert_node({"node_id": "image-1", "node_type": "image"})
            store.upsert_node({"node_id": "text-1", "node_type": "text"})
            store.upsert_node({"node_id": "text-2", "node_type": "text"})
            store.upsert_edge({
                "edge_id": "edge-active", "src_node_id": "image-1", "dst_node_id": "text-1",
                "edge_type": "image_depicts", "relation": "active relation", "status": "active",
            })
            store.upsert_edge(_soft_delete_verified_edge({
                "edge_id": "edge-hidden", "src_node_id": "image-1", "dst_node_id": "text-2",
                "edge_type": "image_depicts", "relation": "hidden relation", "status": "active", "metadata": {},
            }, decision="contradict"))
            store.flush()

            sampling_view = GraphView(JsonlGraphStore(Path(temp_dir)))
            audit_view = GraphView(JsonlGraphStore(Path(temp_dir)), include_inactive_edges=True)

        self.assertEqual([edge["edge_id"] for edge in sampling_view.neighbors("image-1")], ["edge-active"])
        self.assertEqual(
            {edge["edge_id"] for edge in audit_view.neighbors("image-1")},
            {"edge-active", "edge-hidden"},
        )
        self.assertEqual(sampling_view.get_edge("edge-hidden")["status"], "rejected")

    def test_image_node_complete_status_reports_decision_counts(self) -> None:
        stderr = StringIO()
        with redirect_stderr(stderr):
            _emit_image_node_complete_status(
                image_node_id="image-1",
                records=[
                    {"decision": "support"},
                    {"decision": "contradict"},
                    {"decision": "insufficient", "error_type": "worker_exception"},
                ],
                completed_image_node_count=2,
                total_image_node_count=5,
                elapsed_s=12.34,
            )

        output = stderr.getvalue()
        self.assertIn("image_node_complete", output)
        self.assertIn("support=1", output)
        self.assertIn("contradict=1", output)
        self.assertIn("insufficient=1", output)
        self.assertIn("completed_image_nodes=2/5", output)


if __name__ == "__main__":
    unittest.main()

class LocalImagePathResolutionTests(unittest.TestCase):
    def test_resolves_persisted_local_image_path_without_http_download(self) -> None:
        from synthesis.image_discovery import ImageDiscoveryBuilder, ImageDiscoveryConfig
        from synthesis.post_process.verify_image_text_edges import _resolve_image_node_for_model
        from synthesis.test_image_grounding import _UnusedSearchClient

        # A valid 1x1 PNG. The path deliberately has no URL scheme.
        png_bytes = bytes.fromhex(
            "89504e470d0a1a0a0000000d4948445200000002000000020802000000fdd49a73"
            "0000001049444154789c63fccf00024c609201000d1d010382c971ff0000000049454e44ae426082"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "cached.jpg"
            image_path.write_bytes(png_bytes)
            builder = ImageDiscoveryBuilder(
                search_client=_UnusedSearchClient(),
                config=ImageDiscoveryConfig(precheck_image_urls=False),
            )
            model_url = _resolve_image_node_for_model(
                builder,
                image_node={
                    "image_url": str(image_path),
                    "metadata": {"resolved_image": {"cache_path": str(image_path)}},
                },
            )

        self.assertTrue(str(model_url).startswith("data:image/png;base64,"))
