import tempfile
import unittest
from pathlib import Path

from synthesis.post_process.verify_image_text_edges import (
    _grounding_entity_counts,
    _sample_image_node_ids,
)
from synthesis.store import JsonlGraphStore


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
