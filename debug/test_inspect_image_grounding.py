import tempfile
import unittest
from pathlib import Path

from debug.inspect_image_grounding import build_report
from synthesis.store import JsonlGraphStore


class InspectImageGroundingTests(unittest.TestCase):
    def test_prints_persisted_grounding_request_and_response(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = JsonlGraphStore(Path(temp_dir))
            store.upsert_node(
                {
                    "node_id": "image-1",
                    "node_type": "image",
                    "image_url": "https://example.com/image.jpg",
                    "metadata": {
                        "resolved_image": {"cache_path": "/tmp/image.jpg", "content_type": "image/jpeg"},
                        "grounded_entities": [{"name": "Example Entity"}],
                        "image_grounding": {
                            "check": "mllm_grounding",
                            "model_alias": "image-ground-model",
                            "debug_prompt_system": "System prompt",
                            "debug_prompt_user_text": "User context",
                            "raw_model_output": "<ground>...</ground>",
                            "context": {"provider": "source_page_reader"},
                        },
                    },
                }
            )
            store.flush()

            report = build_report(Path(temp_dir), "image-1")

        self.assertEqual(report["grounding_request"]["system_prompt"], "System prompt")
        self.assertEqual(report["grounding_request"]["user_text"], "User context")
        self.assertEqual(report["grounding_request"]["image_input_references"]["image_url"], "https://example.com/image.jpg")
        self.assertEqual(report["grounding_response"]["raw_model_output"], "<ground>...</ground>")
        self.assertEqual(report["grounding_response"]["grounded_entities"], [{"name": "Example Entity"}])


if __name__ == "__main__":
    unittest.main()
