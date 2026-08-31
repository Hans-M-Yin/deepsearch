from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from synthesis.post_process.terminal_visual_evidence import (
    VqaTerminalImageResolver,
    terminal_observation_image_index,
)


class TerminalVisualEvidenceTests(unittest.TestCase):
    def test_terminal_image_placeholder_maps_to_last_observation_image(self) -> None:
        messages = [
            {"from": "human", "value": "Question <image>"},
            {"from": "observation", "value": "old image <image>"},
            {"from": "gpt", "value": "reasoning"},
            {"from": "observation", "value": "final evidence <image>"},
            {"from": "gpt", "value": "<answer>answer</answer>"},
        ]
        self.assertEqual(terminal_observation_image_index(messages, 4), 2)

    def test_resolver_uses_terminal_graph_node_not_starting_input_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            vqa_dir = root / "vqa"
            graph_dir = root / "graph"
            vqa_dir.mkdir()
            graph_dir.mkdir()
            sample = {
                "sample_id": "sample_path_a",
                "path": {
                    "path_id": "path_a",
                    "node_ids": ["image_start", "text_mid", "image_terminal"],
                    "node_types": ["image", "text", "image"],
                },
                "input_image_url": "https://example.invalid/start.png",
                "question_terminal_bridge": {"terminal_image_node_id": "image_terminal"},
            }
            (vqa_dir / "samples.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")
            node_rows = [
                {"node_id": "image_start", "node_type": "image", "image_url": "https://example.invalid/start.png"},
                {"node_id": "image_terminal", "node_type": "image", "image_url": "https://example.invalid/terminal.png"},
            ]
            (graph_dir / "nodes.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in node_rows), encoding="utf-8"
            )
            resolver = VqaTerminalImageResolver(vqa_dir, graph_dir)
            resolved = resolver.resolve({"sample_id": "sample_path_a", "path_id": "path_a"})
            self.assertIsNotNone(resolved)
            assert resolved is not None
            self.assertEqual(resolved.node_id, "image_terminal")
            self.assertEqual(resolved.image_source, "https://example.invalid/terminal.png")
            self.assertEqual(resolved.source, "graph_node")

    def test_resolver_prefers_raw_vqa_terminal_ask_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            vqa_dir = Path(temporary_directory) / "vqa"
            vqa_dir.mkdir()
            sample = {
                "sample_id": "sample_path_a",
                "question_terminal_bridge": {
                    "terminal_image_node_id": "image_terminal",
                    "raw_ask_target": "Besides the raised red card, what other card is in the referee's left hand?",
                    "rewritten_ask_target": "What other card was visible?",
                },
            }
            (vqa_dir / "samples.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")
            resolver = VqaTerminalImageResolver(vqa_dir)
            self.assertEqual(
                resolver.terminal_ask_target({"sample_id": "sample_path_a"}),
                "Besides the raised red card, what other card is in the referee's left hand?",
            )


if __name__ == "__main__":
    unittest.main()
