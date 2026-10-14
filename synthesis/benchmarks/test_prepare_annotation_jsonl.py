from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from synthesis.benchmarks.prepare_annotation_jsonl import _parse_model_json, main
from synthesis.model_worker import ModelResponse


class _FakeTranslator:
    def generate(self, request):
        prompt = request.messages[-1].content
        payload = json.loads(prompt[prompt.index("["):])
        return ModelResponse(
            content=json.dumps(
                {"translations": {item["id"]: f"中文：{item['english']}" for item in payload}},
                ensure_ascii=False,
            )
        )


class PrepareAnnotationJsonlTest(unittest.TestCase):
    def test_translation_preserves_markdown_fences_inside_json_values(self) -> None:
        expected = {"translations": {"segment": "```markdown\n# 加拿大\n加拿大位于北美洲。\n```"}}
        raw = json.dumps(expected, ensure_ascii=False)
        for response in (raw, f"```json\n{raw}\n```", f"Translation result:\n{raw}"):
            with self.subTest(response=response):
                self.assertEqual(_parse_model_json(response), expected)

    def test_join_graph_and_translate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            vqa = root / "vqa"
            graph = root / "graph"
            vqa.mkdir()
            graph.mkdir()

            question = {
                "question_id": "q_1",
                "sample_id": "sample_1",
                "path_id": "path_1",
                "question": "What is shown?",
                "answer": "A ship",
            }
            sample = {
                "sample_id": "sample_1",
                "path": {
                    "path_id": "path_1",
                    "node_ids": ["image_1", "text_1", "image_2"],
                    "edge_ids": ["edge_1", "edge_2"],
                    "target_node_id": "image_2",
                },
                "entry_hop": {
                    "edge_id": "edge_1",
                    "src_node_id": "image_1",
                    "dst_node_id": "text_1",
                    "statement": "The image identifies the event.",
                },
                "hop_chain": [
                    {"edge_id": "edge_1", "src_node_id": "image_1", "dst_node_id": "text_1"},
                    {"edge_id": "edge_2", "src_node_id": "text_1", "dst_node_id": "image_2"},
                ],
                "target_ask": {"ask_target": "What is shown?", "answer": "A ship"},
            }
            (vqa / "questions.jsonl").write_text(json.dumps(question) + "\n", encoding="utf-8")
            (vqa / "samples.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")

            nodes = [
                {
                    "node_id": "image_1",
                    "node_type": "image",
                    "title": "Image: source image",
                    "caption": "A source image",
                    "image_url": "https://cdn.example/source.png",
                    "source_page_url": "https://en.wikipedia.org/wiki/Source",
                    "image_variants": [{
                        "is_primary": True,
                        "image_url": "https://upload.wikimedia.org/source.jpg",
                        "source_page_url": "https://en.wikipedia.org/wiki/Source",
                    }],
                },
                {
                    "node_id": "text_1",
                    "node_type": "text",
                    "title": "An event",
                    "source": {"url": "https://en.wikipedia.org/wiki/Event"},
                },
                {
                    "node_id": "image_2",
                    "node_type": "image",
                    "title": "Image: target image",
                    "image_url": "https://cdn.example/target.png",
                    "image_variants": [{
                        "is_primary": True,
                        "image_url": "https://original.example/target.jpg",
                        "source_page_url": "https://original.example/page",
                    }],
                },
            ]
            (graph / "nodes.jsonl").write_text(
                "".join(json.dumps(item) + "\n" for item in nodes), encoding="utf-8"
            )
            edges = [
                {
                    "edge_id": "edge_1",
                    "src_node_id": "image_1",
                    "dst_node_id": "text_1",
                    "relation": "depicts",
                    "evidence_refs": [{"evidence_id": "evidence_1"}],
                },
                {
                    "edge_id": "edge_2",
                    "src_node_id": "text_1",
                    "dst_node_id": "image_2",
                    "relation": "has image",
                    "evidence_refs": [{"evidence_id": "evidence_2"}],
                },
            ]
            (graph / "edges.jsonl").write_text(
                "".join(json.dumps(item) + "\n" for item in edges), encoding="utf-8"
            )
            evidence = [
                {"evidence_id": "evidence_1", "evidence_type": "image", "content": "The source image depicts the event.", "url": "https://cdn.example/source.png"},
                {"evidence_id": "evidence_2", "evidence_type": "image", "content": "The target image shows a ship.", "url": "https://original.example/target.jpg"},
            ]
            (graph / "evidence.jsonl").write_text(
                "".join(json.dumps(item) + "\n" for item in evidence), encoding="utf-8"
            )

            output = root / "annotation.jsonl"
            result = main(
                [
                    "--vqa-dir", str(vqa),
                    "--graph-dir", str(graph),
                    "--output-jsonl", str(output),
                    "--translation-model-alias", "fake",
                    "--translation-workers", "1",
                ],
                model_worker=_FakeTranslator(),
            )
            self.assertEqual(result, 0)
            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(record["question_id"], "q_1")
            self.assertEqual(record["path_id"], "path_1")
            self.assertEqual(record["record_status"], "complete")
            self.assertEqual(record["question"]["zh"], "中文：What is shown?")
            self.assertEqual(record["search_chain"]["nodes"][2]["original_url"], "https://original.example/target.jpg")
            self.assertEqual(record["final_ask"]["answer_evidence"][0]["text"]["en"], "The target image shows a ship.")


if __name__ == "__main__":
    unittest.main()
