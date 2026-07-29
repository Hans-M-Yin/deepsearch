from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from synthesis.model_worker import ModelResponse
from synthesis.post_process.classify_image_unique_state import (
    classify_graph,
    image_origin,
    parse_judge_response,
)
from synthesis.store import JsonlGraphStore


class FakeWorker:
    def __init__(self, responses: dict[str, str]) -> None:
        self.responses = responses
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        user_text = str(request.messages[-1].content)
        image_id = next(line.split("：", 1)[1] for line in user_text.splitlines() if line.startswith("图片标识："))
        return ModelResponse(content=self.responses[image_id], model="fake-model", usage={"total_tokens": 10})


class ParseJudgeResponseTest(unittest.TestCase):
    def test_parse_and_convert_ready_label(self) -> None:
        parsed = parse_judge_response(
            "分析：这是固定封面。\nimage_abc  分类：唯一性  ｜  理由：固定图案。",
            expected_image_id="image_abc",
        )
        self.assertEqual(parsed["label"], "唯一性")
        self.assertIsNone(parsed["parse_warning"])


class ImageOriginTest(unittest.TestCase):
    def test_wiki_inline_precedes_image_search_source(self) -> None:
        node = {
            "node_id": "image_wiki",
            "node_type": "image",
            "source": {"source_type": "image_search"},
            "metadata": {"image_origin": "wikipedia_inline"},
        }
        self.assertEqual(image_origin(node), "wiki_inline")

    def test_source_text_edge_is_visual_plan_fallback(self) -> None:
        node = {"node_id": "image_old", "node_type": "image", "source": {}}
        edge = {"src_node_id": "text_1", "dst_node_id": "image_old"}
        nodes = {"text_1": {"node_id": "text_1", "node_type": "text"}}
        self.assertEqual(image_origin(node, [edge], nodes_by_id=nodes), "visual_plan")


class ClassifyGraphTest(unittest.TestCase):
    def test_assigns_visual_and_wiki_states_without_sampling_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_dir = Path(tmpdir)
            store = JsonlGraphStore(graph_dir)
            store.upsert_node({"node_id": "text_1", "node_type": "text", "title": "Source"})
            store.upsert_node(
                {
                    "node_id": "image_visual",
                    "node_type": "image",
                    "source": {"source_type": "image_search"},
                    "metadata": {"search_query": "The album cover of Example"},
                }
            )
            store.upsert_node(
                {
                    "node_id": "image_wiki",
                    "node_type": "image",
                    "source": {"source_type": "wikipedia_inline_image"},
                    "metadata": {},
                }
            )
            store.upsert_edge(
                {
                    "edge_id": "edge_1",
                    "src_node_id": "text_1",
                    "dst_node_id": "image_visual",
                    "edge_type": "search_retrieved",
                }
            )
            store.flush()

            worker = FakeWorker(
                {
                    "image_visual": (
                        "分析：特定专辑封面是固定图案。\n"
                        "image_visual  分类：唯一性  ｜  理由：搜索结果对应同一封面。"
                    )
                }
            )
            summary = classify_graph(
                graph_dir=graph_dir,
                judge_model_alias="fake-alias",
                model_client=worker,
                workers=1,
                retries=0,
                results_jsonl=graph_dir / "checkpoint.jsonl",
            )

            reloaded = JsonlGraphStore(graph_dir)
            self.assertEqual(reloaded.get_node("image_visual")["unique_state"], "unique")
            self.assertEqual(reloaded.get_node("image_wiki")["unique_state"], "wiki_inline")
            self.assertTrue(summary["graph_written"])
            self.assertEqual(len(worker.requests), 1)
            request = worker.requests[0]
            self.assertIsNone(request.temperature)
            self.assertIsNone(request.max_tokens)
            self.assertIsNone(request.response_format)

    def test_atomic_default_does_not_write_when_query_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_dir = Path(tmpdir)
            store = JsonlGraphStore(graph_dir)
            store.upsert_node(
                {
                    "node_id": "image_visual",
                    "node_type": "image",
                    "source": {"source_type": "image_search"},
                    "metadata": {},
                }
            )
            store.upsert_node(
                {
                    "node_id": "image_wiki",
                    "node_type": "image",
                    "source": {"source_type": "wikipedia_inline_image"},
                    "metadata": {},
                }
            )
            store.flush()
            summary = classify_graph(
                graph_dir=graph_dir,
                judge_model_alias="fake-alias",
                model_client=FakeWorker({}),
                workers=1,
                retries=0,
            )
            reloaded = JsonlGraphStore(graph_dir)
            self.assertIsNone(reloaded.get_node("image_wiki").get("unique_state"))
            self.assertFalse(summary["graph_written"])
            self.assertEqual(summary["missing_search_query_image_node_ids"], ["image_visual"])


if __name__ == "__main__":
    unittest.main()
