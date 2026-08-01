"""Targeted tests for trajectory-history exposure matching."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from synthesis.vqa.path_sampler import RandomPathSampler, SamplerConfiguration


class HistoryExposureMatchTests(unittest.TestCase):
    def setUp(self) -> None:
        # The helpers under test do not access sampler instance state.
        self.sampler = object.__new__(RandomPathSampler)

    def match(self, title: str, *history: str) -> dict:
        labels = self.sampler._candidate_text_exposure_labels(
            {"title": title, "aliases": []}
        )
        return self.sampler._hard_history_exposure_match(
            labels=labels,
            exposure_parts=[
                {"kind": "edge_relation", "text": text}
                for text in history
            ],
        )

    def test_parenthetical_title_is_blocked_by_exposed_base_name(self) -> None:
        result = self.match(
            "Astor Court (Metropolitan Museum of Art)",
            "venue where a 2012 production was performed in the Astor Court",
        )

        self.assertFalse(result["allow"])
        self.assertEqual(result["matched_label"], "Astor Court")

    def test_national_team_is_blocked_when_country_anchor_was_exposed(self) -> None:
        result = self.match(
            "New Zealand national rugby union team",
            "the photo of New Zealand captain David Kirk lifting the inaugural trophy",
        )

        self.assertFalse(result["allow"])
        self.assertEqual(result["matched_label"], "New Zealand")

    def test_generic_topical_overlap_does_not_block(self) -> None:
        result = self.match(
            "Metropolitan Museum of Art",
            "a play was performed in a museum courtyard",
        )

        self.assertTrue(result["allow"])

    def test_different_country_team_is_not_blocked(self) -> None:
        result = self.match(
            "South Africa national rugby union team",
            "New Zealand captain David Kirk lifting a trophy in Auckland",
        )

        self.assertTrue(result["allow"])


class GenericCategoryScoreCapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sampler = object.__new__(RandomPathSampler)
        self.sampler.config = SamplerConfiguration(llm_generic_category_score_cap=0.15)

    def test_caps_generic_category_score(self) -> None:
        raw, effective, classified = self.sampler._capped_llm_candidate_score(
            {"score": 0.92, "is_generic_category_target": True}
        )
        self.assertEqual(raw, 0.92)
        self.assertEqual(effective, 0.15)
        self.assertTrue(classified)

    def test_does_not_cap_specific_entity_score(self) -> None:
        raw, effective, classified = self.sampler._capped_llm_candidate_score(
            {"score": 0.92, "is_generic_category_target": False}
        )
        self.assertEqual(raw, 0.92)
        self.assertEqual(effective, 0.92)
        self.assertFalse(classified)


class _Graph:
    def __init__(self) -> None:
        self.nodes = {
            "source": {"node_id": "source", "node_type": "text", "title": "See of Constantinople", "summary": "An episcopal see."},
            "text_target": {"node_id": "text_target", "node_type": "text", "title": "1204", "summary": "A year."},
            "image_target": {
                "node_id": "image_target",
                "node_type": "image",
                "caption": "Entry of the Crusaders into Constantinople",
                "metadata": {"search_query": "Entry of the Crusaders into Constantinople", "visual_target": "Delacroix painting"},
            },
        }

    def get_node(self, node_id: str):
        return self.nodes.get(node_id)

    def node_type(self, node_id: str):
        node = self.get_node(node_id)
        return node.get("node_type") if node else None


class _TraversalGraph:
    def __init__(self, unique_state: str | None) -> None:
        image = {"node_id": "image", "node_type": "image"}
        if unique_state is not None:
            image["unique_state"] = unique_state
        self.nodes = {
            "source": {"node_id": "source", "node_type": "text"},
            "image": image,
            "target": {"node_id": "target", "node_type": "text"},
            "page": {"node_id": "page", "node_type": "text"},
        }
        self.out_edges = {
            "image": [
                {
                    "edge_id": "depicts",
                    "edge_type": "image_depicts",
                    "src_node_id": "image",
                    "dst_node_id": "target",
                    "metadata": {},
                },
                {
                    "edge_id": "source_page",
                    "edge_type": "image_source_page",
                    "src_node_id": "image",
                    "dst_node_id": "page",
                    "metadata": {},
                },
            ]
        }

    def get_node(self, node_id: str):
        return self.nodes.get(node_id)

    def node_type(self, node_id: str):
        node = self.get_node(node_id)
        return node.get("node_type") if node else None

    def neighbors(self, node_id: str):
        return list(self.out_edges.get(node_id, []))


class ImageUniqueStateTraversalTests(unittest.TestCase):
    @staticmethod
    def sampler(unique_state: str | None) -> RandomPathSampler:
        return RandomPathSampler(
            graph=_TraversalGraph(unique_state),
            config=SamplerConfiguration(),
        )

    def test_non_unique_image_at_start_can_traverse_to_text(self) -> None:
        for state in ("semi-unique", "no-unique", "wiki_inline", None):
            with self.subTest(unique_state=state):
                sampler = self.sampler(state)
                edges = sampler._traversable_neighbors("image", node_ids=["image"])
                self.assertEqual({edge["edge_id"] for edge in edges}, {"depicts", "source_page"})

    def test_non_unique_intermediate_image_cannot_use_image_depicts_edge(self) -> None:
        for state in ("semi-unique", "no-unique", "wiki_inline", None):
            with self.subTest(unique_state=state):
                sampler = self.sampler(state)
                edges = sampler._traversable_neighbors("image", node_ids=["source", "image"])
                self.assertEqual(edges, [])

    def test_unique_intermediate_image_can_traverse_to_text(self) -> None:
        sampler = self.sampler("unique")
        edges = sampler._traversable_neighbors("image", node_ids=["source", "image"])
        self.assertEqual({edge["edge_id"] for edge in edges}, {"depicts", "source_page"})

    def test_query_overlap_filter_still_applies_to_unique_intermediate_image(self) -> None:
        sampler = self.sampler("unique")
        sampler.graph.out_edges["image"][0]["metadata"] = {"query_overlap_entity": True}
        edges = sampler._traversable_neighbors("image", node_ids=["source", "image"])
        self.assertEqual([edge["edge_id"] for edge in edges], ["source_page"])


class _QueuedModelClient:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return SimpleNamespace(content=json.dumps(self.payload))


class EdgeQualityFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = _Graph()
        self.wiki_edge = {
            "edge_id": "wiki_bad",
            "edge_type": "wiki_link",
            "src_node_id": "source",
            "dst_node_id": "text_target",
            "relation": "the crusaders captured the city in 1204",
            "evidence_refs": [{"quote": "The Fourth Crusade captured Constantinople in 1204."}],
        }
        self.image_edge = {
            "edge_id": "image_bad",
            "edge_type": "search_retrieved",
            "src_node_id": "source",
            "dst_node_id": "image_target",
            "relation": "Delacroix painting of the Fourth Crusade",
            "evidence_refs": [{"metadata": {"snippet": "The painting depicts the sack of Constantinople."}}],
        }

    def sampler(self, response: dict) -> tuple[RandomPathSampler, _QueuedModelClient]:
        client = _QueuedModelClient(response)
        return (
            RandomPathSampler(
                graph=self.graph,
                config=SamplerConfiguration(),
                history_exposure_model_client=client,
                history_exposure_model="overlap-model",
            ),
            client,
        )

    def test_rejects_when_any_text_judgment_fails_and_batches_candidates(self) -> None:
        sampler, client = self.sampler(
            {"evaluations": [
                {"edge_id": "wiki_bad", "relevance_analysis": "The relation omits the see.", "relevance_ok": False,
                 "correctness_analysis": "The evidence concerns a crusade.", "correctness_ok": False,
                 "unambiguous_analysis": "1204 is clear but irrelevant.", "unambiguous_ok": True, "keep": True},
                {"edge_id": "image_bad", "relevance_analysis": "The painting is about the sack, not the see.", "relevance_ok": False,
                 "correctness_analysis": "", "correctness_ok": True,
                 "unambiguous_analysis": "", "unambiguous_ok": True, "keep": False},
            ]}
        )

        rejected = sampler._edge_quality_rejections([self.wiki_edge, self.image_edge])

        self.assertEqual(set(rejected), {"wiki_bad", "image_bad"})
        self.assertFalse(rejected["wiki_bad"]["evaluation"]["keep"])
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(client.requests[0].model, "overlap-model")
        self.assertEqual(
            client.requests[0].metadata,
            {
                "trace_label": "sampler_edge_quality",
                "session_id": "3200636808",
                "prompt_cache_key": "3200636808",
                "user_cache_key": "3200636808",
                "user_id": "3200636808",
                "x_tt_logid": "3200636808",
            },
        )
        payload = json.loads(client.requests[0].messages[1].content)
        self.assertEqual([item["edge_kind"] for item in payload["candidates"]], ["text_to_text", "text_to_image"])

    def test_missing_model_result_fails_open_and_is_cached(self) -> None:
        sampler, client = self.sampler({"evaluations": []})

        self.assertEqual(sampler._edge_quality_rejections([self.wiki_edge]), {})
        self.assertEqual(sampler._edge_quality_rejections([self.wiki_edge]), {})
        self.assertEqual(len(client.requests), 1)


if __name__ == "__main__":
    unittest.main()
