from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest import mock

from synthesis.evidence import Evidence, EvidenceType, SearchEngine, SearchSnapshot
from synthesis.image_discovery import (
    ImageCandidateStatus,
    ImageDiscoveryBuilder,
    ImageDiscoveryConfig,
    ImageSearchCandidate,
    ImageValidationResult,
    ResolvedImageAsset,
)
from synthesis.model_worker import ModelResponse
from synthesis.search_client import ImageSearchResult
from synthesis.visual_planner import SearchQuerySpec, VisualSearchPlan


class _UnusedSearchClient:
    pass


class _RefinementModel:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return ModelResponse(
            content=json.dumps(self.payload, ensure_ascii=False),
            model="fake-refiner",
            usage={"total_tokens": 42},
        )


def _fixture(payload: dict):
    target = Evidence.create(
        EvidenceType.VISUAL_TARGET,
        content="Zhang Yiming attending the opening ceremony",
        node_ids=["text_1"],
    )
    query = SearchQuerySpec.create(
        "Zhang Yiming attending the opening ceremony of the Sixth World Internet Conference in Wuzhen, China, on October 20, 2019",
        target.evidence_id,
    )
    plan = VisualSearchPlan.create(target, queries=[query], source_node_id="text_1")
    search_result = ImageSearchResult(
        title="Zhang Yiming at WIC",
        image_url="https://example.com/zhang.jpg",
        source_page_url="https://example.com/article",
        snippet="Zhang Yiming at the opening ceremony",
        rank=1,
    )
    snapshot = SearchSnapshot.create(
        SearchEngine.OTHER,
        query=query.query,
        request={"query": query.query},
        result_count=1,
    )
    candidate = ImageSearchCandidate(
        candidate_id="candidate_1",
        source_query=query,
        source_snapshot=snapshot,
        search_result=search_result,
        validation=ImageValidationResult(status=ImageCandidateStatus.ACCEPTED, confidence=0.9),
        is_primary=True,
    )
    asset = ResolvedImageAsset(
        cache_key="asset_1",
        original_url=search_result.image_url,
        resolved_url=search_result.image_url,
        source_page_url=search_result.source_page_url,
        model_url="data:image/jpeg;base64,AA==",
        asset_uri=search_result.image_url,
        cache_path=None,
        content_type="image/jpeg",
        width=800,
        height=600,
    )
    client = _RefinementModel(payload)
    builder = ImageDiscoveryBuilder(
        search_client=_UnusedSearchClient(),
        config=ImageDiscoveryConfig(
            precheck_image_urls=False,
            primary_query_refinement_model="query-refiner",
        ),
        model_client=client,
    )
    return builder, client, plan, candidate, asset


class PrimaryQueryRefinementTest(unittest.TestCase):
    def test_accepts_one_short_event_centered_refinement(self) -> None:
        refined = (
            "Zhang Yiming speaking on stage at the opening ceremony of the Sixth World "
            "Internet Conference in Wuzhen, China, on October 20, 2019"
        )
        builder, client, plan, candidate, asset = _fixture(
            {
                "decision": "refine",
                "refined_query": refined,
                "added_constraint": "speaking on stage",
                "constraint_type": "action",
                "reason": "The original query identifies the event but not the pictured sub-moment.",
                "new_named_entities": [],
                "removed_information": [],
                "expected_uniqueness": "unique",
            }
        )

        result = builder._refine_primary_search_query(
            plan=plan,
            candidate=candidate,
            resolved_asset=asset,
            run_id="run_1",
        )

        self.assertTrue(result["applied"])
        self.assertTrue(result["accepted"])
        self.assertEqual(result["refined_query"], refined)
        self.assertEqual(result["added_constraint"], "speaking on stage")
        self.assertEqual(len(client.requests), 1)
        request = client.requests[0]
        self.assertEqual(request.model, "query-refiner")
        self.assertIsNone(request.temperature)
        self.assertIsNone(request.max_tokens)
        self.assertEqual(request.response_format, {"type": "json_object"})
        self.assertIsInstance(request.messages[1].content, list)

    def test_keep_preserves_already_precise_query(self) -> None:
        candidate_query = (
            "Zhang Yiming attending the opening ceremony of the Sixth World Internet "
            "Conference in Wuzhen, China, on October 20, 2019"
        )
        builder, _, plan, candidate, asset = _fixture(
            {
                "decision": "keep",
                "refined_query": candidate_query,
                "added_constraint": "",
                "constraint_type": "none",
                "reason": "Already sufficiently precise.",
                "new_named_entities": [],
                "removed_information": [],
                "expected_uniqueness": "unique",
            }
        )
        self.assertEqual(candidate.source_query.query, candidate_query)

        result = builder._refine_primary_search_query(
            plan=plan,
            candidate=candidate,
            resolved_asset=asset,
            run_id=None,
        )

        self.assertFalse(result["accepted"])
        self.assertEqual(result["decision"], "keep")
        self.assertEqual(result["refined_query"], candidate.source_query.query)

    def test_rejects_refinement_that_adds_named_entities(self) -> None:
        original_query = (
            "Zhang Yiming attending the opening ceremony of the Sixth World Internet "
            "Conference in Wuzhen, China, on October 20, 2019"
        )
        builder, _, plan, candidate, asset = _fixture(
            {
                "decision": "refine",
                "refined_query": original_query + " while speaking beside Jack Ma",
                "added_constraint": "speaking beside Jack Ma",
                "constraint_type": "interaction",
                "reason": "More specific.",
                "new_named_entities": ["Jack Ma"],
                "removed_information": [],
                "expected_uniqueness": "unique",
            }
        )

        result = builder._refine_primary_search_query(
            plan=plan,
            candidate=candidate,
            resolved_asset=asset,
            run_id=None,
        )

        self.assertFalse(result["accepted"])
        self.assertEqual(result["refined_query"], candidate.source_query.query)
        self.assertIn("new_named_entities_not_empty", result["validation_errors"])

    def test_wiki_inline_skips_refinement(self) -> None:
        builder, client, plan, candidate, asset = _fixture({})
        plan.metadata["plan_source"] = "wikipedia_inline_image"

        result = builder._refine_primary_search_query(
            plan=plan,
            candidate=candidate,
            resolved_asset=asset,
            run_id=None,
        )

        self.assertFalse(result["applied"])
        self.assertEqual(result["reason"], "wiki_inline_image")
        self.assertEqual(client.requests, [])

    def test_grounding_context_includes_refined_search_query(self) -> None:
        builder, _, _, candidate, _ = _fixture({})
        builder.config.image_grounding_context_backend = "title_only"

        context = builder._build_image_grounding_context(
            candidate.search_result,
            search_query="Zhang Yiming speaking on stage at the opening ceremony",
        )

        self.assertIn(
            "Visual Search Query: Zhang Yiming speaking on stage at the opening ceremony",
            context.prompt_text,
        )
        self.assertIn("context for disambiguation only", context.prompt_text)


if __name__ == "__main__":
    unittest.main()
